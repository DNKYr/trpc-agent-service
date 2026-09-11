"""HTTP gateway/admin surface for the runnable reference implementation."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field, is_dataclass
from hashlib import sha256
from hmac import compare_digest
from secrets import token_urlsafe
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from trpc_service.agent import AgentFactory, FunctionTool, ImmutableRelease
from trpc_service.channels import (
    AIBotRegistry,
    CallbackRequest,
    ChannelBinding,
    ChannelError,
    MockChannelAdapter,
    ReplyBlock,
    ReplyEnvelope,
    TelegramAdapter,
    WeComAdapter,
    WeComAIBotAdapter,
    WeComAIBotSupervisor,
)
from trpc_service.config import (
    AppSettings,
    EnvironmentSecretProvider,
    MockSecretProvider,
    get_settings,
)
from trpc_service.control import Binding, ControlConflict, ControlNotFound, ControlPlane
from trpc_service.db import PostgresControlPlane
from trpc_service.db.postgres import PostgresConnections
from trpc_service.governance import (
    BudgetFilter,
    InputDLPFilter,
    OutputDLPFilter,
    Principal,
    PrincipalFilter,
    RateLimitFilter,
)
from trpc_service.metrics import bind_trace_context, configure_opentelemetry, extract_trace_context
from trpc_service.runtime import (
    BudgetAccount,
    CommitInput,
    InboundEnvelope,
    InMemoryDeliveryLedger,
    InMemoryMessageBus,
    InMemoryRuntimeStore,
    MemoryIntentDraft,
    PlatformRuntime,
    PostgresDeliveryLedger,
    PostgresRuntimeStore,
    RedisStreamMessageBus,
    ReplyDraft,
    RuntimeErrorBase,
    SessionEventDraft,
    TenantContext,
    ToolCapability,
    ToolStatus,
    stable_id,
    to_primitive,
)
from trpc_service.tool import (
    MockSideEffectTool,
    TicketLookupTool,
    ToolInvocation,
    ToolPolicy,
    ToolPolicyFilter,
)

from .schemas import (
    AgentCreate,
    BudgetCreate,
    ChannelCreate,
    MigrationAction,
    MigrationCreate,
    OperationResponse,
    Page,
    ReleaseCreate,
    ResolutionRequest,
    RunRequest,
    TenantCreate,
    TenantResponse,
)


@dataclass(frozen=True, slots=True)
class AuthPrincipal:
    """The authenticated HTTP caller, never including its credential."""

    kind: str
    tenant_id: str | None = None

    @property
    def is_admin(self) -> bool:
        return self.kind == "admin"


def _json(value: Any) -> Any:
    if is_dataclass(value):
        return to_primitive(value)
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_json(item) for item in value]
    return value


def _context(tenant_id: str, request_id: str, trace_id: str, actor: str = "api") -> TenantContext:
    return TenantContext(
        tenant_id=tenant_id, actor_id=actor, request_id=request_id, trace_id=trace_id
    )


def _session_for_api(
    tenant_id: str, agent_id: str, session_key: str | None, subject_id: str
) -> str:
    material = f"{tenant_id}\x1f{agent_id}\x1f{session_key or subject_id}".encode()
    return "ses_api_" + sha256(material).hexdigest()[:32]


def _api_binding_id(agent_id: str) -> str:
    """Return a reserved durable binding for queued API-originated work."""

    return f"api.{agent_id}"


@dataclass(slots=True)
class ServiceContainer:
    """Process-local composition root.

    The memory store is intentionally the zero-credential mode.  Its methods use
    the same transaction/fence interfaces as the SQL adapter, so API and worker
    code do not rely on a local-only correctness shortcut.
    """

    settings: AppSettings
    control: Any = field(init=False)
    store: Any = field(init=False)
    bus: Any = field(init=False)
    mock_channel: MockChannelAdapter = field(default_factory=MockChannelAdapter)
    processed_bus_events: set[str] = field(default_factory=set)
    delivery_ledger: Any = field(init=False)
    runtime: PlatformRuntime = field(init=False)
    secrets: Any = field(init=False)
    agent_factory: AgentFactory = field(init=False)
    adapters: dict[str, Any] = field(init=False)
    tools: dict[str, Any] = field(init=False)
    aibot_registry: AIBotRegistry = field(init=False)
    principal_filter: PrincipalFilter = field(default_factory=PrincipalFilter)
    rate_limit_filter: RateLimitFilter = field(default_factory=RateLimitFilter)
    budget_filter: BudgetFilter = field(default_factory=BudgetFilter)
    input_dlp_filter: InputDLPFilter = field(default_factory=InputDLPFilter)
    output_dlp_filter: OutputDLPFilter = field(default_factory=OutputDLPFilter)

    def __post_init__(self) -> None:
        if self.settings.runtime_backend == "postgres":
            self.control = PostgresControlPlane(
                self.settings.database_url, self.settings.database_role
            )
            self.store = PostgresRuntimeStore(self.settings.database_url, self.settings.database_role)
            self.bus = RedisStreamMessageBus(self.settings.redis_url)
            self.secrets = EnvironmentSecretProvider()
            self.delivery_ledger = PostgresDeliveryLedger(
                self.settings.database_url, self.settings.database_role
            )
        elif self.settings.runtime_backend == "memory":
            self.control = ControlPlane()
            self.store = InMemoryRuntimeStore()
            self.bus = InMemoryMessageBus()
            self.secrets = MockSecretProvider()
            self.delivery_ledger = InMemoryDeliveryLedger()
        else:
            raise ValueError("TRPC_SERVICE_RUNTIME_BACKEND must be 'memory' or 'postgres'")
        self.runtime = PlatformRuntime(self.store, self.bus)
        self.agent_factory = AgentFactory(self.settings, secrets=self.secrets)
        self.aibot_registry = AIBotRegistry()
        self.adapters = {
            "mock": self.mock_channel,
            "telegram": TelegramAdapter(self.secrets),
            "wecom": WeComAdapter(self.secrets),
            "wecom_aibot": WeComAIBotAdapter(self.aibot_registry),
        }
        # Tools remain registered in the process, but each execution exposes
        # only the immutable release allowlist through FunctionTool wrappers.
        self.tools = {
            "ticket.lookup": TicketLookupTool(),
            "mock.side_effect": MockSideEffectTool(),
        }

    def create_tenant(self, body: TenantCreate) -> dict[str, Any]:
        tenant = self.control.create_tenant(
            body.tenant_id,
            body.display_name,
            audit_policy=body.audit_policy,
            budget_policy=body.budget_policy,
        )
        state = self.store.bootstrap_tenant(body.tenant_id)
        # A service tenant starts with an explicit (generous) hard model-token
        # account.  Operators can replace it through the budget endpoint; there
        # is never an implicit local allowance at claim time.
        self.runtime.put_budget_account(
            _context(body.tenant_id, "bootstrap", "bootstrap"),
            BudgetAccount(body.tenant_id, "model_tokens", "tokens", limit_units=1_000_000),
        )
        result = self.control.serialize(tenant)
        result.update(
            {
                "routing_epoch": state.routing_epoch,
                "security_epoch": state.security_epoch,
                "execution_mode": state.execution_mode.value,
            }
        )
        return result

    def tenant_view(
        self, tenant_id: str, request_id: str = "", trace_id: str = ""
    ) -> dict[str, Any]:
        tenant = self.control.serialize(self.control.tenant(tenant_id))
        state = self.runtime.runtime_state(_context(tenant_id, request_id, trace_id))
        tenant.update(
            {
                "routing_epoch": state.routing_epoch,
                "security_epoch": state.security_epoch,
                "execution_mode": state.execution_mode.value,
                "tool_denylist": sorted(state.tool_denylist),
            }
        )
        return tenant

    def create_agent(self, tenant_id: str, body: AgentCreate) -> dict[str, Any]:
        agent = self.control.create_agent(tenant_id, body.agent_id, body.name)
        self._ensure_api_binding(tenant_id, body.agent_id)
        return self.control.serialize(agent)

    def _ensure_api_binding(self, tenant_id: str, agent_id: str) -> Binding:
        """Create the internal source record required by Inbox/Session FKs.

        Direct runs are first-class queued work, not a pretend external channel
        and not an inline execution shortcut.  The binding is intentionally
        unaddressable: no callback route or adapter exists for ``provider=api``.
        """

        binding_id = _api_binding_id(agent_id)
        try:
            binding = self.control.binding(tenant_id, binding_id)
        except ControlNotFound:
            try:
                binding = self.control.create_binding(
                    tenant_id,
                    binding_id=binding_id,
                    agent_id=agent_id,
                    provider="api",
                    external_account_id=f"internal-api:{tenant_id}:{agent_id}",
                    webhook_key=token_urlsafe(32),
                    secret_ref="internal://direct-run",
                    capabilities={"internal": True, "source_type": "api"},
                )
            except ControlConflict:
                binding = self.control.binding(tenant_id, binding_id)
        if binding.provider != "api" or binding.agent_id != agent_id:
            raise ControlConflict(f"reserved API binding {binding_id!r} is not valid for this agent")
        return binding

    def create_release(self, tenant_id: str, agent_id: str, body: ReleaseCreate) -> dict[str, Any]:
        release = self.control.create_release(
            tenant_id,
            agent_id,
            body.version,
            app_config=body.app_config,
            model_config=body.release_model_config,
            tool_policy=body.tool_policy,
            knowledge_config=body.knowledge_config,
            created_by=body.created_by,
            change_reason=body.change_reason,
        )
        return self.control.serialize(release)

    def activate_release(self, tenant_id: str, agent_id: str, version: int) -> dict[str, Any]:
        return self.control.serialize(self.control.activate_release(tenant_id, agent_id, version))

    def rollback(self, tenant_id: str, agent_id: str) -> dict[str, Any]:
        return self.control.serialize(self.control.rollback_agent(tenant_id, agent_id))

    def create_binding(self, tenant_id: str, body: ChannelCreate) -> dict[str, Any]:
        if body.binding_id.startswith("api."):
            raise ValueError("binding IDs starting with 'api.' are reserved for direct-run sources")
        if body.provider != "wecom_aibot" and not body.webhook_key:
            raise ValueError("webhook_key is required for callback-based channels")
        # Long connections authenticate to WeCom instead of a public callback.
        # The relational locator still requires a non-null unique hash, so make
        # an unguessable internal value that is never exposed or used as a URL.
        webhook_key = body.webhook_key or token_urlsafe(32)
        binding = self.control.create_binding(
            tenant_id,
            binding_id=body.binding_id,
            agent_id=body.agent_id,
            provider=body.provider,
            external_account_id=body.external_account_id,
            webhook_key=webhook_key,
            secret_ref=body.secret_ref,
            capabilities=body.capabilities,
        )
        return self.control.serialize(binding)

    @staticmethod
    def adapter_binding(binding: Binding) -> ChannelBinding:
        return ChannelBinding(
            tenant_id=binding.tenant_id,
            binding_id=binding.binding_id,
            agent_id=binding.agent_id,
            provider=binding.provider,
            external_account_id=binding.external_account_id,
            secret_ref=binding.secret_ref,
            capabilities=binding.capabilities,
            status=binding.status,
        )

    async def accept_callback(
        self,
        provider: str,
        binding_key: str,
        request: CallbackRequest,
        request_id: str,
        trace_id: str,
    ) -> dict[str, Any]:
        binding = self.control.resolve_callback_binding(provider, binding_key)
        adapter = self.adapters[provider]
        adapter_binding = self.adapter_binding(binding)
        envelope = await adapter.validate_and_normalize(adapter_binding, request)
        return self.accept_channel_envelope(adapter_binding, envelope, request_id, trace_id)

    def accept_channel_envelope(
        self,
        binding: ChannelBinding,
        envelope: Any,
        request_id: str,
        trace_id: str,
    ) -> dict[str, Any]:
        """Persist a provider-verified normalized message in the durable Inbox."""

        context = _context(binding.tenant_id, request_id, trace_id, actor=f"callback:{binding.provider}")
        accepted = self.runtime.accept_inbound(
            context,
            InboundEnvelope(
                tenant_id=binding.tenant_id,
                channel_binding_id=binding.binding_id,
                agent_id=binding.agent_id,
                session_id=envelope.session_id,
                idempotency_key=envelope.idempotency_key,
                external_message_id=envelope.external_message_id,
                subject_id=envelope.principal.external_user_id,
                config_version=self.control.active_release(
                    binding.tenant_id, binding.agent_id
                ).version,
                payload={
                    "channel": envelope.channel,
                    "text": envelope.text,
                    "attachments": [_json(item) for item in envelope.attachments],
                    "principal": envelope.principal.external_user_id,
                    "recipient_id": envelope.principal.external_user_id,
                    "external_message_id": envelope.external_message_id,
                    "channel_context": dict(envelope.transport_context),
                },
                request_id=request_id,
                trace_id=trace_id,
            ),
        )
        return {
            "inbox_id": accepted.inbox.inbox_id,
            "duplicate": accepted.duplicate,
            "request_id": request_id,
            "trace_id": trace_id,
        }

    async def accept_aibot_frame(self, binding: ChannelBinding, request: CallbackRequest) -> None:
        """Accept a verified WebSocket frame without exposing an HTTP callback route."""

        adapter = self.adapters["wecom_aibot"]
        envelope = await adapter.validate_and_normalize(binding, request)
        trace_id = envelope.traceparent.split("-")[1] if "-" in envelope.traceparent else ""
        self.accept_channel_envelope(
            binding,
            envelope,
            request_id=f"aibot-{envelope.event_id}",
            trace_id=trace_id,
        )

    def aibot_bindings(self) -> list[ChannelBinding]:
        return [
            self.adapter_binding(binding)
            for binding in self.control.bindings_for_provider("wecom_aibot")
        ]

    def aibot_supervisor(self) -> WeComAIBotSupervisor:
        return WeComAIBotSupervisor(
            bindings=self.aibot_bindings,
            secrets_provider=self.secrets,
            registry=self.aibot_registry,
            on_inbound=self.accept_aibot_frame,
        )

    def readiness(self) -> tuple[bool, dict[str, str]]:
        """Check dependencies used by this process without weakening liveness."""

        if self.settings.runtime_backend == "memory":
            return True, {"runtime_backend": "memory"}
        failures: dict[str, str] = {}
        try:
            PostgresConnections(self.settings.database_url, self.settings.database_role).check()
        except Exception:
            failures["postgres"] = "unavailable"
        try:
            self.bus.check()
        except Exception:
            failures["redis"] = "unavailable"
        if failures:
            return False, failures
        return True, {"runtime_backend": self.settings.runtime_backend}

    async def _run_with_execution_heartbeat(
        self,
        context: TenantContext,
        claim: Any,
        operation: Any,
    ) -> tuple[Any, Any]:
        """Renew the fenced execution lease while an external operation runs."""

        current_claim = claim
        stop = asyncio.Event()
        renewal_error: Exception | None = None
        interval = self.settings.execution_heartbeat_seconds

        async def renew() -> None:
            nonlocal current_claim, renewal_error
            while True:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=interval)
                    return
                except TimeoutError:
                    try:
                        current_claim = self.runtime.renew_execution(
                            context,
                            current_claim,
                            lease_seconds=self.settings.execution_lease_seconds,
                        )
                    except Exception as exc:  # commit must be fenced after model return.
                        renewal_error = exc
                        return

        task = asyncio.create_task(renew())
        try:
            result = await operation
        finally:
            stop.set()
            await task
        if renewal_error is not None:
            raise renewal_error
        return result, current_claim

    def _session_context(self, context: TenantContext, inbox: Mapping[str, Any]) -> dict[str, Any]:
        """Load persisted session events and memory before constructing model input."""

        snapshot = self.runtime.snapshot(context)
        session_id = str(inbox["session_id"])
        subject_id = str(inbox.get("subject_id") or "")
        memories = [
            {
                "memory_id": row["memory_id"],
                "type": row["memory_type"],
                "content": row["content"],
            }
            for row in snapshot["memories"]
            if row.get("content")
            and (
                row.get("session_id") == session_id
                or (subject_id and row.get("subject_id") == subject_id)
            )
        ][-20:]
        history = [
            row
            for row in snapshot["events"]
            if row.get("session_id") == session_id and row.get("role") in {"user", "assistant"}
        ][-12:]
        summary = "\n".join(
            f"{row['role']}: {str(row.get('payload', {}).get('text', ''))[:1000]}"
            for row in history
            if isinstance(row.get("payload"), Mapping)
        )
        return {"memory": memories, "summary": summary}

    def _execution_tools(
        self,
        context: TenantContext,
        claim: Any,
        inbox: Mapping[str, Any],
        release: ImmutableRelease,
    ) -> tuple[FunctionTool, ...]:
        """Bind model-visible tools to the persistent intent ledger and fence."""

        state = self.runtime.runtime_state(context)
        policy = ToolPolicy.from_release(release.tool_policy, live_denylist=state.tool_denylist)
        subject_id = str(inbox.get("subject_id") or "anonymous")
        step = 0
        bound: list[FunctionTool] = []
        for name in sorted(policy.allowlist):
            tool = self.tools.get(name)
            if tool is None or name in policy.denylist:
                continue

            async def handler(arguments: Mapping[str, Any], *, selected_tool=tool) -> Mapping[str, Any]:
                nonlocal step
                invocation = ToolInvocation(
                    name=selected_tool.name,
                    step=step,
                    arguments=dict(arguments),
                    principal_id=subject_id,
                    trace_id=context.trace_id,
                )
                step += 1
                ToolPolicyFilter(policy).authorize(selected_tool, invocation)
                intent = self.runtime.prepare_tool(
                    context,
                    claim,
                    tool_step=invocation.step,
                    tool_name=selected_tool.name,
                    arguments=invocation.arguments,
                    capability=ToolCapability(selected_tool.retry_capability.value),
                )
                started = self.runtime.start_tool(context, claim, intent.tool_call_id)
                if started.status == ToolStatus.SUCCEEDED:
                    return dict(started.result or {})
                if started.status in {
                    ToolStatus.UNKNOWN,
                    ToolStatus.RECONCILING,
                    ToolStatus.MANUAL_REVIEW,
                }:
                    raise RuntimeError("tool operation requires reconciliation or manual resolution")
                try:
                    self.runtime.assert_execution(context, claim)
                    result = await selected_tool.call(
                        invocation.arguments,
                        claim=claim,
                        idempotency_key=intent.provider_idempotency_key,
                    )
                except Exception as exc:
                    self.runtime.finish_tool(
                        context,
                        claim,
                        intent.tool_call_id,
                        status=ToolStatus.UNKNOWN,
                        error_code="tool_provider_ambiguous",
                    )
                    raise RuntimeError("tool provider outcome is unknown") from exc
                completed = self.runtime.finish_tool(
                    context,
                    claim,
                    intent.tool_call_id,
                    status=ToolStatus.SUCCEEDED,
                    result=result,
                    provider_operation_id=(
                        str(result["provider_operation_id"])
                        if result.get("provider_operation_id") is not None
                        else None
                    ),
                )
                return dict(completed.result or {})

            bound.append(
                FunctionTool(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.parameters,
                    handler=handler,
                )
            )
        return tuple(bound)

    async def execute_inbox(
        self, context: TenantContext, inbox_id: str, worker_id: str
    ) -> dict[str, Any]:
        """Run a claimed Inbox using its immutable release and atomically commit output."""

        snapshot = self.runtime.snapshot(context)
        inbox = next((row for row in snapshot["inboxes"] if row["inbox_id"] == inbox_id), None)
        if inbox is None:
            raise KeyError("inbox does not exist")
        release_record = self.control.release(
            context.tenant_id, str(inbox["agent_id"]), int(inbox["config_version"])
        )
        release = ImmutableRelease(
            tenant_id=release_record.tenant_id,
            agent_id=release_record.agent_id,
            config_version=release_record.version,
            app_config=release_record.app_config,
            model_config=release_record.model_config,
            tool_policy=release_record.tool_policy,
            knowledge_config=release_record.knowledge_config,
        )
        max_output_tokens = int(
            release.model_config.get("max_output_tokens", self.settings.model_max_output_tokens)
        )
        if max_output_tokens < 1:
            raise ValueError("release model_config.max_output_tokens must be positive")
        budget_estimates = self.budget_filter.estimates(
            str(inbox["payload"].get("text", "")),
            max_output_tokens=max_output_tokens,
            input_overhead_tokens=self.settings.model_input_overhead_tokens,
        )
        claim = self.runtime.claim_execution(
            context,
            inbox_id,
            worker_id,
            lease_seconds=self.settings.execution_lease_seconds,
            budget_estimates=budget_estimates,
        )
        # Fence/security check immediately before the model call.  Commit repeats
        # the predicates inside its transaction so a stale worker cannot write.
        self.runtime.assert_execution(context, claim)
        if claim.config_version != release.config_version:
            raise RuntimeError("execution claim release version disagrees with its durable Inbox")
        agent = await self.agent_factory.build(
            release, tools=self._execution_tools(context, claim, inbox, release)
        )
        input_text = str(inbox["payload"].get("text", ""))
        self.principal_filter.check(
            Principal(context.tenant_id, str(inbox.get("subject_id") or "anonymous"))
        )
        self.rate_limit_filter.check(context.tenant_id, str(inbox.get("subject_id") or "anonymous"))
        self.input_dlp_filter.check(input_text)
        result, claim = await self._run_with_execution_heartbeat(
            context,
            claim,
            agent.run(input_text, self._session_context(context, inbox)),
        )
        self.runtime.assert_execution(context, claim)
        events = [
            SessionEventDraft(
                event_type="user.message",
                role="user",
                payload={"text": input_text, "inbox_id": inbox_id},
                subject_id=inbox.get("subject_id"),
                external_message_id=inbox.get("external_message_id"),
            ),
            *[
            SessionEventDraft(
                event_type=event.kind,
                role="assistant" if event.kind == "reply.text" else None,
                payload={**dict(event.data), "inbox_id": inbox_id},
            )
            for event in result.events
            ],
        ]
        reply = result.reply.text()
        self.output_dlp_filter.check(reply)
        committed = self.runtime.commit_execution(
            context,
            claim,
            CommitInput(
                expected_session_version=claim.session_version,
                new_state={"last_reply": reply, "last_reply_inbox_id": inbox_id, "model": result.model},
                events=events,
                memories=[
                    MemoryIntentDraft(
                        memory_type="recent_input",
                        content=input_text,
                        subject_id=inbox.get("subject_id"),
                    )
                ],
                # API-originated runs persist their response as a Session event
                # for the operation endpoint.  They never create a fake reply
                # delivery event that a channel dispatcher cannot acknowledge.
                reply=(
                    None
                    if inbox["payload"].get("source_type") == "api"
                    else ReplyDraft(
                        blocks=[{"type": "text", "text": reply}],
                        channel_binding_id=str(inbox["channel_binding_id"]),
                        recipient_id=str(
                            inbox["payload"].get("recipient_id")
                            or inbox.get("subject_id")
                            or ""
                        ),
                        metadata=dict(inbox["payload"].get("channel_context") or {}),
                    )
                ),
                actual_budget_units={
                    "model_tokens": (
                        result.usage["input_tokens"] + result.usage["output_tokens"]
                        if result.usage["input_tokens"] > 0 and result.usage["output_tokens"] > 0
                        else budget_estimates["model_tokens"]
                    )
                },
            ),
        )
        return {
            "inbox_id": inbox_id,
            "execution_id": claim.execution_id,
            "reply": reply,
            "reply_outbox_id": committed.reply_outbox.outbox_id if committed.reply_outbox else None,
        }

    async def run_direct(
        self, tenant_id: str, agent_id: str, body: RunRequest, request_id: str, trace_id: str
    ) -> dict[str, Any]:
        release = self.control.active_release(tenant_id, agent_id)
        context = _context(tenant_id, request_id, trace_id, actor="api-run")
        session_id = _session_for_api(tenant_id, agent_id, body.session_key, body.subject_id)
        binding = self._ensure_api_binding(tenant_id, agent_id)
        accepted = self.runtime.accept_inbound(
            context,
            InboundEnvelope(
                tenant_id=tenant_id,
                channel_binding_id=binding.binding_id,
                agent_id=agent_id,
                session_id=session_id,
                idempotency_key=body.idempotency_key or f"api:{request_id}",
                external_message_id=None,
                subject_id=body.subject_id,
                config_version=release.version,
                payload={
                    "channel": "api",
                    "source_type": "api",
                    "text": body.input,
                    "recipient_id": body.subject_id,
                },
                request_id=request_id,
                trace_id=trace_id,
            ),
        )
        return {
            "inbox_id": accepted.inbox.inbox_id,
            "execution_id": accepted.inbox.execution_id,
            "status": accepted.inbox.status.value,
            "request_id": accepted.inbox.request_id,
        }

    def dispatch(
        self, tenant_id: str, request_id: str = "", trace_id: str = ""
    ) -> list[dict[str, Any]]:
        return [
            _json(event)
            for event in self.runtime.dispatch_once(
                _context(tenant_id, request_id, trace_id), self.settings.dispatcher_id
            )
        ]

    def dispatch_all(
        self, request_id: str = "", trace_id: str = ""
    ) -> dict[str, list[dict[str, Any]]]:
        """Lease/publish pending Outbox work independently for every active tenant."""

        published: dict[str, list[dict[str, Any]]] = {}
        for tenant_id in self.control.tenant_ids():
            self.runtime.reap_expired_reservations(_context(tenant_id, request_id, trace_id))
            published[tenant_id] = self.dispatch(tenant_id, request_id, trace_id)
        return published

    async def process_published(
        self, tenant_id: str | None = None, request_id: str = "", trace_id: str = ""
    ) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        if isinstance(self.bus, RedisStreamMessageBus):
            group = "trpc-agent-workers"
            consumer = self.settings.worker_id
            messages = self.bus.reclaim_idle(group, consumer, min_idle_ms=60_000)
            messages.extend(self.bus.consume(group, consumer, block_ms=250))
            for message_id, record in messages:
                event_type = str(record.get("event_type", ""))
                record_tenant = str(record.get("tenant_id", ""))
                try:
                    if event_type == "inbound.dispatch" and record_tenant:
                        inbox_id = str(
                            record.get("inbox_id") or record.get("payload", {}).get("inbox_id", "")
                        )
                        if inbox_id:
                            outputs.append(
                                await self.execute_inbox(
                                    _context(record_tenant, request_id, trace_id, actor="worker"),
                                    inbox_id,
                                    self.settings.worker_id,
                                )
                            )
                    self.bus.acknowledge(group, message_id)
                except Exception:
                    # Leave the Stream entry pending.  Redis will make it
                    # reclaimable; SQL Inbox/fence state suppresses re-effects.
                    continue
            return outputs
        if tenant_id is None:
            raise ValueError("memory worker requires an explicit tenant_id")
        context = _context(tenant_id, request_id, trace_id, actor="worker")
        for _, event in list(self.bus.published):
            if event.tenant_id != tenant_id or event.outbox_id in self.processed_bus_events:
                continue
            self.processed_bus_events.add(event.outbox_id)
            if event.event_type == "inbound.dispatch" and event.inbox_id:
                outputs.append(
                    await self.execute_inbox(context, event.inbox_id, self.settings.worker_id)
                )
        return outputs

    async def deliver_mock_replies(
        self, tenant_id: str, request_id: str = "", trace_id: str = ""
    ) -> list[dict[str, Any]]:
        """Compatibility entry point for the local demo's mock delivery."""

        return await self.deliver_replies(tenant_id, request_id, trace_id)

    async def _deliver_outbox(
        self, record: Mapping[str, Any], *, provider_filter: str | None = None
    ) -> tuple[dict[str, Any], bool]:
        """Attempt one published reply and state whether its Stream entry is final.

        Failed known requests remain unacknowledged in Redis so they are reclaimed
        after the consumer-idle lease.  Ambiguous effects are recorded for
        reconciliation/manual review and acknowledged to prevent an unsafe replay.
        """

        tenant_id = str(record["tenant_id"])
        outbox_id = str(record["outbox_id"])
        payload = record.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError("reply outbox payload must be an object")
        binding_id = str(payload.get("channel_binding_id") or "")
        if not binding_id or binding_id == "api":
            self.runtime.mark_outbox_delivered(
                _context(tenant_id, "api-reply", str(record.get("trace_id") or ""), actor="dispatcher"),
                outbox_id,
            )
            return {"outbox_id": outbox_id, "status": "api_reply"}, True
        binding = self.control.binding(tenant_id, binding_id)
        # Smart Bot replies must be emitted by the singleton gateway that owns
        # the WebSocket.  General dispatcher replicas acknowledge their own
        # consumer-group copy without ever attempting a second connection.
        if provider_filter is not None and binding.provider != provider_filter:
            return {"outbox_id": outbox_id, "status": "provider_skipped"}, True
        if provider_filter is None and binding.provider == "wecom_aibot":
            return {"outbox_id": outbox_id, "status": "aibot_gateway_required"}, True
        adapter = self.adapters.get(binding.provider)
        if adapter is None:
            raise ChannelError("unsupported_provider", f"no adapter for {binding.provider}")
        recipient_id = str(payload.get("recipient_id") or "")
        if not recipient_id:
            raise ChannelError("missing_recipient", "reply outbox has no recipient", retryable=False)
        reply = ReplyEnvelope(
            tenant_id=tenant_id,
            binding_id=binding.binding_id,
            session_id=str(payload.get("session_id", "")),
            recipient_id=recipient_id,
            delivery_id=outbox_id,
            blocks=tuple(
                ReplyBlock.text_block(str(block.get("text", "")))
                for block in payload.get("blocks", [])
                if isinstance(block, Mapping)
            ),
            traceparent=f"00-{str(record.get('trace_id') or '0' * 32)[:32]}-{'0' * 16}-01",
            metadata=dict(payload.get("metadata") or {}),
        )
        capability_by_provider = {
            "mock": "idempotent",
            "telegram": "non_retriable",
            # The corporate-app send API does not expose a result query by
            # message ID, so treating it as queryable would invite duplicate
            # sends after a timeout.
            "wecom": "non_retriable",
            "wecom_aibot": "non_retriable",
        }
        capability = capability_by_provider.get(binding.provider, "non_retriable")
        attempt = self.delivery_ledger.begin(
            tenant_id=tenant_id,
            outbox_id=outbox_id,
            session_id=reply.session_id,
            channel_binding_id=binding.binding_id,
            capability=capability,
            request_hash=sha256(reply.plain_text().encode()).hexdigest(),
            trace_id=str(record.get("trace_id") or ""),
        )
        if attempt.status != "sending":
            if attempt.status == "accepted":
                self.runtime.mark_outbox_delivered(
                    _context(tenant_id, "delivery-recovery", attempt.trace_id, actor="dispatcher"), outbox_id
                )
                return _json(attempt), True
            if capability == "queryable":
                reconcile = getattr(adapter, "reconcile_delivery", None)
                if not callable(reconcile):
                    final = self.delivery_ledger.finish(
                        attempt,
                        status="manual_review",
                        error_code="queryable_adapter_has_no_reconciliation_operation",
                    )
                    return _json(final), True
                try:
                    reconciled = await reconcile(
                        self.adapter_binding(binding),
                        reply,
                        provider_message_id=attempt.provider_message_id,
                        provider_idempotency_key=attempt.provider_idempotency_key,
                    )
                except Exception as exc:
                    final = self.delivery_ledger.finish(
                        attempt,
                        status="reconciling",
                        error_code=f"reconciliation_{type(exc).__name__}",
                    )
                    return _json(final), False
                if reconciled is None or reconciled.status == "reconciling":
                    final = self.delivery_ledger.finish(
                        attempt,
                        status="reconciling",
                        error_code=(reconciled.error_code if reconciled else "reconciliation_inconclusive"),
                    )
                    return _json(final), False
                if reconciled.status == "accepted":
                    final = self.delivery_ledger.finish(
                        attempt,
                        status="accepted",
                        provider_message_id=reconciled.provider_message_id,
                    )
                    self.runtime.mark_outbox_delivered(
                        _context(tenant_id, "delivery-reconciled", final.trace_id, actor="dispatcher"),
                        outbox_id,
                    )
                    return _json(final), True
                final = self.delivery_ledger.finish(
                    attempt,
                    status="manual_review",
                    error_code=reconciled.error_code or "reconciliation_rejected",
                )
                return _json(final), True
            return _json(attempt), True
        try:
            result = await adapter.deliver(self.adapter_binding(binding), reply)
            final = self.delivery_ledger.finish(
                attempt,
                status=result.status,  # type: ignore[arg-type]
                provider_message_id=result.provider_message_id,
                error_code=result.error_code,
            )
        except ChannelError as exc:
            final = self.delivery_ledger.finish(
                attempt,
                status="failed" if exc.retryable else "unknown",
                error_code=exc.code,
            )
        except Exception as exc:
            final = self.delivery_ledger.finish(
                attempt,
                status="unknown",
                error_code=type(exc).__name__,
            )
        if final.status == "accepted":
            self.runtime.mark_outbox_delivered(
                _context(tenant_id, "delivery", final.trace_id, actor="dispatcher"), outbox_id
            )
            return _json(final), True
        # Only a provider with a deterministic idempotency key may be resent
        # automatically. Queryable providers stay pending for their actual
        # reconciliation operation; non-retriable providers require a human.
        if capability == "idempotent" and final.status == "failed":
            return _json(final), False
        if capability == "queryable" and final.status in {"failed", "unknown", "reconciling"}:
            return _json(final), False
        return _json(final), True

    async def deliver_replies(
        self,
        tenant_id: str,
        request_id: str = "",
        trace_id: str = "",
        *,
        provider_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Deliver currently published replies for one tenant (memory/demo path)."""

        context = _context(tenant_id, request_id, trace_id, actor="dispatcher")
        snapshot = self.runtime.snapshot(context)
        outcomes: list[dict[str, Any]] = []
        for outbox in snapshot["outbox"]:
            if outbox["event_type"] != "reply.dispatch" or outbox["status"] != "published":
                continue
            outcome, _ = await self._deliver_outbox(outbox, provider_filter=provider_filter)
            outcomes.append(outcome)
        return outcomes

    async def process_delivery_published(
        self,
        request_id: str = "",
        trace_id: str = "",
        *,
        provider_filter: str | None = None,
        consumer_group: str | None = None,
    ) -> list[dict[str, Any]]:
        """Consume reply events in a separate Redis group from model workers."""

        if not isinstance(self.bus, RedisStreamMessageBus):
            return [
                outcome
                for tenant_id in self.control.tenant_ids()
                for outcome in await self.deliver_replies(
                    tenant_id, request_id, trace_id, provider_filter=provider_filter
                )
            ]
        group = consumer_group or "trpc-agent-delivery"
        consumer = self.settings.dispatcher_id
        messages = self.bus.reclaim_idle(group, consumer, min_idle_ms=60_000)
        messages.extend(self.bus.consume(group, consumer, block_ms=250))
        outcomes: list[dict[str, Any]] = []
        for message_id, record in messages:
            if str(record.get("event_type", "")) != "reply.dispatch":
                self.bus.acknowledge(group, message_id)
                continue
            try:
                outcome, final = await self._deliver_outbox(record, provider_filter=provider_filter)
                outcomes.append(outcome)
                if final:
                    self.bus.acknowledge(group, message_id)
            except Exception:
                # The entry stays pending for a leased retry.  The SQL attempt
                # ledger is written before any provider call when possible.
                continue
        return outcomes


def create_app(container: ServiceContainer | None = None) -> FastAPI:
    # A supplied container is used by tests and embedded deployments.  Its
    # settings must also govern HTTP authentication instead of consulting the
    # process-global cached settings a second time.
    settings = container.settings if container is not None else get_settings()
    settings.validate_startup()
    services = container or ServiceContainer(settings)
    # Do not expose framework-generated routes that would evade the API
    # authorization policy.  Publish an authenticated specification separately if
    # this service later needs an interactive developer portal.
    app = FastAPI(
        title="tRPC-Agent multi-tenant platform",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.services = services
    app.state.otel_enabled = configure_opentelemetry(
        service_name="trpc-agent-service",
        otlp_endpoint=services.settings.otlp_endpoint,
        fastapi_app=app,
    )

    @app.middleware("http")
    async def trace_request(request: Request, call_next):
        trace = extract_trace_context(request.headers)
        with bind_trace_context(trace):
            response = await call_next(request)
        response.headers["x-request-id"] = trace.request_id
        response.headers["traceparent"] = trace.traceparent
        return response

    def get_services() -> ServiceContainer:
        return app.state.services

    def authentication_error(code: str, response_status: int) -> HTTPException:
        return HTTPException(status_code=response_status, detail={"code": code})

    def bearer_or_api_key(request: Request) -> str | None:
        authorization = request.headers.get("authorization")
        if authorization:
            scheme, _, credential = authorization.partition(" ")
            if scheme.lower() != "bearer" or not credential:
                raise authentication_error("invalid_authorization_header", status.HTTP_401_UNAUTHORIZED)
            return credential
        return request.headers.get("x-api-key")

    def authenticate(request: Request) -> AuthPrincipal:
        """Authenticate without ever treating missing configuration as anonymous."""

        if not settings.authentication_configured:
            raise authentication_error("authentication_unconfigured", status.HTTP_503_SERVICE_UNAVAILABLE)

        admin_credential = request.headers.get("x-admin-key")
        if admin_credential and settings.admin_api_key and compare_digest(
            admin_credential, settings.admin_api_key
        ):
            return AuthPrincipal(kind="admin")

        credential = bearer_or_api_key(request)
        if credential and settings.admin_api_key and compare_digest(
            credential, settings.admin_api_key
        ):
            return AuthPrincipal(kind="admin")
        if credential:
            for tenant_id, tenant_key in settings.tenant_api_keys.items():
                if compare_digest(credential, tenant_key):
                    return AuthPrincipal(kind="tenant", tenant_id=tenant_id)
        raise authentication_error("authentication_required", status.HTTP_401_UNAUTHORIZED)

    def require_admin(request: Request) -> AuthPrincipal:
        principal = authenticate(request)
        if not principal.is_admin:
            raise authentication_error("admin_authorization_required", status.HTTP_403_FORBIDDEN)
        return principal

    def require_tenant(tenant_id: str, request: Request) -> AuthPrincipal:
        principal = authenticate(request)
        if not principal.is_admin and principal.tenant_id != tenant_id:
            raise authentication_error("tenant_authorization_required", status.HTTP_403_FORBIDDEN)
        return principal

    @app.exception_handler(RuntimeErrorBase)
    async def runtime_error(_: Request, exc: RuntimeErrorBase) -> JSONResponse:
        return JSONResponse(status_code=409, content={"code": exc.code, "detail": str(exc)})

    @app.exception_handler(ControlNotFound)
    async def control_not_found(_: Request, exc: ControlNotFound) -> JSONResponse:
        return JSONResponse(status_code=404, content={"code": exc.code, "detail": str(exc)})

    @app.exception_handler(ControlConflict)
    async def control_conflict(_: Request, exc: ControlConflict) -> JSONResponse:
        return JSONResponse(status_code=409, content={"code": exc.code, "detail": str(exc)})

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "live"}

    @app.get("/health/ready")
    async def ready() -> JSONResponse:
        is_ready, details = services.readiness()
        return JSONResponse(
            status_code=status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "ready" if is_ready else "not_ready", **details},
        )

    @app.post(
        "/admin/v1/tenants",
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(require_admin)],
    )
    async def create_tenant(
        body: TenantCreate, request: Request, services: ServiceContainer = Depends(get_services)
    ) -> TenantResponse:
        return TenantResponse(**services.create_tenant(body))

    @app.get("/admin/v1/tenants/{tenant_id}", dependencies=[Depends(require_admin)])
    async def get_tenant(
        tenant_id: str, request: Request, services: ServiceContainer = Depends(get_services)
    ) -> TenantResponse:
        trace = extract_trace_context(request.headers)
        return TenantResponse(**services.tenant_view(tenant_id, trace.request_id, trace.trace_id))

    @app.post(
        "/admin/v1/tenants/{tenant_id}/agents",
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(require_admin)],
    )
    async def create_agent(
        tenant_id: str, body: AgentCreate, services: ServiceContainer = Depends(get_services)
    ) -> dict[str, Any]:
        return services.create_agent(tenant_id, body)

    @app.post(
        "/admin/v1/tenants/{tenant_id}/agents/{agent_id}/releases",
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(require_admin)],
    )
    async def create_release(
        tenant_id: str,
        agent_id: str,
        body: ReleaseCreate,
        services: ServiceContainer = Depends(get_services),
    ) -> dict[str, Any]:
        return services.create_release(tenant_id, agent_id, body)

    @app.post(
        "/admin/v1/tenants/{tenant_id}/agents/{agent_id}/releases/{version}/activate",
        dependencies=[Depends(require_admin)],
    )
    async def activate_release(
        tenant_id: str,
        agent_id: str,
        version: int,
        services: ServiceContainer = Depends(get_services),
    ) -> dict[str, Any]:
        return services.activate_release(tenant_id, agent_id, version)

    @app.post(
        "/admin/v1/tenants/{tenant_id}/agents/{agent_id}/rollback",
        dependencies=[Depends(require_admin)],
    )
    async def rollback(
        tenant_id: str, agent_id: str, services: ServiceContainer = Depends(get_services)
    ) -> dict[str, Any]:
        return services.rollback(tenant_id, agent_id)

    @app.post(
        "/admin/v1/tenants/{tenant_id}/channels",
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(require_admin)],
    )
    async def create_channel(
        tenant_id: str, body: ChannelCreate, services: ServiceContainer = Depends(get_services)
    ) -> dict[str, Any]:
        return services.create_binding(tenant_id, body)

    @app.post(
        "/admin/v1/tenants/{tenant_id}/security/suspend", dependencies=[Depends(require_admin)]
    )
    async def suspend(
        tenant_id: str,
        request: Request,
        emergency: bool = False,
        services: ServiceContainer = Depends(get_services),
    ) -> dict[str, Any]:
        trace = extract_trace_context(request.headers)
        return _json(
            services.runtime.suspend(
                _context(tenant_id, trace.request_id, trace.trace_id), emergency=emergency
            )
        )

    @app.post(
        "/admin/v1/tenants/{tenant_id}/security/tools/{tool_name}/revoke",
        dependencies=[Depends(require_admin)],
    )
    async def revoke_tool(
        tenant_id: str,
        tool_name: str,
        request: Request,
        services: ServiceContainer = Depends(get_services),
    ) -> dict[str, Any]:
        trace = extract_trace_context(request.headers)
        return _json(
            services.runtime.revoke_tool(
                _context(tenant_id, trace.request_id, trace.trace_id), tool_name
            )
        )

    @app.post("/admin/v1/tenants/{tenant_id}/budgets", dependencies=[Depends(require_admin)])
    async def create_budget(
        tenant_id: str,
        body: BudgetCreate,
        request: Request,
        services: ServiceContainer = Depends(get_services),
    ) -> dict[str, Any]:
        trace = extract_trace_context(request.headers)
        account = BudgetAccount(
            tenant_id=tenant_id,
            budget_name=body.budget_name,
            unit=body.unit,
            limit_units=body.limit_units,
            period_start=body.period_start
            or BudgetAccount.__dataclass_fields__["period_start"].default_factory(),
            period_end=body.period_end
            or BudgetAccount.__dataclass_fields__["period_end"].default_factory(),
        )
        return _json(
            services.runtime.put_budget_account(
                _context(tenant_id, trace.request_id, trace.trace_id), account
            )
        )

    @app.post(
        "/admin/v1/tenants/{tenant_id}/migrations",
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(require_admin)],
    )
    async def initiate_migration(
        tenant_id: str,
        body: MigrationCreate,
        request: Request,
        services: ServiceContainer = Depends(get_services),
    ) -> dict[str, Any]:
        trace = extract_trace_context(request.headers)
        return _json(
            services.runtime.initiate_migration(
                _context(tenant_id, trace.request_id, trace.trace_id),
                {"profile": body.target_profile},
                migration_id=stable_id("mig", tenant_id, body.source_profile, body.target_profile),
            )
        )

    @app.get(
        "/admin/v1/tenants/{tenant_id}/migrations/{migration_id}",
        dependencies=[Depends(require_admin)],
    )
    async def get_migration(
        tenant_id: str,
        migration_id: str,
        request: Request,
        services: ServiceContainer = Depends(get_services),
    ) -> dict[str, Any]:
        trace = extract_trace_context(request.headers)
        return _json(
            services.runtime.get_migration(
                _context(tenant_id, trace.request_id, trace.trace_id), migration_id
            )
        )

    @app.post(
        "/admin/v1/tenants/{tenant_id}/migrations/{migration_id}:action",
        dependencies=[Depends(require_admin)],
    )
    async def migration_action(
        tenant_id: str,
        migration_id: str,
        body: MigrationAction,
        request: Request,
        services: ServiceContainer = Depends(get_services),
    ) -> dict[str, Any]:
        trace = extract_trace_context(request.headers)
        context = _context(tenant_id, trace.request_id, trace.trace_id)
        if body.action == "prepare":
            return _json(services.runtime.get_migration(context, migration_id))
        if body.action == "rollback":
            migration = services.runtime.get_migration(context, migration_id)
            runtime_action = (
                "begin_rollback" if migration.status.value == "active" else "complete_rollback"
            )
        else:
            runtime_action = {
                "backfill": "start_backfill",
                "catch_up": "catch_up",
                "drain": "begin_drain",
                "verify": "verify",
                "cutover": "cutover",
                "cancel": "cancel",
            }[body.action]
        return _json(
            services.runtime.migration_action(
                context,
                migration_id,
                runtime_action,
                source_watermark=body.source_watermark,
                target_watermark=body.target_watermark,
                verified=body.verified,
            )
        )

    async def callback(
        provider: str, binding_key: str, request: Request, services: ServiceContainer
    ) -> dict[str, Any]:
        trace = extract_trace_context(request.headers)
        payload = await request.body()
        try:
            return await services.accept_callback(
                provider,
                binding_key,
                CallbackRequest(
                    body=payload, headers=dict(request.headers), query=dict(request.query_params)
                ),
                trace.request_id,
                trace.trace_id,
            )
        except ChannelError as exc:
            raise HTTPException(
                status_code=exc.status_code, detail={"code": exc.code, "retryable": exc.retryable}
            ) from exc

    @app.post("/callbacks/wecom/{binding_key}")
    async def wecom_callback(
        binding_key: str, request: Request, services: ServiceContainer = Depends(get_services)
    ) -> dict[str, Any]:
        return await callback("wecom", binding_key, request, services)

    @app.post("/callbacks/telegram/{binding_key}")
    async def telegram_callback(
        binding_key: str, request: Request, services: ServiceContainer = Depends(get_services)
    ) -> dict[str, Any]:
        return await callback("telegram", binding_key, request, services)

    @app.post("/callbacks/mock/{binding_key}")
    async def mock_callback(
        binding_key: str, request: Request, services: ServiceContainer = Depends(get_services)
    ) -> dict[str, Any]:
        return await callback("mock", binding_key, request, services)

    @app.post(
        "/v1/tenants/{tenant_id}/agents/{agent_id}:run",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_tenant)],
    )
    async def run_agent(
        tenant_id: str,
        agent_id: str,
        body: RunRequest,
        request: Request,
        services: ServiceContainer = Depends(get_services),
    ) -> OperationResponse:
        trace = extract_trace_context(request.headers)
        result = await services.run_direct(
            tenant_id, agent_id, body, trace.request_id, trace.trace_id
        )
        operation_request_id = str(result.pop("request_id", trace.request_id))
        return OperationResponse(request_id=operation_request_id, trace_id=trace.trace_id, **result)

    @app.get("/v1/operations/{request_id}", dependencies=[Depends(require_tenant)])
    async def operation(
        request_id: str,
        tenant_id: str,
        request: Request,
        services: ServiceContainer = Depends(get_services),
    ) -> dict[str, Any]:
        trace = extract_trace_context(request.headers)
        snapshot = services.runtime.snapshot(_context(tenant_id, trace.request_id, trace.trace_id))
        items = [row for row in snapshot["inboxes"] if row["request_id"] == request_id]
        replies = {
            str(event["payload"].get("inbox_id")): str(event["payload"].get("text", ""))
            for event in snapshot["events"]
            if event["event_type"] == "reply.text"
            and isinstance(event.get("payload"), Mapping)
            and event["payload"].get("inbox_id")
        }
        return {
            "items": [{**item, "reply": replies.get(str(item["inbox_id"]))} for item in items],
            "request_id": request_id,
        }

    @app.get("/admin/v1/tenants/{tenant_id}/audit", dependencies=[Depends(require_admin)])
    async def audit(
        tenant_id: str, request: Request, services: ServiceContainer = Depends(get_services)
    ) -> Page:
        trace = extract_trace_context(request.headers)
        return Page(
            items=services.runtime.snapshot(_context(tenant_id, trace.request_id, trace.trace_id))[
                "audit"
            ]
        )

    @app.get(
        "/admin/v1/tenants/{tenant_id}/unknown-operations", dependencies=[Depends(require_admin)]
    )
    async def unknown_operations(
        tenant_id: str, request: Request, services: ServiceContainer = Depends(get_services)
    ) -> Page:
        trace = extract_trace_context(request.headers)
        snapshot = services.runtime.snapshot(_context(tenant_id, trace.request_id, trace.trace_id))
        tools = [row for row in snapshot["tools"] if row["status"] in {"unknown", "manual_review"}]
        deliveries = [_json(row) for row in services.delivery_ledger.unknown(tenant_id)]
        return Page(items=tools + deliveries)

    @app.post(
        "/admin/v1/tenants/{tenant_id}/unknown-operations/{operation_id}:resolve",
        dependencies=[Depends(require_admin)],
    )
    async def resolve_unknown(
        tenant_id: str,
        operation_id: str,
        body: ResolutionRequest,
        request: Request,
        services: ServiceContainer = Depends(get_services),
    ) -> dict[str, Any]:
        # A human resolution is auditable and explicit.  It never silently retries a
        # non-idempotent provider operation; retry is merely recorded for an admin.
        trace = extract_trace_context(request.headers)
        snapshot = services.runtime.snapshot(_context(tenant_id, trace.request_id, trace.trace_id))
        known = next(
            (row for row in snapshot["tools"] if row["tool_call_id"] == operation_id), None
        )
        if known is None:
            try:
                resolved = services.delivery_ledger.resolve(
                    tenant_id, operation_id, action=body.action, note=body.note
                )
                if body.action == "retry" and resolved.status == "reconciling":
                    services.runtime.requeue_outbox(
                        _context(tenant_id, trace.request_id, trace.trace_id, actor="admin"),
                        resolved.outbox_id,
                    )
                return _json(resolved)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail={"code": "not_found"}) from exc
        target = {
            "accepted": ToolStatus.SUCCEEDED,
            "failed": ToolStatus.FAILED,
            "manual_review": ToolStatus.MANUAL_REVIEW,
            # Retry is a human-authorized reconciliation state. It does not call
            # the provider from this endpoint, especially for non-retriable Tool.
            "retry": ToolStatus.RECONCILING,
        }[body.action]
        resolved = services.runtime.resolve_tool(
            _context(tenant_id, trace.request_id, trace.trace_id, actor="admin"),
            operation_id,
            status=target,
            note=body.note,
        )
        return _json(resolved)

    return app


app = create_app()
