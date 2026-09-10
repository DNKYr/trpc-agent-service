"""HTTP gateway/admin surface for the runnable reference implementation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, is_dataclass
from hashlib import sha256
from secrets import token_urlsafe
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from trpc_service.agent import AgentFactory, ImmutableRelease
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
    ToolStatus,
    stable_id,
    to_primitive,
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
    aibot_registry: AIBotRegistry = field(init=False)
    principal_filter: PrincipalFilter = field(default_factory=PrincipalFilter)
    rate_limit_filter: RateLimitFilter = field(default_factory=RateLimitFilter)
    budget_filter: BudgetFilter = field(default_factory=BudgetFilter)
    input_dlp_filter: InputDLPFilter = field(default_factory=InputDLPFilter)
    output_dlp_filter: OutputDLPFilter = field(default_factory=OutputDLPFilter)

    def __post_init__(self) -> None:
        if self.settings.runtime_backend == "postgres":
            self.control = PostgresControlPlane(self.settings.database_url)
            self.store = PostgresRuntimeStore(self.settings.database_url)
            self.bus = RedisStreamMessageBus(self.settings.redis_url)
            self.secrets = EnvironmentSecretProvider()
            self.delivery_ledger = PostgresDeliveryLedger(self.settings.database_url)
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
        return self.control.serialize(
            self.control.create_agent(tenant_id, body.agent_id, body.name)
        )

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

    async def execute_inbox(
        self, context: TenantContext, inbox_id: str, worker_id: str
    ) -> dict[str, Any]:
        """Run a claimed Inbox using its immutable release and atomically commit output."""

        snapshot = self.runtime.snapshot(context)
        inbox = next((row for row in snapshot["inboxes"] if row["inbox_id"] == inbox_id), None)
        if inbox is None:
            raise KeyError("inbox does not exist")
        claim = self.runtime.claim_execution(
            context,
            inbox_id,
            worker_id,
            lease_seconds=60,
            budget_estimates=self.budget_filter.estimates(str(inbox["payload"].get("text", ""))),
        )
        # Fence/security check immediately before the model call.  Commit repeats
        # the predicates inside its transaction so a stale worker cannot write.
        self.runtime.assert_execution(context, claim)
        release_record = self.control.active_release(context.tenant_id, str(inbox["agent_id"]))
        release = ImmutableRelease(
            tenant_id=release_record.tenant_id,
            agent_id=release_record.agent_id,
            config_version=release_record.version,
            app_config=release_record.app_config,
            model_config=release_record.model_config,
            tool_policy=release_record.tool_policy,
            knowledge_config=release_record.knowledge_config,
        )
        agent = await self.agent_factory.build(release)
        input_text = str(inbox["payload"].get("text", ""))
        self.principal_filter.check(
            Principal(context.tenant_id, str(inbox.get("subject_id") or "anonymous"))
        )
        self.rate_limit_filter.check(context.tenant_id, str(inbox.get("subject_id") or "anonymous"))
        self.input_dlp_filter.check(input_text)
        result = await agent.run(input_text, {"memory": [], "summary": ""})
        self.runtime.assert_execution(context, claim)
        events = [
            SessionEventDraft(
                event_type=event.kind,
                role="assistant" if event.kind == "reply.text" else None,
                payload=dict(event.data),
            )
            for event in result.events
        ]
        reply = result.reply.text()
        self.output_dlp_filter.check(reply)
        committed = self.runtime.commit_execution(
            context,
            claim,
            CommitInput(
                expected_session_version=claim.session_version,
                new_state={"last_reply": reply, "model": result.model},
                events=events,
                memories=[
                    MemoryIntentDraft(
                        memory_type="recent_input",
                        content=input_text,
                        subject_id=inbox.get("subject_id"),
                    )
                ],
                reply=ReplyDraft(
                    blocks=[{"type": "text", "text": reply}],
                    channel_binding_id=str(inbox["channel_binding_id"]),
                    recipient_id=str(
                        inbox["payload"].get("recipient_id")
                        or inbox.get("subject_id")
                        or ""
                    ),
                    metadata=dict(inbox["payload"].get("channel_context") or {}),
                ),
                actual_budget_units={
                    "model_tokens": result.usage["input_tokens"] + result.usage["output_tokens"]
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
        accepted = self.runtime.accept_inbound(
            context,
            InboundEnvelope(
                tenant_id=tenant_id,
                channel_binding_id="api",
                agent_id=agent_id,
                session_id=session_id,
                idempotency_key=body.idempotency_key or f"api:{request_id}",
                external_message_id=None,
                subject_id=body.subject_id,
                config_version=release.version,
                payload={"channel": "api", "text": body.input, "recipient_id": body.subject_id},
                request_id=request_id,
                trace_id=trace_id,
            ),
        )
        if accepted.duplicate:
            return {
                "inbox_id": accepted.inbox.inbox_id,
                "execution_id": accepted.inbox.execution_id,
                "status": accepted.inbox.status.value,
            }
        executed = await self.execute_inbox(
            context, accepted.inbox.inbox_id, self.settings.worker_id
        )
        return {**executed, "status": "committed"}

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

        return {
            tenant_id: self.dispatch(tenant_id, request_id, trace_id)
            for tenant_id in self.control.tenant_ids()
        }

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
            "wecom": "queryable",
            "wecom_aibot": "non_retriable",
        }
        capability = capability_by_provider.get(binding.provider, "queryable")
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
        # A provider reported a definite failure; retry it through the durable
        # consumer-group pending list.  Unknown/manual states are intentionally
        # terminal until an operator resolves the ledger entry.
        return _json(final), final.status != "failed"

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
    services = container or ServiceContainer(settings)
    app = FastAPI(title="tRPC-Agent multi-tenant platform", version="0.1.0")
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

    def require_admin(request: Request) -> None:
        configured = getattr(settings, "admin_api_key", None)
        if configured and request.headers.get("x-admin-key") != configured:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail={"code": "admin_auth_required"}
            )

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
    async def ready() -> dict[str, str]:
        return {"status": "ready", "runtime_backend": services.settings.runtime_backend}

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

    @app.post("/v1/tenants/{tenant_id}/agents/{agent_id}:run")
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
        return OperationResponse(request_id=trace.request_id, trace_id=trace.trace_id, **result)

    @app.get("/v1/operations/{request_id}")
    async def operation(
        request_id: str,
        tenant_id: str,
        request: Request,
        services: ServiceContainer = Depends(get_services),
    ) -> dict[str, Any]:
        trace = extract_trace_context(request.headers)
        snapshot = services.runtime.snapshot(_context(tenant_id, trace.request_id, trace.trace_id))
        items = [row for row in snapshot["inboxes"] if row["request_id"] == request_id]
        return {"items": items, "request_id": request_id}

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
                return _json(
                    services.delivery_ledger.resolve(
                        tenant_id, operation_id, action=body.action, note=body.note
                    )
                )
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
