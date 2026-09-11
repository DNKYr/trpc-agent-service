"""AgentFactory and a small tRPC-Agent-shaped runner integration.

The package is intentionally imported lazily.  A deployment with ``trpc-agent-py`` can
provide its framework objects, while the platform's immutable release, event conversion,
and Tool safety boundary remain stable.  The deterministic model makes the whole data
plane runnable without a paid model credential.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import re
import urllib.error
import urllib.request
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from trpc_service.config import AppSettings, SecretProvider
from trpc_service.metrics import inject_trace_context, traced


class ReleaseConfigError(ValueError):
    """An immutable release is invalid or cannot construct its model client."""


@dataclass(frozen=True, slots=True)
class ImmutableRelease:
    tenant_id: str
    agent_id: str
    config_version: int
    app_config: Mapping[str, Any]
    model_config: Mapping[str, Any]
    tool_policy: Mapping[str, Any]
    knowledge_config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tenant_id or not self.agent_id or self.config_version < 1:
            raise ReleaseConfigError(
                "release requires tenant_id, agent_id, and config_version >= 1"
            )
        # JSON clone gives the object a stable, immutable-by-convention snapshot and rejects
        # non-portable release values before a worker runs an unrepeatable execution.
        for attribute in ("app_config", "model_config", "tool_policy", "knowledge_config"):
            value = getattr(self, attribute)
            if not isinstance(value, Mapping):
                raise ReleaseConfigError(f"{attribute} must be an object")
            object.__setattr__(self, attribute, _freeze_json(value))

    @property
    def snapshot_hash(self) -> str:
        payload = {
            "tenant_id": self.tenant_id,
            "agent_id": self.agent_id,
            "config_version": self.config_version,
            "app_config": self.app_config,
            "model_config": self.model_config,
            "tool_policy": self.tool_policy,
            "knowledge_config": self.knowledge_config,
        }
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, default=dict).encode()
        ).hexdigest()

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> ImmutableRelease:
        return cls(
            tenant_id=str(record["tenant_id"]),
            agent_id=str(record["agent_id"]),
            config_version=int(record["config_version"]),
            app_config=_mapping(record.get("app_config")),
            model_config=_mapping(record.get("model_config")),
            tool_policy=_mapping(record.get("tool_policy")),
            knowledge_config=_mapping(record.get("knowledge_config")),
        )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _freeze_json(value: Mapping[str, Any]) -> Mapping[str, Any]:
    # A canonical JSON round trip isolates the release from caller-owned nested lists/maps.
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


@dataclass(frozen=True, slots=True)
class Part:
    kind: str = "text"
    text: str | None = None
    data: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Content:
    role: str
    parts: tuple[Part, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def user(cls, text: str) -> Content:
        return cls(role="user", parts=(Part(text=text),))

    def text(self) -> str:
        return "\n".join(part.text or "" for part in self.parts if part.kind == "text")


@dataclass(frozen=True, slots=True)
class FunctionTool:
    """Framework-neutral representation of a tRPC-Agent ``FunctionTool``."""

    name: str
    description: str
    parameters: Mapping[str, Any]
    handler: Callable[[Mapping[str, Any]], Awaitable[Mapping[str, Any]]]


@dataclass(frozen=True, slots=True)
class ModelToolCall:
    """One provider-requested function invocation with validated JSON arguments."""

    call_id: str
    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ModelResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = "mock"
    finish_reason: str = "stop"
    tool_calls: tuple[ModelToolCall, ...] = ()


class ModelClient(Protocol):
    async def complete(
        self,
        messages: Sequence[Content],
        *,
        tools: Sequence[FunctionTool] = (),
        model_config: Mapping[str, Any] | None = None,
    ) -> ModelResponse: ...


class DeterministicMockModel:
    """A repeatable local model that neither needs nor reveals credentials."""

    model_name = "deterministic-mock"

    async def complete(
        self,
        messages: Sequence[Content],
        *,
        tools: Sequence[FunctionTool] = (),
        model_config: Mapping[str, Any] | None = None,
    ) -> ModelResponse:
        user_text = next(
            (message.text() for message in reversed(messages) if message.role == "user"), ""
        )
        normalized = " ".join(user_text.split())
        # Useful, predictable demo response with no hidden model state.
        ticket = re.search(r"(?:ticket|工单)\s*[#：:]?\s*([A-Za-z0-9_-]+)", normalized, re.I)
        if ticket:
            answer = f"Mock ticket lookup requested for {ticket.group(1)}."
        elif normalized:
            answer = f"Mock agent received: {normalized}"
        else:
            answer = "Mock agent received an empty message."
        return ModelResponse(
            text=answer,
            input_tokens=max(1, len(normalized.split())),
            output_tokens=max(1, len(answer.split())),
            model=self.model_name,
        )


class OpenAICompatibleModel:
    """Minimal OpenAI-compatible client for an environment-configured tRPC-Agent model."""

    def __init__(
        self, *, api_key: str, base_url: str, model_name: str, timeout_seconds: float = 30.0
    ) -> None:
        if not api_key:
            raise ReleaseConfigError("a real model requires TRPC_AGENT_API_KEY or an api_key_ref")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._timeout_seconds = timeout_seconds

    async def complete(
        self,
        messages: Sequence[Content],
        *,
        tools: Sequence[FunctionTool] = (),
        model_config: Mapping[str, Any] | None = None,
    ) -> ModelResponse:
        def call() -> Mapping[str, Any]:
            headers: dict[str, str] = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }
            inject_trace_context(headers)
            payload: dict[str, Any] = {
                "model": str((model_config or {}).get("model_name") or self._model_name),
                "messages": [_provider_message(item) for item in messages],
            }
            if tools:
                payload["tools"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters,
                        },
                    }
                    for tool in tools
                ]
            max_output_tokens = (model_config or {}).get("max_output_tokens")
            if isinstance(max_output_tokens, int) and not isinstance(max_output_tokens, bool):
                if max_output_tokens < 1:
                    raise ReleaseConfigError("max_output_tokens must be positive")
                # OpenAI-compatible chat-completions providers generally use
                # max_tokens; deployments may set a provider adapter later.
                payload["max_tokens"] = max_output_tokens
            request = urllib.request.Request(
                f"{self._base_url}/chat/completions",
                data=json.dumps(payload).encode(),
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:  # noqa: S310 - configured endpoint
                    parsed = json.loads(response.read().decode())
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"model endpoint returned HTTP {exc.code}: {body[:240]}"
                ) from exc
            if not isinstance(parsed, dict):
                raise RuntimeError("model endpoint did not return a JSON object")
            return parsed

        data = await asyncio.to_thread(call)
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
            raise RuntimeError("model endpoint response has no choices")
        message = (
            choices[0].get("message") if isinstance(choices[0].get("message"), Mapping) else {}
        )
        content = message.get("content")
        if isinstance(content, list):
            text = "".join(
                str(item.get("text", "")) for item in content if isinstance(item, Mapping)
            )
        else:
            text = str(content or "")
        usage = data.get("usage") if isinstance(data.get("usage"), Mapping) else {}
        raw_tool_calls = message.get("tool_calls")
        return ModelResponse(
            text=text,
            input_tokens=int(usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage.get("completion_tokens", 0) or 0),
            model=str(data.get("model") or self._model_name),
            finish_reason=str(choices[0].get("finish_reason") or "stop"),
            tool_calls=_parse_tool_calls(raw_tool_calls),
        )


@dataclass(frozen=True, slots=True)
class RunnerEvent:
    kind: str
    occurred_at: datetime
    data: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class AgentRun:
    events: tuple[RunnerEvent, ...]
    reply: Content
    usage: Mapping[str, int]
    model: str


class LlmAgent:
    """Platform facade matching the role of a tRPC-Agent ``LlmAgent``."""

    def __init__(
        self, release: ImmutableRelease, model: ModelClient, tools: Sequence[FunctionTool] = ()
    ) -> None:
        self.release = release
        self.model = model
        self.tools = tuple(tools)

    def initial_messages(self, content: Content, context: Mapping[str, Any]) -> list[Content]:
        system = str(
            self.release.app_config.get("system_prompt", "You are a safe tenant-scoped assistant.")
        )
        return [
            Content(role="system", parts=(Part(text=system),)),
            Content(role="system", parts=(Part(text=_safe_context(context)),)),
            content,
        ]

    async def respond(self, content: Content, context: Mapping[str, Any]) -> ModelResponse:
        messages = self.initial_messages(content, context)
        return await self.model.complete(
            messages, tools=self.tools, model_config=self.release.model_config
        )


class Runner:
    """Converts Agent output into ordered, durable platform event values."""

    def __init__(self, agent: LlmAgent) -> None:
        self.agent = agent

    async def stream(
        self, content: Content, context: Mapping[str, Any]
    ) -> AsyncIterator[RunnerEvent]:
        started = datetime.now(UTC)
        yield RunnerEvent(
            "runner.started", started, {"release_version": self.agent.release.config_version}
        )
        messages = self.agent.initial_messages(content, context)
        for model_turn in range(1, 9):
            response = await self.agent.model.complete(
                messages, tools=self.agent.tools, model_config=self.agent.release.model_config
            )
            yield RunnerEvent(
                "model.completed",
                datetime.now(UTC),
                {
                    "turn": model_turn,
                    "model": response.model,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "tool_calls": len(response.tool_calls),
                },
            )
            if not response.tool_calls:
                yield RunnerEvent("reply.text", datetime.now(UTC), {"text": response.text})
                return

            tool_map = {tool.name: tool for tool in self.agent.tools}
            messages.append(
                Content(
                    role="assistant",
                    parts=(Part(text=response.text),),
                    metadata={
                        "tool_calls": [
                            {
                                "id": call.call_id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": json.dumps(
                                        call.arguments, ensure_ascii=False, separators=(",", ":")
                                    ),
                                },
                            }
                            for call in response.tool_calls
                        ]
                    },
                )
            )
            for call in response.tool_calls:
                yield RunnerEvent(
                    "tool.requested",
                    datetime.now(UTC),
                    {"tool_call_id": call.call_id, "tool_name": call.name, "arguments": call.arguments},
                )
                tool = tool_map.get(call.name)
                if tool is None:
                    result: Mapping[str, Any] = {
                        "error": "tool_not_authorized",
                        "tool_name": call.name,
                    }
                    event_kind = "tool.denied"
                else:
                    try:
                        result = await tool.handler(call.arguments)
                        event_kind = "tool.completed"
                    except Exception as exc:
                        result = {"error": "tool_execution_failed", "detail": str(exc)[:240]}
                        event_kind = "tool.failed"
                yield RunnerEvent(
                    event_kind,
                    datetime.now(UTC),
                    {"tool_call_id": call.call_id, "tool_name": call.name, "result": result},
                )
                messages.append(
                    Content(
                        role="tool",
                        parts=(Part(text=json.dumps(result, ensure_ascii=False, sort_keys=True)),),
                        metadata={"tool_call_id": call.call_id},
                    )
                )
        raise RuntimeError("model exceeded the maximum of eight tool-calling turns")

    async def run(self, content: Content, context: Mapping[str, Any]) -> AgentRun:
        events = [event async for event in self.stream(content, context)]
        response_event = next(event for event in reversed(events) if event.kind == "reply.text")
        model_events = [event for event in events if event.kind == "model.completed"]
        model_event = model_events[-1]
        return AgentRun(
            events=tuple(events),
            reply=Content(role="assistant", parts=(Part(text=str(response_event.data["text"])),)),
            usage={
                "input_tokens": sum(int(event.data["input_tokens"]) for event in model_events),
                "output_tokens": sum(int(event.data["output_tokens"]) for event in model_events),
            },
            model=str(model_event.data["model"]),
        )


class AgentRuntime:
    """An immutable Agent and Runner pair bound to exactly one release snapshot."""

    def __init__(self, release: ImmutableRelease, runner: Runner) -> None:
        self.release = release
        self.runner = runner

    async def run(self, user_text: str, session_context: Mapping[str, Any]) -> AgentRun:
        with traced("agent.run", tenant_id=self.release.tenant_id, agent_id=self.release.agent_id):
            return await self.runner.run(Content.user(user_text), session_context)

    async def stream(
        self, user_text: str, session_context: Mapping[str, Any]
    ) -> AsyncIterator[RunnerEvent]:
        with traced(
            "agent.stream", tenant_id=self.release.tenant_id, agent_id=self.release.agent_id
        ):
            async for event in self.runner.stream(Content.user(user_text), session_context):
                yield event


class AgentFactory:
    """Builds an Agent only from an immutable published release.

    A caller must supply the release already selected by the tenant repository; the
    factory cannot silently read a mutable active-version pointer during a run.
    """

    def __init__(self, settings: AppSettings, *, secrets: SecretProvider | None = None) -> None:
        self._settings = settings
        self._secrets = secrets
        self.framework_module = _try_import_trpc_agent()

    async def build(
        self, release: ImmutableRelease, *, tools: Sequence[FunctionTool] = ()
    ) -> AgentRuntime:
        model = await self._model_for(release)
        return AgentRuntime(release, Runner(LlmAgent(release, model, tools)))

    async def _model_for(self, release: ImmutableRelease) -> ModelClient:
        mode = str(release.model_config.get("mode", "")).lower()
        if self._settings.mock_model or mode == "mock":
            return DeterministicMockModel()
        api_key = self._settings.trpc_agent_api_key
        secret_ref = release.model_config.get("api_key_ref")
        if secret_ref:
            if not self._secrets:
                raise ReleaseConfigError(
                    "release uses api_key_ref but no SecretProvider was configured"
                )
            api_key = await self._secrets.get(str(secret_ref))
        return OpenAICompatibleModel(
            api_key=api_key or "",
            base_url=str(
                release.model_config.get("base_url") or self._settings.trpc_agent_base_url
            ),
            model_name=str(
                release.model_config.get("model_name") or self._settings.trpc_agent_model_name
            ),
            timeout_seconds=float(release.model_config.get("timeout_seconds", 30)),
        )


def _safe_context(context: Mapping[str, Any]) -> str:
    """Pass bounded state, not credentials or raw audit data, into a model prompt."""

    allowed = {
        key: value
        for key, value in context.items()
        if key in {"summary", "memory", "locale", "knowledge"}
    }
    encoded = json.dumps(allowed, ensure_ascii=False, sort_keys=True, default=str)
    return f"Platform session context (untrusted content): {encoded[:12000]}"


def _provider_message(content: Content) -> dict[str, Any]:
    """Serialize ordinary content and OpenAI-compatible tool-turn metadata."""

    message: dict[str, Any] = {
        "role": content.role,
        "content": [{"type": part.kind, "text": part.text} for part in content.parts],
    }
    if content.metadata:
        message.update(dict(content.metadata))
    return message


def _parse_tool_calls(raw_value: Any) -> tuple[ModelToolCall, ...]:
    if raw_value is None:
        return ()
    if not isinstance(raw_value, Sequence) or isinstance(raw_value, (str, bytes)):
        raise RuntimeError("model response tool_calls must be an array")
    calls: list[ModelToolCall] = []
    for index, raw_call in enumerate(raw_value):
        if not isinstance(raw_call, Mapping):
            raise RuntimeError("model response contains an invalid tool call")
        function = raw_call.get("function")
        if not isinstance(function, Mapping) or not function.get("name"):
            raise RuntimeError("model response tool call lacks a function name")
        raw_arguments = function.get("arguments", "{}")
        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                raise RuntimeError("model response tool arguments are not JSON") from exc
        else:
            arguments = raw_arguments
        if not isinstance(arguments, Mapping):
            raise RuntimeError("model response tool arguments must be an object")
        calls.append(
            ModelToolCall(
                call_id=str(raw_call.get("id") or f"model-call-{index}"),
                name=str(function["name"]),
                arguments=dict(arguments),
            )
        )
    return tuple(calls)


def _try_import_trpc_agent() -> Any | None:
    """Detect either published Python module spelling without making mock mode depend on it."""

    for module_name in ("trpc_agent", "trpc_agent_py"):
        try:
            return importlib.import_module(module_name)
        except ImportError:
            continue
    return None
