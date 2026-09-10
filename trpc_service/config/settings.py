"""Environment-only application settings and secret providers.

The module intentionally has no framework dependency.  It is used by the API, worker,
and CLI before optional libraries (FastAPI, tRPC-Agent, OTEL) are initialized.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol


class SecretNotFoundError(LookupError):
    """A configured secret reference cannot be resolved."""


class SecretProvider(Protocol):
    """Resolves an opaque secret reference at the last responsible moment."""

    async def get(self, reference: str) -> str:
        """Return secret material for *reference* without logging it."""


class EnvironmentSecretProvider:
    """Secret provider for local deployment.

    References use ``env://VARIABLE_NAME`` (or ``secret://env/VARIABLE_NAME``).
    A bare environment variable name is accepted only for backwards-compatible local
    configuration.  Database configuration should contain the URI form.
    """

    def __init__(self, environ: Mapping[str, str] | None = None) -> None:
        self._environ = environ if environ is not None else os.environ

    async def get(self, reference: str) -> str:
        key = _environment_key(reference)
        try:
            value = self._environ[key]
        except KeyError as exc:
            raise SecretNotFoundError(f"secret reference is unavailable: {reference!r}") from exc
        if not value:
            raise SecretNotFoundError(f"secret reference resolved to an empty value: {reference!r}")
        return value


class MockSecretProvider:
    """Deterministic provider for tests, local demo, and credential-free mock mode."""

    def __init__(self, values: Mapping[str, str] | None = None) -> None:
        self._values = dict(values or {})

    async def get(self, reference: str) -> str:
        try:
            return self._values[reference]
        except KeyError as exc:
            raise SecretNotFoundError(
                f"mock secret reference is unavailable: {reference!r}"
            ) from exc

    def put(self, reference: str, value: str) -> None:
        self._values[reference] = value


def _environment_key(reference: str) -> str:
    if reference.startswith("env://"):
        return reference.removeprefix("env://")
    if reference.startswith("secret://env/"):
        return reference.removeprefix("secret://env/")
    if "://" in reference:
        raise SecretNotFoundError(
            f"environment provider cannot resolve non-environment reference: {reference!r}"
        )
    return reference


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class AppSettings:
    """Settings accepted by all platform processes.

    The API key is loaded from an environment variable only.  Its value is intentionally
    not exposed by repr, logging helpers, or release records.
    """

    environment: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    runtime_backend: str = "memory"
    database_url: str = "postgresql+asyncpg://trpc:trpc@localhost:5432/trpc_agent"
    redis_url: str = "redis://localhost:6379/0"
    object_store_endpoint: str = "http://localhost:9000"
    object_store_bucket: str = "trpc-artifacts"
    object_store_access_key: str | None = None
    object_store_secret_key: str | None = None
    admin_api_key: str | None = None
    worker_id: str = "local-worker"
    dispatcher_id: str = "local-dispatcher"
    trpc_agent_api_key: str | None = None
    trpc_agent_base_url: str = "https://api.openai.com/v1"
    trpc_agent_model_name: str = "gpt-4o-mini"
    mock_model: bool = True
    otlp_endpoint: str | None = None
    log_level: str = "INFO"
    sensitive_fields: tuple[str, ...] = ()

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> AppSettings:
        env = environ if environ is not None else os.environ

        def configured(service_key: str, legacy_key: str, default: str = "") -> str:
            return env.get(service_key, env.get(legacy_key, default))

        raw_fields = configured("TRPC_SERVICE_SENSITIVE_FIELDS", "TRPC_SENSITIVE_FIELDS")
        fields = tuple(field.strip() for field in raw_fields.split(",") if field.strip())
        configured_mock = _as_bool(env.get("TRPC_AGENT_MOCK_MODE"), default=False)
        # No key must always be safe and functional for a local demo.
        api_key = env.get("TRPC_AGENT_API_KEY") or None
        return cls(
            environment=configured("TRPC_SERVICE_ENVIRONMENT", "TRPC_ENV", "development"),
            host=configured("TRPC_SERVICE_HOST", "TRPC_HOST", "0.0.0.0"),
            port=int(configured("TRPC_SERVICE_PORT", "TRPC_PORT", "8000")),
            runtime_backend=configured(
                "TRPC_SERVICE_RUNTIME_BACKEND", "TRPC_RUNTIME_BACKEND", "memory"
            ),
            database_url=configured(
                "TRPC_SERVICE_DATABASE_URL",
                "DATABASE_URL",
                "postgresql+asyncpg://trpc:trpc@localhost:5432/trpc_agent",
            ),
            redis_url=configured("TRPC_SERVICE_REDIS_URL", "REDIS_URL", "redis://localhost:6379/0"),
            object_store_endpoint=configured(
                "TRPC_SERVICE_OBJECT_STORE_ENDPOINT",
                "OBJECT_STORE_ENDPOINT",
                "http://localhost:9000",
            ),
            object_store_bucket=configured(
                "TRPC_SERVICE_OBJECT_STORE_BUCKET", "OBJECT_STORE_BUCKET", "trpc-artifacts"
            ),
            object_store_access_key=configured(
                "TRPC_SERVICE_OBJECT_STORE_ACCESS_KEY", "OBJECT_STORE_ACCESS_KEY"
            )
            or None,
            object_store_secret_key=configured(
                "TRPC_SERVICE_OBJECT_STORE_SECRET_KEY", "OBJECT_STORE_SECRET_KEY"
            )
            or None,
            admin_api_key=configured("TRPC_SERVICE_ADMIN_API_KEY", "TRPC_ADMIN_API_KEY") or None,
            worker_id=configured("TRPC_SERVICE_WORKER_ID", "TRPC_WORKER_ID", "local-worker"),
            dispatcher_id=configured(
                "TRPC_SERVICE_DISPATCHER_ID", "TRPC_DISPATCHER_ID", "local-dispatcher"
            ),
            trpc_agent_api_key=api_key,
            trpc_agent_base_url=env.get("TRPC_AGENT_BASE_URL", "https://api.openai.com/v1").rstrip(
                "/"
            ),
            trpc_agent_model_name=env.get("TRPC_AGENT_MODEL_NAME", "gpt-4o-mini"),
            mock_model=configured_mock or api_key is None,
            otlp_endpoint=configured("TRPC_SERVICE_OTEL_ENDPOINT", "OTEL_EXPORTER_OTLP_ENDPOINT")
            or None,
            log_level=configured("TRPC_SERVICE_LOG_LEVEL", "LOG_LEVEL", "INFO").upper(),
            sensitive_fields=fields,
        )

    def redacted_dict(self) -> dict[str, object]:
        """Return configuration safe for diagnostics and structured logging."""

        return {
            "environment": self.environment,
            "runtime_backend": self.runtime_backend,
            "trpc_agent_api_key": "[REDACTED]" if self.trpc_agent_api_key else None,
            "object_store_secret_key": "[REDACTED]" if self.object_store_secret_key else None,
            "trpc_agent_base_url": self.trpc_agent_base_url,
            "trpc_agent_model_name": self.trpc_agent_model_name,
            "mock_model": self.mock_model,
            "otlp_endpoint": self.otlp_endpoint,
            "log_level": self.log_level,
            "sensitive_fields": self.sensitive_fields,
        }


# Compatibility names retained for the API and runtime modules.  New integration code
# uses ``AppSettings`` to make it clear that no framework settings singleton is needed.
Settings = AppSettings


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings.from_env()


def parse_secret_json(value: str) -> Mapping[str, object]:
    """Parse a structured provider secret while retaining a clear validation error."""

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("expected JSON secret material") from exc
    if not isinstance(parsed, dict):
        raise ValueError("expected a JSON object secret")
    return parsed
