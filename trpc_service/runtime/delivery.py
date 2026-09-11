"""Persisted outbound delivery-attempt state and conservative recovery rules."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from hashlib import sha256
from threading import RLock
from typing import Any, Literal

from trpc_service.db.postgres import PostgresConnections

from .models import TenantContext

DeliveryCapability = Literal["idempotent", "queryable", "non_retriable"]
DeliveryStatus = Literal[
    "prepared", "sending", "accepted", "failed", "unknown", "reconciling", "manual_review"
]


@dataclass(frozen=True, slots=True)
class DeliveryAttempt:
    tenant_id: str
    delivery_id: str
    attempt_no: int
    outbox_id: str
    session_id: str
    channel_binding_id: str
    capability: DeliveryCapability
    request_hash: str
    status: DeliveryStatus = "prepared"
    provider_idempotency_key: str | None = None
    provider_message_id: str | None = None
    error_code: str | None = None
    trace_id: str = ""
    started_at: datetime = datetime.min.replace(tzinfo=UTC)
    finished_at: datetime | None = None


class InMemoryDeliveryLedger:
    """Small durable-style ledger for the demo and focused delivery tests.

    It makes no exactly-once claim: only an idempotent provider can safely retry
    with the original delivery key. Queryable providers require reconciliation,
    and ambiguous non-retriable delivery requires a human decision.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._attempts: dict[tuple[str, str], list[DeliveryAttempt]] = {}

    @staticmethod
    def delivery_id(tenant_id: str, outbox_id: str) -> str:
        return "del_" + sha256(f"{tenant_id}\x1f{outbox_id}".encode()).hexdigest()[:32]

    def begin(
        self,
        *,
        tenant_id: str,
        outbox_id: str,
        session_id: str,
        channel_binding_id: str,
        capability: DeliveryCapability,
        request_hash: str,
        trace_id: str,
    ) -> DeliveryAttempt:
        with self._lock:
            delivery_id = self.delivery_id(tenant_id, outbox_id)
            attempts = self._attempts.setdefault((tenant_id, delivery_id), [])
            if attempts:
                latest = attempts[-1]
                if latest.status == "accepted":
                    return latest
                if capability == "queryable":
                    # A queryable provider must be reconciled through its query
                    # API; never turn an old unknown/failed request into a new
                    # send attempt merely because a Stream message was replayed.
                    return latest
                if latest.status in {"unknown", "manual_review"} and capability == "non_retriable":
                    return latest
                if latest.request_hash != request_hash:
                    raise ValueError("delivery request changed for deterministic outbox delivery")
            attempt = DeliveryAttempt(
                tenant_id=tenant_id,
                delivery_id=delivery_id,
                attempt_no=len(attempts) + 1,
                outbox_id=outbox_id,
                session_id=session_id,
                channel_binding_id=channel_binding_id,
                capability=capability,
                request_hash=request_hash,
                status="sending",
                provider_idempotency_key=delivery_id if capability == "idempotent" else None,
                trace_id=trace_id,
                started_at=datetime.now(UTC),
            )
            attempts.append(attempt)
            return attempt

    def finish(
        self,
        attempt: DeliveryAttempt,
        *,
        status: DeliveryStatus,
        provider_message_id: str | None = None,
        error_code: str | None = None,
    ) -> DeliveryAttempt:
        if status not in {"accepted", "failed", "unknown", "reconciling", "manual_review"}:
            raise ValueError("delivery must finish in a concrete provider state")
        with self._lock:
            attempts = self._attempts[(attempt.tenant_id, attempt.delivery_id)]
            current = attempts[attempt.attempt_no - 1]
            if current.status == "accepted":
                return current
            final_status: DeliveryStatus = status
            if status == "unknown" and current.capability == "non_retriable":
                final_status = "manual_review"
            final = replace(
                current,
                status=final_status,
                provider_message_id=provider_message_id,
                error_code=error_code,
                finished_at=datetime.now(UTC),
            )
            attempts[attempt.attempt_no - 1] = final
            return final

    def unknown(self, tenant_id: str) -> list[DeliveryAttempt]:
        with self._lock:
            return [
                row
                for (row_tenant, _), attempts in self._attempts.items()
                if row_tenant == tenant_id
                for row in attempts
                if row.status in {"unknown", "manual_review"}
            ]

    def resolve(
        self, tenant_id: str, delivery_id: str, *, action: str, note: str = ""
    ) -> DeliveryAttempt:
        """Record a human delivery decision without promising provider replay."""

        with self._lock:
            attempts = self._attempts.get((tenant_id, delivery_id))
            if not attempts:
                raise KeyError("delivery operation does not exist in this tenant")
            current = attempts[-1]
            if current.status not in {"unknown", "manual_review", "reconciling", "failed"}:
                raise ValueError("delivery does not need manual resolution")
            if action == "accepted":
                status: DeliveryStatus = "accepted"
            elif action == "failed":
                status = "failed"
            elif action == "retry" and current.capability in {"idempotent", "queryable"}:
                # The dispatcher must query/retry later; this endpoint never
                # invokes a provider inline and never retries ambiguous effects.
                status = "reconciling"
            else:
                status = "manual_review"
            final = replace(
                current,
                status=status,
                error_code=note or current.error_code,
                finished_at=datetime.now(UTC),
            )
            attempts[-1] = final
            return final


class PostgresDeliveryLedger:
    """Tenant-scoped PostgreSQL delivery ledger.

    The delivery transport is at-least-once.  This ledger turns that transport
    into an auditable provider-attempt history and preserves the original
    idempotency key for providers that support one.  An advisory transaction
    lock serializes attempts for one deterministic delivery ID, including the
    otherwise racy first attempt.
    """

    def __init__(self, database_url: str, database_role: str | None = None) -> None:
        self._connections = PostgresConnections(database_url, database_role)

    @staticmethod
    def delivery_id(tenant_id: str, outbox_id: str) -> str:
        return InMemoryDeliveryLedger.delivery_id(tenant_id, outbox_id)

    @staticmethod
    def _attempt(row: dict[str, Any]) -> DeliveryAttempt:
        return DeliveryAttempt(
            tenant_id=str(row["tenant_id"]),
            delivery_id=str(row["delivery_id"]),
            attempt_no=int(row["attempt_no"]),
            outbox_id=str(row["outbox_id"]),
            session_id=str(row["session_id"]),
            channel_binding_id=str(row["channel_binding_id"]),
            capability=str(row["retry_capability"]),  # type: ignore[arg-type]
            request_hash=str(row["request_hash"]),
            status=str(row["status"]),  # type: ignore[arg-type]
            provider_idempotency_key=row["provider_idempotency_key"],
            provider_message_id=row["provider_message_id"],
            error_code=row["last_error_code"],
            trace_id=str(row["trace_id"]),
            started_at=row["started_at"],
            finished_at=row["finished_at"],
        )

    def begin(
        self,
        *,
        tenant_id: str,
        outbox_id: str,
        session_id: str,
        channel_binding_id: str,
        capability: DeliveryCapability,
        request_hash: str,
        trace_id: str,
    ) -> DeliveryAttempt:
        context = TenantContext(tenant_id, actor_id="dispatcher", trace_id=trace_id)
        delivery_id = self.delivery_id(tenant_id, outbox_id)
        with self._connections.tenant(context) as connection:
            connection.execute(
                "SELECT pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended(%s, 0))",
                (f"{tenant_id}\x1f{delivery_id}",),
            )
            latest = connection.execute(
                """
                SELECT * FROM delivery_attempt
                WHERE tenant_id = %s AND delivery_id = %s
                ORDER BY attempt_no DESC LIMIT 1 FOR UPDATE
                """,
                (tenant_id, delivery_id),
            ).fetchone()
            if latest is not None:
                prior = self._attempt(latest)
                if prior.status == "accepted":
                    return prior
                if capability == "queryable":
                    return prior
                if prior.status in {"unknown", "manual_review"} and capability == "non_retriable":
                    return prior
                if prior.request_hash != request_hash:
                    raise ValueError("delivery request changed for deterministic outbox delivery")
                attempt_no = prior.attempt_no + 1
            else:
                attempt_no = 1
            row = connection.execute(
                """
                INSERT INTO delivery_attempt (
                    tenant_id, delivery_id, attempt_no, outbox_id, session_id,
                    channel_binding_id, retry_capability, provider_idempotency_key,
                    request_hash, status, trace_id, started_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'sending', %s, now())
                RETURNING *
                """,
                (
                    tenant_id,
                    delivery_id,
                    attempt_no,
                    outbox_id,
                    session_id,
                    channel_binding_id,
                    capability,
                    delivery_id if capability == "idempotent" else None,
                    request_hash,
                    trace_id,
                ),
            ).fetchone()
            return self._attempt(row)

    def finish(
        self,
        attempt: DeliveryAttempt,
        *,
        status: DeliveryStatus,
        provider_message_id: str | None = None,
        error_code: str | None = None,
    ) -> DeliveryAttempt:
        if status not in {"accepted", "failed", "unknown", "reconciling", "manual_review"}:
            raise ValueError("delivery must finish in a concrete provider state")
        context = TenantContext(attempt.tenant_id, actor_id="dispatcher", trace_id=attempt.trace_id)
        with self._connections.tenant(context) as connection:
            current = connection.execute(
                """
                SELECT * FROM delivery_attempt
                WHERE tenant_id = %s AND delivery_id = %s AND attempt_no = %s FOR UPDATE
                """,
                (attempt.tenant_id, attempt.delivery_id, attempt.attempt_no),
            ).fetchone()
            if current is None:
                raise KeyError("delivery operation does not exist in this tenant")
            existing = self._attempt(current)
            if existing.status == "accepted":
                return existing
            final_status: DeliveryStatus = status
            if status == "unknown" and existing.capability == "non_retriable":
                final_status = "manual_review"
            row = connection.execute(
                """
                UPDATE delivery_attempt
                SET status = %s, provider_message_id = %s, last_error_code = %s, finished_at = now()
                WHERE tenant_id = %s AND delivery_id = %s AND attempt_no = %s
                RETURNING *
                """,
                (
                    final_status,
                    provider_message_id,
                    error_code,
                    attempt.tenant_id,
                    attempt.delivery_id,
                    attempt.attempt_no,
                ),
            ).fetchone()
            return self._attempt(row)

    def unknown(self, tenant_id: str) -> list[DeliveryAttempt]:
        context = TenantContext(tenant_id, actor_id="dispatcher")
        with self._connections.tenant(context) as connection:
            rows = connection.execute(
                """
                SELECT * FROM delivery_attempt
                WHERE tenant_id = %s AND status IN ('unknown', 'manual_review')
                ORDER BY delivery_id, attempt_no
                """,
                (tenant_id,),
            ).fetchall()
            return [self._attempt(row) for row in rows]

    def resolve(
        self, tenant_id: str, delivery_id: str, *, action: str, note: str = ""
    ) -> DeliveryAttempt:
        context = TenantContext(tenant_id, actor_id="admin")
        with self._connections.tenant(context) as connection:
            current = connection.execute(
                """
                SELECT * FROM delivery_attempt
                WHERE tenant_id = %s AND delivery_id = %s
                ORDER BY attempt_no DESC LIMIT 1 FOR UPDATE
                """,
                (tenant_id, delivery_id),
            ).fetchone()
            if current is None:
                raise KeyError("delivery operation does not exist in this tenant")
            existing = self._attempt(current)
            if existing.status not in {"unknown", "manual_review", "reconciling", "failed"}:
                raise ValueError("delivery does not need manual resolution")
            if action == "accepted":
                final_status: DeliveryStatus = "accepted"
            elif action == "failed":
                final_status = "failed"
            elif action == "retry" and existing.capability in {"idempotent", "queryable"}:
                final_status = "reconciling"
            else:
                final_status = "manual_review"
            row = connection.execute(
                """
                UPDATE delivery_attempt SET status = %s, last_error_code = %s, finished_at = now()
                WHERE tenant_id = %s AND delivery_id = %s AND attempt_no = %s
                RETURNING *
                """,
                (final_status, note or existing.error_code, tenant_id, delivery_id, existing.attempt_no),
            ).fetchone()
            return self._attempt(row)
