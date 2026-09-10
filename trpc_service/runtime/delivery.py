"""Persisted outbound delivery-attempt state and conservative recovery rules."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from hashlib import sha256
from threading import RLock
from typing import Literal

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
            elif action == "retry" and current.capability != "non_retriable":
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
