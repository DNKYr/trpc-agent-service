from __future__ import annotations

from datetime import UTC, datetime, timedelta
from threading import Thread

import pytest

from trpc_service.runtime import (
    BudgetAccount,
    BudgetExceeded,
    CommitInput,
    ExecutionMode,
    InboundEnvelope,
    InMemoryMessageBus,
    InMemoryRuntimeStore,
    MemoryIntentDraft,
    MigrationStatus,
    PlatformRuntime,
    ReplyDraft,
    SecurityRejected,
    SessionEventDraft,
    StaleFence,
    TenantContext,
    TenantMismatch,
    ToolCapability,
    ToolRecoveryAction,
    ToolStatus,
)


class Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 1, 1, tzinfo=UTC)

    def now(self):
        return self.value

    def advance(self, seconds: int) -> None:
        self.value += timedelta(seconds=seconds)


def make_runtime():
    clock = Clock()
    store = InMemoryRuntimeStore(now=clock.now)
    store.bootstrap_tenant("tenant-a", storage_profile={"sql": "primary"})
    bus = InMemoryMessageBus()
    return clock, PlatformRuntime(store, bus), TenantContext("tenant-a"), bus


def envelope(key: str, *, session: str = "session-1") -> InboundEnvelope:
    return InboundEnvelope(
        tenant_id="tenant-a",
        channel_binding_id="binding-1",
        agent_id="agent-1",
        session_id=session,
        idempotency_key=key,
        external_message_id=key,
        payload={"type": "text", "text": "hello"},
        trace_id="trace-1",
        request_id="request-1",
    )


def test_inbox_and_inbound_outbox_are_atomic_and_duplicate_safe():
    _, runtime, context, _ = make_runtime()
    accepted = runtime.accept_inbound(context, envelope("provider:1"))
    duplicate = runtime.accept_inbound(context, envelope("provider:1"))

    assert not accepted.duplicate
    assert duplicate.duplicate
    assert duplicate.inbox.inbox_id == accepted.inbox.inbox_id
    snapshot = runtime.snapshot(context)
    assert len(snapshot["inboxes"]) == 1
    assert len(snapshot["outbox"]) == 1
    assert snapshot["outbox"][0]["event_type"] == "inbound.dispatch"

    runtime.suspend(context)
    with pytest.raises(SecurityRejected):
        runtime.accept_inbound(context, envelope("provider:2"))
    assert len(runtime.snapshot(context)["inboxes"]) == 1


def test_tenant_context_cannot_access_or_write_another_tenant():
    store = InMemoryRuntimeStore()
    store.bootstrap_tenant("tenant-a")
    store.bootstrap_tenant("tenant-b")
    runtime = PlatformRuntime(store, InMemoryMessageBus())
    tenant_a = TenantContext("tenant-a")
    tenant_b = TenantContext("tenant-b")
    with pytest.raises(TenantMismatch):
        runtime.accept_inbound(
            tenant_a,
            InboundEnvelope(
                tenant_id="tenant-b",
                channel_binding_id="binding-b",
                agent_id="agent-b",
                session_id="session-b",
                idempotency_key="cross-tenant",
                external_message_id="cross-tenant",
                payload={"text": "forbidden"},
            ),
        )
    accepted = runtime.accept_inbound(
        tenant_b,
        InboundEnvelope(
            tenant_id="tenant-b",
            channel_binding_id="binding-b",
            agent_id="agent-b",
            session_id="session-b",
            idempotency_key="owned",
            external_message_id="owned",
            payload={"text": "owned"},
        ),
    )
    assert accepted.inbox.tenant_id == "tenant-b"
    assert runtime.snapshot(tenant_a)["inboxes"] == []


def test_dispatch_is_recoverable_and_never_required_for_acceptance():
    _, runtime, context, bus = make_runtime()
    accepted = runtime.accept_inbound(context, envelope("provider:1"))
    bus.fail_next_publish = True
    assert runtime.dispatch_once(context, "dispatcher") == []
    assert runtime.snapshot(context)["outbox"][0]["status"] == "pending"

    assert len(runtime.dispatch_once(context, "dispatcher")) == 1
    assert len(bus.published) == 1
    assert bus.published[0][1].inbox_id == accepted.inbox.inbox_id


def test_only_one_worker_can_own_fence_and_stale_commit_fails():
    clock, runtime, context, _ = make_runtime()
    inbox = runtime.accept_inbound(context, envelope("provider:1")).inbox
    first = runtime.claim_execution(context, inbox.inbox_id, "worker-a", lease_seconds=5)

    clock.advance(6)
    second = runtime.claim_execution(context, inbox.inbox_id, "worker-b", lease_seconds=5)
    assert second.execution_id == first.execution_id
    assert second.lease_fence == first.lease_fence + 1
    with pytest.raises(StaleFence):
        runtime.commit_execution(
            context, first, CommitInput(expected_session_version=0, new_state={})
        )

    committed = runtime.commit_execution(
        context,
        second,
        CommitInput(
            expected_session_version=0,
            new_state={"last": "ok"},
            events=[
                SessionEventDraft(
                    event_type="assistant.message", role="assistant", payload={"text": "ok"}
                )
            ],
        ),
    )
    assert committed.session.version == 1
    assert committed.inbox.status.value == "committed"


def test_concurrent_hard_budget_reservation_cannot_overspend():
    _, runtime, context, _ = make_runtime()
    runtime.put_budget_account(
        context, BudgetAccount("tenant-a", "model", "tokens", limit_units=10)
    )
    first = runtime.accept_inbound(context, envelope("provider:1", session="s-1")).inbox
    second = runtime.accept_inbound(context, envelope("provider:2", session="s-2")).inbox
    results: list[str] = []

    def claim(inbox_id: str):
        try:
            runtime.claim_execution(context, inbox_id, "worker", budget_estimates={"model": 10})
            results.append("claimed")
        except BudgetExceeded:
            results.append("rejected")

    threads = [
        Thread(target=claim, args=(first.inbox_id,)),
        Thread(target=claim, args=(second.inbox_id,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(results) == ["claimed", "rejected"]


def test_tool_unknown_is_not_automatically_retried_and_reply_memory_are_durable():
    _, runtime, context, _ = make_runtime()
    inbox = runtime.accept_inbound(context, envelope("provider:1")).inbox
    claim = runtime.claim_execution(context, inbox.inbox_id, "worker")
    intent = runtime.prepare_tool(
        context,
        claim,
        tool_step=0,
        tool_name="effect.create",
        arguments={"name": "x"},
        capability=ToolCapability.NON_RETRIABLE,
    )
    runtime.start_tool(context, claim, intent.tool_call_id)
    runtime.finish_tool(
        context, claim, intent.tool_call_id, status=ToolStatus.UNKNOWN, error_code="timeout"
    )
    assert (
        runtime.tool_recovery_action(context, intent.tool_call_id)
        == ToolRecoveryAction.MANUAL_REVIEW
    )
    resolved = runtime.resolve_tool(
        context,
        intent.tool_call_id,
        status=ToolStatus.MANUAL_REVIEW,
        note="operator notified",
    )
    assert resolved.status == ToolStatus.MANUAL_REVIEW

    result = runtime.commit_execution(
        context,
        claim,
        CommitInput(
            expected_session_version=0,
            new_state={"complete": True},
            memories=[MemoryIntentDraft(memory_type="fact", content="user likes tea")],
            reply=ReplyDraft(blocks=[{"type": "text", "text": "saved"}]),
        ),
    )
    assert result.reply_outbox is not None
    assert len(result.memory_outboxes) == 1
    snapshot = runtime.snapshot(context)
    assert len(snapshot["memories"]) == 1
    assert {row["event_type"] for row in snapshot["outbox"]} == {
        "inbound.dispatch",
        "memory.project",
        "reply.dispatch",
    }


def test_migration_drains_fences_and_cutover_rejects_old_worker():
    clock, runtime, context, _ = make_runtime()
    inbox = runtime.accept_inbound(context, envelope("provider:1")).inbox
    stale_claim = runtime.claim_execution(context, inbox.inbox_id, "worker", lease_seconds=5)
    migration = runtime.initiate_migration(context, {"sql": "secondary"}, migration_id="move-1")
    assert migration.status == MigrationStatus.PREPARING
    runtime.migration_action(context, "move-1", "start_backfill")
    runtime.migration_action(
        context, "move-1", "catch_up", source_watermark="10", target_watermark="10"
    )
    draining = runtime.migration_action(context, "move-1", "begin_drain")
    assert draining.status == MigrationStatus.DRAINING
    assert runtime.runtime_state(context).execution_mode == ExecutionMode.DRAINING
    clock.advance(6)
    runtime.migration_action(context, "move-1", "verify", verified=True)
    active = runtime.migration_action(context, "move-1", "cutover")
    assert active.status == MigrationStatus.ACTIVE
    assert runtime.current_route(context).profile == {"sql": "secondary"}
    with pytest.raises(StaleFence):
        runtime.commit_execution(
            context, stale_claim, CommitInput(expected_session_version=0, new_state={})
        )
