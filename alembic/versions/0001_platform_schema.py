"""Install the reviewed schema, protected locator, and tenant RLS policies."""

from __future__ import annotations

from pathlib import Path

from alembic import op

revision = "0001_platform_schema"
down_revision = None
branch_labels = None
depends_on = None


def _read(name: str) -> str:
    return (Path(__file__).resolve().parents[2] / "docs" / name).read_text(encoding="utf-8")


def upgrade() -> None:
    op.execute(_read("schema.sql"))
    op.execute(_read("rls.sql"))


def downgrade() -> None:
    # Reverse only relations owned by this application; shared deployment roles
    # are deliberately retained rather than accidentally deleted.
    for table in (
        "audit_log",
        "delivery_attempt",
        "tool_execution",
        "budget_reservation",
        "budget_account",
        "execution_attempt",
        "outbox",
        "inbox",
        "artifact",
        "knowledge_document",
        "memory_projection",
        "memory",
        "session_summary",
        "session_event",
        "session",
        "identity_mapping",
        "channel_binding_locator",
        "channel_binding",
        "agent_release",
        "agent_app",
        "storage_route",
        "tenant_runtime_state",
        "tenant",
    ):
        op.execute(f"DROP TABLE IF EXISTS public.{table} CASCADE")
    op.execute("DROP FUNCTION IF EXISTS app_security.resolve_binding(text, text)")
    op.execute("DROP FUNCTION IF EXISTS app_security.sync_binding_locator()")
    op.execute("DROP FUNCTION IF EXISTS app_security.current_tenant_id()")
