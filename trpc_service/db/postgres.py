"""Small synchronous PostgreSQL transaction helpers used by long-lived processes.

The runtime port is intentionally synchronous so a worker can make a complete
fenced state transition in one database transaction.  psycopg is used here
instead of an async ORM session so the same implementation can be called by
FastAPI's worker threads and by the dispatcher/worker CLI processes.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from trpc_service.runtime.models import TenantContext


def psycopg_dsn(database_url: str) -> str:
    """Translate SQLAlchemy-style PostgreSQL URLs to a psycopg-compatible DSN."""

    return database_url.replace("postgresql+asyncpg://", "postgresql://", 1).replace(
        "postgresql+psycopg://", "postgresql://", 1
    )


class PostgresConnections:
    """Open short transactions with an un-leakable tenant GUC scope."""

    def __init__(self, database_url: str) -> None:
        self.dsn = psycopg_dsn(database_url)

    @contextmanager
    def tenant(self, context: TenantContext) -> Iterator[object]:
        from psycopg import connect
        from psycopg.rows import dict_row

        connection = connect(self.dsn, row_factory=dict_row)
        try:
            with connection.transaction():
                connection.execute(
                    "SELECT pg_catalog.set_config('app.tenant_id', %s, true)",
                    (context.tenant_id,),
                )
                yield connection
        finally:
            connection.close()

    @contextmanager
    def bootstrap(self) -> Iterator[object]:
        """Use only for the protected callback locator before tenant resolution."""

        from psycopg import connect
        from psycopg.rows import dict_row

        connection = connect(self.dsn, row_factory=dict_row)
        try:
            with connection.transaction():
                yield connection
        finally:
            connection.close()
