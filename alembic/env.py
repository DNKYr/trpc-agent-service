from __future__ import annotations

import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

config = context.config
if config.config_file_name:
    fileConfig(config.config_file_name)
url = os.environ.get("TRPC_SERVICE_DATABASE_URL", config.get_main_option("sqlalchemy.url"))
url = url.replace("postgresql+asyncpg://", "postgresql+psycopg://")
config.set_main_option("sqlalchemy.url", url)


def run_migrations_offline() -> None:
    context.configure(url=url, literal_binds=True, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
