# Operations and recovery runbook

All runtime processes use `TRPC_SERVICE_RUNTIME_BACKEND=postgres`; PostgreSQL is
the source of truth and Redis Streams is only an at-least-once transport. Start
the complete local stack with `docker compose up --build`. Stateful service
ports bind to localhost so they are not exposed by default.

## Compose integration and failure recovery

After the migration job completes, execute the real-service test suite from an
environment with the project dependencies installed:

```bash
TRPC_TEST_DATABASE_URL=postgresql://trpc:trpc@127.0.0.1:5432/trpc_agent \
TRPC_TEST_REDIS_URL=redis://127.0.0.1:6379/0 \
pytest -m integration
```

The test covers atomic Inbox/Outbox acceptance, duplicate callback suppression,
Redis consumer-group delivery, a fenced execution/commit, delivery status, and
the RLS tenant boundary. The focused unit recovery test also simulates the
crash window after a broker publish but before the Outbox publication mark.

## Backup and restore drill

Run `bash scripts/backup-postgres.sh` with `TRPC_BACKUP_DATABASE_URL` and an
optional `TRPC_BACKUP_DIR`. It writes a PostgreSQL custom dump and SHA-256
checksum without deleting old backups. Copy both files to approved durable,
encrypted storage.

For a recovery drill, provision an *empty isolated* database and set
`TRPC_RESTORE_TARGET_URL`; then run:

```bash
bash scripts/restore-postgres-verify.sh backups/trpc-agent-YYYYMMDDTHHMMSSZ.dump
```

The verifier checks the checksum, refuses a nonempty target, restores, and
checks runtime and Alembic tables. It never drops or overwrites a database.

## Load and live-channel probes

Use a bounded health probe first:

```bash
python scripts/load-smoke.py --url https://service.example --requests 500 --concurrency 25
```

Live provider tests are deliberately opt-in and only send one sandbox message
per configured provider:

```bash
RUN_LIVE_CHANNEL_TESTS=1 \
TRPC_LIVE_TELEGRAM_SECRET='{"bot_token":"..."}' \
TRPC_LIVE_TELEGRAM_RECIPIENT=12345 \
pytest -m live
```

For WeCom use `TRPC_LIVE_WECOM_SECRET` (JSON containing `corp_id` and
`corp_secret`), `TRPC_LIVE_WECOM_AGENT_ID`, and `TRPC_LIVE_WECOM_RECIPIENT`.
Keep all values in an operator shell or secret manager, never in Git.
