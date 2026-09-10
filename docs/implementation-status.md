# Implementation status

This checklist tracks executable behavior, not planned work.  The deterministic
`memory` profile is the verification target; PostgreSQL 16 is the production
fact-store profile and is deployed with the accompanying Alembic/RLS scripts.

## Foundation

- [x] Python 3.12 package metadata with pinned runtime and development dependencies.
- [x] Environment-only settings, mock/environment secret providers, JSON redaction, trace context, and bounded tenant metrics.
- [x] CLI commands: `api`, `worker`, `dispatcher`, `migrate`, `seed`, and `demo`.
- [x] Build, start, stop, clean, format, and coverage scripts; Dockerfile, Compose, MinIO, Redis, PostgreSQL/pgvector, and OTEL Collector definitions.

## Persistence and isolation

- [x] Reference PostgreSQL schema for tenants, releases, bindings, sessions, events, memories, artifacts, Inbox/Outbox, executions, budgets, tools, delivery attempts, and audit logs.
- [x] Alembic initial revision applies schema then RLS; route states include preparation, backfill, catch-up, drain, verification, cutover, read-only, retirement, and failure.
- [x] PostgreSQL tenant transaction helper uses `set_config('app.tenant_id', ..., true)` and pool checkout rollback hygiene.
- [x] Protected exact-match channel locator is implemented in `app_security.resolve_binding`; roles have no direct locator-table grant.
- [x] Repository API requires `TenantContext`; the local transactional repository and runtime expose no cross-tenant enumeration method.

## Runtime safety

- [x] Same-transaction deterministic Inbox plus inbound Outbox acceptance; duplicate callback success is backed by an idempotency index.
- [x] Recoverable leased Outbox dispatcher; a publish/mark crash is intentionally at-least-once and consumers remain idempotent.
- [x] Session fencing, lease renewal/takeover, CAS session commit, epochs, live security envelope, and storage-route migration drain/cutover checks.
- [x] SQL-style conditional hard-budget reservation semantics, settlement/release behavior, and fail-closed missing-account behavior.
- [x] Principal, rate-limit, hard-budget, input/output DLP, Tool-policy, confirmation, and Tool-result filter classes; the worker applies principal/rate/DLP checks before the model and before reply commit.
- [x] Canonical Memory intents plus projection Outbox; reply Outbox is created in the same final Session transaction.
- [x] Deterministic Tool intent IDs, argument divergence detection, persistent running state before requests, and recovery classes (`idempotent`, `queryable`, `non_retriable`).

## Integrations and API

- [x] Immutable Agent releases, a deterministic fake model, and an OpenAI-compatible model client configured through `TRPC_AGENT_*` variables.
- [x] Safe tenant ticket lookup and mock side-effect Tool implementations, with policy/role/confirmation/parameter filters.
- [x] Normalized WeCom, Telegram, and mock channel adapters; mock callbacks and delivery outcomes support duplicate and ambiguous-result simulations.
- [x] FastAPI health, admin, release/security/budget/migration, callback, direct-run, operation, audit, and unknown-operation endpoints.

## Verification

- [x] Focused runtime tests cover atomic Inbox/Outbox, duplicate callbacks, worker fence loss, concurrent budget reservations, unknown non-retriable Tool state, durable reply/memory Outbox, and routing-epoch migration fencing.
- [x] `python -m trpc_service._cli demo` is the end-to-end mock evidence command (two tenants, reply, duplicate, isolation, trace IDs, unknown Tool, budget rejection).
- [ ] PostgreSQL/RLS integration execution requires a reachable PostgreSQL 16 instance.  The scripts and regression SQL are included; the local sandbox has no database service.
