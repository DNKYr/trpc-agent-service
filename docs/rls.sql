-- PostgreSQL tenant isolation policy.
-- Run as the deployment administrator after schema.sql. Workload login roles are
-- environment-specific and must only receive one of these NOLOGIN group roles.

DO $roles$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'platform_schema_owner') THEN
        CREATE ROLE platform_schema_owner NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'platform_migrator') THEN
        CREATE ROLE platform_migrator NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'agent_gateway') THEN
        CREATE ROLE agent_gateway NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'agent_worker') THEN
        CREATE ROLE agent_worker NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'agent_dispatcher') THEN
        CREATE ROLE agent_dispatcher NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'agent_admin') THEN
        CREATE ROLE agent_admin NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'agent_auditor') THEN
        CREATE ROLE agent_auditor NOLOGIN;
    END IF;
END
$roles$;

ALTER ROLE platform_schema_owner NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
ALTER ROLE platform_migrator NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
ALTER ROLE agent_gateway NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
ALTER ROLE agent_worker NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
ALTER ROLE agent_dispatcher NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
ALTER ROLE agent_admin NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
ALTER ROLE agent_auditor NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;

CREATE SCHEMA IF NOT EXISTS app_security AUTHORIZATION platform_schema_owner;
ALTER SCHEMA app_security OWNER TO platform_schema_owner;
REVOKE ALL ON SCHEMA app_security FROM PUBLIC;
GRANT USAGE ON SCHEMA app_security
    TO agent_gateway, agent_worker, agent_dispatcher, agent_admin, agent_auditor;

CREATE OR REPLACE FUNCTION app_security.current_tenant_id()
RETURNS text
LANGUAGE plpgsql
STABLE
SECURITY INVOKER
SET search_path = pg_catalog
AS $function$
DECLARE
    scoped_tenant text;
BEGIN
    scoped_tenant := pg_catalog.btrim(
        pg_catalog.current_setting('app.tenant_id', true)
    );
    IF scoped_tenant IS NULL OR scoped_tenant = '' THEN
        RAISE EXCEPTION USING
            ERRCODE = '42501',
            MESSAGE = 'tenant context is required for this transaction';
    END IF;
    RETURN scoped_tenant;
END
$function$;

ALTER FUNCTION app_security.current_tenant_id() OWNER TO platform_schema_owner;
REVOKE ALL ON FUNCTION app_security.current_tenant_id() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION app_security.current_tenant_id()
    TO agent_gateway, agent_worker, agent_dispatcher, agent_admin, agent_auditor;

CREATE OR REPLACE FUNCTION app_security.resolve_binding(
    requested_webhook_key_hash text,
    requested_provider text
)
RETURNS TABLE (resolved_tenant_id text, resolved_binding_id text)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog
AS $function$
    SELECT locator.resolved_tenant_id, locator.binding_id
    FROM public.channel_binding_locator AS locator
    WHERE locator.webhook_key_hash = requested_webhook_key_hash
      AND locator.provider = requested_provider
      AND locator.enabled
    LIMIT 1
$function$;

CREATE OR REPLACE FUNCTION app_security.sync_binding_locator()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog
AS $function$
BEGIN
    IF TG_OP = 'DELETE'
       OR (TG_OP = 'UPDATE' AND OLD.webhook_key_hash <> NEW.webhook_key_hash) THEN
        DELETE FROM public.channel_binding_locator
        WHERE webhook_key_hash = OLD.webhook_key_hash;
    END IF;

    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    END IF;

    INSERT INTO public.channel_binding_locator (
        webhook_key_hash, resolved_tenant_id, binding_id, provider, enabled
    ) VALUES (
        NEW.webhook_key_hash, NEW.tenant_id, NEW.binding_id, NEW.provider,
        NEW.status <> 'disabled'
    )
    ON CONFLICT (webhook_key_hash) DO UPDATE
    SET resolved_tenant_id = EXCLUDED.resolved_tenant_id,
        binding_id = EXCLUDED.binding_id,
        provider = EXCLUDED.provider,
        enabled = EXCLUDED.enabled;

    RETURN NEW;
END
$function$;

INSERT INTO channel_binding_locator (
    webhook_key_hash, resolved_tenant_id, binding_id, provider, enabled
)
SELECT webhook_key_hash, tenant_id, binding_id, provider, status <> 'disabled'
FROM channel_binding
ON CONFLICT (webhook_key_hash) DO UPDATE
SET resolved_tenant_id = EXCLUDED.resolved_tenant_id,
    binding_id = EXCLUDED.binding_id,
    provider = EXCLUDED.provider,
    enabled = EXCLUDED.enabled;

ALTER TABLE channel_binding_locator OWNER TO platform_schema_owner;
REVOKE ALL ON channel_binding_locator FROM PUBLIC;
ALTER FUNCTION app_security.resolve_binding(text, text) OWNER TO platform_schema_owner;
ALTER FUNCTION app_security.sync_binding_locator() OWNER TO platform_schema_owner;
REVOKE ALL ON FUNCTION app_security.resolve_binding(text, text) FROM PUBLIC;
REVOKE ALL ON FUNCTION app_security.sync_binding_locator() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION app_security.resolve_binding(text, text) TO agent_gateway;

DROP TRIGGER IF EXISTS channel_binding_locator_sync ON channel_binding;
CREATE TRIGGER channel_binding_locator_sync
AFTER INSERT OR UPDATE OR DELETE ON channel_binding
FOR EACH ROW EXECUTE FUNCTION app_security.sync_binding_locator();

DO $policies$
DECLARE
    table_name text;
BEGIN
    FOREACH table_name IN ARRAY ARRAY[
        'tenant',
        'tenant_runtime_state',
        'storage_route',
        'agent_app',
        'agent_release',
        'channel_binding',
        'identity_mapping',
        'session',
        'session_event',
        'session_summary',
        'memory',
        'memory_projection',
        'knowledge_document',
        'artifact',
        'inbox',
        'outbox',
        'execution_attempt',
        'budget_account',
        'budget_reservation',
        'tool_execution',
        'delivery_attempt',
        'audit_log'
    ]
    LOOP
        EXECUTE pg_catalog.format('ALTER TABLE public.%I OWNER TO platform_schema_owner', table_name);
        EXECUTE pg_catalog.format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY', table_name);
        EXECUTE pg_catalog.format('ALTER TABLE public.%I FORCE ROW LEVEL SECURITY', table_name);
        EXECUTE pg_catalog.format('DROP POLICY IF EXISTS tenant_isolation ON public.%I', table_name);
        EXECUTE pg_catalog.format(
            'CREATE POLICY tenant_isolation ON public.%I USING '
            || '(tenant_id = app_security.current_tenant_id()) WITH CHECK '
            || '(tenant_id = app_security.current_tenant_id())',
            table_name
        );
        EXECUTE pg_catalog.format('REVOKE ALL ON public.%I FROM PUBLIC', table_name);
    END LOOP;
END
$policies$;

-- Gateway: resolve bindings and durably accept/inspect Inbox + Outbox records.
GRANT SELECT ON tenant, tenant_runtime_state, agent_app, agent_release,
    channel_binding, identity_mapping, session TO agent_gateway;
GRANT SELECT, INSERT, UPDATE ON inbox, outbox TO agent_gateway;

-- Worker: all Session-side operations remain tenant scoped; no tenant/config mutation.
GRANT SELECT ON tenant, tenant_runtime_state, storage_route, agent_app, agent_release,
    channel_binding, identity_mapping, knowledge_document TO agent_worker;
GRANT SELECT, INSERT, UPDATE ON session, session_summary, memory,
    memory_projection, artifact, inbox, outbox, execution_attempt, budget_account,
    budget_reservation, tool_execution TO agent_worker;
GRANT SELECT, INSERT ON session_event TO agent_worker;
GRANT INSERT ON audit_log TO agent_worker;

-- Dispatcher/Reconciler: publish Outbox and update projections/delivery attempts.
GRANT SELECT ON tenant, tenant_runtime_state, storage_route, channel_binding,
    session, inbox, memory TO agent_dispatcher;
GRANT SELECT, INSERT, UPDATE ON outbox, memory_projection, delivery_attempt,
    execution_attempt TO agent_dispatcher;
GRANT INSERT ON audit_log TO agent_dispatcher;

-- Admin is still tenant scoped. Cross-tenant jobs iterate one SET LOCAL scope at a time.
GRANT SELECT, INSERT, UPDATE ON tenant, tenant_runtime_state,
    agent_app, channel_binding, identity_mapping, budget_account
    TO agent_admin;
GRANT SELECT, INSERT ON agent_release, storage_route TO agent_admin;
GRANT UPDATE (release_status) ON agent_release TO agent_admin;
GRANT UPDATE (route_status, source_watermark, target_watermark, activated_at)
    ON storage_route TO agent_admin;
GRANT SELECT, INSERT ON audit_log TO agent_admin;

GRANT SELECT ON audit_log TO agent_auditor;

-- Deployment automation may SET ROLE to this NOLOGIN role. It is deliberately not
-- granted to application roles and is still subject to FORCE RLS for data access.
GRANT platform_schema_owner TO platform_migrator;
