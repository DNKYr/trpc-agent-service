from __future__ import annotations

import asyncio

import httpx

from trpc_service.config import AppSettings
from trpc_service.web.app import ServiceContainer, create_app


def test_admin_callback_and_direct_run_share_the_durable_runtime() -> None:
    """Exercise the documented HTTP boundary without opening a network socket."""

    async def scenario() -> None:
        app = create_app(ServiceContainer(AppSettings(admin_api_key=None)))
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://platform.test"
        ) as client:
            tenant = await client.post(
                "/admin/v1/tenants", json={"tenant_id": "tenant-api", "display_name": "API tenant"}
            )
            assert tenant.status_code == 201
            assert tenant.json()["tenant_id"] == "tenant-api"

            agent = await client.post(
                "/admin/v1/tenants/tenant-api/agents",
                json={"agent_id": "support", "name": "Support"},
            )
            assert agent.status_code == 201
            release = await client.post(
                "/admin/v1/tenants/tenant-api/agents/support/releases",
                json={
                    "version": 1,
                    "model_config": {"mode": "mock"},
                    "tool_policy": {"allow": ["ticket.lookup"]},
                },
            )
            assert release.status_code == 201
            activated = await client.post(
                "/admin/v1/tenants/tenant-api/agents/support/releases/1/activate"
            )
            assert activated.status_code == 200

            binding = await client.post(
                "/admin/v1/tenants/tenant-api/channels",
                json={
                    "binding_id": "mock-main",
                    "agent_id": "support",
                    "provider": "mock",
                    "external_account_id": "test-account",
                    "webhook_key": "test-webhook-key-123456",
                    "capabilities": {"callback_secret": "test-secret"},
                },
            )
            assert binding.status_code == 201

            callback = await client.post(
                "/callbacks/mock/test-webhook-key-123456",
                headers={"x-mock-secret": "test-secret", "x-request-id": "callback-1"},
                json={"message_id": "message-1", "user_id": "alice", "text": "ticket 42"},
            )
            assert callback.status_code == 200
            first = callback.json()
            assert first["duplicate"] is False
            duplicate = await client.post(
                "/callbacks/mock/test-webhook-key-123456",
                headers={"x-mock-secret": "test-secret"},
                json={"message_id": "message-1", "user_id": "alice", "text": "ticket 42"},
            )
            assert duplicate.json()["duplicate"] is True
            assert duplicate.json()["inbox_id"] == first["inbox_id"]

            direct = await client.post(
                "/v1/tenants/tenant-api/agents/support:run",
                headers={"x-request-id": "run-1"},
                json={"input": "ticket 42", "idempotency_key": "direct-1"},
            )
            assert direct.status_code == 200
            assert direct.json()["status"] == "committed"
            assert "ticket" in direct.json()["reply"].lower()
            assert direct.headers["x-request-id"] == "run-1"

    asyncio.run(scenario())
