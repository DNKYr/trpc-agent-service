# Helm deployment

Create the referenced Kubernetes Secret through External Secrets, Sealed Secrets,
or your cloud secret manager. It must provide at least
`TRPC_SERVICE_DATABASE_URL` and `TRPC_SERVICE_REDIS_URL`; model and channel
credentials are optional for mock mode. Never commit a populated Secret manifest.

```bash
helm upgrade --install trpc-agent-service . \
  --namespace trpc-agent --create-namespace \
  --set image.repository=registry.example/trpc-agent-service \
  --set image.tag=<immutable-image-digest-or-tag>
```

The migration Job runs as a Helm post-install/post-upgrade hook. Set
`migration.hook=false` only when migrations are operated by a separate approved
pipeline. HPA requires metrics-server; use KEDA with Redis Stream lag for
queue-aware worker scaling in clusters that provide it.

To enable a WeCom Smart Bot long connection, add
`TRPC_LIVE_WECOM_AIBOT_SECRET` to the referenced Secret as JSON containing
`bot_id` and `secret`, create a `wecom_aibot` channel binding, and set
`wecomAibot.enabled=true`. The gateway is intentionally one replica: WeCom
allows only one active WebSocket per Bot ID.
