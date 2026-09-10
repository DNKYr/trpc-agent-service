# Kubernetes deployment

The `base` directory deploys API, Worker, Dispatcher, a migration Job, HPA,
PDB, ingress policy, and a singleton WeCom Smart Bot gateway. Create
`trpc-agent-service-secrets` using an approved secret manager before applying
it:

```bash
kubectl apply -k deploy/kubernetes/base
kubectl -n trpc-agent get pods
```

`secrets.example.yaml` documents keys only and must never be populated or
committed with credentials. Replace the example image with an immutable
registry digest through a Kustomize overlay or `kustomize edit set image`.

`backup-cronjob.example.yaml` is deliberately not included in the base. Review
PVC encryption, retention, and off-cluster copy policy, replace its PVC name,
then apply it separately. The recovery drill is documented in
`docs/operations.md`.

The Smart Bot gateway stays idle until a `wecom_aibot` binding exists. Add
`TRPC_LIVE_WECOM_AIBOT_SECRET` (JSON `{ "bot_id": "...", "secret": "..." }`)
to the referenced Secret before binding a bot. Keep the gateway at one replica;
it must not be horizontally scaled.
