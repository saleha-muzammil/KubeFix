#!/usr/bin/env bash
set -euo pipefail

NS=stateful-rbac
POD=$(kubectl -n "$NS" get pod -l app=redis -o jsonpath='{.items[0].metadata.name}')

echo "[*] Redis pod status:"
kubectl -n "$NS" get pod "$POD" -o wide

echo "[*] Checking redis responsiveness via redis-cli from a sidecar pod"
kubectl -n "$NS" run redis-client --rm -i --restart=Never --image=redis:7-alpine \
  -- sh -c "redis-cli -h redis -p 6379 ping"

