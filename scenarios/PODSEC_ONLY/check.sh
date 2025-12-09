#!/usr/bin/env bash
set -euo pipefail

NS=podsec-only
POD=$(kubectl -n "$NS" get pod -l app=insecure-web -o jsonpath='{.items[0].metadata.name}')

echo "[*] Pod status:"
kubectl -n "$NS" get pod "$POD" -o wide

SVC_IP=$(kubectl -n "$NS" get pod "$POD" -o jsonpath='{.status.podIP}')
echo "[*] Curling pod IP ($SVC_IP:80) from transient busybox"

kubectl run curl-tester-podsec-only \
  --rm -i --restart=Never --image=busybox \
  -- sh -c "wget -qO- http://${SVC_IP}:80 | head -n 5"

