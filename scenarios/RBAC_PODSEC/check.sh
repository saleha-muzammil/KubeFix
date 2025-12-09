#!/usr/bin/env bash
set -euo pipefail

NS=rbac-podsec
POD=$(kubectl -n "$NS" get pod -l app=nginx -o jsonpath='{.items[0].metadata.name}')

echo "[*] Pod status:"
kubectl -n "$NS" get pod "$POD" -o wide

SVC_IP=$(kubectl -n "$NS" get svc nginx -o jsonpath='{.spec.clusterIP}')
echo "[*] Curling nginx service ($SVC_IP:80) from a transient busybox pod"

kubectl run curl-tester-rbac-podsec \
  --rm -i --restart=Never --image=busybox \
  -- sh -c "wget -qO- http://${SVC_IP}:80 | head -n 5"

