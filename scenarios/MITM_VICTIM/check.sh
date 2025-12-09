#!/usr/bin/env bash
set -euo pipefail

NS=kubeproxy-mitm
POD=$(kubectl -n "$NS" get pod -l app=echoserver -o jsonpath='{.items[0].metadata.name}')

echo "[*] echoserver pod status:"
kubectl -n "$NS" get pod "$POD" -o wide

SVC_IP=$(kubectl -n "$NS" get svc mitm-lb -o jsonpath='{.spec.clusterIP}')
echo "[*] Curling echoserver directly via Service IP: $SVC_IP"

kubectl -n "$NS" run curl-tester-mitm \
  --rm -i --restart=Never --image=busybox \
  -- sh -c "wget -qO- http://${SVC_IP}:80 | head -n 5"

