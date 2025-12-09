#!/usr/bin/env bash
set -euo pipefail

NS_PAY=payments
PAY_POD=$(kubectl -n "$NS_PAY" get pod -l app=payments-api -o jsonpath='{.items[0].metadata.name}')

echo "[*] payments-api pod status:"
kubectl -n "$NS_PAY" get pod "$PAY_POD" -o wide

SVC_IP=$(kubectl -n "$NS_PAY" get svc payments-api -o jsonpath='{.spec.clusterIP}')
echo "[*] Curling payments-api from within payments namespace:"

kubectl -n "$NS_PAY" run curl-tester-netpol \
  --rm -i --restart=Never --image=busybox \
  -- sh -c "wget -qO- http://${SVC_IP}:80 | head -n 5"

