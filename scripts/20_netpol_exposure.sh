# scripts/20_netpol_exposure.sh
#!/usr/bin/env bash
set -euo pipefail
NS=${NS:-demo}

kubectl create ns "$NS" 2>/dev/null || true
kubectl -n "$NS" create deployment nginx --image=nginx:1.27 2>/dev/null || true
kubectl -n "$NS" expose deploy/nginx --port=80 --target-port=80 --name=nginx 2>/dev/null || true
kubectl -n "$NS" rollout status deploy/nginx

# wide-open: remove all policies
kubectl -n "$NS" delete networkpolicy --all 2>/dev/null || true

# attacker pod with curl preinstalled
kubectl create ns attacker 2>/dev/null || true
kubectl -n attacker delete pod curl --force --grace-period=0 2>/dev/null || true
kubectl -n attacker run curl --image=curlimages/curl:8.7.1 --restart=Never -- sleep 3600 2>/dev/null || true
kubectl -n attacker wait --for=condition=Ready pod/curl --timeout=60s

# verify cross-namespace access (should output HTML)
kubectl -n attacker exec curl -- curl -sS nginx.$NS.svc.cluster.local:80 | head -n1 || true
echo "NetPol exposure scenario ready in ns=$NS (service: nginx)."

