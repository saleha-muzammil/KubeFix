#!/usr/bin/env bash
set -euo pipefail
NS=${1:-demo}
kubectl create ns "$NS" 2>/dev/null || true
kubectl -n "$NS" delete networkpolicy --all 2>/dev/null || true
echo "Namespace $NS now has no NetworkPolicies."
