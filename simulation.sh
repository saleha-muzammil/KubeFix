#!/usr/bin/env bash
set -euo pipefail
NS_PREFIX=${NS_PREFIX:-sim}
N_NS=20
PER_NS=10

for n in $(seq 1 $N_NS); do
  ns="${NS_PREFIX}-${n}"
  kubectl get ns "$ns" >/dev/null 2>&1 || kubectl create ns "$ns"
  for m in $(seq 1 $PER_NS); do
    kubectl -n "$ns" create deployment "nginx-$m" --image=nginx:1.27 --dry-run=client -o yaml \
      | kubectl apply -f -
  done
done

echo "Created $((N_NS*PER_NS)) deployments across $N_NS namespaces."

