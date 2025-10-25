#!/usr/bin/env bash
set -euo pipefail
. "$(dirname "$0")/00_target.sh"

kubectl get ns "$NS" >/dev/null 2>&1 || kubectl create ns "$NS"

cat <<'YAML' | kubectl -n "$NS" apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx
  labels: { app: nginx }
spec:
  replicas: 1
  selector: { matchLabels: { app: nginx } }
  template:
    metadata: { labels: { app: nginx } }
    spec:
      containers:
      - name: nginx
        image: nginx:1.27
        ports: [{containerPort: 80}]
YAML

kubectl -n "$NS" rollout status deploy/nginx
echo "Seeded baseline nginx in $NS."

