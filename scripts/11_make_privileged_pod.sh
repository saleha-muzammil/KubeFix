#!/usr/bin/env bash
set -euo pipefail
NS=${1:-demo}
kubectl create ns "$NS" 2>/dev/null || true

cat <<YAML | kubectl -n "$NS" apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx
spec:
  replicas: 1
  selector: { matchLabels: { app: nginx } }
  template:
    metadata: { labels: { app: nginx } }
    spec:
      serviceAccountName: default
      containers:
      - name: nginx
        image: nginx:1.27
        securityContext:
          runAsUser: 0
          runAsGroup: 0
          allowPrivilegeEscalation: true
          capabilities:
            add: ["NET_ADMIN","SYS_ADMIN","SYS_PTRACE"]
          readOnlyRootFilesystem: false
        ports:
        - containerPort: 80
YAML
kubectl -n "$NS" rollout status deploy/nginx
echo "Privileged pod applied in namespace $NS."
