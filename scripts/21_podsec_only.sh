# scripts/21_podsec_only.sh
#!/usr/bin/env bash
set -euo pipefail
NS=${NS:-podsec-only}
kubectl create ns "$NS" 2>/dev/null || true

cat <<'YAML' | kubectl -n "$NS" apply -f -
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
      containers:
      - name: nginx
        image: nginx:1.27
        securityContext:
          runAsUser: 0
          allowPrivilegeEscalation: true
          readOnlyRootFilesystem: false
          capabilities:
            add: ["NET_ADMIN","SYS_ADMIN"]
        ports:
        - containerPort: 80
YAML

kubectl -n "$NS" rollout status deploy/nginx
echo "PodSecurity-only scenario ready in ns=$NS."

