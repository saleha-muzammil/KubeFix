#!/usr/bin/env bash
set -euo pipefail
. "$(dirname "$0")/00_target.sh"

cat <<'YAML' | kubectl -n "$NS" apply -f -
apiVersion: apps/v1
kind: Deployment
metadata: { name: nginx }
spec:
  template:
    spec:
      hostPID: true
      hostNetwork: true
      containers:
      - name: nginx
        securityContext:
          privileged: true
          allowPrivilegeEscalation: true
          capabilities:
            add: ["NET_ADMIN","SYS_ADMIN"]
        volumeMounts:
        - name: host-var
          mountPath: /host/var
      volumes:
      - name: host-var
        hostPath: { path: /var, type: "" }
YAML

kubectl -n "$NS" rollout status deploy/nginx
echo "Made nginx privileged with hostPath in $NS."

