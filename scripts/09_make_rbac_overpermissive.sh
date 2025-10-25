#!/usr/bin/env bash
set -euo pipefail
NS=${NS:-demo}
kubectl -n "$NS" create sa over-sa --dry-run=client -o yaml | kubectl apply -f -
cat <<YAML | kubectl apply -f -
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata: { name: over-perm }
rules:
- apiGroups: ["*"]
  resources: ["*"]
  verbs: ["*"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata: { name: over-perm-binding }
roleRef: { apiGroup: rbac.authorization.k8s.io, kind: ClusterRole, name: over-perm }
subjects:
- kind: ServiceAccount
  name: over-sa
  namespace: ${NS}
YAML
kubectl -n "$NS" set serviceaccount deploy/nginx over-sa
kubectl -n "$NS" rollout status deploy/nginx
echo "Injected over-permissive RBAC in $NS."
