#!/usr/bin/env bash
set -euo pipefail
NS=${NS:-demo}
kubectl -n "$NS" delete networkpolicy --all || true
echo "Removed all NetPols in $NS."
