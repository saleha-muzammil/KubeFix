#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-http://localhost:8085}   # <-- set to your port-forward; you used 8085
NS=${NS:-demo}
TRIALS=${TRIALS:-10}
OUT=${OUT:-results.csv}

# pick a pod
POD=$(kubectl -n "$NS" get pod -l app=nginx -o jsonpath='{.items[0].metadata.name}')

# header once
echo "model,trial,elapsed_sec,rbac_patch,netpol_patch,podsec_patch,confidence" > "$OUT"

MODEL=${GEMINI_MODEL:-gemini-2.0-flash}
for i in $(seq 1 "$TRIALS"); do
  t0=$(date +%s)
  payload=$(printf '{"mic":{"kind":"Pod","namespace":"%s","name":"%s"}}' "$NS" "$POD")
  resp=$(echo "$payload" | curl -s -X POST "$BASE/generate" -H 'Content-Type: application/json' --data-binary @-)
  t1=$(date +%s)
  elapsed=$((t1 - t0))

  # booleans: true if non-null/non-empty string
  rbac=$(echo "$resp" | jq -r '(.gen.rbac_patch   | (type=="string" and length>0))')
  netp=$(echo "$resp" | jq -r '(.gen.netpol_patch | (type=="string" and length>0))')
  podp=$(echo "$resp" | jq -r '(.gen.podsec_patch | (type=="string" and length>0))')
  conf=$(echo "$resp" | jq -r '.gen.confidence // 0')

  echo "$MODEL,$i,$elapsed,$rbac,$netp,$podp,$conf" | tee -a "$OUT" >/dev/null
done

echo "Wrote $OUT"

