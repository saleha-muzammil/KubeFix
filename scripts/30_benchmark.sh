# scripts/30_benchmark.sh
#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-http://localhost:8085}
TRIALS=${TRIALS:-10}
MODEL=${GEMINI_MODEL:-gemini-2.0-flash}
OUTCSV=${OUTCSV:-results.csv}

ensure_pf() {
  if ! curl -sSf "$BASE/" >/dev/null 2>&1; then
    pkill -f "kubectl -n kubefix port-forward svc/kubefix" || true
    kubectl -n kubefix port-forward svc/kubefix 8085:8080 >/tmp/kubefix-pf.log 2>&1 &
    for i in {1..30}; do
      curl -sSf "$BASE/" >/dev/null 2>&1 && break
      sleep 1
    done
  fi
}

[ -s "$OUTCSV" ] || echo "scenario,model,trial,elapsed_sec,rbac_patch,netpol_patch,podsec_patch,confidence,out_dir" > "$OUTCSV"

bench_one() {
  local scenario=$1 ns=$2 sel=${3:-app=nginx}
  ensure_pf
  local pod
  pod=$(kubectl -n "$ns" get pod -l "$sel" -o jsonpath='{.items[0].metadata.name}')
  for i in $(seq 1 $TRIALS); do
    ensure_pf
    local payload resp elapsed t0 t1
    payload=$(printf '{"mic":{"kind":"Pod","namespace":"%s","name":"%s"}}' "$ns" "$pod")
    t0=$(date +%s)
    # Use --fail but don't kill the script on HTTP error
    set +e
    resp=$(curl -s -X POST "$BASE/generate" -H 'Content-Type: application/json' --data-binary "$payload")
    rc=$?
    set -e
    t1=$(date +%s)
    elapsed=$((t1 - t0))

    # Validate JSON before jq filters
    if ! echo "$resp" | jq -e 'type=="object" and (.gen|type=="object")' >/dev/null 2>&1; then
      echo "$scenario,$MODEL,$i,$elapsed,false,false,false,0,ERROR" | tee -a "$OUTCSV"
      continue
    fi

    has_rbac=$(echo "$resp" | jq -r '(.gen.rbac_patch   != null and (.gen.rbac_patch   | length>0))')
    has_netp=$(echo "$resp" | jq -r '(.gen.netpol_patch != null and (.gen.netpol_patch | length>0))')
    has_podp=$(echo "$resp" | jq -r '(.gen.podsec_patch != null and (.gen.podsec_patch | length>0))')
    conf=$(echo "$resp" | jq -r '.gen.confidence // 0')
    outd=$(echo "$resp" | jq -r '.out_dir // "NA"')

    echo "$scenario,$MODEL,$i,$elapsed,$has_rbac,$has_netp,$has_podp,$conf,$outd" | tee -a "$OUTCSV"
  done
}

bench_one RBAC_PODSEC demo
bench_one NETPOL demo
bench_one PODSEC_ONLY podsec-only

