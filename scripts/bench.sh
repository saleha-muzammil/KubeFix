#!/usr/bin/env bash
set -euo pipefail

: "${BASE:=http://kubefix.kubefix.svc.cluster.local:8080}"
: "${TRIALS:=10}"

bench_ns="kubefix"
bench_pod="bench"
model="${GEMINI_MODEL:-gemini-2.0-flash}"
out_csv="results123.csv"

echo "Using in-cluster URL: $BASE"
kubectl -n "$bench_ns" get pod "$bench_pod" >/dev/null 2>&1 || \
  kubectl -n "$bench_ns" run "$bench_pod" --image=curlimages/curl:8.11.1 --restart=Never -- sh -lc 'sleep 36000'
kubectl -n "$bench_ns" wait --for=condition=Ready pod/"$bench_pod" --timeout=120s >/dev/null

if ! [ -f "$out_csv" ]; then
  echo "scenario,model,elapsed_sec,rbac_patch,netpol_patch,podsec_patch,confidence,status" > "$out_csv"
fi

call_generate () {
  local ns="$1" pod="$2" scenario="$3"

  local payload; payload=$(printf '{"mic":{"kind":"Pod","namespace":"%s","name":"%s"}}' "$ns" "$pod")
  local t0 t1 elapsed; t0=$(date +%s)

  # Encode inside pod to avoid control chars/newlines
  set +e
  local resp_b64
  resp_b64=$(kubectl -n "$bench_ns" exec "$bench_pod" -- sh -lc \
    "printf '%s' '$payload' \
     | curl -s -m 30 -H 'Accept: application/json' -H 'Content-Type: application/json' \
       -X POST '$BASE/generate' --data-binary @- \
     | base64 -w0")
  rc=$?
  set -e

  t1=$(date +%s); elapsed=$((t1 - t0))
  if [ $rc -ne 0 ] || [ -z "${resp_b64:-}" ]; then
    echo "$scenario,$model,$elapsed,false,false,false,0,HTTP_ERR" | tee -a "$out_csv"
    return
  fi

  # Decode locally
  local resp; resp=$(echo "$resp_b64" | base64 -d 2>/dev/null || true)
  if ! echo "$resp" | jq -e . >/dev/null 2>&1; then
    echo "$scenario,$model,$elapsed,false,false,false,0,PARSE_ERR" | tee -a "$out_csv"
    return
  fi

  local rbac netpol podsec conf status
  rbac=$(echo "$resp" | jq -r '.gen.rbac_patch   | (.|type=="string")')
  netpol=$(echo "$resp" | jq -r '.gen.netpol_patch | (.|type=="string")')
  podsec=$(echo "$resp" | jq -r '.gen.podsec_patch | (.|type=="string")')
  conf=$(echo "$resp" | jq -r '.gen.confidence // 0')
  status="OK"

  echo "$scenario,$model,$elapsed,$rbac,$netpol,$podsec,$conf,$status" | tee -a "$out_csv"
}

run_trials () {
  local scenario="$1" ns="$2" label="$3"
  echo "Running scenario=$scenario ns=$ns label=$label trials=$TRIALS"

  for _ in $(seq 1 "$TRIALS"); do
    local pod
    pod=$(kubectl -n "$ns" get pod -l "$label" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    if [ -z "$pod" ]; then
      echo "$scenario,$model,0,false,false,false,0,NO_POD" | tee -a "$out_csv"
      continue
    fi
    call_generate "$ns" "$pod" "$scenario" || true
    sleep 1
  done
}

# Scenarios
run_trials "RBAC_PODSEC"  "demo"         "app=nginx"
run_trials "NETPOL"       "demo"         "app=nginx"
run_trials "PODSEC_ONLY"  "podsec-only"  "app=nginx"

echo "Benchmark complete → $out_csv"

