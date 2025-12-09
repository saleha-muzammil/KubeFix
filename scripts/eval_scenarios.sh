#!/usr/bin/env bash
set -euo pipefail

# Root of the repo
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# KubeFix HTTP endpoint
BASE=${BASE:-http://localhost:8085}

# Python interpreter
PYTHON=${PYTHON:-python3}

SCENARIO=${1:-RBAC_PODSEC}

# If user asked for ALL, loop over all known scenarios and exit.
if [[ "$SCENARIO" == "ALL" ]]; then
  ALL_SCENARIOS=(
    "RBAC_PODSEC"
    "PODSEC_ONLY"
    "NETPOL_MISSING"
    "MITM_VICTIM"
    "CVE-2023-2431"
    "CVE-2022-3162"
    "CVE-2019-11247"
  )

  for S in "${ALL_SCENARIOS[@]}"; do
    echo
    echo "########################################"
    echo "[*] eval_scenarios.sh ALL → $S"
    echo "########################################"
    "$0" "$S"
  done
  exit 0
fi

# Map scenario -> namespace + label selector for focus pod
case "$SCENARIO" in
  RBAC_PODSEC)
    NS="rbac-podsec"
    LABEL="app=nginx"
    BASE_YAML="$ROOT_DIR/scenarios/RBAC_PODSEC/base.yaml"
    ;;
  PODSEC_ONLY)
    NS="podsec-only"
    LABEL="app=insecure-web"
    BASE_YAML="$ROOT_DIR/scenarios/PODSEC_ONLY/base.yaml"
    ;;
  NETPOL_MISSING)
    NS="payments"
    LABEL="app=payments-api"
    BASE_YAML="$ROOT_DIR/scenarios/NETPOL_MISSING/base.yaml"
    ;;
  MITM_VICTIM)
    NS="default"
    LABEL="run=victim"
    BASE_YAML="$ROOT_DIR/scenarios/MITM_VICTIM/base.yaml"
    ;;
  CVE-2023-2431)
    NS="cve-2023-2431"
    LABEL="app=my-app"
    BASE_YAML="$ROOT_DIR/scenarios/CVE-2023-2431/base.yaml"
    ;;
  CVE-2022-3162)
    NS="default"
    LABEL="app=my-app"
    BASE_YAML="$ROOT_DIR/scenarios/CVE-2022-3162/base.yaml"
    ;;
  CVE-2019-11247)
    NS="default"
    LABEL="app=my-app"
    BASE_YAML="$ROOT_DIR/scenarios/CVE-2019-11247/base.yaml"
    ;;
  *)
    echo "[!] Unknown scenario: $SCENARIO" >&2
    exit 1
    ;;
esac

echo
echo "=============================="
echo "[*] Evaluating scenario: $SCENARIO"
echo "=============================="

# 1) Apply base manifests
echo "[*] Applying base manifests for $SCENARIO"
kubectl apply -f "$BASE_YAML"

# 2) Wait for pods in NS to be Ready
echo "[*] Waiting for pods in namespace '$NS' to be Ready (if any)..."
kubectl -n "$NS" wait --for=condition=Ready pod --all --timeout=120s || true

# 3) Pick focus pod
POD=$(kubectl -n "$NS" get pod -l "$LABEL" -o jsonpath='{.items[0].metadata.name}')
if [[ -z "${POD:-}" ]]; then
  echo "[!] No pod found with label '$LABEL' in namespace '$NS'" >&2
  exit 1
fi
echo "[*] Selecting pod via label '$LABEL' in namespace '$NS'"
echo "[*] Focus: kind=Pod ns=$NS pod=$POD"

# 4) Call /generate once (for correctness evaluation)
echo "[*] Calling KubeFix /generate..."
PAYLOAD=$(printf '{"mic":{"kind":"Pod","namespace":"%s","name":"%s"}}' "$NS" "$POD")
RESP=$(echo "$PAYLOAD" | curl -s -X POST "$BASE/generate" \
  -H 'Content-Type: application/json' --data-binary @-)

echo "[*] Raw response from /generate:"
echo "$RESP" | jq .

OUT_DIR_IN_POD=$(echo "$RESP" | jq -r '.out_dir')
if [[ -z "${OUT_DIR_IN_POD:-}" || "$OUT_DIR_IN_POD" == "null" ]]; then
  echo "[!] Could not parse out_dir from response" >&2
  exit 1
fi
echo "[*] Out dir in pod: $OUT_DIR_IN_POD"

# 5) Copy artifacts out of kubefix pod
KFPOD=$(kubectl -n kubefix get pod -l app=kubefix -o jsonpath='{.items[0].metadata.name}')
TS=$(date +%s)
LOCAL_ROOT="$ROOT_DIR/kubefix-out/${SCENARIO}-${TS}"
mkdir -p "$LOCAL_ROOT"

echo "[*] Copying artifacts from kubefix pod..."
# NOTE: the trailing '/.' copies the contents of OUT_DIR_IN_POD into LOCAL_ROOT
kubectl -n kubefix cp "${KFPOD}:${OUT_DIR_IN_POD}/." "$LOCAL_ROOT"

MIC="$LOCAL_ROOT/mic.json"
PATCH_DIR="$LOCAL_ROOT/patches"

echo "[*] MIC path:    $MIC"
echo "[*] Patches dir: $PATCH_DIR"

if [[ ! -f "$MIC" ]]; then
  echo "[!] mic.json not found at $MIC" >&2
  exit 1
fi

if [[ ! -d "$PATCH_DIR" ]]; then
  echo "[!] patches/ directory not found at $PATCH_DIR" >&2
else
  if ls "$PATCH_DIR"/*.yaml >/dev/null 2>&1; then
    echo "[*] Found patch YAML files:"
    ls "$PATCH_DIR"/*.yaml
  else
    echo "[!] No patch YAML files found in $PATCH_DIR – nothing to apply."
  fi
fi

# 6) Optional functional check.sh
CHECK="$ROOT_DIR/scenarios/${SCENARIO}/check.sh"
if [[ -x "$CHECK" ]]; then
  echo "[*] Running functional check.sh for $SCENARIO ..."
  "$CHECK" || echo "[!] check.sh reported a failure"
else
  echo "[!] No executable check.sh for $SCENARIO (expected at $CHECK)"
fi

# 7) Run kubefix.py verify (policy-level evaluation)
EVAL_LOG="$ROOT_DIR/eval-${SCENARIO}.log"
echo "[*] Running kubefix.py verify..."
"$PYTHON" "$ROOT_DIR/kubefix.py" verify --mic "$MIC" --patches "$PATCH_DIR" | tee "$EVAL_LOG"

echo
echo "=============================="
echo "[*] Evaluation complete: $SCENARIO"
echo "[*] Artifacts directory: $LOCAL_ROOT"
echo "[*] Verify log:          $EVAL_LOG"
echo "=============================="

