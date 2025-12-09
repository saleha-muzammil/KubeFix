#!/usr/bin/env bash
set -euo pipefail

# Root of the repo
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

SCENARIO=${1:-RBAC_PODSEC}
TRIALS=${2:-10}

# Models to test for each scenario
MODELS=(
  "gemini-2.0-flash"
  "deepseek-ai/deepseek-coder-6.7b-instruct"
)

# Map scenario -> namespace + label selector + base manifest
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
    NS="cve-2022-3162"
    LABEL="app=my-app"
    BASE_YAML="$ROOT_DIR/scenarios/CVE-2022-3162/base.yaml"
    ;;
  CVE-2019-11247)
    NS="default"
    LABEL="app=my-app"
    BASE_YAML="$ROOT_DIR/scenarios/CVE-2019-11247/base.yaml"
    ;;
  ALL)
    # Special mode: iterate all scenarios one by one
    for s in RBAC_PODSEC PODSEC_ONLY NETPOL_MISSING MITM_VICTIM CVE-2023-2431 CVE-2022-3162 CVE-2019-11247; do
      echo
      echo "########################################"
      echo "[*] run_scenaro.sh ALL → $s"
      echo "########################################"
      "$0" "$s" "$TRIALS"
    done
    exit 0
    ;;
  *)
    echo "[!] Unknown scenario: $SCENARIO" >&2
    exit 1
    ;;
esac

echo
echo "=============================="
echo "[*] Scenario: $SCENARIO"
echo "=============================="
echo "[*] Applying manifests: $BASE_YAML"

kubectl apply -f "$BASE_YAML"

echo "[*] Waiting for pods in namespace '$NS' to be Ready (if any)..."
kubectl -n "$NS" wait --for=condition=Ready pod --all --timeout=120s || true

# Pick focus pod once (same pod for all models)
POD=$(kubectl -n "$NS" get pod -l "$LABEL" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
if [[ -z "${POD:-}" ]]; then
  echo "[!] No pod found with label '$LABEL' in namespace '$NS'" >&2
  exit 1
fi

echo "[*] Focus: kind=Pod ns=$NS pod=$POD"

for m in "${MODELS[@]}"; do
  echo
  echo "########################################"
  echo "[*] run_scenaro.sh $SCENARIO → $m"
  echo "########################################"

  # Per-model run: set env vars for benchmark.sh
  SCENARIO="$SCENARIO" \
  MODEL="$m" \
  NS="$NS" \
  LABEL="$LABEL" \
  POD="$POD" \
  TRIALS="$TRIALS" \
    "$ROOT_DIR/benchmark.sh"
done

