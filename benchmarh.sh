#!/usr/bin/env bash
set -euo pipefail

# KubeFix HTTP endpoint
BASE=${BASE:-http://localhost:8085}

# Scenario + model come from run_scenaro.sh
SCENARIO=${SCENARIO:-RBAC_PODSEC}
MODEL=${MODEL:-gemini-2.0-flash}

# How many times to call /generate
TRIALS=${TRIALS:-10}

# Namespace + label used to pick the focus pod
NS=${NS:-demo}
LABEL=${LABEL:-app=nginx}

# Allow caller to override POD directly; otherwise derive from label
if [[ -z "${POD:-}" ]]; then
  POD=$(kubectl -n "$NS" get pod -l "$LABEL" -o jsonpath='{.items[0].metadata.name}')
fi

if [[ -z "${POD:-}" ]]; then
  echo "[!] benchmark.sh: no pod found in ns='$NS' with label '$LABEL'" >&2
  exit 1
fi

# Sanitize model name for filename (replace '/' with '_')
SAFE_MODEL=${MODEL//\//_}

# Default output file: one per (scenario,model)
OUT=${OUT:-"results-${SCENARIO}-${SAFE_MODEL}.csv"}

# CSV header
echo "scenario,model,trial,elapsed_sec,rbac_patch,netpol_patch,podsec_patch,confidence,prompt_tokens,completion_tokens,total_tokens,input_cost_usd,output_cost_usd,total_cost_usd" > "$OUT"

for i in $(seq 1 "$TRIALS"); do
  # 1) Call /generate and measure wall-clock latency
  t0=$(date +%s)
  payload=$(printf '{"mic":{"kind":"Pod","namespace":"%s","name":"%s"}}' "$NS" "$POD")

  resp=$(echo "$payload" | curl -s -X POST "$BASE/generate" \
    -H 'Content-Type: application/json' \
    --data-binary @-)

  t1=$(date +%s)
  elapsed=$((t1 - t0))

  # Make sure we got JSON back; if not, log and skip this trial
  if ! echo "$resp" | jq -e . >/dev/null 2>&1; then
    echo "[!] benchmark.sh: non-JSON response for $SCENARIO / $MODEL / trial $i" >&2
    echo "[!] Raw response: $resp" >&2
    continue
  fi

  # 2) Basic patch presence + confidence
  rbac=$(echo "$resp" | jq -r '(.gen.rbac_patch   | (type=="string" and length>0))')
  netp=$(echo "$resp" | jq -r '(.gen.netpol_patch | (type=="string" and length>0))')
  podp=$(echo "$resp" | jq -r '(.gen.podsec_patch | (type=="string" and length>0))')
  conf=$(echo "$resp" | jq -r '.gen.confidence // 0')

  ########################################
  # 3) Token + cost accounting
  ########################################

  # First, try to use real usage stats if the backend provides them
  prompt_tokens=$(echo "$resp" | jq -r '.usage.prompt_tokens // empty')
  completion_tokens=$(echo "$resp" | jq -r '.usage.completion_tokens // empty')
  total_tokens=$(echo "$resp" | jq -r '.usage.total_tokens // empty')

  # If usage is missing, approximate from generated YAML + summary
  if [[ -z "$total_tokens" || "$total_tokens" == "null" ]]; then
    patch_text=$(echo "$resp" | jq -r '
      [
        .gen.rbac_patch,
        .gen.netpol_patch,
        .gen.podsec_patch,
        .gen.summary
      ]
      | map(select(. != null))
      | join("\n")
    ')

    # Rough heuristic: 1 token ≈ 4 characters
    approx_chars=$(printf '%s' "$patch_text" | wc -c | tr -d ' ')
    completion_tokens=$(( (approx_chars + 3) / 4 ))

    # Prompt tokens (MIC + instructions) – just use a small constant if not present
    if [[ -z "$prompt_tokens" || "$prompt_tokens" == "null" ]]; then
      prompt_tokens=64
    fi

    total_tokens=$((prompt_tokens + completion_tokens))
  fi

  # 4) Per-model prices (USD per 1,000,000 tokens)
  #    *** EDIT THESE to match your actual pricing ***
  case "$MODEL" in
    gemini-2.0-flash)
      in_price_per_million=0.35      # example: $0.35 / 1M input
      out_price_per_million=1.05     # example: $1.05 / 1M output
      ;;
    deepseek-ai/deepseek-coder-6.7b-instruct)
      # Put DeepSeek's real prices here
      in_price_per_million=0.14
      out_price_per_million=0.14
      ;;
    *)
      in_price_per_million=0
      out_price_per_million=0
      ;;
  esac

  # 5) Compute USD costs with awk (floating point)
  input_cost_usd=$(awk -v t="$prompt_tokens" -v p="$in_price_per_million" \
    'BEGIN { printf "%.8f", t * p / 1000000 }')

  output_cost_usd=$(awk -v t="$completion_tokens" -v p="$out_price_per_million" \
    'BEGIN { printf "%.8f", t * p / 1000000 }')

  total_cost_usd=$(awk -v a="$input_cost_usd" -v b="$output_cost_usd" \
    'BEGIN { printf "%.8f", a + b }')

  # 6) Write one CSV row
  echo "$SCENARIO,$MODEL,$i,$elapsed,$rbac,$netp,$podp,$conf,$prompt_tokens,$completion_tokens,$total_tokens,$input_cost_usd,$output_cost_usd,$total_cost_usd" >> "$OUT"
done

echo "Wrote $OUT"

