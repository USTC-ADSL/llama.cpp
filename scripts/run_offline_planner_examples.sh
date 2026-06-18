#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-configs/offline_profile.yaml}"
DRY_RUN=0
RESUME=0
PROMPT_LEN="${PROMPT_LEN:-400}"
OUTPUT_LEN="${OUTPUT_LEN:-300}"
ALPHA="${ALPHA:-0.8}"
INCLUDE_TRANSITION="${INCLUDE_TRANSITION:-true}"
DEFAULT_TRANSITION_ENERGY_MJ="${DEFAULT_TRANSITION_ENERGY_MJ:-2000}"
DEFAULT_TRANSITION_LATENCY_MS="${DEFAULT_TRANSITION_LATENCY_MS:-50}"
OUTPUT="${OUTPUT:-profiles/request_plan_example.csv}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --resume) RESUME=1; shift ;;
        --prompt-len) PROMPT_LEN="$2"; shift 2 ;;
        --output-len) OUTPUT_LEN="$2"; shift 2 ;;
        --alpha) ALPHA="$2"; shift 2 ;;
        --include-transition) INCLUDE_TRANSITION="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

cmd=(
    python3 "${ROOT_DIR}/profiles/offline_bucket_planner.py"
    --config "${CONFIG}"
    --prompt-len "${PROMPT_LEN}"
    --output-len "${OUTPUT_LEN}"
    --alpha "${ALPHA}"
    --include-transition "${INCLUDE_TRANSITION}"
    --default-transition-energy-mj "${DEFAULT_TRANSITION_ENERGY_MJ}"
    --default-transition-latency-ms "${DEFAULT_TRANSITION_LATENCY_MS}"
    --output "${OUTPUT}"
)

echo "[run_offline_planner_examples] output: ${ROOT_DIR}/${OUTPUT}"
if (( DRY_RUN )); then
    printf '[dry-run]'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    exit 0
fi
exec "${cmd[@]}"
