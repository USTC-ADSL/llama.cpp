#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-configs/offline_profile.yaml}"
DRY_RUN=0
RESUME=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --resume) RESUME=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

cmd=(python3 "${ROOT_DIR}/profiles/suggest_refinement.py" --config "${CONFIG}")

echo "[run_suggest_refinement] output: ${ROOT_DIR}/profiles/refinement_plan.csv"
if (( DRY_RUN )); then
    printf '[dry-run]'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    exit 0
fi
exec "${cmd[@]}"
