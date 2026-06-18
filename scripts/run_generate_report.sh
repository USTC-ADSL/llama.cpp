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

cmd=(python3 "${ROOT_DIR}/reports/generate_offline_report.py" --config "${CONFIG}")

echo "[run_generate_report] output: ${ROOT_DIR}/reports/offline_profile_summary.md"
if (( DRY_RUN )); then
    printf '[dry-run]'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    exit 0
fi
exec "${cmd[@]}"
