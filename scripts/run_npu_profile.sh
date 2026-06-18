#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-configs/offline_profile.yaml}"
DRY_RUN=0
RESUME=0
PHASE="${PHASE:-all}"
SANITY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --resume) RESUME=1; shift ;;
        --phase) PHASE="$2"; shift 2 ;;
        --sanity) SANITY=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

cmd=(python3 "${ROOT_DIR}/profiles/run_profile.py" --backend NPU --phase "${PHASE}" --config "${CONFIG}")
(( DRY_RUN )) && cmd+=(--dry-run)
(( RESUME )) && cmd+=(--resume)
(( SANITY )) && cmd+=(--sanity)

echo "[run_npu_profile] logs: ${ROOT_DIR}/logs"
echo "[run_npu_profile] manifests: ${ROOT_DIR}/profiles/manifests"
if (( DRY_RUN )); then
    printf '[dry-run]'
    printf ' %q' "${cmd[@]}"
    printf '\n'
fi
exec "${cmd[@]}"
