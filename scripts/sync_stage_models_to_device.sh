#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-$REPO_ROOT/models}"
TARGET_ROOT="/data/local/tmp/acom-stage-models"
DEVICE_SERIAL=""
SYNC_ITEMS=()

print_help() {
    cat <<EOF
Usage:
  $(basename "$0") --device <serial> [--target-root <abs-path>] [--item <name>]...

Description:
  Sync selected model/AoT directories from the repository models/ tree to one
  device-side models root so multiple binaries can share the same artifacts.

Options:
  --device <serial>         adb device serial (required)
  --target-root <path>      device models root (default: $TARGET_ROOT)
  --item <name>             one directory under local models/ to sync; can be
                            repeated. Defaults to:
                            Qwen2-AoT Qwen3-AoT Qwen3-1.7B-Q4_0 Qwen3-1.7B-Q8_0
  -h, --help                show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)
            DEVICE_SERIAL="$2"
            shift 2
            ;;
        --target-root)
            TARGET_ROOT="$2"
            shift 2
            ;;
        --item)
            SYNC_ITEMS+=("$2")
            shift 2
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            print_help >&2
            exit 1
            ;;
    esac
done

if [[ -z "$DEVICE_SERIAL" ]]; then
    echo "--device is required" >&2
    exit 1
fi

if [[ "$TARGET_ROOT" != /* ]]; then
    echo "--target-root must be an absolute device path" >&2
    exit 1
fi

if [[ ${#SYNC_ITEMS[@]} -eq 0 ]]; then
    SYNC_ITEMS=(
        "Qwen2-AoT"
        "Qwen3-AoT"
        "Qwen3-1.7B-Q4_0"
        "Qwen3-1.7B-Q8_0"
    )
fi

if ! command -v adb >/dev/null 2>&1; then
    echo "adb not found in PATH" >&2
    exit 1
fi

adb -s "$DEVICE_SERIAL" shell "mkdir -p '$TARGET_ROOT'"

for item in "${SYNC_ITEMS[@]}"; do
    src="$LOCAL_MODELS_ROOT/$item"
    if [[ ! -e "$src" ]]; then
        echo "skip missing local item: $src" >&2
        continue
    fi

    echo "sync $src -> $TARGET_ROOT/"
    adb -s "$DEVICE_SERIAL" push "$src" "$TARGET_ROOT/"
done

echo "done: device=$DEVICE_SERIAL target_root=$TARGET_ROOT"
