#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SRC_ROOT="${SRC_ROOT:-$REPO_ROOT/models/Qwen3-AoT}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/models/Qwen3-AoT-runtime}"

usage() {
    cat <<EOF
Usage:
  $(basename "$0") [--src-root <path>] [--out-root <path>]

Description:
  Build a runtime-only Qwen3 AoT bundle for stage-level QNN routes.
  The output keeps only:
    - split-graph layer .bin files
    - merged route JSON files under qwen3-qnn-full/

  It intentionally drops export-only artifacts such as:
    - onnx_model/
    - x86_64-linux-clang/
    - data/
    - build logs
    - input lists

Options:
  --src-root <path>   source Qwen3 AoT root (default: $SRC_ROOT)
  --out-root <path>   output runtime bundle root (default: $OUT_ROOT)
  -h, --help          show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --src-root)
            SRC_ROOT="$2"
            shift 2
            ;;
        --out-root)
            OUT_ROOT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ ! -d "$SRC_ROOT" ]]; then
    echo "missing source root: $SRC_ROOT" >&2
    exit 1
fi

link_or_copy() {
    local src="$1"
    local dst="$2"

    mkdir -p "$(dirname "$dst")"
    rm -f "$dst"
    if ! ln "$src" "$dst" 2>/dev/null; then
        cp -p "$src" "$dst"
    fi
}

copy_layer_bins() {
    local subdir="$1"
    local src_dir="$SRC_ROOT/$subdir"
    local dst_dir="$OUT_ROOT/$subdir"

    if [[ ! -d "$src_dir" ]]; then
        echo "skip missing split dir: $src_dir" >&2
        return
    fi

    while IFS= read -r -d '' file; do
        local rel="${file#$SRC_ROOT/}"
        link_or_copy "$file" "$OUT_ROOT/$rel"
    done < <(find "$src_dir" -mindepth 3 -maxdepth 3 -type f -name '*.bin' -print0 | sort -z)
}

copy_route_jsons() {
    local src_dir="$SRC_ROOT/qwen3-qnn-full"
    local dst_dir="$OUT_ROOT/qwen3-qnn-full"

    mkdir -p "$dst_dir"
    while IFS= read -r -d '' file; do
        local rel="${file#$SRC_ROOT/}"
        link_or_copy "$file" "$OUT_ROOT/$rel"
    done < <(find "$src_dir" -maxdepth 1 -type f -name '*.json' -print0 | sort -z)
}

calc_bytes() {
    local target="$1"
    python - "$target" <<'PY'
import os
import sys

root = sys.argv[1]
total = 0
for base, _, files in os.walk(root):
    for name in files:
        total += os.path.getsize(os.path.join(base, name))
print(total)
PY
}

rm -rf "$OUT_ROOT"
mkdir -p "$OUT_ROOT"

copy_layer_bins "qwen3-qnn-attn-proj"
copy_layer_bins "qwen3-qnn-attn-core"
copy_layer_bins "qwen3-qnn-attn-core-f16"
copy_layer_bins "qwen3-qnn-ffn"
copy_route_jsons

src_bytes="$(calc_bytes "$SRC_ROOT")"
out_bytes="$(calc_bytes "$OUT_ROOT")"

echo "runtime bundle prepared"
echo "src_root=$SRC_ROOT"
echo "out_root=$OUT_ROOT"
echo "src_bytes=$src_bytes"
echo "out_bytes=$out_bytes"
