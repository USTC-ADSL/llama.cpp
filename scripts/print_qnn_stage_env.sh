#!/bin/bash

set -euo pipefail

MODELS_ROOT="/data/local/tmp/acom-stage-models"
MODEL_FAMILY=""
ROUTE="attn_core"
GGUF_PATH=""

print_help() {
    cat <<EOF
Usage:
  eval "\$($(basename "$0") --model qwen3 --route attn_core [--models-root <path>] [--gguf <path>])"

Description:
  Print export commands for GGML_QNN_AOT_CONFIG / GGML_QNN_AOT_MODEL_DIR from
  one shared device-side models root, so mixed-route experiments do not need to
  hardcode per-device AoT paths in every command.

Options:
  --model <qwen2|qwen3>     model family (required)
  --route <name>            one of: fullgraph, attn_proj, attn_core, attn_core_f16, ffn,
                            attnproj_ffn, attnproj_attncore_ffn
  --models-root <path>      shared device-side models root
  --gguf <path>             optional device-side GGUF path; if omitted, a Qwen3
                            default is used and Qwen2 leaves LLAMA_MODEL unset
  -h, --help                show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL_FAMILY="$2"
            shift 2
            ;;
        --route)
            ROUTE="$2"
            shift 2
            ;;
        --models-root)
            MODELS_ROOT="$2"
            shift 2
            ;;
        --gguf)
            GGUF_PATH="$2"
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

if [[ "$MODELS_ROOT" != /* ]]; then
    echo "--models-root must be an absolute device path" >&2
    exit 1
fi

if [[ -z "$MODEL_FAMILY" ]]; then
    echo "--model is required" >&2
    exit 1
fi

case "$MODEL_FAMILY" in
    qwen2)
        aot_root="$MODELS_ROOT/Qwen2-AoT"
        fullgraph_dir="$aot_root/qwen2-qnn-full/qnn"
        default_gguf=""
        case "$ROUTE" in
            fullgraph)
                aot_config="$fullgraph_dir/config.json"
                aot_model_dir="$fullgraph_dir"
                ;;
            attn_proj)
                aot_config="$aot_root/qwen2-qnn-full/qnn_attn_proj_combined.json"
                aot_model_dir="$aot_root"
                ;;
            attn_core)
                aot_config="$aot_root/qwen2-qnn-full/qnn_attn_core_combined.json"
                aot_model_dir="$aot_root"
                ;;
            ffn)
                aot_config="$aot_root/qwen2-qnn-full/qnn_ffn_combined.json"
                aot_model_dir="$aot_root"
                ;;
            attnproj_ffn)
                aot_config="$aot_root/qwen2-qnn-full/qnn_attnproj_ffn_combined.json"
                aot_model_dir="$aot_root"
                ;;
            attnproj_attncore_ffn)
                aot_config="$aot_root/qwen2-qnn-full/qnn_attnproj_attncore_ffn_combined.json"
                aot_model_dir="$aot_root"
                ;;
            *)
                echo "unsupported route for qwen2: $ROUTE" >&2
                exit 1
                ;;
        esac
        ;;
    qwen3)
        aot_root="$MODELS_ROOT/Qwen3-AoT"
        fullgraph_dir="$aot_root/qwen3-qnn-full/qnn"
        default_gguf="$MODELS_ROOT/Qwen3-1.7B-Q4_0/qwen3-1.7b-q4_0.gguf"
        case "$ROUTE" in
            fullgraph)
                aot_config="$fullgraph_dir/config.json"
                aot_model_dir="$fullgraph_dir"
                ;;
            attn_proj)
                aot_config="$aot_root/qwen3-qnn-full/qnn_attn_proj_combined.json"
                aot_model_dir="$aot_root"
                ;;
            attn_core)
                aot_config="$aot_root/qwen3-qnn-full/qnn_attn_core_combined.json"
                aot_model_dir="$aot_root"
                ;;
            attn_core_f16)
                aot_config="$aot_root/qwen3-qnn-full/qnn_attn_core_f16_combined.json"
                aot_model_dir="$aot_root"
                ;;
            ffn)
                aot_config="$aot_root/qwen3-qnn-full/qnn_ffn_combined.json"
                aot_model_dir="$aot_root"
                ;;
            attnproj_ffn)
                aot_config="$aot_root/qwen3-qnn-full/qnn_attnproj_ffn_combined.json"
                aot_model_dir="$aot_root"
                ;;
            attnproj_attncore_ffn)
                aot_config="$aot_root/qwen3-qnn-full/qnn_attnproj_attncore_ffn_combined.json"
                aot_model_dir="$aot_root"
                ;;
            *)
                echo "unsupported route for qwen3: $ROUTE" >&2
                exit 1
                ;;
        esac
        ;;
    *)
        echo "unsupported model family: $MODEL_FAMILY" >&2
        exit 1
        ;;
esac

if [[ -z "$GGUF_PATH" ]]; then
    GGUF_PATH="$default_gguf"
fi

printf "export GGML_QNN_AOT_CONFIG='%s'\n" "$aot_config"
printf "export GGML_QNN_AOT_MODEL_DIR='%s'\n" "$aot_model_dir"
if [[ -n "$GGUF_PATH" ]]; then
    printf "export LLAMA_MODEL='%s'\n" "$GGUF_PATH"
fi
