#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"

QNN_SDK="${QNN_SDK:-/mnt/sda1/yzh/qairt_2.44/qairt/2.44.0.260225}"
CONDA_BASE="${CONDA_BASE:-/home/miog/miniconda3}"
QNN_CONDA_ENV="${QNN_CONDA_ENV:-powerserve}"
EXPORT_DEVICE="${EXPORT_DEVICE:-cpu}"
QNN_CONVERTER_THREADS="${QNN_CONVERTER_THREADS:-16}"
QNN_BUILD_THREADS="${QNN_BUILD_THREADS:-1}"
QNN_MAX_SAMPLES="${QNN_MAX_SAMPLES:-16}"
SOC="${SOC:-8750}"
HTP_VERSION="${HTP_VERSION:-79}"

MODEL_FOLDER="${MODEL_FOLDER:-$REPO_ROOT/models/Qwen2.5/Qwen2.5-3B-Safe}"
MODEL_NAME="${MODEL_NAME:-qwen2.5_3b}"
ARTIFACT_NAME="${ARTIFACT_NAME:-qwen2.5_3b_stage}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/models/Qwen2.5/Qwen2.5-3B-StageQNN-$STAMP}"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/tmp/qwen2.5-3b-stage-qnn-$STAMP}"
STAGE_ROOT="${STAGE_ROOT:-/dev/shm/qwen2.5-3b-stage-qnn-$STAMP}"

PROMPT_FILE="${PROMPT_FILE:-$REPO_ROOT/ref/PowerServe/assets/calibration_data/service_lab_intro_qwen2.txt}"
SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-$REPO_ROOT/ref/PowerServe/assets/system_prompts/qwen2.txt}"
MAX_N_TOKENS="${MAX_N_TOKENS:-128}"
N_MODEL_CHUNKS="${N_MODEL_CHUNKS:-2}"
BATCH_SIZES="${BATCH_SIZES:-1}"
STAGES="${STAGES:-attn_proj attention ffn}"
LAYERS="${LAYERS:-all}"
DOWNLOAD_MODEL="${DOWNLOAD_MODEL:-0}"
BUILD_QNN="${BUILD_QNN:-1}"
ONNX_ONLY="${ONNX_ONLY:-0}"
COMBINE_BATCH_BIN="${COMBINE_BATCH_BIN:-0}"
PRUNE_RUNTIME="${PRUNE_RUNTIME:-0}"
HF_MODEL_ID="${HF_MODEL_ID:-Qwen/Qwen2.5-3B-Instruct}"
HF_HOME="${HF_HOME:-$REPO_ROOT/tmp/huggingface}"
EXPORT_WORK_DIR="${EXPORT_WORK_DIR:-$BUILD_DIR/export}"

if [[ -z "${LOG_DIR:-}" ]]; then
  if [[ "$PRUNE_RUNTIME" == "1" ]]; then
    LOG_DIR="$BUILD_DIR/logs"
  else
    LOG_DIR="$OUTPUT_DIR/logs"
  fi
fi

if [[ "$ONNX_ONLY" == "1" ]]; then
  BUILD_QNN=0
fi

if [[ "$COMBINE_BATCH_BIN" == "1" && "$BUILD_QNN" != "1" ]]; then
  echo "COMBINE_BATCH_BIN=1 requires BUILD_QNN=1." >&2
  exit 1
fi

if [[ ! -d "$QNN_SDK" ]]; then
  echo "QNN SDK folder not found: $QNN_SDK" >&2
  exit 1
fi

if [[ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  echo "conda init script not found: $CONDA_BASE/etc/profile.d/conda.sh" >&2
  exit 1
fi

if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "Prompt file not found: $PROMPT_FILE" >&2
  exit 1
fi

if [[ ! -f "$SYSTEM_PROMPT_FILE" ]]; then
  echo "System prompt file not found: $SYSTEM_PROMPT_FILE" >&2
  exit 1
fi

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "OUTPUT_DIR already exists, refusing to overwrite: $OUTPUT_DIR" >&2
  exit 1
fi

if [[ -e "$BUILD_DIR" ]]; then
  echo "BUILD_DIR already exists, refusing to overwrite: $BUILD_DIR" >&2
  exit 1
fi

if [[ "$LOG_DIR" == "$OUTPUT_DIR"* && -e "$LOG_DIR" ]]; then
  echo "LOG_DIR already exists, refusing to overwrite: $LOG_DIR" >&2
  exit 1
fi

if [[ ! "$QNN_CONVERTER_THREADS" =~ ^[1-9][0-9]*$ ]]; then
  echo "QNN_CONVERTER_THREADS must be a positive integer, got: $QNN_CONVERTER_THREADS" >&2
  exit 1
fi

if [[ ! "$QNN_BUILD_THREADS" =~ ^[1-9][0-9]*$ ]]; then
  echo "QNN_BUILD_THREADS must be a positive integer, got: $QNN_BUILD_THREADS" >&2
  exit 1
fi

if [[ ! "$QNN_MAX_SAMPLES" =~ ^[0-9]+$ ]]; then
  echo "QNN_MAX_SAMPLES must be a non-negative integer, got: $QNN_MAX_SAMPLES" >&2
  exit 1
fi

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

resolve_layers() {
  if [[ "$LAYERS" == "all" ]]; then
    seq 0 35
  else
    printf '%s\n' $LAYERS
  fi
}

stage_exporter_for() {
  case "$1" in
    attn_proj) echo "$REPO_ROOT/tools/qnn-aot-export/export_attn_proj_to_onnx.py" ;;
    attention) echo "$REPO_ROOT/tools/qnn-aot-export/export_attention_to_onnx.py" ;;
    ffn) echo "$REPO_ROOT/tools/qnn-aot-export/export_ffn_to_onnx.py" ;;
    *)
      echo "Unknown stage: $1" >&2
      exit 1
      ;;
  esac
}

layer_dir_name() {
  case "$1" in
    attn_proj) echo "attn_proj_layer_$2" ;;
    attention) echo "attention_layer_$2" ;;
    ffn) echo "ffn_layer_$2" ;;
    *)
      echo "Unknown stage: $1" >&2
      exit 1
      ;;
  esac
}

graph_prefix() {
  case "$1" in
    attn_proj) echo "attn_proj_layer_$2" ;;
    attention) echo "attention_layer_$2" ;;
    ffn) echo "ffn_layer_$2" ;;
    *)
      echo "Unknown stage: $1" >&2
      exit 1
      ;;
  esac
}

download_model_if_requested() {
  if compgen -G "$MODEL_FOLDER/*.safetensors" >/dev/null; then
    return
  fi

  if [[ "$DOWNLOAD_MODEL" != "1" ]]; then
    echo "Model safetensors not found in $MODEL_FOLDER" >&2
    echo "Set MODEL_FOLDER to a local Qwen2.5-3B safetensors directory or run with DOWNLOAD_MODEL=1." >&2
    exit 1
  fi

  mkdir -p "$MODEL_FOLDER" "$HF_HOME"
  export HF_HOME
  run_cmd hf download "$HF_MODEL_ID" --local-dir "$MODEL_FOLDER"
}

merge_configs() {
  python - "$OUTPUT_DIR" "$BATCH_SIZES" "$STAGES" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
batch_sizes = sys.argv[2].split()
stages = sys.argv[3].split()

merged = {"model_parameters": {}, "qnn_parameters": {}, "graphs": [], "embeddings": []}

for batch in batch_sizes:
    for stage in stages:
        cfg_path = output_dir / f"batch_{batch}" / stage / f"config_batch_{batch}.json"
        if not cfg_path.exists():
            continue
        with cfg_path.open("r") as f:
            data = json.load(f)
        if not merged["model_parameters"]:
            merged["model_parameters"] = data.get("model_parameters", {})
        if not merged["qnn_parameters"]:
            merged["qnn_parameters"] = data.get("qnn_parameters", {})
        prefix = f"batch_{batch}/{stage}"
        for graph in data.get("graphs", []):
            graph = dict(graph)
            if graph.get("model_path"):
                graph["model_path"] = f"{prefix}/{graph['model_path']}"
            if graph.get("kv_path_format"):
                graph["kv_path_format"] = f"{prefix}/{graph['kv_path_format']}"
            merged["graphs"].append(graph)

with (output_dir / "config.json").open("w") as f:
    json.dump(merged, f, indent=2)
PY
}

merge_combined_configs() {
  python - "$OUTPUT_DIR" "$EXPORT_WORK_DIR" "$BATCH_SIZES" "$STAGES" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
export_work_dir = Path(sys.argv[2])
batch_sizes = sys.argv[3].split()
stages = sys.argv[4].split()

merged = {"model_parameters": {}, "qnn_parameters": {}, "graphs": [], "embeddings": []}

for batch in batch_sizes:
    for stage in stages:
        cfg_path = export_work_dir / f"batch_{batch}" / stage / f"config_batch_{batch}.json"
        if not cfg_path.exists():
            continue
        with cfg_path.open("r") as f:
            data = json.load(f)
        if not merged["model_parameters"]:
            merged["model_parameters"] = data.get("model_parameters", {})
        if not merged["qnn_parameters"]:
            merged["qnn_parameters"] = data.get("qnn_parameters", {})
        for graph in data.get("graphs", []):
            graph = dict(graph)
            model_path = graph.get("model_path")
            if model_path:
                layer_dir = Path(model_path).parts[0]
                graph["model_path"] = f"{stage}/{layer_dir}/{layer_dir}.bin"
            merged["graphs"].append(graph)

with (output_dir / "config.json").open("w") as f:
    json.dump(merged, f, indent=2)
PY
}

copy_qnn_runtime_libs() {
  run_cmd cp "$QNN_SDK/lib/aarch64-android/libQnnSystem.so" "$OUTPUT_DIR/"
  run_cmd cp "$QNN_SDK/lib/aarch64-android/libQnnHtp.so" "$OUTPUT_DIR/"
  run_cmd cp "$QNN_SDK/lib/aarch64-android/libQnnHtpV${HTP_VERSION}Stub.so" "$OUTPUT_DIR/"

  local htp_lib_folder="$QNN_SDK/lib/hexagon-v${HTP_VERSION}/unsigned"
  run_cmd cp "$htp_lib_folder/libQnnHtpV${HTP_VERSION}.so" "$OUTPUT_DIR/"
  run_cmd cp "$htp_lib_folder/libQnnHtpV${HTP_VERSION}Skel.so" "$OUTPUT_DIR/"
  if [[ -f "$htp_lib_folder/libQnnHexagonSkel_dspApp.so" ]]; then
    run_cmd cp "$htp_lib_folder/libQnnHexagonSkel_dspApp.so" "$OUTPUT_DIR/"
  fi
}

build_layer_qnn_binary() {
  local stage="$1"
  local batch="$2"
  local layer="$3"
  local stage_out="$OUTPUT_DIR/batch_${batch}/${stage}"
  local layer_name
  local graph_base
  local graph_name

  layer_name="$(layer_dir_name "$stage" "$layer")"
  graph_base="$(graph_prefix "$stage" "$layer")"
  graph_name="${graph_base}_batch_${batch}"

  run_cmd python "$REPO_ROOT/ref/PowerServe/tools/qnn_converter/build_shared_object_staged.py" \
    --silent \
    --model "$stage_out/$layer_name/onnx_model/$graph_name.onnx" \
    --encoding "$stage_out/$layer_name/$graph_name.encodings" \
    --io-spec "$stage_out/$layer_name/$graph_name.io.json" \
    --input-list "$stage_out/$layer_name/input_list.txt" \
    --output-folder "$stage_out/$layer_name" \
    --artifact-name "$layer_name" \
    --graph-names "$graph_name" \
    --stage-root "$STAGE_ROOT" \
    --stage-name "${stage}_b${batch}_l${layer}" \
    --max-samples "$QNN_MAX_SAMPLES"

  run_cmd python "$REPO_ROOT/ref/PowerServe/tools/qnn_converter/generate_binary.py" \
    --silent \
    --build-folder "$stage_out/$layer_name" \
    --artifact-name "$layer_name" \
    --graph-names "$graph_name" \
    --soc "$SOC"
}

stage_output_dir_for() {
  local stage="$1"
  local batch="$2"
  if [[ "$COMBINE_BATCH_BIN" == "1" ]]; then
    echo "$EXPORT_WORK_DIR/batch_${batch}/${stage}"
  else
    echo "$OUTPUT_DIR/batch_${batch}/${stage}"
  fi
}

build_layer_qnn_library_into() {
  local stage="$1"
  local batch="$2"
  local layer="$3"
  local combined_layer_out="$4"
  local stage_out
  local layer_name
  local graph_base
  local graph_name

  stage_out="$(stage_output_dir_for "$stage" "$batch")"
  layer_name="$(layer_dir_name "$stage" "$layer")"
  graph_base="$(graph_prefix "$stage" "$layer")"
  graph_name="${graph_base}_batch_${batch}"

  run_cmd python "$REPO_ROOT/ref/PowerServe/tools/qnn_converter/build_shared_object_staged.py" \
    --silent \
    --model "$stage_out/$layer_name/onnx_model/$graph_name.onnx" \
    --encoding "$stage_out/$layer_name/$graph_name.encodings" \
    --io-spec "$stage_out/$layer_name/$graph_name.io.json" \
    --input-list "$stage_out/$layer_name/input_list.txt" \
    --output-folder "$combined_layer_out" \
    --artifact-name "$layer_name" \
    --graph-names "$graph_name" \
    --stage-root "$STAGE_ROOT" \
    --stage-name "${stage}_b${batch}_l${layer}" \
    --max-samples "$QNN_MAX_SAMPLES"
}

generate_combined_layer_qnn_binary() {
  local stage="$1"
  local layer="$2"
  local combined_layer_out="$3"
  local layer_name
  local graph_base
  local graph_names=()
  local batch

  layer_name="$(layer_dir_name "$stage" "$layer")"
  graph_base="$(graph_prefix "$stage" "$layer")"
  for batch in $BATCH_SIZES; do
    graph_names+=("${graph_base}_batch_${batch}")
  done

  run_cmd python "$REPO_ROOT/ref/PowerServe/tools/qnn_converter/generate_binary.py" \
    --silent \
    --build-folder "$combined_layer_out" \
    --artifact-name "$layer_name" \
    --graph-names "${graph_names[@]}" \
    --soc "$SOC"
}

prune_runtime_package() {
  find "$OUTPUT_DIR" -type d -name x86_64-linux-clang -prune -exec rm -rf {} +
  find "$OUTPUT_DIR" -type f \( \
    -name '*.onnx' -o \
    -name '*.encodings' -o \
    -name '*.io.json' -o \
    -name 'input_list.txt' -o \
    -name 'build_so.log' -o \
    -name 'build_bin.log' -o \
    -name 'htp_config.json' -o \
    -name 'htp_setting.json' \
  \) -delete
}

export_stage_onnx() {
  local batch="$1"
  local stage="$2"
  local graph_name="batch_${batch}"
  local exporter
  local stage_out
  local log_file

  exporter="$(stage_exporter_for "$stage")"
  stage_out="$(stage_output_dir_for "$stage" "$batch")"
  mkdir -p "$stage_out"

  cmd=(
    python "$exporter"
    --n-threads "$QNN_CONVERTER_THREADS"
    --model-folder "$MODEL_FOLDER"
    --model-name "$MODEL_NAME"
    --graph-name "$graph_name"
    --device "$EXPORT_DEVICE"
    --prompt-file "$PROMPT_FILE"
    --output-folder "$stage_out"
    --max-n-tokens "$MAX_N_TOKENS"
    --n-model-chunks "$N_MODEL_CHUNKS"
    --layers "${SELECTED_LAYERS[@]}"
  )
  if [[ "$stage" == "attention" || "$stage" == "ffn" ]]; then
    cmd+=("--system-prompt-file" "$SYSTEM_PROMPT_FILE")
  fi

  log_file="$LOG_DIR/export_${stage}_${graph_name}.log"
  {
    printf '+'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    "${cmd[@]}"
  } 2>&1 | tee "$log_file"
}

download_model_if_requested

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$BUILD_DIR"

{
  printf 'date=%s\n' "$(date -Is)"
  printf 'repo=%s\n' "$REPO_ROOT"
  printf 'git_commit=%s\n' "$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || true)"
  printf 'model_folder=%s\n' "$MODEL_FOLDER"
  printf 'model_name=%s\n' "$MODEL_NAME"
  printf 'artifact_name=%s\n' "$ARTIFACT_NAME"
  printf 'output_dir=%s\n' "$OUTPUT_DIR"
  printf 'build_dir=%s\n' "$BUILD_DIR"
  printf 'qnn_sdk=%s\n' "$QNN_SDK"
  printf 'conda_env=%s\n' "$QNN_CONDA_ENV"
  printf 'batch_sizes=%s\n' "$BATCH_SIZES"
  printf 'stages=%s\n' "$STAGES"
  printf 'layers=%s\n' "$LAYERS"
  printf 'max_n_tokens=%s\n' "$MAX_N_TOKENS"
  printf 'qnn_max_samples=%s\n' "$QNN_MAX_SAMPLES"
  printf 'build_qnn=%s\n' "$BUILD_QNN"
  printf 'combine_batch_bin=%s\n' "$COMBINE_BATCH_BIN"
  printf 'prune_runtime=%s\n' "$PRUNE_RUNTIME"
  printf 'export_work_dir=%s\n' "$EXPORT_WORK_DIR"
  printf 'log_dir=%s\n' "$LOG_DIR"
  printf 'argv=%q' "$0"
  printf ' %q' "$@"
  printf '\n'
} > "$OUTPUT_DIR/command.txt"

git -C "$REPO_ROOT" status --short > "$OUTPUT_DIR/git_status.txt" || true
git -C "$REPO_ROOT" rev-parse HEAD > "$OUTPUT_DIR/git_commit.txt" 2>/dev/null || true

set +u
source "$QNN_SDK/bin/envsetup.sh"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$QNN_CONDA_ENV"
set -u

export QNN_SDK_ROOT="$QNN_SDK"
export HF_HOME

mapfile -t SELECTED_LAYERS < <(resolve_layers)

for batch in $BATCH_SIZES; do
  for stage in $STAGES; do
    export_stage_onnx "$batch" "$stage"

    if [[ "$BUILD_QNN" == "1" && "$COMBINE_BATCH_BIN" != "1" ]]; then
      for layer in "${SELECTED_LAYERS[@]}"; do
        build_log="$LOG_DIR/build_${stage}_b${batch}_l${layer}.log"
        build_layer_qnn_binary "$stage" "$batch" "$layer" 2>&1 | tee "$build_log"
      done
    fi
  done
done

if [[ "$COMBINE_BATCH_BIN" == "1" ]]; then
  for stage in $STAGES; do
    for layer in "${SELECTED_LAYERS[@]}"; do
      layer_name="$(layer_dir_name "$stage" "$layer")"
      combined_layer_out="$OUTPUT_DIR/$stage/$layer_name"
      mkdir -p "$combined_layer_out"
      for batch in $BATCH_SIZES; do
        build_log="$LOG_DIR/build_${stage}_b${batch}_l${layer}.log"
        build_layer_qnn_library_into "$stage" "$batch" "$layer" "$combined_layer_out" 2>&1 | tee "$build_log"
      done
      build_log="$LOG_DIR/build_${stage}_combined_l${layer}.log"
      generate_combined_layer_qnn_binary "$stage" "$layer" "$combined_layer_out" 2>&1 | tee "$build_log"
    done
  done
  merge_combined_configs
else
  merge_configs
fi

if [[ "$BUILD_QNN" == "1" ]]; then
  copy_qnn_runtime_libs
fi

if [[ "$PRUNE_RUNTIME" == "1" ]]; then
  prune_runtime_package
fi

cat > "$OUTPUT_DIR/summary.md" <<EOF
# Qwen2.5-3B Stage QNN Export

- Date: $(date -Is)
- Model folder: \`$MODEL_FOLDER\`
- Model name: \`$MODEL_NAME\`
- Stages: \`$STAGES\`
- Batch sizes: \`$BATCH_SIZES\`
- Layers: \`$LAYERS\`
- QNN binaries built: \`$BUILD_QNN\`
- Combined batch binaries: \`$COMBINE_BATCH_BIN\`
- Runtime package pruned: \`$PRUNE_RUNTIME\`
- Config: \`$OUTPUT_DIR/config.json\`
- Logs: \`$LOG_DIR\`
EOF

echo "Export complete: $OUTPUT_DIR"
