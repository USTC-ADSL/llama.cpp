#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

cat > "${TMP_DIR}/bench.log" <<'EOF_LOG'
llama-bench: benchmark 1/1: round 1/1: starting
AOT_LOAD_TRACE kind=ensure_graph_loaded graph_name=attn_proj_layer_0_batch_16 model_path=attn_proj/attn_proj_layer_0/attn_proj_layer_0.bin batch_size=16 cache_size=1920 context_size=2048 graph_cache_hit=1 context_cache_hit=1 lock_wait_us=0 context_resolve_us=0 graph_create_us=0 seed_kv_us=0 total_us=0 success=1
FG_TRACE backend=qnn-npu layer=0 subgraph=qnn_proj begin_us=100 end_us=180 duration_us=80 tokens=16 step_tokens=16 offset=0 graph=attn_proj_layer_0 ok=1
[aot-assign] name=norm-1 reason=aot-unsupported backend=qnn-npu supported=0
FG_SYNC_TRACE from=qnn to=gpu us=11
AOT_LOAD_TRACE kind=ensure_graph_loaded graph_name=ffn_layer_0_batch_16 model_path=ffn/ffn_layer_0/ffn_layer_0.bin batch_size=16 cache_size=1920 context_size=2048 graph_cache_hit=1 context_cache_hit=1 lock_wait_us=0 context_resolve_us=0 graph_create_us=0 seed_kv_us=0 total_us=0 success=1
FG_TRACE backend=qnn-npu layer=0 subgraph=qnn_ffn begin_us=300 end_us=520 duration_us=220 tokens=16 step_tokens=16 offset=0 graph=ffn_layer_0 ok=1
FG_SYNC_TRACE from=gpu to=qnn us=13
AOT_LOAD_TRACE kind=ensure_graph_loaded graph_name=attn_proj_layer_1_batch_16 model_path=attn_proj/attn_proj_layer_1/attn_proj_layer_1.bin batch_size=16 cache_size=1920 context_size=2048 graph_cache_hit=1 context_cache_hit=1 lock_wait_us=0 context_resolve_us=0 graph_create_us=0 seed_kv_us=0 total_us=0 success=1
FG_TRACE backend=qnn-npu layer=1 subgraph=qnn_proj begin_us=600 end_us=680 duration_us=80 tokens=16 step_tokens=16 offset=0 graph=attn_proj_layer_1 ok=1
FG_SYNC_TRACE from=qnn to=gpu us=11
AOT_LOAD_TRACE kind=ensure_graph_loaded graph_name=ffn_layer_1_batch_16 model_path=ffn/ffn_layer_1/ffn_layer_1.bin batch_size=16 cache_size=1920 context_size=2048 graph_cache_hit=1 context_cache_hit=1 lock_wait_us=0 context_resolve_us=0 graph_create_us=0 seed_kv_us=0 total_us=0 success=1
FG_TRACE backend=qnn-npu layer=1 subgraph=qnn_ffn begin_us=800 end_us=1020 duration_us=220 tokens=16 step_tokens=16 offset=0 graph=ffn_layer_1 ok=1
FG_SYNC_TRACE from=gpu to=qnn us=13
OPENCL_KERNEL_TRACE total=7
llama-bench: benchmark 1/1: round 1/1: finished (1000.00 ms)
{"model_filename":"fixture.gguf","n_prompt":0,"n_gen":16,"n_depth":512,"test_time":"2026-05-24T00:00:00Z","avg_ns":1000000000,"stddev_ns":0,"avg_ts":20.0,"stddev_ts":0.0}
EOF_LOG

cat > "${TMP_DIR}/power_samples.csv" <<'EOF_SAMPLES'
timestamp_ms,voltage_uv,current_ua,power_mw,temp_raw,temp_c,gpu_busy,gpu_clock_hz,qnn_workpoint
1,4000000,-1000000,4000,300,30.0,40%,967000000,burst
2,4000000,-1200000,4800,310,31.0,60%,967000000,burst
EOF_SAMPLES

cat > "${TMP_DIR}/cl_stage_profiling.csv" <<'EOF_OPENCL'
stage,exec_total_ms,count,exec_avg_ms,exec_min_ms,exec_max_ms,exec_percentage,host_total_ms,host_avg_ms,host_percentage
ATTN_CORE,0.150,2,0.075,0.070,0.080,100.00,0.010,0.005,100.00
TOTAL,0.150,2,,,,100.00,0.010,,100.00
EOF_OPENCL

cat > "${TMP_DIR}/local_command.sh" <<'EOF_COMMAND'
DEVICE='fixture-device' MODEL_PATH='/data/local/tmp/model.gguf' bash scripts/run_fgsplit_synthetic.sh
EOF_COMMAND

cat > "${TMP_DIR}/remote_command.sh" <<'EOF_COMMAND'
taskset 80 /data/local/tmp/bin/llama-bench -m /data/local/tmp/model.gguf
EOF_COMMAND

python3 "${ROOT_DIR}/tools/parse_fgsplit_trace.py" \
    --bench-log "${TMP_DIR}/bench.log" \
    --sample-log "${TMP_DIR}/power_samples.csv" \
    --opencl-stage-profile "${TMP_DIR}/cl_stage_profiling.csv" \
    --command "${TMP_DIR}/remote_command.sh" \
    --local-command "${TMP_DIR}/local_command.sh" \
    --output-csv "${TMP_DIR}/out.csv" \
    --summary-md "${TMP_DIR}/summary.md" \
    --device fixture-device \
    --model-path /data/local/tmp/model.gguf \
    --git-commit abc123 \
    --output-dir "${TMP_DIR}" \
    --remote-output-dir /data/local/tmp/fgsplit_fixture \
    --mode synthetic \
    --backend-policy fine_grained_qnn_gpu \
    --state-id fg_qnn_burst_gpu_967 \
    --context-len 512 \
    --prompt-tokens 512 \
    --decode-tokens 16 \
    --layers 2 \
    --rounds 2 \
    --gpu-freq-mhz 967 \
    --qnn-workpoint burst \
    --temp-limit-c 38.0 \
    --cooldown-temp-c 37.0

python3 "${ROOT_DIR}/tools/parse_fgsplit_trace.py" \
    --bench-log "${TMP_DIR}/bench.log" \
    --sample-log "${TMP_DIR}/power_samples.csv" \
    --opencl-stage-profile "${TMP_DIR}/cl_stage_profiling.csv" \
    --output-csv "${TMP_DIR}/out_mismatch.csv" \
    --mode synthetic \
    --backend-policy fine_grained_qnn_gpu \
    --state-id fg_qnn_burst_gpu_967 \
    --context-len 512 \
    --prompt-tokens 512 \
    --decode-tokens 16 \
    --layers 3 \
    --rounds 2 \
    --gpu-freq-mhz 967 \
    --qnn-workpoint burst

cat > "${TMP_DIR}/gpu_only.log" <<'EOF_GPU'
llama_context: CPU fallback to GPUOpenCL host buffers is disabled by default because the current OpenCL host-buffer path can corrupt decode semantics.
OPENCL_KERNEL_TRACE total=4
OPENCL_KERNEL_TRACE stage=ATTN_CORE count=2
{"model_filename":"fixture.gguf","n_prompt":0,"n_gen":16,"n_depth":512,"test_time":"2026-05-24T00:00:00Z","avg_ns":1000000000,"stddev_ns":0,"avg_ts":30.0,"stddev_ts":0.0}
FG_RUN_EXIT_CODE=0
EOF_GPU

python3 "${ROOT_DIR}/tools/parse_fgsplit_trace.py" \
    --bench-log "${TMP_DIR}/gpu_only.log" \
    --sample-log "${TMP_DIR}/power_samples.csv" \
    --opencl-stage-profile "${TMP_DIR}/cl_stage_profiling.csv" \
    --output-csv "${TMP_DIR}/out_gpu.csv" \
    --mode synthetic \
    --backend-policy single_gpu_opencl \
    --state-id single_gpu_967 \
    --context-len 512 \
    --prompt-tokens 512 \
    --decode-tokens 16 \
    --layers 2 \
    --rounds 2 \
    --gpu-freq-mhz 967

cat > "${TMP_DIR}/qnn_only.log" <<'EOF_QNN'
FG_TRACE backend=qnn-npu layer=0 subgraph=qnn_proj begin_us=100 end_us=180 duration_us=80 tokens=16 step_tokens=16 offset=0 graph=attn_proj_layer_0 ok=1
failed to unregister shared buffer view, error=8001 (8001)
FG_TRACE backend=qnn-npu layer=0 subgraph=qnn_ffn begin_us=200 end_us=420 duration_us=220 tokens=16 step_tokens=16 offset=0 graph=ffn_layer_0 ok=1
{"model_filename":"fixture.gguf","n_prompt":0,"n_gen":16,"n_depth":512,"test_time":"2026-05-24T00:00:00Z","avg_ns":1000000000,"stddev_ns":0,"avg_ts":10.0,"stddev_ts":0.0}
FG_RUN_EXIT_CODE=0
EOF_QNN

python3 "${ROOT_DIR}/tools/parse_fgsplit_trace.py" \
    --bench-log "${TMP_DIR}/qnn_only.log" \
    --sample-log "${TMP_DIR}/power_samples.csv" \
    --output-csv "${TMP_DIR}/out_qnn.csv" \
    --mode synthetic \
    --backend-policy single_qnn_npu \
    --state-id single_qnn_burst \
    --context-len 512 \
    --prompt-tokens 512 \
    --decode-tokens 16 \
    --layers 2 \
    --rounds 2 \
    --qnn-workpoint burst

python3 - "${TMP_DIR}/out.csv" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="") as handle:
    row = next(csv.DictReader(handle))

expected = {
    "date": "2026-05-24T00:00:00Z",
    "model": "fixture.gguf",
    "mode": "synthetic",
    "backend_policy": "fine_grained_qnn_gpu",
    "state_id": "fg_qnn_burst_gpu_967",
    "context_len": "512",
    "prompt_tokens": "512",
    "decode_tokens": "16",
    "layers": "2",
    "rounds": "2",
    "semantic_correctness_status": "not_required",
    "throughput_tps": "20.000",
    "latency_per_token_ms": "50.000",
    "latency_per_layer_ms": "25.000",
    "active_power_mw": "4400.000",
    "energy_mj_per_token": "220.000",
    "temp_avg_c": "30.500",
    "temp_max_c": "31.000",
    "gpu_freq_mhz": "967",
    "qnn_workpoint": "burst",
    "gpu_active_ratio": "0.5000",
    "qnn_proj_us": "80.000",
    "gpu_attn_core_us": "75.000",
    "qnn_ffn_us": "220.000",
    "sync_qnn_to_gpu_us": "11.000",
    "sync_gpu_to_qnn_us": "13.000",
    "total_sync_us": "24.000",
    "fallback_used": "0",
    "support_status": "ok",
}

for key, value in expected.items():
    actual = row.get(key)
    if actual != value:
        raise SystemExit(f"{key}: expected {value!r}, got {actual!r}")
PY

python3 - "${TMP_DIR}/out_mismatch.csv" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="") as handle:
    row = next(csv.DictReader(handle))

if row["layers"] != "2":
    raise SystemExit(f"layers should be trace-observed value '2', got {row['layers']!r}")
if row["support_status"] != "unsupported_by_shape":
    raise SystemExit(f"layer mismatch should be unsupported_by_shape, got {row['support_status']!r}")
PY

python3 - "${TMP_DIR}/out_gpu.csv" "${TMP_DIR}/out_qnn.csv" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="") as handle:
    gpu = next(csv.DictReader(handle))
with open(sys.argv[2], newline="") as handle:
    qnn = next(csv.DictReader(handle))

if gpu["backend_policy"] != "single_gpu_opencl":
    raise SystemExit(f"gpu backend policy mismatch: {gpu['backend_policy']!r}")
if gpu["support_status"] != "ok":
    raise SystemExit(f"gpu baseline should be ok, got {gpu['support_status']!r}")
if gpu["fallback_used"] != "0":
    raise SystemExit(f"gpu baseline fallback should be 0, got {gpu['fallback_used']!r}")

if qnn["backend_policy"] != "single_qnn_npu":
    raise SystemExit(f"qnn backend policy mismatch: {qnn['backend_policy']!r}")
if qnn["support_status"] != "ok":
    raise SystemExit(f"qnn baseline should be ok, got {qnn['support_status']!r}")
if qnn["fallback_used"] != "0":
    raise SystemExit(f"qnn baseline fallback should be 0, got {qnn['fallback_used']!r}")
PY

cat > "${TMP_DIR}/missing_qnn_proj.log" <<'EOF_MISSING_PROJ'
FG_TRACE backend=qnn-npu layer=0 subgraph=qnn_ffn begin_us=300 end_us=520 duration_us=220 tokens=16 step_tokens=16 offset=0 graph=ffn_layer_0 ok=1
OPENCL_KERNEL_TRACE total=7
OPENCL_KERNEL_TRACE stage=ATTN_CORE count=2
{"model_filename":"fixture.gguf","n_prompt":0,"n_gen":16,"n_depth":512,"test_time":"2026-05-24T00:00:00Z","avg_ns":1000000000,"stddev_ns":0,"avg_ts":20.0,"stddev_ts":0.0}
FG_RUN_EXIT_CODE=0
EOF_MISSING_PROJ

cat > "${TMP_DIR}/missing_qnn_ffn.log" <<'EOF_MISSING_FFN'
FG_TRACE backend=qnn-npu layer=0 subgraph=qnn_proj begin_us=100 end_us=180 duration_us=80 tokens=16 step_tokens=16 offset=0 graph=attn_proj_layer_0 ok=1
OPENCL_KERNEL_TRACE total=7
OPENCL_KERNEL_TRACE stage=ATTN_CORE count=2
{"model_filename":"fixture.gguf","n_prompt":0,"n_gen":16,"n_depth":512,"test_time":"2026-05-24T00:00:00Z","avg_ns":1000000000,"stddev_ns":0,"avg_ts":20.0,"stddev_ts":0.0}
FG_RUN_EXIT_CODE=0
EOF_MISSING_FFN

cat > "${TMP_DIR}/missing_gpu_attn_core.log" <<'EOF_MISSING_ATTN'
FG_TRACE backend=qnn-npu layer=0 subgraph=qnn_proj begin_us=100 end_us=180 duration_us=80 tokens=16 step_tokens=16 offset=0 graph=attn_proj_layer_0 ok=1
FG_TRACE backend=qnn-npu layer=0 subgraph=qnn_ffn begin_us=300 end_us=520 duration_us=220 tokens=16 step_tokens=16 offset=0 graph=ffn_layer_0 ok=1
OPENCL_KERNEL_TRACE total=7
OPENCL_KERNEL_TRACE stage=FFN count=7
{"model_filename":"fixture.gguf","n_prompt":0,"n_gen":16,"n_depth":512,"test_time":"2026-05-24T00:00:00Z","avg_ns":1000000000,"stddev_ns":0,"avg_ts":20.0,"stddev_ts":0.0}
FG_RUN_EXIT_CODE=0
EOF_MISSING_ATTN

for name in missing_qnn_proj missing_qnn_ffn missing_gpu_attn_core; do
    python3 "${ROOT_DIR}/tools/parse_fgsplit_trace.py" \
        --bench-log "${TMP_DIR}/${name}.log" \
        --sample-log "${TMP_DIR}/power_samples.csv" \
        --output-csv "${TMP_DIR}/out_${name}.csv" \
        --mode synthetic \
        --backend-policy fine_grained_qnn_gpu \
        --state-id fg_qnn_burst_gpu_967 \
        --context-len 512 \
        --prompt-tokens 512 \
        --decode-tokens 16 \
        --layers 1 \
        --rounds 1 \
        --gpu-freq-mhz 967 \
        --qnn-workpoint burst
done

python3 - "${TMP_DIR}/out_missing_qnn_proj.csv" "${TMP_DIR}/out_missing_qnn_ffn.csv" "${TMP_DIR}/out_missing_gpu_attn_core.csv" <<'PY'
import csv
import sys

def read(path):
    with open(path, newline="") as handle:
        return next(csv.DictReader(handle))

missing_proj = read(sys.argv[1])
missing_ffn = read(sys.argv[2])
missing_attn = read(sys.argv[3])

if missing_proj["support_status"] != "unsupported_by_qnn_graph":
    raise SystemExit(f"missing qnn_proj should be unsupported_by_qnn_graph, got {missing_proj['support_status']!r}")
if missing_ffn["support_status"] != "unsupported_by_qnn_graph":
    raise SystemExit(f"missing qnn_ffn should be unsupported_by_qnn_graph, got {missing_ffn['support_status']!r}")
if missing_attn["support_status"] != "unsupported_by_gpu_kernel":
    raise SystemExit(f"missing GPU ATTN_CORE should be unsupported_by_gpu_kernel, got {missing_attn['support_status']!r}")
PY

cat > "${TMP_DIR}/measured_loading.log" <<'EOF_LOADING'
llama-bench: benchmark 1/1: round 1/1: starting
AOT_LOAD_TRACE kind=context model_path=/data/local/tmp/stage/ffn/ffn_layer_0.bin binary_bytes=123 mmap_us=1 context_create_from_binary_us=100 system_context_create_us=1 binary_info_us=1 total_us=103 success=1 reason=ok
AOT_LOAD_TRACE kind=ensure_graph_loaded graph_name=attn_proj_layer_0_batch_16 model_path=attn_proj/attn_proj_layer_0/attn_proj_layer_0.bin batch_size=16 cache_size=1920 context_size=2048 graph_cache_hit=1 context_cache_hit=1 lock_wait_us=0 context_resolve_us=0 graph_create_us=0 seed_kv_us=0 total_us=0 success=1
FG_TRACE backend=qnn-npu layer=0 subgraph=qnn_proj begin_us=100 end_us=180 duration_us=80 tokens=16 step_tokens=16 offset=0 graph=attn_proj_layer_0 ok=1
FG_TRACE backend=qnn-npu layer=0 subgraph=qnn_ffn begin_us=300 end_us=520 duration_us=220 tokens=16 step_tokens=16 offset=0 graph=ffn_layer_0 ok=1
OPENCL_KERNEL_TRACE total=7
OPENCL_KERNEL_TRACE stage=ATTN_CORE count=2
llama-bench: benchmark 1/1: round 1/1: finished (1000.00 ms)
{"model_filename":"fixture.gguf","n_prompt":0,"n_gen":16,"n_depth":512,"test_time":"2026-05-24T00:00:00Z","avg_ns":1000000000,"stddev_ns":0,"avg_ts":20.0,"stddev_ts":0.0}
FG_RUN_EXIT_CODE=0
EOF_LOADING

cat > "${TMP_DIR}/measured_cache_miss.log" <<'EOF_CACHE'
llama-bench: benchmark 1/1: round 1/1: starting
AOT_LOAD_TRACE kind=ensure_graph_loaded graph_name=attn_proj_layer_0_batch_16 model_path=attn_proj/attn_proj_layer_0/attn_proj_layer_0.bin batch_size=16 cache_size=1920 context_size=2048 graph_cache_hit=0 context_cache_hit=1 lock_wait_us=0 context_resolve_us=0 graph_create_us=123 seed_kv_us=0 total_us=123 success=1
FG_TRACE backend=qnn-npu layer=0 subgraph=qnn_proj begin_us=100 end_us=180 duration_us=80 tokens=16 step_tokens=16 offset=0 graph=attn_proj_layer_0 ok=1
FG_TRACE backend=qnn-npu layer=0 subgraph=qnn_ffn begin_us=300 end_us=520 duration_us=220 tokens=16 step_tokens=16 offset=0 graph=ffn_layer_0 ok=1
OPENCL_KERNEL_TRACE total=7
OPENCL_KERNEL_TRACE stage=ATTN_CORE count=2
llama-bench: benchmark 1/1: round 1/1: finished (1000.00 ms)
{"model_filename":"fixture.gguf","n_prompt":0,"n_gen":16,"n_depth":512,"test_time":"2026-05-24T00:00:00Z","avg_ns":1000000000,"stddev_ns":0,"avg_ts":20.0,"stddev_ts":0.0}
FG_RUN_EXIT_CODE=0
EOF_CACHE

cat > "${TMP_DIR}/measured_weight_sync.log" <<'EOF_WEIGHT'
llama-bench: benchmark 1/1: round 1/1: starting
AOT_LOAD_TRACE kind=ensure_graph_loaded graph_name=attn_proj_layer_0_batch_16 model_path=attn_proj/attn_proj_layer_0/attn_proj_layer_0.bin batch_size=16 cache_size=1920 context_size=2048 graph_cache_hit=1 context_cache_hit=1 lock_wait_us=0 context_resolve_us=0 graph_create_us=0 seed_kv_us=0 total_us=0 success=1
FG_TRACE backend=qnn-npu layer=0 subgraph=qnn_proj begin_us=100 end_us=180 duration_us=80 tokens=16 step_tokens=16 offset=0 graph=attn_proj_layer_0 ok=1
FG_SYNC_TRACE from=qnn to=gpu us=17 wait_us=0 bytes=7340032 tensor=blk.2.attn_q.weight split=1 src_backend=qnn-npu dst_backend=OpenCL
FG_TRACE backend=qnn-npu layer=0 subgraph=qnn_ffn begin_us=300 end_us=520 duration_us=220 tokens=16 step_tokens=16 offset=0 graph=ffn_layer_0 ok=1
OPENCL_KERNEL_TRACE total=7
OPENCL_KERNEL_TRACE stage=ATTN_CORE count=2
llama-bench: benchmark 1/1: round 1/1: finished (1000.00 ms)
{"model_filename":"fixture.gguf","n_prompt":0,"n_gen":16,"n_depth":512,"test_time":"2026-05-24T00:00:00Z","avg_ns":1000000000,"stddev_ns":0,"avg_ts":20.0,"stddev_ts":0.0}
FG_RUN_EXIT_CODE=0
EOF_WEIGHT

for name in measured_loading measured_cache_miss measured_weight_sync; do
    python3 "${ROOT_DIR}/tools/parse_fgsplit_trace.py" \
        --bench-log "${TMP_DIR}/${name}.log" \
        --sample-log "${TMP_DIR}/power_samples.csv" \
        --output-csv "${TMP_DIR}/out_${name}.csv" \
        --summary-md "${TMP_DIR}/summary_${name}.md" \
        --mode synthetic \
        --backend-policy fine_grained_qnn_gpu \
        --state-id fg_qnn_burst_gpu_967 \
        --context-len 512 \
        --prompt-tokens 512 \
        --decode-tokens 16 \
        --layers 1 \
        --rounds 1 \
        --gpu-freq-mhz 967 \
        --qnn-workpoint burst
done

python3 - "${TMP_DIR}/out_measured_loading.csv" "${TMP_DIR}/out_measured_cache_miss.csv" "${TMP_DIR}/out_measured_weight_sync.csv" <<'PY'
import csv
import sys

def read(path):
    with open(path, newline="") as handle:
        return next(csv.DictReader(handle))

loading = read(sys.argv[1])
cache_miss = read(sys.argv[2])
weight_sync = read(sys.argv[3])

if loading["support_status"] != "failed_measured_loading":
    raise SystemExit(f"measured context load should fail preload gate, got {loading['support_status']!r}")
if cache_miss["support_status"] != "failed_measured_graph_cache":
    raise SystemExit(f"measured graph cache miss should fail cache gate, got {cache_miss['support_status']!r}")
if weight_sync["support_status"] != "failed_measured_weight_sync":
    raise SystemExit(f"measured stage weight sync should fail weight gate, got {weight_sync['support_status']!r}")
PY

grep -F -- "- measured_quality_status: \`failed_measured_weight_sync\`" "${TMP_DIR}/summary_measured_weight_sync.md" >/dev/null
grep -F -- "- measured_weight_syncs: \`1\`" "${TMP_DIR}/summary_measured_weight_sync.md" >/dev/null
grep -F -- "- measured_weight_sync_bytes: \`7340032\`" "${TMP_DIR}/summary_measured_weight_sync.md" >/dev/null

grep -F "## Git Commit" "${TMP_DIR}/summary.md" >/dev/null
grep -F "## Device and Model" "${TMP_DIR}/summary.md" >/dev/null
grep -F "## Temperature Range" "${TMP_DIR}/summary.md" >/dev/null
grep -F -- "- requested_layers: \`2\`" "${TMP_DIR}/summary.md" >/dev/null
grep -F -- "- observed_fg_layers: \`2\`" "${TMP_DIR}/summary.md" >/dev/null
grep -F "## Power Comparison Against Single GPU and Single QNN" "${TMP_DIR}/summary.md" >/dev/null
grep -F "## Throughput Comparison Against Single GPU and Single QNN" "${TMP_DIR}/summary.md" >/dev/null
grep -F "## Energy Per Token Comparison" "${TMP_DIR}/summary.md" >/dev/null
grep -F "## Synchronization Overhead" "${TMP_DIR}/summary.md" >/dev/null
grep -F "## Fallback or Unsupported Conditions" "${TMP_DIR}/summary.md" >/dev/null
grep -F "## Measured Execution Quality" "${TMP_DIR}/summary.md" >/dev/null
grep -F -- "- measured_quality_status: \`ok\`" "${TMP_DIR}/summary.md" >/dev/null
grep -F -- "- measured_graph_cache_hits: \`4\`" "${TMP_DIR}/summary.md" >/dev/null
grep -F -- "- measured_weight_syncs: \`0\`" "${TMP_DIR}/summary.md" >/dev/null
grep -F "## Data Quality Judgment" "${TMP_DIR}/summary.md" >/dev/null
grep -F "## Whether The Result Supports The Insight" "${TMP_DIR}/summary.md" >/dev/null

grep -F 'QNN_AOT_EVICT_STATELESS_STAGE_GRAPHS="${QNN_AOT_EVICT_STATELESS_STAGE_GRAPHS:-0}"' \
    "${ROOT_DIR}/scripts/run_fgsplit_synthetic.sh" >/dev/null
grep -F "export GGML_QNN_AOT_EVICT_STATELESS_STAGE_GRAPHS=" \
    "${ROOT_DIR}/scripts/run_fgsplit_synthetic.sh" >/dev/null
grep -F 'QNN_AOT_TRACE_LOAD_TIMING="${QNN_AOT_TRACE_LOAD_TIMING:-1}"' \
    "${ROOT_DIR}/scripts/run_fgsplit_synthetic.sh" >/dev/null
grep -F "export GGML_QNN_AOT_TRACE_LOAD_TIMING=" \
    "${ROOT_DIR}/scripts/run_fgsplit_synthetic.sh" >/dev/null
grep -F "QNN_AOT_EVICT_STATELESS_STAGE_GRAPHS=%s" \
    "${ROOT_DIR}/scripts/run_fgsplit_synthetic.sh" >/dev/null
grep -F "LLAMA_BENCH_WARMUP=%s" \
    "${ROOT_DIR}/scripts/run_fgsplit_synthetic.sh" >/dev/null
grep -F "LLAMA_BENCH_QNN_PREWARM_DECODE=%s" \
    "${ROOT_DIR}/scripts/run_fgsplit_synthetic.sh" >/dev/null
grep -F "FGSPLIT_REQUIRE_SUPPORT_OK=1" \
    "${ROOT_DIR}/scripts/run_fgsplit_synthetic.sh" >/dev/null
grep -F 'OPENCL_SIM_BUSY="${OPENCL_SIM_BUSY:-0}"' \
    "${ROOT_DIR}/scripts/run_fgsplit_synthetic.sh" >/dev/null
grep -F "export GGML_OPENCL_SIM_BUSY=1" \
    "${ROOT_DIR}/scripts/run_fgsplit_synthetic.sh" >/dev/null
grep -F "export GGML_OPENCL_SIM_BUSY_GLOBAL=" \
    "${ROOT_DIR}/scripts/run_fgsplit_synthetic.sh" >/dev/null
grep -F "unset GGML_OPENCL_SIM_BUSY_ENABLE" \
    "${ROOT_DIR}/scripts/run_fgsplit_synthetic.sh" >/dev/null
grep -F "OPENCL_SIM_BUSY_ITERS=%s" \
    "${ROOT_DIR}/scripts/run_fgsplit_synthetic.sh" >/dev/null
grep -F "read_profile_status()" \
    "${ROOT_DIR}/scripts/run_fgsplit_synthetic.sh" >/dev/null
grep -F "support gate failed: support_status=" \
    "${ROOT_DIR}/scripts/run_fgsplit_synthetic.sh" >/dev/null
grep -F 'FG_MAX_LAYERS="${FG_MAX_LAYERS:-${LAYERS}}"' \
    "${ROOT_DIR}/scripts/run_fgsplit_synthetic.sh" >/dev/null
grep -F "export GGML_HETERO_FG_MAX_LAYERS=" \
    "${ROOT_DIR}/scripts/run_fgsplit_synthetic.sh" >/dev/null
grep -F "export GGML_HETERO_FG_SYNC_TRACE=1" \
    "${ROOT_DIR}/scripts/run_fgsplit_synthetic.sh" >/dev/null
grep -F "FG_MAX_LAYERS=%s" \
    "${ROOT_DIR}/scripts/run_fgsplit_synthetic.sh" >/dev/null
grep -F "cp -f cl_profiling.csv cl_stage_profiling.csv cl_trace.json" \
    "${ROOT_DIR}/scripts/run_fgsplit_synthetic.sh" >/dev/null
grep -F "rm -f cl_profiling.csv cl_stage_profiling.csv cl_trace.json" \
    "${ROOT_DIR}/scripts/run_fgsplit_synthetic.sh" >/dev/null
grep -F 'BACKEND_POLICY="${BACKEND_POLICY:-fine_grained_qnn_gpu}"' \
    "${ROOT_DIR}/scripts/run_fgsplit_synthetic.sh" >/dev/null
grep -F 'DEFAULT_FG_ROUTE="attn_proj=qnn-npu,attn_core=opencl,attn_out=cpu,ffn=qnn-npu,output=cpu"' \
    "${ROOT_DIR}/scripts/run_fgsplit_synthetic.sh" >/dev/null
grep -F "export LLAMA_BENCH_QNN_PREWARM_DECODE=" \
    "${ROOT_DIR}/scripts/run_fgsplit_synthetic.sh" >/dev/null
grep -F "single_gpu_opencl)" \
    "${ROOT_DIR}/scripts/run_fgsplit_synthetic.sh" >/dev/null
grep -F "single_qnn_npu)" \
    "${ROOT_DIR}/scripts/run_fgsplit_synthetic.sh" >/dev/null
