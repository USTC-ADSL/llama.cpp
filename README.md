# stage-sim-qnn-opencl

这个分支用于在 Android Snapdragon 设备上做 AF split simulation：

- Attention 相关阶段运行在 `GPUOpenCL`
- FFN 运行在 `QNN` / HTP NPU
- 不要求语义正确
- 要求算子分配正确
- 要求 Attention 输出 hidden state 作为 QNN FFN 输入
- 要求能够稳定 decode 至少 128 tokens, `r=5`
- 目标吞吐不低于 `12 t/s`

本分支基于 llama.cpp，主要面向系统实验和后端调度验证，不是通用上游 README。

## 当前验证结果

已在设备 `fd8657d6` 上验证：

```text
route        = attn_proj=opencl,attn_core=opencl,attn_out=opencl,ffn=qnn-npu,output=opencl
layers       = 28
decode       = 128 tokens
rounds       = 5
support      = ok
fallback     = 0
throughput   = 16.885 t/s
target       = >= 12 t/s
```

最终验证产物在本地 worktree 中：

```text
results/fgsplit/af-final-fg28-d128-r5-depthprewarm/fgsplit_power_profile.csv
results/fgsplit/af-final-fg28-d128-r5-depthprewarm/raw/bench.log
docs/实验结果/FGSplit_af_final_fg28_d128_r5_depthprewarm.md
```

注意：`results/` 和 `docs/实验结果/FGSplit_af_*.md` 是本地运行证据，默认不随代码提交。

## 分支核心改动

### 1. 默认 AF route

`scripts/run_fgsplit_synthetic.sh` 中 `fine_grained_qnn_gpu` 默认 route 是：

```text
attn_proj=opencl,attn_core=opencl,attn_out=opencl,ffn=qnn-npu,output=opencl
```

含义：

- `attn_proj=opencl`: attention projection 走 OpenCL
- `attn_core=opencl`: attention core 走 OpenCL
- `attn_out=opencl`: attention output / hidden state 留在 OpenCL
- `ffn=qnn-npu`: FFN AoT graph 走 QNN NPU
- `output=opencl`: output tail 走 OpenCL

### 2. Hidden state handoff trace

QNN FFN 执行时会记录 `ffn_inp-*` 的 handoff：

```text
FG_SYNC_TRACE from=gpu to=qnn ... tensor=ffn_inp-0 ... reason=ffn_input_materialize
```

或：

```text
FG_SYNC_TRACE from=gpu to=qnn ... tensor=ffn_inp-0 ... reason=ffn_input_direct_bind
```

parser 会要求这个证据存在。否则 AF route 会被标记为：

```text
support_status=missing_hidden_handoff
```

这避免只看到 OpenCL kernel 和 QNN FFN trace，却无法证明 Attention hidden state 真的交给 FFN。

### 3. QNN depth prewarm

`llama-bench` 增加了：

```text
LLAMA_BENCH_QNN_PREWARM_DEPTH=1
```

用途是在 measured round 之前预热 context/depth 路径会用到的 QNN graph，避免 measured window 内出现 first-use graph loading。

最终报告中的 measured-quality gate 应该是：

```text
measured_quality_status=ok
measured_context_loads=0
measured_graph_loads=0
measured_graph_cache_misses=0
measured_bad_lines=0
```

## 前置条件

### 本地环境

需要：

- Android NDK
- QNN / QAIRT SDK
- `adb`
- 一台可用 Android Snapdragon 设备
- 本地 GGUF 模型
- 已导出的 QNN FFN AoT graph bundle

本分支验证时使用：

```text
QNN SDK: /mnt/sda1/pzw/HeteroCompute/qairt/2.31.0.250130
model:   /mnt/sda1/pzw/HeteroCompute/llama.cpp-acom/models/Qwen3-1.7B/Qwen3-1.7B-Q4_0/qwen3-1.7b-q4_0.gguf
AoT cfg: /mnt/sda1/pzw/HeteroCompute/llama.cpp-acom/models/Qwen3-1.7B/Qwen3-AoT/qwen3-qnn-full/qnn_ffn_combined.json
AoT bin: /mnt/sda1/pzw/HeteroCompute/llama.cpp-acom/models/Qwen3-1.7B/Qwen3-AoT/qwen3-qnn-ffn
```

### 设备路径

示例设备路径：

```text
REMOTE_BIN=/data/local/tmp/acom-af-qnn-opencl-stage-sim
REMOTE_AOT=/data/local/tmp/acom-stage-models/Qwen3-AoT
REMOTE_MODEL=/data/local/tmp/llama-acom-qnn244/qwen3-1.7b-q4_0.gguf
```

## 构建 Android QNN + OpenCL runtime

在仓库根目录执行：

```bash
QNN_SDK_PATH=/mnt/sda1/pzw/HeteroCompute/qairt/2.31.0.250130 \
./build-npu-opencl.sh build-fgsplit-af arm64-android-snapdragon-release \
  --without-npu --with-gpu --with-qnn \
  --qnn-sdk /mnt/sda1/pzw/HeteroCompute/qairt/2.31.0.250130
```

成功后重点检查：

```text
build-fgsplit-af/bin/llama-bench
build-fgsplit-af/bin/libggml-opencl.so
build-fgsplit-af/bin/libggml-qnn.so
build-fgsplit-af/bin/libllama.so
build-fgsplit-af/bin/libQnn*.so
build-fgsplit-af/bin/libomp.so
```

## 部署 runtime 到设备

```bash
DEVICE=fd8657d6
REMOTE_BIN=/data/local/tmp/acom-af-qnn-opencl-stage-sim

adb -s "$DEVICE" shell "mkdir -p '$REMOTE_BIN'"

for f in \
  build-fgsplit-af/bin/llama-bench \
  build-fgsplit-af/bin/libggml*.so \
  build-fgsplit-af/bin/libllama.so \
  build-fgsplit-af/bin/libQnn*.so \
  build-fgsplit-af/bin/libomp.so; do
  adb -s "$DEVICE" push $f "$REMOTE_BIN/"
done

adb -s "$DEVICE" shell "chmod +x '$REMOTE_BIN/llama-bench'"
```

## 部署模型

如果设备上还没有模型：

```bash
DEVICE=fd8657d6
LOCAL_MODEL=/mnt/sda1/pzw/HeteroCompute/llama.cpp-acom/models/Qwen3-1.7B/Qwen3-1.7B-Q4_0/qwen3-1.7b-q4_0.gguf
REMOTE_MODEL=/data/local/tmp/llama-acom-qnn244/qwen3-1.7b-q4_0.gguf

adb -s "$DEVICE" shell "mkdir -p /data/local/tmp/llama-acom-qnn244"
adb -s "$DEVICE" push "$LOCAL_MODEL" "$REMOTE_MODEL"
```

## 部署 QNN FFN AoT graph

`qnn_ffn_combined.json` 中的 `model_path` 指向：

```text
qwen3-qnn-ffn/batch-1/ffn_layer_*/ffn_layer_*.bin
qwen3-qnn-ffn/batch-128/ffn_layer_*/ffn_layer_*.bin
```

不需要把整个导出目录都推到设备。只需要推送 JSON 和它引用的 `.bin` 文件。

```bash
DEVICE=fd8657d6
LOCAL_AOT=/mnt/sda1/pzw/HeteroCompute/llama.cpp-acom/models/Qwen3-1.7B/Qwen3-AoT
REMOTE_AOT=/data/local/tmp/acom-stage-models/Qwen3-AoT
CONFIG_SRC="$LOCAL_AOT/qwen3-qnn-full/qnn_ffn_combined.json"

adb -s "$DEVICE" shell "rm -rf '$REMOTE_AOT' && mkdir -p '$REMOTE_AOT/qwen3-qnn-full' '$REMOTE_AOT/qwen3-qnn-ffn'" < /dev/null
adb -s "$DEVICE" push "$CONFIG_SRC" "$REMOTE_AOT/qwen3-qnn-full/" < /dev/null

while IFS= read -r rel; do
  src="$LOCAL_AOT/$rel"
  remote_dir="$REMOTE_AOT/$(dirname "$rel")"
  adb -s "$DEVICE" shell "mkdir -p '$remote_dir'" < /dev/null >/dev/null
  adb -s "$DEVICE" push "$src" "$remote_dir/" < /dev/null
done < <(python3 - <<'PY'
import json
p = "/mnt/sda1/pzw/HeteroCompute/llama.cpp-acom/models/Qwen3-1.7B/Qwen3-AoT/qwen3-qnn-full/qnn_ffn_combined.json"
d = json.load(open(p))
for rel in sorted({g["model_path"] for g in d.get("graphs", [])}):
    print(rel)
PY
)

adb -s "$DEVICE" shell "find '$REMOTE_AOT/qwen3-qnn-ffn' -type f -name '*.bin' | wc -l"
```

期望 `.bin` 数量是：

```text
56
```

## 运行 1 层 smoke

用于快速确认 route、OpenCL attention、QNN FFN 和 hidden-state handoff。

```bash
DEVICE=fd8657d6 \
MODEL_PATH=/data/local/tmp/llama-acom-qnn244/qwen3-1.7b-q4_0.gguf \
REMOTE_BIN=/data/local/tmp/acom-af-qnn-opencl-stage-sim \
QNN_AOT_CONFIG=/data/local/tmp/acom-stage-models/Qwen3-AoT/qwen3-qnn-full/qnn_ffn_combined.json \
QNN_AOT_MODEL_DIR=/data/local/tmp/acom-stage-models/Qwen3-AoT \
BACKEND_POLICY=fine_grained_qnn_gpu \
FG_ROUTE='attn_proj=opencl,attn_core=opencl,attn_out=opencl,ffn=qnn-npu,output=opencl' \
FG_MAX_LAYERS=1 \
LAYERS=1 \
CONTEXT_LEN=512 \
PROMPT_TOKENS=512 \
BENCH_PROMPT_TOKENS=0 \
DECODE_TOKENS=16 \
ROUNDS=1 \
TEMP_LIMIT_C=99 \
COOLDOWN_TEMP_C=98 \
OUTPUT_DIR=results/fgsplit/af-smoke-fg1-d16-r1-depthprewarm \
SUMMARY_MD=docs/实验结果/FGSplit_af_smoke_fg1_d16_r1_depthprewarm.md \
RESULTS_CSV=results/fgsplit/fgsplit_power_profile_af_validation.csv \
bash scripts/run_fgsplit_synthetic.sh
```

通过条件：

```text
support_status=ok
fallback_used=0
observed_fg_layers=1
FG_SYNC_TRACE from=gpu to=qnn ... tensor=ffn_inp-0 ...
OPENCL_KERNEL_TRACE stage=ATTN_CORE ...
```

## 运行 28 层 smoke

用于确认 28 层 FFN AoT graph 全部可运行。

```bash
DEVICE=fd8657d6 \
MODEL_PATH=/data/local/tmp/llama-acom-qnn244/qwen3-1.7b-q4_0.gguf \
REMOTE_BIN=/data/local/tmp/acom-af-qnn-opencl-stage-sim \
QNN_AOT_CONFIG=/data/local/tmp/acom-stage-models/Qwen3-AoT/qwen3-qnn-full/qnn_ffn_combined.json \
QNN_AOT_MODEL_DIR=/data/local/tmp/acom-stage-models/Qwen3-AoT \
BACKEND_POLICY=fine_grained_qnn_gpu \
FG_ROUTE='attn_proj=opencl,attn_core=opencl,attn_out=opencl,ffn=qnn-npu,output=opencl' \
FG_MAX_LAYERS=28 \
LAYERS=28 \
CONTEXT_LEN=512 \
PROMPT_TOKENS=512 \
BENCH_PROMPT_TOKENS=0 \
DECODE_TOKENS=16 \
ROUNDS=1 \
TEMP_LIMIT_C=99 \
COOLDOWN_TEMP_C=98 \
OUTPUT_DIR=results/fgsplit/af-full-fg28-d16-r1-depthprewarm \
SUMMARY_MD=docs/实验结果/FGSplit_af_full_fg28_d16_r1_depthprewarm.md \
RESULTS_CSV=results/fgsplit/fgsplit_power_profile_af_validation.csv \
bash scripts/run_fgsplit_synthetic.sh
```

## 运行最终验证

这是本分支的主要验收命令：

```bash
DEVICE=fd8657d6 \
MODEL_PATH=/data/local/tmp/llama-acom-qnn244/qwen3-1.7b-q4_0.gguf \
REMOTE_BIN=/data/local/tmp/acom-af-qnn-opencl-stage-sim \
QNN_AOT_CONFIG=/data/local/tmp/acom-stage-models/Qwen3-AoT/qwen3-qnn-full/qnn_ffn_combined.json \
QNN_AOT_MODEL_DIR=/data/local/tmp/acom-stage-models/Qwen3-AoT \
BACKEND_POLICY=fine_grained_qnn_gpu \
FG_ROUTE='attn_proj=opencl,attn_core=opencl,attn_out=opencl,ffn=qnn-npu,output=opencl' \
FG_MAX_LAYERS=28 \
LAYERS=28 \
CONTEXT_LEN=512 \
PROMPT_TOKENS=512 \
BENCH_PROMPT_TOKENS=0 \
DECODE_TOKENS=128 \
ROUNDS=5 \
TEMP_LIMIT_C=99 \
COOLDOWN_TEMP_C=98 \
OUTPUT_DIR=results/fgsplit/af-final-fg28-d128-r5-depthprewarm \
SUMMARY_MD=docs/实验结果/FGSplit_af_final_fg28_d128_r5_depthprewarm.md \
RESULTS_CSV=results/fgsplit/fgsplit_power_profile_af_validation.csv \
bash scripts/run_fgsplit_synthetic.sh
```

期望结果：

```text
support_status=ok
fallback_used=0
decode_tokens=128
rounds=5
observed_fg_layers=28
throughput_tps >= 12
```

本分支已记录的一次结果：

```text
throughput_tps=16.885
latency_per_token_ms=59.225
support_status=ok
fallback_used=0
```

## 查看结果

每次运行会生成：

```text
<OUTPUT_DIR>/fgsplit_power_profile.csv
<OUTPUT_DIR>/summary.md
<OUTPUT_DIR>/raw/bench.log
<OUTPUT_DIR>/raw/power_samples.csv
<OUTPUT_DIR>/remote/opencl_kernel_trace.csv
```

聚合 CSV：

```text
results/fgsplit/fgsplit_power_profile_af_validation.csv
```

报告：

```text
docs/实验结果/FGSplit_*.md
```

## 快速检查最终 raw log

```bash
python3 - <<'PY'
import csv
import re
from pathlib import Path

csv_path = Path("results/fgsplit/af-final-fg28-d128-r5-depthprewarm/fgsplit_power_profile.csv")
log_path = Path("results/fgsplit/af-final-fg28-d128-r5-depthprewarm/raw/bench.log")

row = next(csv.DictReader(csv_path.open(newline="")))
text = log_path.read_text(errors="ignore")

print("support_status", row["support_status"])
print("fallback_used", row["fallback_used"])
print("throughput_tps", row["throughput_tps"])
print("round_starts", len(re.findall(r"round \d+/5: starting", text)))
print("round_finishes", len(re.findall(r"round \d+/5: finished", text)))
print("qnn_ffn_events", len(re.findall(r"FG_TRACE backend=qnn-npu .*subgraph=qnn_ffn", text)))
print("handoff_events", len(re.findall(r"FG_SYNC_TRACE from=gpu to=qnn .*tensor=ffn_inp-", text)))
print("attn_core", re.findall(r"OPENCL_KERNEL_TRACE stage=ATTN_CORE count=([0-9]+)", text)[-1:])
print("exit_codes", re.findall(r"FG_RUN_EXIT_CODE=([0-9]+)", text))
PY
```

## Host regression tests

修改 parser 或 route 后至少运行：

```bash
bash tests/test-fgsplit-parser.sh
```

```bash
cmake --build build-x64-linux-gcc-release \
  --target test-llama-bench-utils test-hetero-stage-route \
           test-context-qnn-phase-migration test-opencl-cpu-extra-copy \
           test-qnn-aot-support-policy \
  -j "$(nproc)"
```

```bash
./build-x64-linux-gcc-release/bin/test-llama-bench-utils
./build-x64-linux-gcc-release/bin/test-hetero-stage-route
./build-x64-linux-gcc-release/bin/test-context-qnn-phase-migration
./build-x64-linux-gcc-release/bin/test-opencl-cpu-extra-copy
./build-x64-linux-gcc-release/bin/test-qnn-aot-support-policy
```

## Android build verification

修改 `llama-bench`、QNN backend 或 OpenCL/QNN runtime 后运行：

```bash
QNN_SDK_PATH=/mnt/sda1/pzw/HeteroCompute/qairt/2.31.0.250130 \
./build-npu-opencl.sh build-fgsplit-af arm64-android-snapdragon-release \
  --without-npu --with-gpu --with-qnn \
  --qnn-sdk /mnt/sda1/pzw/HeteroCompute/qairt/2.31.0.250130
```

然后重新部署 `build-fgsplit-af/bin/` 中的 runtime 文件到设备。

## 常见问题

### support_status=missing_hidden_handoff

说明 parser 没看到 `ffn_inp-*` 从 GPU/CPU 到 QNN 的 handoff。

检查 raw log：

```bash
rg "FG_SYNC_TRACE .*tensor=ffn_inp-" results/fgsplit/<run>/raw/bench.log
```

如果没有，需要检查 FFN 输入 tensor 是否真的来自 OpenCL，以及 `GGML_QNN_AOT_FG_TRACE=1` 是否生效。

### support_status=failed_measured_loading

说明 measured round 内出现 QNN context/graph loading。

检查是否启用：

```text
LLAMA_BENCH_QNN_PREWARM_DECODE=1
LLAMA_BENCH_QNN_PREWARM_DEPTH=1
```

### support_status=unsupported_by_shape

常见原因是 `FG_MAX_LAYERS` 和 `LAYERS` 不一致。

例如 1 层 smoke 应使用：

```text
FG_MAX_LAYERS=1
LAYERS=1
```

28 层验证应使用：

```text
FG_MAX_LAYERS=28
LAYERS=28
```

### fallback_used=1

说明 runtime 走了 fallback 路径。检查 raw log 中的 `fallback`、`failed to run`、`unmatched cgraph` 等关键字。

```bash
rg -i "fallback|failed to run|unmatched|error:" results/fgsplit/<run>/raw/bench.log
```

## 注意事项

- 本分支不验证语义正确性。
- 本分支不保证 p95 或 tail latency。
- 当前目标是模拟 AF split 的 backend assignment、hidden-state exchange 和吞吐。
- `power_samples.csv` 来自当前脚本的 battery sampler，功耗值只适合作为实验参考，不应直接作为论文级能耗结论。
- 若更换模型或 AoT graph，需要确认 `qnn_ffn_combined.json` 中的 graph name、layer 数、batch size 和 FFN input/output shape 与 runtime 匹配。

## 提交策略

建议提交：

- 源码
- 脚本
- parser
- host tests
- 小型 README / 文档

不建议提交：

- `results/`
- 大型 raw log
- 设备 power sample
- AoT `.bin`
- GGUF 模型
