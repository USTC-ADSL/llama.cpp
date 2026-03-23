# 2026-03-23-025 `db6c02cf` Qwen2/Qwen3 Device Validation and QNN AoT Revalidation

## 任务

在主设备 `db6c02cf` 上完成三件事：

1. 验证 `Qwen2` decode mixed 路线中最后一层 `attn tail` 的 CPU fallback 是否真正消除了 residual unmatched fragment。
2. 重新验证 `Qwen2` full-graph QNN AoT 的静态性能，排除“文件未完整部署”导致的伪慢结论。
3. 重新验证 `Qwen3` full-graph QNN AoT 的 `pp128` 静态性能，排除“只用了 batch1 AoT 图”导致的实验类别错误。

这三项工作都服务于当前 decode-first 主线：

- `Qwen2` mixed decode 修复用于收敛阶段边界 correctness；
- `Qwen2/Qwen3` full-graph QNN AoT 复核用于澄清静态基线口径，避免把配置错误误写成后端能力结论。

## 设备与构建

- 设备：`db6c02cf`
- 运行目录：`/data/local/tmp/acom-stage-matrix-verify`
- 亮屏检查：`mWakefulness=Awake`
- 按规范先重新执行：
  - `./build-npu-opencl.sh build-qnn-current-verify arm64-android-snapdragon-release --without-npu --with-gpu --with-qnn`
- 构建完成后重新同步了：
  - `llama-bench`
  - `libllama.so`
  - `libggml*.so`
  - `libQnn*.so`
  - `libomp.so`
- `--list-devices` 复核通过：
  - `GPUOpenCL`
  - `qnn-npu`
  - `qnn-gpu`
  - `qnn-cpu`

## 一、`Qwen2` Mixed Decode Tail Fallback 真机验证

### 配置

- 模型：`/data/local/tmp/acom-attn-proj-verify/models/qwen2_0.5b/ggml/weights.gguf`
- AoT 配置：
  - `/data/local/tmp/acom-attn-proj-verify/models/qwen2_0.5b/qnn_attn_core_combined.json`
- 路线：
  - `attn_proj=cpu,attn_core=qnn-npu,attn_out=qnn-npu,ffn=cpu,output=cpu`
- bench 参数：
  - `-r 1 -t 1 -p 0 -n 1 -c 2048 -b 1 -ub 1`
  - `-ctk f32 -ctv f32`
  - `-ngl 0 -dev GPUOpenCL --mmap 0 --no-warmup`
- 关键环境变量：
  - `GGML_HEXAGON_EXPERIMENTAL=1`
  - `GGML_HETERO_QNN_SHARED_HOST=1`
  - `GGML_HETERO_KV_LAYOUT=qnn`
  - `GGML_QNN_AOT_CONFIG=/data/local/tmp/acom-attn-proj-verify/models/qwen2_0.5b/qnn_attn_core_combined.json`
  - `GGML_QNN_AOT_MODEL_DIR=/data/local/tmp/acom-attn-proj-verify/models/qwen2_0.5b`
  - `GGML_QNN_AOT_TRACE_ASSIGN=1`
  - `GGML_HETERO_PROFILE=1`
  - `GGML_HETERO_PROFILE_SYNC=1`
  - `GGML_HETERO_PROFILE_FLUSH=1`
  - `GGML_HETERO_PROFILE_LOG=1`
  - `GGML_HETERO_PROFILE_CSV=/data/local/tmp/acom-stage-matrix-verify/p18-qwen2-tailfallback.csv`
  - `LLAMA_GRAPH_REUSE_DISABLE=1`

### 关键结果

日志 `tmp/p18/qwen2-tailfallback-device.log` 中最后一层直接出现：

- `attn_norm-23 reason=hetero-last-layer-attn-cpu-tail-fallback backend=CPU`
- `cache_k_upd-23 reason=hetero-last-layer-attn-cpu-tail-fallback backend=CPU`
- `attn_out-tail-23 reason=hetero-last-layer-attn-cpu-tail-fallback backend=CPU`
- `ffn_inp-23 reason=hetero-last-layer-attn-cpu-tail-fallback backend=CPU`

同时，本轮日志中没有再出现旧问题里的：

- `unmatched cgraph: n_nodes=18 first=cache_k_upd-23 last=attn_out-tail-23`
- `unmatched cgraph: n_nodes=1 first=ffn_inp-23 last=ffn_inp-23`

CSV `tmp/p18/qwen2-tailfallback-device.csv` 的 backend 计数为：

- `split_enqueue|CPU = 25`
- `split_compute|CPU = 25`
- `split_enqueue|qnn-npu = 23`
- `split_compute|qnn-npu = 23`
- `OpenCL = 0`

最后 steady-state mixed 图的尾部已经收敛成单个 CPU tail split：

- `split_enqueue,46,CPU,...,norm-22,result_output`
- `split_compute,46,CPU,...,norm-22,result_output`

### 吞吐

- 带 profile / trace 的 purity 口径：
  - `tg1 = 4.90 ± 0.00`
- 去掉 profile 的 clean mixed 口径：
  - `tg1 = 5.24 ± 0.00`

这个值依然不能和 full-graph `pp128` 的几千 `tok/s` 相比，它只是 decode `tg1` mixed 路线的真机 correctness / fragmentation 验证。

### 判断

强结论：

- `Qwen2` mixed decode 路线里最后一层 `attn tail` 的 scheduler-side CPU fallback 已在真机收口。
- residual `OpenCL` split 与旧的 unmatched fragment 都已消失。

保守结论：

- 这项修复解决的是最后一层阶段边界碎片问题，不等于 mixed decode 路线已经变快。
- mixed decode 仍然只有 `~5 tok/s` 量级，后续解释仍应优先放在阶段切换、同步、host/shared KV、CPU tail 等 runtime overhead 上。

## 二、`Qwen2` Full-Graph QNN AoT 静态性能复核

### 先发现的问题

设备侧最初的 full-graph AoT 目录并不完整：

- 存在：
  - `config.json`
  - `lm_head.bin`
- 缺失：
  - `qwen2_0.5b_0.bin`

这意味着设备一开始并不具备可复现的 `Qwen2` full-graph AoT 条件。

### 修复后的标准口径

- 模型：
  - `/data/local/tmp/acom-attn-proj-verify/models/qwen2_0.5b/ggml/weights.gguf`
- AoT 配置：
  - `/data/local/tmp/acom-attn-proj-verify/models/qwen2_0.5b/qnn/config.json`
- AoT 模型目录：
  - `/data/local/tmp/acom-attn-proj-verify/models/qwen2_0.5b/qnn`
- bench 参数：
  - `-r 1 -n 0 -c 2048`
  - `-p 128 -b 128 -ub 128`
  - `-ctk f32 -ctv f32`
  - `-ngl 99 -dev qnn-npu --mmap 0`

### 结果

补齐 `qwen2_0.5b_0.bin` 后，`tmp/p18/qwen2-fullgraph-aot-pp128.log` 给出：

- `pp128 = 3203.83 ± 0.00 tok/s`

### 判断

强结论：

- 本轮 `Qwen2` full-graph QNN AoT 在主设备上恢复到了 `3k tok/s` 量级。
- 之前“QNN 很慢”的表象，不是 QNN full-graph AoT 本身退化，而是设备端 AoT 产物不完整。

对主线的意义：

- 后续凡是引用 `Qwen2 qnn-npu` 静态性能，都必须注明是否是：
  - static non-AoT
  - full-graph AoT
  - mixed split route
- 否则会把部署问题误写成后端能力问题。

## 三、`Qwen3` Full-Graph QNN AoT `pp128` 复核

### 第一次复测：错误口径

先使用 runtime package 自带的：

- `config.json`
- `qwen3_1.7b_0.bin`
- `lm_head.bin`

这组配置只有 `batch_size=1` 图。

在 `pp128` 口径下，`tmp/p18/qwen3-fullgraph-aot-pp128.log` 的结果是：

- `pp128 = 26.73 ± 0.00 tok/s`

这不能被视为 `Qwen3 full-graph AoT pp128` 的性能结论，因为：

- `config.json` 只含 `batch_1`
- 没有 `batch_128` graph
- `pp128` 因而没有进入正确的 batch128 AoT 路径

### 第二次复测：正确口径

随后补齐了仓库中的 `batch128` 资产：

- `config_hvx8.json`
- `qwen3_1.7b_b128mm_0.bin`
- `lm_head_b128.bin`

其中 `config_hvx8.json` 明确包含：

- `batch_1`
- `batch_128`

再次在同样的 `pp128` 口径下复测，`tmp/p18/qwen3-fullgraph-aot-pp128-hvx8.log` 给出：

- `pp128 = 901.03 ± 0.00 tok/s`

### 判断

强结论：

- `Qwen3` 在主设备上的 `pp128` full-graph QNN AoT 确实可以回到 `~900 tok/s` 量级。
- 先前的 `26.73 tok/s` 不是 `Qwen3 AoT` 真正能力，而是因为只使用了 `batch1` 图。

对主线的意义：

- `Qwen3` 和 `Qwen2` 一样，也必须严格区分：
  - `batch1-only AoT`
  - `batch128-capable full-graph AoT`
- 否则不同 batch 图混用会直接污染静态基线，进而污染后续阶段异构性与 runtime overhead 分析。

## 本轮结论

本轮已经形成三条强结论：

1. `Qwen2` mixed decode 的最后一层 tail fallback 已在真机闭环，旧的 unmatched fragment 与 residual `OpenCL` split 都已收口。
2. `Qwen2` full-graph QNN AoT 的静态性能没有“异常变慢”，此前问题来自设备端缺失 `qwen2_0.5b_0.bin`。
3. `Qwen3` full-graph QNN AoT 的 `pp128` 需要显式使用 `batch128` 图；如果误用 `batch1-only` 配置，会把 `~901 tok/s` 错写成 `26.73 tok/s`。

## 对后续工作的影响

- mixed decode correctness 这条线上，`Qwen2` 的最后一层 fragment 已经不是 blocker。
- 后续如果 mixed decode 仍慢，应优先解释：
  - stage-chain launch 数
  - sync / boundary cost
  - shared-host KV path
  - CPU tail/output tail
- 静态基线这条线上，`Qwen2/Qwen3` 的 `qnn-npu` 结果今后必须绑定具体 AoT 配置与 batch 图，不可再只写“qnn-npu”。

## 产物

- `tmp/p18/db6c02cf-list-devices.txt`
- `tmp/p18/qwen2-tailfallback-device.log`
- `tmp/p18/qwen2-tailfallback-device.csv`
- `tmp/p18/qwen2-tailfallback-clean.log`
- `tmp/p18/qwen2-fullgraph-aot-pp128.log`
- `tmp/p18/qwen3-fullgraph-aot-pp128.log`
- `tmp/p18/qwen3-fullgraph-aot-pp128-hvx8.log`
