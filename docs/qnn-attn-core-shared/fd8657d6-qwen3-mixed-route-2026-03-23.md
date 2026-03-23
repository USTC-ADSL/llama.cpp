# `fd8657d6` `Qwen3` Mixed-Route Validation

## Goal

这轮工作的目标不是测功耗，而是先回答两个更基础的问题：

1. `Qwen3-1.7B` 的 `attn_proj / attn_core / ffn` 三段子图，是否都已经能成功导出 `batch=1 / 128` 的 QNN AoT 产物。
2. 在第二台设备 `fd8657d6` 上，这三段分别走 `qnn-npu`、其余阶段走 `OpenCL` 时，`decode` 与 `prefill` 是否能实际跑通，以及当前主要卡在哪类 runtime 问题上。

这份记录优先服务于 decode-centric 主线，但因为本轮明确导出了 `batch=128` 产物，也同步补了 `pp128`。

## Setup

- 日期：`2026-03-23`
- 设备：`fd8657d6`
- 模型：`/data/local/tmp/Qwen3-1.7B-Q8_0.gguf`
- 构建目录：`/data/local/tmp/acom-fd-qwen23-verify/bin`
- host AoT 根目录：`models/Qwen3-AoT`
- 设备 AoT 根目录：`/data/local/tmp/acom-fd-qwen23-verify/models/qwen3_1.7b/models/Qwen3-AoT`
- 统一 cache 类型：`-ctk f32 -ctv f32`
- 统一 ctx：`2048`
- decode bench：`-p 0 -n 64 -b 1 -ub 1`
- prefill bench：`-p 128 -n 0 -b 128 -ub 128`
- 统一运行时环境：
  - `LLAMA_BENCH_FAST_EXIT=1`
  - `GGML_HEXAGON_EXPERIMENTAL=1`
  - `GGML_HETERO_QNN_SHARED_HOST=1`

补充说明：

- `adb shell input keyevent KEYCODE_WAKEUP` 在这台设备上因为 `INJECT_EVENTS` 权限不足而失败，所以这轮没有做到强制亮屏锁定。
- 因此这轮结果应被理解为“非功耗、非强制亮屏条件下的功能/吞吐验证”。

## Export Status

host 侧 `Qwen3` 三段子图的 `batch=1 / 128` AoT 导出全部完成，且 merged config 已生成：

- `models/Qwen3-AoT/qwen3-qnn-full/qnn_attn_proj_combined.json`
- `models/Qwen3-AoT/qwen3-qnn-full/qnn_attn_core_combined.json`
- `models/Qwen3-AoT/qwen3-qnn-full/qnn_ffn_combined.json`
- `models/Qwen3-AoT/qwen3-qnn-full/qnn_attnproj_ffn_combined.json`
- `models/Qwen3-AoT/qwen3-qnn-full/qnn_attnproj_attncore_ffn_combined.json`

本轮直接回答了此前关于 “`Qwen3` 的 `attn_core` 图和 `Qwen2` 不一样，会不会导致它根本导不出来” 的担心：

- 当前证据表明，`Qwen3 attn_core` 并没有卡在 export/build 阶段。
- 如果设备侧仍出现 `unmatched cgraph`，问题更像是 runtime matcher / boundary residual / buffer contract，而不是 “AoT 子图根本生成失败”。

## Route Matrix

### 同配置 static `GPUOpenCL` 基线

为了避免把 mixed-route 的 `f32 KV` 结果和之前 `f16 KV` 的旧表直接硬比，这里补了两条同设备、同 `f32 KV` 的 static `GPUOpenCL` 基线：

| Phase | Route | Throughput | Log |
| --- | --- | ---: | --- |
| `Decode tg64` | static `GPUOpenCL` | `26.967451 tok/s` | `tmp/qwen3_fd8657d6_mixed_20260323/qwen3_gpu_static_f32_tg64.log` |
| `Prefill pp128` | static `GPUOpenCL` | `517.499837 tok/s` | `tmp/qwen3_fd8657d6_mixed_20260323/qwen3_gpu_static_f32_pp128.log` |

### Mixed-route 结果

| Phase | Route | AoT config | Result | Throughput | vs static GPUOpenCL(f32) | Key evidence | Log |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| `Decode tg64` | `attn_proj=qnn-npu, attn_core=opencl, attn_out=opencl, ffn=opencl, output=cpu` | `qnn_attn_proj_combined.json` | 跑通 | `28.122033 tok/s` | `+4.28%` | `tg1` smoke 未见 `unmatched cgraph` / `ggml_hetero_copy`；日志整体为 share-heavy | `tmp/qwen3_fd8657d6_mixed_20260323/qwen3_attnproj_opencl_tg64.log` |
| `Prefill pp128` | `attn_proj=qnn-npu, attn_core=opencl, attn_out=opencl, ffn=opencl, output=cpu` | `qnn_attn_proj_combined.json` | 跑通 | `652.599417 tok/s` | `+26.11%` | 当前日志未见 residual `unmatched` 或显式 copy 证据 | `tmp/qwen3_fd8657d6_mixed_20260323/qwen3_attnproj_opencl_pp128.log` |
| `Decode tg64` | `attn_proj=opencl, attn_core=qnn-npu, attn_out=qnn-npu, ffn=opencl, output=cpu` | `qnn_attn_core_combined.json` | 跑通但不纯 | `15.061532 tok/s` | `-44.15%` | 最后一层 residual 反复出现：`cache_k_upd-27 -> attn_out-tail-27` 与 `ffn_inp-27` | `tmp/qwen3_fd8657d6_mixed_20260323/qwen3_attncore_opencl_tg64.log` |
| `Prefill pp128` | `attn_proj=opencl, attn_core=qnn-npu, attn_out=qnn-npu, ffn=opencl, output=cpu` | `qnn_attn_core_combined.json` | 跑通但不纯 | `613.846311 tok/s` | `+18.62%` | 同样复现最后一层 residual `unmatched cgraph` | `tmp/qwen3_fd8657d6_mixed_20260323/qwen3_attncore_opencl_pp128.log` |
| `Decode tg1` smoke | `attn_proj=opencl, attn_core=opencl, attn_out=opencl, ffn=qnn-npu, output=cpu` | `qnn_ffn_combined.json` | 失败 | 无 | 无 | 大量 `ggml_hetero_copy` / `tensor_copy` 之后触发 `ggml-opencl.cpp:6384: GGML_ASSERT(dst->extra)` | `tmp/qwen3_fd8657d6_mixed_20260323/qwen3_ffn_opencl_smoke.log` |
| `Decode tg64` | `attn_proj=opencl, attn_core=opencl, attn_out=opencl, ffn=qnn-npu, output=cpu` | `qnn_ffn_combined.json` | 失败 | 无 | 无 | 再次在 `ggml-opencl.cpp:6384` abort | `tmp/qwen3_fd8657d6_mixed_20260323/qwen3_ffn_opencl_tg64.log` |
| `Prefill pp128` | `attn_proj=opencl, attn_core=opencl, attn_out=opencl, ffn=qnn-npu, output=cpu` | `qnn_ffn_combined.json` | 失败 | 无 | 无 | 多次 `ggml_hetero_copy` 发生在 KV read/update 周边，随后同样触发 `GGML_ASSERT(dst->extra)` | `tmp/qwen3_fd8657d6_mixed_20260323/qwen3_ffn_opencl_pp128.log` |

## Main Findings

### 1. `Qwen3 attn_core` 当前不是 “导不出来”

这是本轮最重要的负排除结论之一：

- `attn_core` 的 `batch=1 / 128` AoT 子图都已经成功导出并生成 per-layer `.bin`；
- 设备侧 `attn_core=qnn-npu` 的 `decode tg1 / tg64` 与 `prefill pp128` 都能执行到 bench 出结果；
- 因此，当前 `Qwen3` 上出现的 `unmatched cgraph` 不能再笼统归因到 “Qwen3 的 attn_core 计算图和 Qwen2 不一样，所以 AoT 导出坏了”。

更准确的表述应是：

- `Qwen3 attn_core` 的 AoT export/build 没有卡住；
- 当前问题位于 runtime matcher 对最后一层 residual tail 的吞图不完整。

### 2. `attn_proj=qnn-npu` 是当前最健康的 Qwen3 mixed route

这条路线在本轮表现最好：

- `decode tg64 = 28.12 tok/s`
- `prefill pp128 = 652.60 tok/s`
- 相对同配置 static `GPUOpenCL(f32)` 分别高 `4.28%` 和 `26.11%`

更关键的是，它当前没有暴露出明显的 purity 问题：

- `tg1` smoke 未见 `unmatched cgraph`
- 快速 grep 未见 `ggml_hetero_copy` / `tensor_copy`
- 日志整体表现为 share-heavy，而不是 copy-heavy

这说明在 `Qwen3` 上，`attn_proj -> attn_core` 这条边界依然是当前最成熟的 mixed boundary。

### 3. `attn_core=qnn-npu` 已经可跑，但 decode 不纯度会被反复放大

这条路线的性质和 `attn_proj=qnn` 明显不同：

- `prefill pp128` 仍然能跑到 `613.85 tok/s`
- 但 `decode tg64` 只有 `15.06 tok/s`

根因并不在显式 copy，而在最后一层 residual fragment：

- `cache_k_upd-27 -> attn_out-tail-27`
- `ffn_inp-27`

在 `pp128` 中，它只是尾层 residual；
但在 `tg64` 中，它会随着每个 decode token 重复出现，因此对 decode 吞吐的影响被显著放大。

所以这条路线目前更准确的状态是：

- `功能上可跑`
- `边界 share-heavy`
- `但 route purity 仍不够`

### 4. `ffn=qnn-npu` 当前的 blocker 不是 `attn_core` 图差异，而是 copy/assert

`ffn=qnn-npu` 在 `decode` 和 `prefill` 两侧都失败，而且失败模式高度一致：

- 前面先出现大量 `ggml_hetero_copy`
- copy 集中在 `cache_k_l* / cache_v_l* / cache_k_read-* / cache_v_read-*`
- 随后在 OpenCL 侧触发：

`ggml/src/ggml-opencl/ggml-opencl.cpp:6384: GGML_ASSERT(dst->extra) failed`

这说明：

- 当前 `ffn` mixed route 的主要问题不是 “Qwen3 的 attn_core 和 Qwen2 不一样，所以 `attn_core` 子图匹配不到”
- 而是 `attn_core(attn_out) -> ffn` 这条边界在现有 buffer/copy 契约下仍然不健康，最终把 OpenCL 侧送进了异常状态

换句话说，当前 `Qwen3` 上最值得优先修的，不是 `attn_core` 的 AoT export，而是 `ffn` 路由周边的 copy / buffer placement 路径。

## Storyline Impact

这轮结果对主线的意义很明确：

1. `Qwen3` 上三段子图都已经具备 AoT export 条件，说明 “阶段级 mixed route” 在模型适配层面并没有停在纸面上。
2. `attn_proj` 与 `attn_core` 的行为差异，继续支持 stage heterogeneity 这条主线：
   - `attn_proj=qnn` 是当前更成熟的路线；
   - `attn_core=qnn` 在 prefill 有潜力，但 decode 仍被 residual purity 问题拖住。
3. `ffn=qnn` 的失败再次强调 runtime overhead / boundary contract 才是真正的系统瓶颈：
   - 当前不是 “QNN 算不动 FFN”
   - 而是 “阶段边界一旦退化成 copy-heavy，系统就会在 runtime 层面直接失效或被 assert 卡死”

## Next Work

按当前证据，后续优先级应是：

1. 优先修 `ffn=qnn` 路线的 OpenCL assert
   - 重点看 `attn_core(attn_out) -> ffn` 前后的 copied tensor buffer contract
   - 尤其是 `cache_k_l* / cache_v_l* / cache_k_read-* / cache_v_read-*` 的 OpenCL 目标 buffer 生命周期与 `dst->extra`
2. 其次收敛 `attn_core=qnn` 的最后一层 residual fragment
   - 目标是消掉 `attn_out-tail-27` 与 `ffn_inp-27`
   - 这一步主要改善 decode purity，而不是先去追求更高 prefill tok/s
3. 在上述两点完成前，不应把 `ffn=qnn` 混合路线写成“已支持”
4. 若要对 mixed-route 吞吐做更强对比，应继续补同配置 static `CPU / qnn-npu / GPUOpenCL` 的 `f32 KV` 基线

