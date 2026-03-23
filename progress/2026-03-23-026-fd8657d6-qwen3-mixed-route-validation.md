# `fd8657d6` `Qwen3` Mixed-Route Validation

## Completed

- 导出了 `Qwen3` 的三段 AoT 子图：
  - `attn_proj`
  - `attn_core`
  - `ffn`
- 每段都补齐了：
  - `batch=1`
  - `batch=128`
- 生成并保存了 merged config：
  - `qnn_attn_proj_combined.json`
  - `qnn_attn_core_combined.json`
  - `qnn_ffn_combined.json`
  - `qnn_attnproj_ffn_combined.json`
  - `qnn_attnproj_attncore_ffn_combined.json`
- 将运行时最小 AoT 包推送到设备：
  - 设备目录：`/data/local/tmp/acom-fd-qwen23-verify/models/qwen3_1.7b/models/Qwen3-AoT`
- 在 `fd8657d6` 上完成了 `Qwen3` mixed-route 的：
  - `decode tg1` smoke
  - `decode tg64`
  - `prefill pp128`
- 补了同设备、同 `f32 KV` 的 static `GPUOpenCL` 基线，用于比较 mixed-route 结果。

## Main Result

- `attn_proj=qnn-npu` 是当前最健康的 `Qwen3` mixed route：
  - `tg64 = 28.12 tok/s`
  - `pp128 = 652.60 tok/s`
  - 相对 static `GPUOpenCL(f32)` 分别 `+4.28%` / `+26.11%`
- `attn_core=qnn-npu` 已经能跑通 `decode/prefill`，但最后一层 residual `unmatched cgraph` 仍在：
  - `cache_k_upd-27 -> attn_out-tail-27`
  - `ffn_inp-27`
  - `tg64 = 15.06 tok/s`
  - `pp128 = 613.85 tok/s`
- `ffn=qnn-npu` 当前在 `decode` 和 `prefill` 两边都失败：
  - 大量 `ggml_hetero_copy` 先出现
  - 然后触发 `ggml-opencl.cpp:6384: GGML_ASSERT(dst->extra)`

## Interpretation

- 这轮已经可以排除一种弱假设：
  - `Qwen3 attn_core` 不是“因为图和 Qwen2 不一样，所以 AoT 导不出来”
- 当前更准确的判断是：
  - `attn_core` export/build 没问题
  - `attn_core=qnn` 的主要问题是 runtime matcher 对最后一层 residual tail 吞图不完整
  - `ffn=qnn` 的主要问题是 `attn_core(attn_out) -> ffn` 边界退化成 copy-heavy，最后把 OpenCL 侧送进 assert

## Artifacts

- 结果文档：
  - `docs/qnn-attn-core-shared/fd8657d6-qwen3-mixed-route-2026-03-23.md`
- 结果 CSV：
  - `docs/qnn-attn-core-shared/fd8657d6-qwen3-mixed-route-2026-03-23.csv`
- 原始日志目录：
  - `tmp/qwen3_fd8657d6_mixed_20260323/`
- host 导出日志目录：
  - `tmp/qwen3_split_aot_20260323/`

## Next

1. 优先修 `ffn=qnn` mixed route 的 `OpenCL dst->extra` assert。
2. 然后收敛 `attn_core=qnn` 最后一层 residual unmatched。
3. 在这两点完成前，不把 `ffn=qnn` 视为当前 `Qwen3` decode 主线上的可用后端选项。

