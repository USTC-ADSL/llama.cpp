# 任务 005：采集主设备 `db6c02cf` 的 prefill 单后端 baseline

日期：2026-03-22

## 背景与目标

在主设备 decode baseline 完成后，下一步是补齐 `Prefill` 路径的静态单后端对照，避免后续：

- full-graph AoT vs split prefill
- stage overhead 分解
- power-tunable space

只依赖 decode 侧证据。

本任务的最小目标是为主设备 `db6c02cf` 采集：

- `CPU`
- `GPUOpenCL`
- `qnn-npu`

三类后端在：

- `pp128`
- `pp256`
- `pp512`

下的 baseline。

## 执行内容

设备：

- `db6c02cf`

二进制目录：

- `/data/local/tmp/acom-stage-matrix-verify`

模型：

- `/data/local/tmp/acom-attn-proj-verify/models/qwen2_0.5b/ggml/weights.gguf`

统一参数：

- `-r 1`
- `-n 0`
- `-c 2048`
- `-ctk f32 -ctv f32`
- `--mmap 0`

按 prompt 长度设置：

- `pp128`: `-p 128 -b 128 -ub 128`
- `pp256`: `-p 256 -b 256 -ub 256`
- `pp512`: `-p 512 -b 512 -ub 512`

统一环境变量：

- `LD_LIBRARY_PATH=.`
- `ADSP_LIBRARY_PATH=.`
- `LLAMA_BENCH_FAST_EXIT=1`

NPU 路线额外使用：

- `GGML_HEXAGON_EXPERIMENTAL=1`

本轮共采集 9 条 baseline，并产出 CSV：

- `docs/qnn-attn-core-shared/db6c02cf-prefill-baseline-2026-03-22.csv`

## 关键证据

### 1. 主设备 prefill baseline 结果

| 后端 | `pp128` | `pp256` | `pp512` |
|------|--------:|--------:|--------:|
| CPU | `193.86` | `196.32` | `146.64` |
| GPUOpenCL | `1295.68` | `1705.44` | `1686.60` |
| qnn-npu | `201.17` | `195.47` | `150.46` |

对应日志：

- `tmp/db6c02cf_cpu_prefill_pp128_fast_exit.log`
- `tmp/db6c02cf_cpu_prefill_pp256_fast_exit.log`
- `tmp/db6c02cf_cpu_prefill_pp512_fast_exit.log`
- `tmp/db6c02cf_gpu_prefill_pp128_fast_exit.log`
- `tmp/db6c02cf_gpu_prefill_pp256_fast_exit.log`
- `tmp/db6c02cf_gpu_prefill_pp512_fast_exit.log`
- `tmp/db6c02cf_npu_prefill_pp128_fast_exit.log`
- `tmp/db6c02cf_npu_prefill_pp256_fast_exit.log`
- `tmp/db6c02cf_npu_prefill_pp512_fast_exit.log`

### 2. static `GPUOpenCL` 在主设备 prefill 上明显领先

当前最直接的静态观察是：

- `GPUOpenCL` 在 `pp128/256/512` 全部显著领先
- `CPU` 和 `qnn-npu` 的数值非常接近
- 到 `pp512` 时，`CPU` 与 `qnn-npu` 都有明显下滑

这和 decode 静态 baseline 一样，说明主设备当前的“静态最强后端”不是 NPU，而是 GPUOpenCL。

### 3. static `qnn-npu` prefill 没有天然赢过 CPU

这一点对研究主线很关键：

- `pp128`: `qnn-npu 201.17` vs `CPU 193.86`
- `pp256`: `qnn-npu 195.47` vs `CPU 196.32`
- `pp512`: `qnn-npu 150.46` vs `CPU 146.64`

也就是说，当前主设备 static `qnn-npu` prefill 只是在 CPU 附近波动，并没有像 GPU 一样明显拉开。

这意味着：

- 后续不能把 split/full-graph prefill 的价值建立在“静态 NPU 本来就最强”这个前提上
- 更合理的叙述应该是：
  - 存在可利用的后端差异
  - 但 runtime overhead 可能足以吞掉潜在收益

## 当前结论

主设备 `Prefill` baseline 已经满足最小采集目标：

- `CPU / GPUOpenCL / qnn-npu`
- `pp128 / pp256 / pp512`

当前从主设备静态 baseline 能直接读出的结论是：

1. `GPUOpenCL` 是当前主设备上 prefill 的最强静态后端。
2. `CPU` 与 `qnn-npu` 在 prefill 上接近，且都远低于 `GPUOpenCL`。
3. 到较大 prompt (`pp512`) 时，`CPU` 和 `qnn-npu` 都出现明显下降，说明后续还需要结合功率与稳定性判断不同工作点。

## 还缺什么

当前这组 baseline 仍有几个限制：

1. 只有 `r=1`，还没有稳定方差。
2. 还没有同步功率/能耗测量。
3. 目前只完成了“单后端 static baseline”，还没有把它和：
   - full-graph AoT prefill
   - split prefill
   放进同一张对照表。

## 下一步

1. 将主设备 `Decode + Prefill` baseline 合并，形成正式的静态后端对照表。
2. 进入阶段异构性矩阵整理：
   - 用 static baseline 提供 phase-level 上下文
   - 用已有 mixed-route / split-route 数据解释 stage-level 差异
3. 在后续 overhead 分解里重点解释：
   - 为什么 static `GPUOpenCL` 在 prefill 上远强于 static `qnn-npu`
   - 为什么 split prefill 即使真实执行了 `24 x 3` stage graphs，仍可能显著慢于 full graph。
