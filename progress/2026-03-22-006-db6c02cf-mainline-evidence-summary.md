# 任务 006：整理主设备 `db6c02cf` 的主线证据综合摘要

日期：2026-03-22

## 背景与目标

在完成：

- decode 单后端 baseline
- prefill 单后端 baseline
- decode 三段 mixed-route 验证
- `qnn-npu` AoT prefill full-vs-split 对照

之后，现有证据已经足够回答一个更高层但当前非常关键的问题：

- 主设备当前的静态最强后端是谁；
- decode 阶段级异构是否已经有可用实证；
- prefill split 的主要问题究竟是“没跑到”还是“runtime overhead 太重”。

如果不先把这些材料收敛到同一个口径里，后续做：

- 阶段异构性矩阵
- overhead 分解
- Prefill/Decode 后端分离与切换优化

时会持续出现“证据分散、表述过强或口径不一致”的问题。

## 执行内容

新增文档：

- `docs/qnn-attn-core-shared/db6c02cf-mainline-evidence-summary-2026-03-22.md`

本次整理将三类证据放到了同一个主线视角下：

1. `db6c02cf` 的 decode static baseline
2. `db6c02cf` 的 prefill static baseline
3. decode 三段 mixed-route 的已验证结果
4. `qnn-npu` AoT prefill `full graph vs split` 的 warm/cold 对照与 trace 解释

## 关键证据

### 1. 当前主设备上，static 最强后端是 `GPUOpenCL`

decode：

- `GPUOpenCL tg1 = 71.06`
- `qnn-npu tg1 = 10.87`
- `CPU 1c tg1 = 14.59`

prefill：

- `GPUOpenCL pp128 = 1295.68`
- `qnn-npu pp128 = 201.17`
- `CPU pp128 = 193.86`

因此后续不能把主线建立在“static NPU 本来就最强”这个假设上。

### 2. decode 阶段异构已经有实证价值

当前至少已有 6 条 decode route 证明：

- `attn_proj`
- `attn_core`
- `ffn`

三段都能分别交给 `qnn-npu`。

并且本地补测的 mixed route 中，最好值达到：

- `attn_proj=opencl, attn_core=qnn-npu, ffn=opencl`
- `tg1 = 24.35`

这已经明显高于：

- static `qnn-npu tg1 = 10.87`
- static `CPU 1c tg1 = 14.59`

同时日志整体 share-heavy，且未见显式 `tensor_copy`。

### 3. prefill split 的问题已经从“能不能跑”转成“overhead 太重”

当前 trace 已证明 split `pp128` prompt 路线真实执行了：

- `24 x attn_proj`
- `24 x attn_core`
- `24 x ffn`

但性能上：

- warm：
  - full graph `2605.81 ± 94.02 tok/s`
  - split `1531.31 ± 10.85 tok/s`
  - split 慢约 `1.70x`
- cold：
  - full graph `2054.86 ± 892.91 tok/s`
  - split `1099.35 ± 761.50 tok/s`
  - split 慢约 `1.87x`

结合 trace，可解释的主要 overhead 包括：

- `72` 次 graph launch
- fragment copyback / materialization
- 每层 `attn_core` 的 shared-host KV writeback
- 更重的 `qnn-npu-host` footprint

## 当前结论

这一步整理后，主线叙事可以更严格地落到下面三点：

1. `Decode` 的阶段异构证据已经足够强，值得继续推进阶段矩阵与边界 overhead 分解。
2. `Prefill` 的 split 路线已经不是 correctness 问题，而是 runtime overhead 问题。
3. 当前还不能宣称系统已获得端到端动态调度收益，因为：
   - 还缺正式的 per-stage latency 矩阵；
   - 还缺 overhead 分解；
   - 还缺 `SLO-aware` 闭环。

## 下一步

按当前优先级，下一项应进入：

1. `P3-1 Decode 分阶段 profiling`
2. `P4-1 Decode 边界 overhead 分解`
3. `P4-2 Prefill full-vs-split overhead 分解`

其中最先落地的入口建议是：

- 先补 `Decode` 的 `phase × stage × backend` 延迟矩阵；
- 再解释为什么 mixed decode route 仍明显低于 static `GPUOpenCL`；
- 最后把 prefill 的 `1.70x` warm gap 拆成可优化的 runtime overhead 项。
