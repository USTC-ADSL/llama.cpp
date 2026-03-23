# `db6c02cf` ideal vs actual 第一刀

更新日期：2026-03-23

## 目标

这份记录补齐 `P4-3 ideal vs actual 对比` 的第一版结论，重点回答三个问题：

1. 如果只在同一条 `CPU / qnn-npu` runtime 家族里做阶段级“理想选后端”，理论上还能剩多少 headroom。
2. 这个 headroom 在 `Prefill/Decode`、以及不同 token 规模下是否稳定。
3. 已有的 `pp128` full-graph AoT / split AoT 结果，究竟是在说明“阶段局部最优”，还是在说明“runtime 家族差异与 overhead”。

## 数据来源与口径

### 1. 同工具上界：只用 `llama-stage-profiler`

本文件中“ideal”默认都指：

- 在同一组 `llama-stage-profiler` 结果里；
- 对每个 stage 取 `CPU` 和 `qnn-npu` 中较小的 `mean_us`；
- 再按实际 layer 数累加出来的 **stage-local upper bound**。

这不是“全系统全后端的全局理想值”，原因有二：

1. 当前还缺 `GPUOpenCL` 的 stage-profiler 列。
2. `llama-stage-profiler` 的 static `qnn-npu` 路径，与 full-graph QNN AoT 不是同一条 runtime 家族。

### 2. 使用的原始文件

- 最小尺度 `p16 / n16 / c512`：
  - `docs/qnn-attn-core-shared/db6c02cf-stage-profiler-p16n16c512-2026-03-22.csv`
- 匹配尺度 `pp128 / c2048`：
  - `tmp/p43stage/db6c02cf_cpu_prefill_stage_p128_n0_c2048_20260323.json`
  - `tmp/p43stage/db6c02cf_qnn_prefill_stage_p128_n0_c2048_20260323.json`
- `pp128` 端到端 runtime 参考：
  - `docs/qnn-attn-core-shared/db6c02cf-prefill-baseline-2026-03-22.csv`
  - `docs/qnn-attn-core-shared/db6c02cf-prefill-overhead-2026-03-23.md`

说明：

- `pp128` 的 CPU stage-profiler 进程退出码是 `134`，但 JSON 已成功写出，因此本轮仍按已落盘 JSON 作为有效结果。

### 3. 汇总表

- `docs/qnn-attn-core-shared/db6c02cf-ideal-vs-actual-2026-03-23.csv`

## 同工具 stage-local upper bound

| 场景 | CPU total | qnn-npu total | stage-local ideal | 当前 best static | ideal 相对 best static | ideal picks |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| `Prefill p16 / c512` | `155.39 ms` | `155.10 ms` | `62.57 ms` | `qnn-npu` | `59.66%` 更低 | `attn_proj=qnn` `kv=cpu` `attn_core=cpu` `ffn=qnn` |
| `Decode p16 / n16 / c512` | `1349.07 ms` | `1697.99 ms` | `1335.71 ms` | `cpu` | `0.99%` 更低 | `attn_proj=qnn` `kv=cpu` `attn_core=cpu` `ffn=cpu` |
| `Prefill pp128 / c2048` | `622.46 ms` | `534.29 ms` | `527.77 ms` | `qnn-npu` | `1.22%` 更低 | `attn_proj=cpu` `kv=cpu` `attn_core=qnn` `ffn=cpu` |

## `pp128` 的端到端 runtime 参考

下面这组数据是端到端真实结果，但它们 **不属于同一个 runtime 家族**，因此不能直接和上表的 stage-local upper bound 混成同一个“全局 ideal”：

| 路线 | `tok/s` | 总时长 |
| --- | ---: | ---: |
| static `CPU` | `193.86` | `660.27 ms` |
| static `qnn-npu` | `201.17` | `636.28 ms` |
| static `GPUOpenCL` | `1295.68` | `98.79 ms` |
| full-graph QNN AoT | `2617.88` | `48.89 ms` |
| split QNN AoT | `1558.00` | `82.16 ms` |

这里最重要的不是排名本身，而是：

- full-graph AoT 和 split AoT 都远快于当前 `llama-stage-profiler` 的 static `CPU / qnn-npu`；
- 但 split 仍比 full-graph 慢 `33.26 ms`；
- 这个差距已经在 `P4-2` 里被证明主要落在 QNN backend 内部 stage-chain，而不是外层 scheduler `tensor_copy`。

## 关键发现

### 1. 小尺度 `Prefill p16` 给出的理想收益，不能直接外推到 `pp128`

在 `p16 / c512` 上：

- stage-local ideal 只有 `62.57 ms`；
- 相对当前 best static `155.10 ms` 看起来有 `59.66%` 的 headroom。

但一旦切到匹配 `pp128 / c2048` 的同工具结果：

- stage-local ideal 变成 `527.77 ms`；
- 相对 best static `534.29 ms` 只剩 `1.22%` 的 headroom。

这说明：

- `Prefill` 的阶段最优后端判断是 **强 scale-sensitive** 的；
- 之前 `p16` 看到的强烈“`FFN=qnn, Attn_Core=CPU`”信号，在更长 prompt 上并没有原样保持；
- 如果不做 matched-scale 对比，就会严重高估后续异构调度的潜在系统收益。

### 2. 当前主设备 `Decode` 的 `CPU/qnn` 阶段混排 headroom 本来就很小

`Decode p16 / n16 / c512` 的 stage-local ideal 只有：

- `1335.71 ms`

而 static `CPU` 已经是：

- `1349.07 ms`

二者只差：

- `13.36 ms`
- 约 `0.99%`

因此在当前主设备、当前 `CPU/qnn` 这两列里：

- `Decode` 的主要问题已经不再是“还没找到更好的 stage 归属”；
- 而更像是：
  - route purity
  - mixed boundary residual tail
  - runtime overhead

这和 `P4-1` 的结论是一致的。

### 3. `pp128` 的 full-vs-split 主要是 runtime family / overhead 问题，不是 stage-local ideal 问题

在 matched-scale `pp128` 的同工具结果里：

- static `qnn-npu = 534.29 ms`
- stage-local ideal = `527.77 ms`

也就是说，在当前 `CPU/qnn` static runtime 家族里，理想混排上界只比 static `qnn-npu` 少：

- `6.52 ms`

但另一边，QNN AoT 的端到端结果里：

- full-graph = `48.89 ms`
- split = `82.16 ms`

两者差：

- `33.26 ms`

因此当前更可靠的解释不是：

- “split 没拿到某种巨大的 stage-local 理想收益”

而是：

- split 与 full-graph 本来就在不同的 graph family / runtime overhead 条件下运行；
- 真正拉开它们差距的是 QNN backend 内部的 stage-chain fragmentation / shared-host materialization；
- 这正是 `P4-2` 已经量化出来的那部分。

### 4. `ideal vs actual` 结论必须绑定到“尺度 + runtime 家族”

截至目前，更稳妥的表述应该是：

- `p16` 可以证明“存在潜在阶段异构性”；
- `pp128` matched-scale 可以证明“这份潜在收益在更真实 prompt 长度下可能大幅收缩”；
- AoT full/split 可以证明“runtime 实现方式本身足以比 stage 归属更重要”。

如果把这三类结果混成一个单一“理想 vs 实际”故事，就会错误地把：

- stage-local backend choice
- runtime family difference
- graph fragmentation overhead

三件事混在一起。

## 对主线的意义

这轮 `P4-3` 第一刀把主线里的两个风险明确化了：

1. 不能再用 `p16` 的强异构信号，直接宣称 `Prefill` 动态调度一定有大系统收益。
2. 对当前主设备的 `Decode` 来说，`CPU/qnn` 阶段切换的理论 headroom 已经接近耗尽；后续更值得投入的是：
   - route purity
   - mixed tail residual 收口
   - 更完整的 `GPUOpenCL` 列

## 当前强结论与弱结论

强结论：

- `Prefill` 的 stage-local ideal upper bound 对 prompt 长度高度敏感。
- 当前 `Decode` 的 `CPU/qnn` 两列里，理论混排空间很小。
- `pp128` split-vs-fullgraph 的主差距不是外层 copy，而是 runtime 内部实现方式。

弱结论：

- 还不能给出包含 `GPUOpenCL` 的全局最优阶段矩阵。
- 还不能把 `llama-stage-profiler` 的 static `qnn` ideal，直接与 full-graph AoT 当作同一比较面。

## 下一步

1. 继续做 `P4-4 微基准与系统级对照`，把 `hetero-switch-bench` 的 shared-host / memcpy 级别成本和 `P4-1/P4-2` 的系统级结论放到一张图里。
2. 在 `P3-3/P3-4` 里把这份“尺度敏感 + runtime family 敏感”的结论写进正式阶段矩阵，而不是只列 stage winner。
3. 回到 decode 主线，优先收口：
   - route purity
   - tail residual / unmatched fragment
4. 再进入最小 `SLO-aware` 决策闭环，避免在 headroom 已很小的场景里过度设计调度器。
