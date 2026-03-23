# 任务 014：补齐 `ideal vs actual` 第一刀

日期：2026-03-23

## 背景与目标

`P4-1/P4-2` 已经分别说明：

- decode mixed route 的显式 scheduler copy 不是主因；
- prefill full-vs-split 的显式 scheduler copy 也不是主因。

接下来必须补的一块就是：

- **阶段局部最优到底还能剩多少 headroom**
- **这个 headroom 是否会在更真实的 token 规模下收缩**

否则后续很容易把：

- 小尺度 stage-profiler 的强异构信号
- 和真实端到端系统收益

错误地直接画等号。

## 本次执行

### 1. 统一了 `ideal` 的口径

本次把 `ideal` 明确限定为：

- 只在同一条 `llama-stage-profiler` runtime 家族中；
- 对每个 stage 在 `CPU / qnn-npu` 之间取较小 `mean_us`；
- 再按 layer 数累加。

这样做的原因是：

- 当前没有 `GPUOpenCL` 的 stage-profiler 列；
- static `qnn-npu` 与 full-graph QNN AoT 不是同一条 runtime 家族。

### 2. 补算了三组 upper bound

- `Prefill p16 / c512`
- `Decode p16 / n16 / c512`
- `Prefill pp128 / c2048`

其中 `pp128` 这一组是关键，因为它避免继续被 `p16` 的小尺度结果误导。

### 3. 把 `pp128` 的真实 runtime 结果单列出来

额外放入：

- static `CPU`
- static `qnn-npu`
- static `GPUOpenCL`
- full-graph QNN AoT
- split QNN AoT

但文档里明确标注这些结果 **不能** 和 stage-profiler 的 upper bound 混成同一个“全局 ideal”，因为 runtime 家族不同。

## 关键结果

### 1. `Prefill p16` 的理想 headroom 很大，但 `pp128` 几乎消失

- `p16 / c512`
  - best static：`155.10 ms`
  - stage-local ideal：`62.57 ms`
  - 看起来有 `59.66%` headroom
- `pp128 / c2048`
  - best static：`534.29 ms`
  - stage-local ideal：`527.77 ms`
  - 只剩 `1.22%` headroom

这说明小尺度 `p16` 的阶段 winner 不能直接拿来证明 `pp128` 的真实调度空间。

### 2. 当前主设备 `Decode` 的 `CPU/qnn` 混排空间本来就很小

- static `CPU`：`1349.07 ms`
- stage-local ideal：`1335.71 ms`
- 差距只有 `0.99%`

因此 decode 主线后续更应优先解决：

- route purity
- mixed tail residual
- runtime overhead

而不是继续假设“只要再找个更好的 `CPU/qnn` stage mix 就会显著变快”。

### 3. `pp128` full-vs-split 的大 gap 属于 runtime 问题，不属于 stage-local ideal 问题

- matched-scale static `qnn-npu` vs stage-local ideal，只差 `6.52 ms`
- 但 full-graph AoT vs split AoT，差了 `33.26 ms`

这进一步坐实：

- `P4-2` 观察到的主瓶颈确实是 QNN backend 内部 stage-chain fragmentation / materialization
- 而不是“split 只是没选到对的 stage backend”

## 产出

- `docs/qnn-attn-core-shared/db6c02cf-ideal-vs-actual-2026-03-23.md`
- `docs/qnn-attn-core-shared/db6c02cf-ideal-vs-actual-2026-03-23.csv`

## 对后续任务的影响

1. `P3-3/P3-4` 在写正式阶段矩阵时，必须把“尺度敏感”写进去，不能只列单一 winner。
2. `P4-4` 需要把微基准与系统级结论并列，帮助说明为什么外层 copy 不重，但端到端仍会慢。
3. decode 侧后续优先做：
   - route purity
   - tail residual 收口
4. 再进入最小 `SLO-aware` cost model 闭环，避免在 headroom 已很小的场景里过度设计。
