# `db6c02cf` phase × stage × backend 矩阵

更新日期：2026-03-23

## 目标

这份文档补齐：

- `P3-3 阶段×后端矩阵汇总`
- `P3-4 阶段最优后端判定`

并把最近两天新增的 `P4-1/P4-4` 结果也纳入解释层，避免只看 stage winner 而忽略：

- scale sensitivity
- runtime family difference
- runtime overhead

## 数据来源

### 1. 分阶段数据

- `docs/qnn-attn-core-shared/db6c02cf-stage-profiler-p16n16c512-2026-03-22.csv`
- `tmp/p43stage/db6c02cf_cpu_prefill_stage_p128_n0_c2048_20260323.json`
- `tmp/p43stage/db6c02cf_qnn_prefill_stage_p128_n0_c2048_20260323.json`

### 2. 端到端与 overhead 参考

- `docs/qnn-attn-core-shared/db6c02cf-decode-baseline-2026-03-22.csv`
- `docs/qnn-attn-core-shared/db6c02cf-prefill-baseline-2026-03-22.csv`
- `docs/qnn-attn-core-shared/db6c02cf-decode-boundary-overhead-2026-03-23.md`
- `docs/qnn-attn-core-shared/db6c02cf-prefill-overhead-2026-03-23.md`
- `docs/qnn-attn-core-shared/db6c02cf-ideal-vs-actual-2026-03-23.md`
- `docs/qnn-attn-core-shared/db6c02cf-hetero-switch-bench-2026-03-23.md`

### 3. 当前口径限制

这张正式矩阵现在只有 `CPU / qnn-npu` 两列的 stage 数据，原因是：

- `GPUOpenCL` 的 `llama-stage-profiler` 路线仍然卡在模型加载阶段：
  - `unable to allocate OpenCL buffer`

因此：

- `GPUOpenCL` 目前只能作为端到端 static baseline 存在；
- 不能被写成“正式 stage winner”。

这点必须在结论里保留，而不能用全局 static 吞吐把 stage-level 缺口遮过去。

## 正式矩阵

汇总 CSV：

- `docs/qnn-attn-core-shared/db6c02cf-phase-stage-backend-matrix-2026-03-23.csv`

| Phase | Scale | Stage | CPU | qnn-npu | 本 scope winner | margin |
| --- | --- | --- | ---: | ---: | --- | ---: |
| `Decode` | `p16+n16 / c512` | `Attn_Proj` | `906.19 us` | `871.39 us` | `qnn-npu` | `3.84%` |
| `Decode` | `p16+n16 / c512` | `KV_Cache` | `1631.12 us` | `1736.14 us` | `CPU` | `6.05%` |
| `Decode` | `p16+n16 / c512` | `Attn_Core` | `111.68 us` | `579.72 us` | `CPU` | `80.74%` |
| `Decode` | `p16+n16 / c512` | `FFN_Block` | `864.21 us` | `1234.62 us` | `CPU` | `30.00%` |
| `Prefill` | `p16 / c512` | `Attn_Proj` | `194.84 us` | `180.11 us` | `qnn-npu` | `7.56%` |
| `Prefill` | `p16 / c512` | `KV_Cache` | `124.09 us` | `274.23 us` | `CPU` | `54.75%` |
| `Prefill` | `p16 / c512` | `Attn_Core` | `312.42 us` | `4017.45 us` | `CPU` | `92.22%` |
| `Prefill` | `p16 / c512` | `FFN_Block` | `5843.39 us` | `1990.58 us` | `qnn-npu` | `65.93%` |
| `Prefill` | `pp128 / c2048` | `Attn_Proj` | `1245.92 us` | `1279.31 us` | `CPU` | `2.61%` |
| `Prefill` | `pp128 / c2048` | `KV_Cache` | `315.94 us` | `360.16 us` | `CPU` | `12.28%` |
| `Prefill` | `pp128 / c2048` | `Attn_Core` | `8724.57 us` | `4779.14 us` | `qnn-npu` | `45.22%` |
| `Prefill` | `pp128 / c2048` | `FFN_Block` | `15649.25 us` | `15843.36 us` | `CPU` | `1.23%` |

## 端到端上下文

这张矩阵如果脱离端到端基线，会被误读，所以这里把最重要的上下文单独列出来：

### 1. static baseline

主设备 static baseline：

- `Decode`
  - `GPUOpenCL tg1 = 71.06`
  - `CPU 1c tg1 = 14.59`
  - `qnn-npu tg1 = 10.87`
- `Prefill`
  - `GPUOpenCL pp128 = 1295.68`
  - `CPU pp128 = 193.86`
  - `qnn-npu pp128 = 201.17`

因此当前主设备上的端到端事实仍然是：

- `GPUOpenCL` 是最强 static backend；
- 但它还没有正式进入 stage matrix。

### 2. matched-scale `ideal vs actual`

`P4-3` 已经表明：

- `Prefill p16` 的强异构 headroom，到了 `pp128` 后会急剧收缩；
- 当前 `Decode` 的 `CPU/qnn` 混排 headroom 已经接近耗尽。

所以这张矩阵的正确读法不是：

- “哪里赢了就直接硬编码到调度器里”

而是：

- “哪些 stage 倾向稳定，哪些 stage 高度依赖 scale / runtime family”

## 阶段最优后端判定

下面这张表是解释层，不是原始数据层。

| Phase | Stage | 当前判断 | 证据强度 | 依据 | runtime 风险 |
| --- | --- | --- | --- | --- | --- |
| `Decode` | `Attn_Proj` | 不应作为当前首要 `qnn-npu` 推进方向 | 弱到中 | `qnn-npu` 只比 CPU 快 `3.84%`；`attn_proj=qnn` route 可跑，但不是最强 mixed route | winner 幅度小，容易被切换开销抵消 |
| `Decode` | `KV_Cache` | 优先留在 `CPU / host-visible` 侧 | 强 | `CPU` 在 decode 与 prefill 两个 phase 的现有数据里都更稳；KV 明显更像 overhead 约束项 | 极易被 placement / sync / layout 成本放大 |
| `Decode` | `Attn_Core` | 当前 static winner 是 `CPU`；mixed `qnn-npu` 仍值得研究，但重点已转向 purity/overhead | 中 | `p16` static 下 CPU 明显更快；但 mixed decode 里最成熟的 `qnn-npu` 段仍是 `attn_core` | route purity、tail residual、外围壳层 split |
| `Decode` | `FFN_Block` | 当前 static winner 是 `CPU`；`qnn-npu` 可行但不是现阶段主攻瓶颈 | 中 | `p16` static 下 CPU 快 `30%`；`ffn=qnn` route 能跑但没有超越最佳 mixed 路线 | `attn_core -> ffn -> output` 是当前 decode 风险边界 |
| `Prefill` | `Attn_Proj` | 没有稳定 winner，不应写死 | 弱 | `p16` 下 `qnn-npu` 略优，`pp128` 下 CPU 略优，margin 都很小 | 极易受规模变化与 runtime family 影响 |
| `Prefill` | `KV_Cache` | 优先留在 `CPU / host-visible` 侧 | 强 | `CPU` 在 `p16` 和 `pp128` 上都更快；同时它还是 prefill overhead 的主要敏感项 | shared-host writeback/materialization |
| `Prefill` | `Attn_Core` | 高度 scale-sensitive，暂不硬编码 winner | 中 | `p16` 下 CPU 压倒性更快，`pp128` 下 `qnn-npu` 明显更快 | 容易与 graph family、KV 路径、shared-host 管理耦合 |
| `Prefill` | `FFN_Block` | 不能简单写成“`FFN=qnn` 永远最优” | 中 | `p16` 下 `qnn-npu` 很强，但 `pp128` matched-scale 已接近持平并翻到 CPU；同时 full-graph AoT 又远快于 split/static | graph family 与 runtime implementation 比 stage label 更重要 |

## 当前强结论

1. `KV_Cache` 是当前最稳定的 `CPU / host-visible` 倾向 stage。
2. `Decode` 侧当前真正需要继续追的，不是“再找一个更好的 static `CPU/qnn` stage mix”，而是：
   - route purity
   - tail residual
   - mixed boundary overhead
3. `Prefill` 侧的 stage winner 不能只看 `p16`，必须带上 matched-scale `pp128` 一起看。
4. `GPUOpenCL` 虽然是最强 static backend，但还不能被写成正式 stage winner，因为关键的 stage-profiler 列缺失。

## 当前弱结论

1. 还不能给出三后端 `CPU / GPUOpenCL / qnn-npu` 的全局正式 stage winner。
2. 还不能把 `Prefill Attn_Core` 或 `FFN_Block` 的现有 winner 写成稳定 cost table，而不附带 scale 条件。
3. 还不能把 full-graph AoT 的强结果，直接解释成“某个单独 stage 一定适合 `qnn-npu`”。

## 对主线的意义

这张矩阵把主线里的优先级进一步收紧成了两条：

1. `Decode`
   - 继续收口 purity / residual / overhead
   - 而不是继续扩张 stage 数量
2. `Prefill`
   - 优先解释 graph family、materialization、KV 路径
   - 而不是把 `p16` 的单次 winner 直接写进调度策略

## 下一步

1. 回到代码层，优先处理 decode route purity 与 tail residual。
2. 然后做最小 `SLO-aware` cost model 闭环，但 cost table 只先吸收：
   - `KV_Cache -> CPU/host-visible`
   - `Decode CPU/qnn` headroom 很小
   - `Prefill` winner 需要 scale 条件
3. `GPUOpenCL` 的 stage-profiler blocker 仍是当前最重要的 missing evidence 之一。
