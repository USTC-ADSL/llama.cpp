# 任务 011：基于当前证据给出阶段后端倾向的初判

日期：2026-03-22

## 背景与目标

在完成：

- 主设备 `CPU / GPUOpenCL / qnn-npu` static baseline
- decode 三段 mixed-route 验证
- prefill full-vs-split overhead 解释
- `CPU / qnn-npu` 最小 stage-profiler

之后，当前非功耗主线已经不再缺“能不能跑”的证据，开始需要一个更具体的判断：

- **接下来应该优先把哪个阶段继续往哪个后端上推进**

如果这一步不先收口，后续：

- `P4-1 Decode 边界 overhead 分解`
- `P3-4 阶段最优后端判定`
- `P6` 里的 cost model 设计

都会继续在“到底优先盯 `attn_core`、`ffn` 还是 `attn_proj`”之间来回切换。

## 执行内容

本次没有新增代码或设备实验，而是把以下现有证据收敛到同一个判断框架里：

- `progress/2026-03-22-004-db6c02cf-decode-baseline.md`
- `progress/2026-03-22-005-db6c02cf-prefill-baseline.md`
- `progress/2026-03-22-006-db6c02cf-mainline-evidence-summary.md`
- `progress/2026-03-22-010-db6c02cf-stage-profiler-p16n16c512.md`
- `docs/qnn-attn-core-shared/decode-stage-backend-support-matrix-2026-03-22.md`
- `docs/qnn-attn-core-shared/host-validation-2026-03-22.md`
- `docs/qnn-attn-core-shared/db6c02cf-stage-profiler-p16n16c512-2026-03-22.md`

## 关键证据

### 1. `Decode` 的 static 最强后端仍然是 `GPUOpenCL`

主设备 static baseline：

- `GPUOpenCL tg1 = 71.06`
- `CPU 1c tg1 = 14.59`
- `qnn-npu tg1 = 10.87`

因此任何后续阶段级方案都不应建立在：

- “整条 decode 静态 NPU 本来就最强”

这个前提上。

### 2. `Decode` 的 mixed-route 最强证据仍然围绕 `attn_core`

当前主设备已验证的 mixed route 中，最好值是：

- `attn_proj=opencl, attn_core=qnn-npu, ffn=opencl`
- `tg1 = 24.35`

同时：

- `attn_proj=cpu, attn_core=qnn-npu, ffn=opencl`
  - `tg1 = 14.13`
- `attn_proj=cpu, attn_core=qnn-npu, ffn=cpu`
  - `tg1 = 19.20`

并且已有 trace 明确表明：

- `attn_core` 与相邻 `CPU / OpenCL` 子图之间可以走 shared host
- 未观察到显式 `tensor_copy`

这说明当前 decode 主线里，最值得继续押注的 `qnn-npu` stage 仍然是：

- `attn_core`

但这里要特别注意：

- 这是 **AoT mixed-stage 路线** 的优势
- 不是“static qnn-npu 自身的 `attn_core` 一定更快”

### 3. `Decode` 里 `ffn=qnn-npu` 也有价值，但优先级略低于 `attn_core`

当前已有独立 route：

- `attn_proj=cpu, attn_core=cpu, ffn=qnn-npu`
- `tg1 = 23.31`

这说明 `FFN` 单独交给 `qnn-npu` 也确实能形成有价值的 decode 路线。

同时主设备 stage-profiler 也给出了一条重要补充：

- `Prefill` 里 static `qnn-npu` 的 `FFN_Block` 明显快于 `CPU`
  - `1990.58 us` vs `5843.39 us`

因此 `FFN` 仍然是一个值得继续推进的强候选。

但与 `attn_core` 相比，它当前的优先级略低，原因是：

1. `attn_core` 已经是 decode 路线里 shared-host boundary 最成熟的一段。
2. `attn_core -> ffn` 正好又是当前最值得做 overhead 分解的一条边界。

### 4. `attn_proj=qnn-npu` 已可行，但当前没有显示出最高优先级

当前已有独立 route：

- `attn_proj=qnn-npu, attn_core=cpu, ffn=cpu`
- `tg1 = 21.80`

而在 static stage-profiler 里：

- `Decode Attn_Proj`
  - `CPU = 906.19 us`
  - `qnn-npu = 871.39 us`

这说明：

- `attn_proj` 并不是完全没有价值；
- 但它目前更像：
  - **可行候选**
  - 而不是“最值得优先攻的瓶颈段”

### 5. `Prefill` 当前最像 `qnn-npu` 优势段的是 `FFN`，不是 `attn_core`

`Prefill` 的最小 stage-profiler 给出的方向很清楚：

- `FFN_Block`
  - `qnn-npu = 1990.58 us`
  - `CPU = 5843.39 us`
- `Attn_Core`
  - `qnn-npu = 4017.45 us`
  - `CPU = 312.42 us`
- `KV_Cache`
  - `qnn-npu = 274.23 us`
  - `CPU = 124.09 us`

再结合 full-vs-split 证据：

- split warm gap 仍约 `1.70x`
- `72` 次 graph launch
- 更重的 `qnn-npu-host`
- 每层 `attn_core` 的 shared-host KV writeback

当前更合理的判断是：

- `Prefill` 如果继续做细粒度 stage split，第一优先不是把 `attn_core` 往 `qnn-npu` 上堆；
- 更像应该优先观察：
  - `FFN`
  - 或更粗粒度的合并 stage

### 6. `KV_Cache` 当前更应被视为 overhead 敏感段，而不是独立候选后端

无论在：

- static `qnn-npu` stage-profiler
- 还是 split prefill trace

里，`KV_Cache` 都没有表现出“天然适合 `qnn-npu`”的证据。

相反，当前更强的事实是：

- decode static `qnn-npu` 的 `KV_Cache` 比 CPU 略慢
- prefill static `qnn-npu` 的 `KV_Cache` 比 CPU 明显更慢
- split prefill 的 shared-host KV writeback 还是当前主要 overhead 候选之一

因此当前更合理的策略不是“给 KV 单独选后端”，而是：

- 把 `KV_Cache` 当作阶段边界设计中的约束项
- 优先降低它带来的 runtime overhead

## 当前结论

截至目前，当前证据支持下面这个阶段后端优先级判断：

1. `Decode`
   - 第一优先：`attn_core=qnn-npu`
   - 第二优先：`ffn=qnn-npu`
   - 第三优先：`attn_proj=qnn-npu`
   - static fallback：`GPUOpenCL`
2. `Prefill`
   - 第一优先候选：`FFN`
   - `attn_core` 当前不应作为细粒度 split 的首选推进方向
   - 更合理的是优先控制 graph granularity 与 KV/runtime overhead
3. `KV_Cache`
   - 当前不应被当成“独立可赢的 stage”
   - 更应被当成：
     - runtime overhead
     - data placement
     - stage-boundary 约束

这组判断的主线意义是：

- 下一步不应该平均用力；
- 应该把实验与分析资源优先压到：
  - `Decode attn_core / ffn`
  - `Prefill FFN`
  - `KV` 边界 overhead

## 下一步

1. 继续推进 `P4-1 Decode 边界 overhead 分解`：
   - 重点盯 `attn_proj -> attn_core`
   - 以及 `attn_core -> ffn`
2. `Prefill` 侧优先做：
   - 更粗粒度图
   - 或 `FFN` 候选验证
   而不是继续把 `attn_core` 做成过碎 split
3. 等 `GPUOpenCL` 的 stage-profiler blocker 解决后，再把这份“初判”升级成更正式的三后端阶段最优判定。
