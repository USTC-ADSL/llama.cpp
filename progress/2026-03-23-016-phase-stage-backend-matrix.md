# 任务 016：形成正式的 `phase × stage × backend` 矩阵与后端判定

日期：2026-03-23

## 背景与目标

在 `P4-1 ~ P4-4` 完成后，当前主线已经不缺“零散证据”，缺的是：

- 一张正式矩阵，说明不同 phase / stage 在当前设备上的真实倾向；
- 一份判定，明确哪些结论够硬，哪些只能算条件判断。

否则后续：

- route purity 修复
- `SLO-aware` cost model
- 论文式结论表达

都还会继续在“小尺度 winner”和“系统级约束”之间打架。

## 本次执行

### 1. 把矩阵层和解释层拆开

矩阵层只保留：

- 可复现的 `CPU / qnn-npu` stage 数据
- `phase + scale + stage` 三维索引

解释层才加入：

- static baseline
- ideal-vs-actual
- overhead
- `hetero-switch-bench`

这样后续引用时不会再把“测到的数字”和“研究者判断”写混。

### 2. 正式把 `pp128` matched-scale 引入矩阵

这一步是关键，因为它把之前容易误导的 `p16` winner 收住了：

- `Prefill Attn_Core`
  - `p16`：CPU 明显更快
  - `pp128`：`qnn-npu` 明显更快
- `Prefill FFN_Block`
  - `p16`：`qnn-npu` 明显更快
  - `pp128`：几乎持平并翻到 CPU

因此现在可以明确说：

- `Prefill` 的 stage winner 不能脱离 prompt 规模与 runtime family 单独陈述。

### 3. 明确了当前最硬的结论

最硬的不是“谁永远最快”，而是：

- `KV_Cache` 当前最稳定地倾向 `CPU / host-visible`
- `Decode` 当前最需要继续解决的是 purity / residual / overhead，而不是继续找 `CPU/qnn` 静态 mix
- `GPUOpenCL` 虽然 static 最强，但由于 stage-profiler 缺列，不能被冒进地写成正式 stage winner

## 产出

- `docs/qnn-attn-core-shared/db6c02cf-phase-stage-backend-matrix-2026-03-23.md`
- `docs/qnn-attn-core-shared/db6c02cf-phase-stage-backend-matrix-2026-03-23.csv`

## 对后续任务的影响

1. decode 侧后续优先修：
   - route purity
   - tail residual
   - mixed boundary overhead
2. `SLO-aware` cost table 第一版只能先吸收稳定结论，不能把 `Prefill FFN=qnn` 这类小尺度结论直接写死。
3. `GPUOpenCL` stage-profiler blocker 仍是重要缺口，但它已经不再阻塞当前先做 decode purity / SLO 最小闭环。
