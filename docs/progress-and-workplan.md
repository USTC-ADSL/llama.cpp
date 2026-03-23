# 阶段级异构调度研究原型：进度评估与工作计划

> 更新日期：2026-03-23

## 一、研究主线与优先级

当前主线以 [AGENTS.md](/mnt/sda1/pzw/HeteroCompute/llama.cpp-acom/AGENTS.md) 为准，已明确扩展为同时覆盖 `Prefill/Decode`：

1. 系统性证明端侧 LLM `Prefill/Decode` 阶段存在显著的阶段异构性。
2. 系统性证明端侧 LLM `Prefill/Decode` 在不同硬件后端和工作点下存在可利用的功率可调空间。
3. 基于上述观察，构建一个满足 `SLO` 的阶段级功率感知调度框架。
4. 量化揭示 runtime overhead 是释放系统收益的关键瓶颈。

但执行优先级仍保持：

- `Decode` 优先于 `Prefill`
- 阶段级优先于算子级
- runtime overhead 优先于“理想最优”
- `SLO` 优先于单纯最小能耗

这意味着：

- `Prefill` 已经进入主线，不能再视为附属问题。
- 但资源投入顺序仍然应先保证 `Decode` 路径能稳定、可解释、可测量，再把同样的方法扩展到 `Prefill`。

## 二、当前实现与证据进度

### 2.1 阶段边界与路由框架

| 组件 | 状态 | 说明 |
|------|------|------|
| `llama_hetero_route_stage` 阶段枚举 | ✅ | `ATTN / ATTN_PROJ / ATTN_CORE / ATTN_OUT / FFN / OUTPUT` 已具备 |
| `llama_hetero_route_spec` | ✅ | 支持按阶段指定后端 |
| 阶段分类器与 layer id 提取 | ✅ | `ggml-profiler.h` 与路由代码均已可用 |
| `llama_hetero_execution_plan` | ✅ | 路由规格与 KV contract 已进入统一执行计划 |
| `build_hetero_cb` 按阶段 pin 后端 | ✅ | 已在图构建回调中工作 |

当前判断：

- 阶段级接口已经具备，不再是 blocker。
- 主要问题已经从“能不能切”转向“切完后的 fragment 是否稳定、是否值得”。

### 2.2 动态路由与 SLO 框架

| 组件 | 状态 | 说明 |
|------|------|------|
| `PHASE_HEURISTIC` 动态路由 | ✅ | `Prefill/Decode` 已分离 |
| `fallback` 路由 | ✅ | 已实现 |
| `route_switches` 统计 | ✅ | 已实现 |
| `GGML_HETERO_DYNAMIC_SLO_US` | 🔶 | 已接入配置，但尚未进入真正决策 |
| `COST_MODEL_RESERVED` | ⬜ | 仅占位，尚未实现真正的 cost model routing |

当前判断：

- 动态路由框架只有“切换骨架”，还没有形成研究主线意义上的 `SLO-aware` 调度器。
- 这部分不能过度表述为“已完成”，更准确的说法是“接口与决策插点已就绪，策略尚未完成”。

### 2.3 KV cache 跨后端管理

| 组件 | 状态 | 说明 |
|------|------|------|
| `llama_hetero_kv_contract` | ✅ | 已支持 producer/consumer/storage/layout/transfer 建模 |
| `STAGE_SHARED` | ✅ | CPU↔OpenCL 零拷贝路径已存在 |
| `QNN_RPCMEM` / `qnn-npu-host` | 🔶 | 已有结构与部分路径，但仍属条件支持 |
| KV contract 兼容性检查 | ✅ | 已在路由/plan 应用中生效 |
| shared-host `attn_core` 边界 | ✅ | 当前 decode 主线的重要基础 |

当前判断：

- `attn_proj -> attn_core` 已经是当前最强的跨后端边界。
- `attn_core(attn_out) -> ffn` 仍是更高风险的 mixed boundary。

### 2.4 Profiling 与 overhead 观测能力

| 组件 | 状态 | 说明 |
|------|------|------|
| `ggml-profiler.h` | ✅ | 阶段与 phase 分类器已就绪 |
| CPU per-op profiling | ✅ | 已可记录 phase/stage/layer/duration |
| `GGML_HETERO_PROFILE` | ✅ | 已可记录 `split` / `tensor_copy` / `sync` |
| OpenCL profiling | ✅ | 已具备 |
| `hetero-switch-bench` | ✅ | 已能测 shared host vs memcpy 微基准 |

当前判断：

- “看得见 overhead” 这件事已经完成。
- `db6c02cf` 上已经补出一组同尺度 `p16 / n16 / c512` 的 `CPU / qnn-npu` stage-profiler 数据：
  - `Decode` 中，static `qnn-npu` 主要输在 `Attn_Core` 与 `FFN_Block`
  - `Prefill` 中，static `qnn-npu` 的 `FFN_Block` 更快，但 `Attn_Core` 与 `KV_Cache` 更慢
- `GPUOpenCL` 的 stage-profiler 路线当前仍被 OpenCL buffer 分配阻塞，因此系统级结论仍缺完整的三后端 `ideal vs actual` 分解。

### 2.5 Decode 主线进度

当前 `Decode` 已有明确进展：

- `attn_proj / attn_core / ffn` 三段子图的后端支持矩阵已整理：
  - [decode-stage-backend-support-matrix-2026-03-22.md](/mnt/sda1/pzw/HeteroCompute/llama.cpp-acom/docs/qnn-attn-core-shared/decode-stage-backend-support-matrix-2026-03-22.md)
- `attn_core=qnn-npu` 的 shared-host 边界验证已完成：
  - [host-validation-2026-03-22.md](/mnt/sda1/pzw/HeteroCompute/llama.cpp-acom/docs/qnn-attn-core-shared/host-validation-2026-03-22.md)
- 目前至少已有 6 条 decode 路线证明三段中的每一段都可以在特定组合下单独交给 `qnn-npu`。
- 其中 4 条本地 decode 实测显示：
  - 路线可跑通
  - 日志整体 share-heavy
  - 未观察到 `ggml_hetero_copy`
  - 未观察到 `tensor_copy` / `tensor_copy_wait`
- `db6c02cf` 上已经补出同尺度 `p16 / n16 / c512` 的 `CPU / qnn-npu` decode stage-profiler 数据：
  - `CPU`：
    - `Attn_Proj = 906.19 us`
    - `KV_Cache = 1631.12 us`
    - `Attn_Core = 111.68 us`
    - `FFN_Block = 864.21 us`
  - `qnn-npu`：
    - `Attn_Proj = 871.39 us`
    - `KV_Cache = 1736.14 us`
    - `Attn_Core = 579.72 us`
    - `FFN_Block = 1234.62 us`
- 这说明当前主设备 static `qnn-npu` decode 的主要短板不是 `Attn_Proj`，而是：
  - `Attn_Core`
  - `FFN_Block`

当前 `Decode` 仍未解决的问题：

- `attn_core=qnn-npu` 的最后层 residual 仍可能出现：
  - `cache_k_upd-23 -> attn_out-23`
  - `ffn_inp-23`
- 这更像 mixed-stage residual guard 与 tail fragment 切碎，而不是“Qwen3 适配直接破坏了 Qwen2 matcher”。
- `fd8657d6` 上 static `qnn-npu` baseline 已可测，但它们必须与 full-graph AoT 区分使用：
  - `Qwen2` full-graph AoT prefill 已复核到 `pp128 = 4280.48`、`pp256 = 4565.30`
  - `Qwen3` 当前仍缺 second-device full-graph AoT 产物
  - AoT decode 仍不稳定，不能作为正式 AoT 对照基线。

### 2.6 Prefill 主线进度

当前 `Prefill` 已经进入主线，且不再是“完全空白”：

- `db6c02cf` 上 split `batch128-only` prompt 路径已经真实执行：
  - `24 x attn_proj`
  - `24 x attn_core`
  - `24 x ffn`
- full-graph AoT 与 split AoT 已经有可比的 `pp128` 数据：
  - warm：
    - full graph `2605.81 ± 94.02 tok/s`
    - split `1531.31 ± 10.85 tok/s`
  - cold：
    - full graph `2054.86 ± 892.91 tok/s`
    - split `1099.35 ± 761.50 tok/s`
- 当前最强结论已经是：
  - split prefill 确实跑到了
  - full-vs-split 的差距是真实端到端 runtime overhead，而不再是“split 根本没执行”
- 同一批 `p16 / n16 / c512` stage-profiler 也给出了一组最小 `Prefill` 分阶段数据：
  - `CPU`：
    - `Attn_Proj = 194.84 us`
    - `KV_Cache = 124.09 us`
    - `Attn_Core = 312.42 us`
    - `FFN_Block = 5843.39 us`
  - `qnn-npu`：
    - `Attn_Proj = 180.11 us`
    - `KV_Cache = 274.23 us`
    - `Attn_Core = 4017.45 us`
    - `FFN_Block = 1990.58 us`
- 这说明 `Prefill` 的阶段异构已经开始出现清晰方向：
  - `FFN_Block` 更像适合 `qnn-npu`
  - `Attn_Core` 与 `KV_Cache` 当前更像不适合 static `qnn-npu`

当前 `Prefill` 仍未解决的问题：

- split warm gap 仍约 `1.70x`
- `2026-03-23` 的 event-level CSV 第一刀已经表明：
  - 显式 scheduler `tensor_copy` 只有 `2` 次、`7168 B`、`2 us`
  - full-vs-split 的 `33.168 ms` gap 中约 `29.808 ms`（`89.9%`）落在 split 路线那个大的 `qnn-npu` compute 区间内部
- 关键 overhead 候选包括：
  - `72` 次 graph launch
  - fragment I/O direct-bind 命中率不足
  - per-layer shared-host KV writeback/materialization
- 最后一层 `FFN tokens=1` 已确认是 prompt eval 默认 `n_outputs=1` 导致的 output-tail 视图，而不是 split correctness regression；
  - 它会影响 per-stage accounting；
  - 但不改变 “split prefill 已真实执行、gap 主要来自 runtime overhead” 这个主结论。

## 三、主线覆盖度评估

| 主线 | 当前状态 | 证据强度 | 说明 |
|------|------|------|------|
| ① `Prefill/Decode` 阶段异构性 | 🔶 部分完成 | 中 | 接口、切分、部分实测已具备，但还缺正式的 `phase × stage × backend` 延迟矩阵 |
| ② `Prefill/Decode` 功率可调空间 | ⬜ 基本未完成 | 弱 | 路由能力已存在，但仍缺正式功率/能耗数据 |
| ③ `SLO-aware` 调度框架 | 🔶 部分完成 | 弱到中 | 路由骨架已具备，但 `cost model` 与 `slo_us` 尚未形成真正决策闭环 |
| ④ runtime overhead 量化 | 🔶 部分完成 | 中到强 | Decode 已补出第一版 event-level CSV 分解，确认当前 mixed decode 无显式 `tensor_copy`，主要风险转向 split fragmentation、CPU output tail 与 route purity；Prefill 也已补出第一版统一分解表，确认 warm `pp128` full-vs-split gap 的约 `89.9%` 落在 qnn backend 内部 stage-chain 区间，而非外层 scheduler `tensor_copy` |

辅助判断：

- `Decode` 主线进度明显领先于 `Prefill`。
- `Prefill` 已经进入“解释瓶颈”的阶段，而不再只是“验证能不能跑”。
- 真正最薄弱的两环仍然是：
  - 功率/能耗主线
  - `SLO-aware` 调度闭环

## 四、按优先级重排后的工作计划

### Phase 0：主线口径与记录体系

**目标**：把 `Prefill/Decode` 双主线和 `Decode` 优先级统一到文档与执行记录中。

| 任务 | 状态 | 说明 | 产出 |
|------|------|------|------|
| P0-1 建立 `progress/` 记录目录 | ✅ | 逐任务落档 | `progress/README.md` |
| P0-2 调查 `Qwen3` 适配是否导致 `Qwen2` unmatched | ✅ | 初步排除“直接主因” | `progress/2026-03-22-001-qwen3-attn-core-unmatch-investigation.md` |
| P0-3 更新主线工作计划文档 | ✅ | 同步新版 `AGENTS.md` 与现有证据 | 本文档 |

### Phase 1：双路径稳定性止血

**目标**：先把 `Decode` 与 `Prefill` 两条路径都稳定到可持续采数的程度。**

| 任务 | 优先级 | 说明 | 产出 |
|------|------|------|------|
| P1-1 收口 decode tail residual | 最高 | 重点跟踪 `cache_k_upd-23 -> attn_out-23` 与 `ffn_inp-23` 的分裂原因 | 代码补丁 + 日志 |
| P1-2 保证 unmatched residual 统一走 CPU | 高 | 避免 residual 掉回 QNN JIT 小图 | 代码补丁 + 复现命令 |
| P1-3 确认 prefill tail `FFN tokens=1` 语义 | 已完成 | 已确认它是 prompt eval 默认 `n_outputs=1` 下的 output-tail 视图，主要影响 per-stage accounting | `progress/2026-03-22-009-prefill-tail-ffn-output-tail-semantics.md` |
| P1-4 `fd8657d6` 降级为辅助设备 | 中 | 保留 static baseline 能力，不把关键 AoT 实验压在其上 | 设备状态说明 |

### Phase 2：单后端基线采集

**目标**：为 `Prefill/Decode` 两条主线同时建立可比较 baseline。**

| 任务 | 优先级 | 说明 | 产出 |
|------|------|------|------|
| P2-1 主设备 decode baseline | 最高 | `CPU(1c/2c) / GPUOpenCL / qnn-npu`，至少采 `tg1` 与 `tg128` | CSV |
| P2-2 主设备 prefill baseline | 高 | `CPU / GPUOpenCL / qnn-npu`，至少采 `pp128 / pp256 / pp512` | CSV |
| P2-3 第二设备 baseline 补充 | 中 | 仅用于 cross-device sanity check | CSV + 说明 |

### Phase 3：阶段异构性矩阵

**目标**：构建 `Decode` 与 `Prefill` 两张正式的阶段×后端矩阵。**

| 任务 | 优先级 | 说明 | 产出 |
|------|------|------|------|
| P3-1 Decode 分阶段 profiling | 进行中 | 已补出 `CPU / qnn-npu` 的 `p16 / n16 / c512` per-stage latency，`GPUOpenCL` 仍被 OpenCL buffer 分配阻塞 | CSV |
| P3-2 Prefill 分阶段 profiling | 进行中 | 同一批 stage-profiler 已拿到最小 `p16` prompt 数据，但还缺 `GPUOpenCL` 与更长 prompt | CSV |
| P3-3 阶段×后端矩阵汇总 | 高 | 构建 `phase × stage × backend` 矩阵 | 图表/文档 |
| P3-4 阶段最优后端判定 | 高 | 解释计算/访存/KV 依赖和 overhead 风险 | 分析文档 |

### Phase 4：runtime overhead 系统级量化

**目标**：把 “为什么理想收益没有落到端到端收益” 定量拆出来。**

| 任务 | 优先级 | 说明 | 产出 |
|------|------|------|------|
| P4-1 Decode 边界 overhead 分解 | 最高 | 第一刀已完成：确认 decode mixed route 当前无显式 `tensor_copy`，重点瓶颈转向 split fragmentation、`result_output` CPU tail 与 route purity | CSV + 分析 |
| P4-2 Prefill full-vs-split overhead 分解 | 最高 | 第一刀已完成：确认 warm `pp128` gap 的主因不是外层 scheduler copy，而是 qnn backend 内部 stage-chain fragmentation / shared-host materialization | CSV + 分析 |
| P4-3 ideal vs actual 对比 | 高 | 比较阶段最优之和与真实端到端结果 | 差距分析 |
| P4-4 微基准与系统级对照 | 中 | 用 `hetero-switch-bench` 对照端到端观测 | 对比文档 |

- `2026-03-23`：`P4-1` 已补出第一版 decode boundary-overhead 文档：
  - `docs/qnn-attn-core-shared/db6c02cf-decode-boundary-overhead-2026-03-23.md`
- `2026-03-23`：`P4-2` 已补出第一版 prefill full-vs-split overhead 文档：
  - `docs/qnn-attn-core-shared/db6c02cf-prefill-overhead-2026-03-23.md`

### Phase 5：功率可调空间测量

**目标**：正式补上主线 ② 的证据闭环。**

| 任务 | 优先级 | 说明 | 产出 |
|------|------|------|------|
| P5-1 Decode 路线功率测量 | 高 | 测 `CPU/GPU/NPU + 1~2` 条 mixed route | 功率/能耗表 |
| P5-2 Prefill 路线功率测量 | 高 | 测 full-graph 与 split prefill | 功率/能耗表 |
| P5-3 统一实验状态控制 | 高 | 亮屏、频率、warm/cold、device awake | 规范说明 |

### Phase 6：SLO-aware cost model 与动态调度

**目标**：真正实现主线 ③，而不是只保留 `phase heuristic` 骨架。**

| 任务 | 优先级 | 说明 | 产出 |
|------|------|------|------|
| P6-1 建 cost table | 高 | 以 Phase 2~5 的测量结果为输入 | 数据表 |
| P6-2 实现 `COST_MODEL_RESERVED` | 高 | 让动态路由真正使用成本模型 | 代码 |
| P6-3 接入 `slo_us` 约束 | 高 | 在决策时显式比较 route 是否满足 SLO | 代码 |
| P6-4 Prefill→Decode 切换成本验证 | 高 | 验证 route switch 与 KV contract 的运行时代价 | 实验结果 |
| P6-5 SLO 达标率实验 | 高 | 统计不同 phase / route 的达标率 | 数据与分析 |

### Phase 7：针对性工程优化

**目标**：只优化已经被测量证明是主瓶颈的环节。**

| 任务 | 优先级 | 说明 | 产出 |
|------|------|------|------|
| P7-1 提升 direct-bind 命中率 | 视 P4 结果 | 若 fragment I/O copy 主导，则优先做此项 | 代码补丁 |
| P7-2 更粗粒度 AoT family | 视 P4 结果 | 若 graph launch 主导，则尝试 `attention/transformer` 粗化 | 配置/代码 |
| P7-3 shared-host / RPCMEM 优化 | 视 P4 结果 | 若 KV host-visible 管理主导，则优先优化 KV 路径 | 代码补丁 |

## 五、总体优先级排序

```text
P0 主线口径与记录体系
> P1 双路径稳定性止血
> P2 Decode/Prefill 单后端基线
> P3 Decode/Prefill 阶段异构性矩阵
> P4 Decode/Prefill runtime overhead 分解
> P5 功率可调空间测量
> P6 SLO-aware cost model 与动态调度
> P7 针对性工程优化
```

排序原则：

- `Decode` 仍是第一主路径，`Prefill` 是第二主路径。
- `Prefill` 已进入主线，但不会抢占 `Decode` 稳定性与基础数据的优先级。
- 如果没有 `P1~P4` 的测量结果，就不应在 `P6~P7` 上过度设计。

## 六、当前最小可验证方案

如果按主线最小闭环推进，下一批最值得完成的是：

1. 收口 `Decode` 最后一层 tail residual 的 unmatched 问题。
2. 采集主设备 `Decode + Prefill` 的单后端 baseline。
3. 形成 `phase × stage × backend` 的阶段矩阵初稿。
4. 对 `Decode` 和 `Prefill` 各做一版 runtime overhead 分解。

成功意味着：

- `Decode` 与 `Prefill` 都有稳定、可复现、可比较的主路径。
- 主线 ① 与主线 ④ 从“原型存在”提升到“有系统测量支撑”。

失败意味着：

- 仍然无法分辨阶段收益与 runtime overhead 谁在主导。
- `SLO-aware` 调度只能停留在接口层，而无法形成可信的研究结论。

## 七、关键参考文件

| 文件 | 用途 |
|------|------|
| [AGENTS.md](/mnt/sda1/pzw/HeteroCompute/llama.cpp-acom/AGENTS.md) | 当前主线与优先级的唯一权威口径 |
| [decode-stage-backend-support-matrix-2026-03-22.md](/mnt/sda1/pzw/HeteroCompute/llama.cpp-acom/docs/qnn-attn-core-shared/decode-stage-backend-support-matrix-2026-03-22.md) | Decode 三段子图支持矩阵与 4 组补测结果 |
| [host-validation-2026-03-22.md](/mnt/sda1/pzw/HeteroCompute/llama.cpp-acom/docs/qnn-attn-core-shared/host-validation-2026-03-22.md) | `attn_core` shared-host decode 与 prefill 证据 |
| [README.md](/mnt/sda1/pzw/HeteroCompute/llama.cpp-acom/docs/qnn-attn-core-shared/README.md) | 当前 prefill/decode runtime overhead 工作日志 |
| [fd8657d6-baseline-2026-03-22.md](/mnt/sda1/pzw/HeteroCompute/llama.cpp-acom/docs/qnn-attn-core-shared/fd8657d6-baseline-2026-03-22.md) | 第二设备 baseline 与不稳定性说明 |
| `src/llama-dyn-route.h/cpp` | 动态路由框架 |
| `src/llama-context.cpp` | 阶段 pin、plan 应用、动态路由决策点 |
| `ggml/src/ggml-qnn/qnn/aot.cpp` | AoT matcher / execute / prefill/decode 路径 |
| `ggml/src/ggml-backend.cpp` | scheduler、copy、hetero profile |
| `progress/` | 逐任务执行记录 |
