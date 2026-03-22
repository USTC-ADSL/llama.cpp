# `db6c02cf` 主线证据综合摘要

更新日期：2026-03-22

## 目标

这份文档把当前最关键的三类实证材料放到同一处，服务后续：

- 阶段异构性矩阵
- runtime overhead 分解
- 功率可调空间与 `SLO-aware` 调度

这里不试图证明“动态异构调度已经端到端优于所有静态方案”，而是更严格地回答：

1. 主设备当前的静态最强后端是谁。
2. `attn_proj / attn_core / ffn` 这三个 decode 子图是否已经证明存在可用的阶段异构空间。
3. `Prefill` split 路线当前的主要瓶颈究竟是“没跑到”，还是“跑到了但被 runtime overhead 吞掉了收益”。

## 使用口径

- 主设备：`db6c02cf`
- 模型：`qwen2_0.5b`
- 本文混合使用三类证据：
  - 单后端 static baseline
  - decode mixed-stage route 验证
  - `qnn-npu` AoT 下 prefill `full-graph vs split` 对照

这些证据的目的不同，因此不应把所有数字机械地放在同一张“绝对优劣”表里比较。更合理的用法是：

- static baseline 负责给出 phase-level 上下文；
- mixed-route 负责证明阶段异构在 decode 路径上已经具备可执行性；
- full-vs-split prefill 负责证明 runtime overhead 已经是系统收益释放的主瓶颈之一。

## 1. Decode 单后端 baseline

来源：

- `progress/2026-03-22-004-db6c02cf-decode-baseline.md`
- `docs/qnn-attn-core-shared/db6c02cf-decode-baseline-2026-03-22.csv`

统一口径：

- `llama-bench`
- `-r 1 -p 0 -c 2048 -b 1 -ub 1 --mmap 0`
- `tg1 / tg128`

结果：

| 后端 | 配置 | `tg1` | `tg128` |
| --- | --- | ---: | ---: |
| CPU | `taskset 80 -t 1 -ngl 0` | `14.59` | `14.63` |
| CPU | `taskset C0 -t 2 -ngl 0` | `9.84` | `15.33` |
| GPUOpenCL | `taskset 80 -t 1 -ngl 99 -dev GPUOpenCL` | `71.06` | `68.34` |
| qnn-npu | `taskset 80 -t 1 -ngl 99 -dev qnn-npu` | `10.87` | `10.89` |

当前可直接使用的结论：

- `GPUOpenCL` 是当前主设备上最强的静态 decode backend。
- `qnn-npu` 静态 decode 明显弱于 `GPUOpenCL`，也弱于 `CPU 1c`。
- `CPU 1c` 与 `CPU 2c` 不能合并看待：
  - `tg1` 上 `CPU 2c` 更慢；
  - `tg128` 上 `CPU 2c` 略快。

这说明后续若做功率/延迟二元权衡，`CPU 1c` 与 `CPU 2c` 应视为不同 operating points，而不是同一个 CPU backend 的线性放大。

## 2. Decode 阶段级 mixed-route 证据

来源：

- `docs/qnn-attn-core-shared/decode-stage-backend-support-matrix-2026-03-22.md`
- `docs/qnn-attn-core-shared/host-validation-2026-03-22.md`

当前至少已有 6 条 decode 路线证明三段子图中的每一段都可以单独交给 `qnn-npu`：

| 路线 | 角色 | `tg1` | 关键观察 |
| --- | --- | ---: | --- |
| `attn_proj=opencl, attn_core=qnn-npu, ffn=cpu` | `attn_core` 单独上 `qnn-npu` | `13.09` | layer 0 `x/qcur/cache_k/cache_v/out/kcur/vcur` direct-bind 到 `qnn-npu-host` |
| `attn_proj=cpu, attn_core=qnn-npu, ffn=opencl` | `attn_core` 单独上 `qnn-npu` | `14.13` | `CPU -> qnn-npu` 与 `qnn-npu -> OpenCL` 边界都走 shared host |
| `attn_proj=cpu, attn_core=qnn-npu, ffn=cpu` | `attn_core` 单独上 `qnn-npu` | `19.20` | `ggml_hetero_copy / tensor_copy / tensor_copy_wait = 0 / 0 / 0` |
| `attn_proj=opencl, attn_core=qnn-npu, ffn=opencl` | `attn_core` 单独上 `qnn-npu` | `24.35` | 当前测得的 mixed decode 最优值 |
| `attn_proj=qnn-npu, attn_core=cpu, ffn=cpu` | `attn_proj` 单独上 `qnn-npu` | `21.80` | 快速 grep 未见 unmatched/tensor_copy 告警 |
| `attn_proj=cpu, attn_core=cpu, ffn=qnn-npu` | `ffn` 单独上 `qnn-npu` | `23.31` | 快速 grep 未见 unmatched residual fragment |

其中本地补测的 4 条 route 还满足：

- 日志整体是 share-heavy；
- 未观察到 `ggml_hetero_copy`；
- 未观察到 `tensor_copy` / `tensor_copy_wait`；
- 但 `attn_core=qnn-npu` 的两条 route 仍可见最后层 unmatched residual：
  - `cache_k_upd-23 -> attn_out-23`
  - `ffn_inp-23`

把这些 mixed-route 数字与 static baseline 放在一起看，当前主设备上的 decode 结论更清晰：

- mixed decode route 已经明显优于 static `qnn-npu`：
  - 最好值 `24.35` 对比 static `qnn-npu tg1 = 10.87`
- 也普遍优于 static `CPU 1c`：
  - 最好值 `24.35` 对比 static `CPU 1c tg1 = 14.59`
- 但仍显著落后于 static `GPUOpenCL`：
  - static `GPUOpenCL tg1 = 71.06`

因此当前最合理的解释不是“QNN 已经是 decode 最强后端”，而是：

- decode 路径确实存在阶段异构空间；
- 把整个 decode 一把梭交给 `qnn-npu` 在主设备上并不好；
- 但让 `qnn-npu` 只承担特定子图已经能带来比 static `CPU` / static `qnn-npu` 更好的吞吐；
- 现阶段 mixed-route 的主要拦路项已从“显式 activation copy”收缩为“尾部 residual fragment 与其他 runtime overhead”。

## 3. Prefill 单后端 baseline

来源：

- `progress/2026-03-22-005-db6c02cf-prefill-baseline.md`
- `docs/qnn-attn-core-shared/db6c02cf-prefill-baseline-2026-03-22.csv`

统一口径：

- `llama-bench`
- `-r 1 -n 0 -c 2048 --mmap 0`
- `pp128 / pp256 / pp512`

结果：

| 后端 | `pp128` | `pp256` | `pp512` |
| --- | ---: | ---: | ---: |
| CPU | `193.86` | `196.32` | `146.64` |
| GPUOpenCL | `1295.68` | `1705.44` | `1686.60` |
| qnn-npu | `201.17` | `195.47` | `150.46` |

当前可直接使用的结论：

- `GPUOpenCL` 是当前主设备上最强的静态 prefill backend。
- static `qnn-npu` prefill 并没有明显赢过 CPU，只是在 CPU 附近波动。
- 到 `pp512` 时，`CPU` 与 `qnn-npu` 都有明显下滑。

这意味着后续不能把 `Prefill` split/full-graph 研究建立在“static NPU 本来就最强”这个前提上。更稳妥的主线表述应当是：

- 存在可利用的后端差异；
- 但 runtime overhead 很可能足以吞掉其中一大部分潜在收益。

## 4. `qnn-npu` AoT Prefill：full graph vs split

来源：

- `docs/qnn-attn-core-shared/host-validation-2026-03-22.md`

这组数据不再回答“哪个静态后端更强”，而是回答：

- split `attn_proj / attn_core / ffn` prompt 路线是否真的执行了；
- 如果执行了，full-vs-split 的差距是否主要来自 runtime overhead。

### 4.1 split prefill 已真实执行

当前最关键的 trace 证据是：

- `tmp/qnn_split_batch128_only_prefill_pp128_trace_exec_after_kcache_fix.log`
- 连续打印：
  - `24 x execute attn_proj graph`
  - `24 x execute attn_core graph`
  - `24 x execute ffn graph`

因此当前已经不能再把 split prefill 解释成“其实没跑到”。

### 4.2 full-vs-split 差距是真实存在的

在同一台 `db6c02cf`、同一 `pp128` 设置下：

| 模式 | full graph | split | split 相对 full graph |
| --- | ---: | ---: | --- |
| cold `--no-warmup` | `2054.86 ± 892.91` | `1099.35 ± 761.50` | `1.87x` 更慢，吞吐低 `46.5%` |
| warm | `2605.81 ± 94.02` | `1531.31 ± 10.85` | `1.70x` 更慢，吞吐低 `41.2%` |

这说明即使尽量剥掉 lazy-init 影响，split prefill 仍然存在稳定的端到端 slowdown。

### 4.3 当前 gap 的主因是 runtime overhead

full-graph trace：

- 只执行：
  - `transformer`
  - `lm_head`
- `graph splits = 2`
- `qnn-npu compute buffer size = 74.62 MiB`
- `qnn-npu-host compute buffer size = 2.00 MiB`

split trace：

- 实际执行 `24 x 3 = 72` 个 stage graph
- cold path 要 lazy-load `72` 个 binary
- `graph splits = 4`
- `qnn-npu compute buffer size = 15.75 MiB`
- `qnn-npu-host compute buffer size = 75.06 MiB`

更重要的是 `attn_core` 的 direct-bind 命中形态：

- KV 相关 IO 能 direct-bind：
  - `cache_k=1`
  - `cache_v=1`
- 大多数非-KV IO 仍不能 direct-bind：
  - `x=0`
  - `q=0`
  - `k=0`
  - `v=0`
  - `out=0`

这意味着当前 split prefill 即使“算到了 NPU/AoT 上”，runtime 里仍然要承担：

- 更细粒度 graph launch 开销；
- fragment 间额外的 copyback / materialization；
- 每层 `attn_core` 的 shared-host KV writeback；
- 更重的 `qnn-npu-host` buffer 管理成本。

所以当前对 `Prefill` 的最强结论应该是：

- split route 已真实执行；
- full-vs-split 的差距主要是端到端 runtime overhead；
- 这正是后续必须优先量化和优化的瓶颈，而不是再回头纠缠“matcher 到底有没有命中”。

## 5. 对研究主线意味着什么

把以上三组证据放在一起后，当前主线叙事可以更严格地收敛成下面几点：

### 5.1 阶段异构性已经有实证，但强度在 `Decode` 明显高于 `Prefill`

- `Decode` 侧已经证明：
  - `attn_proj / attn_core / ffn` 都可以单独交给 `qnn-npu`
  - 至少部分 mixed-route 可以 share-heavy 地真实执行
  - 当前 mixed-route 吞吐已经优于 static `CPU` 与 static `qnn-npu`
- `Prefill` 侧已经证明：
  - split route 能真实执行
  - 但当前收益释放被 runtime overhead 强烈限制

### 5.2 当前主设备上，static 最强后端是 `GPUOpenCL`

这一点在 `Decode` 与 `Prefill` 上都成立。

因此如果后续要证明动态调度有价值，不能建立在“静态 NPU 就是最强”的假设上，而应当建立在：

- 阶段 heterogeneity
- 不同 operating points
- 功率/能耗差异
- SLO 约束

这些更完整的系统视角上。

### 5.3 当前最强 decode 边界是 `attn_proj -> attn_core`

原因是它已有专门的 KV shared-host / zero-copy 契约，并且现有 decode 日志已证明：

- `CPU / OpenCL -> qnn-npu attn_core`
- 关键 shared KV tensor
- 边界 activation

都可以通过 `qnn-npu-host` 保持 share-heavy。

### 5.4 当前风险最大的边界仍是 `attn_core(attn_out) -> ffn`

原因不是它完全不能跑，而是：

- 虽然已有 `ffn=qnn-npu` 的 decode 路线跑通；
- 但这条边界没有等价于 KV contract 的专门协议；
- `attn_core=qnn-npu` 路线仍暴露了最后层 residual fragment；
- 在 `Prefill` 中，大量非-KV IO 仍无法 direct-bind，说明这条边界相关的 runtime overhead 仍然很可能是主瓶颈之一。

### 5.5 现在还不能声称“系统已经获得端到端动态调度收益”

因为仍然缺：

- 正式的 `phase × stage × backend` latency 矩阵；
- decode/prefill 边界 overhead 分解；
- 功率/能耗测量；
- 计入 `SLO` 约束的 route 选择结果。

当前更准确的说法是：

- 已经证明存在可利用的阶段异构空间；
- 已经观测到明显 runtime overhead；
- 但还没有完成“收益与开销同口径计入”的系统级闭环。

## 6. 下一步的最小进入点

基于当前证据，下一优先级建议收敛为三项：

1. `P3-1 Decode 分阶段 profiling`
   - 目标：补齐 `attn_proj / attn_core / ffn / output` 的 per-stage latency
   - 意义：把“mixed-route 能跑”推进为“哪个阶段更适合哪个后端”

2. `P4-1 Decode 边界 overhead 分解`
   - 重点：`attn_proj -> attn_core` 与 `attn_core(attn_out) -> ffn`
   - 意义：验证当前 decode mixed-route 为何仍远低于 static `GPUOpenCL`

3. `P4-2 Prefill full-vs-split overhead 分解`
   - 重点：graph launch、fragment copyback、KV writeback、shared-host footprint
   - 意义：把当前 `1.70x` warm gap 拆成可优化项，而不是停留在总吞吐差异

在这三步完成前，最稳妥的对外表述应保持为：

- `Decode` 已经具备研究级的阶段异构证据；
- `Prefill` 已经具备研究级的 overhead 证据；
- 但功率可调空间和 `SLO-aware` 动态调度闭环仍需后续实验支持。
