# Decode 三段子图后端支持矩阵

更新日期：2026-03-22

## 目标与范围

这份文档回答的是 decode 路径里 `attn_proj / attn_core / ffn` 三段子图的后端支持问题，重点是：

1. 当前代码是否允许三段分别指定 `CPU / OpenCL / qnn-npu`。
2. 哪些组合只是“接口可配”，哪些已经达到“静态代码上较强支持”，哪些仍然只是“条件支持”。
3. 哪些跨后端边界具备专门的 shared-host / zero-copy 路径，哪些仍然可能被 scheduler 插入 `tensor_copy`。

本结论基于当前仓库代码静态分析，不把它外推成“所有组合都已稳定可执行”或“系统已经端到端无拷贝”。研究口径仍然遵循 decode-centric、stage-centric、overhead-conscious。

## 结论先行

- 三段独立路由接口已经存在：`attn_proj`、`attn_core`、`attn_out`、`ffn`、`output` 都可以单独指定后端，且 `attn_out` 默认继承 `attn_core`。见 `src/llama-hetero-route.h` 与 `src/llama-context.cpp`。
- 但“任意组合都可稳定运行”这个说法过强，尤其是涉及 `qnn-npu` 时并不成立。
  - `qnn-npu` 在阶段路由里是 AoT-gated 的；没有 `GGML_QNN_AOT_CONFIG` 时，`graph_get_cb()` 根本拿不到 `qnn_aot_backend`，阶段 pinning 不会真正落到 `qnn-npu`。
  - 混合 `QNN + 非 QNN` 的路由还要通过 mixed-stage guard；guard 命中时，QNN 残图会被拒绝，不能简单等价为“只要路由字符串能写就支持”。
- “跨后端不会有张量拷贝”也不成立。
  - scheduler 会先检查 buffer type 是否兼容；不兼容就显式创建副本并在 split 执行前执行 `tensor_copy`。
  - 目前唯一有专门 zero-copy 契约设计的是 `attn_proj -> attn_core` 的 KV 边界；`attn_core(attn_out) -> ffn` 没有等价级别的专门契约，只能依赖 shared-host compute buffer 和 scheduler 的 buffer compatibility，仍然可能 copy。

## 判定图例

- `较强静态支持`：代码路径完整，且没有 `qnn-npu` 的 AoT / mixed-stage 专属门槛。
- `条件支持`：接口可配，但要满足额外前提，例如 `GGML_QNN_AOT_CONFIG`、mixed-stage guard、F32 KV、AoT graph 覆盖范围等。
- `已验证`：仓库现有 decode 文档/日志已验证过该类组合，不是本次新实验。
- `可能拷贝`：scheduler 没有全局 zero-copy 保证，buffer 不兼容时会插入 `tensor_copy`。
- `优先避免`：接口虽可写，但当前证据弱、跨后端边界多、runtime overhead 高，暂不宜当作强支持结论。

下文所有 `qnn-npu` 单元格默认都隐含一个前提：

- 必须设置 `GGML_QNN_AOT_CONFIG`，否则 `qnn-npu` 不会成为阶段回调中的可用目标后端。

对三段矩阵还默认采用下面这个三段切分假设：

- 只讨论 `attn_proj / attn_core / ffn` 三段；
- `attn_out` 不单独设置时默认跟随 `attn_core`；
- `output` 尾部和 `embd` / `inp_tokens` 不纳入本矩阵主结论。

## 关键代码依据

### 1. 三段独立路由接口已存在

- 路由字段与继承关系定义在 `src/llama-hetero-route.h`：
  - `ATTN_PROJ`
  - `ATTN_CORE`
  - `ATTN_OUT`
  - `FFN`
  - `OUTPUT`
- `backend_for()` 明确了 `attn_out` 默认继承 `attn_core`，`output` 默认再向上继承。

### 2. 三段张量边界已被显式命名

- `attn_proj`：`norm-*`、`attn_norm-*`、`Qcur*`、`Kcur*`、`Vcur*`
- `attn_core`：`__fattn__-*`、`cache_k_*`、`cache_v_*`、`kq*`、`kqv*`、`v_cont-*`
- `attn_out`：`attn_out-*`、`ffn_inp-*`
- `ffn`：`ffn*`（去掉 `ffn_inp-*`）和 `l_out-*`

这说明三段边界已经被 graph callback 用名字分类，而不是只靠算子级猜测。

### 3. graph callback 会按阶段尝试 pin 到指定后端

- `src/llama-context.cpp` 的 `graph_get_cb()` 里会解析：
  - `hetero_attn_proj_backend`
  - `hetero_attn_core_backend`
  - `hetero_attn_out_backend`
  - `hetero_ffn_backend`
  - `hetero_output_backend`
- 然后按阶段类型决定每个 tensor 的目标 backend。

### 4. 真正落到目标后端前还要过 `supports_op`

- 无论普通 hetero 路由还是 QNN AoT 路由，最终都要先检查 `ggml_backend_supports_op(target_backend, cur)`。
- 所以“路由接口能写”不等于“该子图一定实际在那个后端上执行”。

### 5. `qnn-npu` 是 AoT-gated 的

- `src/llama-context.cpp` 里只有在 `GGML_QNN_AOT_CONFIG` 存在时才把 `qnn-npu` 视为 `qnn_aot_backend`。
- `ggml/src/ggml-qnn/qnn/backend-ops.cpp` 还对 mixed-stage QNN AoT 做了 guard：
  - mixed-stage QNN AoT graph 不可用时，会跳过把 transformer fragment 指派给 QNN；
  - mixed-stage route 未显式要求 `qnn-cpu` 时，还会把 residual fragment 留在 plain CPU，避免额外的 `qnn-cpu` split。

### 6. scheduler 会显式插入 copy

- `ggml/src/ggml-backend.cpp` 先用 `ggml_backend_sched_buffer_supported()` 判断 buffer type 是否兼容。
- 若不兼容，就创建 `tensor_copy(...)` 对应的副本，并在 split 执行前显式做 copy。
- `GGML_HETERO_PROFILE=1` 时还会记录 `tensor_copy` 与 `tensor_copy_wait`。

### 7. `attn_proj -> attn_core` 是唯一有专门 KV 契约的边界

- `src/llama-hetero-route.h` 为这一条边界单独定义了 `llama_hetero_kv_contract`。
- CPU/OpenCL 混合时用 `CPU_OPENCL_ZERO_COPY`。
- QNN/CPU 或 QNN/OpenCL 混合时用 `QNN_RPCMEM` / `qnn-npu-host`。
- `src/llama-context.cpp` 还会在该边界不 zero-copy safe 时直接告警。

### 8. `attn_core=qnn-npu` 还有额外 F32 KV 约束

- `src/llama-context.cpp` 明确告警：当前 `attn_core=qnn-npu` 的实验 shared-KV 路线要求 `type_k=F32` 且 `type_v=F32`，否则不能当作有效 zero-copy KV route。

## 阶段到后端支持矩阵

这张表只回答“这个阶段有没有独立路由接口，以及代码里有没有对应的后端执行路径”，不直接回答跨后端 overhead。

| 阶段 | CPU | OpenCL | qnn-npu | 说明 |
| --- | --- | --- | --- | --- |
| `attn_proj` | 支持 | 支持 | 条件支持 | `qnn-npu` 侧已有 `match_attn_proj_graph()` / `execute_attn_proj()`，但需要 AoT runtime，且 IO 要能匹配 QNN graph 形状与类型。 |
| `attn_core` | 支持 | 支持 | 条件支持 | `qnn-npu` 侧已有 `match_attn_core_graph()` / `execute_attn_core()`；如果想主打 `attn_proj -> attn_core` 无拷贝 KV，还额外需要 F32 KV cache。 |
| `ffn` | 支持 | 支持 | 条件支持 | `qnn-npu` 侧已有 `match_ffn_graph()` / `execute_ffn()`，说明 FFN 阶段本身具备独立 AoT 路径。 |

补充说明：

- `attn_out` 不单列成第三段矩阵主轴，但代码里已支持单独路由；若不单独设置，它默认跟随 `attn_core`。
- `qnn-npu` 的“条件支持”含义是：
  - 要有 `GGML_QNN_AOT_CONFIG`；
  - QNN AoT runtime 必须真能覆盖该 fragment；
  - mixed-stage guard 不能把残图挡回 CPU/JIT fallback；
  - 即使 graph 能执行，跨后端边界也不自动等于无拷贝。

## 关键边界支持矩阵

### 1. `attn_proj -> attn_core`

| 生产端 -> 消费端 | 当前结论 | 是否有专门 zero-copy / shared-host 契约 | 备注 |
| --- | --- | --- | --- |
| 同后端 | 较强静态支持 | 不需要专门契约 | 三段内部不会因为这个边界发生跨后端 copy。 |
| CPU -> OpenCL / OpenCL -> CPU | 较强静态支持 | 有 | 用 `CPU_OPENCL_ZERO_COPY`，前提是 `opencl-host` buffer 可用。 |
| CPU/OpenCL -> qnn-npu | 条件支持 | 有 | 用 `QNN_RPCMEM` / `qnn-npu-host`。若 `attn_core=qnn-npu`，想把它当成有效 zero-copy KV 路线还需要 `-ctk f32 -ctv f32`。 |
| qnn-npu -> CPU/OpenCL | 条件支持 | 有 | 代码层也建了同类 `QNN_RPCMEM` 契约，但仓库现有 decode 证据主要覆盖“非 QNN 产 Q/K/V，再送 QNN attn_core”的方向。 |

这一条边界是当前三段切分里支持最强的一条，因为它不只是 scheduler 层面的“尽量共享”，而是显式建了 KV contract。

### 2. `attn_core(attn_out) -> ffn`

| 生产端 -> 消费端 | 当前结论 | 是否有专门 zero-copy 契约 | 备注 |
| --- | --- | --- | --- |
| 同后端 | 较强静态支持 | 不需要 | 没有跨后端边界。 |
| CPU -> OpenCL / OpenCL -> CPU | 较强静态支持，但可能拷贝 | 没有 | 只能依赖 shared-host compute buffer 与 scheduler 的 buffer compatibility；不是像 KV 那样的专门契约。 |
| CPU/OpenCL -> qnn-npu 或 qnn-npu -> CPU/OpenCL | 条件支持，且可能拷贝 | 没有 | 没有等价于 KV contract 的专门边界协议；是否 direct-bind 取决于实际 buffer layout 和 AoT graph I/O 匹配情况。 |

这一条边界是当前三段切分中更需要 profile 的一条，因为它直接决定了 FFN 异构切分的 runtime overhead 会不会把阶段异构收益抵消掉。

## 组合级支持矩阵

### 图例

- `S`：较强静态支持
- `C`：条件支持，且所有 `qnn-npu` 单元格默认都需要 `GGML_QNN_AOT_CONFIG`
- `K`：`attn_proj -> attn_core` 有专门 KV shared-host / zero-copy 路径
- `F`：若要把 `attn_core=qnn-npu` 当作有效 zero-copy KV route，还需要 `-ctk f32 -ctv f32`
- `B`：`attn_core(attn_out) -> ffn` 这一侧只有 best-effort share，仍可能被 scheduler 插入 copy
- `V`：仓库现有 decode 文档或日志已验证“该组合在 decode `batch=1` 下至少能跑通一次”；这不等价于“所有 shape / ctx / 动态切换下都无拷贝”
- `A`：优先避免，当前静态路径虽在，但两侧混后端过多或证据明显不足

### `attn_core = cpu`

| `attn_proj \\ ffn` | `cpu` | `opencl` | `qnn-npu` |
| --- | --- | --- | --- |
| `cpu` | `S` | `S+B` | `C+B+V` |
| `opencl` | `S+K` | `S+K+B` | `C+K+B` |
| `qnn-npu` | `C+K+V` | `C+K+B` | `C+A` |

解释：

- 纯 `CPU/OpenCL` 家族都属于“较强静态支持”，但只要 `ffn` 与 `attn_core` 分后端，就不能再声称“无拷贝”。
- `attn_proj=qnn-npu, attn_core=cpu, ffn=cpu` 与 `attn_proj=cpu, attn_core=cpu, ffn=qnn-npu` 现在都已有本地 decode 实测，说明 `qnn-npu` 单独承担 `attn_proj` 或 `ffn` 的 AoT stage path 不只是 matcher 存在，而是能在 decode 路径中真正跑通。
- 除了 `attn_proj=qnn-npu, attn_core=cpu, ffn=cpu` 之外，其余 `attn_proj=qnn-npu` 组合目前仍主要停留在静态可达层面，不应当一概读成强支持。
- `qnn-npu -> cpu -> qnn-npu` 这类“两侧都跨 QNN”的路线当前优先避免。

### `attn_core = opencl`

| `attn_proj \\ ffn` | `cpu` | `opencl` | `qnn-npu` |
| --- | --- | --- | --- |
| `cpu` | `S+K+B` | `S+K` | `C+K+B` |
| `opencl` | `S+B` | `S` | `C+B` |
| `qnn-npu` | `C+K+B` | `C+K` | `C+A` |

解释：

- 与上一张表对称：`CPU/OpenCL` 混合仍然属于“较强静态支持”，但 `attn_core -> ffn` 跨后端仍然只是 best-effort share。
- `attn_proj=qnn-npu, attn_core=opencl` 这类路线在代码上有特殊 KV contract，但仍缺 decode 证据。
- `qnn-npu -> opencl -> qnn-npu` 同样应视为高 overhead / 低证据组合。

### `attn_core = qnn-npu`

| `attn_proj \\ ffn` | `cpu` | `opencl` | `qnn-npu` |
| --- | --- | --- | --- |
| `cpu` | `C+F+K+B+V` | `C+F+K+B+V` | `C+F+K` |
| `opencl` | `C+F+K+B+V` | `C+F+K+B+V` | `C+F+K` |
| `qnn-npu` | `C+F+B` | `C+F+B` | `C+F` |

解释：

- 这整张表都不该读成“任意组合已强支持”，因为 `attn_core=qnn-npu` 本身就有 AoT 和 F32 KV 条件。
- 结合仓库既有 host validation 与本文补充日志，`attn_core=qnn-npu` 现在已有四个 decode 已验证单元格：
  - `attn_proj=cpu, attn_core=qnn-npu, ffn=cpu`
  - `attn_proj=cpu, attn_core=qnn-npu, ffn=opencl`
  - `attn_proj=opencl, attn_core=qnn-npu, ffn=cpu`
  - `attn_proj=opencl, attn_core=qnn-npu, ffn=opencl`
- 对 `attn_core=qnn-npu` 来说，`attn_proj -> attn_core` 这一侧是当前支持最强的 mixed boundary，因为这里有专门 KV contract。
- 但 `attn_core(attn_out) -> ffn` 这一侧仍不是专门 zero-copy 协议，所以凡是 `ffn != qnn-npu` 的组合，都仍然要把 `B` 当真。
- `attn_proj=qnn-npu, attn_core=qnn-npu, ffn=qnn-npu` 这类 all-QNN 组合虽然不再有三段之间的跨后端边界，但它仍然是 AoT 路线，不应被误读成“普通 hetero 路由随便切都能等价达到”。

## 2026-03-22 decode 实测补充

这轮补测只回答一个更具体的问题：在 decode `batch=1` 下，`attn_proj / attn_core / ffn` 三段里分别只让一段或两段走 `qnn-npu` 时，是否真的能执行完，以及日志里是否已经出现 scheduler 级显式 tensor copy。

统一配置：

- 设备：`db6c02cf`
- 构建：`./build-npu-opencl.sh build-qnn-shared-host-verify arm64-android-snapdragon-release --without-npu --with-gpu --with-qnn`
- 二进制目录：`/data/local/tmp/acom-stage-matrix-verify`
- 模型：`/data/local/tmp/acom-attn-proj-verify/models/qwen2_0.5b/ggml/weights.gguf`
- 统一 bench 参数：`-r 1 -t 1 -p 0 -n 1 -c 2048 -b 1 -ub 1 -ctk f32 -ctv f32 -ngl 0 --mmap 0`
- 统一环境变量：`GGML_HEXAGON_EXPERIMENTAL=1`、`GGML_HETERO_QNN_SHARED_HOST=1`、`GGML_HETERO_TRACE_SHARE=1`、`GGML_HETERO_PROFILE=1`、`GGML_HETERO_PROFILE_LOG=1`、`GGML_QNN_AOT_TRACE_MATCH=1`

结果汇总：

| 组合 | AoT config | 日志 | 结果 | `tg1` | `ggml_hetero_share` | `ggml_hetero_copy` / `tensor_copy` / `tensor_copy_wait` | 告警/备注 |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| `attn_proj=cpu, attn_core=qnn-npu, attn_out=qnn-npu, ffn=cpu, output=cpu` | `qnn_attn_core_combined.json` | `tmp/stage_matrix_decode_cpu_qnn_cpu_ctx2048_tg1.log` | 跑通 | `19.20 ± 0.00` | 4880 | `0 / 0 / 0` | 有 `mixed-stage` guard；两次出现 unmatched `cache_k_upd-23 -> attn_out-23` 与 `ffn_inp-23` residual fragment |
| `attn_proj=opencl, attn_core=qnn-npu, attn_out=qnn-npu, ffn=opencl, output=cpu` | `qnn_attn_core_combined.json` | `tmp/stage_matrix_decode_opencl_qnn_opencl_ctx2048_tg1.log` | 跑通 | `24.35 ± 0.00` | 2929 | `0 / 0 / 0` | 有 `mixed-stage` guard；两次出现 unmatched `cache_k_upd-23 -> attn_out-23` 与 `ffn_inp-23` residual fragment |
| `attn_proj=qnn-npu, attn_core=cpu, attn_out=cpu, ffn=cpu, output=cpu` | `qnn_attn_proj_combined.json` | `tmp/stage_matrix_decode_qnn_cpu_cpu_ctx2048_tg1.log` | 跑通 | `21.80 ± 0.00` | 4047 | `0 / 0 / 0` | 快速 grep 未见 `mixed-stage` / `unmatched cgraph` / `tensor_copy` 告警 |
| `attn_proj=cpu, attn_core=cpu, attn_out=cpu, ffn=qnn-npu, output=cpu` | `qnn_ffn_combined.json` | `tmp/stage_matrix_decode_cpu_cpu_qnn_ctx2048_tg1.log` | 跑通 | `23.31 ± 0.00` | 3611 | `0 / 0 / 0` | 有 `mixed-stage` guard；快速 grep 未见 unmatched residual fragment |

这些补测目前能支撑的结论是：

- `attn_proj`、`attn_core`、`ffn` 三段都已经至少有一条本地 decode 路线证明“该阶段可单独交给 `qnn-npu` 执行”；
- 在这 4 组具体配置下，日志整体表现为 share-heavy，且没有观察到 `ggml_hetero_copy`、`tensor_copy` 或 `tensor_copy_wait`；
- 但这还不能提升成“跨后端一定无拷贝”的普适结论，因为 `attn_core=qnn-npu` 的两组日志仍有 residual unmatched 片段，说明 mixed-stage AoT route 还没有完全消除尾部碎片与 guard 路径。

补充说明：

- `llama-bench` 结果行里的 `backend=OpenCL,qualcomm` 只说明进程加载并可见 OpenCL 后端，不应将其误读为所有非 OpenCL 组合都实际在 OpenCL 上执行；阶段归属应以 `GGML_HETERO_STAGE_ROUTE` 与 trace/profile 日志为准。
- 不同日志里的 `ggml_hetero_share` 计数不能直接拿来比较“哪条路线更优”，这里只把它用作“是否主要走 share 而不是显式 copy”的证据。

## 已有 decode 证据

当前仓库内，能直接支撑三段组合结论的 decode 证据，主要来自：

- `docs/qnn-attn-core-shared/host-validation-2026-03-22.md`
- 本文补充的 4 个本地 decode 日志：
  - `tmp/stage_matrix_decode_cpu_qnn_cpu_ctx2048_tg1.log`
  - `tmp/stage_matrix_decode_opencl_qnn_opencl_ctx2048_tg1.log`
  - `tmp/stage_matrix_decode_qnn_cpu_cpu_ctx2048_tg1.log`
  - `tmp/stage_matrix_decode_cpu_cpu_qnn_ctx2048_tg1.log`

截至目前，已验证的 decode 路线至少包括六条：

1. `attn_proj=opencl, attn_core=qnn-npu, ffn=cpu`
2. `attn_proj=cpu, attn_core=qnn-npu, ffn=opencl`
3. `attn_proj=cpu, attn_core=qnn-npu, ffn=cpu`
4. `attn_proj=opencl, attn_core=qnn-npu, ffn=opencl`
5. `attn_proj=qnn-npu, attn_core=cpu, ffn=cpu`
6. `attn_proj=cpu, attn_core=cpu, ffn=qnn-npu`

这些证据足以说明：

- `attn_core=qnn-npu` 与相邻 `CPU / OpenCL` 子图之间，shared-host boundary 在 decode 路径上已经能打通；
- `attn_proj -> attn_core` 的 shared KV layout 已经不是主障碍；
- `attn_proj` 与 `ffn` 也都已各自至少有一条“单独交给 `qnn-npu`”的 decode 实测路径；
- 但这些证据仍不足以推出“所有 27 个三元组组合都已稳定支持”，更不足以推出“动态切换时不会触发 KV contract reject 或额外 runtime overhead”。

## 权重驻留矩阵

阶段级异构是否“对称可选”，还受模型权重 buffer type 自动路由影响。

`src/llama-model-loader.cpp` 的当前状态是：

| 阶段 | CPU 权重自动路由 | OpenCL 权重偏好 | QNN host-readable 权重自动路由 |
| --- | --- | --- | --- |
| `attn_proj` | 有 | 有 | 有 |
| `attn_out` | 有 | 有 | 有 |
| `ffn` | 有 | 有 | 有 |
| `output` | 有 | 有 | 无单独 QNN 路由逻辑 |
| `attn_core` | 无对称的独立权重路由逻辑 | 无 | 无 |

这再次说明：阶段级接口虽然是分开的，但工程支持并不是对所有阶段完全对称，`attn_core` 更依赖 AoT graph/runtime 自身，而不是像 `attn_proj/ffn` 那样在 loader 里有明显的阶段型权重驻留策略。

## 为什么“任意切换后端”仍然不是强结论

除了上面的组合矩阵，当前代码还有两点会直接限制“任意切换”：

### 1. 动态 plan 更新会因为 KV contract 不兼容被拒绝

`src/llama-context.cpp` 的 `apply_hetero_plan()` 会在新 plan 的 `attn_proj -> attn_core` KV contract 与当前上下文已分配 contract 不兼容时直接 reject。

这意味着：

- 即使两个路由字符串都“合法”，也不代表可以在同一个已创建 context 上任意切换；
- 当前没有真正意义上的 KV migration，所以动态调度空间仍受上下文创建时的 KV contract 约束。

### 2. scheduler 的共享与拷贝是运行时决定的

即使某个组合在静态上“看起来合理”，最终是 `ggml_hetero_share` 还是 `ggml_hetero_copy`，仍然由：

- tensor 的实际 buffer type
- 目标 backend 是否支持该 buft
- AoT graph 是否能 direct-bind 其输入输出

共同决定。

所以要对“无拷贝”下强结论，必须至少看一次带 trace/profile 的 decode 实测。

## 最小实验补强方案

如果要把上面的“条件支持”升级成强结论，建议优先做最小 decode 验证，而不是先跑 prefill。

### 目标

只回答两个问题：

1. 某个三元组组合在 decode `batch=1` 下是否真的能执行完，而不是只停留在 route-api 层。
2. 关键边界到底走的是 `ggml_hetero_share` 还是 `ggml_hetero_copy`。

### 统一配置

- 先执行 `build-npu-opencl.sh`
- 工具：`llama-bench`
- decode 配置优先保持：
  - `-p 0 -n 1 -b 1 -ub 1 -c 2048`
- 若测试 `attn_core=qnn-npu`，优先使用：
  - `-ctk f32 -ctv f32`
- 推荐同时打开：
  - `GGML_HETERO_TRACE_SHARE=1`
  - `GGML_HETERO_PROFILE=1`
  - `GGML_HETERO_PROFILE_LOG=1`

这样可以直接看到：

- `ggml_hetero_share`
- `ggml_hetero_copy`
- `tensor_copy`
- `tensor_copy_wait`

### 第一批优先补测组合（2026-03-22 已完成）

上面 4 组组合已经按这里的优先级完成，结果见“2026-03-22 decode 实测补充”。这里保留它们，主要是为了说明为什么这 4 组最值得先测。

按 decode 主线和 overhead 价值，当时优先补测的是这四类：

1. `attn_proj=cpu, attn_core=qnn-npu, ffn=cpu`
   - 目的：看 `attn_core=qnn-npu` 在只有一个 mixed boundary 时能否稳定工作。
2. `attn_proj=opencl, attn_core=qnn-npu, ffn=opencl`
   - 目的：看 `CPU` 不参与时，OpenCL 与 `qnn-npu` 的双边界是否仍主要 share 而不是 copy。
3. `attn_proj=qnn-npu, attn_core=cpu, ffn=cpu`
   - 目的：验证 “QNN 只做 `attn_proj`” 是否真有稳定 decode 路径，而不是只有 stage matcher 存在。
4. `attn_proj=cpu, attn_core=cpu, ffn=qnn-npu`
   - 目的：验证 “QNN 只做 `FFN`” 的独立 AoT 路径在 decode 下是否足够稳定。

### 成功/失败判据

- 成功：
  - `llama-bench` 正常返回；
  - 关键边界出现 `ggml_hetero_share`，而不是大面积 `ggml_hetero_copy`；
  - `GGML_HETERO_PROFILE` 中 `tensor_copy` / `tensor_copy_wait` 不是主导项。
- 失败：
  - route 能写但运行时 residual fragment 被 mixed-stage guard 挡回；
  - 边界大量 `tensor_copy`，导致“接口支持”不能转化为低 overhead 的阶段级异构执行；
  - `attn_core=qnn-npu` 在非 F32 KV 下退化，无法满足“无拷贝 KV route”的研究目标。

## 当前最稳妥的表述

截至当前代码状态，更准确的说法是：

- `attn_proj / attn_core / ffn` 三段独立选后端的接口已经具备；
- `CPU` 和 `OpenCL` 之间的三段混合路由属于较强静态支持，但仍不能自动等价成“无拷贝”；
- `qnn-npu` 相关的三段混合路由仍属于条件支持，但现在已经有 6 条 decode 路线证明“三段中的每一段都可以在特定组合下单独交给 `qnn-npu`”；
- 当前真正有专门 zero-copy/shared-host 契约加持的是 `attn_proj -> attn_core` 的 KV 边界；
- `attn_core(attn_out) -> ffn` 仍然是 runtime overhead 风险最大的 mixed boundary 之一；本轮 `ffn=qnn-npu` 的 decode 补测说明它在当前配置下可以做到 share-heavy 且未见显式 tensor copy，但还需要更多 shape / 后端组合来支撑更强结论。
