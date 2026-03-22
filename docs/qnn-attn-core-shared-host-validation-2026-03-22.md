# QNN `attn_core` Shared-Host Validation

更新日期：2026-03-22

## 目标

本次验证只回答两个 decode 路径问题：

1. 在 `attn_proj / attn_core / ffn` 三段切分下，`attn_core=qnn-npu` 是否已经可以和相邻 `CPU / OpenCL` 子图通过 shared host buffer 共享边界张量，从而避免额外 tensor copy。
2. `attn_proj -> attn_core` 之间的 KV cache layout 是否已经与 CPU / OpenCL 路径对齐，使 `AttentionProjection` 与 `attn_core` 落在不同后端时不必再做一轮 KV cache 迁移。

本次不试图证明“任意路由组合都已经完全无 overhead”，也不把结论扩展到 prefill 端到端收益。

## 推荐的三段切分

当前最适合继续沿着研究主线推进的 decode 子图切分仍然是：

- `attn_proj`
- `attn_core`
- `ffn`

原因如下：

- `attn_proj` 主要是 Q/K/V 生产，是 `AttentionProjection` 与 KV cache 写入边界最清晰的位置。
- `attn_core` 聚合了 KV 读写、mask/bias materialization 和主要 attention core 计算，最适合单独观察 KV layout 统一是否真的降低了跨后端 overhead。
- `ffn` 已经有独立 AoT 路径，且 `attn_core -> ffn` 边界天然就是 decode 中最值得优化的一条 activation 边界。

如果再往更细粒度拆，会更快碰到切分/同步/搬移开销主导的问题，不符合当前 `AGENTS.md` 中的 stage-centric 优先级。

## 当前代码状态

为打通上述三段切分，本轮相关代码已经具备以下能力：

- `src/llama-graph.cpp` 将 `cache_k_upd / cache_v_upd / cache_k_read / cache_v_read` 显式暴露成 attention 子图边界，允许 `attn_proj` 写入后的 cache 视图直接喂给 `attn_core`。
- `src/llama-kv-cache.cpp` / `src/llama-kv-cache.h` 增加了基于外部 cache tensor 构造 KV view 的接口，避免重新回到旧的内部 cache tensor 上。
- `src/llama-context.cpp` 允许在 `GGML_HETERO_QNN_SHARED_HOST=1` 时让 `CPU / OpenCL / qnn-npu` 共用 `qnn-npu-host` 作为 decode 计算边界空间。
- `ggml/src/ggml-qnn/qnn/buffer.hpp` 让同一块 shared host 内存可以按 view 形状复用 QNN HTP tensor view，而不是把 view 缓存死绑到单个 `ggml_tensor *` 指针。
- `ggml/src/ggml-qnn/qnn/aot.cpp` 补齐了 `attn_core` decode `batch=1` 路径的 rank-collapsed mask/KV 处理、shared KV 读回以及转置 `V` writeback。

## 验证配置

设备：`db6c02cf`

模型：`qwen2_0.5b`

统一参数：

- `llama-bench`
- `decode`
- `-p 0 -n 1 -b 1 -ub 1 -c 2048`
- `-ctk f32 -ctv f32`
- `-t 1 -ngl 0 -dev GPUOpenCL`
- `GGML_HETERO_QNN_SHARED_HOST=1`
- `GGML_QNN_AOT_CONFIG=/data/local/tmp/acom-attn-proj-verify/models/qwen2_0.5b/qnn_attn_core_combined.json`

## 已验证 route

### 1. `attn_proj=opencl, attn_core=qnn-npu, ffn=cpu`

日志：

- `tmp/attncore_decode_opencl_qnn_cpu_ctx2048_rerun.log`

关键证据：

- layer 0 的 `x / qcur / cache_k / cache_v / out / kcur / vcur` 全部 direct-bind 到 `qnn-npu-host`
- 日志中没有出现：
  - `failed to materialize attn bias`
  - `failed to update shared KV cache`
- 最终 `llama-bench` 结果正常产出：`tg1 = 13.09 tok/s`

### 2. `attn_proj=cpu, attn_core=qnn-npu, ffn=opencl`

日志：

- `tmp/attncore_decode_cpu_qnn_opencl_ctx2048.log`

关键证据：

- layer 0 的 `x / qcur / cache_k / cache_v / out / kcur / vcur` 同样全部 direct-bind 到 `qnn-npu-host`
- `ggml_hetero_share` 明确显示：
  - `CPU -> qnn-npu` 的 `Qcur / Kcur / Vcur`
  - `OpenCL -> qnn-npu` 的 `cache_k_read / cache_v_read / l_out`
  - `qnn-npu -> OpenCL` 的 `ffn_inp`
  都走的是 `qnn-npu-host`
- 最终 `llama-bench` 结果正常产出：`tg1 = 14.13 tok/s`

## 可以得出的结论

截至当前提交前状态，可以给出下面这个范围受限但已经比较强的结论：

- 在 decode `batch=1`、`ctx=2048`、`attn_core=qnn-npu` 的 AoT 路径下，`attn_core` 与相邻的 `CPU` 或 `OpenCL` 子图之间，边界 activation 与 shared KV tensor 已经可以通过 `qnn-npu-host` 共享空间，而不是强制额外 copy 到私有后端 buffer。
- `attn_proj -> attn_core` 的 shared KV layout 现在已经能在 `CPU / OpenCL / QNN AoT` 三者之间互通，因此把 `AttentionProjection` 和 `attn_core` 放到不同后端时，不再必然触发 KV cache copy。
- 这正是当前三段切分 `attn_proj / attn_core / ffn` 值得继续沿着 decode 主线推进的原因：它已经把“相邻异构子图边界上的张量 copy”从主要障碍，缩小成了更局部的剩余图匹配问题。

## 仍然不能过度声称的部分

下面这些结论现在还不能直接下：

- 不能说“所有任意后端组合都已完全无 copy”。
- 不能说“attention 全路径都已完整落进 QNN AoT”。
- 不能说“系统已经获得端到端收益”，因为这里只证明了潜在 overhead 被显著压低，还没有把动态调度端到端 runtime overhead 全量计入。

## 当前剩余问题

### 1. 最后一层仍有 residual unmatched cgraph

两条成功日志里都还能看到：

- `unmatched cgraph: n_nodes=17 first=cache_k_upd-23 last=attn_out-23`
- `unmatched cgraph: n_nodes=1 first=ffn_inp-23 last=ffn_inp-23`

这说明当前 AoT `attn_core` 路径对大多数层已打通，但最后一层 residual attention 相关残图还没有完全吞进去。

### 2. 本轮只重新验证了 decode `batch=1`

当前 AoT 配置 `tmp/device_configs/qnn_attn_core_combined.json` 中同时包含：

- `batch_size=1` 的 24 张 `attn_core` 图
- `batch_size=128` 的 24 张 `attn_core` 图

也就是说，配置层面已经满足“decode 用 batch=1、prefill 用 batch=128”的分离要求。

但是本轮端到端运行只重新验证了 decode `batch=1`，还没有把 `batch=128` 的 prefill 路径重新跑一遍，所以 prefill 侧只能说“图已经准备好”，不能说“运行已经完成验证”。

### 3. static `qnn-npu` baseline 仍有独立问题

在第二台设备的单后端 baseline 尝试中，`qnn-npu` 仍然会在 scheduler 预分配阶段因为 `SET_ROWS` 与 `cache_k_upd-0` 的 buffer placement 冲突而 abort。

这属于 shared-host 三段切分之外的另一条静态单后端问题，不影响本次“跨后端边界共享空间”已经成立的判断，但会影响后续做静态 NPU baseline。

## 对研究主线的意义

这轮结果对当前叙事最有价值的地方，不是“系统已经赢了”，而是：

- 证明 decode 子图确实可以做 stage-level heterogeneous split；
- 证明 `attn_proj / attn_core / ffn` 这组三段切分的边界已经足够清晰，且 runtime overhead 可以被 shared memory 机制显著压低；
- 把剩余瓶颈进一步收敛到了：
  - 最后一层 residual attention 子图匹配
  - prefill `batch=128` 路径复验
  - 静态 `qnn-npu` baseline 的单后端调度问题

因此，下一阶段最小可验证方案应继续围绕这三点推进，而不是再往更细的算子级切分扩散。
