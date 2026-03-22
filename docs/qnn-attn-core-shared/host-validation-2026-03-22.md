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

### 2. `batch=128` prefill 当前仍未跑通端到端

当前 AoT 配置 `tmp/device_configs/qnn_attn_core_combined.json` 中同时包含：

- `batch_size=1` 的 24 张 `attn_core` 图
- `batch_size=128` 的 24 张 `attn_core` 图

也就是说，配置层面已经满足“decode 用 batch=1、prefill 用 batch=128”的分离要求。

但是这次已经实际补跑了两条 `pp128` route：

- `attn_proj=opencl, attn_core=qnn-npu, attn_out=qnn-npu, ffn=cpu, output=cpu`
- `attn_proj=cpu, attn_core=qnn-npu, attn_out=qnn-npu, ffn=opencl, output=cpu`

对应日志：

- `tmp/attncore_prefill_opencl_qnn_cpu_ctx2048_pp128.log`
- `tmp/attncore_prefill_cpu_qnn_opencl_ctx2048_pp128.log`

两条路线的失败模式完全一致：

- `[aot] mixed-stage AoT route does not explicitly request qnn-cpu; keep transformer residual fragments on plain CPU to avoid extra qnn-cpu splits.`
- `[aot] unmatched cgraph: n_nodes=18 first=cache_k_upd-0 last=ffn_inp-0`
- `[aot] rejecting unmatched cgraph before JIT fallback`
- `test_prompt: failed to decode prompt batch, res = -3`

所以当前更准确的表述已经更新为：

- `batch=128` split prefill 在 `db6c02cf` 上已经真实跑通，不再卡在第一层 residual/AoT matching
- full-graph AoT 和 split AoT 现在已经有了真正的 apples-to-apples `pp128` 对比
- 两者仍然存在明显速度差，但这次可以把差异拆成 cold lazy-load 开销和 warm steady-state runtime overhead 两部分来解释

### 2.1 split `batch=128` prefill 现在已经真实执行了 `24 x 3` stage graphs

这轮在 runtime 里继续补了两类修复：

- 保留之前的 lazy-load，避免一次性注册全部 split contexts 时的 `5005`
- 把 merged prompt `qnn-npu` split 重新分解回每层的 `attn_proj / attn_core / ffn`
- 修正 `attn_core` shared KV writeback 对 `Kcur` 2D prompt layout 的错误假设

验证过程分成两步：

- `tmp/qnn_split_batch128_only_prefill_pp128_trace_exec_after_fragment_fix.log`
  - 这里已经能看到：
    - `[aot] decompose transformer cgraph into 24 fragment chains`
    - `[aot] execute attn_proj graph ... layer=0 tokens=128`
    - `[aot] execute attn_core graph ... layer=0 tokens=128`
  - 但第一层 `attn_core` 还会在 shared KV writeback 处失败
- `tmp/qnn_split_batch128_only_prefill_pp128_trace_exec_after_kcache_fix.log`
  - 这里已经连续打印：
    - `24 x execute attn_proj graph`
    - `24 x execute attn_core graph`
    - `24 x execute ffn graph`
  - 并且 benchmark 成功结束

因此当前对 split prefill 的最强事实已经变成：

- prompt-time `batch=128` 的 `attn_proj / attn_core / ffn` 子图现在确实被匹配并执行了
- 之前“split prefill 其实只碰到了最后层 FFN tail”这一结论已经过时

### 2.2 QNN AoT full graph vs split subgraph 的 `prefill` 速度差异依然很大，而且这次可以解释

同一台 `db6c02cf`、同一构建、同一设置：

- `-p 128 -n 0 -c 2048 -b 128 -ub 128`
- `-ngl 99 -dev qnn-npu -t 1`
- `-ctk f32 -ctv f32`

full-graph AoT：

- `GGML_QNN_AOT_CONFIG=/data/local/tmp/acom-attn-proj-verify/models/qwen2_0.5b/qnn/config-only128.json`
- `GGML_QNN_AOT_MODEL_DIR=/data/local/tmp/acom-attn-proj-verify/models/qwen2_0.5b/qnn`

split AoT：

- `GGML_QNN_AOT_CONFIG=/data/local/tmp/acom-attn-proj-verify/models/qwen2_0.5b/qnn_split_batch128_only.json`
- `GGML_QNN_AOT_MODEL_DIR=/data/local/tmp/acom-attn-proj-verify/models/qwen2_0.5b`
- `GGML_HETERO_STAGE_ROUTE=attn_proj=qnn-npu,attn_core=qnn-npu,attn_out=qnn-npu,ffn=qnn-npu,output=qnn-npu`

先看 cold `--no-warmup` 结果：

- full-graph：
  - `tmp/qnn_fullgraph_prefill_qnn_ctx2048_pp128_r3_after_fragment_fix.log`
  - `pp128 = 2054.86 ± 892.91 tok/s`
- split：
  - `tmp/qnn_split_batch128_only_prefill_qnn_ctx2048_pp128_r3_after_fragment_fix.log`
  - `pp128 = 1099.35 ± 761.50 tok/s`

这里 split 大约慢：

- `2054.86 / 1099.35 = 1.87x`
- 吞吐下降约 `46.5%`

再看 warm（保留 warmup，尽量把 lazy-init 从测量里剥掉）：

- full-graph：
  - `tmp/qnn_fullgraph_prefill_qnn_ctx2048_pp128_r3_warm_after_fragment_fix.log`
  - `pp128 = 2605.81 ± 94.02 tok/s`
- split：
  - `tmp/qnn_split_batch128_only_prefill_qnn_ctx2048_pp128_r3_warm_after_fragment_fix.log`
  - `pp128 = 1531.31 ± 10.85 tok/s`

这时 split 仍然慢：

- `2605.81 / 1531.31 = 1.70x`
- 吞吐下降约 `41.2%`

所以现在可以更明确地回答用户的问题：

- **是的，QNN AoT 下完整 full-graph prefill 和 split prefill 之间仍然存在很明显的速度差**
- **而且这不是只有 cold start 或旧的 `5005` 初始化异常才会出现**
- **即使把 lazy-init 的影响尽量剥掉，split prefill 仍然有稳定的 end-to-end slowdown**

### 2.3 当前差异的主要来源是 runtime overhead，而不是“split 其实没有跑到”

full-graph trace 的关键证据：

- 日志：`tmp/qnn_fullgraph_prefill_qnn_ctx2048_pp128_trace_exec.log`
- 执行：
  - `[aot] execute transformer graph batch_128 tokens=128 batch=128`
  - `[aot] execute lm_head graph batch_128 tokens=1 batch=128`
- reserve / buffer 形态：
  - `graph splits = 2`
  - `qnn-npu compute buffer size = 74.62 MiB`
  - `qnn-npu-host compute buffer size = 2.00 MiB`
- cold path只加载：
  - `transformer`
  - `lm_head`

split trace 的关键证据：

- 日志：`tmp/qnn_split_batch128_only_prefill_pp128_trace_exec_after_kcache_fix.log`
- 执行：
  - `24 x execute attn_proj graph`
  - `24 x execute attn_core graph`
  - `24 x execute ffn graph`
- cold path加载：
  - `72` 个 `batch=128` context binary
- reserve / buffer 形态：
  - `graph splits = 4`
  - `qnn-npu compute buffer size = 15.75 MiB`
  - `qnn-npu-host compute buffer size = 75.06 MiB`

更关键的是 `attn_core` 的 bind trace：

- 24 层里，shared KV 可以 direct-bind：
  - `cache_k=1`
  - `cache_v=1`
- 但大多数非-KV IO 都没有 direct-bind：
  - `x=0`
  - `q=0`
  - `k=0`
  - `v=0`
  - `out=0`

结合 `ggml/src/ggml-qnn/qnn/aot.cpp` 的实现，这意味着 split 路径虽然把算子留在了 NPU/AoT 上，但 runtime 里仍然要额外承担：

- `72` 次子图启动，而不是一次 `transformer` 启动
- 频繁的 graph internal buffer copy / output copyback
- 每层 `attn_core` 的 shared-host KV writeback/materialization
- 更重的 `qnn-npu-host` buffer footprint

因此当前更合理的表述是：

- **已经可以把 full-vs-split 的差异解释成真实的端到端 runtime overhead**
- **而不是再把它归因成“split prefill 根本没跑到”**

### 2.4 这类差异部分可修，但不能指望只靠 matcher 小修小补就完全消失

从现有证据看，当前 gap 至少分成两部分：

- **cold path gap**：
  - split 第一次触发时要 lazy-load `72` 个 binary
  - 这部分可以通过 prewarm / eager preload / prefill-decode runtime 分离来明显缩小
- **warm steady-state gap**：
  - 这部分仍然有 `1.70x` slowdown
  - 说明即使 runtime 已经“热起来”，图粒度过细带来的 launch/sync/copy/KV 管理开销依然存在

因此修复优先级应该是：

- **优先减少 stage boundary 的 runtime 开销**
  - 提高 direct-bind 命中率
  - 减少 fragment 之间的 copy fallback
  - 尽量把更多 KV 路径保留在 device-local contract 下
- **或者改回更粗粒度的图**
  - 例如 `attention` family
  - 或者直接 `transformer`
  - 这样更有机会保留 decode-centric 的 stage hook，同时减少 `72` 次 graph launch

这里还剩一个需要额外标记的边角：

- split trace 里的最后一层仍然打印：
  - `execute ffn graph ffn_layer_23_batch_128 layer=23 tokens=1 batch=128`

这说明当前 `match_ffn_graph()` 是从 `ffn_inp-23` 推出 `n_tokens`，而最终 prompt tail 的 `ffn_inp-23` 只保留了 last-token 视图。

我的判断是：

- 这更像是 final prompt graph/logit tail 的图形状，而不是“split prefill 又没跑通”
- 它会影响 per-stage token accounting
- 但它不是当前 full-vs-split 性能差异的主因，因为即使带着这个“layer 23 only 1 token”的尾部，split 仍然整体慢于 full-graph

### 3. static `qnn-npu` baseline 已从 reserve-abort 推进到可测，但仍非正式基线

在第二台设备 `fd8657d6` 的单后端 baseline 补跑中，static `qnn-npu` decode 已经不再在 scheduler 预分配阶段因为 `SET_ROWS` 与 `cache_k_upd-0` 的 buffer placement 冲突而 abort。

关键证据：

- `tmp/fd8657d6_qnn_npu_decode_tg1_verbose_after_kvhost.log` 明确打印：
  - `static qnn-npu KV cache uses qnn-npu-host`
  - 24 层 KV cache 全部放在 `qnn-npu-host`
  - `qnn-npu-host KV buffer size = 24.00 MiB`
- `tmp/fd8657d6_qnn_npu_decode_tg128_r1_fast_exit.log` 已经产出：
  - `tg128 = 10.83 ± 0.00`

这说明 static `qnn-npu` baseline 现在至少已经进入“可测”状态。

但这里仍然不能直接把它当成理想单后端 NPU 对照：

- 当前 static 路径通过 `qnn-npu-host` 让 `SET_ROWS/GET_ROWS` 落在 host-visible buffer 上，本身就会引入额外 runtime overhead
- `src/llama-context.cpp` 的去重补丁已经清掉了重复 `qnn-npu` free 这条症状，但 crash buffer 说明更深层的 abort 来自 `libQnnHtpPrepare.so` 在 `__cxa_finalize` 阶段的 heap corruption；见 `tmp/fd8657d6_crash_logcat_after_ctxdedup.log`
- 因此现在更准确的表述应是：**single-backend qnn-npu decode 存在可运行路径，但它给出的 `tg128` 仍然是一个受 host-visible KV 管理影响、且当前依赖 bench fast-exit workaround 才能干净收尾的 baseline**

补齐 AoT 产物之后，`fd8657d6` 上的 AoT decode 状态也更清楚了：

- `qnn_attn_core_batch1`
- `qnn_attn_core_batch128`

这两组目录已经补推到设备，所以旧的：

- `failed to mmap ... attn_core_layer_0.bin, errno=2`

不再是当前 blocker。

但我在本地复现的最新 AoT decode 结果仍然说明这条路径不稳定：

- 无显式 route：
  - `tmp/fd8657d6_qnn_aot_decode_reserve_abort_after_artifacts.log`
  - reserve abort 在：
    - `pre-allocated tensor (blk.0.attn_norm.weight) in a buffer (qnn-npu) that cannot run the operation (NONE)`
- 加显式 stage route：
  - `tmp/fd8657d6_qnn_aot_decode_with_route_reserve_abort_after_artifacts.log`
  - reserve abort 在：
    - `pre-allocated tensor (output_norm.weight) in a buffer (qnn-npu) that cannot run the operation (NONE)`

因此第二台设备现在可以给出的更稳妥结论是：

- **static `qnn-npu` baseline 已经可测**
- **AoT decode 的缺文件问题已解掉**
- **但 AoT decode 仍然会在 scheduler reserve / param placement 这类更早阶段 abort，尚不能作为稳定的第二设备 AoT 基线**

## 对研究主线的意义

这轮结果对当前叙事最有价值的地方，不是“系统已经赢了”，而是：

- 证明 decode 子图确实可以做 stage-level heterogeneous split；
- 证明 `attn_proj / attn_core / ffn` 这组三段切分的边界已经足够清晰，而且现在已经能在 prompt `batch=128` 下真实执行；
- 同时也定量暴露了：即使全部 stage 都放回 `qnn-npu`，split 仍然会因为 graph launch / copy fallback / shared-host KV 管理而比 full-graph 慢很多；
- 把剩余瓶颈进一步收敛到了：
  - split prefill 的 runtime overhead 降低
  - 最后一层 FFN prompt-tail `tokens=1` 的 stage accounting
  - 静态 `qnn-npu` baseline 的 host-visible KV overhead
  - 第二台设备 AoT decode 的 reserve / param placement abort
  - `libQnnHtpPrepare.so` exit-time finalizer crash

因此，下一阶段最小可验证方案应继续围绕这些瓶颈推进，而不是再往更细的算子级切分扩散。
