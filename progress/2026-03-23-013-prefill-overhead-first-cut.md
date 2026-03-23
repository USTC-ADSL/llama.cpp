# 任务 013：补齐主设备 `db6c02cf` 的 prefill full-vs-split overhead 第一刀

日期：2026-03-23

## 背景与目标

在 `P4-1 Decode 边界 overhead 分解` 完成后，当前非功耗主线里最关键的下一个缺口就是：

- **split prefill 的 warm gap 到底主要是外层 scheduler copy，还是 qnn backend 内部 stage-chain 成本**

之前已经知道：

- split prefill 真实执行了；
- warm `pp128` 下 full graph `2605.81`，split `1531.31`，约慢 `1.70x`；

但还没有 event-level 证据把这个 gap 继续拆开。

## 执行内容

### 1. 在主设备上补抓 `pp128` 的 event-level CSV

full-graph AoT：

- `GGML_QNN_AOT_CONFIG=/data/local/tmp/acom-attn-proj-verify/models/qwen2_0.5b/qnn/config-only128.json`
- `GGML_QNN_AOT_MODEL_DIR=/data/local/tmp/acom-attn-proj-verify/models/qwen2_0.5b/qnn`
- `-ngl 99 -dev qnn-npu`

split AoT：

- `GGML_QNN_AOT_CONFIG=/data/local/tmp/acom-attn-proj-verify/models/qwen2_0.5b/qnn_split_batch128_only.json`
- `GGML_QNN_AOT_MODEL_DIR=/data/local/tmp/acom-attn-proj-verify/models/qwen2_0.5b`
- `GGML_HETERO_STAGE_ROUTE=attn_proj=qnn-npu,attn_core=qnn-npu,attn_out=qnn-npu,ffn=qnn-npu,output=qnn-npu`
- `GGML_HETERO_QNN_SHARED_HOST=1`

统一参数：

- `-r 1 -t 1 -p 128 -n 0 -c 2048 -b 128 -ub 128 -ctk f32 -ctv f32 --mmap 0`
- `GGML_HETERO_PROFILE=1`
- `GGML_HETERO_PROFILE_SYNC=1`
- `GGML_HETERO_PROFILE_FLUSH=1`
- `LLAMA_BENCH_FAST_EXIT=1`

### 2. 复用已有内部 trace 做第二层分解

为了避免把结论错误写成“prefill 只有 4 个 split”，本次同时复用了已有 trace：

- `tmp/qnn_fullgraph_prefill_qnn_ctx2048_pp128_trace_exec.log`
- `tmp/qnn_split_batch128_only_prefill_pp128_trace_exec_after_kcache_fix.log`

因为 `GGML_HETERO_PROFILE` 只能看到 scheduler 外层，而 split prefill 的 `72` 个 AoT graph launch 是藏在 qnn backend 内部的。

## 关键证据

### 1. warm `pp128` 的本轮 profile 值与已有 warm baseline 对得上

本轮：

- full graph：`2617.88 tok/s`
- split：`1558.00 tok/s`

与已有 warm baseline：

- full graph：`2605.81`
- split：`1531.31`

基本一致，因此可以把本轮 CSV 视为 warm steady-state overhead 分解。

### 2. 外层 scheduler 看到的 split 很少

计量 pass：

- full graph：`2` 个 split
- split：`4` 个 split

所以如果只看 scheduler 外层，很容易误判：

- “split prefill 的问题不在 graph granularity”

但这其实是不完整的。

### 3. split 路线的显式 scheduler `tensor_copy` 极小

计量 pass 里只记录到了两次 copy：

- `l_out-23`
- `output_norm.weight`

总计：

- `7168 B`
- `2 us`

因此 split warm gap 当前不能再解释成：

- “主要慢在 scheduler 外层 activation memcpy”

### 4. 外层 gap 主要落在 `qnn-npu` compute 区间

计量 pass：

- full graph：
  - `split_compute_total = 48.598 ms`
  - `qnn-npu = 48.512 ms`
  - `CPU = 0.086 ms`
- split：
  - `split_compute_total = 81.766 ms`
  - `qnn-npu = 78.320 ms`
  - `CPU = 3.446 ms`

因此 gap：

- 总 gap：`33.168 ms`
- 其中 `qnn-npu` 内部 gap：`29.808 ms`
- CPU tail gap：`3.360 ms`

这意味着：

- gap 的 `~90%` 落在 split 路线那个大的 `qnn-npu` compute 区间里；
- CPU 小尾巴只占次要部分。

### 5. 这个内部 gap 与 `72` 个 AoT graph launch 一致

full graph trace：

- `execute transformer graph = 1`
- `execute lm_head graph = 1`
- `loaded context binary = 2`
- `graph splits = 2`

split trace：

- `execute attn_proj graph = 24`
- `execute attn_core graph = 24`
- `execute ffn graph = 24`
- `loaded context binary = 72`
- `graph splits = 4`

所以当前更准确的口径是：

- scheduler 外层只看到一个大的 qnn split；
- 但这个 qnn split 内部其实展开成了 `72` 个 AoT stage graph；
- split warm gap 的主因就在这里。

### 6. split prefill 当前仍高度依赖 `qnn-npu-host`

trace buffer 形态：

- full graph：
  - `qnn-npu = 74.62 MiB`
  - `qnn-npu-host = 2.00 MiB`
- split：
  - `qnn-npu = 15.75 MiB`
  - `qnn-npu-host = 75.06 MiB`

再结合 bind trace：

- `attn_core` 的 `cache_k/cache_v`：`24/24` direct-bind
- 但 `x/q/k/v/out` 基本都没有 direct-bind

这说明当前 split prefill 的主形态不是“内部纯 NPU 连续执行”，而是：

- 更依赖 shared-host
- 更依赖阶段边界 materialization

## 当前结论

这轮 `P4-2` 第一刀已经足够支撑下面这句更强的结论：

- **split prefill warm gap 的主因不是 scheduler 外层 copy，而是 qnn backend 内部的 stage-chain fragmentation 与 shared-host materialization 成本**

因此当前 prefill 的优化方向不应优先放在：

- 再抠外层 scheduler copy

而应优先放在：

- graph coarsening
- direct-bind 命中率
- qnn backend 内部 stage launch / materialization

## 下一步

1. 继续做 `P4-3 ideal vs actual`
2. 用 `hetero-switch-bench` 做 `P4-4` 对照
3. 如果继续做 prefill 侧工程优化，优先尝试更粗粒度 AoT graph
