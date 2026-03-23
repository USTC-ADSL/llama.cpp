# `db6c02cf` Prefill full-vs-split overhead 分解

更新日期：2026-03-23

## 目标

这份记录补齐 `P4-2 Prefill full-vs-split overhead 分解` 的第一版定量证据，回答三个问题：

1. `pp128` 下 full-graph AoT 和 split AoT 的 warm gap，外层 scheduler 看到的差异到底是什么。
2. 这条 gap 里，显式 `tensor_copy` 是否真的是主因。
3. 如果外层 scheduler 只看到很少的 split，为何 split prefill 仍然会稳定慢于 full graph。

## 实验配置

- 设备：`db6c02cf`
- 构建：
  - `./build-npu-opencl.sh build-qnn-prof-db arm64-android-snapdragon-release --without-npu --with-gpu --with-qnn --with-profiling`
- 二进制目录：
  - `/data/local/tmp/acom-stage-profiler-qwen2`
- 模型：
  - `/data/local/tmp/acom-attn-proj-verify/models/qwen2_0.5b/ggml/weights.gguf`
- 统一 bench 参数：
  - `-r 1 -t 1 -p 128 -n 0 -c 2048 -b 128 -ub 128 -ctk f32 -ctv f32 --mmap 0`
- 统一环境变量：
  - `GGML_HEXAGON_EXPERIMENTAL=1`
  - `ADSP_LIBRARY_PATH=.`
  - `LD_LIBRARY_PATH=.`
  - `LLAMA_BENCH_FAST_EXIT=1`
  - `GGML_HETERO_PROFILE=1`
  - `GGML_HETERO_PROFILE_SYNC=1`
  - `GGML_HETERO_PROFILE_FLUSH=1`
  - `GGML_HETERO_PROFILE_CSV=<device csv path>`

full-graph AoT：

- `GGML_QNN_AOT_CONFIG=/data/local/tmp/acom-attn-proj-verify/models/qwen2_0.5b/qnn/config-only128.json`
- `GGML_QNN_AOT_MODEL_DIR=/data/local/tmp/acom-attn-proj-verify/models/qwen2_0.5b/qnn`
- `-ngl 99 -dev qnn-npu`

split AoT：

- `GGML_HETERO_QNN_SHARED_HOST=1`
- `GGML_QNN_AOT_TRACE_MATCH=1`
- `GGML_QNN_AOT_CONFIG=/data/local/tmp/acom-attn-proj-verify/models/qwen2_0.5b/qnn_split_batch128_only.json`
- `GGML_QNN_AOT_MODEL_DIR=/data/local/tmp/acom-attn-proj-verify/models/qwen2_0.5b`
- `GGML_HETERO_STAGE_ROUTE=attn_proj=qnn-npu,attn_core=qnn-npu,attn_out=qnn-npu,ffn=qnn-npu,output=qnn-npu`
- `-ngl 99 -dev qnn-npu`

## 原始产物

本轮新增产物：

- `tmp/p42f/db6c02cf_p42f_fullgraph_pp128_20260323.csv`
- `tmp/p42f/db6c02cf_p42f_fullgraph_pp128_20260323.log`
- `tmp/p42f/db6c02cf_p42f_split_pp128_20260323.csv`
- `tmp/p42f/db6c02cf_p42f_split_pp128_20260323.log`

沿用的已有内部 trace：

- `tmp/qnn_fullgraph_prefill_qnn_ctx2048_pp128_trace_exec.log`
- `tmp/qnn_split_batch128_only_prefill_pp128_trace_exec_after_kcache_fix.log`

汇总表：

- `docs/qnn-attn-core-shared/db6c02cf-prefill-overhead-2026-03-23.csv`

## 口径说明

### 1. 本轮只看 warm `pp128`

本轮的目标不是重测 cold lazy-load，而是回答 warm steady-state 下 split 为什么仍然慢。

对应参考值沿用 2026-03-22 的 warm baseline：

- full graph：`2605.81 tok/s`
- split：`1531.31 tok/s`

本轮 profile 结果非常接近：

- full graph：`2617.88 tok/s`
- split：`1558.00 tok/s`

因此可以直接把本轮 CSV 当作 warm steady-state 的 overhead 分解。

### 2. `pp128` CSV 同样包含 bench 内部的两个 pass

两条路线的 CSV 都出现了两轮 `split_id` 从高值回到 `0`。本轮同样按 `split_id` 回退切分 pass，并只取最后一轮计量 pass：

- full graph：`pass=1`
- split：`pass=1`

用这个口径时：

- full graph：
  - `split_compute = 48.598 ms`
  - `128 / 2617.88 = 48.89 ms`
- split：
  - `split_compute = 81.766 ms`
  - `128 / 1558.00 = 82.16 ms`

两者是对得上的。

## 汇总表

| 模式 | 2026-03-22 warm `pp128` | 2026-03-23 profiled `pp128` | 计量 pass | `tensor_copy` | `split_compute` 数 | `split_compute` 总时长 | CPU | qnn-npu | trace `graph splits` | trace `execute graph` 数 | trace `loaded binary` 数 | `qnn-npu` buffer | `qnn-npu-host` buffer |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full-graph AoT | `2605.81` | `2617.88` | `1` | `0` | `2` | `48.598 ms` | `0.086 ms` | `48.512 ms` | `2` | `2` | `2` | `74.62 MiB` | `2.00 MiB` |
| split batch128-only AoT | `1531.31` | `1558.00` | `1` | `2` 次 / `7168 B` / `2 us` | `4` | `81.766 ms` | `3.446 ms` | `78.320 ms` | `4` | `72` | `72` | `15.75 MiB` | `75.06 MiB` |

## 关键发现

### 1. 外层 scheduler 看到的 split 数其实很少

在 warm `pp128` 的计量 pass 里：

- full graph：
  - `CPU embd`
  - `qnn-npu transformer+lm_head`
- split：
  - `CPU embd`
  - `qnn-npu norm-0 -> l_out-23`
  - `CPU norm`
  - `CPU result_norm -> result_output`

也就是说，从 `GGML_HETERO_PROFILE` 的外层视角看：

- full graph 只有 `2` 个 split；
- split 也只有 `4` 个 split。

因此如果只看 scheduler 外层，你会低估 split prefill 的真实 fragment granularity。

### 2. split warm gap 里，显式 scheduler `tensor_copy` 几乎可以忽略

split 路线只记录到了两次显式 copy：

1. `l_out-23`
2. `output_norm.weight`

总量只有：

- `7168 B`
- `2 us`

而 full-vs-split 的外层 `split_compute` 差距是：

- `81.766 ms - 48.598 ms = 33.168 ms`

所以当前 prefill warm gap 显然 **不是外层 scheduler 显式 memcpy 在主导**。

### 3. gap 的绝大部分落在 split 路线那一个大的 `qnn-npu` compute 区间里

计量 pass 中：

- full graph `qnn-npu split_compute = 48.512 ms`
- split `qnn-npu split_compute = 78.320 ms`

两者相差：

- `29.808 ms`

而外层 CPU 部分只多了：

- `3.446 ms - 0.086 ms = 3.360 ms`

因此外层 profile 看到的 `33.168 ms` gap 中，约：

- `89.9%` 落在 `qnn-npu` compute 区间内部
- `10.1%` 才是 CPU tail

这说明 split prefill 的主 overhead 不是“外层多了几个 CPU 小尾巴”，而是：

- **split AoT 在 qnn backend 内部执行得比 full-graph AoT 明显更贵**

### 4. 这个内部 gap 与 `72` 个 AoT graph launch 是对齐的

已有内部 trace 很清楚：

full graph：

- `execute transformer graph = 1`
- `execute lm_head graph = 1`
- `loaded context binary = 2`
- `graph splits = 2`

split：

- `execute attn_proj graph = 24`
- `execute attn_core graph = 24`
- `execute ffn graph = 24`
- `loaded context binary = 72`
- `graph splits = 4`

因此当前更准确的 prefill 表述应该是：

- scheduler 外层只看到一个大的 `qnn-npu` split；
- 但这个大 split 在 qnn backend 内部其实会展开成 `72` 个 AoT stage graph；
- warm gap 的主要部分正是沉在这个内部 stage-chain 里。

### 5. split prefill 的共享主机内存仍然是重度依赖项

trace 里的 buffer 形态差异非常明显：

full graph：

- `qnn-npu compute buffer size = 74.62 MiB`
- `qnn-npu-host compute buffer size = 2.00 MiB`

split：

- `qnn-npu compute buffer size = 15.75 MiB`
- `qnn-npu-host compute buffer size = 75.06 MiB`

这说明 split prefill 当前的主要运行形态不是“在 NPU 内部持有大块私有工作区”，而是：

- 更依赖 host-visible shared buffer；
- 更容易把阶段边界、KV writeback 和 I/O materialization 变成 runtime 成本。

### 6. `attn_core` 的 KV direct-bind 虽然工作了，但大多数非-KV IO 仍未 direct-bind

从 `tmp/qnn_split_batch128_only_prefill_pp128_trace_exec_after_kcache_fix.log` 的 bind trace 来看：

- `attn_core`：
  - `cache_k=1`、`cache_v=1`：`24 / 24`
  - `out=0`：`24 / 24`
  - `x=0, q=0, k=0, v=0, cache_k=1, cache_v=1, out=0`：`23 / 24`
  - 只有第 `0` 层出现过一次 `x=1`
- `attn_proj`：
  - `x=1, q=0, k=0, v=0`：只有第 `0` 层命中一次

这意味着当前 split prefill 的 shared-host 设计只把 KV 这条边界打通了，远没有把整条 stage chain 的 I/O 都变成 direct-bind。

因此即使外层没有看到大量显式 `tensor_copy`，backend 内部仍然很可能在反复做：

- host-visible input materialization
- intermediate output copyback
- KV 相关 shared-host 管理

## 当前结论

`P4-2` 的第一版结论可以写得比之前更硬一些：

1. split prefill warm gap 现在已经可以被定量拆成：
   - 极小的外层 scheduler 显式 copy
   - 一个很小的 CPU output tail
   - 一个占主导的 qnn backend 内部 stage-chain 成本
2. 因此 split prefill 当前最像主瓶颈的是：
   - `72` 个 AoT graph launch
   - shared-host I/O / KV materialization
   - direct-bind 命中率不足
3. 这进一步说明：
   - prefill 的 runtime overhead 主导项不是“hetero scheduler 外层 copy”
   - 而是 **split AoT 自身在 qnn backend 内部的碎片化执行**

## 对主线的意义

这份结果对当前主线有两个直接帮助：

1. 它把 prefill 的 `1.70x` warm gap 从“现象”推进成了可分解结构：
   - 外层 copy 几乎不是主因；
   - 内部 stage-chain 才是主因。
2. 它也给后续工程方向定了优先级：
   - 如果要继续做 prefill 提升，优先级不应放在再抠 scheduler 外层 copy；
   - 更应该放在：
     - graph coarsening
     - 提高 direct-bind 命中率
     - 降低 qnn backend 内部 stage-chain launch / materialization 成本

## 还缺什么

当前还没有把 prefill gap 完整切成：

- launch
- bind/materialize
- KV writeback

三个精确时间桶，因为 `GGML_HETERO_PROFILE` 只能看到 scheduler 外层，不能直接看到 qnn backend 内部每个 AoT graph 的单独 latency。

所以当前最强结论仍应写成：

- **已经证明 split prefill 的主 gap 位于 qnn backend 内部 stage-chain，而不是外层 scheduler copy**

而不是过度写成：

- “已经精确量化每一个内部子项的 us 占比”

## 下一步

1. 进入 `P4-3 ideal vs actual`：
   - 用现有 stage-profiler 和 `P4-1/P4-2` 结果，把“阶段局部潜在收益”与“真实端到端收益”放到一起
2. 做 `P4-4 微基准与系统级对照`：
   - 用 `hetero-switch-bench` 给 shared-host / memcpy 的量级一个系统外对照
3. 如果继续优化 prefill：
   - 首先尝试更粗粒度 AoT graph，而不是继续增加 split stage 数量
