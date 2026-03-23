# `db6c02cf` Decode Route Purity Revalidation

更新日期：2026-03-23

## 目标

验证 `Qwen2-0.5B` decode `tg1` 路线中，`AoT bootstrap CPU correction` 是否已经真正变成 CPU-only，而不再因为 scheduler 仍持有 `OpenCL` backend 而泄漏成第二张大 `OpenCL` 图。

这项验证直接服务于 decode 主线里的两个问题：

1. `force_cpu_graph` 是否真的只保留 CPU。
2. 当前 mixed decode 路线里剩余的 `OpenCL` split，到底来自 bootstrap correction，还是来自 steady-state mixed graph 本身。

## 配置

- 设备：`db6c02cf`
- 模型：`/data/local/tmp/acom-attn-proj-verify/models/qwen2_0.5b/ggml/weights.gguf`
- AoT 配置：`/data/local/tmp/acom-attn-proj-verify/models/qwen2_0.5b/qnn_attn_core_combined.json`
- 路线：`attn_proj=cpu,attn_core=qnn-npu,attn_out=qnn-npu,ffn=cpu,output=cpu`
- KV layout：`qnn`
- 命令口径：`--no-warmup -p 0 -n 1 -b 1 -ub 1 -c 2048 -t 1 -ngl 0 -dev GPUOpenCL`
- 关键环境变量：
  - `GGML_HETERO_QNN_SHARED_HOST=1`
  - `GGML_QNN_AOT_TRACE_ASSIGN=1`
  - `GGML_HETERO_PROFILE=1`
  - `GGML_HETERO_PROFILE_SYNC=1`
  - `GGML_HETERO_PROFILE_FLUSH=1`
  - `LLAMA_GRAPH_REUSE_DISABLE=1`

原始产物：

- `tmp/p17/purity-cpu-qnn-cpu-noreuse.log`
- `tmp/p17/purity-cpu-qnn-cpu-noreuse.csv`

## 代码修复

修复点在 [src/llama-context.cpp](/mnt/sda1/pzw/HeteroCompute/llama.cpp-acom/src/llama-context.cpp)：

- `AoT bootstrap CPU correction` 不再只靠 `GGML_QNN_DISABLE_BACKEND=1`。
- 在 correction graph 期间，运行时会临时把 `sched` 替换成 CPU-only scheduler。
- 这个临时 scheduler 会一直保留到输出读取完成并 `synchronize()`，然后再恢复 steady-state 主 scheduler。

因此，这次修复解决的是 scheduler backend 集合问题，而不是单纯的 stage pin 问题。

## 结果

### 1. bootstrap correction 已经是 CPU-only

`tmp/p17/purity-cpu-qnn-cpu-noreuse.log` 中，bootstrap pass 的 `aot-assign` 已经全部变成：

- `reason=bootstrap-cpu backend=CPU supported=1`

尾部可以直接看到：

- `cache_k_upd-23`
- `cache_v_upd-23`
- `kq`
- `kqv`
- `attn_out-23`
- `ffn_inp-23`
- `result_norm`
- `result_output`

全部都被 pin 到 `CPU`。

### 2. CSV 不再出现第二张大 OpenCL correction graph

`tmp/p17/purity-cpu-qnn-cpu-noreuse.csv` 的 backend 计数为：

- `CPU = 52`
- `qnn-npu = 50`
- `OpenCL = 2`

这里的 `OpenCL = 2` 仅对应一组 `split_enqueue + split_compute` 事件，而不再是之前那种 correction pass 中 `CPU 122 / OpenCL 121` 的大图泄漏。

### 3. 剩余 OpenCL 只来自 intended mixed graph 的一个小 residual split

CSV 中唯一的 `OpenCL` split 是：

- `split_id = 48`
- `node_start = 929`
- `node_end = 931`
- `latency_us = 1541`

最后一个 correction pass 则已经收敛为单个 CPU split：

- `split_id = 0`
- `backend = CPU`
- `latency_us = 14346`

因此可以明确区分：

- bootstrap correction 的 `OpenCL` 泄漏已被修复；
- 当前 remaining `OpenCL` residual 属于第一张 intended mixed graph，而不是第二张 correction graph。

## 判断

### 强结论

- `force_cpu_graph` 的原始问题已经收口：`AoT bootstrap CPU correction` 现在确实是 CPU-only。
- 这意味着此前 `tg1` purity 分析里那个“第二张大 OpenCL correction graph”不再成立。
- `P1` 里的主要 blocker 已从“bootstrap 机制错误”降级为“steady-state mixed graph 仍有一个很小的 residual split”。

### 保守结论

- 当前还不能把 decode route purity 表述为“完全纯净”。
- 更准确的说法是：`bootstrap correction purity` 已闭环，但 steady-state mixed graph 仍残留 `1` 个很小的 `OpenCL` split。

## 对主线的意义

- 本次结果直接服务于 decode-first 主线，因为它把 runtime overhead 的解释从“第二张错误 correction graph”剥离掉了。
- 之后如果 mixed decode 仍然慢，就应该优先解释：
  - steady-state split fragmentation
  - 小 residual split
  - CPU output tail
  - QNN stage-chain enqueue / sync

而不是继续把主要责任归因到 bootstrap correction。
