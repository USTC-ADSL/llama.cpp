# 2026-03-23-032 Qwen3 Last-Layer Attention CPU Tail Fallback Validation

## Target Decode Problem

- 目标问题：
  - 修复 `Qwen3` mixed route
    - `attn_proj=opencl,attn_core=qnn-npu,attn_out=qnn-npu,ffn=opencl,output=cpu`
    在 decode 路径上的最后一层 residual fragment / unmatched 问题。
- 这项工作服务于主线中的 decode-stage runtime correctness：
  - 关注的是 `attn_core(QNN)` 到 `ffn(OpenCL)` 的阶段边界；
  - 不是证明“QNN 一定更快”，而是先证明阶段级 mixed route 能在真实 runtime overhead 下稳定执行，不被最后一层 residual tail 打碎。

## Code Change Under Validation

- 文件：
  - `src/llama-context.cpp`
- 本轮验证的最小补丁有两部分：
  1. 把 `last_layer_attention_cpu_tail_fallback` 从仅 `ffn=cpu` 扩展到：
     - 只要最后一层 attention 当前要走 `qnn-npu`
     - 且下游 `ffn` 明确走非 QNN backend
     - 就把最后一层 attention 整体落到 CPU，避免 `attn_out-tail` / `ffn_inp` 被切成 unmatched residual fragment。
  2. 修复 graph callback lambda 对局部变量的 capture：
     - 显式按值 capture `qnn_gpu_backend`
     - 显式按值 capture `qnn_cpu_backend`
     - 显式按值 capture `hetero_route_uses_opencl`
   - 之前 layer 0 直接变成 `cache_k_upd-0 -> ffn_inp-0` unmatched，根因就是局部变量被 lambda 按引用带出函数，形成悬空引用。

## Build And Device Setup

- 按规范重新构建：
  - `./build-npu-opencl.sh build-qnn-current-verify arm64-android-snapdragon-release --without-npu --with-gpu --with-qnn`
- 设备：
  - `fd8657d6`
- 设备运行目录：
  - `/data/local/tmp/acom-fd-qwen23-verify/bin`
- 模型：
  - `/data/local/tmp/acom-fd-qwen23-verify/models/Qwen3-1.7B-Q4_0/qwen3-1.7b-q4_0.gguf`
- AoT config：
  - `/data/local/tmp/acom-fd-qwen23-verify/models/qwen3_1.7b/models/Qwen3-AoT/qwen3-qnn-full/qnn_attn_core_f16_combined.json`
- 统一环境：
  - `LD_LIBRARY_PATH=.`
  - `ADSP_LIBRARY_PATH=.`
  - `GGML_HEXAGON_EXPERIMENTAL=1`
  - `GGML_HETERO_QNN_SHARED_HOST=1`
  - `LLAMA_BENCH_FAST_EXIT=1`

## Main Results

### 1. `tg1` 已恢复通过，layer 0 不再 unmatched

- 命令口径：
  - `-p 0 -n 1 -b 1 -ub 1 -c 2048 -ctk f16 -ctv f16`
- 结果：
  - `tg1 = 11.06 tok/s`
- 日志：
  - `tmp/qwen3_fd8657d6_tailcpu_fix2_20260323/qwen3_attncore_f16_tg1_tailcpu_clean.log`

对照旧失败日志：

- 旧日志：
  - `tmp/qwen3_fd8657d6_tailcpu_20260323/qwen3_attncore_f16_tg1_tailcpu_clean.log`
- 旧错误：
  - `[aot] unmatched cgraph: n_nodes=18 first=cache_k_upd-0 last=ffn_inp-0`
  - `[aot] rejecting unmatched cgraph before JIT fallback`
  - `test_gen: failed to decode generation batch, res = -3`

因此这一轮已经可以确认：

- layer 0 的崩坏不是 tail fallback 设计本身导致的；
- 而是前一版 patch 的 lambda capture 悬空引用导致的调度错乱；
- capture 修复后，最小 `tg1` decode 已能端到端跑通。

### 2. 最后一层 tail fallback 已真实命中，并且不再生成 unmatched fragment

- trace 日志：
  - `tmp/qwen3_fd8657d6_tailcpu_fix2_20260323/qwen3_attncore_f16_tg1_tailcpu_trace_full.log`
- 关键证据：
  - layer 27 出现：
    - `reason=hetero-last-layer-attn-cpu-tail-fallback`
  - 对应的 tail 节点也被整体压到 CPU：
    - `cache_k_upd-27`
    - `cache_v_upd-27`
    - `kq`
    - `kqv`
    - `attn_out-27`
    - `attn_out-tail-27`
    - `ffn_inp-27`
- 同一轮 trace/full clean 日志中都没有再出现：
  - `unmatched cgraph`
  - `rejecting unmatched cgraph before JIT fallback`
  - `ggml-opencl.cpp:10031`

这说明 stopgap 的作用机制已经成立：

- 不是去改 `attn_core` matcher；
- 而是在最后一层把 QNN attention 整体提前收束到 CPU；
- 以避免 `QNN attention tail -> OpenCL FFN` 这一残缺边界在 runtime 上碎成 residual fragment。

### 3. `tg64` 也已通过，说明不是只修好单 token smoke

- 命令口径：
  - `-p 0 -n 64 -b 1 -ub 1 -c 2048 -ctk f16 -ctv f16`
- 结果：
  - `tg64 = 19.72 tok/s`
- 日志：
  - `tmp/qwen3_fd8657d6_tailcpu_fix2_20260323/qwen3_attncore_f16_tg64_tailcpu_clean.log`

这说明：

- 修复并不只是在 `tg1` 上侥幸躲过 warmup；
- decode 更长一点的 steady-state 路线也已经稳定通过。

### 4. `pp128 + n1` 仍可运行，batch=128 路线未被这轮补丁误伤

- 命令口径：
  - `-p 128 -n 1 -b 128 -ub 128 -c 2048 -ctk f16 -ctv f16`
- 结果：
  - `pp128 = 579.42 tok/s`
  - `tg1 = 10.76 tok/s`
- 日志：
  - `tmp/qwen3_fd8657d6_tailcpu_fix2_20260323/qwen3_attncore_f16_pp128_n1_tailcpu_clean.log`

这一项的主要价值是回归稳定性，而不是性能结论：

- 它说明 batch=128 路线没有被最后层 tail fix 打坏；
- 但由于当前设备口径和 bootstrap correction 影响，不能直接把这一轮数值拿去和之前的高水位结果做强性能结论。

## Interpretation For The Storyline

- 本轮已经支持一个强 correctness 结论：
  - `Qwen3` 在
    - `attn_proj=opencl`
    - `attn_core=qnn-npu`
    - `attn_out=qnn-npu`
    - `ffn=opencl`
    - `output=cpu`
    的 mixed route 下，
    最后一层 residual tail unmatched 问题已经被最小 scheduler patch 挡住。
- 这非常符合当前主线：
  - 真正卡住阶段级异构执行收益释放的，不是“某个阶段理论上不能上 NPU”；
  - 而是 runtime boundary 是否会在最后一层 residual / tail / fallback 处碎裂。

更具体地说：

- 这轮收益不是减少算子时间本身；
- 而是减少了
  - unmatched split
  - reject-before-JIT fallback
  - 以及后续 OpenCL assert
  这些 runtime correctness overhead。

## Important Caveat About Performance Numbers

- `tg1` 单 token 口径会触发 `AoT bootstrap CPU correction`；
- trace 中出现的大量 `bootstrap-qnn-cpu` 属于 correction pass，而不是 steady-state mixed route 自己静默退回 CPU；
- 因此：
  - `tg1` 更适合做“功能/边界 correctness smoke”
  - 不适合直接代表 steady-state QNN/OpenCL mixed route 的真实 decode 吞吐

所以这轮最稳妥的表述应是：

- `tg1` / `tg64` / `pp128+n1` 已证明路线稳定；
- 但若要讨论“QNN 为什么没有达到预期 tok/s”，需要把 bootstrap correction 与 steady-state token 分开量化。

## What Is Still Missing

若要把本轮从“correctness fix 已闭环”继续推进到“更强性能解释”，还缺：

1. 单独量化 bootstrap correction 对 `tg1` 的影响
   - 否则无法把 `tg1` 直接解释成 steady-state route 性能。
2. 对更长 decode 长度做同口径复测
   - 例如 `tg128` 或更长，以观察 tail fix 后的稳态趋势。
3. 若要继续解释 mixed route 性能偏低的原因，还需要：
   - 分离 `OpenCL -> CPU(last layer attention)` 边界成本
   - 分离 bootstrap correction 成本
   - 再对照纯 OpenCL / 纯 QNN / 其他 mixed route

## Bottom Line

- 这轮最小 patch 已经完成了它应完成的目标：
  - 修掉了 `Qwen3` mixed route 中最后一层 tail unmatched 的 correctness blocker。
- 从主线角度，这个结果的价值在于：
  - 它再次说明 runtime boundary contract 才是阶段级异构调度能否落地的关键瓶颈；
  - 只要最后一层 residual tail 边界没处理好，阶段级 mixed route 的潜在收益就会在 runtime 中被直接吞掉。
