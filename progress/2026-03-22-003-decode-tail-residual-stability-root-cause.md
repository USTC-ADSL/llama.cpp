# 任务 003：收口 decode tail residual unmatched 的根因

日期：2026-03-22

## 背景与目标

当前最高优先级稳定性问题是 decode 尾部 residual 在 mixed-stage `attn_core=qnn-npu` 路线下仍出现 unmatched：

- `n_nodes=17 first=cache_k_upd-23 last=attn_out-23`
- `n_nodes=1 first=ffn_inp-23 last=ffn_inp-23`

本任务要回答四个问题：

1. 这是不是 `Qwen3` 适配直接把 `Qwen2 attn_core` matcher 打坏了。
2. 真正的 root cause 更接近 matcher、stage pinning 还是 scheduler fragment 切分。
3. 当前 stopgap 是否已经足够保证 route 可跑通。
4. 下一步最小可验证动作应该是什么。

## 执行内容

检查了以下代码和证据：

- `ggml/src/ggml-qnn/qnn/aot.cpp`
- `ggml/src/ggml-qnn/qnn/aot.hpp`
- `ggml/src/ggml-qnn/qnn/backend-ops.cpp`
- `src/llama-context.cpp`
- `src/llama-hetero-route.h`
- `tmp/stage_matrix_decode_cpu_qnn_cpu_ctx2048_tg1.log`
- `tmp/stage_matrix_decode_opencl_qnn_opencl_ctx2048_tg1.log`
- `tmp/attncore_decode_cpu_qnn_opencl_ctx2048.log`
- `tmp/attncore_decode_opencl_qnn_cpu_ctx2048_rerun.log`

并额外做了一个编译层 sanity check：

- 执行了 `./build-npu-opencl.sh build-qnn-current-verify arm64-android-snapdragon-release --without-npu --with-gpu --with-qnn`
- 在中断完整 `ninja` 之前，已实际编译通过本轮相关改动文件：
  - `ggml/src/ggml-qnn/qnn/aot.cpp`
  - `ggml/src/ggml-qnn/qnn/backend-ops.cpp`
  - `src/llama-context.cpp`
  - `src/llama-kv-cache.cpp`
  - `tools/llama-bench/llama-bench.cpp`

## 关键证据

### 1. 当前 unmatched 不是 `Qwen3` 命名适配直接导致的

上一条记录已经确认：

- `Qwen3` 适配主要扩展了 `Qcur_normed/Kcur_normed`、`self_kq_mask_swa_cnv` 等名字识别
- 未直接改坏当前 `Qwen2` 使用的 `Qcur/Kcur/Vcur/attn_out/ffn_inp` 主路径

因此这次问题应继续看 fragment 边界，而不是回到模型命名差异本身。

### 2. 最后一层 residual 的真实形态是“两个 qnn 小碎片”，不是一个完整 `attn_core` 子图

多份 decode 日志都稳定出现：

- `[aot-match] attn_core reject: x=<null> ... out=<null> ... layer=23 n_nodes=17`
- `[aot] unmatched cgraph: n_nodes=17 first=cache_k_upd-23 last=attn_out-23`
- `[aot] unmatched cgraph: n_nodes=1 first=ffn_inp-23 last=ffn_inp-23`

最关键的是第一条 reject 信息里：

- `x=<null>`
- `out=<null>`

这说明 scheduler 交给 qnn backend 的 17 节点残图只覆盖到了：

- `cache_k/cache_v` 更新
- attention core 计算
- `attn_out`

但没有把 residual 输入 `x` 和最终边界 `ffn_inp` 一起带进来。

换句话说，当前交到 qnn backend 的最后层 tail fragment，天然就不是现有 `match_attn_core_graph()` 期待的那种：

- 现有 AoT `attn_core` 图更像 `attn_core + attn_out + residual add -> ffn_inp`
- 实际 scheduler 在最后层切出来的是：
  - `cache_k_upd -> attn_out`
  - 单独的 `ffn_inp`

### 3. 直接把 matcher 从 `ffn_inp` 放宽到 `attn_out` 并不稳

代码上可以看到当前 `execute_attn_core()` 强依赖：

- `config.x_name`
- `config.out_name`
- `match.embd`
- `match.out`

而 `attn_core` 图配置里也保留了 `x` 输入。

这更像是在执行：

- attention core
- output projection
- residual add

如果把 matcher 粗暴放宽到 “只要到 `attn_out` 就算匹配”，很可能把一个本应写到 `ffn_inp` 的 AoT 输出错误地绑定到 `attn_out` 张量上，语义风险太高。

因此这一步不能靠放松 matcher 草率解决。

### 4. 当前 route 的可运行性已经由 plain CPU fallback 保住

日志同时稳定出现：

- `[aot] cpu fallback for unmatched residual cgraph: n_nodes=17 first=cache_k_upd-23 last=attn_out-23`
- `[aot] cpu fallback for unmatched residual cgraph: n_nodes=1 first=ffn_inp-23 last=ffn_inp-23`

说明当前 `should_cpu_fallback_unmatched_aot_cgraph()` 已经把这些尾部 residual 统一留在 plain CPU，而没有掉回更不可控的 QNN JIT 小图路径。

这就是当前 decode mixed route 仍能稳定产出 benchmark 结果的直接原因。

### 5. 现阶段更可信的 root cause 是“图边界不对齐”

综合代码和日志，当前更合理的根因判断是：

- `Qwen2` 模型图本身在 layer tail 处有清晰的 `attn_out -> ffn_inp` 边界
- 现有 AoT `attn_core` family 期望吞到 `ffn_inp`
- mixed-stage scheduler 在最后层把它切成了：
  - `attn_out` attention residual fragment
  - 单节点 `ffn_inp`
- 所以最后层会稳定出现 `attn_core reject`

这更像：

- AoT graph coverage boundary
- stage pinning / scheduler fragment boundary

三者没有完全对齐，而不是某个模型适配提交单点打坏。

## 当前结论

当前可以形成一个比之前更强的结论：

- `Qwen3` 适配不是当前 `Qwen2` decode tail unmatched 的直接主因。
- 当前 decode tail unmatched 的核心原因是：
  - `attn_core` AoT 图期待的输出边界在 `ffn_inp`
  - mixed-stage scheduler 交给 qnn backend 的最后层尾部残图只到 `attn_out`
  - `ffn_inp` 又被拆成单节点 fragment
- 现有 plain CPU fallback 已经把这类 residual 从“不稳定的 QNN JIT fallback”收口到“可解释的 CPU stopgap”。

因此，这一阶段最重要的稳定性目标已经从：

- “为什么会 unmatched”

收敛为：

- “如何把这类 tail residual 继续统一留在 CPU，并量化它的 runtime overhead”

而不是贸然去修改 `attn_core` matcher 语义。

## 下一步

1. 继续保持当前 unmatched residual 统一走 plain CPU 的保守路径，不引入语义风险更高的 matcher 放宽。
2. 开始主设备 `db6c02cf` 的单后端 baseline 采集：
   - decode：`CPU(1c/2c) / GPUOpenCL / qnn-npu`，至少 `tg1` 与 `tg128`
   - prefill：`CPU / GPUOpenCL / qnn-npu`，至少 `pp128 / pp256 / pp512`
3. 在后续 overhead 阶段把这类 tail residual CPU fallback 明确计入：
   - 它说明存在潜在收益
   - 但也说明当前系统端到端仍被最后层 runtime overhead 限制
