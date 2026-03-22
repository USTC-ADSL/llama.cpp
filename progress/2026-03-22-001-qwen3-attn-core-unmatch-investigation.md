# 任务 001：调查 `Qwen3` 适配是否导致 `Qwen2 attn_core` unmatched

日期：2026-03-22

## 背景与目标

用户补充指出：此前为了适配 `Qwen3-1.7B`，可能修改过 `attn_core` 相关逻辑；而 `Qwen3` 与 `Qwen2` 的 `attn_core` 计算图并不完全相同，因此需要确认当前 `Qwen2` decode 中出现的 unmatched residual 是否可能由这类适配引入。

本任务只回答两个问题：

1. `Qwen3` 相关改动是否直接修改了当前 `attn_core` matcher / execute 核心逻辑。
2. 当前 `Qwen2` unmatched 更像是“模型图命名差异”还是“mixed-stage residual 被切碎”。

## 执行内容

检查了以下证据：

- `git log -- ... aot.cpp aot.hpp llama-context.cpp llama-graph.cpp llama-kv-cache.cpp`
- `git show 53f6ef1d9 -- ggml/src/ggml-qnn/qnn/aot.cpp src/llama-context.cpp`
- `git show 3b27272a7 -- ggml/src/ggml-qnn/qnn/aot.cpp src/llama-graph.cpp src/llama-kv-cache.cpp`
- `src/models/qwen2.cpp`
- `src/models/qwen3.cpp`
- `ggml/src/ggml-qnn/qnn/aot.cpp`
- `ggml/src/ggml-qnn/qnn/backend-ops.cpp`
- `docs/qnn-attn-core-shared/host-validation-2026-03-22.md`
- `docs/qnn-attn-core-shared/decode-stage-backend-support-matrix-2026-03-22.md`

## 关键证据

### 1. `Qwen3` 适配提交存在，但没有直接改当前 `attn_core` matcher 主体

提交：

- `53f6ef1d9 qnn: stabilize qwen3 aot integration`

该提交对 `ggml/src/ggml-qnn/qnn/aot.cpp` 和 `src/llama-context.cpp` 的主要影响是：

- 增加 `Qcur_normed-` / `Kcur_normed-`
- 增加 `self_kq_mask_swa_cnv`
- 放宽 transformer / lm_head 输出候选识别

未看到它直接修改当前 `match_attn_core_graph()` 的核心匹配条件，也未看到它把 `Qwen2` 使用的 `Qcur/Kcur/Vcur/attn_out/ffn_inp` 名称改成另一套。

### 2. `Qwen2` 与 `Qwen3` 的 attention 子图命名确实不同

`Qwen2`：

- `Qcur`
- `Kcur`
- `Vcur`
- `attn_out`
- `ffn_inp`

`Qwen3`：

- 既有 `Qcur/Kcur/Vcur`
- 也有 `Qcur_normed/Kcur_normed`

因此 `Qwen3` 支持确实要求扩充 matcher / stage-name 识别范围，但这类扩充本身不会改变 `Qwen2` 现有名字的语义。

### 3. 当前 `Qwen2` unmatched 形态更像 mixed-stage residual 被切碎

现有 `Qwen2` decode 日志中的 unmatched 形态是：

- `n_nodes=17 first=cache_k_upd-23 last=attn_out-23`
- `n_nodes=1 first=ffn_inp-23 last=ffn_inp-23`

这说明最后一层在 `attn_out` 与 `ffn_inp` 之间被切成了两个 residual 片段，而不是整个 `attn_core -> attn_out -> ffn_inp` 作为一个完整子图进入 AoT。

同时，运行时还有明确日志：

- `mixed-stage AoT route does not explicitly request qnn-cpu; keep transformer residual fragments on plain CPU to avoid extra qnn-cpu splits.`

这与“最后一层 residual 被 guard 留在 plain CPU，导致片段化 unmatched”更加一致。

### 4. 更接近当前问题的改动是 shared-host `attn_core` 边界重构

提交：

- `3b27272a7 Enable shared-host QNN attn_core boundaries`

该提交直接改动了：

- `cache_k_upd / cache_v_upd / cache_k_read / cache_v_read` 显式边界
- `llama_kv_cache::get_k/get_v` 的外部 cache view 路径
- `execute_attn_core()` 的 shared-KV derive / writeback / alias 处理

这些都比 `Qwen3` 名字适配更接近当前 `cache_k_upd-23 -> attn_out-23` 的 unmatched 形态。

## 当前结论

当前证据支持以下判断：

- `Qwen3` 适配“有可能”改变 AoT 路由覆盖面，但**不太像当前 `Qwen2` unmatched 的直接主因**。
- 当前 `Qwen2` unmatched 更像是：
  - mixed-stage residual guard
  - `attn_out -> ffn_inp` 尾部 residual 分裂
  - shared-host `attn_core` 边界重构后的最后层片段化
  共同导致的结果。

因此，后续稳定性排查应优先围绕：

- 为什么最后一层 `attn_out-23` 和 `ffn_inp-23` 被拆开
- 为什么这部分 residual 未被 `attn_core` / `attn_out` AoT 路径吞掉
- mixed-stage guard 是否过早把尾部 fragment 留给 CPU

而不是优先怀疑 `Qwen3` 命名适配本身。

## 下一步

1. 更新主线工作计划文档，使其与新版 `AGENTS.md` 的 `Prefill/Decode` 主线一致。
2. 把“最后层 `attn_out-23` / `ffn_inp-23` residual 分裂”列为最高优先级稳定性问题。
3. 在后续稳定性记录里专门跟踪：
   - tail residual 是否仍拆成两段
   - mixed-stage guard 是否仍保留 plain CPU residual
   - 修复后是否能消除 `unmatched cgraph` 日志
