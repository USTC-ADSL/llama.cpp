# 任务 009：确认 split prefill 尾部 `FFN tokens=1` 的语义

日期：2026-03-22

## 背景与目标

当前 `Prefill` 主线里还有一个高优先级疑点：

- split `batch128-only` prompt trace 的最后一层会打印：
  - `execute ffn graph ffn_layer_23_batch_128 layer=23 tokens=1 batch=128`

需要确认这到底意味着：

1. split prefill 在最后一层又退化错了。
2. 还是 llama.cpp prompt eval 本来就只需要最后一个输出 token，因此最后一层 `FFN` 合法地缩成了 output tail。

## 执行内容

检查了三类证据：

- trace / support 日志：
  - `tmp/qnn_split_batch128_only_prefill_pp128_trace_exec_after_kcache_fix.log`
  - `tmp/qnn_split_batch128_only_prefill_pp128_support.log`
- `FFN` matcher 与 fragment 切分实现：
  - `ggml/src/ggml-qnn/qnn/aot.cpp`
- llama.cpp prompt 输出与 `n_outputs` 语义：
  - `src/llama-batch.cpp`
  - `src/llama-context.cpp`

## 关键证据

### 1. 最后一层之前的 stage 仍然是完整 `128` token prompt

同一条 trace 中，layer 23 的前两个 stage 明确仍是：

- `execute attn_proj graph ... layer=23 tokens=128 batch=128`
- `execute attn_core graph ... layer=23 tokens=128 batch=128`

只有最后的：

- `execute ffn graph ... layer=23 tokens=1 batch=128`

发生缩小。

这说明问题不是“整条 split prompt 路线在最后一层都塌成了 decode”，而是更局部的：

- **只有最终 `FFN` fragment 变成了单输出 token 视图**

### 2. `match_ffn_graph()` 的 token 数直接来自 `ffn_inp-*`

`ggml/src/ggml-qnn/qnn/aot.cpp` 中：

- `match_ffn_graph()` 先找 `ffn_inp-*`
- 然后用：
  - `result.n_tokens = static_cast<size_t>(result.embd->ne[1]);`

因此日志里的 `tokens=1` 不是 QNN runtime 自己猜的，而是：

- **当前 FFN fragment 输入张量本身就是 `1` token 视图**

### 3. 普通 prompt eval 默认只请求最后一个输出 token

`src/llama-batch.cpp` 中，若 `batch.logits` 为空且不是 embedding 模式：

- `output.resize(batch.n_tokens, false);`
- `output[output.size() - 1] = true;`

也就是：

- **默认只把最后一个 prompt token 标记为输出**

`src/llama-context.cpp` 随后会根据 `ubatch.output[i]` 统计：

- `n_outputs_new += (int32_t) (ubatch.output[i] != 0);`

也就是：

- **图构建时的 `n_outputs` 会真实收缩到最后输出 token 数**

### 4. trace 里也明确出现了 `n_outputs = 1`

`tmp/qnn_split_batch128_only_prefill_pp128_trace_exec_after_kcache_fix.log` 中直接打印：

- `sched_reserve: worst-case: n_tokens = 128, n_seqs = 1, n_outputs = 1`

support log 里也重复出现：

- `graph_reserve: reserving a graph for ubatch with n_tokens = 128, n_outputs = 1`

同时日志尾部仍能看到：

- `result_norm`
- `result_output`

这些输出 tail 节点。

因此更合理的解释是：

- **最后一层 `FFN` 正在服务“只为最后一个 token 产出 logits”这条 output tail**

### 5. 这在语义上是成立的

对 causal transformer 的 prompt eval 来说：

- 前面各层仍必须处理整段 prompt，以便构造正确的 KV 和记住最后 token 的上下文；
- 但到了最后一层，若只需要最后一个 token 的 logits，那么更早 prompt token 的最终 `FFN` 输出不会再被后续消费。

所以最后一层 `FFN` 收缩成 `tokens=1` 的最强解释是：

- **合法的 prompt output-tail 视图**
- 不是 split prefill correctness 回退

## 当前结论

当前可以把这个问题收口成一个更强结论：

1. split prefill 的 `FFN layer 23 tokens=1` 不是“split 没真正跑到”的新证据。
2. 它来自 llama.cpp prompt eval 默认只请求最后一个输出 token，导致 `n_outputs=1`，最后一层 `FFN` fragment 相应收缩成 output tail。
3. 这不会推翻当前主线结论：
   - split prefill 已真实执行
   - full-vs-split gap 主要仍是 runtime overhead
4. 它真正影响的是：
   - **per-stage accounting 口径**
   - 不能把最后一层 `FFN` 简单当成一个完整 `128` token stage 去累计

## 下一步

1. 在主线文档里把这个语义澄清补进去，避免后续误把它当成 correctness regression。
2. 做 `Decode` 分阶段 profiling 时，明确区分：
   - “完整 stage latency”
   - “output tail / residual tail” 这类只覆盖最后输出 token 的 fragment
3. 后续若需要更精确的 `Prefill` per-stage accounting，再单独给 output-tail 视图加标签，而不是把它并回普通 `FFN` 主体。
