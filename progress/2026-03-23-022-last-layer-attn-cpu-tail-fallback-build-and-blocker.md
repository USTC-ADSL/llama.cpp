# 2026-03-23-022 最后一层 attention CPU tail fallback、本地构建与设备阻塞

## 任务

继续处理 decode mixed-stage 路线里最后一层 residual / fragment 问题：

- 目标问题：`attn_core=qnn-npu`、`ffn=cpu` 路线中，最后一层仍会出现
  - `unmatched cgraph: n_nodes=18 first=cache_k_upd-23 last=attn_out-tail-23`
  - `unmatched cgraph: n_nodes=1 first=ffn_inp-23 last=ffn_inp-23`
- 约束：优先做最小补丁，不直接改大范围 AoT matcher 语义，也不引入新的 tail-specific AoT 二进制依赖。

## 根因收敛

这轮进一步确认了两点：

1. 当前 `attn_core` AoT 图仍是“常规层”形态
   - 它消费 `x/q/k/v/cache/attn_bias`
   - 输出 `out`
   - 从 `x` 这个输入存在可以推断，当前 AoT `attn_core` 图包含 residual add 边界，而不是只到 `attn_out`

2. 最后一层 decode 图是另一种 tail 形态
   - `Qwen2/Qwen3` 在最后一层插入了 `GET_ROWS`
   - 也就是从常规的 `attn_out + x -> ffn_inp`
   - 变成了 `get_rows(attn_out) + get_rows(x) -> ffn_inp`

因此，当前残留问题并不只是 matcher “少认一个名字”，而是：

- mixed-stage scheduler 还在尝试把最后一层 attention 相关节点交给 QNN
- 但这组 tail 节点不再符合现有 `attn_core` AoT 二进制的边界假设
- 结果就是最后一层被切成 residual fragments，再落入 unmatched CPU fallback

## 本次实现

修改文件：

- `src/llama-context.cpp`

新增一个局部调度保护：

- 当同时满足以下条件时，把**最后一层 attention 相关节点**直接钉到 CPU：
  - `qnn_aot_enabled`
  - `hetero_stage_enabled`
  - 当前层是最后一层
  - 当前节点属于 `attn_proj / attn_core / attn_out`
  - `ffn` 明确路由到 CPU
  - `attn_proj / attn_core / attn_out` 中至少有一个明确路由到 QNN AoT backend

实现意图：

- 不再让最后一层 attention tail 继续进入“QNN split 后 unmatched 再 CPU fallback”的路径
- 直接在 scheduler 决策点把这部分收口到 CPU
- 保持 AoT matcher / execute 语义不变

这属于 stopgap，不是最终完美修复，但它满足当前主线里的三个优先级：

- `Decode-centric`
- `Minimal patching`
- `Overhead-conscious`

## 本地构建结果

按仓库要求执行：

- `./build-npu-opencl.sh build-qnn-current-verify arm64-android-snapdragon-release --without-npu --with-gpu --with-qnn`

结果：

- 构建成功
- `libllama.so`、`llama-bench`、`libggml-qnn.so` 等产物均重新生成
- 本轮新增的 `llama-context.cpp` warning 已顺手收口，不再有该处的 sign-compare 警告

## 设备验证阻塞

尝试检查设备状态：

- `adb -s db6c02cf get-state`
- `adb -s fd8657d6 get-state`

结果均为：

- `device not found`

因此当前**不能**给出新的真机结论，也不能宣称本次 stopgap 已经在设备上完全消除了最后一层 unmatched。

按照仓库约束，这里应停止设备侧动作，等待设备恢复在线后再复测。

## 当前判断

### 如果这次 stopgap 在设备上成立

预期收益是：

- 最后一层 `qnn unmatched residual fragment` 消失
- 不再出现
  - `cache_k_upd-23 -> attn_out-tail-23` 的 QNN unmatched split
  - `ffn_inp-23` 的单节点 QNN unmatched split
- 路线纯度会更清晰
- runtime overhead 解释会更干净，因为不再是“先切给 QNN，再 fallback 回 CPU”

代价是：

- 最后一层 attention 不再走 QNN
- 对 mixed route 的理想吞吐会有轻微损失
- 但这是用一层的静态让步，换掉一组运行时 fragment / fallback 开销

### 如果这次 stopgap 仍不能完全收口

那就说明问题已经超出“调度层最后一层收口”本身，后续要么：

- 增加 tail-specific AoT 图
- 要么重构 `attn_core` AoT 二进制边界

这两条都已经属于更大改动，不能再算最小修复。

## 对主线故事的影响评估

即使当前最后一层 residual / fragment 还没有完全根治，它对研究主线的影响也是**有限且可解释的**：

1. 它不会推翻 `decode stage heterogeneity`
   - Attention / FFN 在不同后端上的差异仍然存在

2. 它不会推翻 `power-tunable space`
   - CPU / GPU / NPU 的不同工作点仍然可以被静态和分阶段实验刻画

3. 它不会推翻“阶段级调度可行”
   - 它只说明当前 prototype 在最后一层 tail 边界上还有 runtime limitation

4. 它反而强化了 “runtime overhead / fragment boundary 是关键瓶颈” 这条主线
   - 当前问题不是算力不足
   - 而是 tail graph 边界与 AoT coverage 不完全对齐
   - 这是非常典型的 runtime overhead / execution-boundary limitation

因此，如果最后仍保留这个限制，正确的论文/文档表述应是：

- 当前系统证明了存在潜在阶段级收益
- 但实际系统收益仍明显受限于最后一层 tail boundary、后端切换和 residual fragment 等 runtime overhead

而不是：

- 声称所有 mixed decode 路线都已完全 clean

## 下一步

设备恢复后，优先做一条真机复测：

- 设备：`db6c02cf` 或 `fd8657d6`
- 路线：`attn_proj=cpu,attn_core=qnn-npu,attn_out=qnn-npu,ffn=cpu,output=cpu`
- AoT 配置：`qnn_attn_core_combined.json`

重点观测：

- `unmatched cgraph` 是否消失
- `OpenCL residual split` 是否仍保持为 0
- mixed route tok/s 是否出现明显回退

成功意味着：

- 当前 stopgap 足以把最后一层 tail residual 从“不干净的 QNN fragment”收口到“显式 CPU stage”

失败意味着：

- 需要决定是否投入更大改动去做 tail-specific AoT coverage
