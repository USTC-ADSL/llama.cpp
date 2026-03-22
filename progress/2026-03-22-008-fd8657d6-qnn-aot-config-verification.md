# 任务 008：复核 `fd8657d6` 上 `Qwen2/Qwen3` 的 QNN AoT 实验类别

日期：2026-03-22

## 背景与目标

用户指出，第二设备上的 `qnn-npu` 结果明显过慢：

- `Qwen2` 全图 QNN AoT 不应该只有 `~323 tok/s`
- `Qwen3` 全图 QNN AoT 也不应该只有 `~66 tok/s`

这项任务需要先回答一个更基础的问题：

1. 之前记录下来的 `qnn-npu` 数字，究竟是不是 full-graph AoT。
2. 如果不是，`Qwen2` 的 full-graph AoT 真实量级是多少。
3. `Qwen3` 为什么还不能在第二设备上做同类复核。

## 执行内容

### 1. 复查已有 bench 日志

重点检查了以下日志：

- `tmp/fd8657d6_qwen2_qnn_prefill_pp128_pp256_pp512_warm_20260323.log`
- `tmp/fd8657d6_qwen3_1p7b_qnn_prefill_pp64_pp128_pp256_warm_20260323.log`
- `tmp/fd8657d6_qwen2_qnn_decode_tg1_tg128_warm_20260323.log`

这些日志都只有普通 `llama-bench` 输出，没有体现：

- `GGML_QNN_AOT_CONFIG`
- `GGML_QNN_AOT_MODEL_DIR`

因此它们对应的是：

- **static non-AoT `qnn-npu`**

而不是：

- **QNN AoT full-graph**

### 2. 在第二设备上重新验证 `Qwen2` full-graph AoT

复核时显式使用：

- `GGML_QNN_AOT_CONFIG=/data/local/tmp/acom-attn-proj-verify/models/qwen2_0.5b/qnn/config.json`
- `GGML_QNN_AOT_MODEL_DIR=/data/local/tmp/acom-attn-proj-verify/models/qwen2_0.5b/qnn`
- `GGML_HEXAGON_EXPERIMENTAL=1`
- `LLAMA_BENCH_FAST_EXIT=1`
- `taskset 80 -t 1 -ngl 99 -dev qnn-npu --mmap 0`

并记录了：

- `pp128`
- `pp256`

结果已同步到：

- `docs/qnn-attn-core-shared/fd8657d6-qwen2-aot-fullgraph-prefill-2026-03-22.csv`

### 3. 检查第二设备上的 `Qwen3` AoT 产物

设备侧实际检查显示：

- `/data/local/tmp/acom-attn-proj-verify/models/qwen2_0.5b/qnn/` 下存在：
  - `config.json`
  - `qwen2_0.5b_0.bin`
  - `lm_head.bin`
- 当前没有发现 `Qwen3` 对应的：
  - full-graph `config.json`
  - `lm_head.bin`
  - graph `.bin`

因此第二设备上当前还不具备 `Qwen3` full-graph AoT 复核条件。

## 关键证据

### 1. 先前被误读的 `qnn-npu` 数字其实是 static non-AoT

已有日志中：

- `Qwen2 pp128 = 323.28`
- `Qwen2 pp256 = 325.31`
- `Qwen3 pp128 = 66.02`
- `Qwen3 pp256 = 65.93`

这些值可以作为第二设备 static baseline 使用，但不能被写成 full-graph AoT 吞吐。

### 2. `Qwen2` full-graph AoT 的真实吞吐是 `4k tok/s` 量级

重新验证后得到：

- `pp128 = 4280.48 tok/s`
- `pp256 = 4565.30 tok/s`

对应日志：

- `tmp/fd8657d6_qwen2_qnn_aot_fullgraph_prefill_pp128_20260322.log`
- `tmp/fd8657d6_qwen2_qnn_aot_fullgraph_prefill_pp256_20260322.log`

这说明用户对“这组数不应该这么慢”的判断是正确的，真正的问题是实验类别混淆。

### 3. `Qwen3` 当前缺 second-device AoT 产物

因此当前不能得出：

- “`Qwen3` second-device full-graph AoT 很慢”

当前只能得出：

- “`Qwen3` second-device static non-AoT `qnn-npu` baseline 很慢”

## 当前结论

当前可以形成三条更严格的结论：

1. `fd8657d6` 上先前记录的 `Qwen2/Qwen3 qnn-npu` 数字属于 static non-AoT baseline，不是 full-graph AoT。
2. `Qwen2` 在第二设备上的 full-graph AoT prefill 已重新验证到 `4280.48 / 4565.30 tok/s`，明显高于 static non-AoT `323.28 / 325.31 tok/s`。
3. `Qwen3` 在第二设备上尚缺 full-graph AoT 产物，因此当前不能给出对应 full-graph AoT 吞吐，只能保留 static baseline 结论。

这对主线的意义是：

- 第二设备上的 `qnn-npu` 数据今后必须显式区分：
  - static non-AoT
  - QNN AoT full-graph
- 否则会把“静态 NPU 不强”误写成“full-graph AoT 也不强”，进而污染后续阶段异构性和 overhead 分析。

## 下一步

1. 在文档中统一更正第二设备 `qnn-npu` 的实验类别口径。
2. 将 `Qwen2` full-graph AoT 复核结果与 static baseline 分开存档。
3. 若要补 `Qwen3` full-graph AoT，对第二设备先补齐 AoT 产物，再做同类验证。
