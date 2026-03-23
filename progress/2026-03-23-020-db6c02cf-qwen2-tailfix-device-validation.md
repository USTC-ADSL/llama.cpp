# 2026-03-23-020 `db6c02cf` Qwen2 Decode Tail-Fix Device Validation

## 任务

在真实设备上验证 `Qwen2` decode `cpu/qnn/cpu` 路线里，最后一层输出裁剪 `GET_ROWS` 的局部修复是否已经消除 residual `OpenCL` split。

## 设备与配置

- 设备：`db6c02cf`
- 运行目录：`/data/local/tmp/acom-stage-matrix-verify`
- 模型：`/data/local/tmp/acom-attn-proj-verify/models/qwen2_0.5b/ggml/weights.gguf`
- AoT 配置：`/data/local/tmp/acom-attn-proj-verify/models/qwen2_0.5b/qnn_attn_core_combined.json`
- 路线：`attn_proj=cpu,attn_core=qnn-npu,attn_out=qnn-npu,ffn=cpu,output=cpu`
- KV layout：`qnn`
- bench 参数：
  - `-r 1 -t 1 -p 0 -n 1 -c 2048 -b 1 -ub 1`
  - `-ctk f32 -ctv f32`
  - `-ngl 0 -dev GPUOpenCL --mmap 0 --no-warmup`
- 关键环境变量：
  - `LD_LIBRARY_PATH=.`
  - `ADSP_LIBRARY_PATH=.`
  - `LLAMA_BENCH_FAST_EXIT=1`
  - `GGML_HEXAGON_EXPERIMENTAL=1`
  - `GGML_HETERO_QNN_SHARED_HOST=1`
  - `GGML_HETERO_KV_LAYOUT=qnn`
  - `LLAMA_GRAPH_REUSE_DISABLE=1`

## 完成内容

- 重新执行 `build-npu-opencl.sh`。
- 将更新后的 `libllama.so` 与 `llama-bench` 推送到设备运行目录。
- 先执行 `--list-devices`，确认设备仍能正常加载：
  - `GPUOpenCL`
  - `qnn-npu`
  - `qnn-gpu`
  - `qnn-cpu`
- 分别执行两条复测：
  - `scheddebug` 口径：`tmp/p17/purity-cpu-qnn-cpu-tailfix-scheddebug.log`
  - `noreuse` 口径：`tmp/p17/purity-cpu-qnn-cpu-tailfix-noreuse.log`

## 结果

### 1. residual `OpenCL` split 已消失

旧 CSV：

- `tmp/p17/purity-cpu-qnn-cpu-noreuse.csv`
- backend 计数：`CPU=52, qnn-npu=50, OpenCL=2`
- 唯一 `OpenCL` split：
  - `split_id=48`
  - `node_start=929`
  - `node_end=931`
  - `node_first=node_929`
  - `node_last=node_930`

新 CSV：

- `tmp/p17/purity-cpu-qnn-cpu-tailfix-noreuse.csv`
- backend 计数：`CPU=54, qnn-npu=50`
- `OpenCL=0`

这说明原来那组 residual `OpenCL` split 已经被 CPU/QNN 路径吸收，不再单独落到 `OpenCL`。

### 2. 最后一层输出裁剪节点已进入显式 stage route

`scheddebug` 日志中可直接看到：

- `attn_out-tail-23 reason=aot-qnn backend=qnn-npu`
- `l_out-tail-23 reason=hetero-stage backend=CPU`

同时，旧日志中反复出现的匿名 `node_929 / node_930 / node_931` 已经不再出现为 `OpenCL` split 节点。

### 3. residual unmatched fragment 仍在，但形态变了

新日志里仍有：

- `unmatched cgraph: n_nodes=18 first=cache_k_upd-23 last=attn_out-tail-23`
- `cpu fallback for unmatched residual cgraph: n_nodes=18 first=cache_k_upd-23 last=attn_out-tail-23`
- `unmatched cgraph: n_nodes=1 first=ffn_inp-23 last=ffn_inp-23`

因此本次修复解决的是“不该出现的 residual `OpenCL` split”，不是把最后一层 residual fragment 完全消掉。

## 判断

### 强结论

- `Qwen2` 这条 decode `cpu/qnn/cpu` 路线中的 residual `OpenCL` split 已在真机上收口。
- 局部修复是有效的：最后一层输出裁剪 `GET_ROWS` 现在不再因为匿名节点而被 scheduler 抬到 `OpenCL`。

### 保守结论

- 当前还不能说这条 mixed decode 图已经“完全纯净”。
- 更准确的说法是：
  - residual `OpenCL` split 已消失；
  - 但最后一层 residual unmatched cgraph 仍存在，只是现在明确落回 CPU fallback。

## 吞吐解释

- 新 `noreuse` 口径下 `tg1 = 5.28 ± 0.00`
- 旧 `noreuse` 口径下 `tg1 = 5.14 ± 0.00`

因此这次修复没有引入 purity 口径下的额外明显性能退化。

注意：

- 这个 `5.x t/s` 是带 `hetero profile`、`graph reuse disable`、纯度排查口径的结果；
- 不能直接与之前 stage-matrix 文档中的 `19.20 t/s` 静态 route 结果横向比较。

## 当前边界

- 这次只验证了 `Qwen2`。
- `db6c02cf` 与 `fd8657d6` 当前都没有已部署的 `Qwen3` 模型与 AoT 配置，因此 `Qwen3` 的设备复测尚未进行。

## 产物

- `tmp/p17/purity-cpu-qnn-cpu-tailfix-scheddebug.log`
- `tmp/p17/purity-cpu-qnn-cpu-tailfix-scheddebug.csv`
- `tmp/p17/purity-cpu-qnn-cpu-tailfix-noreuse.log`
- `tmp/p17/purity-cpu-qnn-cpu-tailfix-noreuse.csv`
