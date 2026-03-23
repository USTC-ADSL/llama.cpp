# 任务 012：补齐主设备 `db6c02cf` 的 decode 边界 overhead 第一刀

日期：2026-03-23

## 背景与目标

在完成：

- decode static baseline
- decode mixed-route 支持矩阵
- 最小 stage-profiler
- 阶段后端倾向初判

之后，当前非功耗主线里最缺的一块证据已经变成：

- **decode mixed route 的 runtime overhead 到底主要掉在 copy、launch、还是 residual tail 上**

如果这一步不补，后面：

- `P4-2 Prefill full-vs-split overhead 分解`
- `P4-3 ideal vs actual`
- `P6 cost model`

都会继续停留在“知道有 gap，但不知道 gap 结构”的阶段。

## 执行内容

### 1. 重新按规范构建 profiling 版本

执行：

- `./build-npu-opencl.sh build-qnn-prof-db arm64-android-snapdragon-release --without-npu --with-gpu --with-qnn --with-profiling`

然后把 `build-qnn-prof-db/bin` 同步到：

- `/data/local/tmp/acom-stage-profiler-qwen2`

### 2. 确认设备处于亮屏唤醒态

确认：

- `mWakefulness=Awake`
- `mHoldingDisplaySuspendBlocker=true`

### 3. 在主设备上抓取 event-level hetero profile CSV

统一模型：

- `/data/local/tmp/acom-attn-proj-verify/models/qwen2_0.5b/ggml/weights.gguf`

统一参数：

- `-r 1 -t 1 -p 0 -n 1 -c 2048 -b 1 -ub 1 -ctk f32 -ctv f32 --mmap 0`

统一 profile 开关：

- `GGML_HETERO_PROFILE=1`
- `GGML_HETERO_PROFILE_SYNC=1`
- `GGML_HETERO_PROFILE_FLUSH=1`
- `GGML_HETERO_PROFILE_CSV=<csv path>`
- `LLAMA_BENCH_FAST_EXIT=1`

抓了三条路线：

1. static `GPUOpenCL`
2. `attn_proj=opencl, attn_core=qnn-npu, attn_out=qnn-npu, ffn=opencl, output=cpu`
3. `attn_proj=cpu, attn_core=qnn-npu, attn_out=qnn-npu, ffn=cpu, output=cpu`

### 4. 把设备 CSV 拉回本地并按内部 pass 重分解

本轮一个关键发现是：

- `llama-bench tg1` 生成的 CSV 不止一个 pass；
- mixed route 会出现多次 `split_id` 从大值回退到 `0`。

因此不能直接把整份 CSV 求和，而是要按 `split_id` 回退切开，再选择真正贴近最终 `tg1` 的计量 pass：

- static `GPUOpenCL`：取 `pass=1`
- mixed route：取 `pass=2+3`

## 关键证据

### 1. 本轮 decode event CSV 已经成功落盘

本地归档：

- `tmp/p41f/db6c02cf_p41f_static_gpu_tg1_20260323.csv`
- `tmp/p41f/db6c02cf_p41f_opencl_qnn_opencl_tg1_20260323.csv`
- `tmp/p41f/db6c02cf_p41f_cpu_qnn_cpu_tg1_20260323.csv`

正式文档：

- `docs/qnn-attn-core-shared/db6c02cf-decode-boundary-overhead-2026-03-23.md`
- `docs/qnn-attn-core-shared/db6c02cf-decode-boundary-overhead-2026-03-23.csv`

### 2. 这三条路线的计量 pass 里都没有显式 `tensor_copy`

最终计量 pass 的结果都是：

- `tensor_copy = 0`
- `tensor_copy_wait = 0`

因此 decode mixed route 当前更不能简单说成：

- “主要慢在跨后端 activation memcpy”

### 3. `opencl -> qnn -> opencl` 的主 overhead 是 split fragmentation 和 CPU `result_output` tail

计量 pass：

- `split_compute = 104`
- `split_compute_total = 86.742 ms`
- backend 分布：
  - `OpenCL = 51 splits / 51.632 ms`
  - `qnn-npu = 25 splits / 19.508 ms`
  - `CPU = 28 splits / 15.602 ms`

其中 CPU 最重的两项都是：

- `result_norm -> result_output`
  - 合计 `14.031 ms`

这说明：

- 这条路线现在的关键风险不在显式 copy；
- 而在 `attn_core(attn_out) -> ffn/output` 一侧的 tail residual 与 split 数。

### 4. `cpu -> qnn -> cpu` route string 当前并不纯

计量 pass 里实际观测到：

- `CPU = 148 splits / 23.760 ms`
- `OpenCL = 123 splits / 37.413 ms`
- `qnn-npu = 25 splits / 17.712 ms`

也就是说，这条路线虽然 route string 写成了 `cpu/qnn/cpu`，但实际 trace 里仍混入了大量 `OpenCL` split。

因此当前更准确的解释是：

- 这是一个“CPU/QNN 主意图”的控制组；
- 但它还不是纯净的 runtime backend 对照。

### 5. 两条 mixed route 的 QNN 段成本接近

两条 mixed route 的计量 pass 中：

- `opencl -> qnn -> opencl`
  - `qnn-npu = 25 splits / 19.508 ms`
- `cpu -> qnn -> cpu`
  - `qnn-npu = 25 splits / 17.712 ms`

这说明当前 decode mixed route 的大头差异，更多来自：

- QNN 外围壳层的 fragmentation
- CPU tail
- route purity

而不是 `attn_core=qnn-npu` 自己突然变成主瓶颈。

## 当前结论

这轮 `P4-1` 第一刀已经把 decode 边界 overhead 的口径明显推进了一步：

1. decode mixed route 目前没有观察到显式 `tensor_copy`；
2. 当前更像主瓶颈的是：
   - split 数过多
   - enqueue / launch
   - `result_output` CPU tail
   - route string 与实际 backend purity 的偏差
3. 因此主线里关于 runtime overhead 的表述现在应该更精确地写成：
   - **decode 收益当前主要被 fragmentation / residual tail / purity 问题限制**
   - 而不是被“显式张量拷贝”限制

## 下一步

1. 继续推进 `P4-2 Prefill full-vs-split overhead 分解`
2. 额外补一个 decode purity 检查：
   - 为什么 `cpu -> qnn -> cpu` 仍有大量 `OpenCL` split
3. 再做 `P4-3 ideal vs actual`：
   - 把阶段最优收益与真实端到端 gap 对上
