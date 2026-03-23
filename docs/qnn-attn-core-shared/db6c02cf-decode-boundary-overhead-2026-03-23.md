# `db6c02cf` Decode 边界 overhead 第一刀

更新日期：2026-03-23

## 目标

这份记录回答 `P4-1 Decode 边界 overhead 分解` 的第一个更具体问题：

1. 在 decode `tg1` 的 mixed-stage 路线里，`attn_proj -> attn_core` 与 `attn_core(attn_out) -> ffn` 的 overhead 现在主要来自什么。
2. 之前日志里“share-heavy / no-copy”的判断，能否被 event-level CSV 证据进一步坐实。
3. 如果显式 `tensor_copy` 仍然为零，端到端 gap 又主要落在哪些 split / backend / residual tail 上。

这份文档只做 **decode first-cut**。

- 它不替代 2026-03-22 的 uninstrumented `tg1` baseline。
- 它也不把当前 profile 结果外推成“所有 mixed route 都已低 overhead”。

## 实验配置

- 设备：`db6c02cf`
- 构建：
  - `./build-npu-opencl.sh build-qnn-prof-db arm64-android-snapdragon-release --without-npu --with-gpu --with-qnn --with-profiling`
- 设备二进制目录：
  - `/data/local/tmp/acom-stage-profiler-qwen2`
- 模型：
  - `/data/local/tmp/acom-attn-proj-verify/models/qwen2_0.5b/ggml/weights.gguf`
- 统一 bench 参数：
  - `-r 1 -t 1 -p 0 -n 1 -c 2048 -b 1 -ub 1 -ctk f32 -ctv f32 --mmap 0`
- 统一保护开关：
  - `LLAMA_BENCH_FAST_EXIT=1`
- 统一 profile 开关：
  - `GGML_HETERO_PROFILE=1`
  - `GGML_HETERO_PROFILE_SYNC=1`
  - `GGML_HETERO_PROFILE_FLUSH=1`
  - `GGML_HETERO_PROFILE_CSV=<device csv path>`

本轮抓了三条 decode 控制路线：

1. `static GPUOpenCL`
2. `attn_proj=opencl, attn_core=qnn-npu, attn_out=qnn-npu, ffn=opencl, output=cpu`
3. `attn_proj=cpu, attn_core=qnn-npu, attn_out=qnn-npu, ffn=cpu, output=cpu`

QNN mixed route 额外使用：

- `GGML_HEXAGON_EXPERIMENTAL=1`
- `ADSP_LIBRARY_PATH=.`
- `GGML_HETERO_QNN_SHARED_HOST=1`
- `GGML_QNN_AOT_TRACE_MATCH=1`
- `GGML_QNN_AOT_CONFIG=/data/local/tmp/acom-attn-proj-verify/models/qwen2_0.5b/qnn_attn_core_combined.json`
- `GGML_QNN_AOT_MODEL_DIR=/data/local/tmp/acom-attn-proj-verify/models/qwen2_0.5b`

## 原始产物

本地归档在：

- `tmp/p41f/db6c02cf_p41f_static_gpu_tg1_20260323.csv`
- `tmp/p41f/db6c02cf_p41f_static_gpu_tg1_20260323.log`
- `tmp/p41f/db6c02cf_p41f_opencl_qnn_opencl_tg1_20260323.csv`
- `tmp/p41f/db6c02cf_p41f_opencl_qnn_opencl_tg1_20260323.log`
- `tmp/p41f/db6c02cf_p41f_cpu_qnn_cpu_tg1_20260323.csv`
- `tmp/p41f/db6c02cf_p41f_cpu_qnn_cpu_tg1_20260323.log`

汇总表同步写入：

- `docs/qnn-attn-core-shared/db6c02cf-decode-boundary-overhead-2026-03-23.csv`

## 口径说明

### 1. `split_compute` 是 inclusive 时间

当前 `ggml-backend.cpp` 的 profile 口径是：

- `split_enqueue`：`ggml_backend_graph_compute_async()` 调用返回前的时间
- `split_compute`：从进入 `ggml_backend_graph_compute_async()` 前开始，直到 `ggml_backend_synchronize()` 返回

因此：

- `split_compute` 已经包含 `split_enqueue`
- 不能把两者简单相加当成 token latency

### 2. `tg1` CSV 里包含 bench 内部的预热/计量 pass

`llama-bench -n 1 -r 1` 生成的 CSV 里，`split_id` 序列会多次从大值回到 `0`。本轮按“`split_id` 回退”把 CSV 切成内部 pass，再用下面的规则选取 **计量 pass**：

- `static GPUOpenCL`：取最后一个 pass（`pass=1`）
- 两条 mixed route：取最后两个 pass（`pass=2+3`）

理由是这些 pass 的 `split_compute` 总时长最接近日志里的最终 `tg1`：

- `static GPUOpenCL`：
  - `split_compute = 14.288 ms`
  - `1 / 69.12 tok/s = 14.47 ms`
- `opencl -> qnn -> opencl`：
  - `split_compute = 86.742 ms`
  - `1 / 10.82 tok/s = 92.42 ms`
- `cpu -> qnn -> cpu`：
  - `split_compute = 78.885 ms`
  - `1 / 11.92 tok/s = 83.89 ms`

因此这份文档里的表和结论，默认都只引用这些计量 pass。

### 3. profile 结果只用于 overhead 分解，不用于 route 排名

本轮 profile 会明显改变 mixed route 的绝对 `tg1`，所以：

- route 性能排序仍以 2026-03-22 的 uninstrumented 结果为准：
  - `opencl -> qnn -> opencl = 24.35`
  - `cpu -> qnn -> cpu = 19.20`
- 本文只用 2026-03-23 的 profile 结果回答：
  - **时间到底掉在哪些 split 上**

## 汇总表

| 路线 | 2026-03-22 `tg1` | 2026-03-23 profiled `tg1` | 计量 pass | `tensor_copy` / `wait` | `split_compute` 数 | `split_compute` 总时长 | `split_enqueue` 总时长 | CPU | OpenCL | qnn-npu | 关键尾部 |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| static `GPUOpenCL` | `71.06` | `69.12` | `1` | `0 / 0` | `2` | `14.288 ms` | `7.940 ms` | `0.023 ms` | `14.265 ms` | `0` | 无 |
| `opencl -> qnn -> opencl` | `24.35` | `10.82` | `2+3` | `0 / 0` | `104` | `86.742 ms` | `47.087 ms` | `15.602 ms` | `51.632 ms` | `19.508 ms` | `CPU result_norm -> result_output = 14.031 ms` |
| `cpu -> qnn -> cpu` | `19.20` | `11.92` | `2+3` | `0 / 0` | `296` | `78.885 ms` | `43.966 ms` | `23.760 ms` | `37.413 ms` | `17.712 ms` | `CPU result_norm -> result_output = 5.850 ms` |

## 关键发现

### 1. 这三条 decode 路线里都没有观察到显式 `tensor_copy`

在最终计量 pass 中：

- `tensor_copy = 0`
- `tensor_copy_wait = 0`

这意味着当前 decode mixed route 的边界成本，至少在这三条具体配置下：

- **不是 scheduler 显式插入 memcpy 在主导**

因此之前“share-heavy / no explicit copy”的日志判断，现在已经有了 event-level CSV 支撑。

### 2. `opencl -> qnn -> opencl` 的主要成本不是 QNN copy，而是 split fragmentation 和 CPU output tail

计量 pass 的 `split_compute` 分布：

- `OpenCL`：`51` 个 split，`51.632 ms`，占总 `59.5%`
- `qnn-npu`：`25` 个 split，`19.508 ms`，占总 `22.5%`
- `CPU`：`28` 个 split，`15.602 ms`，占总 `18.0%`

其中最突出的 CPU 项不是一串分散的小 residual，而是两次：

- `result_norm -> result_output`
  - 合计 `14.031 ms`
  - 占全部 CPU 时间的 `89.9%`
  - 占全部 `split_compute` 的 `16.2%`

这说明当前这条路线里更值得优先解释的 overhead 是：

1. `OpenCL` shell 仍被切成很多段；
2. 最后 `result_norm -> result_output` CPU tail 很重；
3. 而不是“QNN 边界触发了显式 activation copy”。

### 3. `cpu -> qnn -> cpu` 这条 route string 目前不能当成“纯 CPU 壳 + QNN 中段”

计量 pass 的 backend 分布是：

- `CPU`：`148` 个 split，`23.760 ms`
- `OpenCL`：`123` 个 split，`37.413 ms`
- `qnn-npu`：`25` 个 split，`17.712 ms`

也就是说，即便 route string 写的是：

- `attn_proj=cpu`
- `attn_core=qnn-npu`
- `ffn=cpu`

实际 event CSV 里仍然出现了大量 `OpenCL` split。

因此当前更准确的说法应该是：

- 这是一个“以 `cpu/qnn` 路由意图为主，但运行时仍混入大量 `OpenCL` fragment”的控制路线；
- 它还不能被当成一个完全纯净的 `CPU -> QNN -> CPU` 对照组。

这件事本身就是后续主线里还要继续追的 runtime purity 问题。

### 4. 两条 mixed route 的 QNN 段成本接近，真正放大差距的是 QNN 外围壳层

计量 pass 里：

- `opencl -> qnn -> opencl`
  - `qnn-npu = 25 splits / 19.508 ms`
- `cpu -> qnn -> cpu`
  - `qnn-npu = 25 splits / 17.712 ms`

QNN 段本身很接近，说明当前 mixed-route 的大头差异并不主要来自：

- `attn_core=qnn-npu` 自己突然变得极慢

而主要来自 QNN 之外的壳层：

- 前后两侧被切成多少 split
- 最后一段 CPU output tail 有多重
- runtime 里是否还混入额外 backend fragment

这与当前 decode-centric 主线是对齐的：

- `attn_core` 仍然是可利用的异构段；
- 但系统收益能不能释放出来，更依赖它周围的 runtime overhead。

### 5. `attn_core -> ffn / output` 一侧仍是当前 decode mixed route 的主要瓶颈边界

本轮最重的非 QNN 尾部都集中在：

- `attn_out-*`
- `ffn_inp-*`
- `result_norm -> result_output`

尤其是 `opencl -> qnn -> opencl` 路线里，最后两次 CPU `result_output` tail 直接吃掉了 `14.031 ms`。

因此当前更可靠的 decode overhead 结论是：

- `attn_proj -> attn_core` 这一侧目前没有显式 copy 主导；
- `attn_core(attn_out) -> ffn -> output` 这一侧的 residual tail 和 fragment granularity，仍然是 mixed route 的主要 runtime 风险。

## 对主线的意义

这轮 `P4-1` 第一刀已经能支撑两个更强的结论：

1. `Decode` mixed route 的边界成本，目前不应再笼统归因于“跨后端就会 memcpy”。
2. 真正更像主瓶颈的是：
   - split 数过多；
   - enqueue/launch 开销；
   - `result_output` CPU tail；
   - 以及 route intent 与实际 backend purity 之间的偏差。

因此这轮结果更符合主线 ④ 的表述：

- runtime overhead 是释放系统收益的关键瓶颈；
- 但这个 overhead 现在看起来主要是 **fragmentation / tail residual / backend purity**，
- 而不是显式 tensor copy。

## 还缺什么

当前还不能直接宣称“decode mixed route 已经获得最佳系统收益”，原因有三点：

1. 这轮 profile 会改变 mixed route 的绝对 `tg1`，所以它只能回答 overhead 结构，不能直接做 route 排名。
2. `cpu -> qnn -> cpu` 路线仍混入了大量 `OpenCL` split，说明 route purity 还没被完全钉死。
3. 这轮只做了 decode `tg1`；还没有把同样的 event-level 口径扩展到 prefill full-vs-split。

## 下一步

1. 进入 `P4-2 Prefill full-vs-split overhead 分解`：
   - 重点拆 `graph launch`
   - `shared-host KV writeback`
   - fragment I/O copy / direct-bind 命中率
2. 在 decode 侧补一个更干净的 route purity 检查：
   - 为什么 `cpu -> qnn -> cpu` 的计量 pass 仍出现 `123` 个 `OpenCL` split
3. 把这份第一刀结果和已有 uninstrumented `tg1` 合并，做 `P4-3 ideal vs actual`：
   - “阶段最优之和” 与 “真实端到端” 的差距到底落在哪几类 overhead 上
