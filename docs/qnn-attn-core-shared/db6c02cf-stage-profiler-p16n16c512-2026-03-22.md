# `db6c02cf` 最小分阶段 profiling：`p16 / n16 / c512`

更新日期：2026-03-22

## 目标

这轮实验不测功耗，也不直接证明动态调度收益，而是优先回答两个更基础的问题：

1. 在同一组轻量配置下，`CPU` 与 `qnn-npu` 的 `Prefill/Decode` 分阶段时间分布是什么。
2. `GPUOpenCL` 为什么暂时还没进入这张正式矩阵。

这份结果服务于当前主线中的：

- `P3-1 Decode 分阶段 profiling`
- `P3-2 Prefill 分阶段 profiling`
- `P4-1 Decode 边界 overhead 分解`

## 配置

设备与构建：

- 设备：`db6c02cf`
- 构建：
  - `./build-npu-opencl.sh build-qnn-prof-db arm64-android-snapdragon-release --without-npu --with-gpu --with-qnn --with-profiling`
- 部署目录：
  - `/data/local/tmp/acom-stage-profiler-qwen2`
- 运行前确认：
  - `mWakefulness=Awake`
  - `mHoldingDisplaySuspendBlocker=true`

模型：

- `/data/local/tmp/acom-attn-proj-verify/models/qwen2_0.5b/ggml/weights.gguf`

统一参数：

- `-p 16`
- `-n 16`
- `-c 512`
- `-t 1`
- `--ignore-eos`

成功采集的两条路线：

- `CPU`
  - `LD_LIBRARY_PATH=. ./llama-stage-profiler ... -ngl 0 -fit off`
- `qnn-npu`
  - `GGML_HEXAGON_EXPERIMENTAL=1 ADSP_LIBRARY_PATH=. LD_LIBRARY_PATH=. ./llama-stage-profiler ... -ngl 99 -dev qnn-npu -fit off`

失败的 `GPUOpenCL` 路线：

- `p16 / n16 / c512 / -fit off / -ngl 99 -dev GPUOpenCL`
- `p8 / n8 / c256 / -fit on / -ngl 99 -dev GPUOpenCL`

对应文件：

- 成功：
  - `docs/qnn-attn-core-shared/db6c02cf-stage-profiler-p16n16c512-2026-03-22.csv`
  - `tmp/db6c02cf_cpu_decode_stage_p16_n16_c512_20260322.json`
  - `tmp/db6c02cf_cpu_decode_stage_p16_n16_c512_20260322.log`
  - `tmp/db6c02cf_qnn_decode_stage_p16_n16_c512_env_20260322.json`
  - `tmp/db6c02cf_qnn_decode_stage_p16_n16_c512_env_20260322.log`
- `GPUOpenCL` blocker：
  - `tmp/db6c02cf_gpu_decode_stage_p16_n16_c512_20260322.log`
  - `tmp/db6c02cf_gpu_decode_stage_p8_n8_c256_20260322.log`

说明：

- `CPU` 与 `qnn-npu` 两条路线都会在结果落盘后于 teardown 阶段以 `134` 退出。
- 但日志明确打印了：
  - `main: output written to ...json`
- 因此当前应将：
  - **JSON 是否成功写出**
  - **日志是否显示 prefill/decode 都完成**
  视为有效结果标准，而不是单纯看进程返回码。

## `Decode`：`CPU` vs `qnn-npu`

`Decode` 使用 `16` 次生成，对应 `24 x 16 = 384` 个 layer-stage 观测。

| Stage | CPU mean (us) | qnn-npu mean (us) | qnn/CPU |
| --- | ---: | ---: | ---: |
| `Attn_Proj` | `906.19` | `871.39` | `0.96x` |
| `KV_Cache` | `1631.12` | `1736.14` | `1.06x` |
| `Attn_Core` | `111.68` | `579.72` | `5.19x` |
| `FFN_Block` | `864.21` | `1234.62` | `1.43x` |

端到端 `Decode` 时间：

- `CPU`：
  - `1349.07 ms / 16 tokens`
  - 约 `84.32 ms/token`
- `qnn-npu`：
  - `1697.99 ms / 16 tokens`
  - 约 `106.12 ms/token`

直接观察：

- 当前 static `qnn-npu` decode 并不是“所有阶段都更快”。
- `Attn_Proj` 与 `CPU` 基本打平，甚至略快。
- `KV_Cache` 只比 `CPU` 略慢。
- 真正把 static `qnn-npu` decode 拉慢的关键阶段是：
  - `Attn_Core`
  - `FFN_Block`

这和已有 static baseline 的方向一致：

- `qnn-npu tg1 = 10.87`
- `CPU 1c tg1 = 14.59`

也就是说，当前主设备上 static `qnn-npu` decode 表现差，不是因为“每一段都差”，而是因为：

- **特定关键阶段仍明显不适合这条 static qnn 路线**

## `Prefill`：`CPU` vs `qnn-npu`

`Prefill` 使用 `16` 个 prompt token，对应 `24` 个 layer-stage 观测。

| Stage | CPU mean (us) | qnn-npu mean (us) | qnn/CPU |
| --- | ---: | ---: | ---: |
| `Attn_Proj` | `194.84` | `180.11` | `0.92x` |
| `KV_Cache` | `124.09` | `274.23` | `2.21x` |
| `Attn_Core` | `312.42` | `4017.45` | `12.86x` |
| `FFN_Block` | `5843.39` | `1990.58` | `0.34x` |

端到端 `Prefill` 时间：

- `CPU`：`155.39 ms`
- `qnn-npu`：`155.10 ms`

直接观察：

- 在这个短 prompt 配置下，`CPU` 与 `qnn-npu` 的总时间几乎一样。
- 但阶段分布完全不同：
  - `qnn-npu` 的 `FFN_Block` 明显更快；
  - `qnn-npu` 的 `Attn_Core` 明显更慢；
  - `KV_Cache` 也更重。

这说明当前主线最需要的“阶段异构性”信号已经出来了：

- 并不是整条 `Prefill` 或整条 `Decode` 有一个统一最优后端；
- 而是不同阶段在不同后端上的优势方向不同；
- 如果后续想做阶段级调度，必须同时把：
  - `Attn_Core`
  - `FFN_Block`
  - `KV_Cache`
  分开看，而不能只看整条路径的单个吞吐数。

## `GPUOpenCL` blocker

当前 `GPUOpenCL` 的 stage-profiler 路线还没进入正式矩阵，原因已经收敛得比较明确：

- 即使缩到：
  - `p16 / n16 / c512 / -fit off`
  - 甚至 `p8 / n8 / c256 / -fit on`
- 都仍然在模型加载阶段失败：
  - `llama_model_load: error loading model: unable to allocate OpenCL buffer`

因此现阶段更准确的表述应当是：

- `GPUOpenCL` 的端到端静态 baseline 已经可测；
- 但 `llama-stage-profiler` 路径下，它还受限于 OpenCL buffer 分配；
- 所以当前 `phase × stage × backend` 矩阵只拿到了 `CPU / qnn-npu` 两列，还缺 `GPUOpenCL` 这一列。

## 当前结论

截至这轮实验，最强的非功耗结论有四条：

1. `P3-1` 已经拿到同尺度的 `CPU / qnn-npu` `Decode` 分阶段数据，`GPUOpenCL` 当前被 OpenCL buffer 分配阻塞。
2. 在主设备 static 路线上，`Decode` 里真正拖慢 `qnn-npu` 的关键阶段是：
   - `Attn_Core`
   - `FFN_Block`
   而不是 `Attn_Proj`。
3. `Prefill` 里已经出现明显的阶段异构：
   - `FFN_Block` 更像适合 `qnn-npu`
   - `Attn_Core` 与 `KV_Cache` 当前更像不适合 static `qnn-npu`
4. 这批结果说明“存在潜在阶段级收益空间”，但还不能直接外推成系统级收益，因为：
   - `GPUOpenCL` 列缺失；
   - 还没有把 runtime overhead 单独拆出来；
   - 这里仍是 static backend 路线，不是完整的 `AoT mixed-stage` 动态执行。

## 下一步

按当前主线优先级，后续应继续：

1. 解决或绕开 `GPUOpenCL` 的 stage-profiler 内存分配问题，补齐第三列。
2. 用这组 `CPU / qnn-npu` 分阶段数据优先推进：
   - `P4-1 Decode 边界 overhead 分解`
3. 在 `Prefill` 侧继续验证：
   - 当前 `FFN` 的优势能否在更长 prompt 下保持；
   - `Attn_Core` 的高成本是否与 KV/host-visible buffer 管理直接相关。
