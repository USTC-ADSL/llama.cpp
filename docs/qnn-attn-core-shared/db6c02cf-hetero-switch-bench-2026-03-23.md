# `db6c02cf` hetero-switch-bench 对照

更新日期：2026-03-23

## 目标

这份记录补齐 `P4-4 微基准与系统级对照`，回答两个问题：

1. `shared host ptr` 和显式 `memcpy` 的量级差，在一个独立微基准里到底有多大。
2. 这个量级是否足以解释 `P4-1/P4-2` 里 decode / prefill 的系统级 gap。

## 实验配置

- 设备：`db6c02cf`
- 构建：
  - `./build-npu-opencl.sh build-qnn-prof-db arm64-android-snapdragon-release --without-npu --with-gpu --with-qnn --with-profiling`
- 目标：
  - `build-qnn-prof-db/bin/hetero-switch-bench`
- 设备目录：
  - `/data/local/tmp/acom-stage-profiler-qwen2`
- 设备状态：
  - `mWakefulness=Awake`
  - `mHoldingDisplaySuspendBlocker=true`
- 运行命令：
  - `LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./hetero-switch-bench --warmup 5 --iters 50 --sizes 7168,65536,1048576,16777216 --csv /data/local/tmp/acom-stage-profiler-qwen2/hetero-switch-bench-db6c02cf-20260323.csv`

说明：

- 本轮专门加入 `7168 B`，因为它正好对应 `P4-2` 里唯一观测到的显式 scheduler copy 总量。

## 原始产物

- `tmp/p44switch/db6c02cf_hetero_switch_bench_20260323.log`
- `tmp/p44switch/db6c02cf_hetero_switch_bench_20260323.csv`

汇总表：

- `docs/qnn-attn-core-shared/db6c02cf-hetero-switch-bench-2026-03-23.csv`

## 结果汇总

### 1. `host_write_to_opencl_read`

这是更接近“CPU 写入边界数据，然后 OpenCL 读”的方向。

| 大小 | shared-host | memcpy | shared/memcpy | 结论 |
| --- | ---: | ---: | ---: | --- |
| `7168 B` | `864.010 us` | `864.294 us` | `1.000x` | 几乎完全一样 |
| `64 KB` | `6425.406 us` | `6785.330 us` | `0.947x` | shared 略快 |
| `1 MB` | `85041.678 us` | `86255.060 us` | `0.986x` | 仍然几乎同量级 |
| `16 MB` | `1364761.000 us` | `1368890.000 us` | `0.997x` | 大尺寸差距仍然很小 |

有效性：

- `shared-host`：`50 / 50` 全部有效
- `memcpy`：`50 / 50` 全部有效

### 2. `opencl_write_to_host_read`

这是更接近“OpenCL 写回 host，然后 CPU 读”的方向。

| 大小 | shared-host | memcpy | shared/memcpy | 有效性 | 结论 |
| --- | ---: | ---: | ---: | --- | --- |
| `7168 B` | `249.802 us` | `254.164 us` | `0.983x` | `0 / 50` vs `50 / 50` | shared 虽快但无效 |
| `64 KB` | `731.499 us` | `689.156 us` | `1.061x` | `0 / 50` vs `50 / 50` | shared 无效 |
| `1 MB` | `835.655 us` | `1021.502 us` | `0.818x` | `0 / 50` vs `50 / 50` | shared 无效 |
| `16 MB` | `1529.488 us` | `4863.001 us` | `0.315x` | `0 / 50` vs `50 / 50` | shared 很快但仍无效 |

这里的关键不是 shared 表面上快不快，而是：

- 当前 microbench 里，`CL_MEM_USE_HOST_PTR + clFinish` 之后直接读 host buffer，并 **不构成一个正确的 `opencl -> host` 语义等价路径**；
- shared-host 在这个方向上 `0 / 50` 全部校验失败；
- memcpy 则 `50 / 50` 全部通过。

## 与系统级观察的对照

### 1. `P4-2`：prefill warm gap 不是外层 copy 主导

`P4-2` 已经测到：

- 唯一显式 scheduler copy 总量只有 `7168 B`
- 总时间只有 `2 us`
- full-vs-split warm gap 是 `33.168 ms`

而本轮 microbench 在同样 `7168 B` 尺寸上显示：

- `host_write_to_opencl_read`
  - shared 与 memcpy 的差只有 `0.283 us`

因此这轮对照进一步说明：

- **7 KB 级别的边界 copy 选择 shared 还是 memcpy，量级都不可能解释 `33 ms` 级别的 prefill gap**

这和 `P4-2` 已经给出的系统级结论是一致的：

- 真正的 gap 在 QNN backend 内部 stage-chain fragmentation / shared-host materialization
- 不在外层 scheduler copy 本身

### 2. `P4-1`：decode mixed-route 的主要问题也不是外层 memcpy

`P4-1` 已经表明 decode mixed route 的计量 pass 里：

- `tensor_copy = 0`
- `tensor_copy_wait = 0`

而这轮微基准说明即使真的做了一个 `host -> OpenCL` 方向的数据交接：

- shared 和 memcpy 的差距也很小；
- 至少在当前设备上，它不是一个足以解释 `10~80 ms` 级 runtime gap 的主导项。

这进一步支持 `P4-1` 的判断：

- decode 侧更值得解释的是：
  - split fragmentation
  - `result_output` CPU tail
  - route purity

### 3. raw shared-host 不是“免费且正确”的 `OpenCL -> host` 替代

本轮最有价值的反例是：

- `opencl_write_to_host_read` 方向上，raw shared-host 的 apparent latency 很低；
- 但它在 `4` 个尺寸上都是 `0 / 50` 校验通过。

这意味着：

- 不能把“shared-host 看起来快”直接等价成“系统里可以无代价替代所有正确的 materialization / readback”
- 一旦系统需要 **正确且可见** 的 `device -> host` 结果，就仍然可能支付：
  - cache coherence
  - synchronization
  - materialization
  - explicit copyback

这和 `P4-2` 里对 prefill split 的判断是对齐的：

- runtime 里真正昂贵的，很可能不是外层 scheduler 显式 copy；
- 而是 backend 内部为了把 shared-host 变成“正确可消费状态”所付出的 materialization / 管理成本。

## 当前结论

`P4-4` 的第一版结论可以明确写成两句：

1. **对 `host -> OpenCL` 方向，shared-host 与 memcpy 在当前设备上的量级几乎一致，不足以解释 `P4-1/P4-2` 的系统级 gap。**
2. **对 `OpenCL -> host` 方向，raw shared-host 虽快但不正确，因此系统里任何正确的 shared-host readback 都不能被当作“零成本”路径。**

## 对主线的意义

这轮微基准把前两份系统级文档的口径又收紧了一步：

- `P4-1/P4-2` 里“不是外层 memcpy 主导”的说法，现在既有系统级 evidence，也有系统外微基准量级对照。
- 同时，shared-host 也不能被过度浪漫化成“天然免费”，因为它在 `OpenCL -> host` 方向上还需要额外 correctness 保障。

因此后续更合理的优化顺序仍然是：

1. 减少 fragment 数量
2. 提高 direct-bind 命中率
3. 降低 backend 内部 materialization / synchronization 成本

而不是优先去抠外层那一点 scheduler copy。

## 下一步

1. 进入 `P3-3/P3-4`，把这份“外层 copy 不是主因，但 shared-host 仍有 correctness / materialization 代价”的结论写进正式阶段矩阵解释。
2. 回到 decode 主线，继续收口 route purity 与 tail residual。
3. 在此基础上，再做最小 `SLO-aware` 决策闭环，避免把不稳定或无效的 shared-host 假设写进 cost model。
