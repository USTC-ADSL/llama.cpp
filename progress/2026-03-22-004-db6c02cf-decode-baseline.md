# 任务 004：采集主设备 `db6c02cf` 的 decode 单后端 baseline

日期：2026-03-22

## 背景与目标

在完成 decode tail residual 稳定性收口后，下一优先级任务是补齐主设备 `db6c02cf` 的单后端 decode baseline，作为后续：

- 阶段异构性矩阵
- runtime overhead 分解
- SLO-aware 路由

的静态参考。

本任务的最小目标是：

- `CPU(1c/2c)`
- `GPUOpenCL`
- `qnn-npu`

三类后端都至少拿到：

- `tg1`
- `tg128`

## 执行内容

设备：

- `db6c02cf`

二进制目录：

- `/data/local/tmp/acom-stage-matrix-verify`

模型：

- `/data/local/tmp/acom-attn-proj-verify/models/qwen2_0.5b/ggml/weights.gguf`

统一参数：

- `-r 1`
- `-p 0`
- `-c 2048`
- `-b 1 -ub 1`
- `-ctk f32 -ctv f32`
- `--mmap 0`

统一环境变量：

- `LD_LIBRARY_PATH=.`
- `ADSP_LIBRARY_PATH=.`
- `LLAMA_BENCH_FAST_EXIT=1`

NPU 路线额外使用：

- `GGML_HEXAGON_EXPERIMENTAL=1`

本轮实际采集了 8 条 decode baseline：

1. `CPU 1c tg1`
2. `CPU 1c tg128`
3. `CPU 2c tg1`
4. `CPU 2c tg128`
5. `GPUOpenCL tg1`
6. `GPUOpenCL tg128`
7. `qnn-npu tg1`
8. `qnn-npu tg128`

同时产出 CSV：

- `docs/qnn-attn-core-shared/db6c02cf-decode-baseline-2026-03-22.csv`

## 关键证据

### 1. 主设备也存在 exit-time crash，但 `FAST_EXIT` 足以保证 baseline 可采

未加 `LLAMA_BENCH_FAST_EXIT=1` 时，`CPU tg1` 已打印出结果：

- `tg1 = 14.53 ± 0.00`

随后进程以 `Segmentation fault` 结束。

加入 `LLAMA_BENCH_FAST_EXIT=1` 后，baseline 可以稳定结束并保留结果，因此本轮后续所有 baseline 都统一使用该开关。

### 2. 当前 `llama-bench` 的 NPU 设备枚举名是 `qnn-npu`，不是 `HTP0`

设备枚举输出为：

- `GPUOpenCL`
- `qnn-npu`
- `qnn-gpu`
- `qnn-cpu`

因此本轮主设备 baseline 采用：

- `-dev GPUOpenCL`
- `-dev qnn-npu`

而没有使用 `HTP0` 这个别名。

### 3. 主设备 decode baseline 结果

| 后端 | 配置 | `tg1` | `tg128` |
|------|------|------:|--------:|
| CPU | `taskset 80 -t 1 -ngl 0` | `14.59` | `14.63` |
| CPU | `taskset C0 -t 2 -ngl 0` | `9.84` | `15.33` |
| GPUOpenCL | `taskset 80 -t 1 -ngl 99 -dev GPUOpenCL` | `71.06` | `68.34` |
| qnn-npu | `taskset 80 -t 1 -ngl 99 -dev qnn-npu` | `10.87` | `10.89` |

对应日志：

- `tmp/db6c02cf_cpu_decode_tg1_fast_exit.log`
- `tmp/db6c02cf_cpu_decode_tg128_fast_exit.log`
- `tmp/db6c02cf_cpu2_decode_tg1_fast_exit.log`
- `tmp/db6c02cf_cpu2_decode_tg128_fast_exit.log`
- `tmp/db6c02cf_gpu_decode_tg1_fast_exit.log`
- `tmp/db6c02cf_gpu_decode_tg128_fast_exit.log`
- `tmp/db6c02cf_npu_decode_tg1_fast_exit.log`
- `tmp/db6c02cf_npu_decode_tg128_fast_exit.log`

## 当前结论

本轮主设备 decode baseline 已经满足最小采集目标：

- `CPU(1c/2c) / GPUOpenCL / qnn-npu`
- `tg1 / tg128`

从当前结果看，有三个直接可用的观察：

1. `GPUOpenCL` 明显是当前主设备上最强的静态 decode backend。
2. `qnn-npu` 静态 decode baseline 明显落后于 `GPUOpenCL`，也低于 `CPU 1c`。
3. `CPU 2c` 的结果并不单调优于 `CPU 1c`：
   - `tg1` 反而明显更慢
   - `tg128` 则略快

这说明：

- decode 的“工作点”不应只看后端，也要看 CPU 线程/核掩码；
- 后续功率可调空间与 SLO routing 里，`CPU 1c/2c` 应当被视为不同 operating points，而不是同一个 CPU backend 的简单放大版。

## 还缺什么

当前这组 baseline 仍有几个限制：

1. 只有 `r=1`，还不能作为高置信方差结论。
2. 只覆盖了 decode，还缺主设备 `prefill` baseline。
3. 还没有同步功率/能耗读数，因此目前只能作为性能 baseline。

## 下一步

1. 继续采集主设备 `prefill` baseline：
   - `CPU / GPUOpenCL / qnn-npu`
   - 至少 `pp128 / pp256 / pp512`
2. 将这组 decode baseline 与现有 mixed-route 结果放在一起，开始构造正式的“静态后端对照 + 阶段异构性”分析表。
3. 在 overhead 阶段重点解释：
   - 为什么 static `GPUOpenCL` 当前显著强于 static `qnn-npu`
   - 为什么 mixed stage route 仍然可能有研究价值，因为调度目标是 `stage × SLO × power`，而不是单纯静态单后端最强。
