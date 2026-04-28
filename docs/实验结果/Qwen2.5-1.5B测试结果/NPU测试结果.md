# Qwen2.5-1.5B NPU 测试结果

## 2026-04-28：CPU WALT 不锁频下的 `tg128 r=3` 与 `pp512 r=50`

### 测试配置

- 设备：`192.168.1.113:42977`
- 模型：`/data/local/tmp/restart-static-switch-validation/models/Qwen2.5-1.5B-AoT/ggml/weights.gguf`
- QNN 目录：`/data/local/tmp/restart-static-switch-validation/models/Qwen2.5-1.5B-AoT/qnn`
- 后端：`qnn-npu`，`-ngl 99 -dev qnn-npu`
- 线程绑定：`taskset 80 -t 1`
- CPU 背景策略：`walt`，不手动锁频；本次设备实际限制为 `policy0=384000-1996800`，`policy6=1017600-2649600`
- `tg128` 测试形态：`-p 0 -n 128 -b 1 -ub 1 -r 3`
- `pp512` 测试形态：`-p 512 -n 0 -b 128 -ub 128 -r 50`
- 屏幕状态：测试时保持亮屏
- 温控约束：`TEMP_LIMIT_C=38.0`，`COOLDOWN_TEMP_C=37.0`

### 基础功率

- `tg128 r=3` baseline：`514.37 mW`，`21.97 C`
- `pp512 r=50` baseline：`375.05 mW`，`25.48 C`

### 结果表

| Workpoint | TG128 吞吐 tok/s | TG128 功率 mW | TG128 增量 mW | TG128 波动 % | PP512 吞吐 tok/s | PP512 功率 mW | PP512 增量 mW | PP512 波动 % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native | 5.30 | 752.85 | 238.48 | 8.65 | 200.19 | 772.82 | 397.77 | 6.09 |
| burst | 25.39 | 2966.80 | 2452.43 | 49.00 | 963.16 | 5414.64 | 5039.59 | 4.32 |
| high_performance | 20.82 | 2213.92 | 1699.55 | 13.04 | 782.41 | 3043.01 | 2667.96 | 1.75 |
| balanced | 17.84 | 1736.95 | 1222.58 | 6.40 | 668.06 | 2219.36 | 1844.31 | 1.25 |
| low_balanced | 17.57 | 1684.16 | 1169.79 | 4.28 | 594.95 | 1879.65 | 1504.60 | 5.11 |
| high_power_saver | 12.59 | 1315.66 | 801.29 | 8.81 | 449.15 | 1419.62 | 1044.57 | 4.97 |
| power_saver | 12.23 | 1251.60 | 737.23 | 6.30 | 416.74 | 1287.42 | 912.37 | 5.62 |
| low_power_saver | 8.88 | 969.08 | 454.71 | 9.92 | 303.82 | 1001.87 | 626.82 | 6.80 |
| extreme_power_saver | 5.31 | 746.76 | 232.39 | 4.18 | 199.31 | 768.87 | 393.82 | 7.25 |

### 原始结果

- `tg128 r=3` 汇总 CSV：`/tmp/qwen15b-npu-walt-tg128-r3-20260428/results.csv`
- `tg128 r=3` baseline：`/tmp/qwen15b-npu-walt-tg128-r3-20260428/baseline.samples.csv`
- `pp512 r=50` 汇总 CSV：`/tmp/qwen15b-npu-walt-pp512-r50-20260428/results.csv`
- `pp512 r=50` baseline：`/tmp/qwen15b-npu-walt-pp512-r50-20260428/baseline.samples.csv`

### 备注

- `tg128 r=3` 是短 Decode 测试，高性能档位的功率窗口波动较大，尤其 `burst=49.00%`，因此该组更适合看吞吐趋势，不应作为严格稳态功率结论。
- `pp512 r=50` 的功率窗口明显更稳定，其中 `burst` 为 `963.16 tok/s / 5414.64 mW`，`balanced` 为 `668.06 tok/s / 2219.36 mW`。
- 全部测试最高温度低于 `38 C`，`tg128` 最高约 `25.40 C`，`pp512` 最高约 `27.70 C`。
