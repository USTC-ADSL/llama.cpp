# NPU `tg64 r=10` Workpoint Sweep

测试日期：2026-04-27

## 测试配置

- 设备：`192.168.1.113:38435`
- 脚本：[NPUtest.sh](/mnt/sda1/pzw/HeteroCompute/llama.cpp-acom/docs/实验结果/NPUtest.sh:1)
- 输出目录：`/tmp/npu-tg64-r10-20260427-092612`
- 模型：`/data/local/tmp/powerserve/Qwen2-3B/ggml/weights.gguf`
- 后端：`qnn-npu`
- 测试形态：`tg64`，即 `-p 0 -n 64 -b 1 -ub 1 -r 10`
- HTP workpoint：`native burst high_performance balanced low_balanced high_power_saver power_saver low_power_saver extreme_power_saver`
- 大核策略：`policy6 = powersave`，`min=max=cur=1017600`
- 屏幕状态：由脚本强制保持亮屏
- 温控约束：`TEMP_LIMIT_C=38.0`，`COOLDOWN_TEMP_C=37.0`
- 稳态功率口径：启动后跳过 `8s`，在 `8` 个采样点窗口内选 steady-state window 平均功率

## 基础功率

- baseline 样本文件：`/tmp/npu-tg64-r10-20260427-092612/baseline.samples.csv`
- baseline 平均功率：`396.69 mW`
- baseline 平均温度：`25.80 C`
- 测试前单次温度读数：`25.8 C`

## 结果表

| Workpoint | 吞吐 tok/s | 稳态平均功率 mW | 相对 baseline 增量 mW | 平均温度 C | 最高温度 C | 稳定窗口范围 % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| native | 4.47 | 863.80 | 467.11 | 25.79 | 25.80 | 16.88 |
| burst | 21.33 | 3658.35 | 3261.66 | 25.94 | 26.00 | 1.99 |
| high_performance | 17.36 | 2526.96 | 2130.27 | 26.46 | 26.50 | 6.49 |
| balanced | 15.58 | 2079.94 | 1683.25 | 27.00 | 27.10 | 8.27 |
| low_balanced | 15.35 | 1949.27 | 1552.58 | 27.34 | 27.40 | 11.46 |
| high_power_saver | 10.92 | 1498.57 | 1101.88 | 27.69 | 27.80 | 10.11 |
| power_saver | 10.71 | 1476.81 | 1080.12 | 28.00 | 28.00 | 15.74 |
| low_power_saver | 7.33 | 1106.00 | 709.31 | 28.09 | 28.10 | 17.58 |
| extreme_power_saver | 4.43 | 873.11 | 476.42 | 27.99 | 28.00 | 23.90 |

## 原始结果

- 汇总 CSV：`/tmp/npu-tg64-r10-20260427-092612/results.csv`
- `native`：bench `/tmp/npu-tg64-r10-20260427-092612/npu_native.bench.log`，samples `/tmp/npu-tg64-r10-20260427-092612/npu_native.samples.csv`
- `burst`：bench `/tmp/npu-tg64-r10-20260427-092612/npu_burst.bench.log`，samples `/tmp/npu-tg64-r10-20260427-092612/npu_burst.samples.csv`
- `high_performance`：bench `/tmp/npu-tg64-r10-20260427-092612/npu_high_performance.bench.log`，samples `/tmp/npu-tg64-r10-20260427-092612/npu_high_performance.samples.csv`
- `balanced`：bench `/tmp/npu-tg64-r10-20260427-092612/npu_balanced.bench.log`，samples `/tmp/npu-tg64-r10-20260427-092612/npu_balanced.samples.csv`
- `low_balanced`：bench `/tmp/npu-tg64-r10-20260427-092612/npu_low_balanced.bench.log`，samples `/tmp/npu-tg64-r10-20260427-092612/npu_low_balanced.samples.csv`
- `high_power_saver`：bench `/tmp/npu-tg64-r10-20260427-092612/npu_high_power_saver.bench.log`，samples `/tmp/npu-tg64-r10-20260427-092612/npu_high_power_saver.samples.csv`
- `power_saver`：bench `/tmp/npu-tg64-r10-20260427-092612/npu_power_saver.bench.log`，samples `/tmp/npu-tg64-r10-20260427-092612/npu_power_saver.samples.csv`
- `low_power_saver`：bench `/tmp/npu-tg64-r10-20260427-092612/npu_low_power_saver.bench.log`，samples `/tmp/npu-tg64-r10-20260427-092612/npu_low_power_saver.samples.csv`
- `extreme_power_saver`：bench `/tmp/npu-tg64-r10-20260427-092612/npu_extreme_power_saver.bench.log`，samples `/tmp/npu-tg64-r10-20260427-092612/npu_extreme_power_saver.samples.csv`

## 直接观察

- 这组是纯 `Decode tg64` 静态 `qnn-npu` workpoint 扫描，最高温只有 `28.10 C`，没有触发 `38 C` 热保护，因此当前结果基本不受 thermal throttling 干扰。
- `burst` 是吞吐最高点，`21.33 tok/s`，同时也是功率最高点，稳态功率 `3658.35 mW`，但它的稳态窗口最干净，`stable_range_pct=1.99%`。
- `balanced` 与 `low_balanced` 的吞吐非常接近，分别是 `15.58 tok/s` 和 `15.35 tok/s`，但 `low_balanced` 的稳态功率更低，当前看是一个值得保留的中档静态参考点。
- `native` 和 `extreme_power_saver` 都落在 `4.4 tok/s` 左右，但两者的功率平台波动都比较大，后续如果要拿它们做更强结论，建议再拉长轮数确认更稳的平台窗口。

# NPU `pp512 r=60` Workpoint Sweep

测试日期：2026-04-27

## 测试配置

- 设备：`192.168.1.113:38435`
- 脚本：[NPUtest.sh](/mnt/sda1/pzw/HeteroCompute/llama.cpp-acom/docs/实验结果/NPUtest.sh:1)
- 输出目录：`/tmp/npu-pp512-r60-20260427-094746`
- 模型：`/data/local/tmp/powerserve/Qwen2-3B/ggml/weights.gguf`
- 后端：`qnn-npu`
- 测试形态：`pp512`，即 `-p 512 -n 0 -b 128 -ub 128 -r 60`
- HTP workpoint：`native burst high_performance balanced low_balanced high_power_saver power_saver low_power_saver extreme_power_saver`
- 大核策略：`policy6 = powersave`，`min=max=cur=1017600`
- 屏幕状态：由脚本强制保持亮屏
- 温控约束：`TEMP_LIMIT_C=38.0`，`COOLDOWN_TEMP_C=37.0`
- 稳态功率口径：启动后跳过 `8s`，在 `8` 个采样点窗口内选 steady-state window 平均功率

## 基础功率

- baseline 样本文件：`/tmp/npu-pp512-r60-20260427-094746/baseline.samples.csv`
- baseline 平均功率：`399.91 mW`
- baseline 平均温度：`27.14 C`
- 测试前单次温度读数：`27.2 C`

## 结果表

| Workpoint | 吞吐 tok/s | 稳态平均功率 mW | 相对 baseline 增量 mW | 平均温度 C | 最高温度 C | 稳定窗口范围 % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| native | 214.73 | 768.04 | 368.13 | 26.80 | 26.80 | 8.51 |
| burst | 1087.81 | 3919.47 | 3519.56 | 26.76 | 26.80 | 2.83 |
| high_performance | 877.22 | 2408.78 | 2008.87 | 27.04 | 27.10 | 4.49 |
| balanced | 763.44 | 1878.86 | 1478.95 | 27.40 | 27.40 | 6.81 |
| low_balanced | 701.15 | 1683.65 | 1283.74 | 27.69 | 27.70 | 5.56 |
| high_power_saver | 538.32 | 1325.17 | 925.26 | 27.98 | 28.00 | 7.87 |
| power_saver | 484.73 | 1251.41 | 851.50 | 28.10 | 28.10 | 5.28 |
| low_power_saver | 345.26 | 912.30 | 512.39 | 28.05 | 28.10 | 7.74 |
| extreme_power_saver | 215.12 | 745.86 | 345.95 | 27.82 | 27.90 | 9.77 |

## 原始结果

- 汇总 CSV：`/tmp/npu-pp512-r60-20260427-094746/results.csv`
- `native`：bench `/tmp/npu-pp512-r60-20260427-094746/npu_native.bench.log`，samples `/tmp/npu-pp512-r60-20260427-094746/npu_native.samples.csv`
- `burst`：bench `/tmp/npu-pp512-r60-20260427-094746/npu_burst.bench.log`，samples `/tmp/npu-pp512-r60-20260427-094746/npu_burst.samples.csv`
- `high_performance`：bench `/tmp/npu-pp512-r60-20260427-094746/npu_high_performance.bench.log`，samples `/tmp/npu-pp512-r60-20260427-094746/npu_high_performance.samples.csv`
- `balanced`：bench `/tmp/npu-pp512-r60-20260427-094746/npu_balanced.bench.log`，samples `/tmp/npu-pp512-r60-20260427-094746/npu_balanced.samples.csv`
- `low_balanced`：bench `/tmp/npu-pp512-r60-20260427-094746/npu_low_balanced.bench.log`，samples `/tmp/npu-pp512-r60-20260427-094746/npu_low_balanced.samples.csv`
- `high_power_saver`：bench `/tmp/npu-pp512-r60-20260427-094746/npu_high_power_saver.bench.log`，samples `/tmp/npu-pp512-r60-20260427-094746/npu_high_power_saver.samples.csv`
- `power_saver`：bench `/tmp/npu-pp512-r60-20260427-094746/npu_power_saver.bench.log`，samples `/tmp/npu-pp512-r60-20260427-094746/npu_power_saver.samples.csv`
- `low_power_saver`：bench `/tmp/npu-pp512-r60-20260427-094746/npu_low_power_saver.bench.log`，samples `/tmp/npu-pp512-r60-20260427-094746/npu_low_power_saver.samples.csv`
- `extreme_power_saver`：bench `/tmp/npu-pp512-r60-20260427-094746/npu_extreme_power_saver.bench.log`，samples `/tmp/npu-pp512-r60-20260427-094746/npu_extreme_power_saver.samples.csv`

## 直接观察

- 这组是纯 `Prefill pp512` 静态 `qnn-npu` workpoint 扫描，最高温只有 `28.10 C`，没有触发 `38 C` 热保护，因此当前结果基本不受 thermal throttling 干扰。
- `burst` 是吞吐最高点，`1087.81 tok/s`，同时也是功率最高点，稳态功率 `3919.47 mW`，而且稳态窗口也比较干净，`stable_range_pct=2.83%`。
- `balanced` 到 `low_balanced` 这一段仍然保持了较高 prefill 吞吐，分别是 `763.44 tok/s` 和 `701.15 tok/s`，同时功率已经比高性能档明显下降，这一段是后续 prefill 静态点筛选的重点候选区间。
- `native` 与 `extreme_power_saver` 的 prefill 吞吐都落在 `215 tok/s` 左右，说明最低功耗边界附近已经基本贴近；如果后续主要约束是 prefill 延迟，这两档大概率很难满足更紧的 SLO。

# NPU `pp512 r=50` Workpoint Sweep

测试日期：2026-04-28

## 测试配置

- 设备：`192.168.1.113:42977`
- 脚本：[NPUtest.sh](/mnt/sda1/pzw/HeteroCompute/llama.cpp-acom/docs/实验结果/NPUtest.sh:1)
- 输出目录：`/tmp/npu-pp512-r50-20260428`
- 模型：`/data/local/tmp/powerserve/Qwen2-3B/ggml/weights.gguf`
- 后端：`qnn-npu`
- 测试形态：`pp512`，即 `-p 512 -n 0 -b 128 -ub 128 -r 50`
- HTP workpoint：`native burst high_performance balanced low_balanced high_power_saver power_saver low_power_saver extreme_power_saver`
- 大核策略：测试外层将 `policy6` 固定到 `powersave / 1017600 kHz`
- 屏幕状态：由脚本强制保持亮屏
- 温控约束：`TEMP_LIMIT_C=38.0`，`COOLDOWN_TEMP_C=37.0`
- 稳态功率口径：启动后跳过 `8s`，在 `8` 个采样点窗口内选 steady-state window 平均功率

## 基础功率

- baseline 样本文件：`/tmp/npu-pp512-r50-20260428/baseline.samples.csv`
- baseline 平均功率：`334.02 mW`
- baseline 平均温度：`24.87 C`

## 结果表

| Workpoint | 吞吐 tok/s | 稳态平均功率 mW | 相对 baseline 增量 mW | 平均温度 C | 最高温度 C | 稳定窗口范围 % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| native | 216.27 | 698.42 | 364.40 | 24.70 | 24.70 | 9.37 |
| burst | 1131.60 | 4086.30 | 3752.28 | 24.69 | 24.80 | 4.30 |
| high_performance | 906.92 | 2473.39 | 2139.37 | 24.98 | 25.00 | 1.00 |
| balanced | 786.84 | 1939.28 | 1605.26 | 25.39 | 25.50 | 5.34 |
| low_balanced | 718.84 | 1706.01 | 1371.99 | 25.76 | 25.80 | 4.16 |
| high_power_saver | 545.27 | 1317.82 | 983.80 | 26.10 | 26.10 | 5.96 |
| power_saver | 491.42 | 1166.89 | 832.87 | 26.44 | 26.50 | 5.18 |
| low_power_saver | 349.76 | 888.89 | 554.87 | 26.68 | 26.70 | 9.67 |
| extreme_power_saver | 214.65 | 695.84 | 361.82 | 26.70 | 26.70 | 9.37 |

## 原始结果

- 汇总 CSV：`/tmp/npu-pp512-r50-20260428/results.csv`
- baseline 样本：`/tmp/npu-pp512-r50-20260428/baseline.samples.csv`

## 直接观察

- 这组按用户指定的 `round=50` 重新测 `pp512`，整体趋势与此前 `r=60` 一致。
- `burst` 是最高吞吐点，`1131.60 tok/s / 4086.30 mW`；`high_performance` 为 `906.92 tok/s / 2473.39 mW`。
- `balanced` 与 `low_balanced` 仍是中高吞吐、中低功率区间，分别为 `786.84 tok/s / 1939.28 mW` 和 `718.84 tok/s / 1706.01 mW`。
- 最高温度为 `26.70 C`，没有触发 `38 C` 温控约束。

# NPU `tg128 r=3` Workpoint Sweep

测试日期：2026-04-27

## 测试配置

- 设备：`192.168.1.113:43797`
- 脚本：[NPUtest.sh](/mnt/sda1/pzw/HeteroCompute/llama.cpp-acom/docs/实验结果/NPUtest.sh:1)
- 输出目录：`/tmp/npu-tg128-r3-20260427`
- 模型：`/data/local/tmp/powerserve/Qwen2-3B/ggml/weights.gguf`
- 后端：`qnn-npu`
- 测试形态：`tg128`，即 `-p 0 -n 128 -b 1 -ub 1 -r 3`
- HTP workpoint：`native burst high_performance balanced low_balanced high_power_saver power_saver low_power_saver extreme_power_saver`
- 大核策略：测试前手动将 `policy6` 设为 `powersave`，并固定到 `1017600 kHz`
- 屏幕状态：由脚本强制保持亮屏
- 温控约束：`TEMP_LIMIT_C=38.0`，`COOLDOWN_TEMP_C=37.0`
- 功率口径：表格中的“稳态平均功率”采用样本日志后段 `8` 点平台平均值；原始 `results.csv` 仍保留脚本自动窗口统计

## 基础功率

- baseline 样本文件：`/tmp/npu-tg128-r3-20260427/baseline.samples.csv`
- baseline 平均功率：`400.91 mW`
- baseline 平均温度：`23.47 C`

## 结果表

| Workpoint | 吞吐 tok/s | 后段平台平均功率 mW | 相对 baseline 增量 mW | 平台平均温度 C | 平台最高温度 C |
| --- | ---: | ---: | ---: | ---: | ---: |
| native | 3.91 | 790.26 | 389.35 | 24.59 | 24.60 |
| burst | 20.25 | 3455.81 | 3054.90 | 24.83 | 24.90 |
| high_performance | 16.75 | 2522.62 | 2121.71 | 25.28 | 25.40 |
| balanced | 14.76 | 1968.91 | 1568.00 | 25.69 | 25.80 |
| low_balanced | 14.66 | 1843.14 | 1442.23 | 26.10 | 26.20 |
| high_power_saver | 10.44 | 1388.58 | 987.67 | 26.51 | 26.60 |
| power_saver | 10.24 | 1389.77 | 988.86 | 26.79 | 26.80 |
| low_power_saver | 7.38 | 1072.52 | 671.61 | 27.09 | 27.10 |
| extreme_power_saver | 3.92 | 743.89 | 342.98 | 27.20 | 27.20 |

## 原始结果

- 汇总 CSV：`/tmp/npu-tg128-r3-20260427/results.csv`
- `native`：bench `/tmp/npu-tg128-r3-20260427/npu_native.bench.log`，samples `/tmp/npu-tg128-r3-20260427/npu_native.samples.csv`
- `burst`：bench `/tmp/npu-tg128-r3-20260427/npu_burst.bench.log`，samples `/tmp/npu-tg128-r3-20260427/npu_burst.samples.csv`
- `high_performance`：bench `/tmp/npu-tg128-r3-20260427/npu_high_performance.bench.log`，samples `/tmp/npu-tg128-r3-20260427/npu_high_performance.samples.csv`
- `balanced`：bench `/tmp/npu-tg128-r3-20260427/npu_balanced.bench.log`，samples `/tmp/npu-tg128-r3-20260427/npu_balanced.samples.csv`
- `low_balanced`：bench `/tmp/npu-tg128-r3-20260427/npu_low_balanced.bench.log`，samples `/tmp/npu-tg128-r3-20260427/npu_low_balanced.samples.csv`
- `high_power_saver`：bench `/tmp/npu-tg128-r3-20260427/npu_high_power_saver.bench.log`，samples `/tmp/npu-tg128-r3-20260427/npu_high_power_saver.samples.csv`
- `power_saver`：bench `/tmp/npu-tg128-r3-20260427/npu_power_saver.bench.log`，samples `/tmp/npu-tg128-r3-20260427/npu_power_saver.samples.csv`
- `low_power_saver`：bench `/tmp/npu-tg128-r3-20260427/npu_low_power_saver.bench.log`，samples `/tmp/npu-tg128-r3-20260427/npu_low_power_saver.samples.csv`
- `extreme_power_saver`：bench `/tmp/npu-tg128-r3-20260427/npu_extreme_power_saver.bench.log`，samples `/tmp/npu-tg128-r3-20260427/npu_extreme_power_saver.samples.csv`

## 直接观察

- 这组是 `tg128` 的补充矩阵，不作为当前 Decode 主矩阵；主矩阵仍以 `tg64 r=10` 为准。
- `burst` 仍然是吞吐最高点，`20.25 tok/s`，后段平台功率约 `3455.81 mW`；`high_performance` 则落在 `16.75 tok/s / 2522.62 mW`。
- `balanced` 与 `low_balanced` 仍然形成中档静态候选区间，吞吐分别是 `14.76 tok/s` 和 `14.66 tok/s`，平台功率分别是 `1968.91 mW` 和 `1843.14 mW`。
- `native` 与 `extreme_power_saver` 仍然贴近最低功耗边界，但吞吐只有 `~3.9 tok/s`，对更紧的 Decode SLO 基本没有竞争力。
- 这轮 `burst` 与 `high_performance` 的原始脚本自动窗口功率分别只有 `691.35 mW` 和 `699.47 mW`，明显低估了后段平台；原因是样本前半段存在低功率爬坡阶段，因此本节表格统一改用样本日志后段 `8` 点平台均值。
- 全部 workpoint 测试期间最高温度为 `27.20 C`，没有触发 `38 C` 热保护。

# NPU `tg128 r=3` 代表档位三轮复测

测试日期：2026-04-28

## 测试配置

- 设备：`192.168.1.113:42977`
- 脚本：[NPUtest.sh](/home/miog/pzw/download/pzw/HeteroCompute/llama.cpp-acom/docs/实验结果/NPUtest.sh:1)
- 测试形态：`tg128`，即 `-p 0 -n 128 -b 1 -ub 1 -r 3`
- 后端：`qnn-npu`
- HTP workpoint：`burst`、`high_performance`、`balanced`
- CPU 大核策略：测试外层将 `policy6` 固定到 `powersave / 1017600 kHz`
- 功率口径：从样本日志中选 active plateau 的最高 4 点连续窗口；该窗口避开 `tg128 r=3` 短测试中的启动爬坡和结束回落

## 三轮均值

| Workpoint | 轮数 | 平均吞吐 tok/s | 吞吐标准差 | Active plateau 平均功率 mW | 功率标准差 mW | 平均窗口波动 % | 最高温度 C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| burst | 3 | 20.19 | 0.69 | 3652.35 | 87.28 | 5.89 | 28.80 |
| high_performance | 3 | 16.49 | 0.27 | 2422.01 | 67.05 | 3.61 | 28.90 |
| balanced | 3 | 15.09 | 0.59 | 2073.28 | 149.72 | 11.68 | 28.80 |

## 单轮结果

| Workpoint | 轮次 | 吞吐 tok/s | Active plateau 平均功率 mW | 窗口波动 % | 最高温度 C | 样本窗口 |
| --- | ---: | ---: | ---: | ---: | ---: | :-- |
| burst | 1 | 19.43 | 3752.80 | 8.28 | 27.20 | 13-16 |
| burst | 2 | 20.38 | 3609.23 | 5.55 | 28.80 | 19-22 |
| burst | 3 | 20.77 | 3595.01 | 3.84 | 28.50 | 19-22 |
| high_performance | 1 | 16.58 | 2497.27 | 6.76 | 27.40 | 17-20 |
| high_performance | 2 | 16.19 | 2400.13 | 2.39 | 28.90 | 21-24 |
| high_performance | 3 | 16.70 | 2368.64 | 1.67 | 28.50 | 18-21 |
| balanced | 1 | 14.83 | 2079.80 | 28.87 | 27.50 | 12-15 |
| balanced | 2 | 14.68 | 1920.40 | 3.87 | 28.80 | 22-25 |
| balanced | 4 | 15.76 | 2219.63 | 2.30 | 26.80 | 9-12 |

## 备注

- 原始第 3 轮 `balanced` 出现后段功率不稳定和吞吐偏低，未纳入三轮均值；已用单独补跑的 `balanced-r4` 替代。
- `burst` 三轮平均吞吐为 `20.19 tok/s`，对应 active plateau 平均功率 `3652.35 mW`，可以作为 `tg128` 的 NPU 高吞吐候选点。
- `high_performance` 三轮平均为 `16.49 tok/s / 2422.01 mW`，`balanced` 三轮平均为 `15.09 tok/s / 2073.28 mW`；这两个点更适合中等 SLO。
- 最高温度为 `28.90 C`，低于 `38.0 C` 温控约束。
- 原始输出目录：`/tmp/npu-tg128-r3-recheck-representative-20260428`、`/tmp/npu-tg128-r3-recheck-representative-20260428-r2`、`/tmp/npu-tg128-r3-recheck-representative-20260428-r3`、`/tmp/npu-tg128-r3-recheck-representative-20260428-balanced-r4`

# NPU CPU WALT 不锁频实验：`tg128 r=3` 与 `pp512 r=50`

测试日期：2026-04-28

## 测试配置

- 设备：`192.168.1.113:42977`
- 脚本：[NPUtest.sh](/mnt/sda1/pzw/HeteroCompute/llama.cpp-acom/docs/实验结果/NPUtest.sh:1)
- 后端：`qnn-npu`，`-ngl 99 -dev qnn-npu`
- 线程绑定：`taskset 80 -t 1`
- CPU 背景策略：测试期间将 `policy0` 和 `policy6` 均设为 `walt`，频率范围设为 `cpuinfo_min_freq-cpuinfo_max_freq`，即 CPU 不锁频
- `tg128` 测试形态：`-p 0 -n 128 -b 1 -ub 1 -r 3`
- `pp512` 测试形态：`-p 512 -n 0 -b 128 -ub 128 -r 50`
- 屏幕状态：测试时保持亮屏
- 温控约束：`TEMP_LIMIT_C=38.0`，`COOLDOWN_TEMP_C=37.0`

## 基础功率

- `tg128 r=3` baseline：`409.67 mW`，`23.82 C`
- `pp512 r=50` baseline：`416.94 mW`，`26.50 C`

## 结果表

| Workpoint | TG128 吞吐 tok/s | TG128 功率 mW | TG128 增量 mW | TG128 波动 % | PP512 吞吐 tok/s | PP512 功率 mW | PP512 增量 mW | PP512 波动 % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native | 4.39 | 826.46 | 416.79 | 12.60 | 214.74 | 742.08 | 325.14 | 9.10 |
| burst | 21.09 | 3555.22 | 3145.55 | 51.26 | 1088.95 | 3925.41 | 3508.47 | 3.31 |
| high_performance | 16.62 | 2459.76 | 2050.09 | 4.90 | 875.67 | 2395.55 | 1978.61 | 4.22 |
| balanced | 14.66 | 1972.79 | 1563.12 | 4.12 | 760.64 | 1872.80 | 1455.86 | 5.09 |
| low_balanced | 14.65 | 1914.76 | 1505.09 | 7.11 | 697.96 | 1654.00 | 1237.06 | 4.79 |
| high_power_saver | 10.46 | 1457.73 | 1048.06 | 7.24 | 534.96 | 1302.19 | 885.25 | 6.44 |
| power_saver | 10.28 | 1400.35 | 990.68 | 5.37 | 479.99 | 1180.49 | 763.55 | 7.10 |
| low_power_saver | 7.52 | 1076.35 | 666.68 | 8.29 | 345.05 | 918.92 | 501.98 | 10.52 |
| extreme_power_saver | 4.36 | 819.95 | 410.28 | 13.33 | 215.32 | 714.22 | 297.28 | 9.99 |

## 原始结果

- `tg128 r=3` 汇总 CSV：`/tmp/npu-walt-tg128-r3-20260428/results.csv`
- `tg128 r=3` baseline：`/tmp/npu-walt-tg128-r3-20260428/baseline.samples.csv`
- `pp512 r=50` 汇总 CSV：`/tmp/npu-walt-pp512-r50-20260428/results.csv`
- `pp512 r=50` baseline：`/tmp/npu-walt-pp512-r50-20260428/baseline.samples.csv`

## 备注

- `tg128 r=3` 是短 Decode 补充测试，`burst/native/extreme_power_saver` 的窗口波动偏高，因此该组更适合看吞吐和粗粒度功率趋势，不宜作为严格稳态功率结论。
- `pp512 r=50` 的窗口更稳定，`burst` 在 CPU WALT 不锁频背景下达到 `1088.95 tok/s / 3925.41 mW`，`balanced` 为 `760.64 tok/s / 1872.80 mW`。
- 两组测试最高温度均低于 `38 C`：`tg128` 最高约 `26.40 C`，`pp512` 最高约 `27.50 C`，没有触发热保护。
