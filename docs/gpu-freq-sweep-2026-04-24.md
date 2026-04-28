# GPU Frequency Sweep Results 2026-04-24

- Device: `192.168.1.113:42451`
- Screen: kept ON during all tests
- Thermal constraint: kept below `38 C`; observed peak was `36.6 C`
- Baseline idle power at test start: `385.31 mW`
- Baseline idle temperature at test start: `34.46 C`
- Power metric in the table below is the average power over the detected active test window, not the full process lifetime
- Note: `pp32 @ 832 MHz` used a rerun because the first run was pulled back to `1100 MHz` during load

| Set GPU MHz | PP32 Actual MHz | PP32 Throughput t/s | PP32 Active Avg Power mW | TG128 r=3 Actual MHz | TG128 r=3 Throughput t/s | TG128 r=3 Active Avg Power mW |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1100 | 1100 | 163.98 | 4101.80 | 1100 | 25.42 | 4942.56 |
| 1050 | 1050 | 157.90 | 3800.19 | 1050 | 25.09 | 4688.43 |
| 967 | 967 | 147.76 | 3078.98 | 967 | 24.46 | 4156.80 |
| 900 | 900 | 131.48 | 2461.69 | 900 | 23.66 | 3683.54 |
| 832 | 832 | 123.55 | 2288.28 | 832 | 22.83 | 3485.73 |
| 734 | 734 | 100.04 | 1760.83 | 734 | 20.55 | 3023.53 |
| 660 | 660 | 89.98 | 1518.69 | 660 | 18.95 | 2633.06 |
| 607 | 607 | 84.60 | 1359.57 | 607 | 17.63 | 2388.54 |
| 525 | 525 | 68.19 | 1209.53 | 525 | 15.03 | 2080.19 |
| 443 | 443 | 60.16 | 1059.67 | 443 | 12.50 | 1711.61 |
| 389 | 389 | 54.72 | 997.12 | 389 | 11.36 | 1588.39 |
| 342 | 342 | 49.53 | 917.02 | 342 | 9.36 | 1337.03 |
| 222 | 222 | 33.46 | 738.00 | 222 | 7.03 | 1077.13 |
| 160 | 160 | 24.94 | 683.90 | 160 | 3.87 | 829.82 |

Raw result locations:

- Prefill sweep: `/tmp/gpu-pp32-all-20260424-1/results.csv`
- Prefill `832 MHz` rerun: `/tmp/gpu-pp32-832-rerun-20260424/results.csv`
- Decode sweep: `/tmp/gpu-tg128-r3-all-20260424-1/results.csv`
- Baseline samples: `/tmp/gpu_sweep_baseline_samples_20260424.csv`

## Decode 64 Round 4 Sweep

- Test date: `2026-04-24`
- Screen: kept ON during all tests
- Thermal constraint: kept below `38 C`; observed peak was `33.0 C`
- Baseline idle power at test start: `443.49 mW`
- Baseline idle temperature at test start: `28.14 C`
- Power metric in the table below is the average power over the detected active test window, not the full process lifetime

| Set GPU MHz | TG64 r=4 Actual MHz | TG64 r=4 Throughput t/s | TG64 r=4 Active Avg Power mW | Delta vs Baseline mW | Active Max Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1100 | 1100 | 25.55 | 4477.66 | 4034.17 | 28.90 |
| 1050 | 1050 | 25.22 | 4259.24 | 3815.75 | 29.30 |
| 967 | 967 | 24.55 | 3793.66 | 3350.17 | 29.70 |
| 900 | 900 | 23.73 | 3131.16 | 2687.67 | 30.20 |
| 832 | 832 | 22.84 | 3441.21 | 2997.72 | 30.70 |
| 734 | 734 | 20.59 | 2967.99 | 2524.50 | 31.00 |
| 660 | 660 | 18.99 | 2626.00 | 2182.51 | 31.40 |
| 607 | 607 | 17.63 | 2187.94 | 1744.45 | 31.70 |
| 525 | 525 | 15.03 | 2037.63 | 1594.14 | 32.00 |
| 443 | 443 | 12.55 | 1676.32 | 1232.83 | 32.30 |
| 389 | 389 | 11.28 | 1629.27 | 1185.78 | 32.50 |
| 342 | 342 | 9.33 | 1278.69 | 835.20 | 32.70 |
| 222 | 222 | 7.02 | 1042.60 | 599.11 | 32.90 |
| 160 | 160 | 3.86 | 781.33 | 337.84 | 33.00 |

Raw result locations:

- Decode `tg64 r=4` sweep: `/tmp/gpu-tg64-r4-all-20260424-1/results.csv`
- Decode `tg64 r=4` baseline samples: `/tmp/gpu_sweep_baseline_samples_20260424_tg64r4.csv`
