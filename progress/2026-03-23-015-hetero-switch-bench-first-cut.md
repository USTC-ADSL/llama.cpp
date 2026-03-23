# 任务 015：补齐 `hetero-switch-bench` 第一刀

日期：2026-03-23

## 背景与目标

在 `P4-1/P4-2` 之后，当前主线已经基本确认：

- decode / prefill 的大 gap 都不是外层 scheduler 显式 copy 主导。

但还缺一组系统外对照，来回答：

- shared-host 和 memcpy 在独立微基准里到底差多少；
- shared-host 是否真的既快又正确。

## 本次执行

### 1. 按规范重建后补跑 `hetero-switch-bench`

- 构建：
  - `./build-npu-opencl.sh build-qnn-prof-db arm64-android-snapdragon-release --without-npu --with-gpu --with-qnn --with-profiling`
- 设备：
  - `db6c02cf`
- 运行：
  - `LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./hetero-switch-bench --warmup 5 --iters 50 --sizes 7168,65536,1048576,16777216 --csv ...`

这轮特意加入了 `7168 B`，因为它正好对应 `P4-2` 里唯一显式 scheduler copy 的总量。

### 2. 得到了一个很关键的 asymmetric 结果

`host_write_to_opencl_read`：

- shared-host 与 memcpy 几乎同量级
- 且二者都 `50 / 50` 有效

`opencl_write_to_host_read`：

- shared-host 表面上经常更快
- 但 `4` 个尺寸全部都是 `0 / 50` 校验通过
- memcpy 则全部 `50 / 50` 通过

## 关键结论

### 1. 外层 copy 的量级确实不足以解释系统级 gap

在 `7168 B` 上：

- shared-host vs memcpy 的差只有 `0.283 us`

因此这进一步支持：

- `P4-1/P4-2` 里的大 gap 不是外层 scheduler copy 主导

### 2. raw shared-host 不能被当作“免费且正确”的 readback 路径

这轮最值得记住的不是 shared-host 有时更快，而是：

- `OpenCL -> host` 方向上它当前根本不正确

这意味着系统里任何正确的 shared-host readback，都仍可能需要：

- coherence
- sync
- materialization
- copyback

也就是说：

- shared-host 不是天然零成本；
- 只是它的真实代价可能沉在 backend 内部，而不是体现在外层 scheduler `tensor_copy` 上。

## 产出

- `docs/qnn-attn-core-shared/db6c02cf-hetero-switch-bench-2026-03-23.md`
- `docs/qnn-attn-core-shared/db6c02cf-hetero-switch-bench-2026-03-23.csv`
- 原始文件：
  - `tmp/p44switch/db6c02cf_hetero_switch_bench_20260323.log`
  - `tmp/p44switch/db6c02cf_hetero_switch_bench_20260323.csv`

## 对后续任务的影响

1. `P3-3/P3-4` 写阶段矩阵时，不能把 shared-host 简化成“无代价无风险”。
2. decode / prefill 后续优化仍应优先放在：
   - fragment 数量
   - direct-bind
   - backend 内部 materialization
3. cost model 若要纳入 shared-host，必须显式考虑 correctness / materialization 成本，而不是只看外层 copy 次数。
