# 2026-03-23-017 AoT Bootstrap CPU Scheduler Purity Fix

## 任务

收口 decode `tg1` 路线里 `AoT bootstrap CPU correction` 的 route purity 问题，确认它不再泄漏成第二张大 `OpenCL` correction graph。

## 完成内容

- 在 [src/llama-context.cpp](/mnt/sda1/pzw/HeteroCompute/llama.cpp-acom/src/llama-context.cpp) 中，为 bootstrap correction 引入了临时 CPU-only scheduler。
- 该 scheduler 会在输出读取和 `synchronize()` 完成后恢复 steady-state 主 scheduler。
- 重新构建并在 `db6c02cf` 上按原复现实验口径复测。

## 结果

- `bootstrap-cpu` pass 的 `aot-assign` 已全部变成 `backend=CPU`。
- `tmp/p17/purity-cpu-qnn-cpu-noreuse.csv` 中只剩 `OpenCL = 2` 条事件，且都属于第一张 intended mixed graph 的一个小 residual split。
- 第二张 correction pass 不再出现大规模 `OpenCL` split。

## 判断

- bootstrap correction 的 purity 问题已闭环。
- decode route purity 还没有达到“绝对纯净”，因为 steady-state mixed graph 仍残留 `1` 个小 `OpenCL` split。
- 因此本项更准确的状态是：
  - `bootstrap-cpu` 泄漏已修复；
  - mixed decode steady-state residual 仍需后续单独解释。

## 证据

- 正式记录：`docs/qnn-attn-core-shared/db6c02cf-decode-route-purity-2026-03-23.md`
- 原始日志：`tmp/p17/purity-cpu-qnn-cpu-noreuse.log`
- 原始 CSV：`tmp/p17/purity-cpu-qnn-cpu-noreuse.csv`
