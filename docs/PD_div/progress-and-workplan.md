# Prefill/Decode 后端分离优化：进度与工作计划

> 更新日期：2026-05-08

## 当前范围

本工作区当前只维护 `Prefill -> Decode` phase boundary 上的后端分离与切换优化：

- `Prefill` 与 `Decode` 可分别选择 `qnn-npu`、`GPUOpenCL`、`CPU` 等后端；
- 重点验证跨后端语义正确性、KV handoff、scheduler reserve、alias / materialization 成本；
- 以 `llama-bench -pg <prompt>,<gen>` 这类 combined workload 观察真实 phase switch；
- 用 timing trace 和 benchmark log 解释首次切换与 steady-state 切换开销。

本文件不再维护功耗测试方案、功耗矩阵、能耗表或 power-aware planner 任务。相关旧实验结果和独立方案已经从当前工作区清理。

## 已有证据

当前可直接引用的非功耗证据主要分为三类。

### 1. Prefill/Decode 后端切换

- 根目录 [实验结论.md](/home/miog/yzh/Yzh/llama.cpp/实验结论.md) 记录了 `qnn-npu -> GPUOpenCL` 动态切换验证、首次切换开销分解，以及 OpenCL direct-host-ptr alias 预热实验。
- 关键观察是：切换路径语义可跑通，首次 decode 切换开销可以拆到 `kv_migration`、`alias`、`sched_reserve` 等字段；alias 预热后，首次切换中的 alias 成本可以前移到上下文初始化阶段。

### 2. Decode mixed route 与阶段边界

- [docs/qnn-attn-core-shared](/home/miog/yzh/Yzh/llama.cpp/docs/qnn-attn-core-shared) 下保留了 decode stage backend matrix、route purity、boundary overhead、hetero switch bench 等非功耗材料。
- 这些材料说明 decode 路径中 `attn_proj / attn_core / ffn` 的阶段级后端切换已经有可执行证据，但端到端收益仍受 route purity、fragmentation、scheduler reserve 和 residual tail 影响。

### 3. Prefill split overhead

- `qnn-npu` AoT prefill 的 full-graph 与 split 对照已经证明 split prompt route 可以真实执行。
- 当前主要问题不是 matcher 未命中，而是 runtime overhead：更细粒度 graph launch、fragment materialization、shared-host KV writeback，以及更重的 `qnn-npu-host` buffer 管理。

## 当前工程重点

1. 保持 Prefill/Decode combined workload 可复现，优先使用 `-pg` 而不是拆开的 `-p` / `-n`。
2. 在 phase switch trace 中保留并完善：

   ```text
   route_apply_us
   sched_reserve_us
   kv_migration_us
   kv_alias_us
   graph_rebuild_us
   decode_entry_us
   first_token_gap_us
   post_switch_tbt_us
   ```

3. 优化 `qnn-npu -> GPUOpenCL` 的 KV handoff 和 OpenCL alias 路径，避免把一次性 alias 成本计入首个 decode token。
4. 收缩 scheduler reserve / graph reserve 开销，区分首次切换与 steady-state 切换。
5. 对 Prefill split route 做更粗粒度 AoT family 或 direct-bind 命中率优化，只在已有 trace 证明瓶颈后再改 runtime。

## 非目标

以下内容不属于当前工作区默认任务：

- 新增功耗采样脚本；
- 维护 battery current / voltage 采样流程；
- 生成功耗矩阵、能耗表、active-power profile；
- 基于功耗异常构造论文 insight；
- 实现以能耗最小化为主目标的 planner。

如果后续需要恢复功耗实验，应在单独任务中显式说明，并重新建立数据质量规则和输出目录。

## 下一步

当前最有价值的一步是补一组 combined `Prefill -> Decode` 小矩阵：

```text
pp128+tg1
pp128+tg16
pp512+tg1
pp512+tg16
```

每个 workload 至少比较：

```text
qnn-npu -> qnn-npu
qnn-npu -> GPUOpenCL
GPUOpenCL -> GPUOpenCL
CPU -> GPUOpenCL
```

目标是把首次切换开销、steady-state TBT、fallback 情况和 route purity 放在同一个非功耗口径下解释清楚。
