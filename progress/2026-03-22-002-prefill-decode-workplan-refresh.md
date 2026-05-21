# 任务 002：更新 `Prefill/Decode` 双主线工作计划

日期：2026-03-22

## 背景与目标

`AGENTS.md` 已更新为：

- 主线正式覆盖 `Prefill/Decode`
- 但执行优先级仍保持 `Decode` 优先

因此原有 `docs/progress-and-workplan.md` 需要同步：

1. 不能再把 `Prefill` 当作主线外内容。
2. 也不能因此把工作重心从 `Decode` 转移到 `Prefill`。
3. 需要把最近已经拿到的 decode/prefill 证据纳入主文档。

## 执行内容

重写了：

- `docs/progress-and-workplan.md`

本次更新的重点是：

- 将研究主线修正为 `Prefill/Decode` 双主线
- 明确保留 `Decode` 优先级
- 将现有 decode 证据同步入文档：
  - 6 条 decode 验证路线
  - 4 组 share-heavy/no-explicit-copy 的本地补测
- 将现有 prefill 证据同步入文档：
  - split `batch128-only` 已真实执行 `24 x 3` stage graphs
  - full-vs-split prefill 的 warm/cold 对比
  - 当前 gap 已可解释为 runtime overhead 主导
- 重排后续计划为：
  - 先口径与记录体系
  - 再双路径稳定性
  - 再 baseline
  - 再阶段矩阵
  - 再 overhead 分解
  - 再 combined Prefill/Decode phase-switch 复测
  - 最后才是 `SLO-aware` 调度与工程优化

## 当前结论

新的主进度文档已经与当前仓库状态对齐：

- `Prefill` 已进入主线
- `Decode` 仍是第一主路径
- 当前最需要优先完成的不是新策略，而是：
  - 稳定性止血
  - baseline
  - 阶段矩阵
  - runtime overhead 分解

## 下一步

下一项按优先级推进：

1. 收口 `Decode` 最后一层 tail residual unmatched。
2. 确认 `Prefill` 最后一层 `FFN tokens=1` 的会计语义。
3. 将稳定性阶段的结果继续写入 `progress/`。
