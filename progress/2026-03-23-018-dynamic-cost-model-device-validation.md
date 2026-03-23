# 2026-03-23-018 Dynamic Cost-Model Device Validation

## 任务

为 `COST_MODEL_RESERVED + slo_us` 补上设备侧运行闭环，确认它已经进入真实 route decision，而不是继续停留在配置占位。

## 完成内容

- 使用 `db6c02cf`、`Qwen2-0.5B`、`llama-bench --no-warmup -p 16 -n 1` 验证动态路由。
- 设置：
  - base route = `opencl-all`
  - decode route = `cpu + qnn attn_core`
  - fallback route = `cpu-all`
- 通过 `GGML_HETERO_DYNAMIC_*_EST_US` 覆盖值，构造可分离的候选顺序。

## 结果

- 已验证 decode 路线可被 `cost-model` 选中。
- 已验证 fallback 路线可被 `cost-model` 选中。
- 已验证当所有候选都超过 `slo_us` 时，reason 会进入 `cost-best-effort-over-slo`。

## 限制

- 当前 first cut 主要是 `latency-only` 选择器。
- `slo_us` 已进入 runtime 决策路径，但在当前实现下更多体现为：
  - “满足 SLO” vs
  - “best effort over SLO”

而不是完整的多目标调度策略。

## 判断

- `P6-2/P6-3` 的最小运行闭环已经补齐。
- 这可以支持后续继续接入更真实的 cost table、switch overhead 和达标率实验。

## 证据

- 正式记录：`docs/qnn-attn-core-shared/db6c02cf-dynamic-cost-model-validation-2026-03-23.md`
- 原始日志：
  - `tmp/p17/dyn-cost-decode.log`
  - `tmp/p17/dyn-cost-fallback.log`
  - `tmp/p17/dyn-cost-over-slo.log`
