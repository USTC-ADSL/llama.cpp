# `db6c02cf` Dynamic Cost-Model Validation

更新日期：2026-03-23

## 目标

验证 `COST_MODEL_RESERVED + slo_us` 的第一版实现是否已经进入真实决策路径，而不再只是配置占位。

这项验证关注的是：

1. runtime 是否会枚举 `decode / fallback / base` 候选；
2. cost-model trace 是否会打印每个候选的 estimate；
3. runtime 是否会根据 estimate 选择不同 route；
4. `slo_us` 是否已经进入决策结果，而不是仅存在于配置结构中。

## 配置

- 设备：`db6c02cf`
- 模型：`Qwen2-0.5B`
- 命令口径：`--no-warmup -p 16 -n 1 -b 16 -ub 16 -c 2048 -t 1 -ngl 0 -dev GPUOpenCL`
- 静态 base route：
  - `attn_proj=opencl,attn_core=opencl,attn_out=opencl,ffn=opencl,output=opencl`
- 动态 decode route：
  - `attn_proj=cpu,attn_core=qnn-npu,attn_out=qnn-npu,ffn=cpu,output=cpu`
  - `decode_kv=qnn`
- fallback route：
  - `attn_proj=cpu,attn_core=cpu,attn_out=cpu,ffn=cpu,output=cpu`
- trace 开关：
  - `GGML_HETERO_DYNAMIC_MODE=cost-model`
  - `GGML_HETERO_DYNAMIC_TRACE=1`

原始日志：

- `tmp/p17/dyn-cost-decode.log`
- `tmp/p17/dyn-cost-fallback.log`
- `tmp/p17/dyn-cost-over-slo.log`

## 实验 1：decode route 被选中

环境覆盖：

- `GGML_HETERO_DYNAMIC_DECODE_EST_US=1500`
- `GGML_HETERO_DYNAMIC_FALLBACK_EST_US=2500`
- `GGML_HETERO_DYNAMIC_BASE_EST_US=4500`
- `GGML_HETERO_DYNAMIC_SLO_US=2000`

decode 阶段 trace：

- `cost candidate decode estimate_us=1500`
- `cost candidate fallback estimate_us=2500`
- `cost candidate base estimate_us=4500`
- `cost selected decode estimate_us=1500 slo_us=2000 reason=cost-decode-route`

结论：

- runtime 已经在 decode 时选择动态 decode 候选，而不是继续停留在 base route。

## 实验 2：fallback route 被选中

环境覆盖：

- `GGML_HETERO_DYNAMIC_DECODE_EST_US=3500`
- `GGML_HETERO_DYNAMIC_FALLBACK_EST_US=1800`
- `GGML_HETERO_DYNAMIC_BASE_EST_US=4500`
- `GGML_HETERO_DYNAMIC_SLO_US=2000`

decode 阶段 trace：

- `cost candidate decode estimate_us=3500`
- `cost candidate fallback estimate_us=1800`
- `cost candidate base estimate_us=4500`
- `cost selected fallback estimate_us=1800 slo_us=2000 reason=cost-fallback-route`

结论：

- runtime 已经可以在同一套 base/decode/fallback 候选中切到 fallback。

## 实验 3：`slo_us` 已进入 reason 分支

环境覆盖：

- `GGML_HETERO_DYNAMIC_DECODE_EST_US=1500`
- `GGML_HETERO_DYNAMIC_FALLBACK_EST_US=2500`
- `GGML_HETERO_DYNAMIC_BASE_EST_US=4500`
- `GGML_HETERO_DYNAMIC_SLO_US=1000`

decode 阶段 trace：

- `cost candidate decode estimate_us=1500`
- `cost candidate fallback estimate_us=2500`
- `cost candidate base estimate_us=4500`
- `cost selected decode estimate_us=1500 slo_us=1000 reason=cost-best-effort-over-slo`

结论：

- `slo_us` 已经不再是死配置。
- 当候选估计值全部超过 `slo_us` 时，runtime 会进入 `cost-best-effort-over-slo` 路径。

## 关键限制

当前第一版 cost model 仍是明显的 first cut，不应过度表述：

1. 这次验证主要依赖 `GGML_HETERO_DYNAMIC_*_EST_US` 覆盖值来强制分离候选顺序。
2. 当前 estimator 只有单目标“预计时延最小”排序。
3. 因为排序目标仍是单调 latency，`slo_us` 当前更多是在区分：
   - “候选满足 SLO”
   - “没有候选满足 SLO，于是退化为 best effort”

而不是在 latency winner 之外引入新的 Pareto 选择。

## 判断

### 强结论

- `COST_MODEL_RESERVED` 已经不再是纯占位。
- runtime 现在会：
  - 做候选兼容性过滤；
  - 估计 candidate latency；
  - 打印 trace；
  - 在 `decode / fallback / base` 中做真实选择；
  - 把 `slo_us` 反映到最终 reason。

### 保守结论

- 这还不是研究主线意义上的完整 `SLO-aware` scheduler。
- 更准确的表述应是：
  - “已完成第一版 latency-only cost-model 运行闭环”
  - “`slo_us` 已进入 runtime 决策路径”
  - “但还没有把切换成本和达标率统一进正式调度策略”

## 对主线的意义

- 这次验证把主线 ③ 从“只有 phase heuristic 骨架”推进到了“有真实 route selection 的第一版 cost-model runtime”。
- 它的价值不在于已经得到最优调度，而在于：
  - `cost table -> estimate -> candidate selection -> reason trace`
    这条链路已经可以被真实运行和反复验证了。
