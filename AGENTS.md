# AGENT.md

## Role

You are assisting with experiments and lightweight implementation for a mobile LLM inference systems paper.

The current paper direction is:

> Mobile LLM inference on a heterogeneous SoC should select a composite execution state under TTFT/TBT or throughput SLOs. The runtime should use offline profiling, Pareto frontier pruning, and transition-aware online selection.

The current `Host-policy coupling` idea is no longer valid and must not be used as a paper insight. The previous anomalous high-power data was caused by external scene/tool behavior that disabled or disturbed normal CPU regulation. Do not build new claims from that artifact.

The new target Insight B is:

> **Insight B: Frontier selection is a runtime problem because decode behavior and transition profitability depend on context length, remaining output length, and transition overhead.**

The goal is to help test and implement this idea conservatively.

---

## Non-negotiable rules

1. Do not fabricate data.
2. Do not silently drop failed runs.
3. Do not overwrite existing experiment results.
5. Do not change system governors globally unless the experiment explicitly requests it and the script restores the original state.
6. Do not hardcode ADB device IDs. Use `DEVICE` from the environment.
7. Do not hardcode model paths. Use `MODEL_PATH` from the environment.
8. Keep the screen awake during tests if the existing scripts already do so.
9. Respect the temperature limit. Default:
   ```bash
   TEMP_LIMIT_C=38.0
   COOLDOWN_TEMP_C=37.0
````

10. Every experiment must save:

    * raw benchmark log,
    * raw power samples,
    * summary CSV,
    * summary Markdown,
    * exact command line,
    * git commit hash if available.

---

## Current main baseline

The current stable main characterization uses:

* Model: use `MODEL_PATH`; do not assume a specific model name.
* Decode main workload: `tg64 r=10`.
* Decode supplemental workload: `tg128 r=3`.
* Prefill main workload: `pp512`.
* Backends:

  * NPU: `qnn-npu`
  * GPU: `GPUOpenCL`
  * CPU: `-ngl 0`
* Key NPU workpoints:

  * `low_balanced`
  * `balanced`
  * `high_performance`
  * `burst`
* Key GPU frequencies:

  * `734 MHz`
  * `967 MHz`
  * `1100 MHz`
* Key CPU state:

  * `big2` with stable controllable frequency.
  * For `tg64`, use `2649600 kHz` as the safe stable big2 point unless the experiment explicitly targets another point.
  * Avoid claiming high CPU frequencies are controllable if sample logs show frequency fallback.

---

## What to implement

Implement only minimal instrumentation and scripts needed for Insight B.

Allowed changes:

1. Add experiment scripts under:

   ```text
   docs/实验结果/
   docs/experiments/
   scripts/
   tools/
   ```
2. Add lightweight parsers for power logs and benchmark logs.
3. Add optional trace instrumentation for transition timing if existing logs do not expose enough fields.
4. Add a lightweight frontier planner prototype if requested.
5. Add CSV/JSON profile tables.
6. Add Markdown summaries.

Avoid large changes to the inference core unless explicitly necessary for instrumentation.

---

## Insight B experiment goals

Insight B should answer two questions:

### Q1. Does the decode frontier change with effective context length?

We need to test whether the cheapest feasible decode state changes as the context length grows.

A decode benchmark with `-p 0 -n 64` is not enough. It measures decode without a long KV context. We need workloads like:

```text
prefill context length L, then decode 64 tokens
```

The measurement should isolate or clearly report the decode phase after the context has been built.

Recommended context lengths:

```text
0, 512, 2048, 4096
```

Optional if time permits:

```text
8192
```

Recommended states:

```text
NPU low_balanced
NPU burst
GPU 734 MHz
GPU 967 MHz
GPU 1100 MHz
CPU big2 2649600 kHz
```

Optional NPU states:

```text
NPU balanced
NPU high_performance
```

For each state and context length, record:

```text
model
backend
state_id
context_len
decode_tokens
rounds
throughput_tps
tbt_us
active_power_mw
energy_mj_per_token
temperature_avg_c
temperature_max_c
stable_range_pct
actual_gpu_freq_mhz
actual_cpu_freq_khz
raw_log_path
sample_path
```

Compute:

```text
energy_mj_per_token = active_power_mw / throughput_tps
tbt_us = 1e6 / throughput_tps
```

The output table must be:

```text
results/insightB/context_decode_profile.csv
```

---

### Q2. When is a transition worth it?

Even if a target state is lower energy per token, switching is only useful when the remaining output length amortizes the transition cost.

Measure transition overheads between representative states.

Required transitions:

```text
NPU burst -> NPU low_balanced
NPU low_balanced -> NPU burst

NPU burst -> GPU 734
NPU burst -> GPU 967

GPU 734 -> GPU 967
GPU 967 -> GPU 734

GPU 734 -> NPU low_balanced
GPU 967 -> NPU low_balanced

NPU burst -> CPU big2
CPU big2 -> GPU 967
```

For each transition, collect:

```text
from_state
to_state
context_len
decode_tokens_before_switch
decode_tokens_after_switch
decision_us
route_apply_us
policy_apply_us
qnn_workpoint_apply_us
gpu_freq_apply_us
sched_reserve_us
kv_handoff_us
graph_rebuild_us
decode_entry_us
total_blocking_us
first_token_gap_us
post_switch_tbt_us
switch_success
fallback_used
raw_log_path
```

If some fields are not available, add trace points rather than guessing.

Minimum required fields:

```text
from_state
to_state
total_blocking_us
first_token_gap_us
kv_handoff_us
post_switch_tbt_us
switch_success
```

The output table must be:

```text
results/insightB/transition_cost.csv
```

---

## Context-length decode experiment

Create a script:

```text
scripts/run_insightB_context_frontier.sh
```

The script should accept:

```bash
DEVICE
MODEL_PATH
OUTPUT_DIR
TEMP_LIMIT_C
COOLDOWN_TEMP_C
CONTEXT_LIST
DECODE_TOKENS
ROUNDS
```

Example command:

```bash
DEVICE=192.168.1.113:42977 \
MODEL_PATH=/data/local/tmp/powerserve/Qwen2-3B/ggml/weights.gguf \
OUTPUT_DIR=/tmp/insightB-context-frontier-$(date +%Y%m%d-%H%M%S) \
CONTEXT_LIST="0 512 2048 4096" \
DECODE_TOKENS=64 \
ROUNDS=5 \
TEMP_LIMIT_C=38.0 \
COOLDOWN_TEMP_C=37.0 \
bash scripts/run_insightB_context_frontier.sh
```

The script should test the following states:

```text
npu_low_balanced
npu_burst
gpu_734
gpu_967
gpu_1100
cpu_big2_2649
```

Use existing GPU/NPU/CPU test scripts when possible. Do not duplicate logic unnecessarily.

If existing benchmark scripts cannot isolate decode after a nonzero context, add a clear phase marker in the benchmark output:

```text
PHASE_BEGIN decode
PHASE_END decode
```

Then parse only the decode segment.

If phase isolation is not possible, report the measurement as end-to-end and mark:

```text
phase_isolated = 0
```

Do not label end-to-end measurements as decode-only.

---

## Transition overhead experiment

Create a script:

```text
scripts/run_insightB_transition_overhead.sh
```

The script should accept:

```bash
DEVICE
MODEL_PATH
OUTPUT_DIR
CONTEXT_LEN
DECODE_TOKENS_BEFORE_SWITCH
DECODE_TOKENS_AFTER_SWITCH
TRANSITION_LIST
ROUNDS
```

Example command:

```bash
DEVICE=192.168.1.113:42977 \
MODEL_PATH=/data/local/tmp/powerserve/Qwen2-3B/ggml/weights.gguf \
OUTPUT_DIR=/tmp/insightB-transition-$(date +%Y%m%d-%H%M%S) \
CONTEXT_LEN=512 \
DECODE_TOKENS_BEFORE_SWITCH=16 \
DECODE_TOKENS_AFTER_SWITCH=64 \
ROUNDS=5 \
bash scripts/run_insightB_transition_overhead.sh
```

The runtime must emit transition trace lines like:

```text
TRANSITION_TRACE from=npu_burst to=gpu_734 decision_us=... route_apply_us=... kv_handoff_us=... graph_rebuild_us=... total_blocking_us=... first_token_gap_us=... post_switch_tbt_us=... success=1
```

If the current code already emits fields such as:

```text
route_decide_us
route_apply_us
reserve_us
kv_migration_us
bootstrap_sync_us
bootstrap_sched_rebuild_us
```

reuse them and map them into the transition CSV.

Do not rename existing trace fields unless necessary.

---

## Amortization experiment

Create a script:

```text
scripts/run_insightB_amortization.sh
```

Goal: show when switching is or is not worth it.

Use one or two representative scenarios:

### Scenario A: relaxed SLO

```text
Prefill: NPU burst, pp512
Decode choices:
  - stay NPU low_balanced
  - switch GPU 734
  - transition-aware planner
Output lengths:
  16, 32, 64, 128, 256
```

### Scenario B: tighter SLO

```text
Prefill: NPU burst, pp512
Decode choices:
  - NPU burst
  - GPU 967
  - CPU big2 2649600
  - transition-aware planner
Output lengths:
  16, 32, 64, 128, 256
```

Record:

```text
strategy
output_len
prefill_state
decode_state
total_latency_ms
ttft_ms
avg_tbt_ms
energy_mj
slo_met
transition_count
transition_time_ms
```

Output:

```text
results/insightB/amortization.csv
```

If direct energy integration is not available, estimate:

```text
energy_total =
  prefill_energy +
  decode_tokens * decode_energy_per_token +
  transition_energy_estimate
```

Use:

```text
transition_energy_estimate = target_active_power_mw * transition_time_ms / 1000
```

Clearly mark:

```text
energy_source = measured
```

or

```text
energy_source = estimated
```

Never mix measured and estimated values without marking them.

---

## Planner prototype

If asked to implement a planner, implement:

```text
Context- and Amortization-Aware Frontier Selection
```

Use a profile table:

```text
results/insightB/context_decode_profile.csv
```

and a transition table:

```text
results/insightB/transition_cost.csv
```

Planner input:

```cpp
struct RuntimeCtx {
    std::string model;
    std::string current_state;
    int context_len;
    int remaining_tokens;
    double slo_tbt_us;
    double switch_slack_us;
};
```

Planner output:

```cpp
struct PlannerDecision {
    std::string target_state;
    bool should_switch;
    std::string reason;
    double expected_energy_mj;
    double expected_transition_us;
};
```

Algorithm:

```text
1. Bucket context length.
2. Load candidate states from the profile table.
3. Filter states with TBT > SLO_TBT.
4. Estimate total energy:
     remaining_tokens * energy_mj_per_token + transition_energy_mj
5. Pick the lowest-energy feasible state.
6. If current state is feasible, switch only if:
     energy_saving > transition_energy + guard
   and:
     transition_time <= switch_slack
7. Apply hysteresis:
     - immediate upshift on SLO violation
     - conservative downshift only after stable slack window
     - minimum switch interval in tokens
```

Do not use RL.
Do not use online ILP.
Do not use a large optimizer.
This should be table-driven and lightweight.

---

## Planner overhead benchmark

Create:

```text
scripts/bench_frontier_planner.py
```

Run 1000 or more planner decisions using synthetic contexts:

```text
context_len in [0, 512, 2048, 4096]
remaining_tokens in [8, 16, 32, 64, 128, 256]
slo_tbt_us from relaxed to tight
```

Report:

```text
median_us
p95_us
p99_us
max_us
```

Output:

```text
results/insightB/planner_overhead.csv
```

The expected result should be microsecond-level or low-millisecond-level. If it is slower, explain why.

---

## Data quality rules

For each experimental condition:

1. Prefer 3 rounds minimum.
2. If standard deviation is high, do not hide it.
3. If power window fluctuation is high, mark the point as unstable.
4. If CPU or GPU fails to hold requested frequency, mark:

   ```text
   freq_stable = 0
   ```
5. Do not use unstable high-frequency points as controllable operating points.
6. If a state is infeasible under the SLO, mark it as infeasible rather than assigning a fake score.
7. Keep baseline power separate from active power.
8. For paper tables, use active plateau power unless explicitly doing whole-session energy.

---

## Required CSV schemas

### `context_decode_profile.csv`

```text
date,model,backend,state_id,context_len,decode_tokens,rounds,throughput_tps,throughput_std,active_power_mw,power_std,energy_mj_per_token,tbt_us,temp_max_c,stable_range_pct,freq_stable,raw_log_path,sample_path
```

### `transition_cost.csv`

```text
date,model,context_len,from_state,to_state,rounds,total_blocking_us,first_token_gap_us,kv_handoff_us,route_apply_us,policy_apply_us,graph_rebuild_us,post_switch_tbt_us,switch_success_rate,fallback_count,raw_log_path
```

### `amortization.csv`

```text
date,model,scenario,strategy,output_len,prefill_state,decode_state,total_latency_ms,ttft_ms,avg_tbt_ms,energy_mj,energy_source,slo_tbt_us,slo_met,transition_count,transition_time_ms
```

### `planner_overhead.csv`

```text
date,iterations,median_us,p95_us,p99_us,max_us
```

---

## Required Markdown summaries

After each experiment, generate:

```text
docs/实验结果/InsightB_Context_Frontier_<date>.md
docs/实验结果/InsightB_Transition_Cost_<date>.md
docs/实验结果/InsightB_Amortization_<date>.md
```

Each summary must include:

1. Experiment goal.
2. Exact commands.
3. Temperature range.
4. Main result table.
5. Anomalies.
6. Raw output directories.
7. Whether data is paper-ready or needs rerun.

---

## Expected paper interpretation

Do not force a conclusion.

If context length changes the frontier, the paper insight is:

```text
Decode frontiers drift with context length, so one-shot phase-boundary scheduling can become suboptimal.
```

If context length does not change the frontier much, the fallback insight is:

```text
The frontier is stable, but transitions are only profitable when the remaining output length amortizes the switching cost.
```

Both outcomes are useful.

The final system story should be:

```text
Insight A:
  Different SLOs expose an interleaved CPU/GPU/NPU decode frontier.

Insight B:
  Using the frontier online requires transition-aware decisions:
  context length, remaining output length, and switching overhead decide whether moving to another state is worthwhile.

Design:
  Offline profile -> Pareto frontier -> transition table -> online amortized greedy selector.
```

---

## What not to claim

Do not claim:

```text
Host CPU policy coupling is a core insight.
```

Do not claim:

```text
NPU decode is always bad.
```

Do not claim:

```text
GPU is always better for decode.
```

Do not claim:

```text
Switching is always beneficial.
```

Do claim only what the measured data supports.

---

## Final response format after running tasks

When reporting back, use this structure:

```text
Summary:
- What was implemented
- What was tested
- Key results

Changed files:
- file1
- file2

Commands run:
- command1
- command2

Output directories:
- /tmp/...

Data quality:
- stable points
- unstable points
- reruns needed

Next recommended step:
- one concrete next action
```

```

我建议你把 Codex 的第一个任务设成：**只做 `context_decode_profile.csv` 和 `transition_cost.csv`，先不实现完整 planner**。这两张表出来以后，就能判断 Insight B 应该走 “context frontier drift” 还是 “transition amortization”。
```
