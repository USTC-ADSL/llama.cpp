# AGENT.md

## Role

You are assisting with lightweight implementation and experiments for
Prefill/Decode phase separation across heterogeneous mobile backends.

The current working direction is:

> Accelerate mobile LLM inference by running Prefill and Decode on the most
> suitable hardware backends, with conservative runtime changes that preserve
> correctness and make transition overhead visible.

This workspace is not the owner for power-consumption experiments. Do not add
new power-test scripts, power-sampling workflows, energy tables, or paper claims
based on power data unless the user explicitly redirects the task.

---

## Current Focus

Primary optimization path:

```text
Prefill -> Decode phase boundary
QNN/NPU, GPUOpenCL, and CPU backend selection
KV handoff / migration / aliasing
route stability and switch overhead
```

Useful runtime questions:

1. Can Prefill and Decode run on different backends without semantic regressions?
2. How much blocking overhead is introduced at the phase boundary?
3. Which part of the overhead is KV migration, backend aliasing, scheduler reserve,
   graph rebuild, or route instability?
4. Which optimizations reduce the phase-switch cost without large inference-core
   rewrites?

---

## Non-Negotiable Rules

1. Do not fabricate data.
2. Do not silently drop failed runs.
3. Do not overwrite existing experiment results.
4. Do not hardcode ADB device IDs. Use `DEVICE` from the environment.
5. Do not hardcode model paths. Use `MODEL_PATH` from the environment.
6. Avoid changing global device state unless a task explicitly requires it and
   the script restores the original state.
7. Keep runtime changes scoped to Prefill/Decode separation, backend routing,
   KV handoff, and timing instrumentation.
8. Do not revive Host-policy coupling or anomalous high-power behavior as a
   paper insight.

---

## Allowed Changes

Allowed by default:

1. Runtime fixes for Prefill/Decode backend separation.
2. Timing trace instrumentation for phase switching.
3. Lightweight scripts that reproduce functional, latency, or overhead results.
4. Parsers or summaries for benchmark logs that do not depend on power sampling.
5. Markdown notes summarizing correctness, latency, and overhead findings.

Avoid unless explicitly requested:

1. New power-sampling scripts.
2. Battery current/voltage sampling.
3. Energy or active-power profile tables.
4. Governor/frequency sweeps as a power experiment.
5. Planner prototypes whose main objective is energy minimization.

---

## Preferred Measurement Fields

For phase-switch and backend-separation work, prefer trace lines that expose:

```text
prefill_backend
decode_backend
context_len
decode_tokens
route_apply_us
sched_reserve_us
kv_migration_us
kv_alias_us
graph_rebuild_us
decode_entry_us
first_token_gap_us
post_switch_tbt_us
switch_success
fallback_used
raw_log_path
```

If a field is not available, add a trace point rather than guessing.

---

## Current Useful Commands

Use environment variables for device and model paths:

```bash
DEVICE=<adb-serial> \
MODEL_PATH=<device-gguf-path> \
bash <script>
```

For combined Prefill -> Decode behavior, prefer `llama-bench -pg <prompt>,<gen>`
over separate `-p` and `-n` runs. Use verbose runtime logs when validating route
or KV migration behavior.

---

## Final Response Format

When reporting back after running tasks, use:

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
