# EcoFrontier Offline Frontier Compiler

## Purpose

The EcoFrontier offline frontier compiler converts exported mobile LLM profile
results into a compact planner artifact for a future online EcoFrontier runtime
planner. It is an offline host-only tool. It does not profile devices, load QNN
graphs, schedule requests, modify shared-memory or KV handoff behavior, or
change runtime execution when EcoFrontier is disabled.

The repository command is:

```bash
python tools/ecofrontier/compile_frontier.py \
  --input docs/实验结果 \
  --output build/ecofrontier/ecofrontier_frontier.json \
  --summary build/ecofrontier/ecofrontier_frontier_summary.json
```

The compiler uses `pathlib` and UTF-8 reads/writes so the Chinese input path is
handled as a normal filesystem path.

## Source Discovery

The compiler recursively scans `docs/实验结果` and consumes recognized sources in
priority order:

1. `InsightB_ChatGPT_结构化数据_*.json` for context decode profiles, SLO source
   frontiers, transition costs, source caveats, and InsightB energy policy.
2. InsightB CSV exports such as `context_decode_profile*.csv` and
   `transition_cost*.csv` when the structured InsightB JSON is not present.
3. Markdown sweep reports named `CPU测试结果.md`, `GPU测试结果.md`, and
   `NPU测试结果.md`.
4. Optional QNN graph manifest/profile JSON files such as `qnn_graphs.json` or
   `qnn_graph_manifest.json`.
5. QNN AoT config metadata files named `qnn_aot_config.json`, preserved only as
   metadata and caveat evidence.

Every consumed source is recorded in `source_files` with a source type and row
count. Files that match a known parser but fail to parse are recorded in
`skipped_sources` with a reason.

## Accepted Input Formats

### Structured JSON

The InsightB JSON package is expected to contain:

- `tables.context_decode_profile`
- `tables.slo_frontier`
- `tables.transition_cost`
- `paper_ready_caveats`
- `data_quality_summary`
- `metadata.energy_policy`
- optional QNN AoT metadata such as `qnn_aot_cache_size` and
  `qnn_aot_context_size`

The compiler preserves the source SLO frontier rows separately, then rebuilds
its own normalized SLO-feasible frontier from normalized state profiles.

### CSV

State CSV rows use columns such as:

```text
state_id,backend,phase,test_shape,prompt_tokens,decode_tokens,context_len,
cpu_affinity,cpu_freq_khz,actual_cpu_freq_khz,cpu_threads,
gpu_freq_mhz,actual_gpu_freq_mhz,npu_workpoint,graph_id,
throughput_tps,tbt_us,ttft_ms_p50,ttft_ms_p95,
active_power_mw,baseline_power_mw,power_delta_mw,
energy_mj_per_token,energy_mj_per_request,
temperature_avg_c,temperature_max_c,stable_range_pct,power_cv_pct,
support_status,fallback_used,data_quality,energy_source
```

Transition CSV rows use columns such as:

```text
from_state_id,to_state_id,context_len,total_blocking_us,first_token_gap_us,
post_switch_tbt_us,transition_energy_mj,transition_energy_source,
success_rate,fallback_count,support_status,kv_handoff_us,graph_rebuild_us,
decision_us
```

Optional numeric fields remain optional. Missing energy is not converted to
zero.

### Markdown Tables

The markdown loader extracts pipe tables from CPU, GPU, and NPU reports and
normalizes known Chinese headers:

- CPU frequency sweep tables: requested kHz, throughput, steady active power,
  baseline delta, temperature, average CPU frequency, remarks.
- GPU sweep tables: set MHz, actual MHz, TG/PP throughput, active power, power
  window range, notes.
- NPU workpoint tables: workpoint, TG/PP or generic throughput, power, baseline
  delta, temperature, and stable range.

Markdown extraction is recorded with the `markdown_extraction_used` caveat so
the planner artifact is explicit about the extraction path.

## Normalized StateProfile

Each normalized state row includes:

- `state_id`
- `backend`: `CPU`, `GPU`, or `QNN_NPU`
- `phase`: `prefill` or `decode`
- `source_file`
- `test_shape`
- `prompt_tokens`, `decode_tokens`, `rounds`, `context_len`
- CPU fields: `cpu_affinity`, `cpu_freq_khz`, `actual_cpu_freq_khz`,
  `cpu_threads`
- GPU fields: `gpu_freq_mhz`, `actual_gpu_freq_mhz`
- NPU fields: `npu_workpoint`
- optional `graph_id`
- latency/performance fields: `throughput_tps`, `tbt_us`, `ttft_ms_p50`,
  `ttft_ms_p95`
- power/energy fields: `active_power_mw`, `baseline_power_mw`,
  `power_delta_mw`, `energy_mj_per_token`, `energy_mj_per_request`
- thermal/stability fields: `temperature_avg_c`, `temperature_max_c`,
  `stable_range_pct`, `power_cv_pct`
- flags: `support_status`, `fallback_used`, `stable`, `thermal_safe`
- annotations: `data_quality`, `energy_source`, `energy_complete`
- `metadata` for source-only fields such as raw log paths and QNN AoT context
  metadata

Backend and phase strings are normalized case-insensitively.

## Normalized TransitionProfile

Each transition edge includes:

- `from_state_id`
- `to_state_id`
- `source_file`
- `context_len`
- `total_blocking_us`
- `first_token_gap_us`
- `post_switch_tbt_us`
- optional `transition_energy_mj`
- `transition_energy_source`
- `transition_energy_complete`
- `success_rate`
- `fallback_count`
- `support_status`
- optional `kv_handoff_us`, `graph_rebuild_us`, and `decision_us`

The transition graph is sparse. The compiler does not require or synthesize a
full state-to-state transition matrix.

## Output Artifact Schema

`ecofrontier_frontier.json` contains:

- `version`
- `generated_at`
- `input_dir`
- `source_files`
- `skipped_sources`
- `compiler_config`
- `raw_profile_summary`
- `normalized_states`
- `models`
- `frontiers`
- `dominated_states`
- `transition_edges`
- `graph_catalog_summary`
- `data_quality_summary`
- `source_slo_frontiers`
- `source_caveats`
- `energy_policy`
- `caveats`

`ecofrontier_frontier_summary.json` contains a compact host-readable summary:

- raw rows by source
- normalized states by backend and phase
- unstable state count
- number of generated frontiers
- frontier kind counts
- transition edge count
- caveats
- data quality summary

## Fitting And Interpolation Policy

Each `state_id` is treated as a discrete execution state. The compiler never
interpolates across backend, CPU affinity, CPU frequency, GPU frequency, NPU
workpoint, or QNN graph identity.

The model emitted for each state is a discrete-state piecewise-linear
length model:

- Decode uses `context_len` as the length axis and TBT as the latency metric.
- Prefill uses `prompt_tokens` as the length axis and TTFT/prefill latency as
  the latency metric.
- Exact measured buckets are emitted as `model_result=exact_bucket`.
- Length interpolation is allowed only within the same state when
  `allow_length_interpolation=true`; the artifact marks this as
  `allowed_within_state_only`.
- Extrapolation is disabled by default. If a future planner asks for a length
  outside the known buckets while `allow_extrapolation=false`, it must treat the
  model result as unavailable out of range.

## Derived Fields

If `tbt_us` is missing and `throughput_tps` is available for decode:

```text
tbt_us = 1e6 / throughput_tps
```

The state records `tbt_source=derived_from_throughput`.

If prefill TTFT is missing and prefill throughput plus prompt tokens are
available, the compiler records a conservative prefill latency proxy:

```text
ttft_ms_p95 = prompt_tokens * 1000 / throughput_tps
```

The state records `ttft_source=prefill_latency_from_throughput` and the
`prefill_latency_proxy` data quality marker.

## Stability Filtering

Rows are preserved in `normalized_states` even when low quality. The default
frontier filters exclude them only when configured to do so.

Default limits:

- `stable_range_pct_limit=10.0`
- `power_cv_pct_limit=10.0`
- `frequency_mismatch_pct_limit=5.0`
- `filter_unstable=true`
- `filter_fallback_used=true`
- `filter_unsupported=true`
- `filter_thermal_unsafe=true`

CPU and GPU states are marked unstable when actual frequency differs from the
requested frequency by more than the mismatch threshold. CPU markdown rows that
contain `掉频` are also marked unstable. Rows whose `data_quality` contains
`unstable_power_window` are marked unstable. Rows with high `power_cv_pct` or
`stable_range_pct` receive `power_low_confidence`; they remain in raw profiles
and are caveated.

## SLO-Feasible Frontiers

For each phase, available length bucket, and configured SLO:

- Decode candidates use TBT in microseconds.
- Prefill candidates use TTFT or the prefill latency proxy in milliseconds.
- Unsupported, fallback, unstable, and thermal-unsafe states are filtered by
  default.
- If no candidate meets the SLO, the frontier entry is emitted with an empty
  feasible set and `notes=no_state_meets_slo`.

Every feasible candidate preserves state ID, backend, phase, length bucket,
latency, energy if available, active power, energy completeness, data quality,
and source file.

## Pareto Pruning Rule

Within each SLO-feasible set, state A dominates state B only when:

- A latency is less than or equal to B latency.
- A energy is less than or equal to B energy when both energies are measured or
  otherwise comparable.
- When energy is unavailable and power comparison is enabled, A active power is
  less than or equal to B active power.
- A stability, support, fallback status, and data quality rank are not worse
  than B.
- At least one compared dimension is strictly better.

Dominated states are preserved in `dominated_states` with `dominated_by`,
`dominance_reason`, phase, length bucket, and frontier kind.

The `frontier_kind` is one of:

- `measured_energy_frontier`
- `estimated_energy_frontier`
- `latency_power_frontier`
- `latency_only_frontier`

The compiler does not label an energy-incomplete or transition-energy-incomplete
artifact as an energy-optimal frontier.

## Energy Completeness

Explicit measured/profiled `energy_mj_per_token` is preserved and marked
`energy_complete=true` unless the source policy says energy must not be
claimed. InsightB marks this case through its `energy_policy`, so those rows
are treated as incomplete energy even when the table has a derived energy
column.

If energy is missing but active power and TBT are available, the compiler
estimates per-token energy:

```text
estimated_energy_mj_per_token = active_power_mw * tbt_us / 1e6
```

The state records `energy_source=estimated_power_latency` and
`energy_complete=false`. Missing energy without power/latency remains missing,
not zero.

Artifacts include these caveats when applicable:

- `energy_estimated_from_power_latency`
- `energy_incomplete_frontier`
- `latency_only_or_latency_power_frontier`

## Transition Energy Caveat

Transition profiles preserve latency costs such as `total_blocking_us`,
`first_token_gap_us`, `post_switch_tbt_us`, `kv_handoff_us`,
`graph_rebuild_us`, and `decision_us` for future online amortization.

The compiler does not invent transition energy. If `transition_energy_mj` is
missing or `transition_energy_source=unavailable`, the edge has
`transition_energy_complete=false` and the artifact includes
`transition_energy_unavailable`.

## QNN Usable KV Slots Rule

`qnn_aot_context_size` and `qnn_aot_cache_size` are preserved as metadata only.
They are not usable KV capacity.

The compiler creates `usable_kv_slots` only from an explicit `usable_kv_slots`
or `qnn_usable_kv_slots` input field. A graph entry with
`qnn_aot_context_size` but no explicit usable slots is rejected from the graph
catalog summary with `reason=missing_usable_kv_slots`, and the artifact includes
`qnn_context_size_not_usable_kv_slots`.

## Future Online Planner Consumption

The future online EcoFrontier planner should load the artifact, select the
phase and length bucket, apply the appropriate SLO class, and choose only from
the emitted frontier candidates whose caveats and data quality are acceptable
for the request policy.

The online planner remains responsible for:

- request-level SLO policy;
- context-length lookup and same-state interpolation decisions;
- rejecting out-of-range extrapolation unless explicitly allowed;
- QNN graph capacity checks against explicit `usable_kv_slots`;
- transition amortization using sparse `transition_edges`;
- respecting energy completeness caveats before making energy claims.
