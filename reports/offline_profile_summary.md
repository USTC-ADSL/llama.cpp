# Offline Profile Summary

This report summarizes a configurable offline profiling pipeline for context-aware backend-state characterization.
The synthetic relative throughput target is not a p95 TBT guarantee and not a real user SLO; it is a characterization target derived from the fastest stable state in each context bucket.
Pipeline ready; hardware experiments not run in this report.

## 1. Experiment Configuration

| item | value |
| --- | --- |
| model | _not configured_ |
| tokenizer | _not configured_ |
| repeat | 3 |
| context points | 512, 1024, 1536, 2048, 3072, 4096, 5120, 6144 |
| buckets | [0, 512); [512, 1024); [1024, 1536); [1536, 2048); [2048, 3072); [3072, 4096); [4096, 5120); [5120, 6144) |
| decode probe tokens | 64 |
| idle power mW | 0.0 |
| thermal policy | log_only |

NPU graph tier rule: cap2048 represents sub-2048 contexts; cap4096 and cap6144 are profiled as large and xlarge tiers. cap512/cap1024 are intentionally excluded from the main sweep because pre-experiments already validated them as throughput-equivalent to cap2048 in capacity-feasible sub-2048 contexts.

## 2. QNN Graph Capacity Reduction

- cap512/cap1024/cap2048 are treated as throughput-equivalent for sub-2048 capacity-feasible contexts based on prior pre-experiments.
- Main profile uses cap2048 as representative graph for sub-2048 contexts.
- cap4096 and cap6144 remain separate large/xlarge graph tiers.
- `profiles/qnn_large_graph_sanity.csv` is optional; if present, use it to check whether large graphs lose short-context performance.

| graph | length | workpoint | sanity worst tps | cap2048 worst tps | delta | note |
| --- | --- | --- | --- | --- | --- | --- |
| present |  |  |  |  |  | no ok sanity rows yet |

## 3. Profile Coverage

| backend | catalog states | failed raw rows | skipped raw rows |
| --- | --- | --- | --- |
| NPU | 9 | 0 | 0 |
| CPU | 9 | 0 | 0 |
| GPU | 3 | 0 | 0 |

## 4. Stability Summary

| metric | count |
| --- | --- |
| ok raw runs | 0 |
| failed raw runs | 0 |
| skipped raw runs | 0 |
| unstable states | 0 |
| throttled states | 0 |
| insufficient runs | 0 |
| ok states missing energy/token | 0 |

## 5. Performance Frontier Summary

Fastest stable state per length:

_No rows._

Pareto frontier states:

_No rows._

## 6. Relative Target Summary

_No rows._

## 7. Adaptive Refinement Summary

_No rows._

Current coarse points are sufficient only where no refinement rows are emitted and selected-state margins are not near target.

## 8. Offline Planner Summary

_No rows._

When transition profiles are missing, transition-aware replay must use configured conservative defaults and mark `missing_transition_profile`; no-transition replay must mark `no_transition_model`.

## 9. Figures

_matplotlib unavailable or insufficient numeric data; CSV and Markdown generation still succeeded._

## 10. TODO

- Fill `decode_command_template` and `prefill_command_template` with real device measurement commands.
- Add device-specific CPU affinity and CPU frequency control, or keep those fields as log-only TODO.
- Add NPU workpoint and QNN graph selection commands without hardcoding platform assumptions.
- Add GPU frequency control only if the current device exposes a safe interface.
- Add power/energy sampling command or output JSON fields; idle-subtracted energy can be derived from active power and elapsed time.
- Complete real transition measurements in `profiles/transition_profile.csv`; do not insert fake transition rows.
