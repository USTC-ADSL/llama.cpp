# QNN Phase-Boundary Materializer Design

## Goal

Add a general `QNN prefill -> non-QNN decode` phase-switch mechanism that preserves correctness by materializing QNN-prefill KV state into the generic llama KV cache at the decode boundary.

The mechanism must be:

- decode-centric
- stage-centric
- correctness-first
- measurable
- optional and easy to disable

The first implementation target is:

- device: `db6c02cf`
- model: `Qwen2-3B PowerServe AoT`
- route: `prefill=qnn-npu -> decode=cpu`

After that path is correct and measurable, the same mechanism should extend to:

- `prefill=qnn-npu -> decode=opencl`

## Background

The current codebase already has three related pieces:

1. Static `qnn-npu` decode can run correctly with host-visible KV placement on `qnn-npu-host`.
2. AoT full-graph QNN can discover generic cache tensors `cache_k_l*` / `cache_v_l*` and can:
   - write generic KV rows from QNN graph outputs into those tensors
   - import generic KV rows back into QNN private cache tensors
3. Dynamic phase routing currently rejects decode-time QNN boundary switches under AoT by default with `qnn-phase-switch-unsafe`.

Recent evidence shows that current `QNN <-> non-QNN` decode-time switching is not safe even when generic KV writeback is enabled:

- static CPU output is correct
- static QNN output is correct
- mixed `QNN prefill -> CPU decode` can produce incorrect text
- mixed `CPU prefill -> QNN decode` can also produce incorrect text
- the currently exported mixed KV differs from CPU-exported KV

This means the present generic writeback path is not yet a trustworthy correctness bridge for phase-boundary switching.

## Problem Statement

The project needs a low-overhead stage switch between `QNN prefill` and `non-QNN decode`.

The desired steady-state behavior is:

1. Run prefill on QNN.
2. At the phase boundary, place decode-consumable KV state into memory that CPU or OpenCL can immediately consume.
3. Start decode on the target backend without producing incorrect output.

The immediate problem is not merely "copy overhead". The immediate problem is that the existing QNN-to-generic KV bridge does not yet reproduce the same decode-visible KV state as the correct CPU path.

Therefore the first design priority is:

- establish a correct and explicit phase-boundary materialization path

Only after that should the project reduce its overhead.

## Non-Goals

This design does not attempt to:

- prove all dynamic schedules beat all static schedules
- add operator-level switching
- solve `QNN -> OpenCL` crashes unrelated to KV state transfer
- hide the first-token phase-switch cost
- claim end-to-end energy wins before boundary overhead is measured

## Design Overview

Introduce a new experimental runtime path called the **phase-boundary materializer**.

It is a decode-boundary mechanism that runs only when all of the following are true:

- the current phase is switching from prefill to decode
- the active prefill plan uses QNN
- the target decode plan uses a non-QNN backend
- AoT QNN is active
- the experimental materializer flag is enabled

Instead of relying on the current eager/deferred generic KV writeback as the only correctness bridge, the runtime will:

1. export the QNN prefill KV state into a host-side intermediate payload
2. materialize that payload into the generic llama KV cache using a CPU-side layout routine that matches the known-correct CPU export semantics
3. mark the generic KV cache as the decode-visible source of truth
4. allow decode to start on CPU or OpenCL

This design intentionally allows a one-time CPU conversion at the phase boundary. The conversion is acceptable because:

- the user explicitly allows it
- correctness is the first requirement
- it creates a measurable first-token overhead budget
- it is a stepping stone toward lower-overhead shared-placement paths

## Why This Approach

This design is preferred over immediately forcing zero-copy shared placement for three reasons:

1. It isolates correctness from optimization.
   The current issue is semantic mismatch, not only bandwidth cost.
2. It provides a single explicit boundary hook where overhead can be measured.
3. It generalizes cleanly:
   - `QNN -> CPU`
   - `QNN -> OpenCL`
   can share the same materialization semantics, differing only in decode consumer placement.

## High-Level Data Flow

### `QNN prefill -> CPU decode`

1. QNN AoT prefill runs as usual.
2. At the phase boundary, the runtime extracts per-layer, per-head KV rows from the QNN graph outputs.
3. A CPU materializer converts those rows into the same generic cache row layout that CPU decode expects.
4. The generic cache tensors `cache_k_l*` / `cache_v_l*` become the decode-visible KV state.
5. CPU decode begins using those generic cache tensors.

### `QNN prefill -> OpenCL decode`

The same logical steps apply, but the final generic cache placement must also satisfy the chosen OpenCL decode path:

- either host-visible generic KV that OpenCL decode can consume
- or a future OpenCL-compatible shared-host placement, once correctness is already established

The initial implementation should keep the materialization semantics identical and vary only the consumer-compatible placement.

## Core Components

### 1. QNN Boundary Exporter

Responsibility:

- collect QNN prefill KV state at the phase boundary into a neutral host payload

Source of data:

- existing AoT full-graph per-head KV outputs
- existing layer/head discovery logic
- existing token-slot discovery logic based on `k_idxs`, `v_idxs`, or inferred slots from `kq_mask`

Output shape:

- one payload per layer
- each payload contains:
  - layer id
  - token count
  - slot indices
  - key rows in token-major order
  - value rows in token-major order

Important requirement:

- this payload is not yet the final generic cache layout
- it is only the canonical host-side phase-boundary representation

### 2. CPU KV Materializer

Responsibility:

- convert the exporter payload into the generic llama KV cache layout that non-QNN decode consumes correctly

The reference semantics must match the current correct CPU export path:

- key rows are token-major and head-sliced
- value rows follow the same decode-visible semantics as `dump_powerserve_seed_kv()`
- `v_trans` handling must match the cache's actual layout

This component is the core correctness boundary.

It should not depend on:

- QNN runtime internals after export
- backend-specific decode execution details

It should depend only on:

- generic cache tensor metadata
- model head dimensions
- slot indices
- whether V is transposed in the target cache

### 3. Generic KV Committer

Responsibility:

- write the materialized host rows into `cache_k_l*` / `cache_v_l*`

The first version may reuse the existing helper that writes host rows into cache tensors.

However, its call site must be changed so that:

- the commit step consumes the new boundary payload
- correctness validation happens before decode proceeds

### 4. Phase-Switch Controller

Responsibility:

- decide whether the phase-boundary materializer should run
- reject unsafe decode-time switching unless materialization succeeds

Default behavior:

- keep the current `qnn-phase-switch-unsafe` admission guard

Experimental behavior when the new feature is enabled:

- on `QNN prefill -> non-QNN decode`, replace "reject immediately" with:
  - export
  - materialize
  - commit
  - then apply decode route

If any step fails:

- log the exact reason
- keep the current backend
- preserve correctness over route switching

### 5. Overhead Tracing

Responsibility:

- expose the cost of the new phase-boundary path

At minimum the runtime must log:

- `qnn_boundary_export_us`
- `kv_materialize_us`
- `kv_commit_us`
- `phase_switch_total_us`

These must be reported separately from:

- `reserve_us`
- `apply_us`
- `process_ubatch_us`

This preserves the project's measurement-first and overhead-conscious principles.

## API and Control Surface

Add a new experimental control flag for the materializer path.

Suggested environment variable:

- `GGML_HETERO_QNN_PHASE_MATERIALIZE=1`

Optional second flag for scope:

- `GGML_HETERO_QNN_PHASE_MATERIALIZE_TARGETS=cpu,opencl`

Suggested semantics:

- unset or `0`:
  - keep the current safety guard
- `1`:
  - allow `QNN prefill -> non-QNN decode` only if phase-boundary materialization succeeds

The existing `GGML_HETERO_DYNAMIC_ALLOW_UNSAFE_QNN_PHASE_SWITCH=1` should remain as a stronger debug override that bypasses the safety policy entirely.

## Correctness Contract

For the first implementation, the new mechanism is considered correct only if all of the following hold:

1. `QNN prefill -> CPU decode` generates meaningful text on `Qwen2-3B PowerServe AoT`.
2. The materialized generic KV export matches CPU-prefill-exported KV for the same prompt and token prefix.
3. When the materializer path is disabled, the runtime still rejects unsafe decode-time switching.
4. When the materializer path is enabled but materialization fails, the runtime still preserves correctness by refusing the switch.

## Validation Strategy

### Unit-Level Validation

Add focused tests for the CPU materializer:

- slot mapping is correct
- head slicing is correct
- `v_trans` and non-`v_trans` layouts are both correct
- contiguous and non-contiguous slot cases are both correct

These tests should use synthetic data, not device execution.

### Tool-Level Validation

Extend the existing KV export tooling or add a dedicated tool so that the following can be compared:

- CPU prefill exported KV
- QNN boundary materializer exported KV

Comparison granularity:

- per-file exact match for `layer_<layer>_{key,value}_<head>.raw`

The first validation target is:

- prompt used in the current `Qwen2-3B` repro

### End-to-End Validation

Phase 1:

- `prefill=qnn-npu -> decode=cpu`

Success means:

- text output is meaningful
- the route is actually switched
- logs show materialization happened
- boundary overhead is reported

Phase 2:

- `prefill=qnn-npu -> decode=opencl`

Success means:

- text output is meaningful
- the route is actually switched
- the same materializer semantics still hold

## File and Boundary Expectations for Implementation

The likely implementation boundaries are:

- `ggml/src/ggml-qnn/qnn/aot.cpp`
  - add the boundary export path and possibly refactor generic KV assembly helpers
- `ggml/src/ggml-qnn/qnn/aot.hpp`
  - define the new host-side payload type and public materialization entry points
- `src/llama-context.cpp`
  - trigger the materializer at the prefill/decode route boundary
- `src/llama-dyn-route.cpp`
  - update admission logic so the switch is allowed only through the new safe path
- `src/llama-kv-cache.cpp`
  - keep generic KV layout handling centralized and reuse its semantics as the materializer reference
- `tools/qnn-kv-export/`
  - extend or reuse tooling for correctness comparison
- `tests/`
  - unit tests for materialization semantics

The implementation should avoid large `llama.cpp` refactors. The new path must remain local, experimental, and easy to disable.

## Risks and Tradeoffs

### 1. First-token overhead may be noticeable

This is acceptable for the first version because the mechanism is intended to:

- establish correctness
- quantify phase-boundary overhead
- create a baseline for future optimization

### 2. Existing generic writeback and new materializer may diverge

This is acceptable if the new path is clearly marked as the decode-boundary correctness path for `QNN -> non-QNN`.

Later work can unify them once the correct semantics are proven.

### 3. OpenCL consumer compatibility may still require placement work

This does not block the first phase.

The design explicitly allows:

- correctness-first materialization for CPU
- then reusing the same exported payload for OpenCL-compatible cache placement

### 4. There may be non-KV private state hidden inside AoT

If `KV` parity is correct but output is still wrong, the next conclusion should be:

- phase-boundary switching requires additional exported state beyond generic KV

That would still be valuable because it narrows the unresolved boundary to a specific class of hidden state rather than leaving the failure unexplained.

## Minimal Success Criteria

The minimal acceptable result for this project slice is:

1. `QNN prefill -> CPU decode` works on `Qwen2-3B PowerServe AoT`.
2. The runtime logs the exact materialization overhead.
3. The new path is gated and can be turned off.
4. The existing unsafe-switch guard remains the fallback safety mechanism.

If this succeeds, the project gains:

- a correct decode-boundary bridge
- a quantitative first-token overhead measurement
- a reusable mechanism for future `QNN -> OpenCL` work

## Failure Interpretation

Failure after implementing this design should be interpreted carefully:

- If KV files still differ from CPU-exported KV:
  - the boundary exporter or materializer layout is still wrong.
- If KV files match but text is still wrong:
  - QNN private state beyond generic KV is missing from the boundary contract.
- If text is correct but first-token overhead is too high:
  - the mechanism is correct but not yet competitive; optimization should then focus on placement reuse, partial materialization, or shared-host decode entry.

## Recommendation

Proceed in two steps:

1. Build the general phase-boundary materializer abstraction.
2. Validate it first on `QNN prefill -> CPU decode` with `Qwen2-3B`.

Only after that should the implementation be extended to:

- `QNN prefill -> OpenCL decode`

This preserves correctness, keeps the patch localized, and produces the right evidence for the project's decode-centric, overhead-conscious research story.
