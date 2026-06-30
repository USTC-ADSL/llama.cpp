#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "system_benefit_simulation.v1"
PROFILE_FIELDS = [
    "phase",
    "backend",
    "state_name",
    "state_group",
    "bucket_lo",
    "bucket_hi",
    "bucket_tokens",
    "throughput_tps",
    "power_mw",
    "energy_mj_per_token",
    "energy_mj_per_bucket",
]
DEFAULT_BASELINE_DECODE_STATES = {
    "CPU": "B2S4_4320000_3532800",
    "GPU": "1100",
    "NPU": "burst",
}
DEFAULT_BACKEND_TRANSITION_LATENCY_MS = {
    ("NPU", "CPU"): 5.0,
    ("GPU", "CPU"): 15.0,
    ("CPU", "GPU"): 20.0,
    ("NPU", "GPU"): 20.0,
    ("CPU", "NPU"): 80.0,
    ("GPU", "NPU"): 80.0,
}
QNN_GRAPH_CAPACITIES = (
    {"context_size": 2048, "usable_kv_slots": 1920},
    {"context_size": 4096, "usable_kv_slots": 3968},
    {"context_size": 6144, "usable_kv_slots": 6016},
)


@dataclass(frozen=True)
class ProfileRecord:
    phase: str
    backend: str
    state_name: str
    state_group: str
    length: int
    bucket_lo: int
    bucket_hi: int
    throughput_tps: float
    power_mw: float
    stable: bool
    status: str
    metadata: dict[str, Any]

    @property
    def energy_mj_per_token(self) -> float:
        return self.power_mw / self.throughput_tps

    @property
    def mean_tbt_ms(self) -> float:
        return 1000.0 / self.throughput_tps


@dataclass(frozen=True)
class TransitionCost:
    latency_ms: float
    energy_mj: float
    source: str


@dataclass(frozen=True)
class SegmentCandidate:
    segment_id: int
    context_bucket_lo: int
    context_bucket_hi: int
    profile_query_bucket_lo: int
    profile_query_bucket_hi: int
    num_tokens: int
    record: ProfileRecord
    matched_profile_bucket_lo: int
    matched_profile_bucket_hi: int
    match_kind: str
    feasible_for_slo: bool
    selection_mode: str

    @property
    def latency_ms(self) -> float:
        return self.num_tokens / self.record.throughput_tps * 1000.0

    @property
    def energy_mj(self) -> float:
        return self.record.power_mw * self.latency_ms / 1000.0


@dataclass
class DpNode:
    total_energy_mj: float
    total_latency_ms: float
    total_slo_violations: int
    total_slo_miss_ms: float
    prev_state: str | None
    candidate: SegmentCandidate
    transition: TransitionCost
    step_latency_ms: float
    step_energy_mj: float
    step_slo_deadline_ms: float
    step_slo_ok: bool
    step_slo_miss_ms: float


def dp_key(node: DpNode) -> tuple[int, float, float]:
    # SLO is the hard preference: first minimize missed steps, then miss distance,
    # then energy. This implements "lowest energy under SLO; closest if no SLO path".
    return (node.total_slo_violations, node.total_slo_miss_ms, node.total_energy_mj)


def aligned_decode_segments(input_len: int, output_len: int, bucket_size: int) -> Iterable[tuple[int, int, int, int, int, int]]:
    total_context_hi = input_len + output_len
    context_lo = input_len + 1
    profile_hi = ((input_len + bucket_size - 1) // bucket_size + 1) * bucket_size
    segment_id = 0
    while context_lo <= total_context_hi:
        profile_lo = profile_hi - bucket_size + 1
        context_hi = min(total_context_hi, profile_hi)
        num_tokens = context_hi - context_lo + 1
        yield segment_id, context_lo, context_hi, profile_lo, profile_hi, num_tokens
        context_lo = context_hi + 1
        profile_hi += bucket_size
        segment_id += 1


def backend_transition_latency_ms(from_backend: str | None, to_backend: str | None) -> float | None:
    if not from_backend or not to_backend:
        return None
    if from_backend == to_backend:
        return 0.0
    return DEFAULT_BACKEND_TRANSITION_LATENCY_MS.get((from_backend, to_backend))


def qnn_graph_capacity_for_required(required_context: int) -> dict[str, int] | None:
    if required_context <= 0:
        return None
    for capacity in QNN_GRAPH_CAPACITIES:
        if required_context <= capacity["usable_kv_slots"]:
            return capacity
    return None


def qnn_graph_capacity_for_context_size(context_size: int) -> dict[str, int] | None:
    for capacity in QNN_GRAPH_CAPACITIES:
        if context_size == capacity["context_size"]:
            return capacity
    return None


def qnn_context_size_from_state(state_name: str, state_group: str = "") -> int | None:
    text = f"{state_name} {state_group}".lower()
    match = re.search(r"(?:cap|ctx|context)(\d+)", text)
    if not match:
        return None
    context_size = int(match.group(1))
    return context_size if qnn_graph_capacity_for_context_size(context_size) else None


def qnn_profile_record_matches_query_capacity(record: "ProfileRecord", profile_query_bucket_hi: int) -> bool:
    if record.backend != "NPU":
        return True
    query_capacity = qnn_graph_capacity_for_required(profile_query_bucket_hi)
    if query_capacity is None:
        return False
    explicit_context = qnn_context_size_from_state(record.state_name, record.state_group)
    if explicit_context is not None:
        state_capacity = qnn_graph_capacity_for_context_size(explicit_context)
        return bool(state_capacity and profile_query_bucket_hi <= state_capacity["usable_kv_slots"])
    record_capacity = qnn_graph_capacity_for_required(record.bucket_hi)
    return bool(record_capacity and record_capacity["context_size"] == query_capacity["context_size"])


class Scheduler:
    name = "base"

    def plan(self, request: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class EnergyDpScheduler(Scheduler):
    name = "energy-dp"

    def __init__(
        self,
        records: list[ProfileRecord],
        transitions: list[dict[str, Any]],
        *,
        slo_tbt_ms: float,
        context_match: str,
        allowed_backends: set[str],
        fixed_backend: str | None = None,
        fixed_state: str | None = None,
        initial_state: str | None = None,
        default_transition_latency_ms: float = 0.0,
        default_transition_energy_mj: float = 0.0,
    ) -> None:
        self.records = records
        self.transitions = transitions
        self.slo_tbt_ms = slo_tbt_ms
        self.context_match = context_match
        self.allowed_backends = allowed_backends
        self.fixed_backend = fixed_backend
        self.fixed_state = fixed_state
        self.initial_state = initial_state
        self.default_transition_latency_ms = default_transition_latency_ms
        self.default_transition_energy_mj = default_transition_energy_mj
        self.backend_by_state: dict[str, str] = {}
        for record in records:
            self.backend_by_state.setdefault(record.state_name, record.backend)

    def plan_decode(self, input_len: int, output_len: int, bucket_size: int) -> dict[str, Any]:
        if output_len <= 0:
            return empty_decode_plan(self.name)

        segments: list[list[SegmentCandidate]] = []
        for (
            segment_id,
            context_bucket_lo,
            context_bucket_hi,
            profile_query_bucket_lo,
            profile_query_bucket_hi,
            num_tokens,
        ) in aligned_decode_segments(input_len, output_len, bucket_size):
            candidates = self.candidates_for_segment(
                segment_id,
                context_bucket_lo,
                context_bucket_hi,
                profile_query_bucket_lo,
                profile_query_bucket_hi,
                num_tokens,
            )
            if not candidates:
                raise ValueError(
                    "no decode profile candidate for "
                    f"segment {segment_id} context_bucket={context_bucket_lo}-{context_bucket_hi} "
                    f"profile_query_bucket={profile_query_bucket_lo}-{profile_query_bucket_hi}"
                )
            segments.append(candidates)

        dp: list[dict[str, DpNode]] = []
        for index, candidates in enumerate(segments):
            layer: dict[str, DpNode] = {}
            for candidate in candidates:
                state = candidate.record.state_name
                if index == 0:
                    prev_state = self.initial_state
                    transition = self.lookup_transition(prev_state, state) if prev_state else TransitionCost(0.0, 0.0, "initial")
                    node = self.make_node(None, prev_state, candidate, transition)
                    layer[state] = node
                    continue

                best_node: DpNode | None = None
                for prev_state, prev_node in dp[index - 1].items():
                    transition = self.lookup_transition(prev_state, state)
                    node = self.make_node(prev_node, prev_state, candidate, transition)
                    if best_node is None or dp_key(node) < dp_key(best_node):
                        best_node = node
                if best_node:
                    layer[state] = best_node
            dp.append(layer)

        if not dp or not dp[-1]:
            raise ValueError("DP failed to produce a decode plan")

        final_state, final_node = min(dp[-1].items(), key=lambda item: dp_key(item[1]))
        del final_node
        chosen: list[DpNode] = []
        current_state: str | None = final_state
        for index in range(len(dp) - 1, -1, -1):
            if current_state is None:
                raise ValueError("broken DP backpointer")
            node = dp[index][current_state]
            chosen.append(node)
            current_state = node.prev_state
        chosen.reverse()

        segment_rows = []
        decode_latency_ms = 0.0
        decode_compute_latency_ms = 0.0
        decode_transition_latency_ms = 0.0
        decode_energy_mj = 0.0
        decode_compute_energy_mj = 0.0
        decode_transition_energy_mj = 0.0
        profile_slo_met = True
        slo_satisfied_steps = 0
        notes: set[str] = set()
        for node in chosen:
            candidate = node.candidate
            transition = node.transition
            switch_reason = self.switch_reason(node)
            segment_latency = node.step_latency_ms
            segment_energy = node.step_energy_mj
            decode_latency_ms += segment_latency
            decode_compute_latency_ms += candidate.latency_ms
            decode_transition_latency_ms += transition.latency_ms
            decode_energy_mj += segment_energy
            decode_compute_energy_mj += candidate.energy_mj
            decode_transition_energy_mj += transition.energy_mj
            profile_slo_met = profile_slo_met and candidate.feasible_for_slo
            if node.step_slo_ok:
                slo_satisfied_steps += 1
            if transition.source == "default":
                notes.add("missing_transition_profile")
            if candidate.match_kind != "exact":
                notes.add(f"context_match_{candidate.match_kind}")
            if not node.step_slo_ok:
                notes.add("best_effort_bucket_below_slo")
            step_achieved_tps = candidate.num_tokens / (segment_latency / 1000.0) if segment_latency > 0 else math.inf
            energy_saving_vs_prev_mj = self.energy_saving_vs_prev_state(node)
            energy_saving_vs_prev_pct = (
                energy_saving_vs_prev_mj / (energy_saving_vs_prev_mj + segment_energy) * 100.0
                if energy_saving_vs_prev_mj is not None and energy_saving_vs_prev_mj > 0
                else None
            )
            segment_rows.append(
                {
                    "segment_id": candidate.segment_id,
                    "context_len": candidate.context_bucket_hi,
                    "context_bucket_lo": candidate.context_bucket_lo,
                    "context_bucket_hi": candidate.context_bucket_hi,
                    "profile_query_bucket_lo": candidate.profile_query_bucket_lo,
                    "profile_query_bucket_hi": candidate.profile_query_bucket_hi,
                    "matched_profile_length": candidate.matched_profile_bucket_hi,
                    "matched_profile_bucket_lo": candidate.matched_profile_bucket_lo,
                    "matched_profile_bucket_hi": candidate.matched_profile_bucket_hi,
                    "context_match": candidate.match_kind,
                    "num_tokens": candidate.num_tokens,
                    "selected_state": candidate.record.state_name,
                    "backend": candidate.record.backend,
                    "state_group": candidate.record.state_group,
                    "throughput_tps": candidate.record.throughput_tps,
                    "power_mw": candidate.record.power_mw,
                    "mean_tbt_ms": candidate.record.mean_tbt_ms,
                    "energy_mj_per_token": candidate.record.energy_mj_per_token,
                    "transition_from_prev": node.prev_state or "",
                    "switch_reason": switch_reason,
                    "transition_latency_ms": transition.latency_ms,
                    "transition_energy_mj": transition.energy_mj,
                    "transition_source": transition.source,
                    "transition_energy_source": "target_power" if transition.source == "backend_default" else transition.source,
                    "segment_decode_latency_ms": candidate.latency_ms,
                    "segment_decode_energy_mj": candidate.energy_mj,
                    "segment_total_latency_ms": segment_latency,
                    "segment_total_energy_mj": segment_energy,
                    "step_slo_deadline_ms": node.step_slo_deadline_ms,
                    "step_latency_ms": segment_latency,
                    "step_mean_tbt_ms": segment_latency / candidate.num_tokens,
                    "step_achieved_tps": step_achieved_tps,
                    "step_slo_ok": node.step_slo_ok,
                    "step_slo_miss_ms": node.step_slo_miss_ms,
                    "step_slo_margin_ms": node.step_slo_deadline_ms - segment_latency,
                    "profile_feasible_for_slo": candidate.feasible_for_slo,
                    "feasible_for_slo": node.step_slo_ok,
                    "selection_mode": "feasible" if node.step_slo_ok else "best_effort_closest_to_slo",
                    "energy_saving_vs_prev_mj": energy_saving_vs_prev_mj,
                    "energy_saving_vs_prev_pct": energy_saving_vs_prev_pct,
                }
            )

        achieved_decode_tps = output_len / (decode_latency_ms / 1000.0) if decode_latency_ms > 0 else math.inf
        slo_total_steps = len(chosen)
        all_steps_slo_met = slo_satisfied_steps == slo_total_steps
        return {
            "scheduler": self.name,
            "decode_latency_ms": decode_latency_ms,
            "decode_compute_latency_ms": decode_compute_latency_ms,
            "decode_transition_latency_ms": decode_transition_latency_ms,
            "decode_energy_mj": decode_energy_mj,
            "decode_compute_energy_mj": decode_compute_energy_mj,
            "decode_transition_energy_mj": decode_transition_energy_mj,
            "decode_throughput_tps": achieved_decode_tps,
            "slo_satisfied_steps": slo_satisfied_steps,
            "slo_total_steps": slo_total_steps,
            "slo_satisfaction_rate": slo_satisfied_steps / slo_total_steps if slo_total_steps else 1.0,
            "all_segments_profile_slo_met": profile_slo_met,
            "all_steps_slo_met": all_steps_slo_met,
            "slo_met": all_steps_slo_met,
            "segments": segment_rows,
            "decode_schedule_env": build_decode_schedule_env(segment_rows, bucket_size),
            "notes": sorted(notes),
        }

    def make_node(
        self,
        prev_node: DpNode | None,
        prev_state: str | None,
        candidate: SegmentCandidate,
        transition: TransitionCost,
    ) -> DpNode:
        if transition.source == "backend_default":
            transition = TransitionCost(
                transition.latency_ms,
                candidate.record.power_mw * transition.latency_ms / 1000.0,
                transition.source,
            )
        step_latency_ms = transition.latency_ms + candidate.latency_ms
        step_energy_mj = transition.energy_mj + candidate.energy_mj
        step_slo_deadline_ms = self.slo_tbt_ms * candidate.num_tokens
        step_slo_miss_ms = max(0.0, step_latency_ms - step_slo_deadline_ms)
        step_slo_ok = step_slo_miss_ms <= 1e-9
        prev_energy = prev_node.total_energy_mj if prev_node else 0.0
        prev_latency = prev_node.total_latency_ms if prev_node else 0.0
        prev_violations = prev_node.total_slo_violations if prev_node else 0
        prev_miss = prev_node.total_slo_miss_ms if prev_node else 0.0
        return DpNode(
            total_energy_mj=prev_energy + step_energy_mj,
            total_latency_ms=prev_latency + step_latency_ms,
            total_slo_violations=prev_violations + (0 if step_slo_ok else 1),
            total_slo_miss_ms=prev_miss + step_slo_miss_ms,
            prev_state=prev_state,
            candidate=candidate,
            transition=transition,
            step_latency_ms=step_latency_ms,
            step_energy_mj=step_energy_mj,
            step_slo_deadline_ms=step_slo_deadline_ms,
            step_slo_ok=step_slo_ok,
            step_slo_miss_ms=step_slo_miss_ms,
        )

    def candidates_for_segment(
        self,
        segment_id: int,
        context_bucket_lo: int,
        context_bucket_hi: int,
        profile_query_bucket_lo: int,
        profile_query_bucket_hi: int,
        num_tokens: int,
    ) -> list[SegmentCandidate]:
        candidates_by_state: dict[str, tuple[ProfileRecord, str]] = {}
        for record in self.records:
            if record.phase != "decode":
                continue
            if record.backend not in self.allowed_backends:
                continue
            if self.fixed_backend and record.backend != self.fixed_backend:
                continue
            if self.fixed_state and not state_matches(record.backend, record.state_name, self.fixed_state):
                continue
            if not qnn_profile_record_matches_query_capacity(record, profile_query_bucket_hi):
                continue
            match = bucket_match(record, profile_query_bucket_lo, profile_query_bucket_hi, self.context_match)
            if match is None:
                continue
            previous = candidates_by_state.get(record.state_name)
            if previous is None or match[0] < bucket_match_distance(previous[0], profile_query_bucket_lo, profile_query_bucket_hi):
                candidates_by_state[record.state_name] = (record, match[1])

        raw: list[SegmentCandidate] = []
        for record, match_kind in candidates_by_state.values():
            raw.append(
                SegmentCandidate(
                    segment_id=segment_id,
                    context_bucket_lo=context_bucket_lo,
                    context_bucket_hi=context_bucket_hi,
                    profile_query_bucket_lo=profile_query_bucket_lo,
                    profile_query_bucket_hi=profile_query_bucket_hi,
                    num_tokens=num_tokens,
                    record=record,
                    matched_profile_bucket_lo=record.bucket_lo,
                    matched_profile_bucket_hi=record.bucket_hi,
                    match_kind=match_kind,
                    feasible_for_slo=record.mean_tbt_ms <= self.slo_tbt_ms,
                    selection_mode="feasible",
                )
            )
        if not raw:
            return []

        return raw

    def raw_candidate_for_state(
        self,
        state_name: str,
        context_bucket_lo: int,
        context_bucket_hi: int,
        profile_query_bucket_lo: int,
        profile_query_bucket_hi: int,
        num_tokens: int,
    ) -> SegmentCandidate | None:
        best: tuple[float, SegmentCandidate] | None = None
        for record in self.records:
            if record.phase != "decode":
                continue
            if record.backend not in self.allowed_backends:
                continue
            if self.fixed_backend and record.backend != self.fixed_backend:
                continue
            if not state_matches(record.backend, record.state_name, state_name):
                continue
            if not qnn_profile_record_matches_query_capacity(record, profile_query_bucket_hi):
                continue
            match = bucket_match(record, profile_query_bucket_lo, profile_query_bucket_hi, self.context_match)
            if match is None:
                continue
            candidate = SegmentCandidate(
                segment_id=-1,
                context_bucket_lo=context_bucket_lo,
                context_bucket_hi=context_bucket_hi,
                profile_query_bucket_lo=profile_query_bucket_lo,
                profile_query_bucket_hi=profile_query_bucket_hi,
                num_tokens=num_tokens,
                record=record,
                matched_profile_bucket_lo=record.bucket_lo,
                matched_profile_bucket_hi=record.bucket_hi,
                match_kind=match[1],
                feasible_for_slo=record.mean_tbt_ms <= self.slo_tbt_ms,
                selection_mode="candidate_probe",
            )
            item = (match[0], candidate)
            if best is None or item[0] < best[0]:
                best = item
        return best[1] if best else None

    def switch_reason(self, node: DpNode) -> str:
        prev_state = node.prev_state
        current = node.candidate
        if not prev_state or state_matches(current.record.backend, current.record.state_name, prev_state):
            return ""
        previous_candidate = self.raw_candidate_for_state(
            prev_state,
            current.context_bucket_lo,
            current.context_bucket_hi,
            current.profile_query_bucket_lo,
            current.profile_query_bucket_hi,
            current.num_tokens,
        )
        if previous_candidate is None or not previous_candidate.feasible_for_slo:
            return "slo"
        if node.step_energy_mj < previous_candidate.energy_mj:
            return "energy"
        return ""

    def energy_saving_vs_prev_state(self, node: DpNode) -> float | None:
        prev_state = node.prev_state
        current = node.candidate
        if not prev_state or state_matches(current.record.backend, current.record.state_name, prev_state):
            return None
        previous_candidate = self.raw_candidate_for_state(
            prev_state,
            current.context_bucket_lo,
            current.context_bucket_hi,
            current.profile_query_bucket_lo,
            current.profile_query_bucket_hi,
            current.num_tokens,
        )
        if previous_candidate is None:
            return None
        return previous_candidate.energy_mj - node.step_energy_mj

    def lookup_transition(self, from_state: str, to_state: str) -> TransitionCost:
        from_backend = self.backend_by_state.get(from_state) or backend_for_state(self.records, from_state)
        to_backend = self.backend_by_state.get(to_state) or backend_for_state(self.records, to_state)
        for row in self.transitions:
            row_from_state = str(row.get("from_state", ""))
            row_to_state = str(row.get("to_state", ""))
            if transition_state_matches(row_from_state, from_state, from_backend) and transition_state_matches(
                row_to_state,
                to_state,
                to_backend,
            ):
                return TransitionCost(
                    parse_float(row.get("latency_ms"), 0.0),
                    parse_float(row.get("energy_mj"), 0.0),
                    "state_profile",
                )
            if (
                from_backend
                and to_backend
                and str(row.get("from_backend", "")).upper() == from_backend
                and str(row.get("to_backend", "")).upper() == to_backend
                and not row.get("from_state")
                and not row.get("to_state")
            ):
                return TransitionCost(
                    parse_float(row.get("latency_ms"), 0.0),
                    parse_float(row.get("energy_mj"), 0.0),
                    "backend_profile",
                )
        if from_state == to_state or (from_backend and to_backend and from_backend == to_backend):
            return TransitionCost(0.0, 0.0, "same_state")
        backend_latency = backend_transition_latency_ms(from_backend, to_backend)
        if backend_latency is not None:
            return TransitionCost(backend_latency, 0.0, "backend_default")
        return TransitionCost(self.default_transition_latency_ms, self.default_transition_energy_mj, "default")


def parse_float(value: object, default: float | None = None) -> float:
    text = str(value if value is not None else "").strip()
    if not text:
        if default is None:
            raise ValueError("missing float")
        return default
    try:
        parsed = float(text)
    except ValueError:
        if default is None:
            raise
        return default
    if math.isnan(parsed) or math.isinf(parsed):
        if default is None:
            raise ValueError("non-finite float")
        return default
    return parsed


def normalize_backend(value: str) -> str:
    upper = value.strip().upper()
    if upper in {"QNN_NPU", "QNN-NPU", "HTP"}:
        return "NPU"
    if upper.startswith("NPU") or "NPU" in upper:
        return "NPU"
    if upper.startswith("GPU") or upper == "OPENCL":
        return "GPU"
    if upper.startswith("CPU"):
        return "CPU"
    return upper


def state_aliases(backend: str, state_name: str) -> set[str]:
    backend = normalize_backend(backend)
    clean = state_name.strip()
    aliases = {clean}
    if backend == "GPU":
        suffix = clean.removeprefix("gpu_")
        aliases.update({suffix, f"gpu_{suffix}"})
    elif backend == "NPU":
        suffix = clean.removeprefix("npu_")
        aliases.update({suffix, f"npu_{suffix}"})
    elif backend == "CPU":
        suffix = clean.removeprefix("cpu_")
        aliases.update({suffix, f"cpu_{suffix}"})
        simple = re.match(r"(?P<class>[A-Za-z0-9]+)_(?P<big>\d+)_(?P<little>\d+)$", suffix)
        if simple:
            group = simple.group("class")
            big = simple.group("big")
            little = simple.group("little")
            aliases.update(
                {
                    f"{group}_big{big}_little{little}",
                    f"cpu_{group}_big{big}_little{little}",
                }
            )
        verbose = re.match(r"(?P<class>[A-Za-z0-9]+)_big(?P<big>\d+)_little(?P<little>\d+)$", suffix)
        if verbose:
            group = verbose.group("class")
            big = verbose.group("big")
            little = verbose.group("little")
            aliases.update({f"{group}_{big}_{little}", f"cpu_{group}_{big}_{little}"})
    return aliases


def state_matches(backend: str, profile_state_name: str, requested_state: str) -> bool:
    return bool(state_aliases(backend, profile_state_name) & state_aliases(backend, requested_state))


def default_baseline_decode_state(backend: str) -> str:
    normalized = normalize_backend(backend)
    try:
        return DEFAULT_BASELINE_DECODE_STATES[normalized]
    except KeyError as exc:
        raise ValueError(f"no default baseline decode state configured for backend={backend!r}") from exc


def profile_record_from_mapping(item: dict[str, Any], allow_unstable: bool) -> ProfileRecord | None:
    status = str(item.get("status", "ok")).lower()
    stable = bool(item.get("stable", True))
    if not allow_unstable and (status != "ok" or not stable):
        return None
    throughput = parse_float(item.get("throughput_tps"), 0.0)
    power = parse_float(item.get("power_mw"), 0.0)
    bucket_hi = int(parse_float(item.get("bucket_hi") or item.get("length"), 0.0))
    bucket_lo = int(parse_float(item.get("bucket_lo") or item.get("length"), float(bucket_hi)))
    length = int(parse_float(item.get("length") or item.get("bucket_hi"), float(bucket_hi)))
    state_name = str(item.get("state_name") or "")
    if not state_name or throughput <= 0 or power < 0 or length <= 0 or bucket_lo <= 0 or bucket_hi <= 0:
        return None
    return ProfileRecord(
        phase=str(item.get("phase") or "").lower(),
        backend=normalize_backend(str(item.get("backend") or "")),
        state_name=state_name,
        state_group=str(item.get("state_group") or ""),
        length=length,
        bucket_lo=bucket_lo,
        bucket_hi=bucket_hi,
        throughput_tps=throughput,
        power_mw=power,
        stable=stable,
        status=status,
        metadata=dict(item.get("metadata") or {}),
    )


def load_csv_profile(path: Path, allow_unstable: bool) -> tuple[list[ProfileRecord], list[dict[str, Any]], dict[str, Any]]:
    records: list[ProfileRecord] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != PROFILE_FIELDS:
            raise ValueError(f"profile CSV schema mismatch in {path}; got {reader.fieldnames}, expected {PROFILE_FIELDS}")
        for row in reader:
            record = profile_record_from_mapping(row, allow_unstable)
            if record:
                records.append(record)
    return records, [], {"schema_version": "system_benefit_profile.csv.v1"}


def load_json_profile(path: Path, allow_unstable: bool) -> tuple[list[ProfileRecord], list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records: list[ProfileRecord] = []
    for item in payload.get("records", []):
        if not isinstance(item, dict):
            continue
        record = profile_record_from_mapping(item, allow_unstable)
        if record:
            records.append(record)
    return records, list(payload.get("transitions", [])), payload


def load_profile(path: Path, allow_unstable: bool) -> tuple[list[ProfileRecord], list[dict[str, Any]], dict[str, Any]]:
    if path.suffix.lower() == ".json":
        return load_json_profile(path, allow_unstable)
    return load_csv_profile(path, allow_unstable)


def length_distance(length: int, target: int) -> int:
    return abs(length - target)


def length_match(length: int, target: int, mode: str) -> tuple[int, str] | None:
    if length == target:
        return length, "exact"
    if mode == "exact":
        return None
    if mode == "floor" and length <= target:
        return length, "floor"
    if mode == "ceil" and length >= target:
        return length, "ceil"
    if mode == "nearest":
        return length, "nearest"
    return None


def bucket_mid(lo: int, hi: int) -> float:
    return (lo + hi) / 2.0


def bucket_match_distance(record: ProfileRecord, query_lo: int, query_hi: int) -> float:
    if record.bucket_lo == record.bucket_hi:
        return abs(record.length - query_hi)
    return abs(bucket_mid(record.bucket_lo, record.bucket_hi) - bucket_mid(query_lo, query_hi))


def bucket_match(record: ProfileRecord, query_lo: int, query_hi: int, mode: str) -> tuple[float, str] | None:
    # Old point-style profiles use length only; match them against the bucket end.
    if record.bucket_lo == record.bucket_hi:
        point = record.length
        if point == query_hi:
            return 0.0, "exact"
        if mode == "exact":
            return None
        if mode == "floor" and point <= query_hi:
            return abs(point - query_hi), "floor"
        if mode == "ceil" and point >= query_hi:
            return abs(point - query_hi), "ceil"
        if mode == "nearest":
            return abs(point - query_hi), "nearest"
        return None

    if record.bucket_lo == query_lo and record.bucket_hi == query_hi:
        return 0.0, "exact"
    if record.bucket_lo <= query_lo and record.bucket_hi >= query_hi:
        return 0.0, "covering_partial_bucket" if mode == "exact" else "covering_bucket"
    if mode == "exact":
        return None
    if mode == "floor" and record.bucket_hi <= query_hi:
        return bucket_match_distance(record, query_lo, query_hi), "floor"
    if mode == "ceil" and record.bucket_lo >= query_lo:
        return bucket_match_distance(record, query_lo, query_hi), "ceil"
    if mode == "nearest":
        return bucket_match_distance(record, query_lo, query_hi), "nearest"
    return None


def backend_for_state(records: Iterable[ProfileRecord], state_name: str) -> str | None:
    for record in records:
        if state_matches(record.backend, record.state_name, state_name):
            return record.backend
    return None


def transition_state_matches(row_state: str, query_state: str, backend: str | None) -> bool:
    if not row_state:
        return False
    if row_state == query_state:
        return True
    return state_matches(backend, row_state, query_state) if backend else False


def route_spec_for_state(
    backend: str,
    state_name: str,
    state_group: str = "",
    required_context: int | None = None,
) -> str:
    backend = normalize_backend(backend)
    text = f"{state_name} {state_group}".strip()
    if backend == "GPU":
        numbers = [int(item) for item in re.findall(r"\d+", text)]
        if not numbers:
            return "opencl"
        freq_mhz = numbers[-1]
        return f"opencl{{gpu_freq_hz={freq_mhz * 1_000_000}}}"
    if backend == "NPU":
        lowered = text.lower()
        workpoint = state_name.removeprefix("npu_")
        for candidate in [
            "low_balanced",
            "high_performance",
            "high_power_saver",
            "low_power_saver",
            "extreme_power_saver",
            "power_saver",
            "balanced",
            "burst",
            "native",
            "low",
        ]:
            if candidate in lowered:
                workpoint = candidate
                break
        explicit_context = qnn_context_size_from_state(state_name, state_group)
        capacity = (
            qnn_graph_capacity_for_context_size(explicit_context)
            if explicit_context is not None
            else qnn_graph_capacity_for_required(required_context or 0)
        )
        if capacity is None:
            return f"qnn-npu{{workpoint={workpoint}}}"
        return (
            f"qnn-npu{{workpoint={workpoint},qnn_context_size={capacity['context_size']},"
            f"qnn_required_kv_slots={capacity['usable_kv_slots']}}}"
        )
    if backend == "CPU":
        return cpu_route_spec_for_state(state_name, state_group)
    return state_name


def cpu_route_spec_for_state(state_name: str, state_group: str = "") -> str:
    suffix = state_name.removeprefix("cpu_")
    text = f"{suffix} {state_group}"
    numbers = [int(item) for item in re.findall(r"\d+", text)]
    freqs = [item for item in numbers if item >= 100000]
    group = state_group or suffix.split("_", 1)[0]
    group_upper = group.upper()
    if "B2S4" in group_upper and len(freqs) >= 2:
        big, little = freqs[0], freqs[1]
        return f"cpu{{threads=6,affinity=FC,cpu_policy0_freq_khz={little},cpu_policy6_freq_khz={big}}}"
    if "B2S2" in group_upper and len(freqs) >= 2:
        big, little = freqs[0], freqs[1]
        return f"cpu{{threads=4,affinity=CC,cpu_policy0_freq_khz={little},cpu_policy6_freq_khz={big}}}"
    if "S6" in group_upper and freqs:
        return f"cpu{{threads=6,affinity=3F,cpu_policy0_freq_khz={freqs[-1]}}}"
    if "B2" in group_upper and freqs:
        return f"cpu{{threads=2,affinity=C0,cpu_policy6_freq_khz={freqs[0]}}}"
    if "B1" in group_upper and freqs:
        return f"cpu{{threads=1,affinity=40,cpu_policy6_freq_khz={freqs[0]}}}"
    return "cpu"


def build_decode_schedule_env(segments: list[dict[str, Any]], bucket_size: int) -> str:
    entries: list[str] = []
    previous_spec: str | None = None
    generated_before_segment = 0
    for segment in segments:
        spec = route_spec_for_state(
            str(segment.get("backend") or ""),
            str(segment.get("selected_state") or ""),
            str(segment.get("state_group") or ""),
            int(segment.get("profile_query_bucket_hi") or segment.get("context_bucket_hi") or 0),
        )
        if spec == previous_spec:
            generated_before_segment += int(segment.get("num_tokens") or bucket_size)
            continue
        start_token = generated_before_segment + 1
        entries.append(f"{start_token}:{spec}")
        previous_spec = spec
        generated_before_segment += int(segment.get("num_tokens") or bucket_size)
    return ";".join(entries)


def empty_decode_plan(scheduler_name: str) -> dict[str, Any]:
    return {
        "scheduler": scheduler_name,
        "decode_latency_ms": 0.0,
        "decode_energy_mj": 0.0,
        "decode_throughput_tps": math.inf,
        "all_segments_profile_slo_met": True,
        "slo_met": True,
        "segments": [],
        "notes": [],
    }


def select_profile_for_length(
    records: list[ProfileRecord],
    *,
    phase: str,
    length: int,
    backend: str,
    state_name: str | None,
    context_match: str,
) -> tuple[ProfileRecord, str] | None:
    backend = normalize_backend(backend)
    candidates: list[tuple[ProfileRecord, str]] = []
    for record in records:
        if record.phase != phase:
            continue
        if record.backend != backend:
            continue
        if state_name and record.state_name != state_name:
            continue
        match = length_match(record.length, length, context_match)
        if match:
            candidates.append((record, match[1]))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (length_distance(item[0].length, length), item[0].energy_mj_per_token))
    return candidates[0]


def prefill_cost(
    records: list[ProfileRecord],
    *,
    input_len: int,
    backend: str,
    state_name: str | None,
    context_match: str,
    latency_override_ms: float | None,
    power_override_mw: float | None,
    energy_override_mj: float | None,
    missing_policy: str,
) -> dict[str, Any]:
    if latency_override_ms is not None:
        energy = energy_override_mj
        if energy is None and power_override_mw is not None:
            energy = power_override_mw * latency_override_ms / 1000.0
        if energy is None:
            energy = 0.0
        return {
            "state": state_name or f"{normalize_backend(backend).lower()}_prefill_override",
            "backend": normalize_backend(backend),
            "latency_ms": latency_override_ms,
            "energy_mj": energy,
            "throughput_tps": input_len / (latency_override_ms / 1000.0) if latency_override_ms > 0 else math.inf,
            "power_mw": power_override_mw,
            "source": "cli_override",
            "notes": [],
        }

    selected = select_profile_for_length(
        records,
        phase="prefill",
        length=input_len,
        backend=backend,
        state_name=state_name,
        context_match=context_match,
    )
    if selected:
        record, match_kind = selected
        latency_ms = input_len / record.throughput_tps * 1000.0
        return {
            "state": record.state_name,
            "backend": record.backend,
            "latency_ms": latency_ms,
            "energy_mj": record.power_mw * latency_ms / 1000.0,
            "throughput_tps": record.throughput_tps,
            "power_mw": record.power_mw,
            "source": "profile",
            "matched_profile_length": record.length,
            "context_match": match_kind,
            "notes": [] if match_kind == "exact" else [f"prefill_length_match_{match_kind}"],
        }

    if missing_policy == "error":
        raise ValueError(f"missing prefill profile for backend={backend} state={state_name or '*'} input_len={input_len}")
    return {
        "state": state_name or "",
        "backend": normalize_backend(backend),
        "latency_ms": 0.0,
        "energy_mj": 0.0,
        "throughput_tps": math.inf,
        "power_mw": None,
        "source": "missing_zero",
        "notes": ["prefill_profile_missing_zero_cost"],
    }


def merge_prefill_decode(prefill: dict[str, Any], decode: dict[str, Any], output_len: int) -> dict[str, Any]:
    total_latency = prefill["latency_ms"] + decode["decode_latency_ms"]
    total_energy = prefill["energy_mj"] + decode["decode_energy_mj"]
    notes = sorted(set(prefill.get("notes", [])) | set(decode.get("notes", [])))
    result = dict(decode)
    result.update(
        {
            "prefill_state": prefill["state"],
            "prefill_backend": prefill["backend"],
            "prefill_latency_ms": prefill["latency_ms"],
            "prefill_energy_mj": prefill["energy_mj"],
            "prefill_source": prefill["source"],
            "total_latency_ms": total_latency,
            "total_energy_mj": total_energy,
            "e2e_output_throughput_tps": output_len / (total_latency / 1000.0) if total_latency > 0 else math.inf,
            "notes": notes,
        }
    )
    return result


def pct_reduction(new_value: float, baseline_value: float) -> float | None:
    if baseline_value <= 0:
        return None
    return (baseline_value - new_value) / baseline_value * 100.0


def parse_backend_set(value: str) -> set[str]:
    return {normalize_backend(item) for item in value.split(",") if item.strip()}


def optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    return parse_float(value)


def resolve_slo_tbt_ms(args: argparse.Namespace) -> float:
    if args.slo_tbt_ms is not None:
        value = parse_float(args.slo_tbt_ms)
        if value <= 0:
            raise ValueError("--slo-tbt-ms must be positive")
        return value
    if args.slo_tps is not None:
        value = parse_float(args.slo_tps)
        if value <= 0:
            raise ValueError("--slo-tps must be positive")
        return 1000.0 / value
    raise ValueError("missing required SLO: pass --slo-tbt-ms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate bucketed CPU/GPU/NPU decode planning from an offline profile CSV.")
    parser.add_argument("--profile", default="profiles/system_benefit_offline_profile.csv")
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--output-len", type=int, required=True)
    parser.add_argument("--slo-tbt-ms", default=None, help="Per-bucket decode mean TBT SLO in milliseconds. Lower is stricter.")
    parser.add_argument("--slo-tps", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--bucket-size", type=int, default=32)
    parser.add_argument("--scheduler", choices=["energy-dp"], default="energy-dp")
    parser.add_argument("--allowed-backends", default="CPU,GPU,NPU")
    parser.add_argument("--context-match", choices=["exact", "nearest", "floor", "ceil"], default="nearest")
    parser.add_argument("--allow-unstable", action="store_true")
    parser.add_argument("--default-transition-latency-ms", type=float, default=0.0)
    parser.add_argument("--default-transition-energy-mj", type=float, default=0.0)
    parser.add_argument(
        "--initial-decode-state",
        "--current-state",
        dest="initial_decode_state",
        default=None,
        help="Current state before scheduled decode starts; first step includes current->selected transition.",
    )

    parser.add_argument("--prefill-backend", default="NPU")
    parser.add_argument("--prefill-state", default=None)
    parser.add_argument("--prefill-latency-ms", default=None)
    parser.add_argument("--prefill-power-mw", default=None)
    parser.add_argument("--prefill-energy-mj", default=None)
    parser.add_argument("--missing-prefill-policy", choices=["zero", "error"], default="zero")

    parser.add_argument("--baseline-prefill-backend", default="NPU")
    parser.add_argument("--baseline-prefill-state", default=None)
    parser.add_argument("--baseline-prefill-latency-ms", default=None)
    parser.add_argument("--baseline-prefill-power-mw", default=None)
    parser.add_argument("--baseline-prefill-energy-mj", default=None)
    parser.add_argument("--baseline-decode-backend", default="CPU")
    parser.add_argument(
        "--baseline-decode-state",
        default=None,
        help="Fixed baseline decode state. Defaults by backend: CPU=B2S4_4320000_3532800, GPU=1100, NPU=burst.",
    )
    parser.add_argument(
        "--baseline-initial-decode-state",
        default=None,
        help="Current state before baseline decode starts. Omit to keep baseline decode transition-free.",
    )

    parser.add_argument("--output", default=None, help="Write JSON result. If omitted, only a text summary is printed.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print the simulation request without writing output.")
    return parser.parse_args()


def build_decode_scheduler(
    records: list[ProfileRecord],
    transitions: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    fixed_backend: str | None = None,
    fixed_state: str | None = None,
    initial_state: str | None = None,
) -> EnergyDpScheduler:
    allowed = parse_backend_set(args.allowed_backends)
    if fixed_backend:
        allowed = {normalize_backend(fixed_backend)}
    return EnergyDpScheduler(
        records,
        transitions,
        slo_tbt_ms=args.slo_tbt_ms,
        context_match=args.context_match,
        allowed_backends=allowed,
        fixed_backend=normalize_backend(fixed_backend) if fixed_backend else None,
        fixed_state=fixed_state,
        initial_state=initial_state,
        default_transition_latency_ms=args.default_transition_latency_ms,
        default_transition_energy_mj=args.default_transition_energy_mj,
    )


def main() -> int:
    args = parse_args()
    try:
        args.slo_tbt_ms = resolve_slo_tbt_ms(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    profile_path = Path(args.profile)
    if not profile_path.is_absolute():
        profile_path = ROOT / profile_path
    records, transitions, profile_payload = load_profile(profile_path, args.allow_unstable)
    if not records:
        raise SystemExit(f"profile has no usable records: {profile_path}")
    try:
        baseline_decode_state = args.baseline_decode_state or default_baseline_decode_state(args.baseline_decode_backend)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.dry_run:
        decode_records = [record for record in records if record.phase == "decode"]
        prefill_records = [record for record in records if record.phase == "prefill"]
        print(f"profile={profile_path}")
        print(f"decode_records={len(decode_records)} prefill_records={len(prefill_records)} transitions={len(transitions)}")
        print(
            "request="
            f"input_len={args.input_len} output_len={args.output_len} "
            f"slo_tbt_ms={args.slo_tbt_ms} bucket_size={args.bucket_size}"
        )
        print(
            "scheduled="
            f"prefill_backend={normalize_backend(args.prefill_backend)} "
            f"allowed_backends={','.join(sorted(parse_backend_set(args.allowed_backends)))} "
            f"initial_decode_state={args.initial_decode_state or ''}"
        )
        print(
            "baseline="
            f"prefill_backend={normalize_backend(args.baseline_prefill_backend)} "
            f"decode_backend={normalize_backend(args.baseline_decode_backend)} "
            f"decode_state={baseline_decode_state} "
            f"initial_decode_state={args.baseline_initial_decode_state or ''}"
        )
        print(f"dry-run: would write {args.output}" if args.output else "dry-run: no output path requested")
        return 0

    scheduled_prefill = prefill_cost(
        records,
        input_len=args.input_len,
        backend=args.prefill_backend,
        state_name=args.prefill_state,
        context_match=args.context_match,
        latency_override_ms=optional_float(args.prefill_latency_ms),
        power_override_mw=optional_float(args.prefill_power_mw),
        energy_override_mj=optional_float(args.prefill_energy_mj),
        missing_policy=args.missing_prefill_policy,
    )
    scheduled_decode = build_decode_scheduler(records, transitions, args, initial_state=args.initial_decode_state).plan_decode(
        args.input_len,
        args.output_len,
        args.bucket_size,
    )
    scheduled = merge_prefill_decode(scheduled_prefill, scheduled_decode, args.output_len)

    baseline_prefill = prefill_cost(
        records,
        input_len=args.input_len,
        backend=args.baseline_prefill_backend,
        state_name=args.baseline_prefill_state,
        context_match=args.context_match,
        latency_override_ms=optional_float(args.baseline_prefill_latency_ms),
        power_override_mw=optional_float(args.baseline_prefill_power_mw),
        energy_override_mj=optional_float(args.baseline_prefill_energy_mj),
        missing_policy=args.missing_prefill_policy,
    )
    baseline_decode = build_decode_scheduler(
        records,
        transitions,
        args,
        fixed_backend=args.baseline_decode_backend,
        fixed_state=baseline_decode_state,
        initial_state=args.baseline_initial_decode_state,
    ).plan_decode(args.input_len, args.output_len, args.bucket_size)
    baseline = merge_prefill_decode(baseline_prefill, baseline_decode, args.output_len)

    relative = {
        "latency_reduction_pct": pct_reduction(scheduled["total_latency_ms"], baseline["total_latency_ms"]),
        "decode_latency_reduction_pct": pct_reduction(scheduled["decode_latency_ms"], baseline["decode_latency_ms"]),
        "energy_reduction_pct": pct_reduction(scheduled["total_energy_mj"], baseline["total_energy_mj"]),
        "decode_energy_reduction_pct": pct_reduction(scheduled["decode_energy_mj"], baseline["decode_energy_mj"]),
        "e2e_output_throughput_gain_pct": (
            (scheduled["e2e_output_throughput_tps"] / baseline["e2e_output_throughput_tps"] - 1.0) * 100.0
            if baseline["e2e_output_throughput_tps"] > 0
            else None
        ),
    }

    result = {
        "schema_version": SCHEMA_VERSION,
        "profile_schema_version": profile_payload.get("schema_version", ""),
        "request": {
            "input_len": args.input_len,
            "output_len": args.output_len,
            "slo_tbt_ms": args.slo_tbt_ms,
            "bucket_size": args.bucket_size,
            "context_match": args.context_match,
            "allowed_backends": sorted(parse_backend_set(args.allowed_backends)),
            "baseline_decode_backend": normalize_backend(args.baseline_decode_backend),
            "baseline_decode_state": baseline_decode_state,
            "initial_decode_state": args.initial_decode_state,
            "baseline_initial_decode_state": args.baseline_initial_decode_state,
            "transition_default_latency_ms": args.default_transition_latency_ms,
            "transition_default_energy_mj": args.default_transition_energy_mj,
        },
        "scheduled": scheduled,
        "baseline": baseline,
        "relative_to_baseline": relative,
    }

    output_text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text, encoding="utf-8")
        print(f"wrote {output_path}")

    print("scheduled:")
    print(f"  total_latency_ms={scheduled['total_latency_ms']:.3f}")
    print(f"  e2e_output_throughput_tps={scheduled['e2e_output_throughput_tps']:.3f}")
    print(f"  total_energy_mj={scheduled['total_energy_mj']:.3f}")
    print(f"  decode_slo_met={scheduled['slo_met']}")
    print(f"  decode_slo_satisfaction_rate={scheduled['slo_satisfaction_rate']:.3f}")
    print("baseline:")
    print(f"  total_latency_ms={baseline['total_latency_ms']:.3f}")
    print(f"  e2e_output_throughput_tps={baseline['e2e_output_throughput_tps']:.3f}")
    print(f"  total_energy_mj={baseline['total_energy_mj']:.3f}")
    print("relative_to_baseline:")
    for key, value in relative.items():
        print(f"  {key}={value:.3f}" if value is not None else f"  {key}=NA")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
