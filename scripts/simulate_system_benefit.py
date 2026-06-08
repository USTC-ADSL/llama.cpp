#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "system_benefit_simulation.v1"


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
    prev_state: str | None
    candidate: SegmentCandidate
    transition: TransitionCost


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
        slo_tps: float,
        context_match: str,
        allowed_backends: set[str],
        fixed_backend: str | None = None,
        fixed_state: str | None = None,
        default_transition_latency_ms: float = 0.0,
        default_transition_energy_mj: float = 0.0,
    ) -> None:
        self.records = records
        self.transitions = transitions
        self.slo_tps = slo_tps
        self.context_match = context_match
        self.allowed_backends = allowed_backends
        self.fixed_backend = fixed_backend
        self.fixed_state = fixed_state
        self.default_transition_latency_ms = default_transition_latency_ms
        self.default_transition_energy_mj = default_transition_energy_mj

    def plan_decode(self, input_len: int, output_len: int, bucket_size: int) -> dict[str, Any]:
        if output_len <= 0:
            return empty_decode_plan(self.name)

        segments: list[list[SegmentCandidate]] = []
        generated = 0
        segment_id = 0
        while generated < output_len:
            num_tokens = min(bucket_size, output_len - generated)
            context_bucket_lo = input_len + generated + 1
            context_bucket_hi = input_len + generated + num_tokens
            candidates = self.candidates_for_segment(segment_id, context_bucket_lo, context_bucket_hi, num_tokens)
            if not candidates:
                raise ValueError(
                    "no decode profile candidate for "
                    f"segment {segment_id} context_bucket={context_bucket_lo}-{context_bucket_hi}"
                )
            segments.append(candidates)
            generated += num_tokens
            segment_id += 1

        dp: list[dict[str, DpNode]] = []
        for index, candidates in enumerate(segments):
            layer: dict[str, DpNode] = {}
            for candidate in candidates:
                state = candidate.record.state_name
                if index == 0:
                    transition = TransitionCost(0.0, 0.0, "initial")
                    layer[state] = DpNode(candidate.energy_mj, None, candidate, transition)
                    continue

                best_node: DpNode | None = None
                for prev_state, prev_node in dp[index - 1].items():
                    transition = self.lookup_transition(prev_state, state)
                    total = prev_node.total_energy_mj + transition.energy_mj + candidate.energy_mj
                    node = DpNode(total, prev_state, candidate, transition)
                    if best_node is None or node.total_energy_mj < best_node.total_energy_mj:
                        best_node = node
                if best_node:
                    layer[state] = best_node
            dp.append(layer)

        if not dp or not dp[-1]:
            raise ValueError("DP failed to produce a decode plan")

        final_state, final_node = min(dp[-1].items(), key=lambda item: item[1].total_energy_mj)
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
        decode_energy_mj = 0.0
        all_segments_profile_slo_met = True
        notes: set[str] = set()
        for node in chosen:
            candidate = node.candidate
            transition = node.transition
            segment_latency = transition.latency_ms + candidate.latency_ms
            segment_energy = transition.energy_mj + candidate.energy_mj
            decode_latency_ms += segment_latency
            decode_energy_mj += segment_energy
            all_segments_profile_slo_met = all_segments_profile_slo_met and candidate.feasible_for_slo
            if transition.source == "default":
                notes.add("missing_transition_profile")
            if candidate.match_kind != "exact":
                notes.add(f"context_match_{candidate.match_kind}")
            if not candidate.feasible_for_slo:
                notes.add("best_effort_bucket_below_slo")
            segment_rows.append(
                {
                    "segment_id": candidate.segment_id,
                    "context_len": candidate.context_bucket_hi,
                    "context_bucket_lo": candidate.context_bucket_lo,
                    "context_bucket_hi": candidate.context_bucket_hi,
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
                    "transition_latency_ms": transition.latency_ms,
                    "transition_energy_mj": transition.energy_mj,
                    "transition_source": transition.source,
                    "segment_decode_latency_ms": candidate.latency_ms,
                    "segment_decode_energy_mj": candidate.energy_mj,
                    "segment_total_latency_ms": segment_latency,
                    "segment_total_energy_mj": segment_energy,
                    "feasible_for_slo": candidate.feasible_for_slo,
                    "selection_mode": candidate.selection_mode,
                }
            )

        achieved_decode_tps = output_len / (decode_latency_ms / 1000.0) if decode_latency_ms > 0 else math.inf
        return {
            "scheduler": self.name,
            "decode_latency_ms": decode_latency_ms,
            "decode_energy_mj": decode_energy_mj,
            "decode_throughput_tps": achieved_decode_tps,
            "all_segments_profile_slo_met": all_segments_profile_slo_met,
            "slo_met": all_segments_profile_slo_met,
            "segments": segment_rows,
            "notes": sorted(notes),
        }

    def candidates_for_segment(
        self,
        segment_id: int,
        context_bucket_lo: int,
        context_bucket_hi: int,
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
            if self.fixed_state and record.state_name != self.fixed_state:
                continue
            match = bucket_match(record, context_bucket_lo, context_bucket_hi, self.context_match)
            if match is None:
                continue
            previous = candidates_by_state.get(record.state_name)
            if previous is None or match[0] < bucket_match_distance(previous[0], context_bucket_lo, context_bucket_hi):
                candidates_by_state[record.state_name] = (record, match[1])

        raw: list[SegmentCandidate] = []
        for record, match_kind in candidates_by_state.values():
            raw.append(
                SegmentCandidate(
                    segment_id=segment_id,
                    context_bucket_lo=context_bucket_lo,
                    context_bucket_hi=context_bucket_hi,
                    num_tokens=num_tokens,
                    record=record,
                    matched_profile_bucket_lo=record.bucket_lo,
                    matched_profile_bucket_hi=record.bucket_hi,
                    match_kind=match_kind,
                    feasible_for_slo=record.throughput_tps >= self.slo_tps,
                    selection_mode="feasible",
                )
            )
        if not raw:
            return []

        feasible = [candidate for candidate in raw if candidate.feasible_for_slo]
        if feasible:
            return feasible

        best_throughput = max(candidate.record.throughput_tps for candidate in raw)
        return [
            SegmentCandidate(
                segment_id=candidate.segment_id,
                context_bucket_lo=candidate.context_bucket_lo,
                context_bucket_hi=candidate.context_bucket_hi,
                num_tokens=candidate.num_tokens,
                record=candidate.record,
                matched_profile_bucket_lo=candidate.matched_profile_bucket_lo,
                matched_profile_bucket_hi=candidate.matched_profile_bucket_hi,
                match_kind=candidate.match_kind,
                feasible_for_slo=False,
                selection_mode="best_effort_closest_to_slo",
            )
            for candidate in raw
            if candidate.record.throughput_tps == best_throughput
        ]

    def lookup_transition(self, from_state: str, to_state: str) -> TransitionCost:
        if from_state == to_state:
            return TransitionCost(0.0, 0.0, "same_state")
        from_backend = backend_for_state(self.records, from_state)
        to_backend = backend_for_state(self.records, to_state)
        for row in self.transitions:
            if str(row.get("from_state", "")) == from_state and str(row.get("to_state", "")) == to_state:
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


def load_profile(path: Path, allow_unstable: bool) -> tuple[list[ProfileRecord], list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records: list[ProfileRecord] = []
    for item in payload.get("records", []):
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", "ok")).lower()
        stable = bool(item.get("stable", True))
        if not allow_unstable and (status != "ok" or not stable):
            continue
        throughput = parse_float(item.get("throughput_tps"), 0.0)
        power = parse_float(item.get("power_mw"), 0.0)
        length = int(parse_float(item.get("length"), 0.0))
        bucket_lo = int(parse_float(item.get("bucket_lo"), float(length)))
        bucket_hi = int(parse_float(item.get("bucket_hi"), float(length)))
        state_name = str(item.get("state_name") or "")
        if not state_name or throughput <= 0 or power < 0 or length <= 0 or bucket_lo <= 0 or bucket_hi <= 0:
            continue
        records.append(
            ProfileRecord(
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
        )
    return records, list(payload.get("transitions", [])), payload


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
    if mode == "exact":
        return None
    if record.bucket_lo <= query_lo and record.bucket_hi >= query_hi:
        return 0.0, "covering_bucket"
    if mode == "floor" and record.bucket_hi <= query_hi:
        return bucket_match_distance(record, query_lo, query_hi), "floor"
    if mode == "ceil" and record.bucket_lo >= query_lo:
        return bucket_match_distance(record, query_lo, query_hi), "ceil"
    if mode == "nearest":
        return bucket_match_distance(record, query_lo, query_hi), "nearest"
    return None


def backend_for_state(records: Iterable[ProfileRecord], state_name: str) -> str | None:
    for record in records:
        if record.state_name == state_name:
            return record.backend
    return None


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate bucketed CPU/GPU/NPU decode planning from an offline profile JSON.")
    parser.add_argument("--profile", default="profiles/system_benefit_offline_profile.json")
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--output-len", type=int, required=True)
    parser.add_argument("--slo-tps", type=float, required=True, help="Per-bucket decode throughput SLO in tokens/s.")
    parser.add_argument("--bucket-size", type=int, default=32)
    parser.add_argument("--scheduler", choices=["energy-dp"], default="energy-dp")
    parser.add_argument("--allowed-backends", default="CPU,GPU,NPU")
    parser.add_argument("--context-match", choices=["exact", "nearest", "floor", "ceil"], default="nearest")
    parser.add_argument("--allow-unstable", action="store_true")
    parser.add_argument("--default-transition-latency-ms", type=float, default=0.0)
    parser.add_argument("--default-transition-energy-mj", type=float, default=0.0)

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
    parser.add_argument("--baseline-decode-state", default=None)

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
) -> EnergyDpScheduler:
    allowed = parse_backend_set(args.allowed_backends)
    if fixed_backend:
        allowed = {normalize_backend(fixed_backend)}
    return EnergyDpScheduler(
        records,
        transitions,
        slo_tps=args.slo_tps,
        context_match=args.context_match,
        allowed_backends=allowed,
        fixed_backend=normalize_backend(fixed_backend) if fixed_backend else None,
        fixed_state=fixed_state,
        default_transition_latency_ms=args.default_transition_latency_ms,
        default_transition_energy_mj=args.default_transition_energy_mj,
    )


def main() -> int:
    args = parse_args()
    profile_path = Path(args.profile)
    if not profile_path.is_absolute():
        profile_path = ROOT / profile_path
    records, transitions, profile_payload = load_profile(profile_path, args.allow_unstable)
    if not records:
        raise SystemExit(f"profile has no usable records: {profile_path}")
    if args.dry_run:
        decode_records = [record for record in records if record.phase == "decode"]
        prefill_records = [record for record in records if record.phase == "prefill"]
        print(f"profile={profile_path}")
        print(f"decode_records={len(decode_records)} prefill_records={len(prefill_records)} transitions={len(transitions)}")
        print(
            "request="
            f"input_len={args.input_len} output_len={args.output_len} "
            f"slo_tps={args.slo_tps} bucket_size={args.bucket_size}"
        )
        print(
            "scheduled="
            f"prefill_backend={normalize_backend(args.prefill_backend)} "
            f"allowed_backends={','.join(sorted(parse_backend_set(args.allowed_backends)))}"
        )
        print(
            "baseline="
            f"prefill_backend={normalize_backend(args.baseline_prefill_backend)} "
            f"decode_backend={normalize_backend(args.baseline_decode_backend)}"
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
    scheduled_decode = build_decode_scheduler(records, transitions, args).plan_decode(
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
        fixed_state=args.baseline_decode_state,
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
            "slo_tps": args.slo_tps,
            "bucket_size": args.bucket_size,
            "context_match": args.context_match,
            "allowed_backends": sorted(parse_backend_set(args.allowed_backends)),
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
