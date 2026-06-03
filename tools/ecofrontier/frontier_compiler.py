from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .profile_loader import LoadedProfiles
from .profile_schema import CompilerConfig, StateProfile, TransitionProfile, count_by


def compile_frontier(loaded: LoadedProfiles, config: Optional[CompilerConfig] = None) -> Dict[str, Any]:
    config = config or CompilerConfig()
    states = [state.normalized(config) for state in loaded.states]
    transitions = [transition.normalized() for transition in loaded.transitions]
    models = _build_models(states, config)
    frontiers, dominated_states = _build_frontiers(states, config)
    caveats = _build_caveats(states, transitions, frontiers, loaded)

    artifact: Dict[str, Any] = {
        "version": "ecofrontier.offline_frontier.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(loaded.input_dir) if loaded.input_dir else None,
        "source_files": loaded.source_files,
        "skipped_sources": loaded.skipped_sources,
        "compiler_config": config.to_artifact_dict(),
        "raw_profile_summary": _raw_profile_summary(states, transitions, loaded),
        "normalized_states": [state.to_artifact_dict() for state in states],
        "models": models,
        "frontiers": frontiers,
        "dominated_states": dominated_states,
        "transition_edges": [transition.to_artifact_dict() for transition in transitions],
        "graph_catalog_summary": _graph_catalog_summary(loaded),
        "data_quality_summary": _data_quality_summary(states, loaded),
        "source_slo_frontiers": loaded.source_slo_frontiers,
        "source_caveats": loaded.paper_ready_caveats,
        "energy_policy": loaded.energy_policy,
        "caveats": caveats,
    }
    return artifact


def _build_models(states: Iterable[StateProfile], config: CompilerConfig) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[StateProfile]] = defaultdict(list)
    for state in states:
        if state.length_value() is not None and state.latency_value() is not None:
            grouped[(state.state_id, state.phase)].append(state)

    models: List[Dict[str, Any]] = []
    for (state_id, phase), group in sorted(grouped.items()):
        points = []
        for state in sorted(group, key=lambda item: item.length_value() or -1):
            point: Dict[str, Any] = {
                "length_value": state.length_value(),
                "model_result": "exact_bucket",
                "source_file": state.source_file,
                "energy_complete": state.energy_complete,
                "data_quality": state.data_quality,
            }
            if phase == "decode":
                point["tbt_us"] = state.tbt_us
                point["length_axis"] = "context_len"
            else:
                point["ttft_ms_p95"] = state.ttft_ms_p95
                point["length_axis"] = "prompt_tokens"
            if state.energy_mj_per_token is not None:
                point["energy_mj_per_token"] = state.energy_mj_per_token
                point["energy_source"] = state.energy_source
            points.append(point)
        models.append(
            {
                "state_id": state_id,
                "backend": group[0].backend,
                "phase": phase,
                "model_kind": "discrete_state_piecewise_linear_length_model",
                "discrete_state_dimensions": [
                    "backend",
                    "cpu_affinity",
                    "cpu_freq_khz",
                    "gpu_freq_mhz",
                    "npu_workpoint",
                    "graph_id",
                ],
                "length_axis": "context_len" if phase == "decode" else "prompt_tokens",
                "length_interpolation": "allowed_within_state_only" if config.allow_length_interpolation else "disabled",
                "extrapolation": "disabled" if not config.allow_extrapolation else "allowed_conservative",
                "points": points,
            }
        )
    return models


def _build_frontiers(states: List[StateProfile], config: CompilerConfig) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    frontiers: List[Dict[str, Any]] = []
    dominated_by_key: Dict[Tuple[str, int, str, str, str], Dict[str, Any]] = {}
    phase_lengths = sorted(
        {
            (state.phase, state.length_value())
            for state in states
            if state.phase in {"decode", "prefill"} and state.length_value() is not None
        },
        key=lambda item: (item[0], item[1] or -1),
    )

    for phase, length_value in phase_lengths:
        phase_slos = config.slo_tbt_us_values if phase == "decode" else config.slo_ttft_ms_values
        bucket_states = [state for state in states if state.phase == phase and state.length_value() == length_value]
        for slo in phase_slos:
            candidates: List[Dict[str, Any]] = []
            excluded_counts = Counter()
            for state in bucket_states:
                exclusion = _exclusion_reason(state, config)
                if exclusion:
                    excluded_counts[exclusion] += 1
                    continue
                candidate = _candidate_for_state(state)
                if candidate is None:
                    excluded_counts["missing_latency"] += 1
                    continue
                metric = candidate["latency_us"] if phase == "decode" else candidate["latency_ms"]
                if metric <= slo:
                    candidates.append(candidate)

            frontier_kind = _frontier_kind(candidates)
            frontier, dominated = _pareto_prune(candidates, frontier_kind, config)
            for item in dominated:
                key = (item["phase"], int(item["length_value"]), item["state_id"], item["dominated_by"], item["dominance_reason"])
                dominated_by_key[key] = item

            frontiers.append(
                {
                    "phase": phase,
                    "length_axis": "context_len" if phase == "decode" else "prompt_tokens",
                    "length_value": length_value,
                    "slo_kind": "tbt_us" if phase == "decode" else "ttft_ms",
                    "slo_value": slo,
                    "frontier_kind": frontier_kind,
                    "feasible_candidates": [_public_candidate(candidate) for candidate in candidates],
                    "frontier": frontier,
                    "excluded_counts": dict(excluded_counts),
                    "notes": "" if candidates else "no_state_meets_slo",
                }
            )
    return frontiers, list(dominated_by_key.values())


def _candidate_for_state(state: StateProfile) -> Optional[Dict[str, Any]]:
    latency = state.latency_value()
    length = state.length_value()
    if latency is None or length is None:
        return None
    candidate: Dict[str, Any] = {
        "state_id": state.state_id,
        "backend": state.backend,
        "phase": state.phase,
        "length_value": length,
        "active_power_mw": state.active_power_mw,
        "energy_complete": state.energy_complete,
        "data_quality": list(state.data_quality),
        "source_file": state.source_file,
        "_state": state,
    }
    if state.phase == "decode":
        candidate["latency_us"] = latency
        candidate["tbt_us"] = state.tbt_us
    else:
        candidate["latency_ms"] = latency
        candidate["ttft_ms_p95"] = state.ttft_ms_p95
    if state.energy_mj_per_token is not None:
        candidate["energy_mj_per_token"] = state.energy_mj_per_token
        candidate["energy_source"] = state.energy_source
    return candidate


def _exclusion_reason(state: StateProfile, config: CompilerConfig) -> Optional[str]:
    if config.filter_unsupported and state.support_status != "ok":
        return "unsupported"
    if config.filter_fallback_used and state.fallback_used:
        return "fallback_used"
    if config.filter_unstable and not state.stable:
        return "unstable"
    if config.filter_thermal_unsafe and not state.thermal_safe:
        return "thermal_unsafe"
    return None


def _frontier_kind(candidates: List[Dict[str, Any]]) -> str:
    if not candidates:
        return "latency_only_frontier"
    energy_values = [candidate.get("energy_mj_per_token") for candidate in candidates]
    all_energy = all(value is not None for value in energy_values)
    all_measured = all(candidate.get("energy_complete") is True for candidate in candidates)
    if all_energy and all_measured:
        return "measured_energy_frontier"
    if all_energy:
        return "estimated_energy_frontier"
    if any(candidate.get("active_power_mw") is not None for candidate in candidates):
        return "latency_power_frontier"
    return "latency_only_frontier"


def _pareto_prune(
    candidates: List[Dict[str, Any]],
    frontier_kind: str,
    config: CompilerConfig,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    dominated: List[Dict[str, Any]] = []
    frontier: List[Dict[str, Any]] = []
    for candidate in candidates:
        dominator: Optional[Dict[str, Any]] = None
        reason = ""
        for other in candidates:
            if other is candidate:
                continue
            dominates, candidate_reason = _dominates(other, candidate, frontier_kind, config)
            if dominates:
                dominator = other
                reason = candidate_reason
                break
        if dominator is None:
            frontier.append(_public_candidate(candidate))
        else:
            dominated.append(
                {
                    "phase": candidate["phase"],
                    "length_value": candidate["length_value"],
                    "state_id": candidate["state_id"],
                    "dominated_by": dominator["state_id"],
                    "dominance_reason": reason,
                    "frontier_kind": frontier_kind,
                }
            )
    return frontier, dominated


def _dominates(
    a: Dict[str, Any],
    b: Dict[str, Any],
    frontier_kind: str,
    config: CompilerConfig,
) -> Tuple[bool, str]:
    latency_key = "latency_us" if a["phase"] == "decode" else "latency_ms"
    latency_a = a[latency_key]
    latency_b = b[latency_key]
    if latency_a > latency_b:
        return False, ""
    if not _quality_not_worse(a, b):
        return False, ""

    reason = "latency_only"
    second_a: Optional[float] = None
    second_b: Optional[float] = None
    if frontier_kind == "measured_energy_frontier":
        second_a = a.get("energy_mj_per_token")
        second_b = b.get("energy_mj_per_token")
        reason = "measured_energy"
    elif frontier_kind == "estimated_energy_frontier":
        source_a = a.get("energy_source")
        source_b = b.get("energy_source")
        if source_a == source_b or (source_a != "measured_or_profiled" and source_b != "measured_or_profiled"):
            second_a = a.get("energy_mj_per_token")
            second_b = b.get("energy_mj_per_token")
            reason = "estimated_energy"
    if second_a is None and second_b is None and config.enable_power_comparison_when_energy_unavailable:
        second_a = a.get("active_power_mw")
        second_b = b.get("active_power_mw")
        reason = "active_power"

    strict = latency_a < latency_b
    if second_a is not None and second_b is not None:
        if second_a > second_b:
            return False, ""
        strict = strict or second_a < second_b
    if second_a is None or second_b is None:
        reason = "latency_only"
    return strict, reason


def _quality_not_worse(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    state_a: StateProfile = a["_state"]
    state_b: StateProfile = b["_state"]
    if state_a.support_status != "ok" and state_b.support_status == "ok":
        return False
    if state_a.fallback_used and not state_b.fallback_used:
        return False
    if not state_a.stable and state_b.stable:
        return False
    return _quality_rank(state_a.data_quality) <= _quality_rank(state_b.data_quality)


def _quality_rank(quality: List[str]) -> int:
    rank = 0
    if "power_low_confidence" in quality:
        rank += 1
    if "unstable_power_window" in quality:
        rank += 2
    if "frequency_mismatch" in quality:
        rank += 2
    if "fallback_used" in quality or "unsupported" in quality:
        rank += 3
    return rank


def _public_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in candidate.items() if key != "_state" and value is not None}


def _raw_profile_summary(
    states: List[StateProfile],
    transitions: List[TransitionProfile],
    loaded: LoadedProfiles,
) -> Dict[str, Any]:
    return {
        "state_count": len(states),
        "transition_count": len(transitions),
        "graph_count": len(loaded.graphs),
        "rejected_graph_count": len(loaded.rejected_graphs),
        "rows_by_source": {source["path"]: source.get("row_count", 0) for source in loaded.source_files},
        "states_by_backend": count_by(states, lambda item: item.backend),
        "states_by_phase": count_by(states, lambda item: item.phase),
        "unstable_state_count": sum(1 for state in states if not state.stable),
    }


def _data_quality_summary(states: List[StateProfile], loaded: LoadedProfiles) -> Dict[str, Any]:
    quality = Counter()
    for state in states:
        if state.data_quality:
            quality.update(state.data_quality)
        else:
            quality["unspecified"] += 1
    return {
        "by_quality": dict(quality),
        "unstable_state_count": sum(1 for state in states if not state.stable),
        "unsupported_state_count": sum(1 for state in states if state.support_status != "ok"),
        "fallback_state_count": sum(1 for state in states if state.fallback_used),
        "thermal_unsafe_state_count": sum(1 for state in states if not state.thermal_safe),
        "source_data_quality_summary": loaded.source_data_quality_summary,
    }


def _graph_catalog_summary(loaded: LoadedProfiles) -> Dict[str, Any]:
    return {
        "graph_count": len(loaded.graphs),
        "graphs": [graph.to_artifact_dict() for graph in loaded.graphs],
        "rejected_graphs": loaded.rejected_graphs,
        "qnn_aot_metadata": loaded.graph_metadata,
    }


def _build_caveats(
    states: List[StateProfile],
    transitions: List[TransitionProfile],
    frontiers: List[Dict[str, Any]],
    loaded: LoadedProfiles,
) -> List[str]:
    caveats = set()
    if any(not transition.transition_energy_complete for transition in transitions):
        caveats.add("transition_energy_unavailable")
    if any(state.energy_source == "estimated_power_latency" for state in states):
        caveats.add("energy_estimated_from_power_latency")
    if any(not state.energy_complete for state in states):
        caveats.add("energy_incomplete_frontier")
    if any("unstable_power_window" in state.data_quality for state in states):
        caveats.add("unstable_power_windows_present")
    if any(str(source.get("type", "")).startswith("markdown") for source in loaded.source_files):
        caveats.add("markdown_extraction_used")
    if loaded.rejected_graphs or any("qnn_aot_context_size" in state.metadata for state in states) or loaded.graph_metadata:
        caveats.add("qnn_context_size_not_usable_kv_slots")
    if any(frontier["frontier_kind"] in {"latency_power_frontier", "latency_only_frontier"} for frontier in frontiers):
        caveats.add("latency_only_or_latency_power_frontier")
    return sorted(caveats)
