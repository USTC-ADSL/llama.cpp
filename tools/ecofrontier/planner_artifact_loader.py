from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .planner_schema import GraphEntry, PlannerRequest, PlannerState, TransitionEdge, as_bool, as_float, as_int, as_list


@dataclass
class FrontierArtifact:
    path: Path
    raw: Dict[str, Any]
    states: List[PlannerState]
    transitions: List[TransitionEdge]
    graphs: List[GraphEntry]
    caveats: List[str]
    compiler_config: Dict[str, Any]

    def states_for_phase(self, phase: str) -> List[PlannerState]:
        return [state for state in self.states if state.phase == phase]


def load_artifact(path: Path | str) -> FrontierArtifact:
    artifact_path = Path(path)
    raw = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{artifact_path}: expected JSON object")
    state_rows = raw.get("normalized_states") or raw.get("states") or []
    if not isinstance(state_rows, list):
        raise ValueError(f"{artifact_path}: normalized_states must be an array")
    transitions = [_transition_from_mapping(row) for row in raw.get("transition_edges", []) if isinstance(row, dict)]
    graphs = _load_graph_entries(raw.get("graph_catalog_summary", {}))
    return FrontierArtifact(
        path=artifact_path,
        raw=raw,
        states=[state for row in state_rows if isinstance(row, dict) for state in [_state_from_mapping(row)] if state],
        transitions=transitions,
        graphs=graphs,
        caveats=[str(item) for item in as_list(raw.get("caveats"))],
        compiler_config=dict(raw.get("compiler_config") or {}),
    )


def load_requests(path: Path | str) -> List[PlannerRequest]:
    request_path = Path(path)
    raw = json.loads(request_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "requests" in raw:
        raw = raw["requests"]
    if not isinstance(raw, list):
        raise ValueError(f"{request_path}: expected a JSON array or object with requests")
    return [PlannerRequest.from_mapping(item, index) for index, item in enumerate(raw) if isinstance(item, dict)]


def generate_request_grid(artifact: FrontierArtifact) -> List[PlannerRequest]:
    decode_lengths = sorted({state.length_value for state in artifact.states if state.phase == "decode"})
    available = [length for length in (0, 512, 1024, 1536, 1792) if length in decode_lengths]
    if not available:
        available = decode_lengths[:5] or [0]
    prefill_lengths = sorted({state.length_value for state in artifact.states if state.phase == "prefill"})
    prompt_tokens = 512 if 512 in prefill_lengths else (prefill_lengths[-1] if prefill_lengths else 512)
    requests: List[PlannerRequest] = []
    for context_len in available:
        for slo_tbt_us in (45000.0, 55000.0, 70000.0):
            for predicted_hi in (64, 128):
                requests.append(
                    PlannerRequest(
                        request_id=f"grid-L{context_len}-tbt{int(slo_tbt_us)}-hi{predicted_hi}",
                        prompt_tokens=prompt_tokens,
                        context_len=context_len,
                        predicted_output_mean=64,
                        predicted_output_hi=predicted_hi,
                        slo_ttft_ms=4000.0,
                        slo_tbt_us=slo_tbt_us,
                    )
                )
    return requests


def transition_index(transitions: Iterable[TransitionEdge]) -> Dict[Tuple[str, str], List[TransitionEdge]]:
    indexed: Dict[Tuple[str, str], List[TransitionEdge]] = defaultdict(list)
    for edge in transitions:
        indexed[(edge.from_state_id, edge.to_state_id)].append(edge)
    for edges in indexed.values():
        edges.sort(key=lambda edge: -1 if edge.context_len is None else edge.context_len)
    return dict(indexed)


def _state_from_mapping(row: Dict[str, Any]) -> Optional[PlannerState]:
    phase = str(row.get("phase") or "").strip().lower()
    if phase not in {"prefill", "decode"}:
        return None
    state_id = str(row.get("state_id") or "").strip()
    if not state_id:
        return None
    if phase == "decode":
        length_value = as_int(row.get("context_len", row.get("length_value")))
        tbt_us = as_float(row.get("tbt_us", row.get("latency_us")))
        latency = tbt_us
        latency_field = "tbt_us"
        tbt_source = str(row.get("tbt_source") or ("artifact_tbt_us" if row.get("tbt_us") is not None else "artifact_latency_us"))
        ttft_source = "not_applicable"
        latency_quantile = "p95" if row.get("tbt_us_p95") is not None else "mean"
        slo_basis = "p95_tbt" if row.get("tbt_us_p95") is not None else "mean_tbt"
        latency_source = tbt_source
        latency_complete = row.get("tbt_us_p95") is not None
    else:
        length_value = as_int(row.get("prompt_tokens", row.get("length_value")))
        latency = as_float(row.get("ttft_ms_p95", row.get("latency_ms")))
        latency_field = "ttft_ms_p95" if row.get("ttft_ms_p95") is not None else "latency_ms"
        tbt_source = "not_applicable"
        ttft_source = str(row.get("ttft_source") or ("artifact_ttft_ms_p95" if row.get("ttft_ms_p95") is not None else "artifact_latency_ms"))
        latency_quantile = "p95" if row.get("ttft_ms_p95") is not None else "mean"
        slo_basis = "p95_ttft" if row.get("ttft_ms_p95") is not None else "mean_ttft"
        latency_source = ttft_source
        latency_complete = "prefill_latency_proxy" not in [str(item) for item in as_list(row.get("data_quality"))]
    if length_value is None or latency is None:
        return None
    energy_source = str(row.get("energy_source") or "unavailable")
    data_quality = [str(item) for item in as_list(row.get("data_quality"))]
    return PlannerState(
        state_id=state_id,
        backend=str(row.get("backend") or "").strip() or "unknown",
        phase=phase,
        length_value=length_value,
        latency_value=float(latency),
        latency_field=latency_field,
        latency_quantile=latency_quantile,
        slo_check_basis=slo_basis,
        latency_source=latency_source,
        latency_complete=latency_complete,
        tbt_us=as_float(row.get("tbt_us", row.get("latency_us"))) if phase == "decode" else None,
        ttft_ms=as_float(row.get("ttft_ms_p95", row.get("latency_ms"))) if phase == "prefill" else None,
        tbt_source=tbt_source,
        ttft_source=ttft_source,
        active_power_mw=as_float(row.get("active_power_mw")),
        power_basis=str(row.get("power_basis") or ("active_power_mw" if row.get("active_power_mw") is not None else "unavailable")),
        energy_mj_per_token=as_float(row.get("energy_mj_per_token")),
        energy_mj_per_request=as_float(row.get("energy_mj_per_request")),
        energy_source=energy_source,
        energy_complete=as_bool(row.get("energy_complete"), False),
        support_status=str(row.get("support_status") or "ok").strip().lower(),
        fallback_used=as_bool(row.get("fallback_used"), False),
        stable=as_bool(row.get("stable"), True),
        thermal_safe=as_bool(row.get("thermal_safe"), True),
        data_quality=data_quality,
        npu_workpoint=str(row.get("npu_workpoint") or ""),
        graph_id=str(row.get("graph_id") or ""),
        source_file=str(row.get("source_file") or ""),
        raw=dict(row),
    )


def _transition_from_mapping(row: Dict[str, Any]) -> TransitionEdge:
    source = str(row.get("transition_energy_source") or "unavailable")
    energy = as_float(row.get("transition_energy_mj"))
    return TransitionEdge(
        from_state_id=str(row.get("from_state_id") or row.get("from_state") or ""),
        to_state_id=str(row.get("to_state_id") or row.get("to_state") or ""),
        context_len=as_int(row.get("context_len")),
        total_blocking_us=as_float(row.get("total_blocking_us")),
        first_token_gap_us=as_float(row.get("first_token_gap_us")),
        post_switch_tbt_us=as_float(row.get("post_switch_tbt_us")),
        transition_energy_mj=energy,
        transition_energy_source=source,
        transition_energy_complete=as_bool(row.get("transition_energy_complete"), energy is not None and source != "unavailable"),
        success_rate=as_float(row.get("success_rate")),
        fallback_count=as_int(row.get("fallback_count"), 0) or 0,
        support_status=str(row.get("support_status") or "ok").strip().lower(),
        raw=dict(row),
    )


def _load_graph_entries(summary: Any) -> List[GraphEntry]:
    if not isinstance(summary, dict):
        return []
    rows = summary.get("graphs") or []
    if not isinstance(rows, list):
        return []
    graphs: List[GraphEntry] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        graph_id = str(row.get("graph_id") or "").strip()
        phase = str(row.get("phase") or "").strip().lower()
        if not graph_id or phase not in {"prefill", "decode"}:
            continue
        graphs.append(
            GraphEntry(
                graph_id=graph_id,
                phase=phase,
                chunk_size=as_int(row.get("chunk_size")),
                usable_kv_slots=as_int(row.get("usable_kv_slots")),
                safety_margin=as_int(row.get("safety_margin"), 0) or 0,
                supported_workpoints=[str(item) for item in as_list(row.get("supported_workpoints"))],
                profiled_load_us=as_float(row.get("profiled_load_us")),
                profiled_warmup_us=as_float(row.get("profiled_warmup_us")),
                profiled_exec_us=as_float(row.get("profiled_exec_us")),
                profiled_energy_mj=as_float(row.get("profiled_energy_mj")),
                data_quality=[str(item) for item in as_list(row.get("data_quality"))],
                raw=dict(row),
            )
        )
    return graphs
