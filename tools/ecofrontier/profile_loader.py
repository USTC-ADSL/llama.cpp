from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .markdown_table_loader import load_markdown_profiles
from .profile_schema import (
    CompilerConfig,
    GraphProfile,
    StateProfile,
    TransitionProfile,
    infer_phase_from_shape,
    normalize_backend,
    normalize_phase,
    normalize_quality,
    parse_bool,
    parse_float,
    parse_int,
)


@dataclass
class LoadedProfiles:
    input_dir: Optional[Path] = None
    states: List[StateProfile] = field(default_factory=list)
    transitions: List[TransitionProfile] = field(default_factory=list)
    graphs: List[GraphProfile] = field(default_factory=list)
    rejected_graphs: List[Dict[str, Any]] = field(default_factory=list)
    source_files: List[Dict[str, Any]] = field(default_factory=list)
    skipped_sources: List[Dict[str, Any]] = field(default_factory=list)
    source_slo_frontiers: List[Dict[str, Any]] = field(default_factory=list)
    paper_ready_caveats: List[str] = field(default_factory=list)
    source_data_quality_summary: Dict[str, Any] = field(default_factory=dict)
    energy_policy: Optional[str] = None
    graph_metadata: List[Dict[str, Any]] = field(default_factory=list)

    def add_source(self, path: Path, source_type: str, row_count: int = 0, detail: Optional[Dict[str, Any]] = None) -> None:
        entry: Dict[str, Any] = {
            "path": str(path),
            "type": source_type,
            "row_count": row_count,
        }
        if detail:
            entry.update(detail)
        self.source_files.append(entry)

    def add_skipped(self, path: Path, reason: str) -> None:
        self.skipped_sources.append({"path": str(path), "reason": reason})


def load_input_dir(input_dir: Path | str, config: Optional[CompilerConfig] = None) -> LoadedProfiles:
    config = config or CompilerConfig()
    root = Path(input_dir)
    loaded = LoadedProfiles(input_dir=root)
    files = sorted([path for path in root.rglob("*") if path.is_file()], key=lambda p: str(p))

    insightb_jsons = [
        path
        for path in files
        if path.suffix.lower() == ".json" and path.name.startswith("InsightB_ChatGPT_结构化数据")
    ]
    parsed_insightb = False
    for path in insightb_jsons:
        try:
            insight = load_insightb_json(path, config)
        except Exception as exc:  # pragma: no cover - exercised by malformed external data.
            loaded.add_skipped(path, f"insightb_json_parse_error: {exc}")
            continue
        _merge_loaded(loaded, insight)
        parsed_insightb = True

    for path in files:
        if path.name in {"CPU测试结果.md", "GPU测试结果.md", "NPU测试结果.md"}:
            try:
                states = load_markdown_profiles(path, config)
            except Exception as exc:  # pragma: no cover - malformed external markdown.
                loaded.add_skipped(path, f"markdown_parse_error: {exc}")
                continue
            loaded.states.extend(states)
            loaded.add_source(path, "markdown_profile", len(states), {"backend_report": path.name})
        elif _is_graph_profile(path):
            try:
                graphs, rejected = load_graph_profiles_from_json(path)
            except Exception as exc:
                loaded.add_skipped(path, f"graph_profile_parse_error: {exc}")
                continue
            loaded.graphs.extend(graphs)
            loaded.rejected_graphs.extend(rejected)
            loaded.add_source(path, "qnn_graph_profile", len(graphs), {"rejected_entries": len(rejected)})
        elif path.name == "qnn_aot_config.json":
            metadata = _load_qnn_aot_metadata(path)
            if metadata:
                loaded.graph_metadata.append(metadata)
                loaded.add_source(path, "qnn_aot_config_metadata", 1)

    if not parsed_insightb:
        for path in files:
            lower = path.name.lower()
            if path.suffix.lower() == ".csv" and ("context_decode_profile" in lower or lower == "states.csv"):
                try:
                    states = load_state_profiles_from_csv(path, config)
                except Exception as exc:
                    loaded.add_skipped(path, f"state_csv_parse_error: {exc}")
                    continue
                loaded.states.extend(states)
                loaded.add_source(path, "state_csv", len(states))
            elif path.suffix.lower() == ".csv" and ("transition_cost" in lower or lower == "transitions.csv"):
                try:
                    transitions = load_transition_profiles_from_csv(path)
                except Exception as exc:
                    loaded.add_skipped(path, f"transition_csv_parse_error: {exc}")
                    continue
                loaded.transitions.extend(transitions)
                loaded.add_source(path, "transition_csv", len(transitions))

    return loaded


def load_state_profiles_from_csv(path: Path | str, config: Optional[CompilerConfig] = None) -> List[StateProfile]:
    config = config or CompilerConfig()
    rows = _read_csv(Path(path))
    states = [_state_from_row(row, Path(path), config) for row in rows]
    return [state.normalized(config) for state in states]


def load_transition_profiles_from_csv(path: Path | str) -> List[TransitionProfile]:
    rows = _read_csv(Path(path))
    return [_transition_from_row(row, Path(path)).normalized() for row in rows]


def load_graph_profiles_from_json(path: Path | str) -> Tuple[List[GraphProfile], List[Dict[str, Any]]]:
    json_path = Path(path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    entries: Iterable[Dict[str, Any]]
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        entries = data.get("graphs") or data.get("graph_profiles") or []
    else:
        entries = []

    graphs: List[GraphProfile] = []
    rejected: List[Dict[str, Any]] = []
    for entry in entries:
        graph_id = str(entry.get("graph_id") or entry.get("id") or "")
        if "usable_kv_slots" not in entry and "qnn_usable_kv_slots" not in entry:
            rejected.append(
                {
                    "source_file": str(json_path),
                    "graph_id": graph_id,
                    "reason": "missing_usable_kv_slots",
                    "metadata": {
                        key: entry[key]
                        for key in ("qnn_aot_context_size", "qnn_aot_cache_size", "qnn_cache_size")
                        if key in entry
                    },
                }
            )
            continue
        usable_kv_slots = parse_int(entry.get("usable_kv_slots", entry.get("qnn_usable_kv_slots")))
        if usable_kv_slots is None:
            rejected.append({"source_file": str(json_path), "graph_id": graph_id, "reason": "invalid_usable_kv_slots"})
            continue
        graphs.append(
            GraphProfile(
                graph_id=graph_id,
                phase=entry.get("phase", "decode"),
                source_file=str(json_path),
                chunk_size=parse_int(entry.get("chunk_size")),
                usable_kv_slots=usable_kv_slots,
                safety_margin=parse_int(entry.get("safety_margin")),
                supported_workpoints=list(entry.get("supported_workpoints") or []),
                profiled_load_us=parse_float(entry.get("profiled_load_us")),
                profiled_warmup_us=parse_float(entry.get("profiled_warmup_us")),
                profiled_exec_us=parse_float(entry.get("profiled_exec_us")),
                profiled_energy_mj=parse_float(entry.get("profiled_energy_mj")),
                memory_bytes=parse_int(entry.get("memory_bytes")),
                data_quality=normalize_quality(entry.get("data_quality")),
                metadata={
                    key: entry[key]
                    for key in ("qnn_aot_context_size", "qnn_aot_cache_size", "qnn_cache_size")
                    if key in entry
                },
            ).normalized()
        )
    return graphs, rejected


def load_insightb_json(path: Path | str, config: Optional[CompilerConfig] = None) -> LoadedProfiles:
    config = config or CompilerConfig()
    json_path = Path(path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    loaded = LoadedProfiles(input_dir=json_path.parent)
    tables = data.get("tables", {})
    metadata = data.get("metadata", {})
    loaded.paper_ready_caveats = list(data.get("paper_ready_caveats") or [])
    loaded.source_data_quality_summary = dict(data.get("data_quality_summary") or {})
    loaded.energy_policy = metadata.get("energy_policy")

    context_rows = list(tables.get("context_decode_profile") or [])
    for row in context_rows:
        row = dict(row)
        if loaded.energy_policy and "do not claim energy" in loaded.energy_policy.lower():
            if row.get("energy_mj_per_token") not in {None, ""}:
                row["energy_source"] = "insightb_power_latency_no_energy_claim"
        loaded.states.append(_state_from_row(row, json_path, config, default_phase="decode").normalized(config))

    slo_rows = [dict(row) for row in tables.get("slo_frontier") or []]
    loaded.source_slo_frontiers.extend(slo_rows)

    transition_rows = list(tables.get("transition_cost") or [])
    loaded.transitions.extend([_transition_from_row(row, json_path).normalized() for row in transition_rows])

    loaded.graph_metadata.append(
        {
            key: metadata[key]
            for key in ("qnn_aot_cache_size", "qnn_aot_context_size", "qnn_cache_safety_margin")
            if key in metadata
        }
    )
    loaded.add_source(
        json_path,
        "insightb_structured_json",
        len(context_rows) + len(transition_rows) + len(slo_rows),
        {
            "context_rows": len(context_rows),
            "transition_rows": len(transition_rows),
            "slo_frontier_rows": len(slo_rows),
        },
    )
    return loaded


def _state_from_row(
    row: Dict[str, Any],
    path: Path,
    config: CompilerConfig,
    default_phase: Optional[str] = None,
) -> StateProfile:
    backend = _get(row, "backend", "device")
    state_id = _get(row, "state_id", "state", "id")
    test_shape = _get(row, "test_shape", "shape", "workload")
    phase = _get(row, "phase") or default_phase or infer_phase_from_shape(test_shape)

    if not backend:
        raise ValueError(f"{path}: missing required backend")
    if not phase:
        raise ValueError(f"{path}: missing required phase")

    backend_norm = normalize_backend(backend)
    cpu_freq = parse_int(_get(row, "cpu_freq_khz", "requested_cpu_freq_khz", "请求频率 kHz"))
    actual_cpu = parse_int(_get(row, "actual_cpu_freq_khz", "avg_cpu_freq_khz", "平均 CPU 频率 kHz"))
    gpu_freq = parse_int(_get(row, "gpu_freq_mhz", "set_gpu_freq_mhz", "gpu_frequency_mhz"))
    actual_gpu = parse_int(_get(row, "actual_gpu_freq_mhz", "avg_gpu_freq_mhz"))
    npu_workpoint = _get(row, "npu_workpoint", "workpoint")

    if not state_id:
        state_id = _make_state_id(backend_norm, row, cpu_freq, gpu_freq, npu_workpoint)
    if not state_id:
        raise ValueError(f"{path}: missing required state_id")

    metadata = {
        key: row[key]
        for key in (
            "date",
            "model",
            "raw_log_path",
            "sample_path",
            "qnn_aot_cache_size",
            "qnn_aot_context_size",
            "qnn_cache_size",
            "qnn_cache_limit",
            "remarks",
            "notes",
        )
        if key in row and row[key] not in {None, ""}
    }
    if _get(row, "qnn_usable_kv_slots", "usable_kv_slots") not in {None, ""}:
        metadata["usable_kv_slots"] = parse_int(_get(row, "qnn_usable_kv_slots", "usable_kv_slots"))

    state = StateProfile(
        state_id=str(state_id),
        backend=backend_norm,
        phase=normalize_phase(phase),
        source_file=str(path),
        test_shape=str(test_shape) if test_shape else None,
        prompt_tokens=parse_int(_get(row, "prompt_tokens", "prompt_len", "effective_prefill_tokens")),
        decode_tokens=parse_int(_get(row, "decode_tokens", "output_tokens")),
        rounds=parse_int(_get(row, "rounds", "r")),
        context_len=parse_int(_get(row, "context_len", "context_tokens")),
        cpu_affinity=_get(row, "cpu_affinity", "cpu_case"),
        cpu_freq_khz=cpu_freq,
        actual_cpu_freq_khz=actual_cpu,
        cpu_threads=parse_int(_get(row, "cpu_threads", "threads")),
        gpu_freq_mhz=gpu_freq,
        actual_gpu_freq_mhz=actual_gpu,
        npu_workpoint=npu_workpoint,
        graph_id=_get(row, "graph_id"),
        throughput_tps=parse_float(_get(row, "throughput_tps", "throughput_tok_s", "吞吐 tok/s")),
        tbt_us=parse_float(_get(row, "tbt_us", "tbt_us_p95", "tbt_us_p50")),
        ttft_ms_p50=parse_float(_get(row, "ttft_ms_p50")),
        ttft_ms_p95=parse_float(_get(row, "ttft_ms_p95", "prefill_latency_ms", "latency_ms")),
        active_power_mw=parse_float(_get(row, "active_power_mw", "steady_power_mw", "power_mw")),
        baseline_power_mw=parse_float(_get(row, "baseline_power_mw")),
        power_delta_mw=parse_float(_get(row, "power_delta_mw", "delta_power_mw")),
        energy_mj_per_token=parse_float(_get(row, "energy_mj_per_token")),
        energy_mj_per_request=parse_float(_get(row, "energy_mj_per_request")),
        temperature_avg_c=parse_float(_get(row, "temperature_avg_c", "temp_avg_c")),
        temperature_max_c=parse_float(_get(row, "temperature_max_c", "temp_max_c")),
        stable_range_pct=parse_float(_get(row, "stable_range_pct", "stable_window_range_pct")),
        power_cv_pct=parse_float(_get(row, "power_cv_pct")),
        support_status=str(_get(row, "support_status") or "ok"),
        fallback_used=parse_bool(_get(row, "fallback_used"), False),
        data_quality=normalize_quality(_get(row, "data_quality")),
        energy_source=_get(row, "energy_source"),
        metadata=metadata,
    )
    return state


def _transition_from_row(row: Dict[str, Any], path: Path) -> TransitionProfile:
    from_state = _get(row, "from_state_id", "from_state")
    to_state = _get(row, "to_state_id", "to_state")
    if not from_state or not to_state:
        raise ValueError(f"{path}: missing required transition endpoints")
    metadata = {
        key: row[key]
        for key in (
            "date",
            "model",
            "effective_prefill_tokens",
            "decode_tokens_before_switch",
            "decode_tokens_after_switch",
            "rounds",
            "route_apply_us",
            "policy_apply_us",
            "qnn_workpoint_apply_us",
            "gpu_freq_apply_us",
            "sched_reserve_us",
            "decode_entry_us",
            "qnn_aot_cache_size",
            "qnn_aot_context_size",
            "raw_log_path",
        )
        if key in row and row[key] not in {None, ""}
    }
    return TransitionProfile(
        from_state_id=str(from_state),
        to_state_id=str(to_state),
        source_file=str(path),
        context_len=parse_int(_get(row, "context_len")),
        total_blocking_us=parse_float(_get(row, "total_blocking_us")),
        first_token_gap_us=parse_float(_get(row, "first_token_gap_us")),
        post_switch_tbt_us=parse_float(_get(row, "post_switch_tbt_us")),
        transition_energy_mj=parse_float(_get(row, "transition_energy_mj")),
        transition_energy_source=str(_get(row, "transition_energy_source") or "unavailable"),
        success_rate=parse_float(_get(row, "success_rate", "switch_success_rate")),
        fallback_count=parse_int(_get(row, "fallback_count")),
        support_status=str(_get(row, "support_status") or "ok"),
        kv_handoff_us=parse_float(_get(row, "kv_handoff_us")),
        graph_rebuild_us=parse_float(_get(row, "graph_rebuild_us")),
        decision_us=parse_float(_get(row, "decision_us")),
        metadata=metadata,
    )


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _get(row: Dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            return row[name]
    lowered = {str(key).lower(): key for key in row}
    for name in names:
        key = lowered.get(name.lower())
        if key is not None:
            return row[key]
    return None


def _make_state_id(
    backend: str,
    row: Dict[str, Any],
    cpu_freq_khz: Optional[int],
    gpu_freq_mhz: Optional[int],
    npu_workpoint: Optional[str],
) -> Optional[str]:
    if backend == "CPU":
        affinity = _get(row, "cpu_affinity", "cpu_case") or "cpu"
        if cpu_freq_khz is not None:
            return f"cpu_{affinity}_{cpu_freq_khz // 1000}"
    if backend == "GPU" and gpu_freq_mhz is not None:
        return f"gpu_{gpu_freq_mhz}"
    if backend == "QNN_NPU" and npu_workpoint:
        return f"npu_{npu_workpoint}"
    return None


def _is_graph_profile(path: Path) -> bool:
    lower = path.name.lower()
    if path.suffix.lower() != ".json":
        return False
    return lower in {"qnn_graphs.json", "qnn_graph_manifest.json"} or "graph_profile" in lower or "graph_manifest" in lower


def _load_qnn_aot_metadata(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    result: Dict[str, Any] = {"source_file": str(path)}
    for key in ("qnn_aot_cache_size", "qnn_aot_context_size", "qnn_cache_safety_margin"):
        if key in data:
            result[key] = data[key]
    for section in ("context", "aot", "metadata"):
        if isinstance(data.get(section), dict):
            for key in ("qnn_aot_cache_size", "qnn_aot_context_size", "qnn_cache_safety_margin"):
                if key in data[section]:
                    result[key] = data[section][key]
    return result


def _merge_loaded(target: LoadedProfiles, source: LoadedProfiles) -> None:
    target.states.extend(source.states)
    target.transitions.extend(source.transitions)
    target.graphs.extend(source.graphs)
    target.rejected_graphs.extend(source.rejected_graphs)
    target.source_files.extend(source.source_files)
    target.skipped_sources.extend(source.skipped_sources)
    target.source_slo_frontiers.extend(source.source_slo_frontiers)
    target.paper_ready_caveats.extend(source.paper_ready_caveats)
    target.graph_metadata.extend(source.graph_metadata)
    if source.source_data_quality_summary:
        target.source_data_quality_summary.update(source.source_data_quality_summary)
    if source.energy_policy:
        target.energy_policy = source.energy_policy
