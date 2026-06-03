from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .planner_schema import PlanResult


def write_trace(results: Iterable[PlanResult], path: Path | str) -> None:
    trace_path = Path(path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for result in results:
        lines.append(json.dumps(result.to_trace_event(), ensure_ascii=False, sort_keys=True))
        if result.reject_counts:
            lines.append(
                json.dumps(
                    {
                        "event": "ecofrontier_reject_summary",
                        "request_id": result.request.request_id,
                        "reject_counts": result.reject_counts,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
    trace_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build_replay_summary(results: Iterable[PlanResult]) -> Dict[str, Any]:
    items = list(results)
    chosen_decode = Counter(result.chosen_decode_state or "" for result in items)
    chosen_prefill = Counter(result.chosen_prefill_state or "" for result in items)
    selected_by = Counter(result.selected_by for result in items)
    length_match = Counter()
    latency_source = Counter()
    transition_type = Counter(result.transition_type for result in items)
    power_basis = Counter(result.power_basis for result in items)
    artifact_caveats = sorted({caveat for result in items for caveat in result.artifact_caveats})
    heatmap_rows = [
        {
            "context_len": result.request.context_len,
            "slo_tbt_us": result.request.slo_tbt_us,
            "predicted_output_hi": result.request.predicted_output_hi,
            "chosen_decode_state": result.chosen_decode_state,
            "feasible_plan_count": result.feasible_plan_count,
            "selected_by": result.selected_by,
            "status": result.status,
            "prefill_length_match": result.prefill_length_match,
            "decode_length_match": result.decode_length_match,
        }
        for result in items
    ]
    reject_counts = Counter()
    for result in items:
        reject_counts.update(result.reject_counts)
        length_match[f"prefill:{result.prefill_length_match}"] += 1
        length_match[f"decode:{result.decode_length_match}"] += 1
        latency_source[f"prefill:{result.prefill_latency_source}"] += 1
        latency_source[f"decode:{result.decode_latency_source}"] += 1
    request_grid_summary = {
        "context_len_values": sorted({result.request.context_len for result in items}),
        "slo_tbt_us_values": sorted({result.request.slo_tbt_us for result in items}),
        "predicted_output_hi_values": sorted({result.request.predicted_output_hi for result in items}),
        "prompt_tokens_values": sorted({result.request.prompt_tokens for result in items}),
        "current_state_id_values": sorted({result.request.current_state_id for result in items}),
    }
    return {
        "request_count": len(items),
        "feasible_count": sum(1 for result in items if result.status == "Feasible"),
        "best_effort_count": sum(1 for result in items if result.status == "SLO_INFEASIBLE_BEST_EFFORT"),
        "chosen_decode_state_counts": dict(chosen_decode),
        "chosen_prefill_state_counts": dict(chosen_prefill),
        "selected_by_counts": dict(selected_by),
        "length_match_counts": dict(length_match),
        "latency_source_counts": dict(latency_source),
        "transition_type_counts": dict(transition_type),
        "power_basis_counts": dict(power_basis),
        "ttft_complete_count": sum(1 for result in items if result.ttft_complete),
        "ttft_incomplete_count": sum(1 for result in items if not result.ttft_complete),
        "energy_complete_count": sum(1 for result in items if result.energy_complete),
        "energy_incomplete_count": sum(1 for result in items if not result.energy_complete),
        "no_state_meets_slo_count": reject_counts.get("no_state_meets_slo", 0),
        "transition_used_count": sum(1 for result in items if result.transition_used),
        "transition_not_amortized_count": reject_counts.get("transition_not_amortized", 0),
        "graph_capacity_reject_count": reject_counts.get("graph_capacity_unsafe", 0),
        "artifact_caveats_seen": artifact_caveats,
        "request_grid_summary": request_grid_summary,
        "heatmap_rows": heatmap_rows,
    }


def write_summary(results: Iterable[PlanResult], path: Path | str) -> Dict[str, Any]:
    summary = build_replay_summary(results)
    summary_path = Path(path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
