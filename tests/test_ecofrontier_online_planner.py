from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.ecofrontier.online_planner import OnlinePlanner, PlannerOptions
from tools.ecofrontier.planner_artifact_loader import generate_request_grid, load_artifact, load_requests
from tools.ecofrontier.planner_schema import PlannerRequest
from tools.ecofrontier.planner_trace import write_summary, write_trace


FIXTURES = ROOT / "tests" / "fixtures" / "ecofrontier"
ARTIFACT = FIXTURES / "planner_artifact.json"
REQUESTS = FIXTURES / "planner_requests.json"


def _artifact():
    return load_artifact(ARTIFACT)


def _request(**overrides):
    data = {
        "request_id": "r",
        "prompt_tokens": 512,
        "context_len": 512,
        "predicted_output_mean": 32,
        "predicted_output_hi": 64,
        "slo_ttft_ms": 800.0,
        "slo_tbt_us": 45000.0,
        "current_state_id": "",
        "current_graph_id": "",
        "current_temp_c": 30.0,
    }
    data.update(overrides)
    return PlannerRequest.from_mapping(data)


def test_loads_ecofrontier_frontier_json_fixture():
    artifact = _artifact()

    assert artifact.raw["version"] == "ecofrontier.offline_frontier.v1"
    assert {state.phase for state in artifact.states} == {"prefill", "decode"}
    assert len(artifact.transitions) == 2


def test_loads_request_fixture():
    requests = load_requests(REQUESTS)

    assert [request.request_id for request in requests] == ["fixture-feasible", "fixture-best-effort"]
    assert requests[0].prompt_tokens == 512


def test_missing_energy_is_not_treated_as_zero():
    result = OnlinePlanner(_artifact()).plan(_request(slo_tbt_us=43000.0))

    assert result.energy_complete is False
    assert "decode_energy_mj_per_token" in result.missing_energy_terms
    assert result.estimated_energy_mj is not None
    assert result.estimated_energy_mj > 0.0


def test_energy_complete_false_when_transition_energy_is_unavailable():
    result = OnlinePlanner(_artifact()).plan(
        _request(
            request_id="transition-energy-missing",
            current_state_id="cpu_decode_missing_energy",
            predicted_output_mean=128,
            slo_tbt_us=35000.0,
        )
    )

    assert result.transition_used is True
    assert result.transition_energy_complete is False
    assert result.energy_complete is False
    assert "transition_energy_mj" in result.missing_energy_terms


def test_rejects_decode_state_that_violates_tbt_slo():
    result = OnlinePlanner(_artifact()).plan(_request(slo_tbt_us=45000.0))

    assert "violates_tbt_slo" in result.reject_counts
    assert result.chosen_decode_state != "gpu_decode_slow"


def test_rejects_prefill_state_that_violates_ttft_slo():
    result = OnlinePlanner(_artifact()).plan(_request(slo_ttft_ms=800.0))

    assert "violates_ttft_slo" in result.reject_counts
    assert result.chosen_prefill_state != "gpu_prefill_slow"


def test_returns_fastest_safe_best_effort_when_no_state_meets_slo():
    result = OnlinePlanner(_artifact()).plan(_request(slo_ttft_ms=200.0, slo_tbt_us=20000.0))

    assert result.status == "SLO_INFEASIBLE_BEST_EFFORT"
    assert result.selected_by == "fastest_safe_best_effort"
    assert "no_state_meets_slo" in result.reject_reasons
    assert result.chosen_decode_state == "gpu_decode_fast"


def test_filters_unstable_state_from_default_candidates():
    result = OnlinePlanner(_artifact()).plan(_request(slo_tbt_us=26000.0))

    assert "unstable_state" in result.reject_counts
    assert result.chosen_decode_state != "gpu_decode_unstable"


def test_uses_qnn_graph_capacity_guard():
    result = OnlinePlanner(_artifact()).plan(
        _request(
            request_id="qnn-capacity",
            predicted_output_hi=600,
            slo_tbt_us=37000.0,
        )
    )

    assert result.chosen_decode_state != "npu_decode_burst"
    assert "graph_capacity_unsafe" in result.reject_counts


def test_refuses_to_infer_usable_kv_slots_from_qnn_aot_context_size():
    artifact = _artifact()
    graph = next(entry for entry in artifact.graphs if entry.graph_id == "decode_context_size_only")

    assert graph.usable_kv_slots is None
    assert graph.has_explicit_capacity is False


def test_transition_latency_amortization_rejects_non_profitable_switch():
    result = OnlinePlanner(_artifact()).plan(
        _request(
            request_id="not-amortized",
            current_state_id="cpu_decode_missing_energy",
            predicted_output_mean=32,
            slo_tbt_us=35000.0,
        )
    )

    assert result.chosen_decode_state != "gpu_decode_fast"
    assert "transition_not_amortized" in result.reject_counts


def test_transition_latency_amortization_accepts_profitable_switch():
    result = OnlinePlanner(_artifact()).plan(
        _request(
            request_id="amortized",
            current_state_id="gpu_decode_slow",
            predicted_output_mean=32,
            slo_tbt_us=35000.0,
        )
    )

    assert result.status == "Feasible"
    assert result.chosen_decode_state == "gpu_decode_fast"
    assert result.transition_used is True
    assert result.transition_total_blocking_us == pytest.approx(1000.0)


def test_sparse_transition_graph_missing_edge_does_not_crash_when_allowed():
    result = OnlinePlanner(_artifact(), PlannerOptions(allow_missing_transition=True)).plan(
        _request(
            request_id="missing-transition",
            current_state_id="unknown_current",
            slo_tbt_us=35000.0,
        )
    )

    assert result.status == "Feasible"
    assert "transition_missing" not in result.reject_reasons
    assert result.transition_used is False


def test_trace_jsonl_contains_required_fields(tmp_path: Path):
    result = OnlinePlanner(_artifact()).plan(_request())
    trace = tmp_path / "trace.jsonl"

    write_trace([result], trace)
    events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    plan_event = events[0]

    assert plan_event["event"] == "ecofrontier_plan"
    assert "selected_by" in plan_event
    assert "energy_complete" in plan_event
    assert "missing_energy_terms" in plan_event
    assert "reject_reasons" in plan_event
    assert plan_event["selected_by"] == "estimated_energy_incomplete"
    assert plan_event["prefill_length_match"] == "exact"
    assert plan_event["decode_length_match"] == "exact"
    assert plan_event["transition_type"] == "none"
    assert plan_event["prefill_latency_source"] == "artifact_ttft_ms_p95"
    assert plan_event["decode_latency_source"] == "artifact_tbt_us"
    assert "ttft_complete" in plan_event
    assert "power_basis" in plan_event


def test_replay_summary_is_generated(tmp_path: Path):
    results = [OnlinePlanner(_artifact()).plan(request) for request in load_requests(REQUESTS)]
    summary_path = tmp_path / "summary.json"

    summary = write_summary(results, summary_path)
    parsed = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["request_count"] == 2
    assert parsed["heatmap_rows"]
    assert "chosen_decode_state_counts" in parsed
    assert parsed["length_match_counts"]
    assert parsed["power_basis_counts"]


def test_cli_replay_writes_trace_and_summary(tmp_path: Path):
    trace = tmp_path / "planner_replay_trace.jsonl"
    summary = tmp_path / "planner_replay_summary.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "ecofrontier" / "replay_planner.py"),
            "--artifact",
            str(ARTIFACT),
            "--requests",
            str(REQUESTS),
            "--trace",
            str(trace),
            "--summary",
            str(summary),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert trace.exists()
    assert summary.exists()
    assert json.loads(summary.read_text(encoding="utf-8"))["request_count"] == 2


def test_generate_grid_uses_artifact_decode_buckets():
    requests = generate_request_grid(_artifact())

    assert requests
    assert {request.context_len for request in requests} == {512, 1024}
