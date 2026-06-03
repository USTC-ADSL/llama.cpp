from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.ecofrontier.artifact_writer import write_artifacts
from tools.ecofrontier.check_artifact import run_sanity_checks
from tools.ecofrontier.frontier_compiler import compile_frontier
from tools.ecofrontier.markdown_table_loader import load_markdown_profiles
from tools.ecofrontier.profile_loader import (
    load_graph_profiles_from_json,
    load_input_dir,
    load_state_profiles_from_csv,
    load_transition_profiles_from_csv,
)
from tools.ecofrontier.profile_schema import CompilerConfig, StateProfile
from tools.ecofrontier.split_artifact import split_artifact


FIXTURES = ROOT / "tests" / "fixtures" / "ecofrontier"


def _compile_fixture(config: CompilerConfig | None = None):
    loaded = load_input_dir(FIXTURES, config or CompilerConfig())
    return loaded, compile_frontier(loaded, config or CompilerConfig())


def test_loads_state_profile_fixture():
    states = load_state_profiles_from_csv(FIXTURES / "states.csv", CompilerConfig())
    assert len(states) >= 10
    assert states[0].state_id == "cpu_big2_2400"
    assert states[0].backend == "CPU"
    assert states[0].phase == "decode"


def test_reports_missing_required_state_fields(tmp_path: Path):
    bad = tmp_path / "bad_states.csv"
    bad.write_text("state_id,phase,throughput_tps\nmissing_backend,decode,10\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required backend"):
        load_state_profiles_from_csv(bad, CompilerConfig())


def test_parses_markdown_table_fixture():
    config = CompilerConfig()
    cpu_states = load_markdown_profiles(FIXTURES / "CPU测试结果.md", config)
    gpu_states = load_markdown_profiles(FIXTURES / "GPU测试结果.md", config)
    npu_states = load_markdown_profiles(FIXTURES / "NPU测试结果.md", config)
    assert {s.backend for s in cpu_states + gpu_states + npu_states} == {"CPU", "GPU", "QNN_NPU"}
    assert any(s.phase == "decode" and s.cpu_freq_khz == 2400000 for s in cpu_states)
    assert any(s.phase == "prefill" and s.gpu_freq_mhz == 734 for s in gpu_states)
    assert any(s.phase == "decode" and s.npu_workpoint == "burst" for s in npu_states)


def test_handles_utf8_path_names(tmp_path: Path):
    utf8_dir = tmp_path / "实验结果"
    utf8_dir.mkdir()
    (utf8_dir / "states.csv").write_text((FIXTURES / "states.csv").read_text(encoding="utf-8"), encoding="utf-8")
    loaded = load_input_dir(utf8_dir, CompilerConfig())
    assert loaded.states
    assert any("实验结果" in source["path"] for source in loaded.source_files)


def test_missing_optional_energy_is_not_converted_to_zero():
    states = load_state_profiles_from_csv(FIXTURES / "states.csv", CompilerConfig())
    missing = next(s for s in states if s.state_id == "missing_energy_no_power")
    assert missing.energy_mj_per_token is None
    assert "energy_mj_per_token" not in missing.to_artifact_dict()


def test_derives_tbt_from_throughput_correctly():
    states = load_state_profiles_from_csv(FIXTURES / "states.csv", CompilerConfig())
    state = next(s for s in states if s.state_id == "cpu_big2_2400")
    assert state.tbt_us == pytest.approx(50000.0)
    assert state.tbt_source == "derived_from_throughput"


def test_derives_estimated_energy_from_power_and_tbt_with_incomplete_flag():
    states = load_state_profiles_from_csv(FIXTURES / "states.csv", CompilerConfig())
    state = next(s for s in states if s.state_id == "cpu_big2_2400")
    assert state.energy_mj_per_token == pytest.approx(100.0)
    assert state.energy_source == "estimated_power_latency"
    assert state.energy_complete is False


def test_marks_cpu_throttled_state_unstable_on_frequency_mismatch():
    states = load_state_profiles_from_csv(FIXTURES / "states.csv", CompilerConfig())
    throttled = next(s for s in states if s.state_id == "cpu_big2_3200")
    assert throttled.stable is False
    assert "frequency_mismatch" in throttled.data_quality


def test_marks_unstable_power_window_rows_low_quality():
    states = load_state_profiles_from_csv(FIXTURES / "states.csv", CompilerConfig())
    unstable = next(s for s in states if s.state_id == "gpu_967_unstable")
    assert unstable.stable is False
    assert "unstable_power_window" in unstable.data_quality
    assert "power_low_confidence" in unstable.data_quality


def test_filters_unstable_states_from_default_frontier():
    config = CompilerConfig(slo_tbt_us_values=[50000.0], filter_unstable=True)
    _, artifact = _compile_fixture(config)
    decode = [
        entry
        for entry in artifact["frontiers"]
        if entry["phase"] == "decode" and entry["length_value"] == 512 and entry["slo_value"] == 50000.0
    ][0]
    candidate_ids = {candidate["state_id"] for candidate in decode["feasible_candidates"]}
    assert "gpu_967_unstable" not in candidate_ids


def test_builds_decode_slo_feasible_frontier():
    config = CompilerConfig(slo_tbt_us_values=[45000.0], filter_unstable=True)
    _, artifact = _compile_fixture(config)
    decode = [
        entry
        for entry in artifact["frontiers"]
        if entry["phase"] == "decode" and entry["length_value"] == 512 and entry["slo_value"] == 45000.0
    ][0]
    candidate_ids = {candidate["state_id"] for candidate in decode["feasible_candidates"]}
    assert "npu_burst" in candidate_ids
    assert all(candidate["latency_us"] <= 45000.0 for candidate in decode["feasible_candidates"])


def test_builds_prefill_slo_feasible_frontier():
    config = CompilerConfig(slo_ttft_ms_values=[600.0], filter_unstable=True)
    _, artifact = _compile_fixture(config)
    prefill = [
        entry
        for entry in artifact["frontiers"]
        if entry["phase"] == "prefill" and entry["length_value"] == 512 and entry["slo_value"] == 600.0
    ][0]
    candidate_ids = {candidate["state_id"] for candidate in prefill["feasible_candidates"]}
    assert "npu_burst_prefill" in candidate_ids
    assert "gpu_734_prefill" not in candidate_ids


def test_pareto_prunes_dominated_states_when_measured_energy_is_complete():
    config = CompilerConfig(slo_tbt_us_values=[70000.0], filter_unstable=True)
    _, artifact = _compile_fixture(config)
    dominated = {
        item["state_id"]: item
        for item in artifact["dominated_states"]
        if item["length_value"] == 1024 and item["phase"] == "decode"
    }
    assert "measured_slow_high_energy" in dominated
    assert dominated["measured_slow_high_energy"]["dominance_reason"] == "measured_energy"
    frontier = [
        entry
        for entry in artifact["frontiers"]
        if entry["phase"] == "decode" and entry["length_value"] == 1024 and entry["slo_value"] == 70000.0
    ][0]
    assert frontier["frontier_kind"] == "measured_energy_frontier"


def test_does_not_claim_measured_energy_dominance_when_energy_is_incomplete():
    config = CompilerConfig(slo_tbt_us_values=[70000.0], filter_unstable=True)
    states = [
        StateProfile(
            state_id="estimated_a",
            backend="CPU",
            phase="decode",
            source_file="synthetic",
            context_len=128,
            tbt_us=40000.0,
            active_power_mw=1000.0,
        ),
        StateProfile(
            state_id="estimated_b",
            backend="GPU",
            phase="decode",
            source_file="synthetic",
            context_len=128,
            tbt_us=50000.0,
            active_power_mw=2000.0,
        ),
    ]
    loaded = load_input_dir(FIXTURES, config)
    loaded.states = [s.normalized(config) for s in states]
    artifact = compile_frontier(loaded, config)
    assert artifact["frontiers"][0]["frontier_kind"] in {"estimated_energy_frontier", "latency_power_frontier"}
    assert all(item["dominance_reason"] != "measured_energy" for item in artifact["dominated_states"])


def test_loads_transition_profile_and_preserves_transition_energy_unavailable():
    transitions = load_transition_profiles_from_csv(FIXTURES / "transitions.csv")
    assert len(transitions) == 1
    edge = transitions[0]
    assert edge.transition_energy_mj is None
    assert edge.transition_energy_complete is False
    assert edge.transition_energy_source == "unavailable"


def test_stores_sparse_transition_edges():
    _, artifact = _compile_fixture(CompilerConfig())
    assert len(artifact["transition_edges"]) == 1
    edge = artifact["transition_edges"][0]
    assert edge["from_state_id"] == "npu_burst"
    assert edge["to_state_id"] == "gpu_734"


def test_refuses_to_infer_usable_kv_slots_from_qnn_aot_context_size():
    graphs, rejected = load_graph_profiles_from_json(FIXTURES / "qnn_graphs.json")
    assert [graph.graph_id for graph in graphs] == ["decode_ctx2048"]
    assert rejected[0]["graph_id"] == "bad_context_only"
    assert rejected[0]["reason"] == "missing_usable_kv_slots"


def test_emits_artifact_json_with_caveats(tmp_path: Path):
    loaded, artifact = _compile_fixture(CompilerConfig())
    output = tmp_path / "frontier.json"
    summary = tmp_path / "summary.json"
    write_artifacts(artifact, output, summary)
    parsed = json.loads(output.read_text(encoding="utf-8"))
    assert parsed["version"] == "ecofrontier.offline_frontier.v1"
    assert "transition_energy_unavailable" in parsed["caveats"]
    assert "energy_estimated_from_power_latency" in parsed["caveats"]
    assert summary.exists()


def test_artifact_includes_data_quality_summary():
    _, artifact = _compile_fixture(CompilerConfig())
    summary = artifact["data_quality_summary"]
    assert summary["by_quality"]["paper_ready"] > 0
    assert summary["unstable_state_count"] >= 1


def test_compiler_can_run_against_docs_experiment_results(tmp_path: Path):
    input_dir = ROOT / "docs" / "实验结果"
    if not input_dir.exists():
        pytest.skip("docs/实验结果 is not present in this checkout")
    output = tmp_path / "ecofrontier_frontier.json"
    summary = tmp_path / "ecofrontier_frontier_summary.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "ecofrontier" / "compile_frontier.py"),
            "--input",
            str(input_dir),
            "--output",
            str(output),
            "--summary",
            str(summary),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["normalized_states"]
    assert artifact["frontiers"]
    assert any("InsightB_ChatGPT_结构化数据" in source["path"] for source in artifact["source_files"])


def test_artifact_sanity_checker_passes_compiled_fixture(tmp_path: Path):
    _, artifact = _compile_fixture(CompilerConfig())
    output = tmp_path / "frontier.json"
    summary = tmp_path / "summary.json"
    write_artifacts(artifact, output, summary)

    report = run_sanity_checks(output, summary)

    assert report["ok"] is True
    assert report["frontier_kind_counts"]
    assert report["estimated_energy_marked_complete"]["count"] == 0
    assert report["qnn_context_with_usable_kv_slots"]["count"] == 0
    assert report["forged_transition_energy"]["count"] == 0
    assert report["state_identity_preservation"]["normalized_state_rows"] == len(artifact["normalized_states"])
    assert report["state_identity_preservation"]["multi_tuple_state_id_count"] > 0


def test_artifact_sanity_checker_flags_energy_qnn_and_transition_violations(tmp_path: Path):
    artifact = {
        "normalized_states": [
            {
                "state_id": "bad_energy",
                "phase": "decode",
                "context_len": 128,
                "source_file": "synthetic",
                "energy_source": "estimated_power_latency",
                "energy_complete": True,
            },
            {
                "state_id": "bad_qnn_capacity",
                "phase": "decode",
                "context_len": 128,
                "source_file": "synthetic",
                "qnn_aot_context_size": 4096,
                "usable_kv_slots": 4096,
            },
        ],
        "transition_edges": [
            {
                "from_state_id": "a",
                "to_state_id": "b",
                "transition_energy_source": "unavailable",
                "transition_energy_mj": 1.0,
            }
        ],
        "frontiers": [{"frontier_kind": "latency_only_frontier"}],
    }
    summary = {"frontier_kind_counts": {"latency_only_frontier": 1}}
    output = tmp_path / "bad_frontier.json"
    summary_path = tmp_path / "bad_summary.json"
    output.write_text(json.dumps(artifact), encoding="utf-8")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    report = run_sanity_checks(output, summary_path)

    assert report["ok"] is False
    assert report["estimated_energy_marked_complete"]["count"] == 1
    assert report["qnn_context_with_usable_kv_slots"]["count"] == 1
    assert report["forged_transition_energy"]["count"] == 1


def test_artifact_sanity_checker_cli_outputs_json(tmp_path: Path):
    _, artifact = _compile_fixture(CompilerConfig())
    output = tmp_path / "frontier.json"
    summary = tmp_path / "summary.json"
    write_artifacts(artifact, output, summary)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "ecofrontier" / "check_artifact.py"),
            "--artifact",
            str(output),
            "--summary",
            str(summary),
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    parsed = json.loads(result.stdout)
    assert parsed["ok"] is True
    assert parsed["frontier_kind_counts"]


def test_split_artifact_writes_review_parts(tmp_path: Path):
    _, artifact = _compile_fixture(CompilerConfig())
    output = tmp_path / "frontier.json"
    summary = tmp_path / "summary.json"
    split_dir = tmp_path / "review_parts"
    write_artifacts(artifact, output, summary)

    manifest = split_artifact(output, split_dir, summary)

    assert (split_dir / "00_manifest.json").exists()
    assert (split_dir / "01_sources_and_config.json").exists()
    assert (split_dir / "02_quality_caveats_and_policy.json").exists()
    assert (split_dir / "states" / "QNN_NPU_decode.json").exists()
    assert (split_dir / "states" / "GPU_prefill.json").exists()
    assert (split_dir / "frontiers" / "decode.json").exists()
    assert (split_dir / "frontiers" / "prefill.json").exists()
    assert (split_dir / "07_transition_edges.json").exists()
    assert manifest["counts"]["normalized_states"] == len(artifact["normalized_states"])
    state_counts = sum(item["count"] for item in manifest["parts"] if item["category"] == "states")
    assert state_counts == len(artifact["normalized_states"])


def test_split_artifact_cli_writes_review_parts(tmp_path: Path):
    _, artifact = _compile_fixture(CompilerConfig())
    output = tmp_path / "frontier.json"
    summary = tmp_path / "summary.json"
    split_dir = tmp_path / "review_parts"
    write_artifacts(artifact, output, summary)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "ecofrontier" / "split_artifact.py"),
            "--artifact",
            str(output),
            "--summary",
            str(summary),
            "--output-dir",
            str(split_dir),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    parsed = json.loads(result.stdout)
    assert parsed["output_dir"] == str(split_dir)
    assert (split_dir / "00_manifest.json").exists()
