from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RAW_FIELDS = [
    "phase",
    "state_name",
    "length",
    "run_id",
    "n_tokens",
    "elapsed_ms",
    "energy_mj",
    "status",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _row(rows: list[dict[str, str]], **matches: object) -> dict[str, str]:
    for row in rows:
        if all(row.get(key) == str(value) for key, value in matches.items()):
            return row
    raise AssertionError(f"missing row matching {matches!r}")


def _write_test_config(path: Path, output_dir: Path) -> None:
    path.write_text(
        f"""
model_path: /data/local/tmp/model.gguf
tokenizer_path: /data/local/tmp/tokenizer.model
output_dir: {output_dir}
idle_power_mw: 100.0
repeat: 2
decode_probe_tokens: 64
context_points: [512, 1024, 1536, 2048, 3072, 4096, 5120, 6144]
buckets:
  - [0, 512]
  - [512, 1024]
  - [1024, 1536]
  - [1536, 2048]
  - [2048, 3072]
  - [3072, 4096]
  - [4096, 5120]
  - [5120, 6144]
npu_workpoints: [low_balanced, balanced, burst]
qnn_graphs:
  - capacity: 2048
    tier: normal
  - capacity: 4096
    tier: large
  - capacity: 6144
    tier: xlarge
cpu_affinity_classes:
  - name: B1
  - name: B2
  - name: S4
  - name: B2+S2
  - name: allcore
cpu_frequencies: [1804, 2208, 2649]
gpu_frequencies: [305, 587, 734]
thermal_policy: log_only
power_sampling_command: ""
device_command_prefix: ""
decode_command_template: ""
prefill_command_template: ""
dry_run: true
resume: true
alpha_levels: [1.0, 0.9, 0.8, 0.7]
""".lstrip(),
        encoding="utf-8",
    )


def test_aggregate_marks_throttled_from_nested_manifest(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT))
    from profiles.offline_profile_lib import aggregate_rows

    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    (manifest_dir / "cpu_B2_2649.manifest.json").write_text(
        """
{
  "phase": "decode",
  "state_name": "cpu_B2_2649",
  "length": 512,
  "state": {"cpu_freq_mhz": "2649"},
  "measurement": {"measured_cpu_freq_mhz": 2400}
}
""".strip(),
        encoding="utf-8",
    )

    rows = aggregate_rows(
        [
            {
                "phase": "decode",
                "state_name": "cpu_B2_2649",
                "length": "512",
                "run_id": "0",
                "n_tokens": "64",
                "elapsed_ms": "4000",
                "energy_mj": "960",
                "status": "ok",
            },
            {
                "phase": "decode",
                "state_name": "cpu_B2_2649",
                "length": "512",
                "run_id": "1",
                "n_tokens": "64",
                "elapsed_ms": "4000",
                "energy_mj": "960",
                "status": "ok",
            },
        ],
        repeat=2,
        manifest_dir=manifest_dir,
    )

    assert rows[0]["status"] == "throttled"


def test_toy_raw_csv_has_fixed_eight_column_schema() -> None:
    raw_path = ROOT / "profiles" / "toy_perf_profile_raw.csv"
    with raw_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        assert next(reader) == RAW_FIELDS
        for row in reader:
            assert len(row) == 8


def test_toy_pipeline_aggregates_filters_builds_frontier_and_plans(tmp_path: Path) -> None:
    raw = tmp_path / "perf_profile_raw.csv"
    catalog = tmp_path / "state_catalog.csv"
    config = tmp_path / "offline_profile.yaml"
    agg = tmp_path / "perf_profile_agg.csv"
    pareto = tmp_path / "pareto_states.csv"
    frontier = tmp_path / "frontier.csv"
    plan = tmp_path / "request_plan.csv"
    report = tmp_path / "offline_profile_summary.md"

    shutil.copy(ROOT / "profiles" / "toy_perf_profile_raw.csv", raw)
    shutil.copy(ROOT / "profiles" / "toy_state_catalog.csv", catalog)
    _write_test_config(config, tmp_path)

    filter_result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "profiles" / "filter_profiles.py"),
            "--config",
            str(config),
            "--raw",
            str(raw),
            "--state-catalog",
            str(catalog),
            "--agg",
            str(agg),
            "--pareto",
            str(pareto),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert filter_result.returncode == 0, filter_result.stderr

    agg_rows = _read_csv(agg)
    npu_512 = _row(agg_rows, phase="decode", state_name="npu_burst_cap2048", length=512)
    assert npu_512["n_runs"] == "2"
    assert float(npu_512["throughput_worst_tps"]) == pytest.approx(64 / 3.36)
    assert float(npu_512["mean_tbt_ms"]) == pytest.approx(52.5)
    assert float(npu_512["energy_per_token_mj"]) == pytest.approx(20.5)
    assert npu_512["status"] == "ok"

    unstable = _row(agg_rows, phase="decode", state_name="npu_balanced_cap2048", length=512)
    assert unstable["status"] == "unstable"

    pareto_rows = _read_csv(pareto)
    gpu_512 = _row(pareto_rows, phase="decode", state_name="gpu_734", length=512)
    assert gpu_512["dominated"] == "true"
    assert gpu_512["dominated_by"] == "cpu_B2_2649"

    frontier_result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "profiles" / "build_frontier.py"),
            "--config",
            str(config),
            "--agg",
            str(agg),
            "--state-catalog",
            str(catalog),
            "--pareto",
            str(pareto),
            "--output",
            str(frontier),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert frontier_result.returncode == 0, frontier_result.stderr

    frontier_rows = _read_csv(frontier)
    assert {row["phase"] for row in frontier_rows} == {"decode"}

    selected_512_fastest = _row(frontier_rows, phase="decode", length=512, alpha="1.0")
    assert selected_512_fastest["state_name"] == "npu_burst_cap2048"
    assert selected_512_fastest["slo_feasible"] == "true"
    assert selected_512_fastest["selected"] == "true"

    selected_512_slack = _row(frontier_rows, phase="decode", length=512, alpha="0.8")
    assert selected_512_slack["state_name"] == "cpu_B2_2649"
    assert float(selected_512_slack["target_mean_tbt_ms"]) == pytest.approx(
        1000.0 / (0.8 * float(selected_512_slack["qmax_tps"]))
    )

    selected_1024_slack = _row(frontier_rows, phase="decode", length=1024, alpha="0.8")
    assert selected_1024_slack["state_name"] == "npu_burst_cap2048"

    planner_result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "profiles" / "offline_bucket_planner.py"),
            "--config",
            str(config),
            "--agg",
            str(agg),
            "--pareto",
            str(pareto),
            "--state-catalog",
            str(catalog),
            "--transition-profile",
            str(tmp_path / "missing_transition_profile.csv"),
            "--prompt-len",
            "400",
            "--output-len",
            "300",
            "--alpha",
            "0.8",
            "--include-transition",
            "true",
            "--default-transition-energy-mj",
            "2000",
            "--default-transition-latency-ms",
            "50",
            "--output",
            str(plan),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert planner_result.returncode == 0, planner_result.stderr
    plan_rows = _read_csv(plan)
    assert [row["selected_state"] for row in plan_rows] == ["npu_burst_cap2048", "npu_burst_cap2048"]
    assert plan_rows[0]["transition_energy_mj"] == "0.000000"
    assert float(plan_rows[-1]["total_decode_energy_mj"]) == pytest.approx(6432.0)
    assert plan_rows[-1]["slo_ok"] == "true"

    report_result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "reports" / "generate_offline_report.py"),
            "--config",
            str(config),
            "--raw",
            str(raw),
            "--agg",
            str(agg),
            "--pareto",
            str(pareto),
            "--frontier",
            str(frontier),
            "--request-plan",
            str(plan),
            "--output",
            str(report),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert report_result.returncode == 0, report_result.stderr
    text = report.read_text(encoding="utf-8")
    assert "not a p95 TBT guarantee" in text
    assert "cap2048 represents sub-2048 contexts" in text
