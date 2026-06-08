from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_build_profile_and_simulate_system_benefit(tmp_path: Path) -> None:
    input_dir = tmp_path / "Paper_Writing" / "offline"
    profile_path = tmp_path / "system_benefit_profile.json"
    result_path = tmp_path / "sim_result.json"

    _write_csv(
        input_dir / "CPU" / "cpu_fixed_i8mm_summary_20260605.csv",
        [
            {
                "experiment_dir": "toy",
                "run_name": "cpu_B2_1000_decode64_run0",
                "status": "ok",
                "case_name": "B2",
                "cpu_freq_khz": "1000000",
                "decode_tokens": "64",
                "context_tokens": "64",
                "mean_tbt_ms": "83.333333",
                "throughput_tps": "12",
                "avg_power_mw": "1000",
                "median_power_mw": "1000",
                "source_results": "toy/cpu.csv",
            },
            {
                "experiment_dir": "toy",
                "run_name": "cpu_B2_1000_decode96_run0",
                "status": "ok",
                "case_name": "B2",
                "cpu_freq_khz": "1000000",
                "decode_tokens": "64",
                "context_tokens": "96",
                "mean_tbt_ms": "83.333333",
                "throughput_tps": "12",
                "avg_power_mw": "1000",
                "median_power_mw": "1000",
                "source_results": "toy/cpu.csv",
            },
        ],
    )
    _write_csv(
        input_dir / "GPU" / "gpu_freq_sweep_summary_20260605.csv",
        [
            {
                "experiment_dir": "toy",
                "run_name": "gpu_734_decode64_run0",
                "status": "ok",
                "freq_mhz": "734",
                "decode_tokens": "64",
                "context_tokens": "64",
                "mean_tbt_ms": "62.5",
                "throughput_tps": "16",
                "avg_power_mw": "700",
                "median_power_mw": "700",
                "source_results": "toy/gpu.csv",
            },
            {
                "experiment_dir": "toy",
                "run_name": "gpu_734_decode96_run0",
                "status": "ok",
                "freq_mhz": "734",
                "decode_tokens": "64",
                "context_tokens": "96",
                "mean_tbt_ms": "62.5",
                "throughput_tps": "16",
                "avg_power_mw": "700",
                "median_power_mw": "700",
                "source_results": "toy/gpu.csv",
            },
        ],
    )
    _write_csv(
        input_dir / "NPU" / "npu_2k_workpoint_summary_20260605.csv",
        [
            {
                "experiment_dir": "toy",
                "run_name": "npu_burst",
                "workpoint": "burst",
                "status": "ok",
                "decode_tokens": "64",
                "context_tokens": "64",
                "mean_tbt_ms": "50",
                "throughput_tps": "20",
                "avg_power_mw": "4000",
                "delta_vs_baseline_mw": "4000",
                "source_results": "toy/npu.csv",
            },
            {
                "experiment_dir": "toy",
                "run_name": "npu_burst",
                "workpoint": "burst",
                "status": "ok",
                "decode_tokens": "64",
                "context_tokens": "96",
                "mean_tbt_ms": "100",
                "throughput_tps": "10",
                "avg_power_mw": "2500",
                "delta_vs_baseline_mw": "2500",
                "source_results": "toy/npu.csv",
            },
        ],
    )

    prefill_dir = tmp_path / "Paper_Writing" / "ecofrontier" / "review_parts" / "states"
    prefill_dir.mkdir(parents=True, exist_ok=True)
    (prefill_dir / "QNN_NPU_prefill.json").write_text(
        json.dumps(
            [
                {
                    "state_id": "npu_burst",
                    "backend": "QNN_NPU",
                    "phase": "prefill",
                    "prompt_tokens": 64,
                    "npu_workpoint": "burst",
                    "throughput_tps": 100,
                    "active_power_mw": 1000,
                    "support_status": "ok",
                    "stable": True,
                }
            ]
        ),
        encoding="utf-8",
    )

    build = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_system_benefit_profile.py"),
            "--input-dir",
            str(input_dir),
            "--output",
            str(profile_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert build.returncode == 0, build.stderr

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert profile["schema_version"] == "system_benefit_profile.v1"
    decode_records = [row for row in profile["records"] if row["phase"] == "decode"]
    prefill_records = [row for row in profile["records"] if row["phase"] == "prefill"]
    assert len(decode_records) == 6
    assert len(prefill_records) == 1
    gpu_64 = next(row for row in decode_records if row["state_name"] == "gpu_734" and row["length"] == 64)
    assert gpu_64["latency_ms_per_bucket"] == pytest.approx(2000.0)
    assert gpu_64["energy_mj_per_bucket"] == pytest.approx(1400.0)

    sim = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "simulate_system_benefit.py"),
            "--profile",
            str(profile_path),
            "--input-len",
            "64",
            "--output-len",
            "64",
            "--slo-tps",
            "15",
            "--prefill-backend",
            "NPU",
            "--prefill-state",
            "npu_burst",
            "--baseline-prefill-backend",
            "NPU",
            "--baseline-prefill-state",
            "npu_burst",
            "--baseline-decode-backend",
            "CPU",
            "--bucket-size",
            "32",
            "--output",
            str(result_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert sim.returncode == 0, sim.stderr

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert [segment["selected_state"] for segment in result["scheduled"]["segments"]] == ["gpu_734", "gpu_734"]
    assert result["scheduled"]["decode_latency_ms"] == pytest.approx(4000.0)
    assert result["scheduled"]["decode_energy_mj"] == pytest.approx(2800.0)
    assert result["scheduled"]["prefill_latency_ms"] == pytest.approx(640.0)
    assert result["scheduled"]["total_latency_ms"] == pytest.approx(4640.0)
    assert result["scheduled"]["total_energy_mj"] == pytest.approx(3440.0)
    assert result["baseline"]["decode_latency_ms"] == pytest.approx(64 / 12 * 1000)
    assert result["relative_to_baseline"]["latency_reduction_pct"] == pytest.approx(22.3214285714)
    assert result["relative_to_baseline"]["energy_reduction_pct"] == pytest.approx(42.4107142857)
    assert result["scheduled"]["slo_met"] is True


def test_manual_log_mode_builds_context_buckets_and_simulator_uses_decode_ranges(tmp_path: Path) -> None:
    profile_path = tmp_path / "manual_profile.json"
    result_path = tmp_path / "manual_result.json"
    log_dir = tmp_path / "Paper_Writing" / "offline" / "Log"
    power_dir = tmp_path / "Paper_Writing" / "offline" / "Power"
    log_dir.mkdir(parents=True)
    power_dir.mkdir(parents=True)

    gpu_log = log_dir / "gpu_734_decode576_run0_valid_tbt.csv"
    cpu_log = log_dir / "cpu_B2_1000_decode576_run0_valid_tbt.csv"
    with gpu_log.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for context in range(513, 577):
            writer.writerow([context, 62500])
    with cpu_log.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for context in range(513, 577):
            writer.writerow([context, 83333.333333])

    _write_csv(
        power_dir / "gpu_734_decode576_run0_power.csv",
        [{"timestamp_ms": 1, "power_mw_est": 700}, {"timestamp_ms": 2, "power_mw_est": 700}],
    )
    _write_csv(
        power_dir / "cpu_B2_1000_decode576_run0_power.csv",
        [{"timestamp_ms": 1, "power_mw_est": 1000}, {"timestamp_ms": 2, "power_mw_est": 1000}],
    )

    for backend, state, csv_path, max_context in [
        ("GPU", "734", gpu_log, "576"),
        ("CPU", "B2_1000", cpu_log, "576"),
    ]:
        build = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "build_system_benefit_profile.py"),
                "--backend",
                backend,
                "--state",
                state,
                "--csv",
                str(csv_path),
                "--max-context-len",
                max_context,
                "--output",
                str(profile_path),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert build.returncode == 0, build.stderr

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    decode_records = [row for row in profile["records"] if row["phase"] == "decode"]
    assert len(decode_records) == 4
    gpu_buckets = [row for row in decode_records if row["state_name"] == "gpu_734"]
    assert [(row["bucket_lo"], row["bucket_hi"]) for row in gpu_buckets] == [(513, 544), (545, 576)]
    assert gpu_buckets[0]["throughput_tps"] == pytest.approx(16.0)
    assert gpu_buckets[0]["power_mw"] == pytest.approx(700.0)
    assert gpu_buckets[0]["energy_mj_per_bucket"] == pytest.approx(1400.0)

    sim = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "simulate_system_benefit.py"),
            "--profile",
            str(profile_path),
            "--input-len",
            "512",
            "--output-len",
            "64",
            "--slo-tps",
            "15",
            "--prefill-latency-ms",
            "0",
            "--prefill-energy-mj",
            "0",
            "--baseline-prefill-latency-ms",
            "0",
            "--baseline-prefill-energy-mj",
            "0",
            "--baseline-decode-backend",
            "CPU",
            "--context-match",
            "exact",
            "--bucket-size",
            "32",
            "--output",
            str(result_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert sim.returncode == 0, sim.stderr
    result = json.loads(result_path.read_text(encoding="utf-8"))

    assert [segment["selected_state"] for segment in result["scheduled"]["segments"]] == ["gpu_734", "gpu_734"]
    assert [
        (segment["context_bucket_lo"], segment["context_bucket_hi"])
        for segment in result["scheduled"]["segments"]
    ] == [(513, 544), (545, 576)]
    assert [
        (segment["matched_profile_bucket_lo"], segment["matched_profile_bucket_hi"])
        for segment in result["scheduled"]["segments"]
    ] == [(513, 544), (545, 576)]
    assert result["scheduled"]["decode_latency_ms"] == pytest.approx(4000.0)
    assert result["scheduled"]["decode_energy_mj"] == pytest.approx(2800.0)
    assert result["baseline"]["decode_latency_ms"] == pytest.approx(64 / 12 * 1000)
