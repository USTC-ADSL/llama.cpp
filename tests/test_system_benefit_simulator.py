from __future__ import annotations

import csv
import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
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


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_profile(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == PROFILE_FIELDS
        return list(reader)


def _power_rows(head_value: float, kept_value: float, tail_value: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx in range(10):
        rows.append({"timestamp_ms": idx, "power_mw_est": head_value})
    for idx in range(10, 16):
        rows.append({"timestamp_ms": idx, "power_mw_est": kept_value})
    for idx in range(16, 19):
        rows.append({"timestamp_ms": idx, "power_mw_est": tail_value})
    return rows


def test_default_backend_transition_latency_table() -> None:
    simulator = runpy.run_path(str(ROOT / "scripts" / "simulate_system_benefit.py"))
    backend_transition_latency_ms = simulator["backend_transition_latency_ms"]

    assert backend_transition_latency_ms("NPU", "CPU") == pytest.approx(5.0)
    assert backend_transition_latency_ms("GPU", "CPU") == pytest.approx(15.0)
    assert backend_transition_latency_ms("CPU", "GPU") == pytest.approx(20.0)
    assert backend_transition_latency_ms("NPU", "GPU") == pytest.approx(20.0)
    assert backend_transition_latency_ms("CPU", "NPU") == pytest.approx(80.0)
    assert backend_transition_latency_ms("GPU", "NPU") == pytest.approx(80.0)


def test_build_profile_and_simulate_system_benefit(tmp_path: Path) -> None:
    input_dir = tmp_path / "Paper_Writing" / "offline"
    profile_path = tmp_path / "system_benefit_profile.csv"
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

    profile = _read_profile(profile_path)
    decode_records = [row for row in profile if row["phase"] == "decode"]
    prefill_records = [row for row in profile if row["phase"] == "prefill"]
    assert len(decode_records) == 6
    assert len(prefill_records) == 1
    gpu_64 = next(row for row in decode_records if row["state_name"] == "gpu_734" and row["bucket_hi"] == "64")
    assert float(gpu_64["throughput_tps"]) == pytest.approx(16.0)
    assert float(gpu_64["energy_mj_per_bucket"]) == pytest.approx(1400.0)

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
            "--slo-tbt-ms",
            "66.6666667",
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
            "--baseline-decode-state",
            "cpu_B2_1000",
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
    profile_path = tmp_path / "manual_profile.csv"
    result_path = tmp_path / "manual_result.json"
    log_dir = tmp_path / "Paper_Writing" / "offline" / "Log"
    power_dir = tmp_path / "Paper_Writing" / "offline" / "Power"
    log_dir.mkdir(parents=True)
    power_dir.mkdir(parents=True)

    gpu_log = log_dir / "gpu_734_decode576_run0_valid_tbt.csv"
    cpu_log = log_dir / "cpu_B2_1000_decode576_run0_valid_tbt.csv"
    cpu_b2s4_log = log_dir / "cpu_B2S4_4320000_3532800_decode576_run0_valid_tbt.csv"
    with gpu_log.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for context in range(513, 577):
            writer.writerow([context, 62500])
    with cpu_log.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for context in range(513, 577):
            writer.writerow([context, 83333.333333])
    with cpu_b2s4_log.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for context in range(513, 577):
            writer.writerow([context, 50000])

    _write_csv(
        power_dir / "gpu_734_decode576_run0_power.csv",
        _power_rows(head_value=9999, kept_value=700, tail_value=1),
    )
    _write_csv(
        power_dir / "cpu_B2_1000_decode576_run0_power.csv",
        _power_rows(head_value=9999, kept_value=1000, tail_value=1),
    )
    _write_csv(
        power_dir / "cpu_B2S4_4320000_3532800_decode576_run0_power.csv",
        _power_rows(head_value=9999, kept_value=5000, tail_value=1),
    )

    for backend, state, csv_path, max_context in [
        ("GPU", "734", gpu_log, "576"),
        ("CPU", "B2_1000", cpu_log, "576"),
        ("CPU", "B2S4_4320000_3532800", cpu_b2s4_log, "576"),
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

    profile = _read_profile(profile_path)
    decode_records = [row for row in profile if row["phase"] == "decode"]
    assert len(decode_records) == 6
    gpu_buckets = [row for row in decode_records if row["state_name"] == "gpu_734"]
    assert [(int(row["bucket_lo"]), int(row["bucket_hi"])) for row in gpu_buckets] == [(513, 544), (545, 576)]
    assert float(gpu_buckets[0]["throughput_tps"]) == pytest.approx(16.0)
    assert float(gpu_buckets[0]["power_mw"]) == pytest.approx(700.0)
    assert float(gpu_buckets[0]["energy_mj_per_bucket"]) == pytest.approx(1400.0)

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
            "--slo-tbt-ms",
            "100",
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
    assert [segment["selected_state"] for segment in result["baseline"]["segments"]] == [
        "cpu_B2S4_4320000_3532800",
        "cpu_B2S4_4320000_3532800",
    ]
    assert result["baseline"]["decode_latency_ms"] == pytest.approx(64 / 20 * 1000)


def test_default_baseline_decode_state_is_fixed_for_each_backend(tmp_path: Path) -> None:
    profile_path = tmp_path / "baseline_profile.csv"
    result_path = tmp_path / "baseline_result.json"
    rows = []
    for bucket_lo, bucket_hi in [(513, 544), (545, 576)]:
        for backend, state_name, state_group, throughput, power in [
            ("GPU", "gpu_734", "GPU", 16, 700),
            ("GPU", "gpu_1100", "GPU", 22, 6000),
            ("NPU", "npu_low_balanced", "low_balanced", 12, 800),
            ("NPU", "npu_burst", "burst", 20, 5000),
        ]:
            rows.append(
                {
                    "phase": "decode",
                    "backend": backend,
                    "state_name": state_name,
                    "state_group": state_group,
                    "bucket_lo": bucket_lo,
                    "bucket_hi": bucket_hi,
                    "bucket_tokens": 32,
                    "throughput_tps": throughput,
                    "power_mw": power,
                    "energy_mj_per_token": power / throughput,
                    "energy_mj_per_bucket": power / throughput * 32,
                }
            )
    _write_csv(profile_path, rows)

    for backend, expected_state in [("GPU", "gpu_1100"), ("NPU", "npu_burst")]:
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
                "--slo-tbt-ms",
                "100",
                "--prefill-latency-ms",
                "0",
                "--prefill-energy-mj",
                "0",
                "--baseline-prefill-latency-ms",
                "0",
                "--baseline-prefill-energy-mj",
                "0",
                "--baseline-decode-backend",
                backend,
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
        assert [segment["selected_state"] for segment in result["baseline"]["segments"]] == [
            expected_state,
            expected_state,
        ]


def test_decode_switch_reason_marks_slo_and_energy_changes(tmp_path: Path) -> None:
    profile_path = tmp_path / "switch_profile.csv"
    result_path = tmp_path / "switch_result.json"
    rows = []
    for bucket_lo, bucket_hi, gpu_tps, npu_tps in [
        (513, 544, 20, 25),
        (545, 576, 5, 25),
        (577, 608, 20, 25),
    ]:
        rows.extend(
            [
                {
                    "phase": "decode",
                    "backend": "GPU",
                    "state_name": "gpu_734",
                    "state_group": "GPU",
                    "bucket_lo": bucket_lo,
                    "bucket_hi": bucket_hi,
                    "bucket_tokens": 32,
                    "throughput_tps": gpu_tps,
                    "power_mw": 100,
                    "energy_mj_per_token": 100 / gpu_tps,
                    "energy_mj_per_bucket": 100 / gpu_tps * 32,
                },
                {
                    "phase": "decode",
                    "backend": "NPU",
                    "state_name": "npu_burst",
                    "state_group": "burst",
                    "bucket_lo": bucket_lo,
                    "bucket_hi": bucket_hi,
                    "bucket_tokens": 32,
                    "throughput_tps": npu_tps,
                    "power_mw": 1000,
                    "energy_mj_per_token": 1000 / npu_tps,
                    "energy_mj_per_bucket": 1000 / npu_tps * 32,
                },
            ]
        )
    _write_csv(profile_path, rows)

    sim = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "simulate_system_benefit.py"),
            "--profile",
            str(profile_path),
            "--input-len",
            "512",
            "--output-len",
            "96",
            "--slo-tbt-ms",
            "100",
            "--prefill-latency-ms",
            "0",
            "--prefill-energy-mj",
            "0",
            "--baseline-prefill-latency-ms",
            "0",
            "--baseline-prefill-energy-mj",
            "0",
            "--baseline-decode-backend",
            "GPU",
            "--baseline-decode-state",
            "gpu_734",
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
    segments = result["scheduled"]["segments"]
    assert [segment["selected_state"] for segment in segments] == ["gpu_734", "npu_burst", "gpu_734"]
    assert [segment["switch_reason"] for segment in segments] == ["", "slo", "energy"]


def test_decode_dp_counts_initial_transition_in_step_slo(tmp_path: Path) -> None:
    profile_path = tmp_path / "transition_slo_profile.json"
    result_path = tmp_path / "transition_slo_result.json"
    records = []
    for bucket_lo, bucket_hi in [(513, 544), (545, 576)]:
        records.extend(
            [
                {
                    "phase": "decode",
                    "backend": "GPU",
                    "state_name": "gpu_low",
                    "state_group": "GPU",
                    "bucket_lo": bucket_lo,
                    "bucket_hi": bucket_hi,
                    "throughput_tps": 10,
                    "power_mw": 1000,
                },
                {
                    "phase": "decode",
                    "backend": "NPU",
                    "state_name": "npu_fast_low_energy",
                    "state_group": "burst",
                    "bucket_lo": bucket_lo,
                    "bucket_hi": bucket_hi,
                    "throughput_tps": 20,
                    "power_mw": 200,
                },
            ]
        )
    profile_path.write_text(
        json.dumps(
            {
                "schema_version": "toy",
                "records": records,
                "transitions": [
                    {
                        "from_state": "gpu_low",
                        "to_state": "npu_fast_low_energy",
                        "latency_ms": 2000,
                        "energy_mj": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

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
            "--slo-tbt-ms",
            "100",
            "--initial-decode-state",
            "gpu_low",
            "--prefill-latency-ms",
            "0",
            "--prefill-energy-mj",
            "0",
            "--baseline-prefill-latency-ms",
            "0",
            "--baseline-prefill-energy-mj",
            "0",
            "--baseline-decode-backend",
            "GPU",
            "--baseline-decode-state",
            "gpu_low",
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
    segments = result["scheduled"]["segments"]
    assert [segment["selected_state"] for segment in segments] == ["gpu_low", "gpu_low"]
    assert segments[0]["transition_from_prev"] == "gpu_low"
    assert segments[0]["step_slo_ok"] is True
    assert result["scheduled"]["slo_satisfaction_rate"] == pytest.approx(1.0)


def test_decode_dp_chooses_closest_to_slo_when_no_step_is_feasible(tmp_path: Path) -> None:
    profile_path = tmp_path / "best_effort_profile.csv"
    result_path = tmp_path / "best_effort_result.json"
    rows = [
        {
            "phase": "decode",
            "backend": "GPU",
            "state_name": "gpu_cheap_slow",
            "state_group": "GPU",
            "bucket_lo": 513,
            "bucket_hi": 544,
            "bucket_tokens": 32,
            "throughput_tps": 5,
            "power_mw": 100,
            "energy_mj_per_token": 20,
            "energy_mj_per_bucket": 640,
        },
        {
            "phase": "decode",
            "backend": "NPU",
            "state_name": "npu_expensive_closer",
            "state_group": "burst",
            "bucket_lo": 513,
            "bucket_hi": 544,
            "bucket_tokens": 32,
            "throughput_tps": 8,
            "power_mw": 2000,
            "energy_mj_per_token": 250,
            "energy_mj_per_bucket": 8000,
        },
    ]
    _write_csv(profile_path, rows)

    sim = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "simulate_system_benefit.py"),
            "--profile",
            str(profile_path),
            "--input-len",
            "512",
            "--output-len",
            "32",
            "--slo-tbt-ms",
            "50",
            "--prefill-latency-ms",
            "0",
            "--prefill-energy-mj",
            "0",
            "--baseline-prefill-latency-ms",
            "0",
            "--baseline-prefill-energy-mj",
            "0",
            "--baseline-decode-backend",
            "GPU",
            "--baseline-decode-state",
            "gpu_cheap_slow",
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
    segment = result["scheduled"]["segments"][0]
    assert segment["selected_state"] == "npu_expensive_closer"
    assert segment["step_slo_ok"] is False
    assert segment["selection_mode"] == "best_effort_closest_to_slo"
    assert result["scheduled"]["slo_satisfaction_rate"] == pytest.approx(0.0)


def test_default_backend_transition_costs_and_schedule_env_are_exported(tmp_path: Path) -> None:
    profile_path = tmp_path / "backend_transition_profile.csv"
    result_path = tmp_path / "backend_transition_result.json"
    rows = []
    for bucket_lo, bucket_hi, gpu_tps, cpu_tps in [
        (513, 544, 20, 10),
        (545, 576, 9, 20),
    ]:
        rows.extend(
            [
                {
                    "phase": "decode",
                    "backend": "GPU",
                    "state_name": "gpu_967",
                    "state_group": "GPU",
                    "bucket_lo": bucket_lo,
                    "bucket_hi": bucket_hi,
                    "bucket_tokens": 32,
                    "throughput_tps": gpu_tps,
                    "power_mw": 100,
                    "energy_mj_per_token": 100 / gpu_tps,
                    "energy_mj_per_bucket": 100 / gpu_tps * 32,
                },
                {
                    "phase": "decode",
                    "backend": "CPU",
                    "state_name": "cpu_B2S4_4320000_3532800",
                    "state_group": "B2S4",
                    "bucket_lo": bucket_lo,
                    "bucket_hi": bucket_hi,
                    "bucket_tokens": 32,
                    "throughput_tps": cpu_tps,
                    "power_mw": 2000,
                    "energy_mj_per_token": 2000 / cpu_tps,
                    "energy_mj_per_bucket": 2000 / cpu_tps * 32,
                },
            ]
        )
    _write_csv(profile_path, rows)

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
            "--slo-tbt-ms",
            "100",
            "--prefill-latency-ms",
            "0",
            "--prefill-energy-mj",
            "0",
            "--baseline-prefill-latency-ms",
            "0",
            "--baseline-prefill-energy-mj",
            "0",
            "--baseline-decode-backend",
            "GPU",
            "--baseline-decode-state",
            "gpu_967",
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
    segments = result["scheduled"]["segments"]

    assert [segment["selected_state"] for segment in segments] == ["gpu_967", "cpu_B2S4_4320000_3532800"]
    assert segments[1]["transition_latency_ms"] == pytest.approx(15.0)
    assert segments[1]["transition_energy_mj"] == pytest.approx(30.0)
    assert segments[1]["transition_source"] == "backend_default"
    assert result["scheduled"]["decode_schedule_env"] == (
        "1:opencl{gpu_freq_hz=967000000};"
        "33:cpu{threads=6,affinity=FC,cpu_policy0_freq_khz=3532800,cpu_policy6_freq_khz=4320000}"
    )
    assert result["scheduled"]["decode_transition_latency_ms"] == pytest.approx(15.0)
    assert result["scheduled"]["decode_transition_energy_mj"] == pytest.approx(30.0)


def test_exact_context_match_allows_profile_bucket_covering_final_partial_segment(tmp_path: Path) -> None:
    profile_path = tmp_path / "partial_final_bucket_profile.csv"
    result_path = tmp_path / "partial_final_bucket_result.json"
    rows = []
    for bucket_lo, bucket_hi in [(1, 32), (33, 64)]:
        rows.append(
            {
                "phase": "decode",
                "backend": "GPU",
                "state_name": "gpu_967",
                "state_group": "GPU",
                "bucket_lo": bucket_lo,
                "bucket_hi": bucket_hi,
                "bucket_tokens": 32,
                "throughput_tps": 16,
                "power_mw": 800,
                "energy_mj_per_token": 50,
                "energy_mj_per_bucket": 1600,
            }
        )
    _write_csv(profile_path, rows)

    sim = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "simulate_system_benefit.py"),
            "--profile",
            str(profile_path),
            "--input-len",
            "0",
            "--output-len",
            "50",
            "--slo-tbt-ms",
            "100",
            "--prefill-latency-ms",
            "0",
            "--prefill-energy-mj",
            "0",
            "--baseline-prefill-latency-ms",
            "0",
            "--baseline-prefill-energy-mj",
            "0",
            "--baseline-decode-backend",
            "GPU",
            "--baseline-decode-state",
            "gpu_967",
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
    segments = result["scheduled"]["segments"]
    assert [(segment["context_bucket_lo"], segment["context_bucket_hi"]) for segment in segments] == [(1, 32), (33, 50)]
    assert segments[1]["matched_profile_bucket_lo"] == 33
    assert segments[1]["matched_profile_bucket_hi"] == 64
    assert segments[1]["context_match"] == "exact"
    assert segments[1]["num_tokens"] == 18


def test_decode_segments_align_non_multiple_input_to_next_context_bucket(tmp_path: Path) -> None:
    profile_path = tmp_path / "aligned_input_profile.csv"
    result_path = tmp_path / "aligned_input_result.json"
    rows = []
    for bucket_lo, bucket_hi in [(513, 544), (545, 576)]:
        rows.extend(
            [
                {
                    "phase": "decode",
                    "backend": "GPU",
                    "state_name": "gpu_967",
                    "state_group": "GPU",
                    "bucket_lo": bucket_lo,
                    "bucket_hi": bucket_hi,
                    "bucket_tokens": 32,
                    "throughput_tps": 16,
                    "power_mw": 800,
                    "energy_mj_per_token": 50,
                    "energy_mj_per_bucket": 1600,
                },
                {
                    "phase": "decode",
                    "backend": "CPU",
                    "state_name": "cpu_B2_4320000",
                    "state_group": "B2",
                    "bucket_lo": bucket_lo,
                    "bucket_hi": bucket_hi,
                    "bucket_tokens": 32,
                    "throughput_tps": 16 if bucket_lo == 513 else 100,
                    "power_mw": 100000 if bucket_lo == 513 else 1,
                    "energy_mj_per_token": 6250 if bucket_lo == 513 else 0.01,
                    "energy_mj_per_bucket": 200000 if bucket_lo == 513 else 0.32,
                },
            ]
        )
    _write_csv(profile_path, rows)

    sim = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "simulate_system_benefit.py"),
            "--profile",
            str(profile_path),
            "--input-len",
            "500",
            "--output-len",
            "76",
            "--slo-tbt-ms",
            "100",
            "--prefill-latency-ms",
            "0",
            "--prefill-energy-mj",
            "0",
            "--baseline-prefill-latency-ms",
            "0",
            "--baseline-prefill-energy-mj",
            "0",
            "--baseline-decode-backend",
            "GPU",
            "--baseline-decode-state",
            "gpu_967",
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
    segments = result["scheduled"]["segments"]
    assert [(segment["context_bucket_lo"], segment["context_bucket_hi"]) for segment in segments] == [
        (501, 544),
        (545, 576),
    ]
    assert [(segment["profile_query_bucket_lo"], segment["profile_query_bucket_hi"]) for segment in segments] == [
        (513, 544),
        (545, 576),
    ]
    assert [(segment["matched_profile_bucket_lo"], segment["matched_profile_bucket_hi"]) for segment in segments] == [
        (513, 544),
        (545, 576),
    ]
    assert [segment["num_tokens"] for segment in segments] == [44, 32]
    assert result["scheduled"]["decode_schedule_env"] == (
        "1:opencl{gpu_freq_hz=967000000};"
        "45:cpu{threads=2,affinity=C0,cpu_policy6_freq_khz=4320000}"
    )


def test_plot_decode_state_timeline_writes_svg(tmp_path: Path) -> None:
    result_path = tmp_path / "sim_result.json"
    figure_path = tmp_path / "decode_timeline.svg"
    result_path.write_text(
        json.dumps(
            {
                "scheduled": {
                    "segments": [
                        {
                            "segment_id": 0,
                            "context_bucket_lo": 513,
                            "context_bucket_hi": 544,
                            "selected_state": "cpu_B2S4_4320000_3532800",
                            "backend": "CPU",
                            "state_group": "B2S4",
                            "mean_tbt_ms": 50,
                            "energy_mj_per_token": 100,
                        },
                        {
                            "segment_id": 1,
                            "context_bucket_lo": 545,
                            "context_bucket_hi": 576,
                            "selected_state": "gpu_1100",
                            "backend": "GPU",
                            "state_group": "GPU",
                            "mean_tbt_ms": 45,
                            "energy_mj_per_token": 90,
                            "switch_reason": "energy",
                            "energy_saving_vs_prev_mj": 320,
                            "energy_saving_vs_prev_pct": 18.5,
                        },
                        {
                            "segment_id": 2,
                            "context_bucket_lo": 577,
                            "context_bucket_hi": 608,
                            "selected_state": "npu_burst",
                            "backend": "NPU",
                            "state_group": "burst",
                            "mean_tbt_ms": 40,
                            "energy_mj_per_token": 120,
                            "switch_reason": "slo",
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    plot = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "plot_decode_state_timeline.py"),
            "--input",
            str(result_path),
            "--output",
            str(figure_path),
            "--title",
            "Toy Decode Timeline",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert plot.returncode == 0, plot.stderr
    svg = figure_path.read_text(encoding="utf-8")
    assert "<svg" in svg
    assert "Toy Decode Timeline" in svg
    assert "Decode time sharing by selected backend state" in svg
    assert "SLO miss / unavailable" in svg
    assert "lower energy" in svg
    assert "-320mJ" in svg
    assert "SLO" in svg
    assert "CPU" in svg and "GPU" in svg and "NPU" in svg
    assert "cpu_B2S4_4320000_3532800" in svg
    assert "gpu_1100" in svg
    assert "npu_burst" in svg


def test_plot_decode_state_timeline_accepts_e2e_result_json(tmp_path: Path) -> None:
    result_path = tmp_path / "e2e_result.json"
    figure_path = tmp_path / "decode_timeline.svg"
    result_path.write_text(
        json.dumps(
            {
                "strategy": "auto",
                "plan": {
                    "scheduled": {
                        "segments": [
                            {
                                "segment_id": 0,
                                "context_bucket_lo": 1,
                                "context_bucket_hi": 32,
                                "selected_state": "cpu_B2_3513600",
                                "backend": "CPU",
                                "state_group": "B2",
                                "mean_tbt_ms": 34,
                                "energy_mj_per_token": 150,
                                "step_slo_ok": True,
                                "selection_mode": "feasible",
                            },
                            {
                                "segment_id": 1,
                                "context_bucket_lo": 33,
                                "context_bucket_hi": 64,
                                "selected_state": "gpu_1100",
                                "backend": "GPU",
                                "state_group": "GPU",
                                "mean_tbt_ms": 31,
                                "energy_mj_per_token": 90,
                                "energy_saving_vs_prev_mj": 256.4,
                                "energy_saving_vs_prev_pct": 12.3,
                                "step_slo_ok": True,
                                "selection_mode": "feasible",
                            },
                            {
                                "segment_id": 2,
                                "context_bucket_lo": 65,
                                "context_bucket_hi": 96,
                                "selected_state": "cpu_B2S4_4320000_3532800",
                                "backend": "CPU",
                                "state_group": "B2S4",
                                "mean_tbt_ms": 62,
                                "energy_mj_per_token": 190,
                                "energy_saving_vs_prev_mj": -120.0,
                                "step_slo_ok": False,
                                "step_slo_miss_ms": 22.4,
                                "selection_mode": "best_effort_closest_to_slo",
                            },
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    plot = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "plot_decode_state_timeline.py"),
            "--input",
            str(result_path),
            "--output",
            str(figure_path),
            "--title",
            "E2E Decode Timeline",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert plot.returncode == 0, plot.stderr
    svg = figure_path.read_text(encoding="utf-8")
    assert "E2E Decode Timeline" in svg
    assert "cpu_B2_3513600" in svg
    assert "gpu_1100" in svg
    assert "cpu_B2S4_4320000_3532800" in svg
    assert "-256mJ" in svg
    assert "SLO +22ms" in svg
