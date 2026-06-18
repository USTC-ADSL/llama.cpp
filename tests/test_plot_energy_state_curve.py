from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


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


def test_plot_energy_state_curve_reads_states_from_profile(tmp_path: Path) -> None:
    profile_path = tmp_path / "system_benefit_offline_profile.csv"
    output_path = tmp_path / "gpu_energy_curve.svg"

    rows = [
        {
            "phase": "decode",
            "backend": "GPU",
            "state_name": "gpu_734",
            "state_group": "GPU",
            "bucket_lo": 513,
            "bucket_hi": 544,
            "bucket_tokens": 32,
            "throughput_tps": 16,
            "power_mw": 700,
            "energy_mj_per_token": 43.75,
            "energy_mj_per_bucket": 1400,
        },
        {
            "phase": "decode",
            "backend": "GPU",
            "state_name": "gpu_1100",
            "state_group": "GPU",
            "bucket_lo": 513,
            "bucket_hi": 544,
            "bucket_tokens": 32,
            "throughput_tps": 20,
            "power_mw": 1200,
            "energy_mj_per_token": 60,
            "energy_mj_per_bucket": 1920,
        },
        {
            "phase": "decode",
            "backend": "GPU",
            "state_name": "gpu_734",
            "state_group": "GPU",
            "bucket_lo": 545,
            "bucket_hi": 576,
            "bucket_tokens": 32,
            "throughput_tps": 15,
            "power_mw": 720,
            "energy_mj_per_token": 48,
            "energy_mj_per_bucket": 1536,
        },
    ]
    with profile_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PROFILE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "plot_energy_state_curve.py"),
            "--profile",
            str(profile_path),
            "--states",
            "gpu_734,gpu_1100",
            "--bucket-hi",
            "544",
            "--output",
            str(output_path),
            "--format",
            "svg",
            "--title",
            "Toy GPU Energy Curve",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    svg = output_path.read_text(encoding="utf-8")
    assert "Toy GPU Energy Curve" in svg
    assert "gpu_734" in svg
    assert "gpu_1100" in svg
    assert "mJ/token" in svg
    assert "frequency/workpoint" in svg
