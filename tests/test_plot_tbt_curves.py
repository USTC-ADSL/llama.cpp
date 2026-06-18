from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _write_tbt(path: Path, start_context: int, values_us: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for offset, value in enumerate(values_us):
            writer.writerow([start_context + offset, value])


def test_plot_tbt_curves_groups_series_by_backend(tmp_path: Path) -> None:
    gpu_734 = tmp_path / "gpu_734.csv"
    gpu_1100 = tmp_path / "gpu_1100.csv"
    cpu_b2s4 = tmp_path / "cpu_b2s4.csv"
    npu_burst = tmp_path / "npu_burst.csv"
    output_dir = tmp_path / "figures"

    _write_tbt(gpu_734, 513, [62000, 63000, 65000, 67000])
    _write_tbt(gpu_1100, 513, [47000, 48000, 49000, 50000])
    _write_tbt(cpu_b2s4, 513, [52000, 53000, 54000, 55000])
    _write_tbt(npu_burst, 513, [45000, 46000, 47000, 48000])

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "plot_tbt_curves.py"),
            "--series",
            f"GPU:734:{gpu_734}",
            "--series",
            f"GPU:1100:{gpu_1100}",
            "--series",
            f"CPU:B2S4_4320000_3532800:{cpu_b2s4}",
            "--series",
            f"NPU:burst:{npu_burst}",
            "--output-dir",
            str(output_dir),
            "--format",
            "svg",
            "--title-prefix",
            "Toy TBT",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    gpu_svg = (output_dir / "gpu_tbt_curves.svg").read_text(encoding="utf-8")
    cpu_svg = (output_dir / "cpu_tbt_curves.svg").read_text(encoding="utf-8")
    npu_svg = (output_dir / "npu_tbt_curves.svg").read_text(encoding="utf-8")

    assert "Toy TBT GPU" in gpu_svg
    assert "734" in gpu_svg and "1100" in gpu_svg
    assert "Toy TBT CPU" in cpu_svg
    assert "B2S4_4320000_3532800" in cpu_svg
    assert "Toy TBT NPU" in npu_svg
    assert "burst" in npu_svg
    assert "Context length" in gpu_svg
    assert "TBT (ms)" in gpu_svg
