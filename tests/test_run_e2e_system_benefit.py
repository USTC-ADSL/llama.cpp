from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_e2e_runner_dry_run_exports_dynamic_schedule(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    output_path = tmp_path / "result.json"
    plan_path.write_text(
        json.dumps(
            {
                "scheduled": {
                    "decode_schedule_env": (
                        "1:opencl{gpu_freq_hz=967000000};"
                        "33:cpu{threads=6,affinity=FC,cpu_policy0_freq_khz=3532800,cpu_policy6_freq_khz=4320000}"
                    ),
                    "segments": [],
                }
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_e2e_system_benefit.py"),
            "--strategy",
            "auto",
            "--plan-json",
            str(plan_path),
            "--run-command",
            "printf 'bench: 12.5 tok/s\\n'",
            "--mah-command",
            "printf 1000",
            "--output",
            str(output_path),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "GGML_HETERO_DYNAMIC_DECODE_SCHEDULE=" in result.stdout
    assert "1:opencl{gpu_freq_hz=967000000}" in result.stdout
    assert "would run" in result.stdout
    assert not output_path.exists()


def test_e2e_runner_can_wrap_device_commands_with_adb_su(tmp_path: Path) -> None:
    output_path = tmp_path / "result.json"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_e2e_system_benefit.py"),
            "--strategy",
            "fixed",
            "--fixed-backend",
            "gpu_967",
            "--run-command",
            "cd /data/local/tmp/deploy && ./llama-bench -n 64",
            "--run-command-on-device",
            "--mah-command",
            "cat /sys/class/power_supply/battery/charge_counter",
            "--mah-command-on-device",
            "--adb-su",
            "--adb-serial",
            "192.168.1.148:36645",
            "--output",
            str(output_path),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "adb -s 192.168.1.148:36645 shell" in result.stdout
    assert "su 0 sh -c" in result.stdout
    assert "cat /sys/class/power_supply/battery/charge_counter" in result.stdout
    assert "./llama-bench -n 64" in result.stdout


def test_e2e_runner_measures_mah_delta_and_throughput(tmp_path: Path) -> None:
    output_path = tmp_path / "result.json"
    counter_path = tmp_path / "mah_count.txt"
    mah_script = tmp_path / "mah.py"
    mah_script.write_text(
        """
from pathlib import Path
path = Path(__import__('sys').argv[1])
count = int(path.read_text()) if path.exists() else 0
values = [1000000.0, 1002500.0]
path.write_text(str(count + 1))
print(values[count])
""".strip(),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_e2e_system_benefit.py"),
            "--strategy",
            "fixed",
            "--fixed-backend",
            "gpu_967",
            "--output-len",
            "64",
            "--run-command",
            "printf 'total time = 8.0 s\\n'",
            "--mah-command",
            f"{sys.executable} {mah_script} {counter_path}",
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["strategy"] == "fixed"
    assert payload["charge_unit"] == "uah"
    assert payload["energy_mah_delta"] == 2.5
    assert payload["energy_mj"] == 34650.0
    assert payload["throughput_tps"] == 8.0
    assert payload["run"]["returncode"] == 0
    assert "stdout" not in payload["run"]
    assert "stderr" not in payload["run"]
    assert Path(payload["run"]["log_path"]).read_text(encoding="utf-8") == "total time = 8.0 s\n"


def test_e2e_runner_patches_template_decode_and_context_lengths(tmp_path: Path) -> None:
    template = tmp_path / "schedule_test.sh"
    output_path = tmp_path / "result.json"
    template.write_text(
        """
export GGML_HETERO_DYNAMIC_DECODE_SCHEDULE="1:opencl{gpu_freq_hz=967000000}"
N_GEN=${N_GEN:-450}
N_DEPTH=${N_DEPTH:-0}
./llama-bench -v -m model.gguf -t 1 -p 0 -n "$N_GEN" -d "$N_DEPTH" -c 1920 -b 1
""".strip(),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_e2e_system_benefit.py"),
            "--strategy",
            "fixed",
            "--fixed-backend",
            "CPU_B2S4_4320000_3532800",
            "--input-len",
            "512",
            "--output-len",
            "1750",
            "--command-template-file",
            str(template),
            "--run-command",
            "unused",
            "--mah-command",
            "printf 1000",
            "--output",
            str(output_path),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    patched_line = next(line for line in result.stdout.splitlines() if "would run:" in line)
    patched_path = Path(patched_line.split("bash ", 1)[1])
    patched = patched_path.read_text(encoding="utf-8")
    assert "N_GEN=${N_GEN:-1750}" in patched
    assert "N_DEPTH=${N_DEPTH:-512}" in patched
    assert "1:cpu{threads=6,affinity=FC,cpu_policy0_freq_khz=3532800,cpu_policy6_freq_khz=4320000}" in patched
