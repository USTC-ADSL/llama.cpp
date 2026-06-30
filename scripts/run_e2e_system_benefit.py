#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import time
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hetero_route_spec import route_spec_for_fixed_state as route_spec_for_state

SIMULATOR = ROOT / "scripts" / "simulate_system_benefit.py"


def normalize_backend(value: str) -> str:
    upper = value.strip().upper()
    if upper.startswith("GPU") or upper == "OPENCL":
        return "GPU"
    if "NPU" in upper or upper in {"QNN", "QNN_NPU", "HTP"}:
        return "NPU"
    if upper.startswith("CPU"):
        return "CPU"
    return upper


def load_plan_schedule(path: Path) -> tuple[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    schedule = str(payload.get("scheduled", {}).get("decode_schedule_env") or "")
    if not schedule:
        raise ValueError(f"{path} does not contain scheduled.decode_schedule_env")
    return schedule, payload


def run_planner(args: argparse.Namespace, output_dir: Path) -> tuple[str, dict[str, Any], Path]:
    plan_path = output_dir / "planner_result.json"
    command = [
        str(Path(os.environ.get("PYTHON", "python3"))),
        str(SIMULATOR),
        "--profile",
        args.profile,
        "--input-len",
        str(args.input_len),
        "--output-len",
        str(args.output_len),
        "--slo-tbt-ms",
        str(args.slo_tbt_ms),
        "--bucket-size",
        str(args.bucket_size),
        "--context-match",
        args.context_match,
        "--prefill-latency-ms",
        "0",
        "--prefill-energy-mj",
        "0",
        "--baseline-prefill-latency-ms",
        "0",
        "--baseline-prefill-energy-mj",
        "0",
        "--output",
        str(plan_path),
    ]
    if args.current_state:
        command.extend(["--current-state", args.current_state])
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"planner failed with exit {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    schedule, payload = load_plan_schedule(plan_path)
    return schedule, payload, plan_path


def env_for_schedule(schedule: str) -> dict[str, str]:
    return {
        "GGML_HETERO_DYNAMIC_MODE": "phase",
        "GGML_HETERO_DYNAMIC_PREFILL_ROUTE": "qnn-npu",
        "GGML_HETERO_DYNAMIC_DECODE_SCHEDULE": schedule,
        "GGML_HETERO_DYNAMIC_TRACE": "1",
        "GGML_HETERO_DYNAMIC_TRACE_TIMING": "1",
        "GGML_HETERO_DYNAMIC_TRACE_TIMING_DETAIL": "1",
    }


def command_with_placeholders(command: str, schedule: str, args: argparse.Namespace) -> str:
    exports = " ".join(f"export {key}={shlex.quote(value)};" for key, value in env_for_schedule(schedule).items())
    return command.format(
        decode_schedule_env=schedule,
        decode_schedule_env_quoted=shlex.quote(schedule),
        decode_route_exports=exports,
        input_len=args.input_len,
        output_len=args.output_len,
    )


def adb_shell_command(command: str, *, serial: str | None, use_su: bool, su_user: str) -> str:
    device_command = f"su {su_user} sh -c {shlex.quote(command)}" if use_su else command
    adb = ["adb"]
    if serial:
        adb.extend(["-s", serial])
    adb.extend(["shell", device_command])
    return shlex.join(adb)


def maybe_wrap_device_command(command: str, *, on_device: bool, args: argparse.Namespace) -> str:
    if not on_device:
        return command
    return adb_shell_command(command, serial=args.adb_serial, use_su=args.adb_su, su_user=args.adb_su_user)


def patch_template_script(template: Path, schedule: str, output_dir: Path, *, input_len: int, output_len: int) -> Path:
    text = template.read_text(encoding="utf-8")
    replacement = f'export GGML_HETERO_DYNAMIC_DECODE_SCHEDULE="{schedule}"'
    if "GGML_HETERO_DYNAMIC_DECODE_SCHEDULE=" in text:
        text = re.sub(r'export GGML_HETERO_DYNAMIC_DECODE_SCHEDULE="[^"]*"', replacement, text)
    else:
        text += "\n" + replacement + "\n"
    text = re.sub(r"(?<!\S)-n\s+\d+", f"-n {output_len}", text)
    text = re.sub(r"(?<!\S)-d\s+\d+", f"-d {input_len}", text)
    text = re.sub(r"^N_GEN=\$?\{N_GEN:-[^}]+\}", f"N_GEN=${{N_GEN:-{output_len}}}", text, flags=re.MULTILINE)
    text = re.sub(r"^N_DEPTH=\$?\{N_DEPTH:-[^}]+\}", f"N_DEPTH=${{N_DEPTH:-{input_len}}}", text, flags=re.MULTILINE)
    patched = output_dir / f"{template.stem}.patched.sh"
    patched.write_text(text, encoding="utf-8")
    patched.chmod(0o755)
    return patched


def read_charge(command: str) -> float:
    result = subprocess.run(command, shell=True, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"charge command failed with exit {result.returncode}: {result.stderr}")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", result.stdout)
    if not match:
        raise RuntimeError(f"charge command produced no numeric value: {result.stdout!r}")
    return float(match.group(0))


def charge_delta_to_mah(delta: float, unit: str) -> float:
    if unit == "mah":
        return delta
    if unit == "uah":
        return delta / 1000.0
    raise ValueError(f"unsupported charge unit: {unit}")


def write_run_log(path: Path, stdout: str, stderr: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = []
    if stdout:
        parts.append(stdout)
    if stderr:
        if parts and not parts[-1].endswith("\n"):
            parts[-1] += "\n"
        parts.append(stderr)
    path.write_text("".join(parts), encoding="utf-8")


def parse_seconds(stdout: str, stderr: str) -> float | None:
    text = stdout + "\n" + stderr
    patterns = [
        r"total\s+time\s*=\s*([0-9.]+)\s*s",
        r"elapsed(?:_ms)?\s*[=:]\s*([0-9.]+)\s*ms",
        r"elapsed(?:_s)?\s*[=:]\s*([0-9.]+)\s*s",
        r"total_latency_ms\s*[=:]\s*([0-9.]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        value = float(match.group(1))
        return value / 1000.0 if "ms" in pattern else value
    return None


def parse_throughput(stdout: str, stderr: str, output_len: int) -> float | None:
    text = stdout + "\n" + stderr
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:tok/s|tokens/s|t/s)", text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    seconds = parse_seconds(stdout, stderr)
    if seconds and seconds > 0 and output_len > 0:
        return output_len / seconds
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one e2e system-benefit measurement with fixed or planner-generated decode schedule.")
    parser.add_argument("--strategy", choices=["fixed", "auto"], required=True)
    parser.add_argument("--fixed-backend", default=None, help="Fixed state/backend, e.g. gpu_967, npu_burst, cpu_B2S4_4320000_3532800.")
    parser.add_argument("--plan-json", default=None, help="Existing simulate_system_benefit.py result JSON.")
    parser.add_argument("--profile", default="profiles/system_benefit_offline_profile.csv")
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=450)
    parser.add_argument("--slo-tbt-ms", type=float, default=100.0)
    parser.add_argument("--bucket-size", type=int, default=32)
    parser.add_argument("--context-match", choices=["exact", "nearest", "floor", "ceil"], default="nearest")
    parser.add_argument("--current-state", default=None)
    parser.add_argument("--run-command", required=True, help="Command to run. Supports {decode_schedule_env}, {decode_schedule_env_quoted}, {decode_route_exports}.")
    parser.add_argument("--run-command-on-device", action="store_true", help="Treat --run-command as an Android device shell command and wrap it with adb shell.")
    parser.add_argument("--command-template-file", default=None, help="Optional shell script to patch with the generated schedule and execute.")
    parser.add_argument("--mah-command", required=True, help="Command that prints current device battery charge before/after the run.")
    parser.add_argument("--mah-command-on-device", action="store_true", help="Treat --mah-command as an Android device shell command and wrap it with adb shell.")
    parser.add_argument("--adb-serial", default=None, help="Optional adb serial used by --*-command-on-device.")
    parser.add_argument("--adb-su", action="store_true", help="Run device commands as su <user> sh -c '<command>'.")
    parser.add_argument("--adb-su-user", default="0", help="su user for --adb-su; default is 0.")
    parser.add_argument("--charge-unit", choices=["uah", "mah"], default="uah", help="Unit printed by --mah-command. Android charge_counter is usually uAh.")
    parser.add_argument("--battery-voltage-v", type=float, default=3.85, help="Voltage used to convert charge delta to mJ: mAh*V*3600. Default: 3.85.")
    parser.add_argument("--log-output", default=None, help="Write full stdout/stderr here. Defaults next to --output with .log suffix.")
    parser.add_argument("--output", default="reports/e2e_system_benefit_result.json")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plan_payload: dict[str, Any] | None = None
    plan_path: Path | None = None
    if args.strategy == "fixed":
        if not args.fixed_backend:
            raise SystemExit("--fixed-backend is required for --strategy fixed")
        schedule = f"1:{route_spec_for_state(args.fixed_backend)}"
    elif args.plan_json:
        plan_path = Path(args.plan_json)
        schedule, plan_payload = load_plan_schedule(plan_path)
    else:
        schedule, plan_payload, plan_path = run_planner(args, output_path.parent)

    command = args.run_command
    if args.command_template_file:
        patched = patch_template_script(
            Path(args.command_template_file),
            schedule,
            output_path.parent,
            input_len=args.input_len,
            output_len=args.output_len,
        )
        command = f"bash {shlex.quote(str(patched))}"
    else:
        command = command_with_placeholders(command, schedule, args)
    command = maybe_wrap_device_command(command, on_device=args.run_command_on_device, args=args)
    mah_command = maybe_wrap_device_command(args.mah_command, on_device=args.mah_command_on_device, args=args)
    log_path = Path(args.log_output) if args.log_output else output_path.with_suffix(".log")
    if not log_path.is_absolute():
        log_path = ROOT / log_path

    env = os.environ.copy()
    env.update(env_for_schedule(schedule))

    if args.dry_run:
        print(f"strategy={args.strategy}")
        print(f"GGML_HETERO_DYNAMIC_DECODE_SCHEDULE={schedule}")
        print(f"mAh command: {mah_command}")
        print(f"would run: {command}")
        print(f"would log: {log_path}")
        print(f"would write: {output_path}")
        return 0

    charge_before = read_charge(mah_command)
    start = time.time()
    run = subprocess.run(command, cwd=ROOT, shell=True, text=True, capture_output=True, env=env, check=False)
    end = time.time()
    charge_after = read_charge(mah_command)
    write_run_log(log_path, run.stdout, run.stderr)
    charge_delta = charge_after - charge_before
    mah_delta = charge_delta_to_mah(charge_delta, args.charge_unit)
    energy_mj = mah_delta * args.battery_voltage_v * 3600.0
    throughput = parse_throughput(run.stdout, run.stderr, args.output_len)
    elapsed_s = parse_seconds(run.stdout, run.stderr) or (end - start)
    if throughput is None and elapsed_s > 0 and args.output_len > 0:
        throughput = args.output_len / elapsed_s

    payload = {
        "strategy": args.strategy,
        "schedule": schedule,
        "plan_json": str(plan_path) if plan_path else None,
        "plan": plan_payload,
        "charge_before": charge_before,
        "charge_after": charge_after,
        "charge_delta": charge_delta,
        "charge_unit": args.charge_unit,
        "mah_before": charge_delta_to_mah(charge_before, args.charge_unit),
        "mah_after": charge_delta_to_mah(charge_after, args.charge_unit),
        "energy_mah_delta": mah_delta,
        "battery_voltage_v": args.battery_voltage_v,
        "energy_mj": energy_mj,
        "elapsed_s": elapsed_s,
        "output_len": args.output_len,
        "throughput_tps": throughput,
        "run": {
            "command": command,
            "mah_command": mah_command,
            "returncode": run.returncode,
            "log_path": str(log_path),
        },
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {output_path}")
    print(f"throughput_tps={throughput:.6f}" if throughput is not None else "throughput_tps=NA")
    print(f"energy_mah_delta={mah_delta:.6f}")
    if energy_mj is not None:
        print(f"energy_mj={energy_mj:.6f}")
    return run.returncode


if __name__ == "__main__":
    raise SystemExit(main())
