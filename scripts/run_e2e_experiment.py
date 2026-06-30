#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hetero_route_spec import route_spec_for_fixed_state

DEFAULT_DEVICE = "192.168.1.148:37195"
DEFAULT_DEPLOY = "/data/local/tmp/llama-scheduler-plan-20260615"
DEFAULT_MODEL = "/data/local/tmp/models/Qwen2.5-3B-2K"
DEFAULT_SCHEDULE = (
    "1:opencl{gpu_freq_hz=967000000};"
    "64:cpu{threads=6,affinity=FC,cpu_policy0_freq_khz=3532800,cpu_policy6_freq_khz=4320000};"
    "128:qnn-npu{workpoint=burst};"
    "192:cpu{threads=6,affinity=FC,cpu_policy0_freq_khz=3532800,cpu_policy6_freq_khz=4320000};"
    "256:opencl{gpu_freq_hz=967000000};"
    "320:qnn-npu{workpoint=burst};"
    "384:opencl{gpu_freq_hz=967000000}"
)


@dataclass
class PhaseEvent:
    phase: str
    n_tokens: int
    total_wall_us: int
    label: str = ""
    reason: str = ""
    target: str = ""


@dataclass
class SampleMetrics:
    rep: int
    index: int
    prompt_tokens: int = 0
    gen_tokens: int = 0
    elapsed_ms: float | None = None
    tok_s: float | None = None
    phase_events: list[PhaseEvent] = field(default_factory=list)
    ttft_ms: float | None = None
    tbt_ms: list[float] = field(default_factory=list)
    selected_states: Counter[str] = field(default_factory=Counter)


def quote(value: str | os.PathLike[str]) -> str:
    return shlex.quote(str(value))


def adb_prefix(device: str) -> list[str]:
    return ["adb", "-s", device]


def run_cmd(
        cmd: list[str],
        *,
        input_text: str | None = None,
        check: bool = True,
        timeout_s: int | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        cmd,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=timeout_s,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"command failed with exit {result.returncode}: {shlex.join(cmd)}\n{result.stdout}"
        )
    return result


def adb_shell(args: argparse.Namespace, command: str, *, check: bool = True) -> str:
    if args.adb_su:
        shell_cmd = f"su {quote(args.adb_su_user)} sh -c {quote(command)}"
    else:
        shell_cmd = f"sh -c {quote(command)}"
    result = run_cmd(adb_prefix(args.device) + ["shell", shell_cmd], check=check)
    return result.stdout


def adb_shell_script(args: argparse.Namespace, script: str, *, timeout_s: int | None = None) -> subprocess.CompletedProcess[str]:
    if args.adb_su:
        shell_cmd = f"su {quote(args.adb_su_user)} sh -c {quote('sh -s')}"
    else:
        shell_cmd = "sh -s"
    return run_cmd(
        adb_prefix(args.device) + ["shell", shell_cmd],
        input_text=script,
        check=False,
        timeout_s=timeout_s,
    )


def adb_push(args: argparse.Namespace, local: Path, remote: str) -> None:
    if not local.exists():
        raise FileNotFoundError(local)
    remote_dir = os.path.dirname(remote)
    adb_shell(args, f"mkdir -p {quote(remote_dir)}")
    run_cmd(adb_prefix(args.device) + ["push", str(local), remote], check=True)


def maybe_push_runtime(args: argparse.Namespace) -> None:
    if not args.push_runtime:
        return
    bin_dir = Path(args.runtime_dir)
    files = [bin_dir / "llama-e2e-bench"]
    if args.push_shared_libs:
        files.extend(sorted(bin_dir.glob("*.so")))
    for path in files:
        adb_push(args, path, f"{args.deploy}/{path.name}")
    adb_shell(args, f"chmod 755 {quote(args.deploy + '/llama-e2e-bench')}")


def push_profile_if_needed(args: argparse.Namespace, profile: str | None, name: str) -> str | None:
    if not profile:
        return None
    path = Path(profile)
    if path.exists():
        remote = f"{args.deploy}/profiles/{name}"
        adb_push(args, path, remote)
        return remote
    return profile


def read_numeric_from_device(args: argparse.Namespace, command: str) -> float:
    text = adb_shell(args, command)
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        raise RuntimeError(f"device command produced no numeric value: {command!r}, output={text!r}")
    return float(match.group(0))


def read_text_from_device(args: argparse.Namespace, command: str, default: str = "") -> str:
    try:
        return adb_shell(args, command, check=False).strip() or default
    except Exception:
        return default


def charge_to_mah(value: float, unit: str) -> float:
    if unit == "mah":
        return value
    if unit == "uah":
        return value / 1000.0
    raise ValueError(f"unsupported charge unit: {unit}")


def parse_int(value: str | None, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(value)


def parse_float(value: str | None, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    return float(value)


def parse_kv_fields(line: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=([^,\s]+)", line):
        out[key] = value
    return out


def percentile_nearest_rank(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(p * len(ordered)))
    return ordered[min(len(ordered) - 1, rank - 1)]


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "p50": percentile_nearest_rank(values, 0.50),
        "p95": percentile_nearest_rank(values, 0.95),
        "p99": percentile_nearest_rank(values, 0.99),
        "min": min(values),
        "max": max(values),
    }


def state_label_from_route(route: str) -> str:
    route = route.strip()
    if not route or route == "<default>":
        return route or "<unknown>"
    lowered = route.lower()
    if lowered.startswith("opencl"):
        freq = re.search(r"gpu_freq_hz=(\d+)", route)
        if freq:
            return f"GPU_{int(freq.group(1)) // 1_000_000}MHz"
        return "GPU_opencl"
    if lowered.startswith("qnn-npu"):
        workpoint = re.search(r"(?:workpoint|qnn_workpoint)=([^,}]+)", route)
        ctx = re.search(r"qnn_context_size=(\d+)", route)
        label = f"NPU_{workpoint.group(1) if workpoint else 'default'}"
        if ctx:
            label += f"_ctx{ctx.group(1)}"
        return label
    if lowered.startswith("cpu"):
        threads = re.search(r"(?:threads|cpu_threads)=(\d+)", route)
        affinity = re.search(r"(?:affinity|cpu_affinity_mask)=([^,}]+)", route)
        p0 = re.search(r"cpu_policy0_freq_khz=(\d+)", route)
        p6 = re.search(r"cpu_policy6_freq_khz=(\d+)", route)
        parts = ["CPU"]
        if threads:
            parts.append(f"t{threads.group(1)}")
        if affinity:
            parts.append(f"aff{affinity.group(1)}")
        if p0:
            parts.append(f"p0_{p0.group(1)}")
        if p6:
            parts.append(f"p6_{p6.group(1)}")
        return "_".join(parts)
    return route


def parse_schedule_entries(schedule: str) -> list[tuple[int, str]]:
    entries: list[tuple[int, str]] = []
    for raw in schedule.split(";"):
        raw = raw.strip()
        if not raw:
            continue
        start, sep, route = raw.partition(":")
        if not sep:
            continue
        try:
            entries.append((int(start), route.strip()))
        except ValueError:
            continue
    entries.sort(key=lambda item: item[0])
    return entries


def distribution_from_schedule(schedule: str, gen_tokens: int) -> Counter[str]:
    dist: Counter[str] = Counter()
    entries = parse_schedule_entries(schedule)
    if not entries or gen_tokens <= 0:
        return dist
    for index, (start, route) in enumerate(entries):
        end = entries[index + 1][0] - 1 if index + 1 < len(entries) else gen_tokens
        lo = max(1, start)
        hi = min(gen_tokens, end)
        if hi >= lo:
            dist[state_label_from_route(route)] += hi - lo + 1
    return dist


def parse_log_metrics(log_text: str, slo_tbt_ms: float | None) -> tuple[list[SampleMetrics], dict[str, Any]]:
    samples: list[SampleMetrics] = []
    current: SampleMetrics | None = None
    pending: list[SampleMetrics] = []

    sample_begin_re = re.compile(r"SAMPLE_BEGIN\s+rep=(\d+)\s+sample=(\d+)/(\d+)")
    phase_re = re.compile(
        r"timing phase=(\w+)\s+n_tokens=(\d+)\s+.*?total_wall_us=(\d+)"
        r".*?\slabel=([^\s]+)\s+reason=([^\s]+)\s+target=([^\s]+)"
    )

    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        begin = sample_begin_re.search(line)
        if begin:
            current = SampleMetrics(rep=int(begin.group(1)), index=int(begin.group(2)))
            pending.append(current)
            continue

        phase_match = phase_re.search(line)
        if phase_match and current is not None:
            current.phase_events.append(PhaseEvent(
                phase=phase_match.group(1),
                n_tokens=int(phase_match.group(2)),
                total_wall_us=int(phase_match.group(3)),
                label=phase_match.group(4),
                reason=phase_match.group(5),
                target=phase_match.group(6),
            ))
            continue

        if line.startswith("sample,"):
            fields = parse_kv_fields(line)
            rep = parse_int(fields.get("rep"))
            index = parse_int(fields.get("index"))
            target = next((item for item in pending if item.rep == rep and item.index == index), None)
            if target is None:
                target = SampleMetrics(rep=rep, index=index)
                pending.append(target)
            target.prompt_tokens = parse_int(fields.get("prompt_tokens"))
            target.gen_tokens = parse_int(fields.get("gen_tokens"))
            target.elapsed_ms = parse_float(fields.get("elapsed_ms"))
            target.tok_s = parse_float(fields.get("tok_s"))
            samples.append(target)
            pending = [item for item in pending if item is not target]
            current = pending[-1] if pending else None

    for sample in samples:
        decode_events = [event for event in sample.phase_events if event.phase == "decode" and event.n_tokens == 1]
        generated_decode_events = decode_events[-sample.gen_tokens:] if sample.gen_tokens > 0 else []
        if generated_decode_events:
            first_generated = generated_decode_events[0]
            prefix_events: list[PhaseEvent] = []
            for event in sample.phase_events:
                prefix_events.append(event)
                if event is first_generated:
                    break
            sample.ttft_ms = sum(event.total_wall_us for event in prefix_events) / 1000.0
            sample.tbt_ms = [event.total_wall_us / 1000.0 for event in generated_decode_events[1:]]
            active_target = ""
            for event in generated_decode_events:
                if event.target and event.target != "<default>":
                    active_target = event.target
                target = active_target or event.target
                sample.selected_states[state_label_from_route(target)] += 1
        elif sample.elapsed_ms is not None and sample.gen_tokens == 1:
            sample.ttft_ms = sample.elapsed_ms

    all_tbt = [value for sample in samples for value in sample.tbt_ms]
    all_ttft = [sample.ttft_ms for sample in samples if sample.ttft_ms is not None]
    selected = Counter()
    for sample in samples:
        selected.update(sample.selected_states)

    violation_count = 0
    if slo_tbt_ms is not None:
        violation_count = sum(1 for value in all_tbt if value > slo_tbt_ms)
    summary = {
        "ttft_ms": summarize_values([float(value) for value in all_ttft]),
        "tbt_ms": summarize_values(all_tbt),
        "slo_tbt_ms": slo_tbt_ms,
        "slo_violation_count": violation_count,
        "slo_checked_count": len(all_tbt),
        "slo_violation_rate": (violation_count / len(all_tbt)) if all_tbt else None,
        "selected_state_distribution": {
            key: {
                "tokens": count,
                "fraction": count / sum(selected.values()) if selected else 0.0,
            }
            for key, count in selected.most_common()
        },
    }
    return samples, summary


def parse_run_summary(log_text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for line in log_text.splitlines():
        line = line.strip()
        if line.startswith("summary,"):
            fields = parse_kv_fields(line)
            out = {
                "samples": parse_int(fields.get("samples")),
                "reps": parse_int(fields.get("reps")),
                "total_prompt_tokens": parse_int(fields.get("total_prompt_tokens")),
                "total_gen_tokens": parse_int(fields.get("total_gen_tokens")),
                "elapsed_ms": parse_float(fields.get("elapsed_ms")),
                "tok_s": parse_float(fields.get("tok_s")),
            }
    return out


def build_device_script(args: argparse.Namespace, planner_profile_dev: str | None, prefill_profile_dev: str | None) -> str:
    if args.dataset:
        prompt_arg = "-p 0"
        depth_arg = "-d 0"
        dataset_args = f"--dataset {quote(args.dataset)} --limit {args.limit}"
        use_dataset_output_tokens = (
            args.dataset_output_tokens is True or
            (args.dataset_output_tokens is None and args.output_len is None)
        )
        if use_dataset_output_tokens:
            gen_arg = "-n 0"
            dataset_args += " --dataset-output-tokens"
        else:
            gen_arg = f"-n {args.output_len}"
    else:
        if args.input_len is None or args.output_len is None:
            raise ValueError("--input-len and --output-len are required without --dataset")
        # e2e-bench inserts one BOS prompt token for generation when -p 0, so
        # depth is reduced by one to approximate the requested total input.
        depth = max(0, args.input_len - 1)
        prompt_arg = "-p 0"
        depth_arg = f"-d {depth}"
        gen_arg = f"-n {args.output_len}"
        dataset_args = ""

    schedule = args.schedule
    if args.routing == "fixed":
        schedule = "1:" + route_spec_for_fixed_state(args.fixed_state)
    elif args.routing == "schedule" and not schedule:
        schedule = DEFAULT_SCHEDULE

    routing_env = ""
    planner_args = ""
    if args.routing in {"schedule", "fixed"}:
        routing_env = "\n".join([
            "export GGML_HETERO_DYNAMIC_MODE=phase",
            "export GGML_HETERO_DYNAMIC_PREFILL_ROUTE=qnn-npu",
            f"export GGML_HETERO_DYNAMIC_DECODE_SCHEDULE={quote(schedule)}",
            "unset GGML_HETERO_DECODE_ROUTE_SCHEDULE",
            "unset GGML_HETERO_DYNAMIC_DECODE_ROUTE",
        ])
    else:
        if planner_profile_dev is None:
            raise ValueError("--planner-profile is required for --routing planner")
        routing_env = "\n".join([
            "unset GGML_HETERO_DYNAMIC_MODE",
            "unset GGML_HETERO_DYNAMIC_PREFILL_ROUTE",
            "unset GGML_HETERO_DYNAMIC_DECODE_ROUTE",
            "unset GGML_HETERO_DYNAMIC_DECODE_SCHEDULE",
            "unset GGML_HETERO_DECODE_ROUTE_SCHEDULE",
        ])
        planner_args = (
            f" --planner-profile {quote(planner_profile_dev)}"
            f" --planner-ttft-slo-ms {args.ttft_slo_ms}"
            f" --planner-tbt-slo-ms {args.tbt_slo_ms}"
            f" --planner-bucket-size {args.bucket_size}"
            f" --planner-context-match {quote(args.context_match)}"
            f" --planner-max-context {args.planner_max_context}"
        )
        if prefill_profile_dev:
            planner_args += f" --planner-prefill-profile {quote(prefill_profile_dev)}"
        if args.input_len is not None and not args.dataset:
            planner_args += f" --planner-input-len {args.input_len}"
        if args.output_len is not None:
            planner_args += f" --planner-output-len {args.output_len}"

    device_arg = f"-dev {quote(args.device_arg)}" if args.device_arg else ""
    mmap_value = "1" if args.mmap else "0"
    warmup_arg = "" if args.warmup else "--no-warmup"
    wait_arg = "" if args.wait_start else "--no-wait-start"

    return f"""set -eu

DEPLOY={quote(args.deploy)}
MODEL={quote(args.model)}

cd "$DEPLOY"

if [ ! -x "$DEPLOY/llama-e2e-bench" ]; then
    echo "Missing executable: $DEPLOY/llama-e2e-bench" >&2
    exit 1
fi
if [ ! -f "$MODEL/ggml/weights.gguf" ]; then
    echo "Missing model: $MODEL/ggml/weights.gguf" >&2
    exit 1
fi
if [ ! -f "$MODEL/qnn/config.json" ]; then
    echo "Missing QNN config: $MODEL/qnn/config.json" >&2
    exit 1
fi

input keyevent WAKEUP >/dev/null 2>&1 || true
svc power stayon true >/dev/null 2>&1 || true

export LD_LIBRARY_PATH="$DEPLOY:$MODEL/qnn:${{LD_LIBRARY_PATH:-}}"
export ADSP_LIBRARY_PATH="$DEPLOY:$MODEL/qnn:${{ADSP_LIBRARY_PATH:-}}"

export GGML_QNN_AOT_CONFIG="$MODEL/qnn/config.json"
export GGML_QNN_AOT_MODEL_DIR="$MODEL/qnn"
export GGML_QNN_AOT_DISABLE_SEED_KV=1
export GGML_QNN_AOT_WRITE_GENERIC_KV=1
export GGML_QNN_HTP_WORKPOINT={quote(args.qnn_workpoint)}

export GGML_HETERO_GPU_MIN_FREQ_PATH=/sys/class/kgsl/kgsl-3d0/devfreq/min_freq
export GGML_HETERO_GPU_MAX_FREQ_PATH=/sys/class/kgsl/kgsl-3d0/devfreq/max_freq
export GGML_HETERO_GPU_CUR_FREQ_PATH=/sys/class/kgsl/kgsl-3d0/devfreq/cur_freq
export GGML_HETERO_CPU_POLICY0_MIN_FREQ_PATH=/sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq
export GGML_HETERO_CPU_POLICY0_MAX_FREQ_PATH=/sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq
export GGML_HETERO_CPU_POLICY0_CUR_FREQ_PATH=/sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq
export GGML_HETERO_CPU_POLICY6_MIN_FREQ_PATH=/sys/devices/system/cpu/cpufreq/policy6/scaling_min_freq
export GGML_HETERO_CPU_POLICY6_MAX_FREQ_PATH=/sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq
export GGML_HETERO_CPU_POLICY6_CUR_FREQ_PATH=/sys/devices/system/cpu/cpufreq/policy6/scaling_cur_freq
export GGML_HETERO_DYNAMIC_DECODE_CPU_STRICT=1
export GGML_HETERO_DYNAMIC_PRELOAD_QNN_DECODE=1
export GGML_HETERO_ENABLE_OPENCL_CPU_UMA_KV_HANDOFF=1
unset GGML_HETERO_DISABLE_CPU_OPENCL_UMA_KV_HANDOFF
export GGML_HETERO_DYNAMIC_TRACE=1
export GGML_HETERO_DYNAMIC_TRACE_TIMING=1
export GGML_HETERO_DYNAMIC_TRACE_TIMING_DETAIL=1
export LLAMA_E2E_BENCH_FAST_EXIT=1
export LLAMA_BENCH_FAST_EXIT=1

{routing_env}

echo "RUN_E2E_EXPERIMENT_BEGIN routing={args.routing}"
exec ./llama-e2e-bench -v -m "$MODEL/ggml/weights.gguf" -t {args.threads} \\
    {prompt_arg} {gen_arg} {depth_arg} -c {args.ctx_size} -b {args.batch_size} -ub {args.ubatch_size} \\
    -r {args.repetitions} {warmup_arg} --mmap {mmap_value} {wait_arg} {device_arg} \\
    {dataset_args} {planner_args}
"""


def sample_to_json(sample: SampleMetrics) -> dict[str, Any]:
    return {
        "rep": sample.rep,
        "index": sample.index,
        "prompt_tokens": sample.prompt_tokens,
        "gen_tokens": sample.gen_tokens,
        "elapsed_ms": sample.elapsed_ms,
        "tok_s": sample.tok_s,
        "ttft_ms": sample.ttft_ms,
        "tbt_ms": summarize_values(sample.tbt_ms),
        "selected_state_distribution": {
            key: {"tokens": count, "fraction": count / sum(sample.selected_states.values())}
            for key, count in sample.selected_states.most_common()
        } if sample.selected_states else {},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an adb e2e experiment and report energy, TTFT/TBT, SLO, and selected-state distribution."
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="adb serial, default: %(default)s")
    parser.add_argument("--adb-su", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adb-su-user", default="0")
    parser.add_argument("--deploy", default=DEFAULT_DEPLOY)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=None, help="Device-side dataset path. If omitted, use synthetic request mode.")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--dataset-output-tokens", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--input-len", type=int, default=None, help="Synthetic total input tokens; required without --dataset.")
    parser.add_argument("--output-len", type=int, default=None, help="Synthetic/fixed generated tokens.")
    parser.add_argument("--ctx-size", type=int, default=6144)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--ubatch-size", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wait-start", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mmap", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device-arg", default="", help="Optional e2e-bench -dev argument, e.g. qnn-npu/GPUOpenCL.")

    parser.add_argument("--routing", choices=["planner", "schedule", "fixed"], default="planner")
    parser.add_argument("--schedule", default=None, help="Decode schedule for --routing schedule.")
    parser.add_argument("--fixed-state", default="CPU_B2S4_4320000_3532800")
    parser.add_argument("--planner-profile", default=str(ROOT / "profiles" / "system_benefit_offline_profile.csv"))
    parser.add_argument("--planner-prefill-profile", default=None)
    parser.add_argument("--ttft-slo-ms", type=float, default=0.0)
    parser.add_argument("--tbt-slo-ms", type=float, default=100.0)
    parser.add_argument("--bucket-size", type=int, default=32)
    parser.add_argument("--context-match", choices=["exact", "floor", "ceil", "nearest"], default="nearest")
    parser.add_argument("--planner-max-context", type=int, default=6144)
    parser.add_argument("--qnn-workpoint", default="burst")

    parser.add_argument("--charge-path", default="/sys/class/power_supply/battery/charge_counter")
    parser.add_argument("--charge-unit", choices=["uah", "mah"], default="uah")
    parser.add_argument("--battery-status-path", default="/sys/class/power_supply/battery/status")
    parser.add_argument("--battery-capacity-path", default="/sys/class/power_supply/battery/capacity")

    parser.add_argument("--push-runtime", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--push-shared-libs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--runtime-dir", default=str(ROOT / "build-hz" / "bin"))
    parser.add_argument("--output-dir", default=str(ROOT / "reports" / "e2e_experiments"))
    parser.add_argument("--tag", default="")
    parser.add_argument("--timeout-s", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.dataset and (args.input_len is None or args.output_len is None):
        raise SystemExit("--input-len and --output-len are required when --dataset is not set")
    if args.dataset and args.dataset_output_tokens is False and args.output_len is None:
        raise SystemExit("--output-len is required with --dataset when --no-dataset-output-tokens is set")
    if args.routing == "fixed" and not args.fixed_state:
        raise SystemExit("--fixed-state is required for --routing fixed")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = f"-{args.tag}" if args.tag else ""
    base = out_dir / f"e2e{tag}-{stamp}"
    log_path = base.with_suffix(".log")
    json_path = base.with_suffix(".json")

    planner_profile_dev = None
    prefill_profile_dev = None
    if args.dry_run and args.routing == "planner":
        planner_profile_dev = args.planner_profile
        prefill_profile_dev = args.planner_prefill_profile
    elif args.routing == "planner":
        maybe_push_runtime(args)
        planner_profile_dev = push_profile_if_needed(args, args.planner_profile, "system_benefit_offline_profile.csv")
        prefill_profile_dev = push_profile_if_needed(args, args.planner_prefill_profile, "prefill_profile.csv")
    else:
        maybe_push_runtime(args)

    script = build_device_script(args, planner_profile_dev, prefill_profile_dev)
    if args.dry_run:
        print(script)
        return 0

    adb_shell(args, "input keyevent WAKEUP >/dev/null 2>&1 || true; svc power stayon true >/dev/null 2>&1 || true", check=False)
    status_before = read_text_from_device(args, f"cat {quote(args.battery_status_path)}", default="unknown")
    capacity_before = read_text_from_device(args, f"cat {quote(args.battery_capacity_path)}", default="unknown")
    charge_before_raw = read_numeric_from_device(args, f"cat {quote(args.charge_path)}")
    charge_before_mah = charge_to_mah(charge_before_raw, args.charge_unit)

    wall_start = time.time()
    result = adb_shell_script(args, script, timeout_s=args.timeout_s)
    wall_end = time.time()

    charge_after_raw = read_numeric_from_device(args, f"cat {quote(args.charge_path)}")
    charge_after_mah = charge_to_mah(charge_after_raw, args.charge_unit)
    status_after = read_text_from_device(args, f"cat {quote(args.battery_status_path)}", default="unknown")
    capacity_after = read_text_from_device(args, f"cat {quote(args.battery_capacity_path)}", default="unknown")

    log_text = result.stdout
    log_path.write_text(log_text, encoding="utf-8")

    samples, trace_summary = parse_log_metrics(log_text, args.tbt_slo_ms)
    run_summary = parse_run_summary(log_text)
    if not samples and run_summary:
        samples = []

    total_gen_tokens = int(run_summary.get("total_gen_tokens") or sum(sample.gen_tokens for sample in samples))
    if run_summary:
        request_count = int(run_summary.get("samples") or 1) * int(run_summary.get("reps") or args.repetitions)
    else:
        request_count = len(samples) or 1
    wall_elapsed_s = wall_end - wall_start
    measured_elapsed_s = (float(run_summary["elapsed_ms"]) / 1000.0) if run_summary.get("elapsed_ms") is not None else None

    consumed_mah = charge_before_mah - charge_after_mah
    request_energy_mah = consumed_mah / request_count if request_count > 0 else None
    energy_per_token_mah = consumed_mah / total_gen_tokens if total_gen_tokens > 0 else None
    active_power_ma = consumed_mah / (wall_elapsed_s / 3600.0) if wall_elapsed_s > 0 else None
    measured_run_power_ma = (
        consumed_mah / (measured_elapsed_s / 3600.0)
        if measured_elapsed_s is not None and measured_elapsed_s > 0
        else None
    )

    selected_dist = trace_summary["selected_state_distribution"]
    if not selected_dist and args.routing in {"schedule", "fixed"}:
        schedule = args.schedule or DEFAULT_SCHEDULE
        if args.routing == "fixed":
            schedule = "1:" + route_spec_for_fixed_state(args.fixed_state)
        fallback_dist = distribution_from_schedule(schedule, total_gen_tokens)
        selected_dist = {
            key: {
                "tokens": count,
                "fraction": count / sum(fallback_dist.values()) if fallback_dist else 0.0,
            }
            for key, count in fallback_dist.most_common()
        }

    warnings: list[str] = []
    if consumed_mah < 0:
        warnings.append("battery charge increased during the run; energy is negative, likely because the device was charging")
    if status_before.lower() not in {"discharging", "not charging", "unknown"}:
        warnings.append(f"battery status before run is {status_before!r}; mAh delta may include charging effects")

    payload = {
        "device": args.device,
        "routing": args.routing,
        "returncode": result.returncode,
        "command_succeeded": result.returncode == 0,
        "paths": {
            "log": str(log_path),
            "json": str(json_path),
        },
        "battery": {
            "charge_unit_raw": args.charge_unit,
            "charge_before_raw": charge_before_raw,
            "charge_after_raw": charge_after_raw,
            "charge_before_mAh": charge_before_mah,
            "charge_after_mAh": charge_after_mah,
            "status_before": status_before,
            "status_after": status_after,
            "capacity_before_percent": capacity_before,
            "capacity_after_percent": capacity_after,
        },
        "metrics": {
            "request_energy_mAh": request_energy_mah,
            "total_energy_mAh": consumed_mah,
            "energy_per_token_mAh": energy_per_token_mah,
            "active_power_mA": active_power_ma,
            "measured_run_power_mA": measured_run_power_ma,
            "active_elapsed_s": wall_elapsed_s,
            "measured_elapsed_s": measured_elapsed_s,
            "wall_elapsed_s": wall_elapsed_s,
            "request_count": request_count,
            "total_gen_tokens": total_gen_tokens,
            "TTFT_ms": trace_summary["ttft_ms"],
            "TBT_ms": trace_summary["tbt_ms"],
            "SLO_violation_rate": trace_summary["slo_violation_rate"],
            "SLO_violation_count": trace_summary["slo_violation_count"],
            "SLO_checked_count": trace_summary["slo_checked_count"],
            "selected_state_distribution": selected_dist,
        },
        "run_summary": run_summary,
        "samples": [sample_to_json(sample) for sample in samples],
        "warnings": warnings,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote_json={json_path}")
    print(f"wrote_log={log_path}")
    print(f"returncode={result.returncode}")
    print(f"request_energy_mAh={request_energy_mah if request_energy_mah is not None else 'NA'}")
    print(f"energy_per_token_mAh={energy_per_token_mah if energy_per_token_mah is not None else 'NA'}")
    print(f"active_power_mA={active_power_ma if active_power_ma is not None else 'NA'}")
    print(f"TTFT_ms={trace_summary['ttft_ms']}")
    print(f"TBT_ms={trace_summary['tbt_ms']}")
    print(f"SLO_violation_rate={trace_summary['slo_violation_rate']}")
    print(f"selected_state_distribution={selected_dist}")
    if warnings:
        print("warnings=" + "; ".join(warnings))

    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
