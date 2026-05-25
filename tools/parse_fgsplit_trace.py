#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path


PROFILE_FIELDS = [
    "date",
    "model",
    "mode",
    "backend_policy",
    "state_id",
    "workload_type",
    "context_len",
    "prompt_tokens",
    "decode_tokens",
    "layers",
    "rounds",
    "semantic_correctness_required",
    "semantic_correctness_status",
    "throughput_tps",
    "latency_per_token_ms",
    "latency_per_layer_ms",
    "active_power_mw",
    "power_std_mw",
    "energy_mj_per_token",
    "temp_avg_c",
    "temp_max_c",
    "gpu_freq_mhz",
    "cpu_freq_khz",
    "qnn_workpoint",
    "gpu_active_ratio",
    "npu_active_ratio",
    "qnn_proj_us",
    "gpu_attn_core_us",
    "qnn_ffn_us",
    "sync_qnn_to_gpu_us",
    "sync_gpu_to_qnn_us",
    "total_sync_us",
    "fallback_used",
    "support_status",
    "raw_log_path",
    "sample_path",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize fine-grained QNN/OpenCL split traces into the AGENTS.md CSV schema.")
    parser.add_argument("--bench-log", required=True, type=Path)
    parser.add_argument("--sample-log", "--samples", dest="sample_log", type=Path)
    parser.add_argument("--opencl-stage-profile", type=Path)
    parser.add_argument("--opencl-kernel-trace", type=Path)
    parser.add_argument("--command", type=Path)
    parser.add_argument("--local-command", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--summary-md", type=Path)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--date", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--git-commit", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--remote-output-dir", default="")
    parser.add_argument("--mode", default="synthetic")
    parser.add_argument("--backend-policy", default="fine_grained_qnn_gpu")
    parser.add_argument("--fg-route", default="")
    parser.add_argument("--state-id", default="")
    parser.add_argument("--workload-type", default="decode_like")
    parser.add_argument("--context-len", default="")
    parser.add_argument("--prompt-tokens", default="")
    parser.add_argument("--decode-tokens", default="")
    parser.add_argument("--layers", default="")
    parser.add_argument("--rounds", default="")
    parser.add_argument("--semantic-correctness-required", default="0")
    parser.add_argument("--semantic-correctness-status", default="not_required")
    parser.add_argument("--gpu-freq-mhz", default="")
    parser.add_argument("--cpu-freq-khz", default="")
    parser.add_argument("--qnn-workpoint", default="")
    parser.add_argument("--temp-limit-c", default="")
    parser.add_argument("--cooldown-temp-c", default="")
    parser.add_argument("--raw-log-path", default="")
    parser.add_argument("--sample-path", default="")
    return parser.parse_args()


def read_text(path):
    if path is None or not path.exists():
        return ""
    return path.read_text(errors="ignore")


def safe_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in {"NA", "N/A", "NULL", "NONE"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def safe_int(value):
    number = safe_float(value)
    if number is None:
        return None
    return int(number)


def fmt(value, digits=3):
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def mean(values):
    return statistics.mean(values) if values else None


def stdev(values):
    return statistics.stdev(values) if len(values) >= 2 else None


def parse_kv_tokens(line):
    result = {}
    for match in re.finditer(r"([A-Za-z0-9_./:-]+)=([^ \t\r\n]+)", line):
        result[match.group(1)] = match.group(2).strip()
    return result


def parse_gpu_busy_value(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    percent = text.endswith("%")
    if percent:
        text = text[:-1].strip()

    tokens = re.findall(r"-?[0-9]+(?:\.[0-9]+)?", text)
    if not tokens:
        return None

    first = safe_float(tokens[0])
    if first is None:
        return None

    if len(tokens) >= 2 and not percent:
        second = safe_float(tokens[1])
        if second and second > 0:
            return max(0.0, min(1.0, first / second))

    if percent or first > 1.0:
        return first / 100.0
    return first


def parse_fg_route(route_text):
    text = (route_text or "").strip().lower()
    default_route = {
        "attn_proj": "qnn-npu",
        "attn_core": "opencl",
        "attn_out": "cpu",
        "ffn": "qnn-npu",
        "output": "cpu",
    }
    if not text:
        return default_route
    if "=" not in text:
        return {stage: text for stage in default_route}

    route = dict(default_route)
    for token in text.split(","):
        token = token.strip()
        if not token or "=" not in token:
            continue
        stage, backend = token.split("=", 1)
        stage = stage.strip().lower()
        backend = backend.strip().lower()
        if stage:
            route[stage] = backend
    return route


def backend_is_qnn(backend):
    return "qnn" in (backend or "").lower()


def backend_is_opencl(backend):
    text = (backend or "").lower()
    return "opencl" in text or "gpu" in text


def parse_measured_windows(text):
    start_re = re.compile(
        r"llama-bench:\s+benchmark\s+([0-9]+)/([0-9]+):\s+round\s+([0-9]+)/([0-9]+):\s+starting")
    finish_re = re.compile(
        r"llama-bench:\s+benchmark\s+([0-9]+)/([0-9]+):\s+round\s+([0-9]+)/([0-9]+):\s+finished")
    weight_sync_re = re.compile(r"\bFG_SYNC_TRACE\b.*\btensor=blk\.[^\s]*weight\b")
    bad_re = re.compile(
        r"unmatched cgraph|rejecting unmatched|fallback to JIT|failed to run|\berror:\s",
        re.IGNORECASE)

    result = {
        "measured_windows": 0,
        "measured_incomplete_windows": 0,
        "measured_context_loads": 0,
        "measured_graph_loads": 0,
        "measured_ensure_graph_loaded": 0,
        "measured_graph_cache_hits": 0,
        "measured_graph_cache_misses": 0,
        "measured_weight_syncs": 0,
        "measured_weight_sync_bytes": 0,
        "measured_weight_sync_us": 0.0,
        "measured_bad_lines": 0,
        "measured_bad_examples": [],
    }

    active = None
    for lineno, line in enumerate(text.splitlines(), 1):
        start = start_re.search(line)
        if start:
            active = {
                "benchmark_index": safe_int(start.group(1)),
                "benchmark_count": safe_int(start.group(2)),
                "round_index": safe_int(start.group(3)),
                "round_count": safe_int(start.group(4)),
                "start_line": lineno,
            }
            result["measured_windows"] += 1
            continue

        if active is not None:
            if "AOT_LOAD_TRACE" in line:
                kv = parse_kv_tokens(line)
                kind = kv.get("kind", "")
                if kind == "context":
                    result["measured_context_loads"] += 1
                    if len(result["measured_bad_examples"]) < 6:
                        result["measured_bad_examples"].append(
                            f"line {lineno}: measured QNN context load: {line.strip()}")
                elif kind == "graph":
                    result["measured_graph_loads"] += 1
                    if len(result["measured_bad_examples"]) < 6:
                        result["measured_bad_examples"].append(
                            f"line {lineno}: measured QNN graph load: {line.strip()}")
                elif kind == "ensure_graph_loaded":
                    result["measured_ensure_graph_loaded"] += 1
                    graph_hit = kv.get("graph_cache_hit") == "1"
                    context_hit = kv.get("context_cache_hit") == "1"
                    graph_create_us = safe_float(kv.get("graph_create_us"))
                    if graph_hit and context_hit and (graph_create_us is None or graph_create_us == 0):
                        result["measured_graph_cache_hits"] += 1
                    else:
                        result["measured_graph_cache_misses"] += 1
                        if len(result["measured_bad_examples"]) < 6:
                            result["measured_bad_examples"].append(
                                f"line {lineno}: measured QNN graph cache miss: {line.strip()}")

            if weight_sync_re.search(line):
                result["measured_weight_syncs"] += 1
                kv = parse_kv_tokens(line)
                result["measured_weight_sync_bytes"] += safe_int(kv.get("bytes")) or 0
                result["measured_weight_sync_us"] += safe_float(kv.get("us")) or 0.0
                if len(result["measured_bad_examples"]) < 6:
                    result["measured_bad_examples"].append(
                        f"line {lineno}: measured weight sync: {line.strip()}")

            if bad_re.search(line):
                result["measured_bad_lines"] += 1
                if len(result["measured_bad_examples"]) < 6:
                    result["measured_bad_examples"].append(
                        f"line {lineno}: measured failure/fallback signal: {line.strip()}")

        if active is not None and finish_re.search(line):
            active = None

    if active is not None:
        result["measured_incomplete_windows"] += 1

    return result


def parse_llama_bench_jsonl(text):
    tests = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            item = json.loads(stripped.rstrip(","))
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and ("avg_ts" in item or "samples_ts" in item):
            tests.append(item)
    if tests:
        return tests[-1]
    return {}


def parse_llama_bench_markdown_tps(text):
    # Fallback for default markdown output lines ending with "NN.NN +/- MM.MM".
    candidates = []
    for line in text.splitlines():
        if "|" not in line:
            continue
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:±|\+/-)\s*[0-9]+(?:\.[0-9]+)?\s*\|?\s*$", line)
        if match:
            candidates.append(float(match.group(1)))
    return candidates[-1] if candidates else None


def parse_traces(text):
    durations = {
        "qnn_proj": [],
        "gpu_attn_core": [],
        "qnn_ffn": [],
    }
    sync_qnn_to_gpu = []
    sync_gpu_to_qnn = []
    qnn_events = 0
    gpu_events = 0
    qnn_proj_events = 0
    qnn_ffn_events = 0
    gpu_attn_core_events = 0
    observed_layers = set()

    for line in text.splitlines():
        if "FG_TRACE" in line:
            kv = parse_kv_tokens(line)
            subgraph = kv.get("subgraph", "")
            ok = kv.get("ok", "1") != "0"
            duration = safe_float(kv.get("duration_us"))
            if duration is None:
                begin = safe_float(kv.get("begin_us"))
                end = safe_float(kv.get("end_us"))
                if begin is not None and end is not None:
                    duration = end - begin
            if duration is not None and subgraph in durations:
                durations[subgraph].append(duration)
            if ok and subgraph == "qnn_proj":
                qnn_proj_events += 1
            if ok and subgraph == "qnn_ffn":
                qnn_ffn_events += 1
            if ok and subgraph == "gpu_attn_core":
                gpu_attn_core_events += 1
            layer = safe_int(kv.get("layer"))
            if layer is not None and layer >= 0:
                observed_layers.add(layer)
            backend = kv.get("backend", "")
            if backend.startswith("qnn"):
                qnn_events += 1
            if "gpu" in backend.lower() or "opencl" in backend.lower():
                gpu_events += 1

        if "FG_SYNC_TRACE" in line:
            kv = parse_kv_tokens(line)
            duration = safe_float(kv.get("us") or kv.get("duration_us"))
            if duration is None:
                continue
            src = kv.get("from", "")
            dst = kv.get("to", "")
            if src == "qnn" and dst == "gpu":
                sync_qnn_to_gpu.append(duration)
            if src == "gpu" and dst == "qnn":
                sync_gpu_to_qnn.append(duration)

        aot_execute = re.search(r"\[aot\].*execute (attn_proj|ffn) graph", line)
        if aot_execute:
            qnn_events += 1
            if aot_execute.group(1) == "attn_proj":
                qnn_proj_events += 1
            if aot_execute.group(1) == "ffn":
                qnn_ffn_events += 1
        if "OPENCL_KERNEL_TRACE" in line and "total=" in line:
            total = safe_int(parse_kv_tokens(line).get("total"))
            if total:
                gpu_events += total
        if "OPENCL_KERNEL_TRACE" in line and "stage=ATTN_CORE" in line:
            count = safe_int(parse_kv_tokens(line).get("count"))
            if count:
                gpu_attn_core_events += count

    return {
        "qnn_proj_us": mean(durations["qnn_proj"]),
        "gpu_attn_core_us": mean(durations["gpu_attn_core"]),
        "qnn_ffn_us": mean(durations["qnn_ffn"]),
        "sync_qnn_to_gpu_us": mean(sync_qnn_to_gpu),
        "sync_gpu_to_qnn_us": mean(sync_gpu_to_qnn),
        "total_sync_us": sum(v for v in [mean(sync_qnn_to_gpu), mean(sync_gpu_to_qnn)] if v is not None)
                         if (sync_qnn_to_gpu or sync_gpu_to_qnn) else None,
        "qnn_events": qnn_events,
        "gpu_events": gpu_events,
        "qnn_proj_events": qnn_proj_events,
        "qnn_ffn_events": qnn_ffn_events,
        "gpu_attn_core_events": gpu_attn_core_events,
        "observed_layers": len(observed_layers) if observed_layers else None,
        "observed_layer_ids": sorted(observed_layers),
    }


def parse_opencl_stage_profile(path):
    if path is None or not path.exists():
        return {}
    result = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            stage = row.get("stage", "")
            if stage == "ATTN_CORE":
                exec_total_ms = safe_float(row.get("exec_total_ms"))
                count = safe_float(row.get("count"))
                avg_ms = safe_float(row.get("exec_avg_ms"))
                if avg_ms is None and exec_total_ms is not None and count:
                    avg_ms = exec_total_ms / count
                if avg_ms is not None:
                    result["gpu_attn_core_us"] = avg_ms * 1000.0
                    result["gpu_events"] = int(count or 0)
                    result["gpu_attn_core_events"] = int(count or 0)
            elif stage == "TOTAL":
                result["opencl_total_ms"] = safe_float(row.get("exec_total_ms"))
    return result


def parse_opencl_kernel_trace(path):
    if path is None or not path.exists():
        return {}
    result = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("kind") == "stage" and row.get("name") == "ATTN_CORE":
                count = int(safe_float(row.get("count")) or 0)
                result["gpu_events"] = count
                result["gpu_attn_core_events"] = count
            if row.get("kind") == "total" and row.get("name") == "all":
                result["opencl_total_kernels"] = int(safe_float(row.get("count")) or 0)
    return result


def parse_sample_log(path):
    if path is None or not path.exists():
        return {}
    power_values = []
    temp_values = []
    gpu_busy_values = []
    gpu_clock_values = []

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            power = safe_float(row.get("power_mw") or row.get("power_index_mw"))
            if power is None:
                voltage = safe_float(
                    row.get("voltage_uv") or
                    row.get("batt_voltage_now") or
                    row.get("battery_voltage_uv") or
                    row.get("usb_voltage_now"))
                current = safe_float(
                    row.get("current_ua") or
                    row.get("current_raw") or
                    row.get("batt_current_now") or
                    row.get("battery_current_ua") or
                    row.get("usb_current_now"))
                if voltage is not None and current is not None:
                    power = abs(voltage * current) / 1e9
            if power is not None and power > 0:
                power_values.append(power)

            temp = safe_float(row.get("temp_c") or row.get("battery_temp_c"))
            if temp is None:
                raw_temp = safe_float(row.get("batt_temp") or row.get("temp_raw"))
                if raw_temp is not None:
                    # Android battery temp is usually deci-C; thermal-zone temp is usually milli-C.
                    if abs(raw_temp) > 1000:
                        temp = raw_temp / 1000.0
                    elif abs(raw_temp) > 100:
                        temp = raw_temp / 10.0
                    else:
                        temp = raw_temp
            if temp is not None:
                temp_values.append(temp)

            gpu_busy = parse_gpu_busy_value(row.get("gpu_busy_pct") or row.get("gpu_busy"))
            if gpu_busy is not None:
                gpu_busy_values.append(gpu_busy)

            gpu_clock_text = str(row.get("gpu_clock_hz") or row.get("gpu_clock") or "").split()
            gpu_clock = safe_float(gpu_clock_text[0]) if gpu_clock_text else None
            if gpu_clock is not None:
                gpu_clock_values.append(gpu_clock / 1e6 if gpu_clock > 10000 else gpu_clock)

    return {
        "active_power_mw": mean(power_values),
        "power_std_mw": stdev(power_values),
        "temp_avg_c": mean(temp_values),
        "temp_max_c": max(temp_values) if temp_values else None,
        "gpu_active_ratio": mean(gpu_busy_values),
        "actual_gpu_freq_mhz": mean(gpu_clock_values),
        "sample_count": max(len(power_values), len(temp_values), len(gpu_busy_values)),
    }


def detect_fallback(text):
    patterns = [
        r"fallback_used=1",
        r"runtime fallback",
        r"falling back",
        r"fallback to (CPU|GPU|OpenCL|qnn-npu|JIT)",
        r"unsupported-cpu-fallback",
        r"replay fallback",
    ]
    for line in text.splitlines():
        lowered = line.lower()
        if "fallback" in lowered and "disabled" in lowered:
            continue
        if any(re.search(pattern, line, re.IGNORECASE) for pattern in patterns):
            return "1"
    return "0"


def detect_support_status(
        text,
        qnn_events,
        gpu_events,
        backend_policy,
        fg_route="",
        qnn_proj_events=0,
        qnn_ffn_events=0,
        gpu_attn_core_events=0):
    exit_codes = [safe_int(code) for code in re.findall(r"FG_RUN_EXIT_CODE=([0-9]+)", text)]
    if any(code not in (None, 0) for code in exit_codes):
        return "failed"
    if re.search(r"unsupported.*shape|shape.*unsupported", text, re.IGNORECASE):
        return "unsupported_by_shape"
    if re.search(r"failed to initialize|cannot initialize|runtime rejects", text, re.IGNORECASE):
        return "unsupported_by_runtime"
    for line in text.splitlines():
        if "failed to unregister shared buffer view" in line:
            continue
        if re.search(r"segmentation fault|failed to run|\berror:\s|GGML_ABORT", line, re.IGNORECASE):
            return "failed"
    if backend_policy == "single_gpu_opencl":
        if gpu_events > 0:
            return "ok"
        return "unsupported_by_gpu_kernel"
    if backend_policy == "single_qnn_npu":
        if qnn_events > 0:
            return "ok"
        return "unsupported_by_qnn_graph"

    route = parse_fg_route(fg_route)
    if backend_is_qnn(route.get("attn_proj")) and qnn_proj_events <= 0:
        return "unsupported_by_qnn_graph"
    if backend_is_qnn(route.get("ffn")) and qnn_ffn_events <= 0:
        return "unsupported_by_qnn_graph"
    if backend_is_opencl(route.get("attn_core")) and gpu_attn_core_events <= 0:
        return "unsupported_by_gpu_kernel"

    requires_qnn = any(backend_is_qnn(backend) for backend in route.values())
    requires_opencl = any(backend_is_opencl(backend) for backend in route.values())
    if requires_qnn and qnn_events <= 0:
        return "unsupported_by_qnn_graph"
    if requires_opencl and gpu_events <= 0:
        return "unsupported_by_gpu_kernel"
    if qnn_events > 0 and gpu_events > 0:
        return "ok"
    if re.search(r"missing .*qnn|missing .*graph", text, re.IGNORECASE):
        return "unsupported_by_qnn_graph"
    if re.search(r"unsupported.*opencl|opencl.*unsupported|GPUOpenCL backend not available", text, re.IGNORECASE):
        return "unsupported_by_gpu_kernel"
    if requires_qnn and qnn_events == 0:
        return "unsupported_by_qnn_graph"
    if requires_opencl and gpu_events == 0:
        return "unsupported_by_gpu_kernel"
    return "failed"


def write_profile_csv(path, row, append):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = True
    mode = "w"
    if append and path.exists() and path.stat().st_size > 0:
        write_header = False
        mode = "a"
    with path.open(mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROFILE_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_summary(path, row, args, evidence):
    remote_command_text = read_text(args.command).strip() if args.command else ""
    local_command_text = read_text(args.local_command).strip() if args.local_command else ""
    if not local_command_text:
        local_command_text = remote_command_text

    model_path = args.model_path or row["model"]
    git_commit = args.git_commit.strip() if args.git_commit else "unknown"
    temp_range = "not_available"
    if row["temp_avg_c"] or row["temp_max_c"]:
        temp_range = f"avg={row['temp_avg_c'] or 'NA'} C max={row['temp_max_c'] or 'NA'} C"

    if args.backend_policy == "single_gpu_opencl":
        experiment_goal = "Measure a single-backend OpenCL baseline with the same harness, sampling, and output schema."
    elif args.backend_policy == "single_qnn_npu":
        experiment_goal = "Measure a single-backend QNN/NPU baseline with the same harness, sampling, and output schema when supported by the available QNN graphs."
    else:
        experiment_goal = "Measure a synthetic/approximate fine-grained QNN projection + OpenCL attention core + QNN FFN schedule."

    path.parent.mkdir(parents=True, exist_ok=True)
    requested_layers = safe_int(args.layers)
    observed_layers = evidence.get("observed_layers")
    measured_quality_status = evidence.get("measured_quality_status", "not_available")
    if measured_quality_status != "ok" and measured_quality_status != "not_available":
        data_quality_judgment = (
            f"{measured_quality_status}: measured execution included graph/model loading, "
            "graph cache misses, or stage-weight synchronization; row is retained for debugging only.")
    elif requested_layers and observed_layers and requested_layers != observed_layers:
        data_quality_judgment = (
            "workload_shape_mismatch: requested layer count differs from observed FG_TRACE layer count; "
            "row is retained but should not be compared as the requested workload.")
    else:
        data_quality_judgment = (
            "smoke_only: sufficient for harness validation if QNN and GPU events are present with fallback_used=0; "
            "insufficient for paper claims or baseline comparison.")

    lines = [
        "# FGSplit Synthetic",
        "",
        "## Experiment Goal",
        experiment_goal,
        "",
        "## Exact Command",
        "```bash",
        local_command_text,
        "```",
        "",
        "## Remote Command",
        "```bash",
        remote_command_text,
        "```",
        "",
        "## Git Commit",
        f"`{git_commit}`",
        "",
        "## Device and Model",
        f"- device: `{args.device or 'unknown'}`",
        f"- model: `{model_path or 'unknown'}`",
        f"- output_dir: `{args.output_dir or 'unknown'}`",
        f"- remote_output_dir: `{args.remote_output_dir or 'unknown'}`",
        "",
        "## Backend Policy",
        f"- mode: `{row['mode']}`",
        f"- backend_policy: `{row['backend_policy']}`",
        f"- state_id: `{row['state_id']}`",
        f"- qnn_workpoint: `{row['qnn_workpoint']}`",
        f"- gpu_freq_mhz: `{row['gpu_freq_mhz']}`",
        "",
        "## Workload Shape",
        f"- context_len: `{row['context_len']}`",
        f"- prompt_tokens: `{row['prompt_tokens']}`",
        f"- decode_tokens: `{row['decode_tokens']}`",
        f"- layers: `{row['layers']}`",
        f"- requested_layers: `{args.layers or 'unknown'}`",
        f"- observed_fg_layers: `{observed_layers if observed_layers is not None else 'unknown'}`",
        f"- rounds: `{row['rounds']}`",
        "",
        "## Temperature Range",
        f"- measured: `{temp_range}`",
        f"- temp_limit_c: `{args.temp_limit_c or 'unknown'}`",
        f"- cooldown_temp_c: `{args.cooldown_temp_c or 'unknown'}`",
        "",
        "## Main Result",
        "| throughput_tps | active_power_mw | energy_mj_per_token | support_status | fallback_used |",
        "| ---: | ---: | ---: | --- | ---: |",
        f"| {row['throughput_tps']} | {row['active_power_mw']} | {row['energy_mj_per_token']} | {row['support_status']} | {row['fallback_used']} |",
        "",
        "## Power Comparison Against Single GPU and Single QNN",
        "not_computed_in_per_run_summary: compare rows from the aggregate CSV after the matching baseline and fine-grained runs are available.",
        "",
        "## Throughput Comparison Against Single GPU and Single QNN",
        "not_computed_in_per_run_summary: compare rows from the aggregate CSV after the matching baseline and fine-grained runs are available.",
        "",
        "## Energy Per Token Comparison",
        "not_computed_in_per_run_summary: compare rows from the aggregate CSV after the matching baseline and fine-grained runs are available.",
        "",
        "## Subgraph Timing",
        "| qnn_proj_us | gpu_attn_core_us | qnn_ffn_us | sync_qnn_to_gpu_us | sync_gpu_to_qnn_us | total_sync_us |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {row['qnn_proj_us']} | {row['gpu_attn_core_us']} | {row['qnn_ffn_us']} | {row['sync_qnn_to_gpu_us']} | {row['sync_gpu_to_qnn_us']} | {row['total_sync_us']} |",
        "",
        "## Synchronization Overhead",
        f"- sync_qnn_to_gpu_us: `{row['sync_qnn_to_gpu_us']}`",
        f"- sync_gpu_to_qnn_us: `{row['sync_gpu_to_qnn_us']}`",
        f"- total_sync_us: `{row['total_sync_us']}`",
        "",
        "## Fallback or Unsupported Conditions",
        f"- support_status: `{row['support_status']}`",
        f"- fallback_used: `{row['fallback_used']}`",
        "",
        "## Measured Execution Quality",
        f"- measured_quality_status: `{measured_quality_status}`",
        f"- measured_windows: `{evidence.get('measured_windows', 0)}`",
        f"- measured_incomplete_windows: `{evidence.get('measured_incomplete_windows', 0)}`",
        f"- measured_context_loads: `{evidence.get('measured_context_loads', 0)}`",
        f"- measured_graph_loads: `{evidence.get('measured_graph_loads', 0)}`",
        f"- measured_ensure_graph_loaded: `{evidence.get('measured_ensure_graph_loaded', 0)}`",
        f"- measured_graph_cache_hits: `{evidence.get('measured_graph_cache_hits', 0)}`",
        f"- measured_graph_cache_misses: `{evidence.get('measured_graph_cache_misses', 0)}`",
        f"- measured_weight_syncs: `{evidence.get('measured_weight_syncs', 0)}`",
        f"- measured_weight_sync_bytes: `{evidence.get('measured_weight_sync_bytes', 0)}`",
        f"- measured_weight_sync_us: `{fmt(evidence.get('measured_weight_sync_us'))}`",
        f"- measured_bad_lines: `{evidence.get('measured_bad_lines', 0)}`",
        f"- measured_bad_examples: `{'; '.join(evidence.get('measured_bad_examples', [])) or 'none'}`",
        "",
        "## Data Quality",
        f"- semantic_correctness_status: `{row['semantic_correctness_status']}`",
        f"- qnn_events: `{evidence.get('qnn_events', 0)}`",
        f"- qnn_proj_events: `{evidence.get('qnn_proj_events', 0)}`",
        f"- qnn_ffn_events: `{evidence.get('qnn_ffn_events', 0)}`",
        f"- gpu_events: `{evidence.get('gpu_events', 0)}`",
        f"- gpu_attn_core_events: `{evidence.get('gpu_attn_core_events', 0)}`",
        f"- sample_count: `{evidence.get('sample_count', 0)}`",
        "- interpretation: smoke-test data only; do not use for paper claims until reviewed and repeated.",
        "",
        "## Data Quality Judgment",
        data_quality_judgment,
        "",
        "## Whether The Result Supports The Insight",
        "not_evaluated: Task 1 smoke data must not be used to claim a power-performance insight.",
        "",
        "## Raw Evidence",
        f"- raw_log_path: `{row['raw_log_path']}`",
        f"- sample_path: `{row['sample_path']}`",
    ]
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    text = read_text(args.bench_log)
    bench = parse_llama_bench_jsonl(text)
    trace = parse_traces(text)
    measured = parse_measured_windows(text)
    opencl_stage = parse_opencl_stage_profile(args.opencl_stage_profile)
    opencl_kernel = parse_opencl_kernel_trace(args.opencl_kernel_trace)
    samples = parse_sample_log(args.sample_log)

    if opencl_kernel.get("gpu_events"):
        trace["gpu_events"] = max(trace["gpu_events"], opencl_kernel["gpu_events"])
    if opencl_kernel.get("gpu_attn_core_events"):
        trace["gpu_attn_core_events"] = max(
            trace["gpu_attn_core_events"], opencl_kernel["gpu_attn_core_events"])
    if opencl_stage.get("gpu_events"):
        trace["gpu_events"] = max(trace["gpu_events"], opencl_stage["gpu_events"])
    if opencl_stage.get("gpu_attn_core_events"):
        trace["gpu_attn_core_events"] = max(
            trace["gpu_attn_core_events"], opencl_stage["gpu_attn_core_events"])
    if opencl_stage.get("gpu_attn_core_us") is not None:
        trace["gpu_attn_core_us"] = opencl_stage["gpu_attn_core_us"]

    throughput = safe_float(bench.get("avg_ts"))
    if throughput is None:
        throughput = parse_llama_bench_markdown_tps(text)

    avg_ns = safe_float(bench.get("avg_ns"))
    decode_tokens = safe_int(args.decode_tokens) or safe_int(bench.get("n_gen")) or 0
    prompt_tokens = safe_int(args.prompt_tokens) or safe_int(bench.get("n_prompt")) or 0
    total_tokens = prompt_tokens + decode_tokens
    if throughput is None and avg_ns is not None and total_tokens > 0:
        throughput = 1e9 * total_tokens / avg_ns

    latency_per_token_ms = 1000.0 / throughput if throughput and throughput > 0 else None
    requested_layers = safe_int(args.layers) or 0
    observed_layers = safe_int(trace.get("observed_layers"))
    layers = observed_layers or requested_layers
    latency_per_layer_ms = latency_per_token_ms / layers if latency_per_token_ms and layers > 0 else None
    active_power = samples.get("active_power_mw")
    energy = active_power / throughput if active_power is not None and throughput and throughput > 0 else None

    qnn_duration_total = sum(v for v in [
        trace.get("qnn_proj_us"),
        trace.get("qnn_ffn_us"),
    ] if v is not None)
    if qnn_duration_total and latency_per_token_ms:
        npu_active_ratio = min(1.0, (qnn_duration_total * max(layers, 1)) / (latency_per_token_ms * 1000.0))
    else:
        npu_active_ratio = None

    gpu_freq_mhz = args.gpu_freq_mhz
    if not gpu_freq_mhz and samples.get("actual_gpu_freq_mhz") is not None:
        gpu_freq_mhz = fmt(samples.get("actual_gpu_freq_mhz"), 0)

    date_value = args.date or str(bench.get("test_time") or "")
    if not date_value:
        date_value = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    model = args.model or str(bench.get("model_filename") or "") or args.model_path
    raw_log_path = args.raw_log_path or str(args.bench_log)
    sample_path = args.sample_path or (str(args.sample_log) if args.sample_log else "")

    qnn_events = int(trace.get("qnn_events") or 0)
    gpu_events = int(trace.get("gpu_events") or 0)
    qnn_proj_events = int(trace.get("qnn_proj_events") or 0)
    qnn_ffn_events = int(trace.get("qnn_ffn_events") or 0)
    gpu_attn_core_events = int(trace.get("gpu_attn_core_events") or 0)
    support_status = detect_support_status(
        text,
        qnn_events,
        gpu_events,
        args.backend_policy,
        args.fg_route,
        qnn_proj_events,
        qnn_ffn_events,
        gpu_attn_core_events)
    if (support_status == "ok" and args.backend_policy == "fine_grained_qnn_gpu" and
            requested_layers and observed_layers and requested_layers != observed_layers):
        support_status = "unsupported_by_shape"
    measured_quality_status = "not_available"
    if measured.get("measured_windows", 0) > 0:
        measured_quality_status = "ok"
        if measured.get("measured_context_loads", 0) > 0 or measured.get("measured_graph_loads", 0) > 0:
            measured_quality_status = "failed_measured_loading"
        elif measured.get("measured_graph_cache_misses", 0) > 0:
            measured_quality_status = "failed_measured_graph_cache"
        elif measured.get("measured_weight_syncs", 0) > 0:
            measured_quality_status = "failed_measured_weight_sync"
        elif measured.get("measured_bad_lines", 0) > 0:
            measured_quality_status = "failed_measured_runtime"
        elif measured.get("measured_incomplete_windows", 0) > 0:
            measured_quality_status = "failed_measured_window"

    if (support_status == "ok" and args.backend_policy == "fine_grained_qnn_gpu" and
            measured_quality_status not in ("ok", "not_available")):
        support_status = measured_quality_status
    fallback_used = detect_fallback(text)

    row = {
        "date": date_value,
        "model": model,
        "mode": args.mode,
        "backend_policy": args.backend_policy,
        "state_id": args.state_id,
        "workload_type": args.workload_type,
        "context_len": args.context_len or str(bench.get("n_depth") or ""),
        "prompt_tokens": str(prompt_tokens) if prompt_tokens else (args.prompt_tokens or ""),
        "decode_tokens": str(decode_tokens) if decode_tokens else (args.decode_tokens or ""),
        "layers": str(layers) if layers else args.layers,
        "rounds": args.rounds,
        "semantic_correctness_required": args.semantic_correctness_required,
        "semantic_correctness_status": args.semantic_correctness_status,
        "throughput_tps": fmt(throughput),
        "latency_per_token_ms": fmt(latency_per_token_ms),
        "latency_per_layer_ms": fmt(latency_per_layer_ms),
        "active_power_mw": fmt(active_power),
        "power_std_mw": fmt(samples.get("power_std_mw")),
        "energy_mj_per_token": fmt(energy),
        "temp_avg_c": fmt(samples.get("temp_avg_c")),
        "temp_max_c": fmt(samples.get("temp_max_c")),
        "gpu_freq_mhz": gpu_freq_mhz,
        "cpu_freq_khz": args.cpu_freq_khz,
        "qnn_workpoint": args.qnn_workpoint,
        "gpu_active_ratio": fmt(samples.get("gpu_active_ratio"), 4),
        "npu_active_ratio": fmt(npu_active_ratio, 4),
        "qnn_proj_us": fmt(trace.get("qnn_proj_us")),
        "gpu_attn_core_us": fmt(trace.get("gpu_attn_core_us")),
        "qnn_ffn_us": fmt(trace.get("qnn_ffn_us")),
        "sync_qnn_to_gpu_us": fmt(trace.get("sync_qnn_to_gpu_us")),
        "sync_gpu_to_qnn_us": fmt(trace.get("sync_gpu_to_qnn_us")),
        "total_sync_us": fmt(trace.get("total_sync_us")),
        "fallback_used": fallback_used,
        "support_status": support_status,
        "raw_log_path": raw_log_path,
        "sample_path": sample_path,
    }

    if args.output_csv:
        write_profile_csv(args.output_csv, row, args.append)
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=PROFILE_FIELDS)
        writer.writeheader()
        writer.writerow(row)

    evidence = {
        "qnn_events": qnn_events,
        "gpu_events": gpu_events,
        "qnn_proj_events": qnn_proj_events,
        "qnn_ffn_events": qnn_ffn_events,
        "gpu_attn_core_events": gpu_attn_core_events,
        "observed_layers": observed_layers,
        "sample_count": samples.get("sample_count") or 0,
        "measured_quality_status": measured_quality_status,
        "measured_windows": measured.get("measured_windows", 0),
        "measured_incomplete_windows": measured.get("measured_incomplete_windows", 0),
        "measured_context_loads": measured.get("measured_context_loads", 0),
        "measured_graph_loads": measured.get("measured_graph_loads", 0),
        "measured_ensure_graph_loaded": measured.get("measured_ensure_graph_loaded", 0),
        "measured_graph_cache_hits": measured.get("measured_graph_cache_hits", 0),
        "measured_graph_cache_misses": measured.get("measured_graph_cache_misses", 0),
        "measured_weight_syncs": measured.get("measured_weight_syncs", 0),
        "measured_weight_sync_bytes": measured.get("measured_weight_sync_bytes", 0),
        "measured_weight_sync_us": measured.get("measured_weight_sync_us", 0.0),
        "measured_bad_lines": measured.get("measured_bad_lines", 0),
        "measured_bad_examples": measured.get("measured_bad_examples", []),
    }
    if args.summary_md:
        write_summary(args.summary_md, row, args, evidence)


if __name__ == "__main__":
    main()
