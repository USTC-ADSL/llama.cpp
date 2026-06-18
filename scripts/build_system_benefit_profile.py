#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
POWER_SKIP_HEAD_ROWS = 10
POWER_SKIP_TAIL_ROWS = 3
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


@dataclass
class Sample:
    phase: str
    backend: str
    state_name: str
    state_group: str
    length: int
    length_axis: str
    throughput_tps: float
    power_mw: float
    source_file: str
    stable: bool = True
    status: str = "ok"
    bucket_lo: int | None = None
    bucket_hi: int | None = None
    n_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_float(value: object) -> float | None:
    text = str(value if value is not None else "").strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def parse_int(value: object) -> int | None:
    parsed = parse_float(value)
    if parsed is None:
        return None
    return int(round(parsed))


def sanitize_name(value: str) -> str:
    text = value.strip()
    text = text.replace("+", "S")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def normalize_backend(value: str) -> str:
    upper = value.strip().upper()
    if upper in {"QNN_NPU", "QNN-NPU", "HTP"}:
        return "NPU"
    if upper.startswith("CPU"):
        return "CPU"
    if upper.startswith("GPU") or upper == "OPENCL":
        return "GPU"
    if upper.startswith("NPU") or "NPU" in upper:
        return "NPU"
    return upper or "UNKNOWN"


def infer_backend_from_path(path: Path, row: dict[str, str] | None = None) -> str:
    if row:
        backend = row.get("backend") or row.get("device_backend") or ""
        if backend:
            return normalize_backend(backend)
    parts = [part.upper() for part in path.parts]
    if "CPU" in parts:
        return "CPU"
    if "GPU" in parts:
        return "GPU"
    if "NPU" in parts or "QNN" in path.name.upper():
        return "NPU"
    name = path.name.lower()
    if name.startswith("cpu_"):
        return "CPU"
    if name.startswith("gpu_"):
        return "GPU"
    if name.startswith(("npu_", "qnn_npu")):
        return "NPU"
    return "UNKNOWN"


def state_prefix_from_run_name(run_name: str) -> str:
    prefix = run_name.strip()
    prefix = re.sub(r"_run\d+$", "", prefix)
    prefix = re.split(r"_decode\d+|_prefill\d+|_pp\d+", prefix)[0]
    return sanitize_name(prefix)


def parse_frequency_mhz(value: object) -> int | None:
    text = str(value if value is not None else "").strip()
    if not text:
        return None
    numbers = re.findall(r"\d+", text)
    if not numbers:
        return None
    # Most source CPU fields are kHz; GPU summary fields are already MHz.
    number = int(numbers[0])
    if number > 100000:
        return int(round(number / 1000))
    return number


def infer_state(row: dict[str, str], backend: str) -> tuple[str, str, dict[str, Any]]:
    metadata: dict[str, Any] = {}
    run_name = row.get("run_name", "").strip()

    if backend == "GPU":
        freq_mhz = parse_frequency_mhz(row.get("freq_mhz") or row.get("gpu_freq_mhz") or row.get("freq_hz"))
        if freq_mhz and parse_int(row.get("freq_hz")) and parse_int(row.get("freq_hz")) > 1_000_000:
            freq_mhz = int(round((parse_int(row.get("freq_hz")) or 0) / 1_000_000))
        metadata["gpu_freq_mhz"] = freq_mhz
        return f"gpu_{freq_mhz}" if freq_mhz else state_prefix_from_run_name(run_name), "GPU", metadata

    if backend == "NPU":
        workpoint = sanitize_name(row.get("workpoint") or row.get("npu_workpoint") or "")
        if workpoint:
            metadata["npu_workpoint"] = workpoint
            graph_capacity = parse_int(row.get("qnn_graph_capacity") or row.get("graph_capacity"))
            if graph_capacity:
                metadata["qnn_graph_capacity"] = graph_capacity
                return f"npu_{workpoint}_cap{graph_capacity}", workpoint, metadata
            return f"npu_{workpoint}", workpoint, metadata
        return state_prefix_from_run_name(run_name), "NPU", metadata

    if backend == "CPU":
        case_name = sanitize_name(row.get("case_name") or row.get("cpu_affinity") or "")
        cpu_freq = row.get("cpu_freq_khz") or row.get("cpu_freq_mhz") or row.get("freq_mhz") or ""
        metadata["cpu_affinity"] = case_name or None
        metadata["cpu_freq_mhz"] = parse_frequency_mhz(cpu_freq)
        if run_name:
            state_name = state_prefix_from_run_name(run_name)
        else:
            state_name = sanitize_name(f"cpu_{case_name}_{metadata['cpu_freq_mhz']}")
        return state_name, case_name or "CPU", metadata

    return state_prefix_from_run_name(run_name), backend, metadata


def first_float(row: dict[str, str], names: Iterable[str]) -> float | None:
    for name in names:
        parsed = parse_float(row.get(name))
        if parsed is not None:
            return parsed
    return None


def first_int(row: dict[str, str], names: Iterable[str]) -> int | None:
    for name in names:
        parsed = parse_int(row.get(name))
        if parsed is not None:
            return parsed
    return None


def row_to_sample(path: Path, row: dict[str, str], idle_power_mw: float) -> Sample | None:
    status = (row.get("status") or row.get("support_status") or "ok").strip().lower()
    if status and status not in {"ok", "true", "1"}:
        return None

    phase = (row.get("phase") or "decode").strip().lower()
    backend = infer_backend_from_path(path, row)
    if backend == "UNKNOWN":
        return None

    length_axis = "prompt_len" if phase == "prefill" else "context_len"
    length = first_int(row, ["context_tokens", "context_len", "length", "prompt_tokens", "prompt_len"])
    if phase == "prefill":
        length = first_int(row, ["prompt_tokens", "prompt_len", "length", "context_tokens", "context_len"])
    if length is None:
        match = re.search(r"(?:decode|prefill|pp)(\d+)", row.get("run_name", "") or path.name)
        length = int(match.group(1)) if match else None
    if length is None or length <= 0:
        return None

    throughput = first_float(row, ["throughput_tps", "throughput_tok_s", "bench_tps", "bench_throughput_tps"])
    mean_tbt_ms = first_float(row, ["mean_tbt_ms", "tbt_ms"])
    if throughput is None and mean_tbt_ms and mean_tbt_ms > 0:
        throughput = 1000.0 / mean_tbt_ms
    if throughput is None or throughput <= 0:
        return None

    power = first_float(
        row,
        [
            "avg_power_mw",
            "active_power_mw",
            "median_power_mw",
            "power_mw",
            "delta_vs_baseline_mw",
            "power_delta_mw",
        ],
    )
    if power is None:
        return None
    if idle_power_mw:
        power = max(0.0, power - idle_power_mw)

    state_name, state_group, metadata = infer_state(row, backend)
    metadata.update(
        {
            "run_name": row.get("run_name", ""),
            "source_results": row.get("source_results", ""),
            "power_basis": "idle_subtracted" if idle_power_mw else "as_reported",
        }
    )
    stable_text = str(row.get("stable", "true")).strip().lower()
    stable = stable_text not in {"0", "false", "no", "unstable"}

    return Sample(
        phase=phase,
        backend=backend,
        state_name=state_name,
        state_group=state_group,
        length=length,
        length_axis=length_axis,
        throughput_tps=throughput,
        power_mw=power,
        source_file=str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
        stable=stable,
        status="ok",
        metadata=metadata,
    )


def read_dict_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []
        return [dict(row) for row in reader]


def collect_summary_samples(input_dir: Path, idle_power_mw: float) -> list[Sample]:
    samples: list[Sample] = []
    summary_files = sorted(path for path in input_dir.rglob("*.csv") if "summary" in path.name.lower())
    for path in summary_files:
        for row in read_dict_csv(path):
            sample = row_to_sample(path, row, idle_power_mw)
            if sample:
                samples.append(sample)
    return samples


def collect_results_samples(input_dir: Path, idle_power_mw: float) -> list[Sample]:
    samples: list[Sample] = []
    for path in sorted(input_dir.rglob("results.csv")):
        if "source_results" in {part.lower() for part in path.parts}:
            continue
        for row in read_dict_csv(path):
            sample = row_to_sample(path, row, idle_power_mw)
            if sample:
                if sample.power_mw <= 0 and row.get("power_csv"):
                    power = load_power_average(resolve_source_path(path.parent, row["power_csv"]))
                    if power is not None:
                        sample.power_mw = max(0.0, power - idle_power_mw)
                samples.append(sample)
    return samples


def resolve_source_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = base_dir / path
    if candidate.exists():
        return candidate
    return ROOT / path


def read_tbt_values_ms(path: Path) -> list[float]:
    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            numeric = [parse_float(item) for item in row]
            numeric = [item for item in numeric if item is not None]
            if not numeric:
                continue
            value = numeric[-1]
            if value > 1000:
                value = value / 1000.0
            values.append(value)
    return values


def read_tbt_context_rows_ms(path: Path) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    fallback_context = 1
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            numeric = [parse_float(item) for item in row]
            numeric = [item for item in numeric if item is not None]
            if not numeric:
                continue
            if len(numeric) >= 2:
                context = int(round(numeric[0]))
                value = numeric[-1]
            else:
                context = fallback_context
                value = numeric[0]
            fallback_context += 1
            if value > 1000:
                value = value / 1000.0
            rows.append((context, value))
    return rows


def load_power_average(path: Path) -> float | None:
    if not path.exists():
        return None
    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            for row in reader:
                value = first_float(row, ["power_mw_est", "power_mw", "avg_power_mw", "active_power_mw"])
                if value is not None:
                    values.append(value)
        else:
            return None
    if not values:
        return None
    if len(values) <= POWER_SKIP_HEAD_ROWS + POWER_SKIP_TAIL_ROWS:
        return None
    trimmed = values[POWER_SKIP_HEAD_ROWS : len(values) - POWER_SKIP_TAIL_ROWS]
    if not trimmed:
        return None
    return statistics.mean(trimmed)


def find_power_pair(tbt_path: Path) -> Path | None:
    power_name = tbt_path.name.replace("_valid_tbt.csv", "_power.csv")
    candidates = [
        tbt_path.with_name(power_name),
        tbt_path.parent.parent / "Power" / power_name,
        tbt_path.parent.parent / "power" / power_name,
        tbt_path.parent.parent / "Power" / power_name.replace("_run0_", "_"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def manual_state_info(backend: str, state: str) -> tuple[str, str, dict[str, Any]]:
    backend = normalize_backend(backend)
    clean_state = sanitize_name(state)
    metadata: dict[str, Any] = {"manual_state_arg": state}
    if backend == "GPU":
        freq_mhz = parse_frequency_mhz(state)
        metadata["gpu_freq_mhz"] = freq_mhz
        return f"gpu_{freq_mhz}" if freq_mhz else f"gpu_{clean_state}", "GPU", metadata
    if backend == "NPU":
        metadata["npu_workpoint"] = clean_state
        return f"npu_{clean_state}", clean_state, metadata
    if backend == "CPU":
        numbers = [parse_frequency_mhz(item) for item in re.findall(r"\d+", state)]
        numbers = [item for item in numbers if item is not None]
        if numbers:
            metadata["cpu_freq_mhz"] = numbers[0]
        if len(numbers) > 1:
            metadata["cpu_little_freq_mhz"] = numbers[1]
        group_match = re.match(r"([A-Za-z0-9]+)", clean_state)
        group = group_match.group(1) if group_match else "CPU"
        return clean_state if clean_state.startswith("cpu_") else f"cpu_{clean_state}", group, metadata
    return f"{backend.lower()}_{clean_state}", backend, metadata


def build_manual_bucket_samples(
    *,
    backend: str,
    state: str,
    csv_path: Path,
    max_context_len: int,
    bucket_size: int,
    idle_power_mw: float,
    power_csv: Path | None,
) -> list[Sample]:
    backend = normalize_backend(backend)
    if backend not in {"CPU", "GPU", "NPU"}:
        raise ValueError(f"backend must be CPU, GPU, or NPU, got {backend!r}")
    if max_context_len <= 0:
        raise ValueError("--max-context-len must be positive")

    tbt_rows = [(context, tbt_ms) for context, tbt_ms in read_tbt_context_rows_ms(csv_path) if 1 <= context <= max_context_len]
    if not tbt_rows:
        raise ValueError(f"no usable TBT rows in {csv_path}")

    resolved_power_csv = power_csv or find_power_pair(csv_path)
    power = load_power_average(resolved_power_csv) if resolved_power_csv else None
    if power is None:
        raise ValueError(
            f"could not infer power CSV for {csv_path}; pass --power-csv or place a matching file under ../Power/"
        )
    if idle_power_mw:
        power = max(0.0, power - idle_power_mw)

    state_name, state_group, metadata = manual_state_info(backend, state)
    metadata.update(
        {
            "power_file": str(resolved_power_csv) if resolved_power_csv else "",
            "power_basis": "idle_subtracted" if idle_power_mw else "as_reported",
            "power_bucket_policy": "same_average_power_for_each_context_bucket",
            "power_skip_head_rows": POWER_SKIP_HEAD_ROWS,
            "power_skip_tail_rows": POWER_SKIP_TAIL_ROWS,
            "max_context_len": max_context_len,
        }
    )

    grouped: dict[tuple[int, int], list[float]] = {}
    for context, tbt_ms in tbt_rows:
        bucket_lo = ((context - 1) // bucket_size) * bucket_size + 1
        bucket_hi = min(bucket_lo + bucket_size - 1, max_context_len)
        grouped.setdefault((bucket_lo, bucket_hi), []).append(tbt_ms)

    samples: list[Sample] = []
    for (bucket_lo, bucket_hi), tbt_ms_values in sorted(grouped.items()):
        elapsed_ms = sum(tbt_ms_values)
        n_tokens = len(tbt_ms_values)
        if elapsed_ms <= 0 or n_tokens <= 0:
            continue
        throughput = n_tokens / (elapsed_ms / 1000.0)
        samples.append(
            Sample(
                phase="decode",
                backend=backend,
                state_name=state_name,
                state_group=state_group,
                length=bucket_hi,
                length_axis="context_bucket",
                throughput_tps=throughput,
                power_mw=power,
                source_file=str(csv_path.relative_to(ROOT) if csv_path.is_relative_to(ROOT) else csv_path),
                stable=True,
                status="ok",
                bucket_lo=bucket_lo,
                bucket_hi=bucket_hi,
                n_tokens=n_tokens,
                metadata=dict(metadata),
            )
        )
    return samples


def collect_raw_tbt_samples(input_dir: Path, idle_power_mw: float) -> list[Sample]:
    samples: list[Sample] = []
    for tbt_path in sorted(input_dir.rglob("*valid_tbt.csv")):
        match = re.match(r"(?P<prefix>.+)_decode(?P<length>\d+)(?:_run(?P<run>\d+))?_valid_tbt\.csv$", tbt_path.name)
        if not match:
            continue
        tbt_ms = read_tbt_values_ms(tbt_path)
        if not tbt_ms:
            continue
        power_path = find_power_pair(tbt_path)
        power = load_power_average(power_path) if power_path else None
        if power is None:
            continue
        if idle_power_mw:
            power = max(0.0, power - idle_power_mw)
        state_name = sanitize_name(match.group("prefix"))
        backend = infer_backend_from_path(tbt_path)
        mean_tbt_ms = statistics.mean(tbt_ms)
        samples.append(
            Sample(
                phase="decode",
                backend=backend,
                state_name=state_name,
                state_group=infer_group_from_state(state_name, backend),
                length=int(match.group("length")),
                length_axis="context_len",
                throughput_tps=1000.0 / mean_tbt_ms,
                power_mw=power,
                source_file=str(tbt_path.relative_to(ROOT) if tbt_path.is_relative_to(ROOT) else tbt_path),
                stable=True,
                status="ok",
                metadata={
                    "power_file": str(power_path) if power_path else "",
                    "run_id": match.group("run") or "",
                    "power_basis": "idle_subtracted" if idle_power_mw else "as_reported",
                },
            )
        )
    return samples


def infer_group_from_state(state_name: str, backend: str) -> str:
    if backend == "CPU":
        match = re.match(r"cpu_([^_]+)", state_name, flags=re.IGNORECASE)
        return match.group(1) if match else "CPU"
    if backend == "GPU":
        return "GPU"
    if backend == "NPU":
        return state_name.removeprefix("npu_")
    return backend


def collect_ecofrontier_prefill_samples(input_dir: Path, idle_power_mw: float) -> list[Sample]:
    candidates: list[Path] = []
    roots = [
        input_dir / "ecofrontier" / "review_parts" / "states",
        input_dir.parent / "ecofrontier" / "review_parts" / "states",
        input_dir.parent.parent / "ecofrontier" / "review_parts" / "states",
    ]
    for root in roots:
        if root.exists():
            candidates.extend(sorted(root.glob("*_prefill.json")))

    samples: list[Sample] = []
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for item in data:
            if not isinstance(item, dict):
                continue
            status = str(item.get("support_status", "ok")).strip().lower()
            if status not in {"ok", "true", "1"}:
                continue
            length = parse_int(item.get("prompt_tokens") or item.get("prompt_len") or item.get("length_value"))
            throughput = parse_float(item.get("throughput_tps"))
            power = parse_float(item.get("active_power_mw") or item.get("power_mw") or item.get("power_delta_mw"))
            state_name = sanitize_name(str(item.get("state_id") or ""))
            if not length or not throughput or not power or not state_name:
                continue
            backend = normalize_backend(str(item.get("backend") or ""))
            if idle_power_mw:
                power = max(0.0, power - idle_power_mw)
            group = (
                sanitize_name(str(item.get("cpu_affinity") or item.get("npu_workpoint") or ""))
                or ("GPU" if backend == "GPU" else backend)
            )
            samples.append(
                Sample(
                    phase="prefill",
                    backend=backend,
                    state_name=state_name,
                    state_group=group,
                    length=length,
                    length_axis="prompt_len",
                    throughput_tps=throughput,
                    power_mw=power,
                    source_file=str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
                    stable=bool(item.get("stable", True)),
                    status="ok",
                    metadata={
                        "source_file": item.get("source_file", ""),
                        "power_basis": "idle_subtracted" if idle_power_mw else "as_reported",
                    },
                )
            )
    return samples


def coefficient_of_variation(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = statistics.mean(values)
    if mean == 0:
        return 0.0
    return statistics.stdev(values) / mean


def aggregate_samples(
    samples: list[Sample],
    bucket_size: int,
    throughput_cv_threshold: float,
    power_cv_threshold: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, int, int], list[Sample]] = {}
    for sample in samples:
        bucket_lo = sample.bucket_lo if sample.bucket_lo is not None else sample.length
        bucket_hi = sample.bucket_hi if sample.bucket_hi is not None else sample.length
        grouped.setdefault((sample.phase, sample.state_name, sample.length, bucket_lo, bucket_hi), []).append(sample)

    records: list[dict[str, Any]] = []
    for (_phase, _state, _length, _bucket_lo, _bucket_hi), items in sorted(grouped.items()):
        throughput_values = [item.throughput_tps for item in items]
        power_values = [item.power_mw for item in items]
        throughput_cv = coefficient_of_variation(throughput_values)
        power_cv = coefficient_of_variation(power_values)
        stable = all(item.stable for item in items)
        stable = stable and throughput_cv <= throughput_cv_threshold and power_cv <= power_cv_threshold
        throughput_tps = statistics.mean(throughput_values)
        power_mw = statistics.mean(power_values)
        exemplar = items[0]
        bucket_lo = exemplar.bucket_lo if exemplar.bucket_lo is not None else exemplar.length
        bucket_hi = exemplar.bucket_hi if exemplar.bucket_hi is not None else exemplar.length
        tokens_for_record = exemplar.n_tokens or (bucket_size if exemplar.phase == "decode" else exemplar.length)
        latency_ms = tokens_for_record / throughput_tps * 1000.0
        energy_mj = power_mw * latency_ms / 1000.0

        metadata: dict[str, Any] = {}
        for item in items:
            metadata.update({key: value for key, value in item.metadata.items() if value not in {"", None}})

        records.append(
            {
                "phase": exemplar.phase,
                "backend": exemplar.backend,
                "state_name": exemplar.state_name,
                "state_group": exemplar.state_group,
                "length": exemplar.length,
                "length_axis": exemplar.length_axis,
                "bucket_lo": bucket_lo,
                "bucket_hi": bucket_hi,
                "bucket_tokens": tokens_for_record,
                "throughput_tps": throughput_tps,
                "power_mw": power_mw,
                "latency_ms_per_token": 1000.0 / throughput_tps,
                "energy_mj_per_token": power_mw / throughput_tps,
                "latency_ms_per_bucket": latency_ms,
                "energy_mj_per_bucket": energy_mj,
                "source_count": len(items),
                "source_files": sorted({item.source_file for item in items}),
                "stable": stable,
                "status": "ok" if stable else "unstable",
                "throughput_cv": throughput_cv,
                "power_cv": power_cv,
                "metadata": metadata,
            }
        )
    return records


def default_input_dir() -> Path:
    for candidate in [ROOT / "Paper_Writing" / "offline" / "Log", ROOT / "Paper_writing_offline_Log", ROOT / "Paper_Writing" / "offline", ROOT / "test-hz"]:
        if candidate.exists():
            return candidate
    return ROOT / "Paper_writing_offline_Log"


def compact_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase": record["phase"],
        "backend": record["backend"],
        "state_name": record["state_name"],
        "state_group": record["state_group"],
        "bucket_lo": int(record["bucket_lo"]),
        "bucket_hi": int(record["bucket_hi"]),
        "bucket_tokens": int(record["bucket_tokens"]),
        "throughput_tps": float(record["throughput_tps"]),
        "power_mw": float(record["power_mw"]),
        "energy_mj_per_token": float(record["energy_mj_per_token"]),
        "energy_mj_per_bucket": float(record["energy_mj_per_bucket"]),
    }


def compact_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [compact_record(record) for record in records]


def read_profile_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != PROFILE_FIELDS:
            raise ValueError(f"profile CSV schema mismatch in {path}; got {reader.fieldnames}, expected {PROFILE_FIELDS}")
        return [dict(row) for row in reader]


def write_profile_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PROFILE_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record[field] for field in PROFILE_FIELDS})


def count_records(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = f"{record.get('phase')}:{record.get('backend')}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def record_identity(record: dict[str, Any]) -> tuple[str, str, str, int, int]:
    bucket_lo = int(record.get("bucket_lo") or record.get("length") or 0)
    bucket_hi = int(record.get("bucket_hi") or record.get("length") or 0)
    return (
        str(record.get("phase") or ""),
        str(record.get("backend") or ""),
        str(record.get("state_name") or ""),
        bucket_lo,
        bucket_hi,
    )


def merge_records(existing_records: list[dict[str, Any]], new_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = {record_identity(record): record for record in existing_records}
    for record in new_records:
        merged[record_identity(record)] = record
    return sorted(
        merged.values(),
        key=lambda row: (
            str(row.get("phase") or ""),
            str(row.get("backend") or ""),
            str(row.get("state_name") or ""),
            int(row.get("bucket_lo") or row.get("length") or 0),
            int(row.get("bucket_hi") or row.get("length") or 0),
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one compact CSV offline profile for system benefit simulation from Paper_Writing/test-hz logs."
    )
    parser.add_argument("--input-dir", default=None, help="Default tries Paper_Writing/offline/Log, Paper_writing_offline_Log, Paper_Writing/offline, then test-hz.")
    parser.add_argument("--output", default="profiles/system_benefit_offline_profile.csv")
    parser.add_argument("--bucket-size", type=int, default=32)
    parser.add_argument("--idle-power-mw", type=float, default=0.0)
    parser.add_argument("--throughput-cv-threshold", type=float, default=0.10)
    parser.add_argument("--power-cv-threshold", type=float, default=0.15)
    parser.add_argument("--no-prefill-json", action="store_true", help="Do not import EcoFrontier prefill JSON records.")
    parser.add_argument("--backend", default=None, help="Manual add mode: CPU, GPU, or NPU.")
    parser.add_argument("--state", default=None, help="Manual add mode: state/frequency/workpoint, e.g. 734, burst, B2_2649600, B2S2_3513600_2745600.")
    parser.add_argument("--csv", default=None, help="Manual add mode: TBT CSV path from Paper_Writing/offline/Log.")
    parser.add_argument("--max-context-len", type=int, default=None, help="Manual add mode: maximum context length covered by the CSV.")
    parser.add_argument("--power-csv", default=None, help="Optional manual add mode override for power CSV. By default a matching ../Power file is used.")
    parser.add_argument("--dry-run", action="store_true", help="Scan and print counts without writing output.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manual_values = [args.backend, args.state, args.csv, args.max_context_len]
    manual_mode = any(value is not None for value in manual_values)
    if manual_mode and not all(value is not None for value in manual_values):
        raise SystemExit("manual add mode requires all four arguments: --backend --state --csv --max-context-len")

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output

    if manual_mode:
        csv_path = Path(str(args.csv)).resolve()
        if not csv_path.exists():
            raise SystemExit(f"CSV file not found: {csv_path}")
        power_csv = Path(str(args.power_csv)).resolve() if args.power_csv else None
        if power_csv and not power_csv.exists():
            raise SystemExit(f"power CSV file not found: {power_csv}")
        all_samples = build_manual_bucket_samples(
            backend=str(args.backend),
            state=str(args.state),
            csv_path=csv_path,
            max_context_len=int(args.max_context_len),
            bucket_size=args.bucket_size,
            idle_power_mw=args.idle_power_mw,
            power_csv=power_csv,
        )
        source_mode = "manual_log_csv"
        records = aggregate_samples(
            all_samples,
            bucket_size=args.bucket_size,
            throughput_cv_threshold=args.throughput_cv_threshold,
            power_cv_threshold=args.power_cv_threshold,
        )
        compact_new_records = compact_records(records)
        if output.exists():
            try:
                existing_records = read_profile_csv(output)
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
        else:
            existing_records = []
        merged_records = merge_records(existing_records, compact_new_records)
        counts = count_records(merged_records)

        if args.dry_run:
            print(f"manual_csv={csv_path}")
            print(f"backend={normalize_backend(str(args.backend))} state={args.state} max_context_len={args.max_context_len}")
            print(f"samples={len(all_samples)} records_to_add={len(compact_new_records)} output_records_after_merge={len(merged_records)}")
            for key in sorted(counts):
                print(f"{key}: {counts[key]}")
            print(f"dry-run: would write {output}")
            return 0

        write_profile_csv(output, merged_records)
        print(f"wrote {output}")
        print(f"samples={len(all_samples)} records_added_or_updated={len(compact_new_records)} total_records={len(merged_records)}")
        for key in sorted(counts):
            print(f"{key}: {counts[key]}")
        return 0

    input_dir = Path(args.input_dir).resolve() if args.input_dir else default_input_dir().resolve()
    if not input_dir.exists():
        raise SystemExit(f"input directory not found: {input_dir}")

    decode_samples = collect_summary_samples(input_dir, args.idle_power_mw)
    source_mode = "summary_csv"
    if not decode_samples:
        decode_samples = collect_results_samples(input_dir, args.idle_power_mw)
        source_mode = "results_csv"
    if not decode_samples:
        decode_samples = collect_raw_tbt_samples(input_dir, args.idle_power_mw)
        source_mode = "raw_tbt_power_csv"

    prefill_samples: list[Sample] = []
    if not args.no_prefill_json:
        prefill_samples = collect_ecofrontier_prefill_samples(input_dir, args.idle_power_mw)

    all_samples = decode_samples + prefill_samples
    records = aggregate_samples(
        all_samples,
        bucket_size=args.bucket_size,
        throughput_cv_threshold=args.throughput_cv_threshold,
        power_cv_threshold=args.power_cv_threshold,
    )

    compact_output_records = compact_records(records)
    counts = count_records(compact_output_records)

    if args.dry_run:
        print(f"input_dir={input_dir}")
        print(f"source_mode={source_mode}")
        print(f"samples={len(all_samples)} records={len(compact_output_records)}")
        for key in sorted(counts):
            print(f"{key}: {counts[key]}")
        print(f"dry-run: would write {output}")
        return 0

    write_profile_csv(output, compact_output_records)
    print(f"wrote {output}")
    print(f"samples={len(all_samples)} records={len(compact_output_records)}")
    for key in sorted(counts):
        print(f"{key}: {counts[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
