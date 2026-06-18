from __future__ import annotations

import csv
import json
import math
import os
import statistics
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


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

STATE_CATALOG_FIELDS = [
    "state_name",
    "backend",
    "cpu_affinity",
    "cpu_freq_mhz",
    "gpu_freq_mhz",
    "npu_workpoint",
    "qnn_graph_capacity",
    "notes",
]

AGG_FIELDS = [
    "phase",
    "state_name",
    "length",
    "n_runs",
    "throughput_worst_tps",
    "mean_tbt_ms",
    "energy_per_token_mj",
    "status",
]

PARETO_FIELDS = [
    "phase",
    "length",
    "state_name",
    "backend",
    "mean_tbt_ms",
    "throughput_worst_tps",
    "energy_per_token_mj",
    "stable",
    "dominated",
    "dominated_by",
]

FRONTIER_FIELDS = [
    "phase",
    "length",
    "alpha",
    "qmax_tps",
    "target_tps",
    "target_mean_tbt_ms",
    "state_name",
    "backend",
    "throughput_worst_tps",
    "mean_tbt_ms",
    "energy_per_token_mj",
    "slo_feasible",
    "selected",
    "filtered_reason",
]

REFINEMENT_FIELDS = [
    "phase",
    "interval_lo",
    "interval_hi",
    "suggested_length",
    "reason",
    "affected_alpha",
    "old_selected_left",
    "old_selected_right",
    "priority",
    "states_to_retest",
]

TRANSITION_FIELDS = [
    "from_state",
    "to_state",
    "from_backend",
    "to_backend",
    "from_graph_capacity",
    "to_graph_capacity",
    "from_workpoint",
    "to_workpoint",
    "latency_ms",
    "energy_mj",
    "cold_or_warm",
    "affects_boundary_tbt",
    "run_id",
    "stable",
    "notes",
]

REQUEST_PLAN_FIELDS = [
    "prompt_len",
    "output_len",
    "alpha",
    "segment_id",
    "bucket_lo",
    "bucket_hi",
    "num_tokens",
    "selected_state",
    "backend",
    "qnn_graph_capacity",
    "mean_tbt_ms",
    "energy_per_token_mj",
    "transition_from_prev",
    "transition_latency_ms",
    "transition_energy_mj",
    "segment_decode_time_ms",
    "segment_decode_energy_mj",
    "total_decode_time_ms",
    "total_decode_energy_mj",
    "target_tps",
    "achieved_tps",
    "slo_ok",
    "notes",
]

DEFAULT_CONTEXT_POINTS = [512, 1024, 1536, 2048, 3072, 4096, 5120, 6144]
DEFAULT_PREFILL_POINTS = [256, 512, 1024, 2048, 4096]
DEFAULT_BUCKETS = [
    [0, 512],
    [512, 1024],
    [1024, 1536],
    [1536, 2048],
    [2048, 3072],
    [3072, 4096],
    [4096, 5120],
    [5120, 6144],
]
DEFAULT_ALPHAS = [1.0, 0.9, 0.8, 0.7]
DEFAULT_REFINEMENT_POINTS = [768, 1280, 1792, 2560, 3584, 4608, 5632]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_config() -> dict[str, Any]:
    return {
        "model_path": "",
        "tokenizer_path": "",
        "output_dir": ".",
        "idle_power_mw": 0.0,
        "repeat": 3,
        "decode_probe_tokens": 64,
        "context_points": list(DEFAULT_CONTEXT_POINTS),
        "prefill_points": list(DEFAULT_PREFILL_POINTS),
        "buckets": [list(item) for item in DEFAULT_BUCKETS],
        "npu_workpoints": ["low_balanced", "balanced", "burst"],
        "qnn_graphs": [
            {"capacity": 2048, "tier": "normal", "notes": "representative graph for sub-2048 contexts"},
            {"capacity": 4096, "tier": "large", "notes": "large graph tier"},
            {"capacity": 6144, "tier": "xlarge", "notes": "xlarge graph tier"},
        ],
        "cpu_affinity_classes": [
            {"name": "B1", "affinity": "B1", "notes": "1 big core"},
            {"name": "B2", "affinity": "B2", "notes": "2 big cores"},
            {"name": "S4", "affinity": "S4", "notes": "small-core-only reference"},
            {"name": "B2+S2", "affinity": "B2+S2", "notes": "hybrid reference"},
            {"name": "allcore", "affinity": "allcore", "notes": "default governor reference"},
        ],
        "cpu_frequencies": [1804, 2208, 2649],
        "gpu_frequencies": [305, 587, 734],
        "thermal_policy": "log_only",
        "power_sampling_command": "",
        "device_command_prefix": "",
        "decode_command_template": "",
        "prefill_command_template": "",
        "dry_run": True,
        "resume": True,
        "fail_fast": False,
        "alpha_levels": list(DEFAULT_ALPHAS),
        "refinement_points": list(DEFAULT_REFINEMENT_POINTS),
    }


def load_config(path: str | Path | None) -> dict[str, Any]:
    cfg = default_config()
    if not path:
        return cfg
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - this host has PyYAML, but real users may not.
        raise RuntimeError("PyYAML is required to read configs/offline_profile.yaml") from exc
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    cfg.update(loaded)
    return cfg


def output_root(config: dict[str, Any]) -> Path:
    root = Path(str(config.get("output_dir") or "."))
    if not root.is_absolute():
        root = repo_root() / root
    return root


def profiles_dir(config: dict[str, Any]) -> Path:
    return output_root(config) / "profiles"


def logs_dir(config: dict[str, Any]) -> Path:
    return output_root(config) / "logs"


def reports_dir(config: dict[str, Any]) -> Path:
    return output_root(config) / "reports"


def figures_dir(config: dict[str, Any]) -> Path:
    return output_root(config) / "figures"


def manifests_dir(config: dict[str, Any]) -> Path:
    return profiles_dir(config) / "manifests"


def ensure_output_dirs(config: dict[str, Any]) -> None:
    for path in [profiles_dir(config), manifests_dir(config), logs_dir(config), reports_dir(config), figures_dir(config)]:
        path.mkdir(parents=True, exist_ok=True)


def read_csv_rows(path: str | Path, required_fields: list[str] | None = None, missing_ok: bool = False) -> list[dict[str, str]]:
    path = Path(path)
    if missing_ok and not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if required_fields is not None:
            if reader.fieldnames != required_fields:
                raise ValueError(f"{path} header mismatch: expected {required_fields}, got {reader.fieldnames}")
        return [{key: (value if value is not None else "") for key, value in row.items()} for row in reader]


def write_csv_rows(path: str | Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def append_csv_row(path: str | Path, fields: list[str], row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null", "na"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def to_int(value: Any) -> int | None:
    number = to_float(value)
    if number is None:
        return None
    return int(number)


def fmt_float(value: Any, digits: int = 6) -> str:
    number = to_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def fmt_alpha(value: Any) -> str:
    number = to_float(value)
    if number is None:
        return str(value)
    text = f"{number:.6f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return text


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def coefficient_of_variation(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = statistics.mean(values)
    if mean == 0:
        return 0.0
    return statistics.stdev(values) / abs(mean)


def median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def catalog_by_state(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["state_name"]: row for row in rows if row.get("state_name")}


def slugify(value: str) -> str:
    out = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_")


def git_commit(root: Path | None = None) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root or repo_root(),
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def environment_snapshot() -> dict[str, str]:
    keys = [
        "USER",
        "HOSTNAME",
        "ANDROID_SERIAL",
        "ANDROID_SDK_ROOT",
        "ANDROID_NDK_ROOT",
        "QNN_SDK_ROOT",
        "QNN_SDK_PATH",
        "PATH",
    ]
    return {key: os.environ.get(key, "") for key in keys if os.environ.get(key, "")}


def aggregate_rows(
    raw_rows: list[dict[str, str]],
    repeat: int,
    manifest_dir: str | Path | None = None,
) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in raw_rows:
        length = to_int(row.get("length"))
        if row.get("phase") and row.get("state_name") and length is not None:
            grouped[(row["phase"], row["state_name"], length)].append(row)

    output: list[dict[str, str]] = []
    for (phase, state_name, length), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][2], item[0][1])):
        tps_values: list[float] = []
        energy_per_token_values: list[float] = []
        for row in rows:
            if row.get("status") != "ok":
                continue
            n_tokens = to_float(row.get("n_tokens"))
            elapsed_ms = to_float(row.get("elapsed_ms"))
            energy_mj = to_float(row.get("energy_mj"))
            if n_tokens is None or elapsed_ms is None or n_tokens <= 0 or elapsed_ms <= 0:
                continue
            tps = n_tokens / (elapsed_ms / 1000.0)
            tps_values.append(tps)
            if energy_mj is not None:
                energy_per_token_values.append(energy_mj / n_tokens)

        n_runs = len(tps_values)
        throughput_worst = min(tps_values) if tps_values else None
        # Conservative mean TBT: derive from the worst repeated-run throughput, not from
        # the fastest or mean throughput, so feasibility does not depend on a lucky run.
        mean_tbt = (1000.0 / throughput_worst) if throughput_worst and throughput_worst > 0 else None
        ept = median_or_none(energy_per_token_values)

        status = "ok"
        if n_runs < repeat:
            status = "insufficient_runs"
        elif coefficient_of_variation(tps_values) > 0.10:
            status = "unstable"
        elif len(energy_per_token_values) >= 2 and coefficient_of_variation(energy_per_token_values) > 0.15:
            status = "unstable"
        elif manifest_dir and manifests_indicate_throttled(manifest_dir, phase, state_name, length):
            status = "throttled"

        output.append(
            {
                "phase": phase,
                "state_name": state_name,
                "length": str(length),
                "n_runs": str(n_runs),
                "throughput_worst_tps": fmt_float(throughput_worst),
                "mean_tbt_ms": fmt_float(mean_tbt),
                "energy_per_token_mj": fmt_float(ept),
                "status": status,
            }
        )
    return output


def manifests_indicate_throttled(manifest_dir: str | Path, phase: str, state_name: str, length: int) -> bool:
    manifest_dir = Path(manifest_dir)
    if not manifest_dir.exists():
        return False

    def nested_float(data: dict[str, Any], *paths: tuple[str, ...]) -> float | None:
        for path in paths:
            current: Any = data
            for key in path:
                if not isinstance(current, dict):
                    current = None
                    break
                current = current.get(key)
            number = to_float(current)
            if number is not None:
                return number
        return None

    for path in manifest_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("phase") != phase or data.get("state_name") != state_name:
            continue
        if to_int(data.get("length")) != length:
            continue
        requested = nested_float(
            data,
            ("requested_cpu_freq_mhz",),
            ("cpu_freq_mhz",),
            ("requested_frequency_mhz",),
            ("command_variables", "cpu_freq_mhz"),
            ("state", "cpu_freq_mhz"),
            ("measurement", "requested_cpu_freq_mhz"),
            ("measurement", "cpu_freq_mhz"),
            ("measurement", "requested_frequency_mhz"),
        )
        measured = nested_float(
            data,
            ("measured_cpu_freq_mhz",),
            ("measured_frequency_mhz",),
            ("actual_cpu_freq_mhz",),
            ("measurement", "measured_cpu_freq_mhz"),
            ("measurement", "measured_frequency_mhz"),
            ("measurement", "actual_cpu_freq_mhz"),
            ("state", "measured_cpu_freq_mhz"),
        )
        if requested is not None and measured is not None and measured < 0.95 * requested:
            return True
    return False


def pareto_rows(agg_rows: list[dict[str, str]], state_catalog: list[dict[str, str]]) -> list[dict[str, str]]:
    catalog = catalog_by_state(state_catalog)
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in agg_rows:
        length = to_int(row.get("length"))
        if row.get("phase") and length is not None:
            grouped[(row["phase"], length)].append(row)

    output: list[dict[str, str]] = []
    for (phase, length), rows in sorted(grouped.items()):
        dominated_by: dict[str, list[str]] = {row["state_name"]: [] for row in rows}
        stable_rows = [row for row in rows if row.get("status") == "ok"]
        for candidate in stable_rows:
            c_tbt = to_float(candidate.get("mean_tbt_ms"))
            c_energy = to_float(candidate.get("energy_per_token_mj"))
            if c_tbt is None or c_energy is None:
                continue
            for other in stable_rows:
                if other is candidate:
                    continue
                o_tbt = to_float(other.get("mean_tbt_ms"))
                o_energy = to_float(other.get("energy_per_token_mj"))
                if o_tbt is None or o_energy is None:
                    continue
                if o_tbt <= c_tbt and o_energy <= c_energy and (o_tbt < c_tbt or o_energy < c_energy):
                    dominated_by[candidate["state_name"]].append(other["state_name"])

        for row in rows:
            state_name = row.get("state_name", "")
            output.append(
                {
                    "phase": phase,
                    "length": str(length),
                    "state_name": state_name,
                    "backend": catalog.get(state_name, {}).get("backend", ""),
                    "mean_tbt_ms": row.get("mean_tbt_ms", ""),
                    "throughput_worst_tps": row.get("throughput_worst_tps", ""),
                    "energy_per_token_mj": row.get("energy_per_token_mj", ""),
                    "stable": bool_text(row.get("status") == "ok"),
                    "dominated": bool_text(bool(dominated_by.get(state_name))),
                    "dominated_by": ";".join(sorted(dominated_by.get(state_name, []))),
                }
            )
    return output


def stable_non_dominated_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output = []
    for row in rows:
        if row.get("stable") != "true" or row.get("dominated") == "true":
            continue
        if (
            to_float(row.get("throughput_worst_tps")) is None
            or to_float(row.get("mean_tbt_ms")) is None
            or to_float(row.get("energy_per_token_mj")) is None
        ):
            continue
        output.append(row)
    return output


def build_frontier_rows(
    pareto: list[dict[str, str]],
    state_catalog: list[dict[str, str]],
    alphas: Iterable[float] = DEFAULT_ALPHAS,
) -> list[dict[str, str]]:
    catalog = catalog_by_state(state_catalog)
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in stable_non_dominated_rows(pareto):
        length = to_int(row.get("length"))
        if row.get("phase") == "decode" and length is not None:
            grouped[(row["phase"], length)].append(row)

    output: list[dict[str, str]] = []
    for (phase, length), rows in sorted(grouped.items()):
        qmax = max(to_float(row.get("throughput_worst_tps")) or 0.0 for row in rows)
        for alpha in alphas:
            target = alpha * qmax
            feasible = [
                row
                for row in rows
                if (to_float(row.get("throughput_worst_tps")) or -1.0) + 1e-12 >= target
            ]
            if not feasible:
                output.append(
                    {
                        "phase": phase,
                        "length": str(length),
                        "alpha": fmt_alpha(alpha),
                        "qmax_tps": fmt_float(qmax),
                        "target_tps": fmt_float(target),
                        "target_mean_tbt_ms": fmt_float(1000.0 / target if target > 0 else None),
                        "state_name": "",
                        "backend": "",
                        "throughput_worst_tps": "",
                        "mean_tbt_ms": "",
                        "energy_per_token_mj": "",
                        "slo_feasible": "false",
                        "selected": "false",
                        "filtered_reason": "no_feasible_state",
                    }
                )
                continue
            winner = min(
                feasible,
                key=lambda row: (
                    to_float(row.get("energy_per_token_mj")) or float("inf"),
                    to_float(row.get("mean_tbt_ms")) or float("inf"),
                    row.get("state_name", ""),
                ),
            )
            state_name = winner.get("state_name", "")
            output.append(
                {
                    "phase": phase,
                    "length": str(length),
                    "alpha": fmt_alpha(alpha),
                    "qmax_tps": fmt_float(qmax),
                    "target_tps": fmt_float(target),
                    "target_mean_tbt_ms": fmt_float(1000.0 / target if target > 0 else None),
                    "state_name": state_name,
                    "backend": catalog.get(state_name, {}).get("backend", winner.get("backend", "")),
                    "throughput_worst_tps": winner.get("throughput_worst_tps", ""),
                    "mean_tbt_ms": winner.get("mean_tbt_ms", ""),
                    "energy_per_token_mj": winner.get("energy_per_token_mj", ""),
                    "slo_feasible": "true",
                    "selected": "true",
                    "filtered_reason": "",
                }
            )
    return output


def buckets_from_config(config: dict[str, Any]) -> list[tuple[int, int]]:
    buckets: list[tuple[int, int]] = []
    for item in config.get("buckets") or DEFAULT_BUCKETS:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            lo = to_int(item[0])
            hi = to_int(item[1])
            if lo is not None and hi is not None and hi > lo:
                buckets.append((lo, hi))
    return buckets or [(lo, hi) for lo, hi in DEFAULT_BUCKETS]


def alphas_from_config(config: dict[str, Any]) -> list[float]:
    values: list[float] = []
    for item in config.get("alpha_levels") or DEFAULT_ALPHAS:
        number = to_float(item)
        if number is not None and number > 0:
            values.append(number)
    return values or list(DEFAULT_ALPHAS)


def context_points_from_config(config: dict[str, Any]) -> list[int]:
    return [int(x) for x in config.get("context_points") or DEFAULT_CONTEXT_POINTS]


def refinement_points_from_config(config: dict[str, Any]) -> list[int]:
    return [int(x) for x in config.get("refinement_points") or DEFAULT_REFINEMENT_POINTS]


def _frontier_selected_by_length_alpha(frontier_rows: list[dict[str, str]]) -> dict[tuple[str, int, str], dict[str, str]]:
    output = {}
    for row in frontier_rows:
        length = to_int(row.get("length"))
        if length is None:
            continue
        output[(row.get("phase", ""), length, row.get("alpha", ""))] = row
    return output


def _fastest_by_phase_length(pareto: list[dict[str, str]]) -> dict[tuple[str, int], dict[str, str]]:
    output: dict[tuple[str, int], dict[str, str]] = {}
    for row in stable_non_dominated_rows(pareto):
        length = to_int(row.get("length"))
        tps = to_float(row.get("throughput_worst_tps"))
        if length is None or tps is None:
            continue
        key = (row.get("phase", ""), length)
        current = output.get(key)
        if current is None or tps > (to_float(current.get("throughput_worst_tps")) or -1.0):
            output[key] = row
    return output


def _feasible_sets_by_phase_length_alpha(
    pareto: list[dict[str, str]], alphas: list[str]
) -> dict[tuple[str, int, str], set[str]]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in stable_non_dominated_rows(pareto):
        length = to_int(row.get("length"))
        if length is not None:
            grouped[(row.get("phase", ""), length)].append(row)

    output: dict[tuple[str, int, str], set[str]] = {}
    for (phase, length), rows in grouped.items():
        qmax = max(to_float(row.get("throughput_worst_tps")) or 0.0 for row in rows)
        for alpha_text in alphas:
            alpha = to_float(alpha_text)
            if alpha is None:
                continue
            target = alpha * qmax
            output[(phase, length, alpha_text)] = {
                row.get("state_name", "")
                for row in rows
                if row.get("state_name") and (to_float(row.get("throughput_worst_tps")) or -1.0) + 1e-12 >= target
            }
    return output


def _states_for_retest(
    phase: str,
    lo: int,
    hi: int,
    pareto: list[dict[str, str]],
    state_catalog: list[dict[str, str]],
    extra: Iterable[str],
) -> str:
    states: set[str] = {state for state in extra if state}
    backends_seen: set[str] = set()
    for row in stable_non_dominated_rows(pareto):
        length = to_int(row.get("length"))
        if row.get("phase") == phase and length in {lo, hi}:
            state = row.get("state_name", "")
            if state:
                states.add(state)
            backend = row.get("backend", "")
            if backend and backend not in backends_seen:
                states.add(state)
                backends_seen.add(backend)
    if len(backends_seen) < 3:
        for state in state_catalog:
            backend = state.get("backend", "")
            state_name = state.get("state_name", "")
            if backend and state_name and backend not in backends_seen:
                states.add(state_name)
                backends_seen.add(backend)
    return ";".join(sorted(states))


def suggest_refinement_rows(
    frontier: list[dict[str, str]],
    pareto: list[dict[str, str]],
    state_catalog: list[dict[str, str]],
    config: dict[str, Any],
) -> list[dict[str, str]]:
    coarse = context_points_from_config(config)
    candidates = refinement_points_from_config(config)
    selected = _frontier_selected_by_length_alpha(frontier)
    fastest = _fastest_by_phase_length(pareto)
    alphas = [fmt_alpha(a) for a in alphas_from_config(config)]
    feasible_sets = _feasible_sets_by_phase_length_alpha(pareto, alphas)
    phases = sorted({row.get("phase", "") for row in frontier if row.get("phase")}) or ["decode"]

    output: list[dict[str, str]] = []

    def add_row(phase: str, lo: int, hi: int, reason: str, alpha: str, left: str, right: str, priority: str, extra: Iterable[str]) -> None:
        suggested = next((point for point in candidates if lo < point < hi), "")
        output.append(
            {
                "phase": phase,
                "interval_lo": str(lo),
                "interval_hi": str(hi),
                "suggested_length": str(suggested),
                "reason": reason,
                "affected_alpha": alpha,
                "old_selected_left": left,
                "old_selected_right": right,
                "priority": priority,
                "states_to_retest": _states_for_retest(phase, lo, hi, pareto, state_catalog, extra),
            }
        )

    for phase in phases:
        for lo, hi in zip(coarse, coarse[1:]):
            if not any(lo < point < hi for point in candidates):
                continue
            left_fast = fastest.get((phase, lo))
            right_fast = fastest.get((phase, hi))
            if left_fast and right_fast:
                if left_fast.get("state_name") != right_fast.get("state_name"):
                    add_row(
                        phase,
                        lo,
                        hi,
                        "fastest_state_change",
                        "",
                        left_fast.get("state_name", ""),
                        right_fast.get("state_name", ""),
                        "medium",
                        [left_fast.get("state_name", ""), right_fast.get("state_name", "")],
                    )
                left_tps = to_float(left_fast.get("throughput_worst_tps"))
                right_tps = to_float(right_fast.get("throughput_worst_tps"))
                if left_tps and right_tps and abs(right_tps / left_tps - 1.0) > 0.10:
                    priority = "high" if abs(right_tps / left_tps - 1.0) > 0.15 else "medium"
                    add_row(
                        phase,
                        lo,
                        hi,
                        "qmax_throughput_change",
                        "",
                        left_fast.get("state_name", ""),
                        right_fast.get("state_name", ""),
                        priority,
                        [left_fast.get("state_name", ""), right_fast.get("state_name", "")],
                    )

            for alpha in alphas:
                left = selected.get((phase, lo, alpha))
                right = selected.get((phase, hi, alpha))
                if not left or not right:
                    continue
                left_state = left.get("state_name", "")
                right_state = right.get("state_name", "")
                if left_state != right_state:
                    add_row(phase, lo, hi, "selected_state_change", alpha, left_state, right_state, "high", [left_state, right_state])

                left_set = feasible_sets.get((phase, lo, alpha), set())
                right_set = feasible_sets.get((phase, hi, alpha), set())
                if left_set and right_set and left_set != right_set:
                    add_row(
                        phase,
                        lo,
                        hi,
                        "feasible_set_change",
                        alpha,
                        left_state,
                        right_state,
                        "medium",
                        sorted(left_set | right_set),
                    )

                left_energy = to_float(left.get("energy_per_token_mj"))
                right_energy = to_float(right.get("energy_per_token_mj"))
                if left_energy and right_energy and abs(right_energy / left_energy - 1.0) > 0.10:
                    priority = "high" if abs(right_energy / left_energy - 1.0) > 0.15 else "medium"
                    add_row(phase, lo, hi, "selected_energy_change", alpha, left_state, right_state, priority, [left_state, right_state])

                tps = to_float(left.get("throughput_worst_tps"))
                target = to_float(left.get("target_tps"))
                if tps and target:
                    margin = tps / target - 1.0
                    if margin < 0.10:
                        priority = "high" if margin < 0.05 else "medium"
                        add_row(phase, lo, hi, "selected_margin_low_left", alpha, left_state, right_state, priority, [left_state, right_state])
                tps = to_float(right.get("throughput_worst_tps"))
                target = to_float(right.get("target_tps"))
                if tps and target:
                    margin = tps / target - 1.0
                    if margin < 0.10:
                        priority = "high" if margin < 0.05 else "medium"
                        add_row(phase, lo, hi, "selected_margin_low_right", alpha, left_state, right_state, priority, [left_state, right_state])

    # De-duplicate identical suggestions.
    seen: set[tuple[str, str, str, str, str]] = set()
    unique: list[dict[str, str]] = []
    for row in output:
        key = (row["phase"], row["interval_lo"], row["interval_hi"], row["reason"], row["affected_alpha"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def request_segments(prompt_len: int, output_len: int, buckets: list[tuple[int, int]]) -> list[dict[str, int]]:
    req_lo = prompt_len
    req_hi = prompt_len + output_len
    segments: list[dict[str, int]] = []
    for lo, hi in buckets:
        seg_lo = max(lo, req_lo)
        seg_hi = min(hi, req_hi)
        if seg_lo < seg_hi:
            segments.append({"bucket_lo": lo, "bucket_hi": hi, "num_tokens": seg_hi - seg_lo})
    covered = sum(segment["num_tokens"] for segment in segments)
    if covered != output_len:
        raise ValueError(
            f"decode range [{req_lo}, {req_hi}) is not fully covered by configured buckets; covered {covered}/{output_len}"
        )
    return segments


def load_transition_map(path: str | Path | None) -> dict[tuple[str, str], dict[str, str]]:
    if not path:
        return {}
    rows = read_csv_rows(path, TRANSITION_FIELDS, missing_ok=True)
    output = {}
    for row in rows:
        if row.get("from_state") and row.get("to_state") and row.get("stable", "true") != "false":
            output[(row["from_state"], row["to_state"])] = row
    return output


def _frontier_targets_by_length(frontier: list[dict[str, str]], alpha: float) -> dict[int, float]:
    targets: dict[int, float] = {}
    alpha_text = fmt_alpha(alpha)
    for row in frontier:
        length = to_int(row.get("length"))
        target = to_float(row.get("target_tps"))
        if row.get("phase") == "decode" and row.get("alpha") == alpha_text and length is not None and target is not None:
            targets[length] = target
    return targets


def _candidates_by_length(
    pareto: list[dict[str, str]],
    state_catalog: list[dict[str, str]],
    alpha: float,
    frontier: list[dict[str, str]] | None = None,
) -> dict[int, list[dict[str, Any]]]:
    catalog = catalog_by_state(state_catalog)
    frontier_targets = _frontier_targets_by_length(frontier or [], alpha)
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in stable_non_dominated_rows(pareto):
        if row.get("phase") != "decode":
            continue
        length = to_int(row.get("length"))
        if length is not None:
            grouped[length].append(row)

    output: dict[int, list[dict[str, Any]]] = {}
    for length, rows in grouped.items():
        qmax = max(to_float(row.get("throughput_worst_tps")) or 0.0 for row in rows)
        target = frontier_targets.get(length, alpha * qmax)
        candidates: list[dict[str, Any]] = []
        for row in rows:
            tps = to_float(row.get("throughput_worst_tps"))
            mean_tbt = to_float(row.get("mean_tbt_ms"))
            energy = to_float(row.get("energy_per_token_mj"))
            state_name = row.get("state_name", "")
            if tps is None or mean_tbt is None or energy is None or tps + 1e-12 < target:
                continue
            state_meta = catalog.get(state_name, {})
            candidates.append(
                {
                    "state_name": state_name,
                    "backend": state_meta.get("backend", row.get("backend", "")),
                    "qnn_graph_capacity": state_meta.get("qnn_graph_capacity", ""),
                    "throughput_worst_tps": tps,
                    "mean_tbt_ms": mean_tbt,
                    "energy_per_token_mj": energy,
                    "target_tps": target,
                    "qmax_tps": qmax,
                }
            )
        output[length] = candidates
    return output


def _transition_cost(
    prev_state: str | None,
    next_state: str,
    transitions: dict[tuple[str, str], dict[str, str]],
    include_transition: bool,
    default_energy_mj: float,
    default_latency_ms: float,
) -> tuple[float, float, str]:
    if prev_state is None or prev_state == next_state:
        return 0.0, 0.0, ""
    if not include_transition:
        return 0.0, 0.0, "no_transition_model"
    row = transitions.get((prev_state, next_state))
    if row:
        energy = to_float(row.get("energy_mj"))
        latency = to_float(row.get("latency_ms"))
        notes = row.get("notes", "")
        if energy is None:
            energy = default_energy_mj
            notes = ";".join(part for part in [notes, "missing_transition_energy"] if part)
        if latency is None:
            latency = default_latency_ms
            notes = ";".join(part for part in [notes, "missing_transition_latency"] if part)
        return energy, latency, notes
    return default_energy_mj, default_latency_ms, "missing_transition_profile"


@dataclass
class _DpCell:
    energy: float
    time_ms: float
    path: list[dict[str, Any]]


def plan_request_rows(
    pareto: list[dict[str, str]],
    state_catalog: list[dict[str, str]],
    transitions: dict[tuple[str, str], dict[str, str]],
    config: dict[str, Any],
    prompt_len: int,
    output_len: int,
    alpha: float,
    include_transition: bool,
    default_transition_energy_mj: float,
    default_transition_latency_ms: float,
    frontier: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    segments = request_segments(prompt_len, output_len, buckets_from_config(config))
    candidates_by_length = _candidates_by_length(pareto, state_catalog, alpha, frontier)
    layers: list[tuple[dict[str, int], list[dict[str, Any]]]] = []
    for segment in segments:
        length = segment["bucket_hi"]
        candidates = candidates_by_length.get(length, [])
        if not candidates:
            raise ValueError(f"no feasible decode candidates for bucket ending at {length} and alpha={alpha}")
        layers.append((segment, candidates))

    prev_cells: dict[str, _DpCell] = {}
    for index, (segment, candidates) in enumerate(layers):
        next_cells: dict[str, _DpCell] = {}
        n_tokens = segment["num_tokens"]
        for candidate in candidates:
            state_name = candidate["state_name"]
            seg_energy = n_tokens * candidate["energy_per_token_mj"]
            seg_time = n_tokens * candidate["mean_tbt_ms"]
            best: _DpCell | None = None
            if index == 0:
                best = _DpCell(
                    energy=seg_energy,
                    time_ms=seg_time,
                    path=[
                        {
                            "segment": segment,
                            "candidate": candidate,
                            "transition_from_prev": "",
                            "transition_energy_mj": 0.0,
                            "transition_latency_ms": 0.0,
                            "notes": "",
                            "segment_decode_energy_mj": seg_energy,
                            "segment_decode_time_ms": seg_time,
                        }
                    ],
                )
            else:
                for prev_state, prev_cell in prev_cells.items():
                    trans_energy, trans_latency, notes = _transition_cost(
                        prev_state,
                        state_name,
                        transitions,
                        include_transition,
                        default_transition_energy_mj,
                        default_transition_latency_ms,
                    )
                    total_energy = prev_cell.energy + trans_energy + seg_energy
                    total_time = prev_cell.time_ms + trans_latency + seg_time
                    path = prev_cell.path + [
                        {
                            "segment": segment,
                            "candidate": candidate,
                            "transition_from_prev": prev_state,
                            "transition_energy_mj": trans_energy,
                            "transition_latency_ms": trans_latency,
                            "notes": notes,
                            "segment_decode_energy_mj": seg_energy,
                            "segment_decode_time_ms": seg_time,
                        }
                    ]
                    cell = _DpCell(total_energy, total_time, path)
                    if best is None or cell.energy < best.energy:
                        best = cell
            assert best is not None
            current = next_cells.get(state_name)
            if current is None or best.energy < current.energy:
                next_cells[state_name] = best
        prev_cells = next_cells

    best = min(prev_cells.values(), key=lambda cell: cell.energy)
    achieved_tps = output_len / (best.time_ms / 1000.0) if best.time_ms > 0 else 0.0
    rows: list[dict[str, str]] = []
    for segment_id, item in enumerate(best.path):
        segment = item["segment"]
        candidate = item["candidate"]
        rows.append(
            {
                "prompt_len": str(prompt_len),
                "output_len": str(output_len),
                "alpha": fmt_alpha(alpha),
                "segment_id": str(segment_id),
                "bucket_lo": str(segment["bucket_lo"]),
                "bucket_hi": str(segment["bucket_hi"]),
                "num_tokens": str(segment["num_tokens"]),
                "selected_state": candidate["state_name"],
                "backend": candidate["backend"],
                "qnn_graph_capacity": candidate["qnn_graph_capacity"],
                "mean_tbt_ms": fmt_float(candidate["mean_tbt_ms"]),
                "energy_per_token_mj": fmt_float(candidate["energy_per_token_mj"]),
                "transition_from_prev": item["transition_from_prev"],
                "transition_latency_ms": fmt_float(item["transition_latency_ms"]),
                "transition_energy_mj": fmt_float(item["transition_energy_mj"]),
                "segment_decode_time_ms": fmt_float(item["segment_decode_time_ms"]),
                "segment_decode_energy_mj": fmt_float(item["segment_decode_energy_mj"]),
                "total_decode_time_ms": fmt_float(best.time_ms),
                "total_decode_energy_mj": fmt_float(best.energy),
                "target_tps": fmt_float(candidate["target_tps"]),
                "achieved_tps": fmt_float(achieved_tps),
                "slo_ok": "true",
                "notes": item["notes"],
            }
        )
    return rows


def graph_capacity_for_length(config: dict[str, Any], length: int) -> int:
    if length <= 2048:
        return 2048
    if length <= 4096:
        return 4096
    return 6144


def graph_capacities(config: dict[str, Any]) -> list[int]:
    values = []
    for item in config.get("qnn_graphs") or []:
        capacity = to_int(item.get("capacity") if isinstance(item, dict) else item)
        if capacity is not None and capacity not in values:
            values.append(capacity)
    return values or [2048, 4096, 6144]


def state_catalog_rows(config: dict[str, Any], backend: str | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    backends = {backend.upper()} if backend else {"NPU", "CPU", "GPU"}

    if "NPU" in backends:
        graph_notes = {
            to_int(item.get("capacity")): item.get("notes", item.get("tier", ""))
            for item in config.get("qnn_graphs", [])
            if isinstance(item, dict)
        }
        for capacity in graph_capacities(config):
            if capacity not in {2048, 4096, 6144}:
                continue
            for workpoint in config.get("npu_workpoints") or []:
                rows.append(
                    {
                        "state_name": f"npu_{workpoint}_cap{capacity}",
                        "backend": "NPU",
                        "cpu_affinity": "",
                        "cpu_freq_mhz": "",
                        "gpu_freq_mhz": "",
                        "npu_workpoint": str(workpoint),
                        "qnn_graph_capacity": str(capacity),
                        "notes": graph_notes.get(capacity, ""),
                    }
                )

    if "CPU" in backends:
        frequencies = [to_int(item) for item in (config.get("cpu_frequencies") or [])]
        frequencies = [item for item in frequencies if item is not None]
        for affinity in ["B1", "B2"]:
            for freq in frequencies:
                rows.append(
                    {
                        "state_name": f"cpu_{affinity}_{freq}",
                        "backend": "CPU",
                        "cpu_affinity": affinity,
                        "cpu_freq_mhz": str(freq),
                        "gpu_freq_mhz": "",
                        "npu_workpoint": "",
                        "qnn_graph_capacity": "",
                        "notes": f"{affinity} locked frequency",
                    }
                )
        for affinity, name, notes in [
            ("S4", "cpu_S4_default", "small-core-only reference"),
            ("B2+S2", "cpu_B2S2_default", "hybrid reference"),
            ("allcore", "cpu_allcore_default", "default governor reference"),
        ]:
            rows.append(
                {
                    "state_name": name,
                    "backend": "CPU",
                    "cpu_affinity": affinity,
                    "cpu_freq_mhz": "",
                    "gpu_freq_mhz": "",
                    "npu_workpoint": "",
                    "qnn_graph_capacity": "",
                    "notes": notes,
                }
            )

    if "GPU" in backends:
        for freq in config.get("gpu_frequencies") or []:
            freq_text = str(freq)
            rows.append(
                {
                    "state_name": f"gpu_{slugify(freq_text)}",
                    "backend": "GPU",
                    "cpu_affinity": "",
                    "cpu_freq_mhz": "",
                    "gpu_freq_mhz": freq_text if to_float(freq_text) is not None else "",
                    "npu_workpoint": "",
                    "qnn_graph_capacity": "",
                    "notes": "locked GPU frequency if platform control is available; otherwise TODO",
                }
            )
    return rows


def merge_state_catalog(existing: list[dict[str, str]], new_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_name = {row["state_name"]: row for row in existing if row.get("state_name")}
    for row in new_rows:
        by_name[row["state_name"]] = row
    return [by_name[key] for key in sorted(by_name)]


def command_variables(
    config: dict[str, Any],
    phase: str,
    state: dict[str, str],
    length: int,
    run_id: int,
    output_json: Path,
    log_path: Path,
) -> dict[str, str]:
    decode_tokens = str(config.get("decode_probe_tokens", 64))
    return {
        "phase": phase,
        "backend": state.get("backend", ""),
        "state_name": state.get("state_name", ""),
        "length": str(length),
        "context_len": str(length if phase == "decode" else ""),
        "prompt_len": str(length if phase == "prefill" else ""),
        "decode_tokens": decode_tokens,
        "cpu_affinity": state.get("cpu_affinity", ""),
        "cpu_freq_mhz": state.get("cpu_freq_mhz", ""),
        "gpu_freq_mhz": state.get("gpu_freq_mhz", ""),
        "npu_workpoint": state.get("npu_workpoint", ""),
        "qnn_graph_capacity": state.get("qnn_graph_capacity", ""),
        "run_id": str(run_id),
        "output_json": str(output_json),
        "log_path": str(log_path),
        "model_path": str(config.get("model_path", "")),
        "tokenizer_path": str(config.get("tokenizer_path", "")),
    }


class _FormatDict(dict):
    def __missing__(self, key: str) -> str:
        return ""


def render_command(template: str, variables: dict[str, str], prefix: str = "") -> str:
    command = template.format_map(_FormatDict(variables))
    if prefix:
        return f"{prefix} {command}".strip()
    return command.strip()


def parse_measurement_json(path: Path, phase: str, length: int, config: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return {"status": "failed", "error": "output_json_missing"}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "failed", "error": f"output_json_parse_failed: {exc}"}
    n_tokens = to_float(data.get("n_tokens"))
    if n_tokens is None:
        n_tokens = to_float(data.get("decode_tokens")) if phase == "decode" else to_float(data.get("prompt_tokens"))
    if n_tokens is None:
        n_tokens = to_float(config.get("decode_probe_tokens")) if phase == "decode" else float(length)
    elapsed_ms = to_float(data.get("elapsed_ms"))
    energy_mj = to_float(data.get("energy_mj"))
    if energy_mj is None and elapsed_ms is not None:
        active_power = to_float(data.get("active_power_mw"))
        avg_power = to_float(data.get("average_power_mw")) or to_float(data.get("avg_power_mw"))
        idle = to_float(config.get("idle_power_mw")) or 0.0
        if active_power is not None:
            energy_mj = active_power * elapsed_ms / 1000.0
        elif avg_power is not None:
            energy_mj = max(0.0, avg_power - idle) * elapsed_ms / 1000.0
    status = str(data.get("status") or "ok")
    if elapsed_ms is None or elapsed_ms <= 0:
        status = "failed"
    measurement = {
        "n_tokens": n_tokens,
        "elapsed_ms": elapsed_ms,
        "energy_mj": energy_mj,
        "status": status,
    }
    for key in [
        "requested_cpu_freq_mhz",
        "cpu_freq_mhz",
        "requested_frequency_mhz",
        "measured_cpu_freq_mhz",
        "measured_frequency_mhz",
        "actual_cpu_freq_mhz",
    ]:
        if key in data:
            measurement[key] = data.get(key)
    return measurement


def measurement_matrix(config: dict[str, Any], backend: str, phase: str = "all", sanity: bool = False) -> list[dict[str, Any]]:
    backend = backend.upper()
    states = state_catalog_rows(config, backend)
    rows: list[dict[str, Any]] = []
    context_points = [int(x) for x in config.get("context_points") or DEFAULT_CONTEXT_POINTS]
    prefill_points = [int(x) for x in config.get("prefill_points") or DEFAULT_PREFILL_POINTS]

    def add(phase_name: str, state: dict[str, str], length: int) -> None:
        rows.append({"phase": phase_name, "state": state, "length": length})

    if backend == "NPU":
        for state in states:
            capacity = to_int(state.get("qnn_graph_capacity"))
            if capacity is None:
                continue
            if sanity:
                lengths = []
                if capacity == 4096:
                    lengths = [512, 1024, 1536, 2048]
                elif capacity == 6144:
                    lengths = [512, 1024, 1536, 2048, 4096]
            else:
                lengths = [length for length in context_points if graph_capacity_for_length(config, length) == capacity]
            if phase in {"decode", "all"}:
                for length in lengths:
                    add("decode", state, length)
            if phase in {"prefill", "all"} and not sanity:
                for length in prefill_points:
                    if graph_capacity_for_length(config, length) == capacity:
                        add("prefill", state, length)

    elif backend == "CPU":
        short_contexts = [length for length in context_points if length <= 2048]
        long_contexts = [length for length in context_points if length in {4096, 6144}]
        for state in states:
            name = state["state_name"]
            if phase in {"decode", "all"}:
                if name.startswith("cpu_B1_") or name.startswith("cpu_B2_"):
                    for length in short_contexts:
                        add("decode", state, length)
                elif name in {"cpu_S4_default", "cpu_B2S2_default", "cpu_allcore_default"}:
                    for length in short_contexts:
                        add("decode", state, length)
                if name.startswith("cpu_B2_") or name in {"cpu_B2S2_default", "cpu_allcore_default"}:
                    for length in long_contexts:
                        add("decode", state, length)
            if phase in {"prefill", "all"}:
                selected_freq = str(config.get("prefill_cpu_frequency_mhz") or (config.get("cpu_frequencies") or [""])[-1])
                if name in {f"cpu_B2_{selected_freq}", "cpu_B2S2_default", "cpu_allcore_default"}:
                    for length in prefill_points:
                        add("prefill", state, length)

    elif backend == "GPU":
        short_contexts = [length for length in context_points if length <= 2048]
        for state in states:
            if phase in {"decode", "all"}:
                for length in short_contexts:
                    add("decode", state, length)
            if phase in {"prefill", "all"}:
                for length in prefill_points:
                    add("prefill", state, length)

    return rows
