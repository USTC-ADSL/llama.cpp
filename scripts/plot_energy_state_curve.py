#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PALETTE = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1", "#76B7B2"]


@dataclass(frozen=True)
class ProfileRow:
    phase: str
    backend: str
    state_name: str
    state_group: str
    bucket_lo: int
    bucket_hi: int
    energy_mj_per_token: float


@dataclass(frozen=True)
class CurvePoint:
    requested_state: str
    matched_state: str
    backend: str
    x: float
    x_label: str
    energy_mj_per_token: float
    n_records: int
    bucket_label: str


def normalize_backend(value: str) -> str:
    upper = value.strip().upper()
    if "NPU" in upper or upper in {"QNN", "QNN_NPU", "HTP"}:
        return "NPU"
    if upper.startswith("GPU") or upper in {"OPENCL", "VULKAN"}:
        return "GPU"
    if upper.startswith("CPU"):
        return "CPU"
    return upper or "UNKNOWN"


def parse_float(value: object, default: float | None = None) -> float:
    text = str(value if value is not None else "").strip()
    if not text:
        if default is None:
            raise ValueError("missing float")
        return default
    try:
        parsed = float(text)
    except ValueError:
        if default is None:
            raise
        return default
    if math.isnan(parsed) or math.isinf(parsed):
        if default is None:
            raise ValueError("non-finite float")
        return default
    return parsed


def parse_int(value: object, default: int = 0) -> int:
    return int(parse_float(value, float(default)))


def state_aliases(state_name: str) -> set[str]:
    clean = state_name.strip()
    aliases = {clean}
    for prefix in ["gpu_", "cpu_", "npu_"]:
        if clean.startswith(prefix):
            suffix = clean.removeprefix(prefix)
            aliases.add(suffix)
            aliases.add(f"{prefix}{suffix}")
        else:
            aliases.add(f"{prefix}{clean}")
    return aliases


def state_matches(profile_state: str, requested_state: str) -> bool:
    return bool(state_aliases(profile_state) & state_aliases(requested_state))


def read_profile(path: Path) -> list[ProfileRow]:
    rows: list[ProfileRow] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            phase = str(row.get("phase") or "").lower()
            backend = normalize_backend(str(row.get("backend") or ""))
            state_name = str(row.get("state_name") or "").strip()
            bucket_lo = parse_int(row.get("bucket_lo") or row.get("length"), 0)
            bucket_hi = parse_int(row.get("bucket_hi") or row.get("length"), 0)
            throughput = parse_float(row.get("throughput_tps"), 0.0)
            power = parse_float(row.get("power_mw"), 0.0)
            energy = parse_float(row.get("energy_mj_per_token"), math.nan)
            if math.isnan(energy) and throughput > 0:
                energy = power / throughput
            if not phase or not state_name or bucket_hi <= 0 or energy < 0 or math.isnan(energy):
                continue
            rows.append(
                ProfileRow(
                    phase=phase,
                    backend=backend,
                    state_name=state_name,
                    state_group=str(row.get("state_group") or ""),
                    bucket_lo=bucket_lo,
                    bucket_hi=bucket_hi,
                    energy_mj_per_token=energy,
                )
            )
    return rows


def split_states(value: str) -> list[str]:
    states = [item.strip() for item in value.split(",") if item.strip()]
    if not states:
        raise argparse.ArgumentTypeError("--states must contain at least one state")
    return states


def numeric_x_from_state(state_name: str) -> float | None:
    text = state_name.strip().lower()
    if text.startswith("gpu_"):
        numbers = [int(item) for item in re.findall(r"\d+", text.removeprefix("gpu_"))]
        return float(numbers[-1]) if numbers else None
    numbers = [int(item) for item in re.findall(r"\d+", text)]
    freq_like = [item for item in numbers if item >= 100]
    if len(freq_like) == 1:
        return float(freq_like[0])
    return None


def aggregate(values: list[float], mode: str) -> float:
    if mode == "mean":
        return statistics.mean(values)
    return statistics.median(values)


def build_curve_points(
    rows: list[ProfileRow],
    *,
    states: list[str],
    phase: str,
    bucket_lo: int | None,
    bucket_hi: int | None,
    aggregate_mode: str,
    preserve_order: bool,
) -> tuple[list[CurvePoint], str]:
    points: list[CurvePoint] = []
    matched_backends: set[str] = set()
    for index, requested_state in enumerate(states):
        matches = [
            row
            for row in rows
            if row.phase == phase
            and state_matches(row.state_name, requested_state)
            and (bucket_lo is None or row.bucket_lo == bucket_lo)
            and (bucket_hi is None or row.bucket_hi == bucket_hi)
        ]
        if not matches:
            filters = []
            if bucket_lo is not None:
                filters.append(f"bucket_lo={bucket_lo}")
            if bucket_hi is not None:
                filters.append(f"bucket_hi={bucket_hi}")
            suffix = f" ({', '.join(filters)})" if filters else ""
            raise ValueError(f"no profile rows matched state={requested_state!r} phase={phase}{suffix}")
        backends = {row.backend for row in matches}
        if len(backends) != 1:
            raise ValueError(f"state={requested_state!r} matched multiple backends: {sorted(backends)}")
        backend = next(iter(backends))
        matched_backends.add(backend)
        matched_state = matches[0].state_name
        energy = aggregate([row.energy_mj_per_token for row in matches], aggregate_mode)
        numeric_x = numeric_x_from_state(requested_state) or numeric_x_from_state(matched_state)
        x = numeric_x if numeric_x is not None else float(index)
        x_label = str(int(numeric_x)) if numeric_x is not None else requested_state
        if bucket_lo is not None or bucket_hi is not None:
            bucket_label = f"{matches[0].bucket_lo}-{matches[0].bucket_hi}"
        else:
            bucket_label = f"{aggregate_mode} over {len(matches)} buckets"
        points.append(
            CurvePoint(
                requested_state=requested_state,
                matched_state=matched_state,
                backend=backend,
                x=x,
                x_label=x_label,
                energy_mj_per_token=energy,
                n_records=len(matches),
                bucket_label=bucket_label,
            )
        )

    if len(matched_backends) != 1:
        raise ValueError(f"--states must describe one backend; matched backends={sorted(matched_backends)}")
    numeric = all(numeric_x_from_state(point.requested_state) is not None or numeric_x_from_state(point.matched_state) is not None for point in points)
    if numeric and not preserve_order:
        points.sort(key=lambda point: point.x)
    return points, next(iter(matched_backends))


def padded_range(values: list[float], pad_fraction: float = 0.08) -> tuple[float, float]:
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        delta = max(1.0, abs(lo) * 0.05)
        return lo - delta, hi + delta
    pad = (hi - lo) * pad_fraction
    return lo - pad, hi + pad


def tick_values(lo: float, hi: float, count: int = 5) -> list[float]:
    if count <= 1 or math.isclose(lo, hi):
        return [lo]
    return [lo + (hi - lo) * index / (count - 1) for index in range(count)]


def format_tick(value: float) -> str:
    if abs(value) >= 100:
        return str(int(round(value)))
    if abs(value) >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


def render_svg(points: list[CurvePoint], output: Path, title: str, xlabel: str, ylabel: str) -> None:
    width = 1040
    height = 640
    left = 94
    right = 56
    top = 76
    bottom = 110
    plot_w = width - left - right
    plot_h = height - top - bottom

    min_x, max_x = padded_range([point.x for point in points], 0.06)
    min_y, max_y = padded_range([point.energy_mj_per_token for point in points], 0.12)
    min_y = max(0.0, min_y)

    def sx(value: float) -> float:
        return left + (value - min_x) / (max_x - min_x) * plot_w

    def sy(value: float) -> float:
        return top + plot_h - (value - min_y) / (max_y - min_y) * plot_h

    line_points = " ".join(f"{sx(point.x):.2f},{sy(point.energy_mj_per_token):.2f}" for point in points)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FCFCFD"/>',
        f'<text x="{left}" y="38" font-family="Inter,Arial,sans-serif" font-size="22" font-weight="700" fill="#101828">{html.escape(title)}</text>',
        f'<text x="{left}" y="60" font-family="Inter,Arial,sans-serif" font-size="12" fill="#667085">Energy per generated token by frequency/workpoint</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#FFFFFF" stroke="#D0D5DD"/>',
    ]

    for value in tick_values(min_y, max_y):
        y = sy(value)
        parts.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#EAECF0"/>',
                f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-family="Inter,Arial,sans-serif" font-size="12" fill="#475467">{format_tick(value)}</text>',
            ]
        )
    for point in points:
        x = sx(point.x)
        parts.extend(
            [
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#F2F4F7"/>',
                f'<text x="{x:.2f}" y="{top + plot_h + 24}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="12" fill="#475467">{html.escape(point.x_label)}</text>',
            ]
        )

    color = PALETTE[0]
    parts.append(f'<polyline points="{line_points}" fill="none" stroke="{color}" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>')
    for point in points:
        x = sx(point.x)
        y = sy(point.energy_mj_per_token)
        tooltip = f"{point.matched_state}: {point.energy_mj_per_token:.4g} mJ/token, {point.bucket_label}"
        parts.extend(
            [
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5.5" fill="{color}"/>',
                f'<title>{html.escape(tooltip)}</title>',
                f'<text x="{x:.2f}" y="{y - 12:.2f}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="11" fill="#344054">{html.escape(point.matched_state)}</text>',
            ]
        )

    parts.extend(
        [
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#98A2B3" stroke-width="1.2"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#98A2B3" stroke-width="1.2"/>',
            f'<text x="{left + plot_w / 2:.2f}" y="{height - 34}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="14" fill="#344054">{html.escape(xlabel)}</text>',
            f'<text transform="translate(24 {top + plot_h / 2:.2f}) rotate(-90)" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="14" fill="#344054">{html.escape(ylabel)}</text>',
            "</svg>",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts), encoding="utf-8")


def render_matplotlib(points: list[CurvePoint], output: Path, title: str, xlabel: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.4, 6.4))
    ax.plot(
        [point.x for point in points],
        [point.energy_mj_per_token for point in points],
        color=PALETTE[0],
        marker="o",
        linewidth=2.0,
    )
    for point in points:
        ax.annotate(point.matched_state, (point.x, point.energy_mj_per_token), textcoords="offset points", xytext=(0, 8), ha="center")
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks([point.x for point in points], [point.x_label for point in points])
    ax.grid(True, color="#EAECF0")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot mJ/token versus frequency/workpoint for one backend state list.")
    parser.add_argument("--profile", default="profiles/system_benefit_offline_profile.csv")
    parser.add_argument("--states", required=True, type=split_states, help="Comma-separated states, e.g. gpu_1100,gpu_1050,gpu_222.")
    parser.add_argument("--phase", choices=["decode", "prefill"], default="decode")
    parser.add_argument("--bucket-lo", type=int, default=None, help="Optional exact bucket_lo filter.")
    parser.add_argument("--bucket-hi", type=int, default=None, help="Optional exact bucket_hi filter.")
    parser.add_argument("--aggregate", choices=["median", "mean"], default="median", help="How to combine multiple matched buckets per state.")
    parser.add_argument("--preserve-order", action="store_true", help="Keep --states order instead of sorting numeric frequencies.")
    parser.add_argument("--output", default="figures/energy_state_curve.svg")
    parser.add_argument("--format", choices=["svg", "png"], default="svg")
    parser.add_argument("--title", default=None)
    parser.add_argument("--xlabel", default="frequency/workpoint")
    parser.add_argument("--ylabel", default="mJ/token")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile_path = Path(args.profile)
    if not profile_path.is_absolute():
        profile_path = ROOT / profile_path
    rows = read_profile(profile_path)
    if not rows:
        raise SystemExit(f"profile has no usable rows: {profile_path}")
    try:
        points, backend = build_curve_points(
            rows,
            states=args.states,
            phase=args.phase,
            bucket_lo=args.bucket_lo,
            bucket_hi=args.bucket_hi,
            aggregate_mode=args.aggregate,
            preserve_order=args.preserve_order,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    title = args.title or f"{backend} Energy per Token"
    if args.format == "png":
        try:
            render_matplotlib(points, output, title, args.xlabel, args.ylabel)
        except ImportError:
            output = output.with_suffix(".svg")
            render_svg(points, output, title, args.xlabel, args.ylabel)
            print(f"matplotlib not available; wrote {output}")
    else:
        render_svg(points, output, title, args.xlabel, args.ylabel)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
