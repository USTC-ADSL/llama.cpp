#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import math
import statistics
from dataclasses import dataclass
from pathlib import Path


PALETTE = [
    "#4E79A7",
    "#F28E2B",
    "#59A14F",
    "#E15759",
    "#B07AA1",
    "#76B7B2",
    "#EDC948",
    "#9C755F",
    "#BAB0AC",
    "#2F4B7C",
    "#A05195",
    "#00876C",
]
MARKERS = ["circle", "square", "triangle", "diamond", "x"]
MATPLOTLIB_MARKERS = ["o", "s", "^", "D", "x"]


@dataclass(frozen=True)
class Point:
    x: float
    tbt_ms: float


@dataclass(frozen=True)
class Series:
    backend: str
    label: str
    csv_path: Path
    points: list[Point]


def normalize_backend(value: str) -> str:
    upper = value.strip().upper()
    if "NPU" in upper or upper in {"QNN", "QNN_NPU", "HTP"}:
        return "NPU"
    if upper.startswith("GPU") or upper in {"OPENCL", "VULKAN"}:
        return "GPU"
    if upper.startswith("CPU"):
        return "CPU"
    return upper or "UNKNOWN"


def parse_float(value: str) -> float | None:
    text = value.strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def convert_tbt_to_ms(value: float, unit: str) -> float:
    if unit == "ms":
        return value
    if unit == "us":
        return value / 1000.0
    if unit == "ns":
        return value / 1_000_000.0
    if unit == "s":
        return value * 1000.0
    raise ValueError(f"unsupported TBT unit: {unit}")


def infer_auto_unit(values: list[float]) -> str:
    if not values:
        return "ms"
    median = statistics.median(values)
    if median >= 1_000_000:
        return "ns"
    if median > 1000:
        return "us"
    if 0 < median < 1:
        return "s"
    return "ms"


def read_tbt_csv(
    path: Path,
    *,
    unit: str,
    x_offset: float,
    skip_head: int,
    skip_tail: int,
    max_x: float | None,
    stride: int,
    smooth_window: int,
) -> list[Point]:
    raw_rows: list[tuple[float, float]] = []
    tbt_values: list[float] = []
    implicit_x = 0

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            cells = [cell.strip() for cell in row if cell.strip()]
            if not cells or cells[0].startswith("#"):
                continue

            numeric = [parse_float(cell) for cell in cells]
            numbers = [value for value in numeric if value is not None]
            if len(numbers) >= 2:
                x, tbt = numbers[0], numbers[1]
            elif len(numbers) == 1:
                implicit_x += 1
                x, tbt = float(implicit_x), numbers[0]
            else:
                continue

            raw_rows.append((x + x_offset, tbt))
            tbt_values.append(tbt)

    if skip_head:
        raw_rows = raw_rows[skip_head:]
    if skip_tail:
        raw_rows = raw_rows[:-skip_tail] if skip_tail < len(raw_rows) else []
    if stride > 1:
        raw_rows = raw_rows[::stride]
    if max_x is not None:
        raw_rows = [(x, tbt) for x, tbt in raw_rows if x <= max_x]

    actual_unit = infer_auto_unit(tbt_values) if unit == "auto" else unit
    points = [Point(x=x, tbt_ms=convert_tbt_to_ms(tbt, actual_unit)) for x, tbt in raw_rows]
    if smooth_window > 1:
        points = smooth_points(points, smooth_window)
    return points


def smooth_points(points: list[Point], window: int) -> list[Point]:
    if window <= 1 or len(points) < 3:
        return points
    half = window // 2
    smoothed: list[Point] = []
    for index, point in enumerate(points):
        lo = max(0, index - half)
        hi = min(len(points), index + half + 1)
        smoothed.append(Point(point.x, statistics.mean(item.tbt_ms for item in points[lo:hi])))
    return smoothed


def parse_series_arg(value: str, args: argparse.Namespace) -> Series:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--series must be BACKEND:LABEL:CSV")
    backend, label, csv_name = parts
    path = Path(csv_name)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"TBT CSV does not exist: {path}")
    points = read_tbt_csv(
        path,
        unit=args.tbt_unit,
        x_offset=args.x_offset,
        skip_head=args.skip_head,
        skip_tail=args.skip_tail,
        max_x=args.max_context,
        stride=max(1, args.stride),
        smooth_window=max(1, args.smooth_window),
    )
    if not points:
        raise argparse.ArgumentTypeError(f"TBT CSV has no usable numeric rows: {path}")
    return Series(normalize_backend(backend), label.strip() or path.stem, path, points)


def group_by_backend(series: list[Series]) -> dict[str, list[Series]]:
    grouped: dict[str, list[Series]] = {}
    for item in series:
        grouped.setdefault(item.backend, []).append(item)
    preferred = ["GPU", "CPU", "NPU"]
    return {backend: grouped[backend] for backend in preferred if backend in grouped} | {
        backend: grouped[backend] for backend in sorted(grouped) if backend not in preferred
    }


def padded_range(values: list[float], pad_fraction: float = 0.05) -> tuple[float, float]:
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


def svg_marker(kind: str, x: float, y: float, size: float, color: str) -> str:
    half = size / 2
    if kind == "square":
        return f'<rect x="{x-half:.2f}" y="{y-half:.2f}" width="{size:.2f}" height="{size:.2f}" rx="2" fill="{color}"/>'
    if kind == "triangle":
        points = f"{x:.2f},{y-half:.2f} {x-half:.2f},{y+half:.2f} {x+half:.2f},{y+half:.2f}"
        return f'<polygon points="{points}" fill="{color}"/>'
    if kind == "diamond":
        points = f"{x:.2f},{y-half:.2f} {x+half:.2f},{y:.2f} {x:.2f},{y+half:.2f} {x-half:.2f},{y:.2f}"
        return f'<polygon points="{points}" fill="{color}"/>'
    if kind == "x":
        return (
            f'<line x1="{x-half:.2f}" y1="{y-half:.2f}" x2="{x+half:.2f}" y2="{y+half:.2f}" stroke="{color}" stroke-width="2"/>'
            f'<line x1="{x-half:.2f}" y1="{y+half:.2f}" x2="{x+half:.2f}" y2="{y-half:.2f}" stroke="{color}" stroke-width="2"/>'
        )
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{half:.2f}" fill="{color}"/>'


def polyline_points(points: list[Point], sx, sy) -> str:
    return " ".join(f"{sx(point.x):.2f},{sy(point.tbt_ms):.2f}" for point in points)


def marker_indices(length: int) -> list[int]:
    if length <= 0:
        return []
    step = max(1, length // 12)
    indices = list(range(0, length, step))
    if indices[-1] != length - 1:
        indices.append(length - 1)
    return indices


def render_svg(series: list[Series], output: Path, title: str, xlabel: str, ylabel: str) -> None:
    width = 1180
    height = 720
    left = 96
    right = 282
    top = 74
    bottom = 92
    plot_w = width - left - right
    plot_h = height - top - bottom

    all_x = [point.x for item in series for point in item.points]
    all_y = [point.tbt_ms for item in series for point in item.points]
    min_x, max_x = padded_range(all_x, 0.02)
    min_y, max_y = padded_range(all_y, 0.08)
    min_y = max(0.0, min_y)

    def sx(value: float) -> float:
        return left + (value - min_x) / (max_x - min_x) * plot_w

    def sy(value: float) -> float:
        return top + plot_h - (value - min_y) / (max_y - min_y) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FCFCFD"/>',
        f'<text x="{left}" y="38" font-family="Inter,Arial,sans-serif" font-size="23" font-weight="700" fill="#101828">{html.escape(title)}</text>',
        f'<text x="{left}" y="60" font-family="Inter,Arial,sans-serif" font-size="12" fill="#667085">Per-token TBT growth by backend state</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#FFFFFF" stroke="#D0D5DD" stroke-width="1"/>',
    ]

    for value in tick_values(min_y, max_y):
        y = sy(value)
        parts.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#EAECF0" stroke-width="1"/>',
                f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-family="Inter,Arial,sans-serif" font-size="12" fill="#475467">{format_tick(value)}</text>',
            ]
        )

    for value in tick_values(min_x, max_x):
        x = sx(value)
        parts.extend(
            [
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#F2F4F7" stroke-width="1"/>',
                f'<text x="{x:.2f}" y="{top + plot_h + 24}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="12" fill="#475467">{format_tick(value)}</text>',
            ]
        )

    parts.extend(
        [
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#98A2B3" stroke-width="1.2"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#98A2B3" stroke-width="1.2"/>',
            f'<text x="{left + plot_w / 2:.2f}" y="{height - 34}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="14" fill="#344054">{html.escape(xlabel)}</text>',
            f'<text transform="translate(24 {top + plot_h / 2:.2f}) rotate(-90)" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="14" fill="#344054">{html.escape(ylabel)}</text>',
        ]
    )

    for index, item in enumerate(series):
        color = PALETTE[index % len(PALETTE)]
        marker = MARKERS[index % len(MARKERS)]
        parts.append(
            f'<polyline points="{polyline_points(item.points, sx, sy)}" fill="none" stroke="{color}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>'
        )
        for marker_index in marker_indices(len(item.points)):
            point = item.points[marker_index]
            parts.append(svg_marker(marker, sx(point.x), sy(point.tbt_ms), 8.5, color))

    legend_x = left + plot_w + 28
    legend_y = top + 4
    parts.append(f'<text x="{legend_x}" y="{legend_y}" font-family="Inter,Arial,sans-serif" font-size="13" font-weight="700" fill="#344054">State</text>')
    for index, item in enumerate(series):
        color = PALETTE[index % len(PALETTE)]
        marker = MARKERS[index % len(MARKERS)]
        y = legend_y + 28 + index * 28
        parts.append(svg_marker(marker, legend_x + 8, y - 4, 9, color))
        parts.append(
            f'<text x="{legend_x + 24}" y="{y:.2f}" font-family="Inter,Arial,sans-serif" font-size="12" fill="#344054">{html.escape(item.label)}</text>'
        )

    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts), encoding="utf-8")


def render_matplotlib(series: list[Series], output: Path, title: str, xlabel: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11.8, 7.2))
    for index, item in enumerate(series):
        color = PALETTE[index % len(PALETTE)]
        marker = MATPLOTLIB_MARKERS[index % len(MATPLOTLIB_MARKERS)]
        ax.plot(
            [point.x for point in item.points],
            [point.tbt_ms for point in item.points],
            label=item.label,
            color=color,
            linewidth=2.0,
            marker=marker,
            markevery=max(1, len(item.points) // 12),
            markersize=4.5,
        )
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#EAECF0")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-token TBT curves from one or more CSV files. "
            "Repeat --series as BACKEND:LABEL:CSV; each backend is written to a separate figure."
        )
    )
    parser.add_argument("--series", action="append", default=[], help="Series spec: BACKEND:LABEL:CSV")
    parser.add_argument("--output-dir", default="figures/tbt_curves", help="Directory for generated figures.")
    parser.add_argument("--format", choices=["svg", "png"], default="svg", help="Output figure format.")
    parser.add_argument("--title-prefix", default="TBT", help="Figure title prefix; backend name is appended.")
    parser.add_argument("--xlabel", default="Context length")
    parser.add_argument("--ylabel", default="TBT (ms)")
    parser.add_argument("--tbt-unit", choices=["auto", "ms", "us", "ns", "s"], default="auto")
    parser.add_argument("--x-offset", type=float, default=0.0, help="Value added to the CSV x column.")
    parser.add_argument("--max-context", type=float, default=None, help="Drop points with x larger than this value.")
    parser.add_argument("--skip-head", type=int, default=0, help="Drop this many leading numeric rows from each TBT CSV.")
    parser.add_argument("--skip-tail", type=int, default=0, help="Drop this many trailing numeric rows from each TBT CSV.")
    parser.add_argument("--stride", type=int, default=1, help="Keep one point every N rows for lighter SVG output.")
    parser.add_argument("--smooth-window", type=int, default=1, help="Centered moving average window; 1 disables smoothing.")
    args = parser.parse_args()
    if not args.series:
        parser.error("at least one --series BACKEND:LABEL:CSV is required")
    if args.skip_head < 0 or args.skip_tail < 0:
        parser.error("--skip-head and --skip-tail must be non-negative")
    if args.stride <= 0:
        parser.error("--stride must be positive")
    if args.smooth_window <= 0:
        parser.error("--smooth-window must be positive")
    return args


def main() -> int:
    args = parse_args()
    all_series = [parse_series_arg(value, args) for value in args.series]
    grouped = group_by_backend(all_series)
    output_dir = Path(args.output_dir)
    written: list[Path] = []

    for backend, backend_series in grouped.items():
        title = f"{args.title_prefix} {backend}"
        output = output_dir / f"{backend.lower()}_tbt_curves.{args.format}"
        if args.format == "png":
            try:
                render_matplotlib(backend_series, output, title, args.xlabel, args.ylabel)
            except ImportError:
                output = output.with_suffix(".svg")
                render_svg(backend_series, output, title, args.xlabel, args.ylabel)
                print(f"matplotlib not available; wrote {output}")
        else:
            render_svg(backend_series, output, title, args.xlabel, args.ylabel)
        written.append(output)

    for output in written:
        print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
