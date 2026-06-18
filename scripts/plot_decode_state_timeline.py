#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
import re
from pathlib import Path
from typing import Any


STATE_PALETTE = [
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
BACKEND_MARKER = {"CPU": "circle", "GPU": "square", "NPU": "triangle"}
NPU_ORDER = {"low": 0, "low_balanced": 1, "balanced": 2, "burst": 3}


def normalize_backend(value: str) -> str:
    upper = value.strip().upper()
    if "NPU" in upper or upper in {"QNN", "QNN_NPU", "HTP"}:
        return "NPU"
    if upper.startswith("GPU") or upper == "OPENCL":
        return "GPU"
    if upper.startswith("CPU"):
        return "CPU"
    return upper


def segment_start(segment: dict[str, Any]) -> float:
    return float(segment.get("context_bucket_lo", segment.get("context_len", 0)))


def segment_end(segment: dict[str, Any]) -> float:
    return float(segment.get("context_bucket_hi", segment.get("context_len", 0)))


def state_key(segment: dict[str, Any]) -> str:
    return str(segment.get("selected_state") or "")


def parse_state_tier(backend: str, state_name: str, state_group: str) -> float:
    backend = normalize_backend(backend)
    text = f"{state_name} {state_group}".lower()
    if backend == "NPU":
        for name, tier in NPU_ORDER.items():
            if name in text:
                return float(tier)
    numbers = [int(item) for item in re.findall(r"\d+", text)]
    if backend == "CPU":
        freqs = [item for item in numbers if item >= 1000]
        return float(sum(freqs) / len(freqs)) if freqs else float(max(numbers, default=0))
    return float(max(numbers, default=0))


def compact_state_label(segment: dict[str, Any]) -> str:
    backend = normalize_backend(str(segment.get("backend") or ""))
    state_name = state_key(segment)
    state_group = str(segment.get("state_group") or "")
    if backend == "GPU":
        numbers = [int(item) for item in re.findall(r"\d+", state_name)]
        tier = f"{max(numbers)} MHz" if numbers else state_name
    elif backend == "NPU":
        text = f"{state_name} {state_group}".lower()
        tier = state_name
        for name in ["low_balanced", "balanced", "burst", "low"]:
            if name in text:
                tier = name
                break
    elif backend == "CPU":
        tier = state_name.removeprefix("cpu_")
    else:
        tier = state_name
    return f"{backend}  {tier}"


def load_segments(path: Path, plan: str) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    candidates = [
        (f"{plan}.segments", data.get(plan, {}).get("segments")),
        (f"plan.{plan}.segments", data.get("plan", {}).get(plan, {}).get("segments")),
    ]
    segments = []
    matched_path = ""
    for candidate_path, candidate_segments in candidates:
        if isinstance(candidate_segments, list) and candidate_segments:
            segments = candidate_segments
            matched_path = candidate_path
            break
    if not isinstance(segments, list) or not segments:
        searched = ", ".join(candidate_path for candidate_path, _ in candidates)
        raise ValueError(f"{path} has no decode segments under any of: {searched}")
    decoded = [segment for segment in segments if isinstance(segment, dict) and state_key(segment)]
    if not decoded:
        raise ValueError(f"{path} has no usable decode segments under {matched_path}")
    return sorted(decoded, key=lambda segment: (segment_start(segment), segment_end(segment)))


def ordered_states(segments: list[dict[str, Any]]) -> list[str]:
    first_seen: dict[str, int] = {}
    exemplars: dict[str, dict[str, Any]] = {}
    for index, segment in enumerate(segments):
        key = state_key(segment)
        first_seen.setdefault(key, index)
        exemplars.setdefault(key, segment)
    return sorted(
        first_seen,
        key=lambda key: (
            normalize_backend(str(exemplars[key].get("backend") or "")),
            parse_state_tier(
                str(exemplars[key].get("backend") or ""),
                key,
                str(exemplars[key].get("state_group") or ""),
            ),
            first_seen[key],
        ),
    )


def state_colors(states: list[str]) -> dict[str, str]:
    return {state: STATE_PALETTE[index % len(STATE_PALETTE)] for index, state in enumerate(states)}


def infer_switch_reason(prev: dict[str, Any], current: dict[str, Any]) -> str:
    reason = str(current.get("switch_reason") or "").strip().lower()
    try:
        saving_value = float(current.get("energy_saving_vs_prev_mj"))
    except (TypeError, ValueError):
        saving_value = math.nan
    if reason == "energy" and (math.isnan(saving_value) or saving_value > 0):
        return "energy"
    if reason == "slo":
        return reason
    if current.get("step_slo_ok") is False:
        return "slo"
    if str(current.get("selection_mode") or "").strip().lower() == "best_effort_closest_to_slo":
        return "slo"
    if current.get("profile_feasible_for_slo") is False:
        return "slo"
    if not math.isnan(saving_value) and saving_value > 0:
        return "energy"
    if state_key(prev) != state_key(current):
        return "slo"
    return ""


def draw_backend_marker(kind: str, x: float, y: float, size: float, color: str) -> str:
    if kind == "square":
        half = size / 2
        return f'<rect x="{x-half:.2f}" y="{y-half:.2f}" width="{size:.2f}" height="{size:.2f}" rx="2" fill="{color}"/>'
    if kind == "triangle":
        half = size / 2
        points = f"{x:.2f},{y-half:.2f} {x-half:.2f},{y+half:.2f} {x+half:.2f},{y+half:.2f}"
        return f'<polygon points="{points}" fill="{color}"/>'
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{size/2:.2f}" fill="{color}"/>'


def draw_switch_marker(reason: str, x: float, y: float) -> list[str]:
    if reason == "slo":
        color = "#D92D20"
        points = f"{x:.2f},{y-7:.2f} {x+7:.2f},{y:.2f} {x:.2f},{y+7:.2f} {x-7:.2f},{y:.2f}"
        return [
            f'<polygon points="{points}" fill="#ffffff" stroke="{color}" stroke-width="2"/>',
            f'<text x="{x:.2f}" y="{y+3.5:.2f}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="8" font-weight="700" fill="{color}">S</text>',
        ]
    color = "#175CD3"
    return [
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="7" fill="#ffffff" stroke="{color}" stroke-width="2"/>',
        f'<text x="{x:.2f}" y="{y+3.5:.2f}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="8" font-weight="700" fill="{color}">E</text>',
    ]


def switch_annotation(segment: dict[str, Any], reason: str) -> str:
    if reason == "slo":
        try:
            miss_ms = float(segment.get("step_slo_miss_ms"))
        except (TypeError, ValueError):
            miss_ms = 0.0
        return f"SLO +{miss_ms:.0f}ms" if miss_ms > 0 else "SLO"
    if reason == "energy":
        try:
            saving_value = float(segment.get("energy_saving_vs_prev_mj"))
        except (TypeError, ValueError):
            return ""
        if saving_value <= 0:
            return ""
        if saving_value < 10:
            return f"-{saving_value:.1f}mJ"
        return f"-{saving_value:.0f}mJ"
    return ""


def switch_annotation_color(reason: str) -> str:
    return "#D92D20" if reason == "slo" else "#175CD3"


def render_svg(segments: list[dict[str, Any]], output: Path, title: str) -> None:
    states = ordered_states(segments)
    colors = state_colors(states)
    state_to_row = {state: index for index, state in enumerate(states)}
    exemplars = {state_key(segment): segment for segment in reversed(segments)}

    width = 1180
    row_h = 46
    top = 74
    left = 246
    right = 42
    bottom = 88
    plot_h = max(1, len(states)) * row_h
    height = top + plot_h + bottom
    plot_w = width - left - right

    min_x = min(segment_start(segment) for segment in segments)
    max_x = max(segment_end(segment) for segment in segments)
    if math.isclose(min_x, max_x):
        max_x = min_x + 1

    def sx(value: float) -> float:
        return left + (value - min_x) / (max_x - min_x) * plot_w

    def row_y(row: int) -> float:
        return top + row * row_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#FCFCFD"/>',
        f'<text x="{left}" y="36" font-family="Inter,Arial,sans-serif" font-size="22" font-weight="700" fill="#101828">{html.escape(title)}</text>',
        f'<text x="{left}" y="58" font-family="Inter,Arial,sans-serif" font-size="12" fill="#667085">Decode time sharing by selected backend state</text>',
    ]

    for row, state in enumerate(states):
        y = row_y(row)
        cy = y + row_h / 2
        fill = "#F8FAFC" if row % 2 == 0 else "#FFFFFF"
        segment = exemplars[state]
        backend = normalize_backend(str(segment.get("backend") or ""))
        marker = BACKEND_MARKER.get(backend, "circle")
        color = colors[state]
        label = compact_state_label(segment)
        parts.extend(
            [
                f'<rect x="{left}" y="{y:.2f}" width="{plot_w}" height="{row_h}" fill="{fill}" stroke="#EAECF0" stroke-width="1"/>',
                draw_backend_marker(marker, 30, cy, 13, color),
                f'<text x="50" y="{cy-2:.2f}" font-family="Inter,Arial,sans-serif" font-size="13" font-weight="700" fill="#344054">{html.escape(label)}</text>',
                f'<text x="50" y="{cy+15:.2f}" font-family="Inter,Arial,sans-serif" font-size="10" fill="#667085">{html.escape(state)}</text>',
            ]
        )

    for segment in segments:
        state = state_key(segment)
        row = state_to_row[state]
        y = row_y(row) + 8
        h = row_h - 16
        x1 = sx(segment_start(segment))
        x2 = sx(segment_end(segment))
        color = colors[state]
        title_text = (
            f"{state} context {int(segment_start(segment))}-{int(segment_end(segment))} "
            f"mean_tbt={float(segment.get('mean_tbt_ms', 0)):.3f}ms "
            f"energy/token={float(segment.get('energy_mj_per_token', 0)):.3f}mJ"
        )
        parts.extend(
            [
                f'<rect x="{x1:.2f}" y="{y:.2f}" width="{max(1.0, x2-x1):.2f}" height="{h:.2f}" rx="8" fill="{color}" opacity="0.86"/>',
                f'<title>{html.escape(title_text)}</title>',
            ]
        )

    for prev, current in zip(segments, segments[1:]):
        if state_key(prev) == state_key(current):
            continue
        reason = infer_switch_reason(prev, current)
        x = sx(segment_start(current))
        prev_y = row_y(state_to_row[state_key(prev)]) + row_h / 2
        cur_y = row_y(state_to_row[state_key(current)]) + row_h / 2
        top_y = min(prev_y, cur_y)
        bot_y = max(prev_y, cur_y)
        parts.append(f'<line x1="{x:.2f}" y1="{top_y:.2f}" x2="{x:.2f}" y2="{bot_y:.2f}" stroke="#344054" stroke-width="1.4" stroke-dasharray="4 4" opacity="0.62"/>')
        parts.extend(draw_switch_marker(reason, x, (top_y + bot_y) / 2))
        annotation = switch_annotation(current, reason)
        if annotation:
            label_y = max(top + 16, top_y - 8)
            parts.append(
                f'<text x="{x + 10:.2f}" y="{label_y:.2f}" font-family="Inter,Arial,sans-serif" font-size="11" fill="{switch_annotation_color(reason)}">{html.escape(annotation)}</text>'
            )

    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        value = min_x + (max_x - min_x) * frac
        x = sx(value)
        parts.extend(
            [
                f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 6}" stroke="#98A2B3"/>',
                f'<text x="{x:.2f}" y="{top + plot_h + 24}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="12" fill="#475467">{int(round(value))}</text>',
            ]
        )
    parts.extend(
        [
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#98A2B3"/>',
            f'<text x="{left + plot_w / 2:.2f}" y="{height - 28}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="13" fill="#344054">Decode context bucket</text>',
        ]
    )

    legend_x = left + plot_w - 292
    legend_y = 28
    parts.append(f'<g transform="translate({legend_x},{legend_y})">')
    parts.extend(draw_switch_marker("slo", 8, 8))
    parts.append('<text x="24" y="12" font-family="Inter,Arial,sans-serif" font-size="11" fill="#344054">SLO miss / unavailable</text>')
    parts.extend(draw_switch_marker("energy", 160, 8))
    parts.append('<text x="176" y="12" font-family="Inter,Arial,sans-serif" font-size="11" fill="#344054">lower energy</text>')
    parts.append("</g>")
    parts.append("</svg>")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts), encoding="utf-8")


def render_matplotlib(segments: list[dict[str, Any]], output: Path, title: str) -> None:
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    states = ordered_states(segments)
    colors = state_colors(states)
    state_to_row = {state: index for index, state in enumerate(states)}
    fig_h = max(2.6, 1.0 + len(states) * 0.52)
    fig, ax = plt.subplots(figsize=(11.8, fig_h))

    for segment in segments:
        state = state_key(segment)
        row = state_to_row[state]
        x1 = segment_start(segment)
        x2 = segment_end(segment)
        ax.add_patch(
            patches.Rectangle(
                (x1, row - 0.36),
                max(1.0, x2 - x1),
                0.72,
                linewidth=0,
                facecolor=colors[state],
                alpha=0.86,
            )
        )

    for prev, current in zip(segments, segments[1:]):
        if state_key(prev) == state_key(current):
            continue
        reason = infer_switch_reason(prev, current)
        x = segment_start(current)
        y1 = state_to_row[state_key(prev)]
        y2 = state_to_row[state_key(current)]
        ax.plot([x, x], [y1, y2], color="#344054", linewidth=1.2, linestyle=(0, (3, 3)), alpha=0.7)
        marker = "D" if reason == "slo" else "o"
        color = "#D92D20" if reason == "slo" else "#175CD3"
        ax.scatter([x], [(y1 + y2) / 2], marker=marker, s=80, facecolor="white", edgecolor=color, linewidth=1.8, zorder=5)
        annotation = switch_annotation(current, reason)
        if annotation:
            ax.annotate(annotation, (x, min(y1, y2)), textcoords="offset points", xytext=(6, -8), fontsize=8, color=switch_annotation_color(reason))

    ax.set_yticks(range(len(states)))
    ax.set_yticklabels([compact_state_label(next(segment for segment in segments if state_key(segment) == state)) for state in states])
    ax.set_xlabel("Decode context bucket")
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold")
    ax.grid(axis="x", color="#EAECF0")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot decode backend/state time-sharing from simulate_system_benefit.py output.")
    parser.add_argument("--input", required=True, help="Simulation result JSON.")
    parser.add_argument("--output", required=True, help="Output .svg or .png path.")
    parser.add_argument("--plan", choices=["scheduled", "baseline"], default="scheduled")
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    segments = load_segments(input_path, args.plan)
    title = args.title or f"Decode State Time Sharing ({args.plan})"
    if output_path.suffix.lower() == ".png":
        try:
            render_matplotlib(segments, output_path, title)
        except ImportError:
            svg_path = output_path.with_suffix(".svg")
            render_svg(segments, svg_path, title)
            print(f"matplotlib not available; wrote {svg_path}")
            return 0
    else:
        render_svg(segments, output_path, title)
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
