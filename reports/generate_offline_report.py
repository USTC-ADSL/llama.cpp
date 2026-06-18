#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from profiles.offline_profile_lib import (  # noqa: E402
    AGG_FIELDS,
    FRONTIER_FIELDS,
    PARETO_FIELDS,
    RAW_FIELDS,
    REFINEMENT_FIELDS,
    REQUEST_PLAN_FIELDS,
    STATE_CATALOG_FIELDS,
    alphas_from_config,
    aggregate_rows,
    bool_text,
    figures_dir,
    fmt_alpha,
    fmt_float,
    load_config,
    profiles_dir,
    read_csv_rows,
    reports_dir,
    to_float,
    to_int,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Markdown report for offline profiling pipeline.")
    parser.add_argument("--config", default="configs/offline_profile.yaml")
    parser.add_argument("--raw", default=None)
    parser.add_argument("--agg", default=None)
    parser.add_argument("--pareto", default=None)
    parser.add_argument("--frontier", default=None)
    parser.add_argument("--refinement", default=None)
    parser.add_argument("--request-plan", default=None)
    parser.add_argument("--state-catalog", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def md_table(headers: list[str], rows: list[list[Any]], empty: str = "_No rows._") -> str:
    if not rows:
        return empty
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _read_any_csv(path: Path | None, fields: list[str]) -> list[dict[str, str]]:
    if path is None:
        return []
    return read_csv_rows(path, fields, missing_ok=True)


def _fastest_rows(pareto: list[dict[str, str]]) -> list[list[str]]:
    grouped: dict[tuple[str, str], dict[str, str]] = {}
    for row in pareto:
        if row.get("stable") != "true":
            continue
        key = (row.get("phase", ""), row.get("length", ""))
        tps = to_float(row.get("throughput_worst_tps")) or -1.0
        current = grouped.get(key)
        if current is None or tps > (to_float(current.get("throughput_worst_tps")) or -1.0):
            grouped[key] = row
    return [
        [
            phase,
            length,
            row.get("state_name", ""),
            row.get("backend", ""),
            row.get("throughput_worst_tps", ""),
            row.get("mean_tbt_ms", ""),
        ]
        for (phase, length), row in sorted(grouped.items(), key=lambda item: (item[0][0], int(item[0][1] or 0)))
    ]


def _pareto_summary_rows(pareto: list[dict[str, str]]) -> list[list[str]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in pareto:
        grouped[(row.get("phase", ""), row.get("length", ""))].append(row)
    rows = []
    for (phase, length), items in sorted(grouped.items(), key=lambda item: (item[0][0], int(item[0][1] or 0))):
        frontier = [row.get("state_name", "") for row in items if row.get("stable") == "true" and row.get("dominated") != "true"]
        dominated_count = sum(1 for row in items if row.get("dominated") == "true")
        rows.append([phase, length, ", ".join(frontier), str(dominated_count)])
    return rows


def _relative_target_rows(frontier: list[dict[str, str]]) -> list[list[str]]:
    rows = []
    fastest_energy: dict[tuple[str, str], float] = {}
    for row in frontier:
        if row.get("alpha") == "1.0" and row.get("selected") == "true":
            energy = to_float(row.get("energy_per_token_mj"))
            if energy is not None:
                fastest_energy[(row.get("phase", ""), row.get("length", ""))] = energy
    for row in frontier:
        if row.get("phase") != "decode":
            continue
        energy = to_float(row.get("energy_per_token_mj"))
        base = fastest_energy.get((row.get("phase", ""), row.get("length", "")))
        if energy is not None and base and base > 0:
            saving = max(0.0, 100.0 * (1.0 - energy / base))
            saving_text = f"{saving:.2f}%"
        else:
            saving_text = ""
        rows.append(
            [
                row.get("length", ""),
                row.get("alpha", ""),
                row.get("qmax_tps", ""),
                row.get("target_tps", ""),
                row.get("state_name", "") or row.get("filtered_reason", ""),
                row.get("backend", ""),
                row.get("energy_per_token_mj", ""),
                saving_text,
            ]
        )
    return rows


def _coverage_rows(catalog: list[dict[str, str]], raw: list[dict[str, str]]) -> list[list[str]]:
    states_by_backend = Counter(row.get("backend", "") for row in catalog if row.get("backend"))
    raw_status = Counter(row.get("status", "") for row in raw)
    return [
        ["NPU", states_by_backend.get("NPU", 0), raw_status.get("failed", 0), raw_status.get("skipped", 0)],
        ["CPU", states_by_backend.get("CPU", 0), raw_status.get("failed", 0), raw_status.get("skipped", 0)],
        ["GPU", states_by_backend.get("GPU", 0), raw_status.get("failed", 0), raw_status.get("skipped", 0)],
    ]


def _parse_npu_state(state_name: str) -> tuple[str, str] | None:
    if not state_name.startswith("npu_") or "_cap" not in state_name:
        return None
    prefix, capacity = state_name.rsplit("_cap", 1)
    return prefix.removeprefix("npu_"), capacity


def _qnn_sanity_rows(config: dict[str, Any], agg: list[dict[str, str]]) -> list[list[str]]:
    sanity_path = profiles_dir(config) / "qnn_large_graph_sanity.csv"
    if not sanity_path.exists():
        return [["not present", "", "", "", "", "", "run `scripts/run_npu_profile.sh --sanity` if needed"]]

    raw_rows = read_csv_rows(sanity_path, RAW_FIELDS, missing_ok=True)
    if not any(row.get("status") == "ok" for row in raw_rows):
        return [["present", "", "", "", "", "", "no ok sanity rows yet"]]

    sanity_agg = aggregate_rows(raw_rows, repeat=1)
    baseline_by_key = {
        (row.get("phase", ""), row.get("length", ""), row.get("state_name", "")): row
        for row in agg
        if row.get("status") == "ok"
    }
    rows: list[list[str]] = []
    for row in sanity_agg:
        if row.get("phase") != "decode" or row.get("status") not in {"ok", "unstable", "insufficient_runs"}:
            continue
        length = to_int(row.get("length"))
        parsed = _parse_npu_state(row.get("state_name", ""))
        if length is None or length > 2048 or parsed is None:
            continue
        workpoint, capacity = parsed
        if capacity not in {"4096", "6144"}:
            continue
        baseline = baseline_by_key.get(("decode", row.get("length", ""), f"npu_{workpoint}_cap2048"))
        sanity_tps = to_float(row.get("throughput_worst_tps"))
        baseline_tps = to_float(baseline.get("throughput_worst_tps")) if baseline else None
        if sanity_tps is not None and baseline_tps is not None and baseline_tps > 0:
            delta = 100.0 * (sanity_tps / baseline_tps - 1.0)
            note = "loss" if delta < 0 else "no loss"
            delta_text = f"{delta:.2f}%"
        else:
            note = "missing cap2048 baseline"
            delta_text = ""
        rows.append(
            [
                f"cap{capacity}",
                str(length),
                workpoint,
                row.get("throughput_worst_tps", ""),
                baseline.get("throughput_worst_tps", "") if baseline else "",
                delta_text,
                note,
            ]
        )
    return rows or [["present", "", "", "", "", "", "no short-context large graph ok rows yet"]]


def _planner_rows(plan: list[dict[str, str]]) -> list[list[str]]:
    return [
        [
            row.get("segment_id", ""),
            f"[{row.get('bucket_lo', '')}, {row.get('bucket_hi', '')})",
            row.get("num_tokens", ""),
            row.get("selected_state", ""),
            row.get("transition_from_prev", ""),
            row.get("transition_latency_ms", ""),
            row.get("transition_energy_mj", ""),
            row.get("notes", ""),
        ]
        for row in plan
    ]


def _generate_figures(
    config: dict[str, Any],
    pareto: list[dict[str, str]],
    frontier: list[dict[str, str]],
    refinement: list[dict[str, str]],
) -> list[str]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return []

    fig_dir = figures_dir(config)
    fig_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    def save(name: str) -> None:
        path = fig_dir / name
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        generated.append(str(path))

    rows_1024 = [row for row in pareto if row.get("phase") == "decode" and row.get("length") == "1024"]
    if rows_1024:
        plt.figure(figsize=(6, 4))
        for backend in sorted({row.get("backend", "") for row in rows_1024}):
            xs = [to_float(row.get("mean_tbt_ms")) for row in rows_1024 if row.get("backend") == backend]
            ys = [to_float(row.get("energy_per_token_mj")) for row in rows_1024 if row.get("backend") == backend]
            xs2 = [x for x in xs if x is not None]
            ys2 = [y for y in ys if y is not None]
            if xs2 and ys2:
                plt.scatter(xs2, ys2, label=backend)
        plt.xlabel("mean TBT ms")
        plt.ylabel("energy/token mJ")
        plt.title("State frontier at context 1024")
        plt.legend()
        save("state_frontier_context_1024.png")

    decode_frontier = [row for row in frontier if row.get("phase") == "decode" and row.get("selected") == "true"]
    if decode_frontier:
        lengths = sorted({row.get("length", "") for row in decode_frontier}, key=lambda x: int(x or 0))
        alphas = [fmt_alpha(alpha) for alpha in alphas_from_config(config)]
        table = [["" for _ in lengths] for _ in alphas]
        for row in decode_frontier:
            if row.get("alpha") in alphas and row.get("length") in lengths:
                table[alphas.index(row["alpha"])][lengths.index(row["length"])] = row.get("state_name", "")
        plt.figure(figsize=(max(6, len(lengths)), 3))
        plt.imshow([[1 if cell else 0 for cell in line] for line in table], cmap="Greens", aspect="auto")
        plt.xticks(range(len(lengths)), lengths)
        plt.yticks(range(len(alphas)), alphas)
        for y, line in enumerate(table):
            for x, cell in enumerate(line):
                plt.text(x, y, cell.replace("_", "\n"), ha="center", va="center", fontsize=7)
        plt.xlabel("length bucket")
        plt.ylabel("alpha")
        plt.title("Selected state heatmap")
        save("selected_state_heatmap.png")

    for backend, name in [("NPU", "npu_workpoint_frontier.png"), ("CPU", "cpu_affinity_frontier.png")]:
        rows = [row for row in pareto if row.get("phase") == "decode" and row.get("backend") == backend]
        if rows:
            plt.figure(figsize=(7, 4))
            for length in sorted({row.get("length", "") for row in rows}, key=lambda x: int(x or 0)):
                xs = [to_float(row.get("throughput_worst_tps")) for row in rows if row.get("length") == length]
                ys = [to_float(row.get("energy_per_token_mj")) for row in rows if row.get("length") == length]
                xs2 = [x for x in xs if x is not None]
                ys2 = [y for y in ys if y is not None]
                if xs2 and ys2:
                    plt.scatter(xs2, ys2, label=length)
            plt.xlabel("worst throughput tps")
            plt.ylabel("energy/token mJ")
            plt.title(f"{backend} throughput-energy frontier")
            plt.legend(title="length")
            save(name)

    if refinement:
        plt.figure(figsize=(7, 2.5))
        for row in refinement:
            lo = to_int(row.get("interval_lo"))
            hi = to_int(row.get("interval_hi"))
            y = 1 if row.get("priority") == "high" else 0
            if lo is not None and hi is not None:
                plt.plot([lo, hi], [y, y], linewidth=4)
                plt.text((lo + hi) / 2.0, y + 0.05, row.get("reason", ""), ha="center", fontsize=7)
        plt.yticks([0, 1], ["medium", "high"])
        plt.xlabel("context interval")
        plt.title("Refinement plan")
        save("refinement_plan.png")

    return generated


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    raw_path = Path(args.raw) if args.raw else profiles_dir(config) / "perf_profile_raw.csv"
    agg_path = Path(args.agg) if args.agg else profiles_dir(config) / "perf_profile_agg.csv"
    pareto_path = Path(args.pareto) if args.pareto else profiles_dir(config) / "pareto_states.csv"
    frontier_path = Path(args.frontier) if args.frontier else profiles_dir(config) / "frontier.csv"
    refinement_path = Path(args.refinement) if args.refinement else profiles_dir(config) / "refinement_plan.csv"
    request_plan_path = Path(args.request_plan) if args.request_plan else profiles_dir(config) / "request_plan_example.csv"
    catalog_path = Path(args.state_catalog) if args.state_catalog else profiles_dir(config) / "state_catalog.csv"
    output_path = Path(args.output) if args.output else reports_dir(config) / "offline_profile_summary.md"

    raw = _read_any_csv(raw_path, RAW_FIELDS)
    agg = _read_any_csv(agg_path, AGG_FIELDS)
    pareto = _read_any_csv(pareto_path, PARETO_FIELDS)
    frontier = _read_any_csv(frontier_path, FRONTIER_FIELDS)
    refinement = _read_any_csv(refinement_path, REFINEMENT_FIELDS)
    plan = _read_any_csv(request_plan_path, REQUEST_PLAN_FIELDS)
    catalog = _read_any_csv(catalog_path, STATE_CATALOG_FIELDS)
    figures = _generate_figures(config, pareto, frontier, refinement)

    raw_status = Counter(row.get("status", "") for row in raw)
    agg_status = Counter(row.get("status", "") for row in agg)
    missing_energy = sum(1 for row in agg if row.get("status") == "ok" and not row.get("energy_per_token_mj"))
    hardware_note = (
        "Pipeline ready; hardware experiments not run in this report."
        if raw_status.get("ok", 0) == 0
        else "Hardware profile rows are present; verify device metadata before treating the report as final."
    )

    lines = [
        "# Offline Profile Summary",
        "",
        "This report summarizes a configurable offline profiling pipeline for context-aware backend-state characterization.",
        "The synthetic relative throughput target is not a p95 TBT guarantee and not a real user SLO; it is a characterization target derived from the fastest stable state in each context bucket.",
        hardware_note,
        "",
        "## 1. Experiment Configuration",
        "",
        md_table(
            ["item", "value"],
            [
                ["model", config.get("model_path", "") or "_not configured_"],
                ["tokenizer", config.get("tokenizer_path", "") or "_not configured_"],
                ["repeat", config.get("repeat", "")],
                ["context points", ", ".join(str(x) for x in config.get("context_points", []))],
                ["buckets", "; ".join(f"[{lo}, {hi})" for lo, hi in config.get("buckets", []))],
                ["decode probe tokens", config.get("decode_probe_tokens", "")],
                ["idle power mW", config.get("idle_power_mw", "")],
                ["thermal policy", config.get("thermal_policy", "")],
            ],
        ),
        "",
        "NPU graph tier rule: cap2048 represents sub-2048 contexts; cap4096 and cap6144 are profiled as large and xlarge tiers. cap512/cap1024 are intentionally excluded from the main sweep because pre-experiments already validated them as throughput-equivalent to cap2048 in capacity-feasible sub-2048 contexts.",
        "",
        "## 2. QNN Graph Capacity Reduction",
        "",
        "- cap512/cap1024/cap2048 are treated as throughput-equivalent for sub-2048 capacity-feasible contexts based on prior pre-experiments.",
        "- Main profile uses cap2048 as representative graph for sub-2048 contexts.",
        "- cap4096 and cap6144 remain separate large/xlarge graph tiers.",
        "- `profiles/qnn_large_graph_sanity.csv` is optional; if present, use it to check whether large graphs lose short-context performance.",
        "",
        md_table(
            ["graph", "length", "workpoint", "sanity worst tps", "cap2048 worst tps", "delta", "note"],
            _qnn_sanity_rows(config, agg),
        ),
        "",
        "## 3. Profile Coverage",
        "",
        md_table(["backend", "catalog states", "failed raw rows", "skipped raw rows"], _coverage_rows(catalog, raw)),
        "",
        "## 4. Stability Summary",
        "",
        md_table(
            ["metric", "count"],
            [
                ["ok raw runs", raw_status.get("ok", 0)],
                ["failed raw runs", raw_status.get("failed", 0)],
                ["skipped raw runs", raw_status.get("skipped", 0)],
                ["unstable states", agg_status.get("unstable", 0)],
                ["throttled states", agg_status.get("throttled", 0)],
                ["insufficient runs", agg_status.get("insufficient_runs", 0)],
                ["ok states missing energy/token", missing_energy],
            ],
        ),
        "",
        "## 5. Performance Frontier Summary",
        "",
        "Fastest stable state per length:",
        "",
        md_table(["phase", "length", "state", "backend", "worst tps", "mean TBT ms"], _fastest_rows(pareto)),
        "",
        "Pareto frontier states:",
        "",
        md_table(["phase", "length", "non-dominated states", "dominated count"], _pareto_summary_rows(pareto)),
        "",
        "## 6. Relative Target Summary",
        "",
        md_table(
            ["length", "alpha", "Qmax tps", "target tps", "selected state", "backend", "energy/token mJ", "energy saving vs fastest"],
            _relative_target_rows(frontier),
        ),
        "",
        "## 7. Adaptive Refinement Summary",
        "",
        md_table(
            ["priority", "interval", "suggested length", "reason", "alpha", "states to retest"],
            [
                [
                    row.get("priority", ""),
                    f"[{row.get('interval_lo', '')}, {row.get('interval_hi', '')}]",
                    row.get("suggested_length", ""),
                    row.get("reason", ""),
                    row.get("affected_alpha", ""),
                    row.get("states_to_retest", ""),
                ]
                for row in refinement
            ],
        ),
        "",
        "Current coarse points are sufficient only where no refinement rows are emitted and selected-state margins are not near target.",
        "",
        "## 8. Offline Planner Summary",
        "",
        md_table(
            ["segment", "bucket", "tokens", "state", "transition from", "transition latency ms", "transition energy mJ", "notes"],
            _planner_rows(plan),
        ),
        "",
        "When transition profiles are missing, transition-aware replay must use configured conservative defaults and mark `missing_transition_profile`; no-transition replay must mark `no_transition_model`.",
        "",
        "## 9. Figures",
        "",
    ]
    if figures:
        lines.extend(f"- `{path}`" for path in figures)
    else:
        lines.append("_matplotlib unavailable or insufficient numeric data; CSV and Markdown generation still succeeded._")

    lines.extend(
        [
            "",
            "## 10. TODO",
            "",
            "- Fill `decode_command_template` and `prefill_command_template` with real device measurement commands.",
            "- Add device-specific CPU affinity and CPU frequency control, or keep those fields as log-only TODO.",
            "- Add NPU workpoint and QNN graph selection commands without hardcoding platform assumptions.",
            "- Add GPU frequency control only if the current device exposes a safe interface.",
            "- Add power/energy sampling command or output JSON fields; idle-subtracted energy can be derived from active power and elapsed time.",
            "- Complete real transition measurements in `profiles/transition_profile.csv`; do not insert fake transition rows.",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
