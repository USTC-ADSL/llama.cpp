#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from profiles.offline_profile_lib import (  # noqa: E402
    TRANSITION_FIELDS,
    catalog_by_state,
    ensure_output_dirs,
    load_config,
    profiles_dir,
    read_csv_rows,
    render_command,
    slugify,
    state_catalog_rows,
    write_csv_rows,
)


MINIMAL_TRANSITIONS = [
    ("npu_low_balanced_cap2048", "npu_burst_cap2048", "NPU workpoint switch"),
    ("npu_burst_cap2048", "npu_low_balanced_cap2048", "NPU workpoint switch"),
    ("npu_balanced_cap2048", "npu_burst_cap2048", "NPU workpoint switch"),
    ("npu_burst_cap2048", "npu_balanced_cap2048", "NPU workpoint switch"),
    ("npu_burst_cap2048", "npu_burst_cap4096", "QNN graph load/switch"),
    ("npu_burst_cap4096", "npu_burst_cap6144", "QNN graph load/switch"),
    ("cold", "npu_burst_cap2048", "cold load cap2048"),
    ("cold", "npu_burst_cap4096", "cold load cap4096"),
    ("cold", "npu_burst_cap6144", "cold load cap6144"),
    ("warm", "npu_burst_cap2048", "warm load cap2048"),
    ("warm", "npu_burst_cap4096", "warm load cap4096"),
    ("warm", "npu_burst_cap6144", "warm load cap6144"),
    ("npu_burst_cap2048", "cpu_B2_2649", "Backend switch NPU -> CPU"),
    ("cpu_B2_2649", "npu_burst_cap2048", "Backend switch CPU -> NPU"),
    ("npu_burst_cap2048", "gpu_734", "Backend switch NPU -> GPU"),
    ("gpu_734", "npu_burst_cap2048", "Backend switch GPU -> NPU"),
    ("cpu_B2_2649", "gpu_734", "Backend switch CPU -> GPU"),
    ("gpu_734", "cpu_B2_2649", "Backend switch GPU -> CPU"),
    ("cpu_B1_2649", "cpu_B2_2649", "CPU state switch B1 -> B2"),
    ("cpu_B2_1804", "cpu_B2_2649", "CPU state switch B2 lowfreq -> highfreq"),
    ("gpu_305", "gpu_734", "GPU frequency switch low -> high"),
    ("gpu_734", "gpu_305", "GPU frequency switch high -> low"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transition-profile framework entrypoint.")
    parser.add_argument("--config", default="configs/offline_profile.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def write_todo(path: Path, config_path: str, transitions: list[tuple[str, str, str]]) -> None:
    lines = [
        "# Transition Profile TODO",
        "",
        "No fake transition rows should be written. Add rows to `profiles/transition_profile.csv` only after a real measurement.",
        "",
        "Expected CSV schema:",
        "",
        "```csv",
        ",".join(TRANSITION_FIELDS),
        "```",
        "",
        "Command template variables:",
        "",
        "```text",
        "{from_state} {to_state} {from_backend} {to_backend} {from_graph_capacity} {to_graph_capacity} {from_workpoint} {to_workpoint} {run_id} {output_json} {log_path}",
        "```",
        "",
        "Manual command template to adapt in `configs/offline_profile.yaml`:",
        "",
        "```yaml",
        "transition_command_template: >-",
        "  YOUR_TRANSITION_MEASUREMENT_TOOL",
        "  --from-state {from_state}",
        "  --to-state {to_state}",
        "  --from-backend {from_backend}",
        "  --to-backend {to_backend}",
        "  --from-graph-capacity {from_graph_capacity}",
        "  --to-graph-capacity {to_graph_capacity}",
        "  --from-workpoint {from_workpoint}",
        "  --to-workpoint {to_workpoint}",
        "  --run-id {run_id}",
        "  --output-json {output_json}",
        "  > {log_path} 2>&1",
        "```",
        "",
        "After adapting the template, preview or run with:",
        "",
        "```bash",
        "scripts/run_transition_profile.sh --dry-run --resume",
        "scripts/run_transition_profile.sh --resume",
        "```",
        "",
        f"Config: `{config_path}`",
        "",
        "Minimum transition set:",
        "",
    ]
    for from_state, to_state, reason in transitions:
        lines.append(f"- `{from_state}` -> `{to_state}`: {reason}")
    lines.extend(
        [
            "",
            "Expected output JSON from a future transition command:",
            "",
            "```json",
            '{"latency_ms": 1.23, "energy_mj": 4.56, "stable": true, "notes": "measured on device"}',
            "```",
            "",
            "Reasons this remains TODO by default:",
            "",
            "- CPU affinity/frequency control is device-specific.",
            "- NPU workpoint and QNN graph switching APIs are platform-specific.",
            "- GPU frequency control may require privileged sysfs or vendor tooling.",
            "- Power/energy sampling source is not standardized in this repository.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    transition_path = profiles_dir(config) / "transition_profile.csv"

    todo_path = Path("scripts") / "todo_transition_tests.md"
    if not args.dry_run:
        ensure_output_dirs(config)
        if not transition_path.exists():
            write_csv_rows(transition_path, TRANSITION_FIELDS, [])
        write_todo(todo_path, args.config, MINIMAL_TRANSITIONS)

    template = str(config.get("transition_command_template") or "")
    catalog = catalog_by_state(state_catalog_rows(config))
    if not template:
        for from_state, to_state, reason in MINIMAL_TRANSITIONS:
            print(f"[todo] {from_state} -> {to_state}: {reason}")
        print(f"transition_csv={transition_path}")
        print(f"todo={todo_path}")
        if args.dry_run:
            print("[dry-run] no transition CSV or TODO files written")
        return 0

    for run_id, (from_state, to_state, reason) in enumerate(MINIMAL_TRANSITIONS):
        variables = {
            "from_state": from_state,
            "to_state": to_state,
            "from_backend": catalog.get(from_state, {}).get("backend", ""),
            "to_backend": catalog.get(to_state, {}).get("backend", ""),
            "from_graph_capacity": catalog.get(from_state, {}).get("qnn_graph_capacity", ""),
            "to_graph_capacity": catalog.get(to_state, {}).get("qnn_graph_capacity", ""),
            "from_workpoint": catalog.get(from_state, {}).get("npu_workpoint", ""),
            "to_workpoint": catalog.get(to_state, {}).get("npu_workpoint", ""),
            "run_id": str(run_id),
            "output_json": str(profiles_dir(config) / f"transition_{slugify(from_state)}_to_{slugify(to_state)}.json"),
            "log_path": str(profiles_dir(config) / f"transition_{slugify(from_state)}_to_{slugify(to_state)}.log"),
            "reason": reason,
        }
        command = render_command(template, variables)
        print(f"[dry-run]" if args.dry_run else "[todo-runner]", command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
