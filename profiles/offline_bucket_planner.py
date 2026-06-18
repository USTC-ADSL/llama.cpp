#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from profiles.offline_profile_lib import (  # noqa: E402
    FRONTIER_FIELDS,
    PARETO_FIELDS,
    REQUEST_PLAN_FIELDS,
    STATE_CATALOG_FIELDS,
    load_config,
    load_transition_map,
    plan_request_rows,
    profiles_dir,
    read_csv_rows,
    write_csv_rows,
)


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay bucketed decode planning with dynamic programming.")
    parser.add_argument("--config", default="configs/offline_profile.yaml")
    parser.add_argument("--agg", default=None, help="Accepted for pipeline symmetry; planner uses pareto/state catalog.")
    parser.add_argument("--frontier", default=None)
    parser.add_argument("--pareto", default=None)
    parser.add_argument("--state-catalog", default=None)
    parser.add_argument("--transition-profile", default=None)
    parser.add_argument("--prompt-len", type=int, required=True)
    parser.add_argument("--output-len", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--include-transition", type=parse_bool, default=False)
    parser.add_argument("--default-transition-energy-mj", type=float, default=0.0)
    parser.add_argument("--default-transition-latency-ms", type=float, default=0.0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    pareto_path = Path(args.pareto) if args.pareto else profiles_dir(config) / "pareto_states.csv"
    frontier_path = Path(args.frontier) if args.frontier else profiles_dir(config) / "frontier.csv"
    catalog_path = Path(args.state_catalog) if args.state_catalog else profiles_dir(config) / "state_catalog.csv"
    transition_path = Path(args.transition_profile) if args.transition_profile else profiles_dir(config) / "transition_profile.csv"

    pareto = read_csv_rows(pareto_path, PARETO_FIELDS, missing_ok=True)
    frontier = read_csv_rows(frontier_path, FRONTIER_FIELDS, missing_ok=True)
    catalog = read_csv_rows(catalog_path, STATE_CATALOG_FIELDS, missing_ok=True)
    transitions = load_transition_map(transition_path)
    rows = plan_request_rows(
        pareto=pareto,
        state_catalog=catalog,
        transitions=transitions,
        config=config,
        prompt_len=args.prompt_len,
        output_len=args.output_len,
        alpha=args.alpha,
        include_transition=args.include_transition,
        default_transition_energy_mj=args.default_transition_energy_mj,
        default_transition_latency_ms=args.default_transition_latency_ms,
        frontier=frontier,
    )
    write_csv_rows(args.output, REQUEST_PLAN_FIELDS, rows)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
