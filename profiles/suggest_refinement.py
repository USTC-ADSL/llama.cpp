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
    REFINEMENT_FIELDS,
    STATE_CATALOG_FIELDS,
    load_config,
    profiles_dir,
    read_csv_rows,
    suggest_refinement_rows,
    write_csv_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Suggest adaptive stage-2 refinement points from coarse frontier data.")
    parser.add_argument("--config", default="configs/offline_profile.yaml")
    parser.add_argument("--frontier", default=None)
    parser.add_argument("--pareto", default=None)
    parser.add_argument("--state-catalog", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    frontier_path = Path(args.frontier) if args.frontier else profiles_dir(config) / "frontier.csv"
    pareto_path = Path(args.pareto) if args.pareto else profiles_dir(config) / "pareto_states.csv"
    catalog_path = Path(args.state_catalog) if args.state_catalog else profiles_dir(config) / "state_catalog.csv"
    output_path = Path(args.output) if args.output else profiles_dir(config) / "refinement_plan.csv"

    frontier = read_csv_rows(frontier_path, FRONTIER_FIELDS, missing_ok=True)
    pareto = read_csv_rows(pareto_path, PARETO_FIELDS, missing_ok=True)
    catalog = read_csv_rows(catalog_path, STATE_CATALOG_FIELDS, missing_ok=True)
    rows = suggest_refinement_rows(frontier, pareto, catalog, config)
    write_csv_rows(output_path, REFINEMENT_FIELDS, rows)
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
