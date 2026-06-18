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
    STATE_CATALOG_FIELDS,
    alphas_from_config,
    build_frontier_rows,
    load_config,
    profiles_dir,
    read_csv_rows,
    write_csv_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build synthetic relative-throughput frontier table.")
    parser.add_argument("--config", default="configs/offline_profile.yaml")
    parser.add_argument("--agg", default=None, help="Accepted for pipeline symmetry; frontier is built from pareto CSV.")
    parser.add_argument("--state-catalog", default=None)
    parser.add_argument("--pareto", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    catalog_path = Path(args.state_catalog) if args.state_catalog else profiles_dir(config) / "state_catalog.csv"
    pareto_path = Path(args.pareto) if args.pareto else profiles_dir(config) / "pareto_states.csv"
    output_path = Path(args.output) if args.output else profiles_dir(config) / "frontier.csv"

    catalog_rows = read_csv_rows(catalog_path, STATE_CATALOG_FIELDS, missing_ok=True)
    pareto = read_csv_rows(pareto_path, PARETO_FIELDS, missing_ok=True)
    rows = build_frontier_rows(pareto, catalog_rows, alphas_from_config(config))
    write_csv_rows(output_path, FRONTIER_FIELDS, rows)
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
