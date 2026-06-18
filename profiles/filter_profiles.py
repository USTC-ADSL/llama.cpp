#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from profiles.offline_profile_lib import (  # noqa: E402
    AGG_FIELDS,
    PARETO_FIELDS,
    RAW_FIELDS,
    STATE_CATALOG_FIELDS,
    aggregate_rows,
    load_config,
    manifests_dir,
    pareto_rows,
    profiles_dir,
    read_csv_rows,
    write_csv_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate raw offline profile CSV and build Pareto state table.")
    parser.add_argument("--config", default="configs/offline_profile.yaml")
    parser.add_argument("--raw", default=None)
    parser.add_argument("--state-catalog", default=None)
    parser.add_argument("--agg", default=None)
    parser.add_argument("--pareto", default=None)
    parser.add_argument("--repeat", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    raw_path = Path(args.raw) if args.raw else profiles_dir(config) / "perf_profile_raw.csv"
    catalog_path = Path(args.state_catalog) if args.state_catalog else profiles_dir(config) / "state_catalog.csv"
    agg_path = Path(args.agg) if args.agg else profiles_dir(config) / "perf_profile_agg.csv"
    pareto_path = Path(args.pareto) if args.pareto else profiles_dir(config) / "pareto_states.csv"
    repeat = args.repeat if args.repeat is not None else int(config.get("repeat", 1))

    raw_rows = read_csv_rows(raw_path, RAW_FIELDS, missing_ok=True)
    catalog_rows = read_csv_rows(catalog_path, STATE_CATALOG_FIELDS, missing_ok=True)
    agg = aggregate_rows(raw_rows, repeat=repeat, manifest_dir=manifests_dir(config))
    pareto = pareto_rows(agg, catalog_rows)

    write_csv_rows(agg_path, AGG_FIELDS, agg)
    write_csv_rows(pareto_path, PARETO_FIELDS, pareto)
    print(f"wrote {agg_path}")
    print(f"wrote {pareto_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
