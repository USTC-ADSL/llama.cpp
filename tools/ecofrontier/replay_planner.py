#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.ecofrontier.online_planner import OnlinePlanner, PlannerOptions
from tools.ecofrontier.planner_artifact_loader import generate_request_grid, load_artifact, load_requests
from tools.ecofrontier.planner_trace import write_summary, write_trace


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Replay EcoFrontier online planner decisions from a frontier artifact.")
    parser.add_argument("--artifact", required=True, type=Path, help="Path to ecofrontier_frontier.json")
    parser.add_argument("--requests", type=Path, help="JSON request fixture")
    parser.add_argument("--generate-grid", action="store_true", help="Generate a request grid from artifact buckets")
    parser.add_argument("--trace", required=True, type=Path, help="Output JSONL trace path")
    parser.add_argument("--summary", required=True, type=Path, help="Output replay summary JSON path")
    parser.add_argument("--allow-missing-transition", action="store_true", help="Allow missing sparse transition edges")
    args = parser.parse_args(argv)

    if not args.requests and not args.generate_grid:
        parser.error("provide --requests or --generate-grid")

    artifact = load_artifact(args.artifact)
    requests = load_requests(args.requests) if args.requests else generate_request_grid(artifact)
    planner = OnlinePlanner(artifact, PlannerOptions(allow_missing_transition=args.allow_missing_transition))
    results = [planner.plan(request) for request in requests]
    write_trace(results, args.trace)
    summary = write_summary(results, args.summary)
    print(
        json.dumps(
            {
                "artifact": str(args.artifact),
                "request_count": summary["request_count"],
                "feasible_count": summary["feasible_count"],
                "best_effort_count": summary["best_effort_count"],
                "trace": str(args.trace),
                "summary": str(args.summary),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
