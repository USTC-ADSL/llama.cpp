#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.ecofrontier.artifact_writer import write_artifacts
from tools.ecofrontier.frontier_compiler import compile_frontier
from tools.ecofrontier.profile_loader import load_input_dir
from tools.ecofrontier.profile_schema import CompilerConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile EcoFrontier offline frontier artifacts.")
    parser.add_argument("--input", required=True, type=Path, help="Input profile directory, e.g. docs/实验结果")
    parser.add_argument("--output", required=True, type=Path, help="Output ecofrontier_frontier.json path")
    parser.add_argument("--summary", type=Path, help="Output ecofrontier_frontier_summary.json path")
    parser.add_argument("--config", type=Path, help="Optional compiler config JSON")
    parser.add_argument("--paper-output-dir", type=Path, help="Optional directory for paper-writing copies")
    args = parser.parse_args(argv)

    config = _load_config(args.config)
    loaded = load_input_dir(args.input, config)
    artifact = compile_frontier(loaded, config)
    summary = write_artifacts(artifact, args.output, args.summary)

    if args.paper_output_dir:
        args.paper_output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.output, args.paper_output_dir / args.output.name)
        if args.summary:
            shutil.copy2(args.summary, args.paper_output_dir / args.summary.name)

    print(
        json.dumps(
            {
                "output": str(args.output),
                "summary": str(args.summary) if args.summary else None,
                "states": len(artifact["normalized_states"]),
                "frontiers": len(artifact["frontiers"]),
                "transition_edges": len(artifact["transition_edges"]),
                "caveats": artifact["caveats"],
                "frontier_kind_counts": summary["frontier_kind_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _load_config(path: Path | None) -> CompilerConfig:
    if path is None:
        return CompilerConfig()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("--config must point to a JSON object")
    return CompilerConfig.from_mapping(data)


if __name__ == "__main__":
    raise SystemExit(main())
