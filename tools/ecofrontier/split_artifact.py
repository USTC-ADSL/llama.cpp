#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_ARTIFACT = Path("build/ecofrontier/ecofrontier_frontier.json")
DEFAULT_SUMMARY = Path("build/ecofrontier/ecofrontier_frontier_summary.json")
DEFAULT_OUTPUT_DIR = Path("Paper_Writing/ecofrontier/review_parts")


def split_artifact(
    artifact_path: Path | str = DEFAULT_ARTIFACT,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    summary_path: Optional[Path | str] = DEFAULT_SUMMARY,
    clean: bool = True,
) -> Dict[str, Any]:
    artifact_file = Path(artifact_path)
    summary_file = Path(summary_path) if summary_path is not None else None
    out_dir = Path(output_dir)

    artifact = _load_json(artifact_file)
    summary = _load_json(summary_file) if summary_file and summary_file.exists() else None

    if clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parts: List[Dict[str, Any]] = []

    def write_part(name: str, payload: Any, category: str, count: Optional[int] = None) -> None:
        path = out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(path, payload)
        parts.append(
            {
                "file": str(path.relative_to(out_dir)),
                "category": category,
                "count": _infer_count(payload) if count is None else count,
            }
        )

    sources_and_config = {
        "source_files": artifact.get("source_files", []),
        "skipped_sources": artifact.get("skipped_sources", []),
        "compiler_config": artifact.get("compiler_config", {}),
        "raw_profile_summary": artifact.get("raw_profile_summary", {}),
    }
    quality_caveats_policy = {
        "caveats": artifact.get("caveats", []),
        "source_caveats": artifact.get("source_caveats", []),
        "data_quality_summary": artifact.get("data_quality_summary", {}),
        "energy_policy": artifact.get("energy_policy"),
    }

    write_part("01_sources_and_config.json", sources_and_config, "metadata")
    write_part("02_quality_caveats_and_policy.json", quality_caveats_policy, "metadata")
    if summary is not None:
        write_part("03_summary_copy.json", summary, "metadata")

    for key, filename in (
        ("models", "04_models.json"),
        ("dominated_states", "06_dominated_states.json"),
        ("transition_edges", "07_transition_edges.json"),
        ("graph_catalog_summary", "08_graph_catalog_summary.json"),
        ("source_slo_frontiers", "09_source_slo_frontiers.json"),
    ):
        payload = artifact.get(key, [] if key != "graph_catalog_summary" else {})
        write_part(filename, payload, key, _infer_count(payload))

    for filename, payload, count in _split_states(artifact.get("normalized_states", [])):
        write_part(f"states/{filename}", payload, "states", count)

    for filename, payload, count in _split_frontiers(artifact.get("frontiers", [])):
        write_part(f"frontiers/{filename}", payload, "frontiers", count)

    manifest = {
        "artifact": str(artifact_file),
        "summary": str(summary_file) if summary_file is not None else None,
        "output_dir": str(out_dir),
        "version": artifact.get("version"),
        "generated_at": artifact.get("generated_at"),
        "input_dir": artifact.get("input_dir"),
        "counts": {
            "source_files": len(artifact.get("source_files", [])),
            "skipped_sources": len(artifact.get("skipped_sources", [])),
            "normalized_states": len(artifact.get("normalized_states", [])),
            "models": len(artifact.get("models", [])),
            "frontiers": len(artifact.get("frontiers", [])),
            "dominated_states": len(artifact.get("dominated_states", [])),
            "transition_edges": len(artifact.get("transition_edges", [])),
            "source_slo_frontiers": len(artifact.get("source_slo_frontiers", [])),
        },
        "parts": sorted(parts, key=lambda item: item["file"]),
        "review_order": [
            "00_manifest.json",
            "01_sources_and_config.json",
            "02_quality_caveats_and_policy.json",
            "03_summary_copy.json",
            "states/*.json",
            "frontiers/*.json",
            "04_models.json",
            "06_dominated_states.json",
            "07_transition_edges.json",
            "08_graph_catalog_summary.json",
            "09_source_slo_frontiers.json",
        ],
    }
    _write_json(out_dir / "00_manifest.json", manifest)
    return manifest


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Split EcoFrontier frontier artifact into review-sized JSON files.")
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT, help="Path to ecofrontier_frontier.json")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY, help="Path to ecofrontier_frontier_summary.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for split review files")
    parser.add_argument("--no-clean", action="store_true", help="Do not remove an existing output directory before writing")
    args = parser.parse_args(argv)

    manifest = split_artifact(args.artifact, args.output_dir, args.summary, clean=not args.no_clean)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "manifest": str(args.output_dir / "00_manifest.json"),
                "part_count": len(manifest["parts"]),
                "counts": manifest["counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _split_states(states: List[Dict[str, Any]]) -> List[tuple[str, List[Dict[str, Any]], int]]:
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for state in states:
        backend = _safe_name(str(state.get("backend", "unknown")))
        phase = _safe_name(str(state.get("phase", "unknown")))
        grouped[(backend, phase)].append(state)
    return [
        (f"{backend}_{phase}.json", rows, len(rows))
        for (backend, phase), rows in sorted(grouped.items())
    ]


def _split_frontiers(frontiers: List[Dict[str, Any]]) -> List[tuple[str, List[Dict[str, Any]], int]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for frontier in frontiers:
        phase = _safe_name(str(frontier.get("phase", "unknown")))
        grouped[phase].append(frontier)
    return [(f"{phase}.json", rows, len(rows)) for phase, rows in sorted(grouped.items())]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _infer_count(payload: Any) -> int:
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        return len(payload)
    return 1


def _safe_name(value: str) -> str:
    allowed = []
    for char in value:
        if char.isalnum() or char in {"_", "-"}:
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_") or "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
