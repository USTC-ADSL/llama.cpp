from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Optional


def write_artifacts(artifact: Dict[str, Any], output_path: Path | str, summary_path: Optional[Path | str] = None) -> Dict[str, Any]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = build_summary(artifact)
    if summary_path is not None:
        path = Path(summary_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def build_summary(artifact: Dict[str, Any]) -> Dict[str, Any]:
    by_backend_phase: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for state in artifact.get("normalized_states", []):
        by_backend_phase[state.get("backend", "unknown")][state.get("phase", "unknown")] += 1

    frontier_kind_counts = Counter(frontier.get("frontier_kind", "unknown") for frontier in artifact.get("frontiers", []))
    source_rows = {
        source.get("path", "unknown"): source.get("row_count", 0)
        for source in artifact.get("source_files", [])
    }
    return {
        "version": artifact.get("version"),
        "generated_at": artifact.get("generated_at"),
        "input_dir": artifact.get("input_dir"),
        "raw_rows_by_source": source_rows,
        "normalized_states_by_backend_and_phase": {
            backend: dict(phases) for backend, phases in by_backend_phase.items()
        },
        "unstable_states_filtered": artifact.get("raw_profile_summary", {}).get("unstable_state_count", 0),
        "frontiers_generated": len(artifact.get("frontiers", [])),
        "frontier_kind_counts": dict(frontier_kind_counts),
        "transition_edge_count": len(artifact.get("transition_edges", [])),
        "caveats": artifact.get("caveats", []),
        "data_quality_summary": artifact.get("data_quality_summary", {}),
        "source_file_count": len(artifact.get("source_files", [])),
        "skipped_source_count": len(artifact.get("skipped_sources", [])),
    }
