#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


DEFAULT_ARTIFACT = Path("build/ecofrontier/ecofrontier_frontier.json")
DEFAULT_SUMMARY = Path("build/ecofrontier/ecofrontier_frontier_summary.json")


def run_sanity_checks(
    artifact_path: Path | str = DEFAULT_ARTIFACT,
    summary_path: Path | str = DEFAULT_SUMMARY,
    max_matches: int = 20,
) -> Dict[str, Any]:
    artifact_file = Path(artifact_path)
    summary_file = Path(summary_path)
    artifact = _load_json(artifact_file)
    summary = _load_json(summary_file)
    objects = list(_walk_objects(artifact))

    frontier_kind_counts, frontier_kind_source = _frontier_kind_counts(artifact, summary)
    estimated_complete = _matches(
        objects,
        lambda obj: obj.get("energy_source") == "estimated_power_latency" and obj.get("energy_complete") is True,
        max_matches,
    )
    qnn_context_capacity = _matches(
        objects,
        lambda obj: _truthy_jq_value(obj.get("qnn_aot_context_size")) and _truthy_jq_value(obj.get("usable_kv_slots")),
        max_matches,
    )
    forged_transition_energy = _matches(
        objects,
        lambda obj: obj.get("transition_energy_source") == "unavailable"
        and obj.get("transition_energy_mj") is not None
        and obj.get("transition_energy_mj") != "",
        max_matches,
    )
    state_identity = _state_identity_preservation(artifact, objects, max_matches)

    checks = {
        "frontier_kind_counts_present": {
            "ok": bool(frontier_kind_counts),
            "source": frontier_kind_source,
            "counts": frontier_kind_counts,
        },
        "estimated_energy_marked_complete": {
            "ok": estimated_complete["count"] == 0,
            **estimated_complete,
        },
        "qnn_context_with_usable_kv_slots": {
            "ok": qnn_context_capacity["count"] == 0,
            **qnn_context_capacity,
        },
        "forged_transition_energy": {
            "ok": forged_transition_energy["count"] == 0,
            **forged_transition_energy,
        },
        "state_identity_preservation": state_identity,
    }
    ok = all(item.get("ok", False) for item in checks.values())
    return {
        "ok": ok,
        "artifact": str(artifact_file),
        "summary": str(summary_file),
        "frontier_kind_counts": frontier_kind_counts,
        **checks,
    }


def format_text_report(report: Dict[str, Any]) -> str:
    lines = [
        "EcoFrontier artifact sanity check",
        f"artifact: {report['artifact']}",
        f"summary: {report['summary']}",
        "",
        _format_status(
            report["frontier_kind_counts_present"]["ok"],
            f"frontier kind counts ({report['frontier_kind_counts_present']['source']}): "
            f"{json.dumps(report['frontier_kind_counts'], ensure_ascii=False, sort_keys=True)}",
        ),
        _format_status(
            report["estimated_energy_marked_complete"]["ok"],
            f"estimated energy marked complete: {report['estimated_energy_marked_complete']['count']}",
        ),
        _format_status(
            report["qnn_context_with_usable_kv_slots"]["ok"],
            f"qnn_aot_context_size with usable_kv_slots: {report['qnn_context_with_usable_kv_slots']['count']}",
        ),
        _format_status(
            report["forged_transition_energy"]["ok"],
            f"unavailable transition energy with transition_energy_mj: {report['forged_transition_energy']['count']}",
        ),
    ]

    state = report["state_identity_preservation"]
    lines.append(
        _format_status(
            state["ok"],
            "state identity preservation: "
            f"normalized_rows={state['normalized_state_rows']}, "
            f"raw_summary_state_count={state.get('raw_summary_state_count')}, "
            f"unique_state_ids={state['unique_state_ids']}, "
            f"unique_state_tuples={state['unique_state_tuples']}, "
            f"multi_tuple_state_id_count={state['multi_tuple_state_id_count']}",
        )
    )
    if state["multi_tuple_examples"]:
        lines.append("multi_tuple_examples:")
        for item in state["multi_tuple_examples"]:
            lines.append(f"  - {item['state_id']}: tuple_count={item['tuple_count']}")

    for key in (
        "estimated_energy_marked_complete",
        "qnn_context_with_usable_kv_slots",
        "forged_transition_energy",
    ):
        check = report[key]
        if check["count"] == 0:
            continue
        lines.append(f"{key} examples:")
        for match in check["matches"]:
            lines.append(f"  - {match['path']}: {json.dumps(match['object'], ensure_ascii=False, sort_keys=True)}")

    lines.append("")
    lines.append("result: PASS" if report["ok"] else "result: FAIL")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run sanity checks against an EcoFrontier frontier artifact.")
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT, help="Path to ecofrontier_frontier.json")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY, help="Path to ecofrontier_frontier_summary.json")
    parser.add_argument("--max-matches", type=int, default=20, help="Maximum example matches to include per failed check")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON report")
    args = parser.parse_args(argv)

    report = run_sanity_checks(args.artifact, args.summary, max_matches=args.max_matches)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(format_text_report(report))
    return 0 if report["ok"] else 1


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _walk_objects(value: Any, path: str = "$") -> Iterator[Tuple[str, Dict[str, Any]]]:
    if isinstance(value, dict):
        yield path, value
        for key, child in value.items():
            yield from _walk_objects(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_objects(child, f"{path}[{index}]")


def _matches(
    objects: Iterable[Tuple[str, Dict[str, Any]]],
    predicate,
    max_matches: int,
) -> Dict[str, Any]:
    count = 0
    matches: List[Dict[str, Any]] = []
    for path, obj in objects:
        if not predicate(obj):
            continue
        count += 1
        if len(matches) < max_matches:
            matches.append({"path": path, "object": _compact_object(obj)})
    return {"count": count, "matches": matches}


def _frontier_kind_counts(artifact: Dict[str, Any], summary: Dict[str, Any]) -> Tuple[Dict[str, int], str]:
    direct = summary.get("frontier_kind_counts")
    if isinstance(direct, dict) and direct:
        return {str(key): int(value) for key, value in direct.items()}, "summary.frontier_kind_counts"
    nested = summary.get("data_quality_summary", {}).get("frontier_kind_counts")
    if isinstance(nested, dict) and nested:
        return {str(key): int(value) for key, value in nested.items()}, "summary.data_quality_summary.frontier_kind_counts"
    counts = Counter(frontier.get("frontier_kind", "unknown") for frontier in artifact.get("frontiers", []))
    return dict(counts), "artifact.frontiers"


def _state_identity_preservation(
    artifact: Dict[str, Any],
    objects: List[Tuple[str, Dict[str, Any]]],
    max_matches: int,
) -> Dict[str, Any]:
    normalized_states = artifact.get("normalized_states", [])
    raw_summary_count = artifact.get("raw_profile_summary", {}).get("state_count")

    tuples_by_state_id: Dict[str, set[Tuple[Any, Any, Any, Any]]] = defaultdict(set)
    for state in normalized_states:
        if not isinstance(state, dict) or "state_id" not in state:
            continue
        tuples_by_state_id[str(state["state_id"])].add(
            (
                state.get("phase"),
                state.get("context_len"),
                state.get("prompt_tokens"),
                state.get("source_file"),
            )
        )

    all_state_objects = []
    for path, obj in objects:
        if "state_id" not in obj:
            continue
        if len(all_state_objects) < max_matches:
            all_state_objects.append(
                {
                    "path": path,
                    "state_id": obj.get("state_id"),
                    "phase": obj.get("phase"),
                    "context_len": obj.get("context_len"),
                    "prompt_tokens": obj.get("prompt_tokens"),
                    "source_file": obj.get("source_file"),
                }
            )

    multi_tuple = {
        state_id: sorted(tuples)
        for state_id, tuples in tuples_by_state_id.items()
        if len(tuples) > 1
    }
    examples = [
        {
            "state_id": state_id,
            "tuple_count": len(tuples),
            "tuples": [
                {
                    "phase": item[0],
                    "context_len": item[1],
                    "prompt_tokens": item[2],
                    "source_file": item[3],
                }
                for item in tuples[:3]
            ],
        }
        for state_id, tuples in sorted(multi_tuple.items(), key=lambda item: (-len(item[1]), item[0]))[:max_matches]
    ]
    normalized_rows = len(normalized_states)
    raw_count_matches = raw_summary_count is None or raw_summary_count == normalized_rows
    return {
        "ok": normalized_rows > 0 and raw_count_matches,
        "normalized_state_rows": normalized_rows,
        "raw_summary_state_count": raw_summary_count,
        "all_state_id_object_sample": all_state_objects,
        "unique_state_ids": len(tuples_by_state_id),
        "unique_state_tuples": sum(len(tuples) for tuples in tuples_by_state_id.values()),
        "multi_tuple_state_id_count": len(multi_tuple),
        "multi_tuple_examples": examples,
    }


def _compact_object(obj: Dict[str, Any]) -> Dict[str, Any]:
    keep = (
        "state_id",
        "phase",
        "context_len",
        "prompt_tokens",
        "source_file",
        "energy_source",
        "energy_complete",
        "qnn_aot_context_size",
        "usable_kv_slots",
        "from_state_id",
        "to_state_id",
        "transition_energy_source",
        "transition_energy_mj",
    )
    return {key: obj[key] for key in keep if key in obj}


def _truthy_jq_value(value: Any) -> bool:
    return value is not None and value is not False


def _format_status(ok: bool, text: str) -> str:
    return f"{'PASS' if ok else 'FAIL'} {text}"


if __name__ == "__main__":
    raise SystemExit(main())
