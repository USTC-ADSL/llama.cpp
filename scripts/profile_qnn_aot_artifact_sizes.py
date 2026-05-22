#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT_DIR = Path(__file__).resolve().parents[1]


PROFILE_HEADER = [
    "date",
    "artifact_id",
    "config_path",
    "artifact_dir",
    "cache_sizes",
    "context_sizes",
    "batch_sizes",
    "graph_count",
    "transformer_graph_count",
    "embedding_count",
    "transformer_model_files",
    "embedding_model_files",
    "missing_model_files",
    "config_bytes",
    "transformer_bin_bytes",
    "embedding_bin_bytes",
    "referenced_bin_bytes",
    "kv_file_count",
    "kv_bytes",
    "total_referenced_bytes",
    "total_artifact_bytes",
    "largest_file_bytes",
    "largest_file_path",
    "comparison_group",
    "support_status",
    "notes",
]


COMPARISON_HEADER = [
    "date",
    "comparison_group",
    "base_artifact_id",
    "target_artifact_id",
    "base_cache_sizes",
    "target_cache_sizes",
    "base_context_sizes",
    "target_context_sizes",
    "base_total_referenced_bytes",
    "target_total_referenced_bytes",
    "delta_bytes",
    "growth_pct",
    "base_config_path",
    "target_config_path",
]


@dataclass(frozen=True)
class ArtifactProfile:
    date: str
    artifact_id: str
    config_path: str
    artifact_dir: str
    cache_sizes: str
    context_sizes: str
    batch_sizes: str
    graph_count: int
    transformer_graph_count: int
    embedding_count: int
    transformer_model_files: str
    embedding_model_files: str
    missing_model_files: str
    config_bytes: int
    transformer_bin_bytes: int
    embedding_bin_bytes: int
    referenced_bin_bytes: int
    kv_file_count: int
    kv_bytes: int
    total_referenced_bytes: int
    total_artifact_bytes: int
    largest_file_bytes: int
    largest_file_path: str
    comparison_group: str
    support_status: str
    notes: str


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_DIR,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unavailable"


def resolve_config_path(path: Path) -> Path:
    if path.is_dir():
        config = path / "config.json"
        if config.exists():
            return config
        matches = sorted(path.glob("*.json"))
        if len(matches) == 1:
            return matches[0]
        raise FileNotFoundError(f"cannot resolve a unique QNN config under {path}")
    return path


def file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() and path.is_file() else 0


def all_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [path for path in root.rglob("*") if path.is_file()]


def sum_sizes(paths: Iterable[Path]) -> int:
    return sum(path.stat().st_size for path in set(paths) if path.exists() and path.is_file())


def join_ints(values: Iterable[object]) -> str:
    ints = sorted({int(value) for value in values if value not in (None, "")})
    return " ".join(str(value) for value in ints)


def join_paths(paths: Iterable[Path], base: Path) -> str:
    normalized = []
    for path in sorted(set(paths)):
        try:
            normalized.append(str(path.relative_to(base)))
        except ValueError:
            normalized.append(str(path))
    return " ".join(normalized)


def comparison_group_for(config: dict) -> str:
    graphs = config.get("graphs", [])
    embeddings = config.get("embeddings", [])
    graph_shape = [
        {
            "type": graph.get("type"),
            "start_layer_id": graph.get("start_layer_id"),
            "end_layer_id": graph.get("end_layer_id"),
            "batch_size": graph.get("batch_size"),
            "graph_name": graph.get("graph_name"),
        }
        for graph in graphs
        if isinstance(graph, dict)
    ]
    embedding_shape = [
        {
            "batch_size": embedding.get("batch_size"),
            "graph_name": embedding.get("graph_name"),
        }
        for embedding in embeddings
        if isinstance(embedding, dict)
    ]
    payload = {
        "model_parameters": config.get("model_parameters", {}),
        "graph_shape": sorted(graph_shape, key=lambda item: json.dumps(item, sort_keys=True)),
        "embedding_shape": sorted(embedding_shape, key=lambda item: json.dumps(item, sort_keys=True)),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def profile_artifact(config_path: Path | str, artifact_id: str | None = None, date: str | None = None) -> ArtifactProfile:
    config_path = resolve_config_path(Path(config_path)).resolve()
    artifact_dir = config_path.parent
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    graphs = [graph for graph in config.get("graphs", []) if isinstance(graph, dict)]
    transformers = [graph for graph in graphs if graph.get("type") == "transformers"]
    embeddings = [embedding for embedding in config.get("embeddings", []) if isinstance(embedding, dict)]

    transformer_paths = [
        artifact_dir / graph["model_path"]
        for graph in transformers
        if graph.get("model_path")
    ]
    embedding_paths = [
        artifact_dir / embedding["model_path"]
        for embedding in embeddings
        if embedding.get("model_path")
    ]
    referenced_paths = set(transformer_paths) | set(embedding_paths)
    missing_paths = [path for path in sorted(referenced_paths) if not path.exists()]

    kv_paths = all_files(artifact_dir / "kv")
    artifact_files = all_files(artifact_dir)
    largest_file = max(artifact_files, key=lambda path: path.stat().st_size, default=None)

    transformer_existing = [path for path in transformer_paths if path.exists()]
    embedding_existing = [path for path in embedding_paths if path.exists()]
    referenced_existing = [path for path in referenced_paths if path.exists()]
    total_referenced_bytes = sum_sizes(referenced_existing) + sum_sizes(kv_paths)

    status = "ok" if not missing_paths else "missing_referenced_files"
    notes = ""
    if len({graph.get("context_size") for graph in transformers if graph.get("context_size")}) > 1:
        notes = "mixed_context_sizes_in_single_config"

    return ArtifactProfile(
        date=date or utc_timestamp(),
        artifact_id=artifact_id or default_artifact_id(config_path),
        config_path=str(config_path),
        artifact_dir=str(artifact_dir),
        cache_sizes=join_ints(graph.get("cache_size") for graph in transformers),
        context_sizes=join_ints(graph.get("context_size") for graph in transformers),
        batch_sizes=join_ints(graph.get("batch_size") for graph in transformers),
        graph_count=len(graphs),
        transformer_graph_count=len(transformers),
        embedding_count=len(embeddings),
        transformer_model_files=join_paths(transformer_existing, artifact_dir),
        embedding_model_files=join_paths(embedding_existing, artifact_dir),
        missing_model_files=join_paths(missing_paths, artifact_dir),
        config_bytes=file_size(config_path),
        transformer_bin_bytes=sum_sizes(transformer_existing),
        embedding_bin_bytes=sum_sizes(embedding_existing),
        referenced_bin_bytes=sum_sizes(referenced_existing),
        kv_file_count=len(kv_paths),
        kv_bytes=sum_sizes(kv_paths),
        total_referenced_bytes=total_referenced_bytes,
        total_artifact_bytes=sum_sizes(artifact_files),
        largest_file_bytes=largest_file.stat().st_size if largest_file else 0,
        largest_file_path=str(largest_file.relative_to(artifact_dir)) if largest_file else "",
        comparison_group=comparison_group_for(config),
        support_status=status,
        notes=notes,
    )


def default_artifact_id(config_path: Path) -> str:
    artifact_dir = config_path.parent
    if artifact_dir.name == "qnn" and artifact_dir.parent.name:
        return artifact_dir.parent.name
    return artifact_dir.name


def max_context(row: ArtifactProfile) -> int:
    values = [int(item) for item in row.context_sizes.split() if item.isdigit()]
    return max(values) if values else -1


def max_cache(row: ArtifactProfile) -> int:
    values = [int(item) for item in row.cache_sizes.split() if item.isdigit()]
    return max(values) if values else -1


def compare_context_variants(rows: list[ArtifactProfile], date: str | None = None) -> list[dict[str, str]]:
    by_group: dict[str, list[ArtifactProfile]] = {}
    for row in rows:
        if row.support_status != "ok":
            continue
        by_group.setdefault(row.comparison_group, []).append(row)

    comparisons: list[dict[str, str]] = []
    for group, group_rows in sorted(by_group.items()):
        unique_contexts = {row.context_sizes for row in group_rows}
        if len(unique_contexts) < 2:
            continue
        ordered = sorted(group_rows, key=lambda row: (max_context(row), max_cache(row), row.artifact_id))
        base = ordered[0]
        target = ordered[-1]
        delta = target.total_referenced_bytes - base.total_referenced_bytes
        growth = ""
        if base.total_referenced_bytes > 0:
            growth = f"{100.0 * delta / base.total_referenced_bytes:.2f}"
        comparisons.append(
            {
                "date": date or utc_timestamp(),
                "comparison_group": group,
                "base_artifact_id": base.artifact_id,
                "target_artifact_id": target.artifact_id,
                "base_cache_sizes": base.cache_sizes,
                "target_cache_sizes": target.cache_sizes,
                "base_context_sizes": base.context_sizes,
                "target_context_sizes": target.context_sizes,
                "base_total_referenced_bytes": str(base.total_referenced_bytes),
                "target_total_referenced_bytes": str(target.total_referenced_bytes),
                "delta_bytes": str(delta),
                "growth_pct": growth,
                "base_config_path": base.config_path,
                "target_config_path": target.config_path,
            }
        )
    return comparisons


def write_csv(path: Path, header: list[str], rows: Iterable[dict[str, object]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in header})


def mib(num_bytes: int | str) -> str:
    try:
        value = int(num_bytes)
    except Exception:
        return ""
    return f"{value / 1024.0 / 1024.0:.2f}"


def write_markdown(path: Path, rows: list[ArtifactProfile], comparisons: list[dict[str, str]], command: str, commit: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# QNN AoT Artifact Size Profile\n\n")
        f.write("## Goal\n\n")
        f.write(
            "Check whether generated QNN AoT artifact file sizes differ across cache/context sizes. "
            "This is a static file-size check only; it does not measure runtime PSS/RSS, HTP memory, or spill-fill allocation.\n\n"
        )
        f.write("## Exact Command\n\n")
        f.write("```bash\n")
        f.write(command)
        f.write("\n```\n\n")
        f.write(f"- Git commit: `{commit}`\n\n")
        f.write("## Artifact Table\n\n")
        f.write(
            "| artifact_id | cache_sizes | context_sizes | batch_sizes | referenced MiB | artifact-dir MiB | support_status |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            f.write(
                f"| {row.artifact_id} | {row.cache_sizes} | {row.context_sizes} | {row.batch_sizes} | "
                f"{mib(row.total_referenced_bytes)} | {mib(row.total_artifact_bytes)} | {row.support_status} |\n"
            )
        f.write("\n")

        f.write("## Cross-Context Comparison\n\n")
        if comparisons:
            f.write(
                "| group | base | target | base context | target context | base MiB | target MiB | delta MiB | growth pct |\n"
            )
            f.write("|---|---|---|---:|---:|---:|---:|---:|---:|\n")
            for comp in comparisons:
                f.write(
                    f"| {comp['comparison_group']} | {comp['base_artifact_id']} | {comp['target_artifact_id']} | "
                    f"{comp['base_context_sizes']} | {comp['target_context_sizes']} | "
                    f"{mib(comp['base_total_referenced_bytes'])} | {mib(comp['target_total_referenced_bytes'])} | "
                    f"{mib(comp['delta_bytes'])} | {comp['growth_pct']} |\n"
                )
            f.write("\n")
        else:
            f.write(
                "No cross-context comparison is available because the scanned artifacts do not contain two valid configs "
                "with different `context_size` values in the same comparison group.\n\n"
            )

        f.write("## Interpretation\n\n")
        if comparisons:
            high_growth = [
                comp for comp in comparisons
                if comp.get("growth_pct") and float(comp["growth_pct"]) >= 30.0
            ]
            if high_growth:
                f.write(
                    "At least one same-group context variant shows `>=30%` referenced-artifact file-size growth. "
                    "This supports treating long-context QNN AoT graph selection as a capacity/cost tradeoff, pending runtime-memory validation.\n"
                )
            else:
                f.write(
                    "The scanned same-group context variants do not show `>=30%` referenced-artifact file-size growth. "
                    "This weakens a file-size-only claim for dynamic AoT graph switching, though runtime memory may still differ.\n"
                )
        else:
            f.write(
                "The scan is inconclusive for the 2k-vs-4k question. It can confirm the active generated artifacts' sizes, "
                "but it cannot prove or reject context-size growth without a real larger-context AoT artifact.\n"
            )
        f.write("\n")

        unsupported = [row for row in rows if row.support_status != "ok"]
        f.write("\n## Anomalies\n\n")
        if unsupported:
            for row in unsupported:
                f.write(f"- `{row.artifact_id}`: {row.support_status}; missing `{row.missing_model_files}`\n")
        else:
            f.write("- None in the static file-size scan.\n")


def copy_global_outputs(output_dir: Path, csv_path: Path, comparison_path: Path):
    global_dir = ROOT_DIR / "results" / "insightB"
    global_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(csv_path, global_dir / "qnn_aot_artifact_size.csv")
    shutil.copy2(comparison_path, global_dir / "qnn_aot_artifact_size_comparison.csv")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile generated QNN AoT artifact file sizes.")
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        help="QNN artifact directory or config JSON. May be passed multiple times.",
    )
    parser.add_argument(
        "--scan-root",
        action="append",
        default=[],
        help="Directory to recursively scan for config.json files. May be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to results/insightB/qnn-aot-size-<date>.",
    )
    parser.add_argument("--date", default=None, help="Run date label for output filenames.")
    parser.add_argument("--no-global-copy", action="store_true", help="Do not update results/insightB summary CSV copies.")
    return parser.parse_args(argv)


def discover_configs(args: argparse.Namespace) -> list[Path]:
    configs: list[Path] = []
    for artifact in args.artifact:
        configs.append(resolve_config_path(Path(artifact)))
    for root_arg in args.scan_root:
        root = Path(root_arg)
        configs.extend(sorted(root.rglob("config.json")))
    unique = []
    seen = set()
    for config in configs:
        resolved = config.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    date_label = args.date or run_date()
    output_dir = args.output_dir or (ROOT_DIR / "results" / "insightB" / f"qnn-aot-size-{date_label}")
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = discover_configs(args)
    if not configs:
        print("No QNN artifact configs found. Pass --artifact or --scan-root.", file=sys.stderr)
        return 2

    command = " ".join([shlex_quote(sys.executable), shlex_quote(__file__), *map(shlex_quote, sys.argv[1:])])
    (output_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    commit = git_commit()
    (output_dir / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")

    date = utc_timestamp()
    rows = [
        profile_artifact(config, artifact_id=default_artifact_id(config), date=date)
        for config in configs
    ]
    comparisons = compare_context_variants(rows, date=date)

    csv_path = output_dir / "qnn_aot_artifact_size.csv"
    comparison_path = output_dir / "qnn_aot_artifact_size_comparison.csv"
    md_path = output_dir / f"QNN_AoT_Artifact_Size_{date_label}.md"

    write_csv(csv_path, PROFILE_HEADER, [asdict(row) for row in rows])
    write_csv(comparison_path, COMPARISON_HEADER, comparisons)
    write_markdown(md_path, rows, comparisons, command, commit)

    if not args.no_global_copy:
        copy_global_outputs(output_dir, csv_path, comparison_path)
        docs_dir = ROOT_DIR / "docs" / "实验结果"
        docs_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(md_path, docs_dir / md_path.name)

    print(f"profile_csv={csv_path}")
    print(f"comparison_csv={comparison_path}")
    print(f"summary_md={md_path}")
    if comparisons:
        print("comparison_status=available")
    else:
        print("comparison_status=inconclusive_no_context_variants")
    return 0


def shlex_quote(value: object) -> str:
    text = str(value)
    if not text:
        return "''"
    safe = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_@%+=:,./-"
    if all(char in safe for char in text):
        return text
    return "'" + text.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    raise SystemExit(main())
