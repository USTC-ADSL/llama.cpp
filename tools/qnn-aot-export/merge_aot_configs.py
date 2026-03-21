from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List


SECTION_KEYS = [
    "graphs",
    "transformer_graphs",
    "attention_graphs",
    "attn_proj_graphs",
    "attn_core_graphs",
    "ffn_graphs",
    "embeddings",
    "lm_head_graphs",
]


def parse_entry(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("expected PREFIX=CONFIG_PATH")
    prefix, path = text.split("=", 1)
    return prefix.strip("/"), Path(path)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: Path, obj: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def prefix_relpath(prefix: str, relpath: str) -> str:
    if not prefix:
        return relpath
    return str(Path(prefix) / relpath).replace("\\", "/")


def rewrite_graph_paths(graphs: List[Dict[str, Any]], prefix: str):
    for graph in graphs:
        if "model_path" in graph and isinstance(graph["model_path"], str):
            graph["model_path"] = prefix_relpath(prefix, graph["model_path"])
        if "kv_path_format" in graph and isinstance(graph["kv_path_format"], str):
            graph["kv_path_format"] = prefix_relpath(prefix, graph["kv_path_format"])


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple AoT configs into one runtime-selectable config by prefixing model paths."
    )
    parser.add_argument(
        "--entry",
        type=parse_entry,
        nargs="+",
        required=True,
        help="PREFIX=CONFIG_PATH. PREFIX is prepended to model_path / kv_path_format from that config.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    merged: Dict[str, Any] = {}

    for index, (prefix, config_path) in enumerate(args.entry):
        config = load_json(config_path)

        if index == 0:
            merged["model_parameters"] = deepcopy(config.get("model_parameters", {}))
            if "qnn_parameters" in config:
                merged["qnn_parameters"] = deepcopy(config["qnn_parameters"])
        else:
            if config.get("model_parameters", {}) != merged.get("model_parameters", {}):
                raise RuntimeError(f"model_parameters mismatch in {config_path}")
            if "qnn_parameters" in config:
                if "qnn_parameters" not in merged:
                    merged["qnn_parameters"] = deepcopy(config["qnn_parameters"])
                elif config["qnn_parameters"] != merged["qnn_parameters"]:
                    raise RuntimeError(f"qnn_parameters mismatch in {config_path}")

        for key in SECTION_KEYS:
            if key not in config:
                continue
            graphs = deepcopy(config[key])
            rewrite_graph_paths(graphs, prefix)
            merged.setdefault(key, [])
            merged[key].extend(graphs)

    write_json(args.output, merged)


if __name__ == "__main__":
    main()
