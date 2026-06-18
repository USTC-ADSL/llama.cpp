#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from profiles.offline_profile_lib import (  # noqa: E402
    RAW_FIELDS,
    STATE_CATALOG_FIELDS,
    append_csv_row,
    command_variables,
    ensure_output_dirs,
    environment_snapshot,
    git_commit,
    load_config,
    logs_dir,
    manifests_dir,
    measurement_matrix,
    merge_state_catalog,
    now_utc,
    parse_measurement_json,
    profiles_dir,
    read_csv_rows,
    render_command,
    slugify,
    state_catalog_rows,
    write_csv_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run configurable offline profiling sweep.")
    parser.add_argument("--backend", choices=["NPU", "CPU", "GPU"], required=True)
    parser.add_argument("--phase", choices=["decode", "prefill", "all"], default="all")
    parser.add_argument("--config", default="configs/offline_profile.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--sanity", action="store_true", help="Run optional QNN large-graph sanity profile.")
    return parser.parse_args()


def _existing_ok_keys(raw_path: Path) -> set[tuple[str, str, str, str]]:
    rows = read_csv_rows(raw_path, RAW_FIELDS, missing_ok=True)
    return {
        (row.get("phase", ""), row.get("state_name", ""), row.get("length", ""), row.get("run_id", ""))
        for row in rows
        if row.get("status") == "ok"
    }


def _write_manifest(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    dry_run = bool(config.get("dry_run", False)) or args.dry_run
    resume = bool(config.get("resume", True))
    if args.resume:
        resume = True
    if args.no_resume:
        resume = False
    repeat = int(config.get("repeat", 1))
    fail_fast = bool(config.get("fail_fast", False))

    if not dry_run:
        ensure_output_dirs(config)
    raw_name = "qnn_large_graph_sanity.csv" if args.sanity else "perf_profile_raw.csv"
    raw_path = profiles_dir(config) / raw_name
    catalog_path = profiles_dir(config) / "state_catalog.csv"

    existing_catalog = read_csv_rows(catalog_path, STATE_CATALOG_FIELDS, missing_ok=True)
    merged_catalog = merge_state_catalog(existing_catalog, state_catalog_rows(config, args.backend))
    if not dry_run:
        write_csv_rows(catalog_path, STATE_CATALOG_FIELDS, merged_catalog)

    planned = measurement_matrix(config, args.backend, args.phase, args.sanity)
    existing_ok = _existing_ok_keys(raw_path) if resume else set()
    template_by_phase = {
        "decode": str(config.get("decode_command_template") or ""),
        "prefill": str(config.get("prefill_command_template") or ""),
    }
    prefix = str(config.get("device_command_prefix") or "")
    exit_code = 0

    for item in planned:
        phase = item["phase"]
        state = item["state"]
        length = int(item["length"])
        for run_id in range(repeat):
            key = (phase, state["state_name"], str(length), str(run_id))
            slug = slugify(f"{phase}_{state['state_name']}_{length}_r{run_id}")
            log_path = logs_dir(config) / f"{slug}.log"
            output_json = logs_dir(config) / f"{slug}.metrics.json"
            manifest_path = manifests_dir(config) / f"{slug}.manifest.json"
            variables = command_variables(config, phase, state, length, run_id, output_json, log_path)
            template = template_by_phase[phase]
            command = render_command(template, variables, prefix)
            manifest = {
                "created_at_utc": now_utc(),
                "phase": phase,
                "state_name": state["state_name"],
                "length": length,
                "run_id": run_id,
                "command": command,
                "command_variables": variables,
                "state": state,
                "config_path": str(Path(args.config).resolve()),
                "config": config,
                "git_commit": git_commit(),
                "environment": environment_snapshot(),
                "log_path": str(log_path),
                "output_json": str(output_json),
            }

            if resume and key in existing_ok:
                print(f"[resume] skip ok run {phase} {state['state_name']} length={length} run_id={run_id}")
                continue

            if not command:
                message = (
                    f"[todo] no {phase}_command_template configured for "
                    f"{state['state_name']} length={length} run_id={run_id}"
                )
                print(message)
                if dry_run:
                    print(f"[dry-run] log_path={log_path}")
                    print(f"[dry-run] manifest_path={manifest_path}")
                if not dry_run:
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    log_path.write_text(message + "\n", encoding="utf-8")
                    manifest["completed_at_utc"] = now_utc()
                    manifest["exit_code"] = None
                    manifest["measurement"] = {"status": "skipped", "reason": "missing_command_template"}
                    _write_manifest(manifest_path, manifest)
                    append_csv_row(
                        raw_path,
                        RAW_FIELDS,
                        {
                            "phase": phase,
                            "state_name": state["state_name"],
                            "length": length,
                            "run_id": run_id,
                            "n_tokens": config.get("decode_probe_tokens") if phase == "decode" else length,
                            "elapsed_ms": "",
                            "energy_mj": "",
                            "status": "skipped",
                        },
                    )
                continue

            if dry_run:
                print(f"[dry-run] {command}")
                print(f"[dry-run] log_path={log_path}")
                print(f"[dry-run] manifest_path={manifest_path}")
                continue

            _write_manifest(manifest_path, manifest)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as log_file:
                log_file.write(f"$ {command}\n")
                result = subprocess.run(command, shell=True, text=True, stdout=log_file, stderr=subprocess.STDOUT, check=False)

            measurement = parse_measurement_json(output_json, phase, length, config)
            status = str(measurement.get("status") or "failed")
            if result.returncode != 0 and status == "ok":
                status = "failed"
            manifest["completed_at_utc"] = now_utc()
            manifest["exit_code"] = result.returncode
            manifest["measurement"] = measurement
            _write_manifest(manifest_path, manifest)

            append_csv_row(
                raw_path,
                RAW_FIELDS,
                {
                    "phase": phase,
                    "state_name": state["state_name"],
                    "length": length,
                    "run_id": run_id,
                    "n_tokens": measurement.get("n_tokens", config.get("decode_probe_tokens") if phase == "decode" else length),
                    "elapsed_ms": measurement.get("elapsed_ms", ""),
                    "energy_mj": measurement.get("energy_mj", ""),
                    "status": status,
                },
            )

            if status not in {"ok", "skipped"}:
                exit_code = 1
                print(f"[failed] {phase} {state['state_name']} length={length} run_id={run_id} log={log_path}")
                if fail_fast:
                    return exit_code
            else:
                print(f"[ok] {phase} {state['state_name']} length={length} run_id={run_id} log={log_path}")

    if dry_run:
        print("[dry-run] no commands executed and no profiling CSV rows written")
    else:
        print(f"raw_csv={raw_path}")
        print(f"state_catalog={catalog_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
