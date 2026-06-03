from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .profile_schema import CompilerConfig, StateProfile, add_quality, normalize_quality, parse_float, parse_int


@dataclass
class MarkdownTable:
    headers: List[str]
    rows: List[Dict[str, str]]
    context: str


def parse_markdown_tables(path: Path | str) -> List[MarkdownTable]:
    md_path = Path(path)
    lines = md_path.read_text(encoding="utf-8", errors="replace").splitlines()
    tables: List[MarkdownTable] = []
    index = 0
    while index < len(lines) - 1:
        line = lines[index].strip()
        next_line = lines[index + 1].strip()
        if line.startswith("|") and _is_separator(next_line):
            table_lines = [line]
            index += 2
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            headers = _split_row(table_lines[0])
            rows = []
            for raw in table_lines[1:]:
                cells = _split_row(raw)
                if len(cells) < len(headers):
                    cells.extend([""] * (len(headers) - len(cells)))
                rows.append(dict(zip(headers, cells)))
            context = "\n".join(lines[max(0, index - len(table_lines) - 40) : index])
            tables.append(MarkdownTable(headers=headers, rows=rows, context=context))
            continue
        index += 1
    return tables


def load_markdown_profiles(path: Path | str, config: CompilerConfig) -> List[StateProfile]:
    md_path = Path(path)
    name = md_path.name
    tables = parse_markdown_tables(md_path)
    states: List[StateProfile] = []
    for table in tables:
        if name == "CPU测试结果.md":
            states.extend(_parse_cpu_table(md_path, table, config))
        elif name == "GPU测试结果.md":
            states.extend(_parse_gpu_table(md_path, table, config))
        elif name == "NPU测试结果.md":
            states.extend(_parse_npu_table(md_path, table, config))
    return states


def _parse_cpu_table(path: Path, table: MarkdownTable, config: CompilerConfig) -> List[StateProfile]:
    if not _has_header(table, "请求频率") or not _has_header(table, "吞吐"):
        return []
    shape = _shape_from_context(table.context) or "tg"
    phase = "prefill" if shape.lower().startswith("pp") else "decode"
    tokens = _shape_tokens(shape)
    affinity = _cpu_affinity_from_context(table.context)
    threads = _threads_from_context(table.context)
    states: List[StateProfile] = []
    for row in table.rows:
        freq = parse_int(_cell(row, "请求频率"))
        throughput = parse_float(_cell(row, "吞吐"))
        if freq is None or throughput is None:
            continue
        remarks = _cell(row, "备注") or ""
        actual_freq = parse_int(_cell(row, "平均 CPU 频率")) or freq
        row_affinity = _clean_case(_cell(row, "Case")) or affinity
        quality = normalize_quality(None)
        if "窗口波动高" in remarks or "功率/吞吐异常" in remarks:
            add_quality(quality, "power_low_confidence")
        state = StateProfile(
            state_id=f"cpu_{row_affinity}_{freq // 1000}",
            backend="CPU",
            phase=phase,
            source_file=str(path),
            test_shape=shape,
            prompt_tokens=tokens if phase == "prefill" else 0,
            decode_tokens=tokens if phase == "decode" else 0,
            rounds=_rounds_from_context(table.context),
            context_len=0 if phase == "decode" else None,
            cpu_affinity=row_affinity,
            cpu_freq_khz=freq,
            actual_cpu_freq_khz=actual_freq,
            cpu_threads=threads,
            throughput_tps=throughput,
            active_power_mw=parse_float(_cell(row, "稳态平均功率") or _cell(row, "Active平均功率") or _cell(row, "功率")),
            power_delta_mw=parse_float(_cell(row, "增量")),
            temperature_avg_c=parse_float(_cell(row, "平均温度")),
            temperature_max_c=parse_float(_cell(row, "最高温度")),
            stable_range_pct=parse_float(_cell(row, "稳定窗口范围") or _cell(row, "窗口波动")),
            data_quality=quality,
            metadata={"remarks": remarks} if remarks else {},
        )
        states.append(state.normalized(config))
    return states


def _parse_gpu_table(path: Path, table: MarkdownTable, config: CompilerConfig) -> List[StateProfile]:
    if not (_has_header(table, "设定频率") or _has_header(table, "GPU 频率")):
        return []
    states: List[StateProfile] = []
    for row in table.rows:
        freq = parse_int(_cell(row, "设定频率") or _cell(row, "GPU 频率"))
        if freq is None:
            continue
        states.extend(_gpu_state_for_prefix(path, table, row, config, freq, "TG", "decode"))
        states.extend(_gpu_state_for_prefix(path, table, row, config, freq, "PP", "prefill"))
        if not states or not any(s.source_file == str(path) and s.gpu_freq_mhz == freq for s in states):
            phase = _phase_from_context(table.context)
            if phase:
                throughput = parse_float(_cell(row, "平均吞吐", "吞吐"))
                if throughput is not None:
                    shape = _shape_from_context(table.context) or ("pp" if phase == "prefill" else "tg")
                    states.append(
                        _make_gpu_state(
                            path,
                            row,
                            config,
                            freq,
                            phase,
                            shape,
                            throughput,
                            parse_float(_cell(row, "Active plateau 平均功率", "Active平均功率", "功率")),
                            parse_int(_cell(row, "实际频率")) or freq,
                            parse_float(_cell(row, "平均窗口波动", "窗口波动")),
                        )
                    )
    return states


def _parse_npu_table(path: Path, table: MarkdownTable, config: CompilerConfig) -> List[StateProfile]:
    if not _has_header(table, "Workpoint"):
        return []
    states: List[StateProfile] = []
    for row in table.rows:
        workpoint = _cell(row, "Workpoint")
        if not workpoint:
            continue
        emitted_prefixed = False
        for prefix, phase in (("TG", "decode"), ("PP", "prefill")):
            throughput = parse_float(_cell(row, f"{prefix}", "吞吐"))
            power = parse_float(_cell(row, f"{prefix}", "功率"))
            if throughput is None:
                continue
            emitted_prefixed = True
            shape = _shape_for_prefix(table, prefix, default=f"{prefix.lower()}{_shape_tokens(_shape_from_context(table.context)) or ''}")
            tokens = _shape_tokens(shape)
            state = StateProfile(
                state_id=f"npu_{workpoint}",
                backend="QNN_NPU",
                phase=phase,
                source_file=str(path),
                test_shape=shape,
                prompt_tokens=tokens if phase == "prefill" else 0,
                decode_tokens=tokens if phase == "decode" else 0,
                rounds=_rounds_from_context(table.context),
                context_len=0 if phase == "decode" else None,
                npu_workpoint=workpoint,
                throughput_tps=throughput,
                active_power_mw=power,
                power_delta_mw=parse_float(_cell(row, f"{prefix}", "增量")),
                stable_range_pct=parse_float(_cell(row, f"{prefix}", "波动", "稳定窗口范围")),
            )
            states.append(state.normalized(config))
        if emitted_prefixed:
            continue
        phase = _phase_from_context(table.context)
        throughput = parse_float(_cell(row, "吞吐"))
        if phase is None or throughput is None:
            continue
        shape = _shape_from_context(table.context) or ("pp" if phase == "prefill" else "tg")
        tokens = _shape_tokens(shape)
        state = StateProfile(
            state_id=f"npu_{workpoint}",
            backend="QNN_NPU",
            phase=phase,
            source_file=str(path),
            test_shape=shape,
            prompt_tokens=tokens if phase == "prefill" else 0,
            decode_tokens=tokens if phase == "decode" else 0,
            rounds=_rounds_from_context(table.context),
            context_len=0 if phase == "decode" else None,
            npu_workpoint=workpoint,
            throughput_tps=throughput,
            active_power_mw=parse_float(_cell(row, "稳态平均功率") or _cell(row, "功率")),
            power_delta_mw=parse_float(_cell(row, "增量")),
            temperature_avg_c=parse_float(_cell(row, "平均温度")),
            temperature_max_c=parse_float(_cell(row, "最高温度")),
            stable_range_pct=parse_float(_cell(row, "稳定窗口范围") or _cell(row, "波动")),
        )
        states.append(state.normalized(config))
    return states


def _gpu_state_for_prefix(
    path: Path,
    table: MarkdownTable,
    row: Dict[str, str],
    config: CompilerConfig,
    freq: int,
    prefix: str,
    phase: str,
) -> List[StateProfile]:
    matching_headers = [header for header in table.headers if prefix.lower() in header.lower()]
    if not matching_headers:
        return []
    throughput = parse_float(_cell(row, prefix, "吞吐"))
    if throughput is None:
        return []
    shape = _shape_for_prefix(table, prefix, default=f"{prefix.lower()}{_shape_tokens(_shape_from_context(table.context)) or ''}")
    return [
        _make_gpu_state(
            path,
            row,
            config,
            freq,
            phase,
            shape,
            throughput,
            parse_float(_cell(row, prefix, "功率")),
            parse_int(_cell(row, prefix, "实际频率")) or freq,
            parse_float(_cell(row, prefix, "波动")),
        )
    ]


def _make_gpu_state(
    path: Path,
    row: Dict[str, str],
    config: CompilerConfig,
    freq: int,
    phase: str,
    shape: str,
    throughput: float,
    power: Optional[float],
    actual_freq: int,
    stable_range: Optional[float],
) -> StateProfile:
    remarks = _cell(row, "备注") or ""
    quality = normalize_quality(None)
    if "波动高" in remarks or "功率窗口波动偏高" in remarks:
        add_quality(quality, "unstable_power_window")
    tokens = _shape_tokens(shape)
    state = StateProfile(
        state_id=f"gpu_{freq}",
        backend="GPU",
        phase=phase,
        source_file=str(path),
        test_shape=shape,
        prompt_tokens=tokens if phase == "prefill" else 0,
        decode_tokens=tokens if phase == "decode" else 0,
        rounds=_rounds_from_context(shape),
        context_len=0 if phase == "decode" else None,
        gpu_freq_mhz=freq,
        actual_gpu_freq_mhz=actual_freq,
        throughput_tps=throughput,
        active_power_mw=power,
        stable_range_pct=stable_range,
        data_quality=quality,
        metadata={"remarks": remarks} if remarks else {},
    )
    return state.normalized(config)


def _cell(row: Dict[str, str], *needles: str) -> Optional[str]:
    for header, value in row.items():
        if all(needle.lower() in header.lower() for needle in needles):
            return value.strip()
    return None


def _has_header(table: MarkdownTable, needle: str) -> bool:
    return any(needle.lower() in header.lower() for header in table.headers)


def _split_row(line: str) -> List[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _is_separator(line: str) -> bool:
    cells = _split_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{2,}:?", cell.strip()) for cell in cells)


def _shape_from_context(context: str) -> Optional[str]:
    matches = re.findall(r"\b(?:`)?((?:tg|pp)\s*[0-9]+)(?:`)?", context, flags=re.IGNORECASE)
    if matches:
        return matches[-1].replace(" ", "").lower()
    return None


def _shape_for_prefix(table: MarkdownTable, prefix: str, default: str) -> str:
    for header in table.headers:
        match = re.search(rf"({prefix}\s*[0-9]+)", header, flags=re.IGNORECASE)
        if match:
            return match.group(1).replace(" ", "").lower()
    shape = _shape_from_context(table.context)
    if shape and shape.lower().startswith(prefix.lower()):
        return shape
    return default


def _shape_tokens(shape: Optional[str]) -> Optional[int]:
    if not shape:
        return None
    match = re.search(r"([0-9]+)", shape)
    return int(match.group(1)) if match else None


def _rounds_from_context(context: str) -> Optional[int]:
    match = re.search(r"\br\s*=?\s*([0-9]+)", context, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"Round\s*([0-9]+)", context, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _cpu_affinity_from_context(context: str) -> str:
    match = re.search(r"CPU case[：:]\s*`?([A-Za-z0-9_-]+)", context, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"(big2small1|big2|big1|all-core)", context, flags=re.IGNORECASE)
    return match.group(1) if match else "cpu"


def _threads_from_context(context: str) -> Optional[int]:
    match = re.search(r"threads\s*=\s*([0-9]+)", context, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _clean_case(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    text = re.sub(r"\s*\(.*?\)", "", value).strip()
    return text or None


def _phase_from_context(context: str) -> Optional[str]:
    lower = context.lower()
    if "prefill" in lower or "pp" in lower:
        return "prefill"
    if "decode" in lower or "tg" in lower:
        return "decode"
    return None
