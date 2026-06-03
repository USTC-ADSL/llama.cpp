from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional


BACKEND_ALIASES = {
    "cpu": "CPU",
    "gpu": "GPU",
    "gpuopencl": "GPU",
    "opencl": "GPU",
    "qnn": "QNN_NPU",
    "qnn-npu": "QNN_NPU",
    "qnn_npu": "QNN_NPU",
    "npu": "QNN_NPU",
}


PHASE_ALIASES = {
    "prefill": "prefill",
    "pp": "prefill",
    "prompt": "prefill",
    "decode": "decode",
    "tg": "decode",
    "token_generation": "decode",
}


def parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "n/a", "na", "-", "unavailable"}:
        return None
    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def parse_int(value: Any) -> Optional[int]:
    parsed = parse_float(value)
    if parsed is None:
        return None
    return int(round(parsed))


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "ok"}:
        return True
    if text in {"0", "false", "no", "n", ""}:
        return False
    return default


def normalize_backend(value: Any) -> str:
    text = str(value or "").strip()
    key = text.replace(" ", "").replace("_", "-").lower()
    return BACKEND_ALIASES.get(key, BACKEND_ALIASES.get(text.lower(), text.upper()))


def normalize_phase(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("pp"):
        return "prefill"
    if text.startswith("tg"):
        return "decode"
    return PHASE_ALIASES.get(text, text)


def infer_phase_from_shape(shape: Optional[str]) -> Optional[str]:
    if not shape:
        return None
    lower = shape.lower()
    if "pp" in lower:
        return "prefill"
    if "tg" in lower:
        return "decode"
    return None


def normalize_quality(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw = value
    else:
        raw = str(value).replace(";", ",").replace("|", ",").split(",")
    result: List[str] = []
    for item in raw:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return result


def add_quality(quality: List[str], marker: str) -> None:
    if marker not in quality:
        quality.append(marker)


def _without_none(data: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in data.items():
        if value is None:
            continue
        if value == {}:
            continue
        result[key] = value
    return result


@dataclass
class CompilerConfig:
    slo_ttft_ms_values: List[float] = field(default_factory=lambda: [500.0, 1000.0, 2000.0, 4000.0])
    slo_tbt_us_values: List[float] = field(default_factory=lambda: [45000.0, 55000.0, 70000.0])
    stable_range_pct_limit: float = 10.0
    power_cv_pct_limit: float = 10.0
    frequency_mismatch_pct_limit: float = 5.0
    allow_estimated_energy: bool = True
    allow_length_interpolation: bool = True
    allow_extrapolation: bool = False
    allow_p50_as_proxy: bool = False
    filter_unstable: bool = True
    filter_fallback_used: bool = True
    filter_unsupported: bool = True
    filter_thermal_unsafe: bool = True
    thermal_max_c_limit: Optional[float] = 38.0
    enable_power_comparison_when_energy_unavailable: bool = True

    @classmethod
    def from_mapping(cls, data: Dict[str, Any]) -> "CompilerConfig":
        config = cls()
        for key, value in data.items():
            if not hasattr(config, key):
                continue
            current = getattr(config, key)
            if isinstance(current, bool):
                setattr(config, key, parse_bool(value, current))
            elif isinstance(current, float) or current is None:
                parsed = parse_float(value)
                setattr(config, key, parsed if parsed is not None else current)
            elif isinstance(current, list):
                setattr(config, key, [float(v) for v in value])
            else:
                setattr(config, key, value)
        return config

    def to_artifact_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StateProfile:
    state_id: str
    backend: str
    phase: str
    source_file: str
    test_shape: Optional[str] = None
    prompt_tokens: Optional[int] = None
    decode_tokens: Optional[int] = None
    rounds: Optional[int] = None
    context_len: Optional[int] = None
    cpu_affinity: Optional[str] = None
    cpu_freq_khz: Optional[int] = None
    actual_cpu_freq_khz: Optional[int] = None
    cpu_threads: Optional[int] = None
    gpu_freq_mhz: Optional[int] = None
    actual_gpu_freq_mhz: Optional[int] = None
    npu_workpoint: Optional[str] = None
    graph_id: Optional[str] = None
    throughput_tps: Optional[float] = None
    tbt_us: Optional[float] = None
    tbt_source: Optional[str] = None
    ttft_ms_p50: Optional[float] = None
    ttft_ms_p95: Optional[float] = None
    ttft_source: Optional[str] = None
    active_power_mw: Optional[float] = None
    baseline_power_mw: Optional[float] = None
    power_delta_mw: Optional[float] = None
    energy_mj_per_token: Optional[float] = None
    energy_mj_per_request: Optional[float] = None
    temperature_avg_c: Optional[float] = None
    temperature_max_c: Optional[float] = None
    stable_range_pct: Optional[float] = None
    power_cv_pct: Optional[float] = None
    support_status: str = "ok"
    fallback_used: bool = False
    stable: bool = True
    thermal_safe: bool = True
    data_quality: List[str] = field(default_factory=list)
    energy_source: Optional[str] = None
    energy_complete: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalized(self, config: CompilerConfig) -> "StateProfile":
        self.backend = normalize_backend(self.backend)
        self.phase = normalize_phase(self.phase)
        self.support_status = str(self.support_status or "ok").strip().lower()
        self.data_quality = normalize_quality(self.data_quality)

        if self.tbt_us is None and self.throughput_tps and self.throughput_tps > 0 and self.phase == "decode":
            self.tbt_us = 1_000_000.0 / self.throughput_tps
            self.tbt_source = "derived_from_throughput"

        if self.ttft_ms_p95 is None and self.phase == "prefill" and self.throughput_tps and self.throughput_tps > 0:
            tokens = self.prompt_tokens or _tokens_from_shape(self.test_shape) or 1
            self.ttft_ms_p95 = tokens * 1000.0 / self.throughput_tps
            self.ttft_source = "prefill_latency_from_throughput"
            add_quality(self.data_quality, "prefill_latency_proxy")

        if self.energy_mj_per_token is not None:
            source = self.energy_source or "measured_or_profiled"
            self.energy_source = source
            self.energy_complete = source not in {
                "estimated_power_latency",
                "unavailable",
                "missing",
                "insightb_power_latency_no_energy_claim",
            }
        elif self.active_power_mw is not None and self.tbt_us is not None:
            self.energy_mj_per_token = self.active_power_mw * self.tbt_us / 1_000_000.0
            self.energy_source = "estimated_power_latency"
            self.energy_complete = False
        else:
            self.energy_complete = False

        if self.support_status != "ok":
            add_quality(self.data_quality, "unsupported")

        if self.fallback_used:
            add_quality(self.data_quality, "fallback_used")

        remarks = str(self.metadata.get("remarks", ""))
        if "掉频" in remarks or "throttle" in remarks.lower():
            self.stable = False
            add_quality(self.data_quality, "frequency_mismatch")

        if "unstable_power_window" in self.data_quality:
            self.stable = False

        if self.backend == "CPU":
            self._mark_frequency_mismatch(self.cpu_freq_khz, self.actual_cpu_freq_khz, config)
        elif self.backend == "GPU":
            self._mark_frequency_mismatch(self.gpu_freq_mhz, self.actual_gpu_freq_mhz, config)

        if self.stable_range_pct is not None and self.stable_range_pct > config.stable_range_pct_limit:
            add_quality(self.data_quality, "power_low_confidence")
        if self.power_cv_pct is not None and self.power_cv_pct > config.power_cv_pct_limit:
            add_quality(self.data_quality, "power_low_confidence")

        if config.thermal_max_c_limit is not None and self.temperature_max_c is not None:
            self.thermal_safe = self.temperature_max_c <= config.thermal_max_c_limit
            if not self.thermal_safe:
                add_quality(self.data_quality, "thermal_unsafe")

        return self

    def _mark_frequency_mismatch(
        self,
        requested: Optional[int],
        actual: Optional[int],
        config: CompilerConfig,
    ) -> None:
        if requested is None or actual is None or requested <= 0:
            return
        mismatch_pct = abs(float(requested - actual)) * 100.0 / float(requested)
        if mismatch_pct > config.frequency_mismatch_pct_limit:
            self.stable = False
            add_quality(self.data_quality, "frequency_mismatch")
            self.metadata["frequency_mismatch_pct"] = round(mismatch_pct, 3)

    def latency_value(self) -> Optional[float]:
        if self.phase == "decode":
            return self.tbt_us
        if self.phase == "prefill":
            return self.ttft_ms_p95 or self.ttft_ms_p50
        return None

    def length_value(self) -> Optional[int]:
        if self.phase == "decode":
            return self.context_len
        if self.phase == "prefill":
            return self.prompt_tokens
        return None

    def to_artifact_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["data_quality"] = list(self.data_quality)
        return _without_none(data)


@dataclass
class TransitionProfile:
    from_state_id: str
    to_state_id: str
    source_file: str
    context_len: Optional[int] = None
    total_blocking_us: Optional[float] = None
    first_token_gap_us: Optional[float] = None
    post_switch_tbt_us: Optional[float] = None
    transition_energy_mj: Optional[float] = None
    transition_energy_source: str = "unavailable"
    transition_energy_complete: bool = False
    success_rate: Optional[float] = None
    fallback_count: Optional[int] = None
    support_status: str = "ok"
    kv_handoff_us: Optional[float] = None
    graph_rebuild_us: Optional[float] = None
    decision_us: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "TransitionProfile":
        self.support_status = str(self.support_status or "ok").strip().lower()
        self.transition_energy_source = str(self.transition_energy_source or "unavailable").strip().lower()
        self.transition_energy_complete = (
            self.transition_energy_mj is not None and self.transition_energy_source not in {"", "unavailable", "missing"}
        )
        return self

    def to_artifact_dict(self) -> Dict[str, Any]:
        return _without_none(asdict(self))


@dataclass
class GraphProfile:
    graph_id: str
    phase: str
    source_file: str
    chunk_size: Optional[int] = None
    usable_kv_slots: Optional[int] = None
    safety_margin: Optional[int] = None
    supported_workpoints: List[str] = field(default_factory=list)
    profiled_load_us: Optional[float] = None
    profiled_warmup_us: Optional[float] = None
    profiled_exec_us: Optional[float] = None
    profiled_energy_mj: Optional[float] = None
    memory_bytes: Optional[int] = None
    data_quality: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "GraphProfile":
        self.phase = normalize_phase(self.phase)
        self.data_quality = normalize_quality(self.data_quality)
        return self

    def to_artifact_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["data_quality"] = list(self.data_quality)
        return _without_none(data)


def _tokens_from_shape(shape: Optional[str]) -> Optional[int]:
    if not shape:
        return None
    import re

    match = re.search(r"(?:pp|tg)\s*([0-9]+)", shape, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def count_by(items: Iterable[Any], key_fn) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        key = str(key_fn(item))
        counts[key] = counts.get(key, 0) + 1
    return counts
