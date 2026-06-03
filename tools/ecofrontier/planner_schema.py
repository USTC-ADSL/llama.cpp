from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan", "n/a", "na", "-", "unavailable"}:
        return default
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return default


def as_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    parsed = as_float(value)
    if parsed is None:
        return default
    return int(round(parsed))


def as_bool(value: Any, default: bool = False) -> bool:
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


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


@dataclass(frozen=True)
class PlannerRequest:
    request_id: str
    prompt_tokens: int
    context_len: int
    predicted_output_mean: int
    predicted_output_hi: int
    slo_ttft_ms: float
    slo_tbt_us: float
    current_state_id: str = ""
    current_graph_id: str = ""
    current_temp_c: Optional[float] = None

    @classmethod
    def from_mapping(cls, data: Dict[str, Any], index: int = 0) -> "PlannerRequest":
        request_id = str(data.get("request_id") or f"request-{index}")
        prompt_tokens = as_int(data.get("prompt_tokens"))
        context_len = as_int(data.get("context_len"))
        predicted_output_mean = as_int(data.get("predicted_output_mean"))
        predicted_output_hi = as_int(data.get("predicted_output_hi"), predicted_output_mean)
        slo_ttft_ms = as_float(data.get("slo_ttft_ms"))
        slo_tbt_us = as_float(data.get("slo_tbt_us"))
        missing = [
            name
            for name, value in (
                ("prompt_tokens", prompt_tokens),
                ("context_len", context_len),
                ("predicted_output_mean", predicted_output_mean),
                ("predicted_output_hi", predicted_output_hi),
                ("slo_ttft_ms", slo_ttft_ms),
                ("slo_tbt_us", slo_tbt_us),
            )
            if value is None
        ]
        if missing:
            raise ValueError(f"{request_id}: missing required request fields: {', '.join(missing)}")
        return cls(
            request_id=request_id,
            prompt_tokens=int(prompt_tokens),
            context_len=int(context_len),
            predicted_output_mean=int(predicted_output_mean),
            predicted_output_hi=int(predicted_output_hi),
            slo_ttft_ms=float(slo_ttft_ms),
            slo_tbt_us=float(slo_tbt_us),
            current_state_id=str(data.get("current_state_id") or ""),
            current_graph_id=str(data.get("current_graph_id") or ""),
            current_temp_c=as_float(data.get("current_temp_c")),
        )


@dataclass(frozen=True)
class PlannerState:
    state_id: str
    backend: str
    phase: str
    length_value: int
    latency_value: float
    latency_field: str
    latency_quantile: str
    slo_check_basis: str
    latency_source: str
    latency_complete: bool
    tbt_us: Optional[float] = None
    ttft_ms: Optional[float] = None
    tbt_source: str = "unknown"
    ttft_source: str = "unknown"
    active_power_mw: Optional[float] = None
    power_basis: str = "active_power_mw"
    energy_mj_per_token: Optional[float] = None
    energy_mj_per_request: Optional[float] = None
    energy_source: str = "unavailable"
    energy_complete: bool = False
    support_status: str = "ok"
    fallback_used: bool = False
    stable: bool = True
    thermal_safe: bool = True
    data_quality: List[str] = field(default_factory=list)
    npu_workpoint: str = ""
    graph_id: str = ""
    source_file: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def supported(self) -> bool:
        return self.support_status.lower() == "ok"


@dataclass(frozen=True)
class TransitionEdge:
    from_state_id: str
    to_state_id: str
    context_len: Optional[int] = None
    total_blocking_us: Optional[float] = None
    first_token_gap_us: Optional[float] = None
    post_switch_tbt_us: Optional[float] = None
    transition_energy_mj: Optional[float] = None
    transition_energy_source: str = "unavailable"
    transition_energy_complete: bool = False
    success_rate: Optional[float] = None
    fallback_count: int = 0
    support_status: str = "ok"
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def supported(self) -> bool:
        return self.support_status.lower() == "ok" and self.fallback_count == 0


@dataclass(frozen=True)
class GraphEntry:
    graph_id: str
    phase: str
    chunk_size: Optional[int] = None
    usable_kv_slots: Optional[int] = None
    safety_margin: int = 0
    supported_workpoints: List[str] = field(default_factory=list)
    profiled_load_us: Optional[float] = None
    profiled_warmup_us: Optional[float] = None
    profiled_exec_us: Optional[float] = None
    profiled_energy_mj: Optional[float] = None
    data_quality: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_explicit_capacity(self) -> bool:
        return self.usable_kv_slots is not None

    def supports_workpoint(self, workpoint: str) -> bool:
        if not workpoint or not self.supported_workpoints:
            return True
        return workpoint in self.supported_workpoints

    def required_kv(self, request: PlannerRequest) -> int:
        return request.context_len + request.predicted_output_hi + self.safety_margin


@dataclass
class PlanResult:
    request: PlannerRequest
    status: str
    selected_by: str
    chosen_prefill_state: Optional[str]
    chosen_decode_state: Optional[str]
    chosen_prefill_graph: Optional[str]
    chosen_decode_graph: Optional[str]
    feasible_plan_count: int
    rejected_plan_count: int
    estimated_ttft_ms: Optional[float]
    estimated_tbt_us: Optional[float]
    estimated_energy_mj: Optional[float]
    energy_complete: bool
    missing_energy_terms: List[str]
    slo_check_basis: str
    latency_quantile: str
    prefill_length_match: str
    decode_length_match: str
    prefill_latency_source: str
    decode_latency_source: str
    tbt_source: str
    ttft_source: str
    ttft_complete: bool
    power_basis: str
    transition_used: bool
    transition_type: str
    transition_total_blocking_us: Optional[float]
    transition_energy_complete: bool
    transition_not_amortized_but_best_effort: bool
    graph_required_kv: Optional[int]
    graph_usable_kv_slots: Optional[int]
    reject_reasons: List[str]
    reject_counts: Dict[str, int]
    artifact_caveats: List[str]

    def to_trace_event(self) -> Dict[str, Any]:
        request = self.request
        return {
            "event": "ecofrontier_plan",
            "request_id": request.request_id,
            "prompt_tokens": request.prompt_tokens,
            "context_len": request.context_len,
            "predicted_output_mean": request.predicted_output_mean,
            "predicted_output_hi": request.predicted_output_hi,
            "slo_ttft_ms": request.slo_ttft_ms,
            "slo_tbt_us": request.slo_tbt_us,
            "chosen_prefill_state": self.chosen_prefill_state,
            "chosen_decode_state": self.chosen_decode_state,
            "chosen_prefill_graph": self.chosen_prefill_graph,
            "chosen_decode_graph": self.chosen_decode_graph,
            "feasible_plan_count": self.feasible_plan_count,
            "rejected_plan_count": self.rejected_plan_count,
            "estimated_ttft_ms": self.estimated_ttft_ms,
            "estimated_tbt_us": self.estimated_tbt_us,
            "estimated_energy_mj": self.estimated_energy_mj,
            "energy_complete": self.energy_complete,
            "missing_energy_terms": self.missing_energy_terms,
            "selected_by": self.selected_by,
            "status": self.status,
            "slo_check_basis": self.slo_check_basis,
            "latency_quantile": self.latency_quantile,
            "prefill_length_match": self.prefill_length_match,
            "decode_length_match": self.decode_length_match,
            "prefill_latency_source": self.prefill_latency_source,
            "decode_latency_source": self.decode_latency_source,
            "tbt_source": self.tbt_source,
            "ttft_source": self.ttft_source,
            "ttft_complete": self.ttft_complete,
            "power_basis": self.power_basis,
            "transition_used": self.transition_used,
            "transition_type": self.transition_type,
            "transition_total_blocking_us": self.transition_total_blocking_us,
            "transition_energy_complete": self.transition_energy_complete,
            "transition_not_amortized_but_best_effort": self.transition_not_amortized_but_best_effort,
            "graph_required_kv": self.graph_required_kv,
            "graph_usable_kv_slots": self.graph_usable_kv_slots,
            "reject_reasons": self.reject_reasons,
            "artifact_caveats": self.artifact_caveats,
        }
