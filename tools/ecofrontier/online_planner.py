from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .planner_artifact_loader import FrontierArtifact, transition_index
from .planner_schema import GraphEntry, PlanResult, PlannerRequest, PlannerState, TransitionEdge


@dataclass
class PlannerOptions:
    allow_missing_transition: bool = False
    allow_fallback_rows: bool = False
    allow_unstable: bool = False
    allow_thermal_unsafe: bool = False


@dataclass
class CandidatePlan:
    prefill: PlannerState
    decode: PlannerState
    prefill_graph: Optional[GraphEntry]
    decode_graph: Optional[GraphEntry]
    transition: Optional[TransitionEdge]
    transition_used: bool
    transition_missing: bool
    transition_not_amortized: bool
    estimated_energy_mj: Optional[float]
    energy_complete: bool
    missing_energy_terms: List[str]
    graph_required_kv: Optional[int]
    graph_usable_kv_slots: Optional[int]


class OnlinePlanner:
    def __init__(self, artifact: FrontierArtifact, options: Optional[PlannerOptions] = None):
        self.artifact = artifact
        self.options = options or PlannerOptions()
        self._transitions = transition_index(artifact.transitions)

    def plan(self, request: PlannerRequest) -> PlanResult:
        reject_counts: Counter[str] = Counter()
        prefill_candidates = self._phase_candidates("prefill", request.prompt_tokens, request, reject_counts)
        decode_candidates = self._phase_candidates("decode", request.context_len, request, reject_counts)

        feasible: List[CandidatePlan] = []
        rejected = 0
        for prefill in prefill_candidates:
            for decode in decode_candidates:
                candidate = self._compose_candidate(prefill, decode, request, reject_counts)
                rejected += candidate is None
                if candidate is not None:
                    feasible.append(candidate)

        if feasible:
            chosen = min(feasible, key=self._score)
            return self._result_from_candidate(request, chosen, "Feasible", len(feasible), rejected, reject_counts)

        best_effort = self._best_effort(prefill_candidates, decode_candidates, request, reject_counts)
        if best_effort is not None:
            if "no_state_meets_slo" not in reject_counts:
                reject_counts["no_state_meets_slo"] += 1
            return self._result_from_candidate(
                request,
                best_effort,
                "SLO_INFEASIBLE_BEST_EFFORT",
                0,
                rejected,
                reject_counts,
                forced_selected_by="fastest_safe_best_effort",
                extra_reject_reasons=["no_state_meets_slo"],
            )

        return PlanResult(
            request=request,
            status="NO_SAFE_PLAN",
            selected_by="no_safe_plan",
            chosen_prefill_state=None,
            chosen_decode_state=None,
            chosen_prefill_graph=None,
            chosen_decode_graph=None,
            feasible_plan_count=0,
            rejected_plan_count=rejected,
            estimated_ttft_ms=None,
            estimated_tbt_us=None,
            estimated_energy_mj=None,
            energy_complete=False,
            missing_energy_terms=["prefill_state", "decode_state"],
            slo_check_basis="unknown",
            latency_quantile="unknown",
            prefill_length_match="unavailable_out_of_range",
            decode_length_match="unavailable_out_of_range",
            prefill_latency_source="unavailable",
            decode_latency_source="unavailable",
            tbt_source="unavailable",
            ttft_source="unavailable",
            ttft_complete=False,
            power_basis="unavailable",
            transition_used=False,
            transition_type="none",
            transition_total_blocking_us=None,
            transition_energy_complete=False,
            transition_not_amortized_but_best_effort=False,
            graph_required_kv=None,
            graph_usable_kv_slots=None,
            reject_reasons=sorted(reject_counts),
            reject_counts=dict(reject_counts),
            artifact_caveats=list(self.artifact.caveats),
        )

    def _phase_candidates(
        self,
        phase: str,
        requested_length: int,
        request: PlannerRequest,
        reject_counts: Counter[str],
        enforce_slo: bool = True,
    ) -> List[PlannerState]:
        candidates: List[PlannerState] = []
        for state in self._states_for_length(phase, requested_length):
            reason = self._state_filter_reason(state)
            if reason:
                reject_counts[reason] += 1
                continue
            if enforce_slo and phase == "prefill" and state.latency_value > request.slo_ttft_ms:
                reject_counts["violates_ttft_slo"] += 1
                continue
            if enforce_slo and phase == "decode" and state.latency_value > request.slo_tbt_us:
                reject_counts["violates_tbt_slo"] += 1
                continue
            candidates.append(state)
        return candidates

    def _states_for_length(self, phase: str, requested_length: int) -> List[PlannerState]:
        states = [state for state in self.artifact.states if state.phase == phase]
        exact = [state for state in states if state.length_value == requested_length]
        if exact:
            return exact
        if not self.artifact.compiler_config.get("allow_length_interpolation", False):
            return []
        lower = [state.length_value for state in states if state.length_value <= requested_length]
        upper = [state.length_value for state in states if state.length_value >= requested_length]
        if not lower or not upper:
            return []
        selected_lengths = {max(lower), min(upper)}
        return [state for state in states if state.length_value in selected_lengths]

    def _state_filter_reason(self, state: PlannerState) -> Optional[str]:
        if not state.supported:
            return "unsupported_state"
        if state.fallback_used and not self.options.allow_fallback_rows:
            return "fallback_state"
        if not state.stable and not self.options.allow_unstable:
            return "unstable_state"
        if not state.thermal_safe and not self.options.allow_thermal_unsafe:
            return "thermal_unsafe_state"
        return None

    def _compose_candidate(
        self,
        prefill: PlannerState,
        decode: PlannerState,
        request: PlannerRequest,
        reject_counts: Counter[str],
        enforce_transition: bool = True,
        allow_missing_transition: Optional[bool] = None,
    ) -> Optional[CandidatePlan]:
        prefill_graph, prefill_graph_reason = self._graph_for_state(prefill, request)
        if prefill_graph_reason:
            reject_counts[prefill_graph_reason] += 1
            return None
        decode_graph, decode_graph_reason = self._graph_for_state(decode, request)
        if decode_graph_reason:
            reject_counts[decode_graph_reason] += 1
            return None

        transition, transition_missing, transition_not_amortized = self._transition_for_candidate(decode, request)
        missing_transition_allowed = self.options.allow_missing_transition if allow_missing_transition is None else allow_missing_transition
        if enforce_transition and transition_missing and not missing_transition_allowed:
            reject_counts["transition_missing"] += 1
            return None
        if enforce_transition and transition_not_amortized:
            reject_counts["transition_not_amortized"] += 1
            return None

        energy_complete, estimated_energy_mj, missing_terms = self._energy_for_candidate(
            prefill,
            decode,
            request,
            transition,
            prefill_graph,
            decode_graph,
        )
        graph_required, graph_usable = self._graph_capacity_trace(prefill_graph, decode_graph, request)
        return CandidatePlan(
            prefill=prefill,
            decode=decode,
            prefill_graph=prefill_graph,
            decode_graph=decode_graph,
            transition=transition,
            transition_used=transition is not None,
            transition_missing=transition_missing,
            transition_not_amortized=transition_not_amortized,
            estimated_energy_mj=estimated_energy_mj,
            energy_complete=energy_complete,
            missing_energy_terms=missing_terms,
            graph_required_kv=graph_required,
            graph_usable_kv_slots=graph_usable,
        )

    def _graph_for_state(self, state: PlannerState, request: PlannerRequest) -> Tuple[Optional[GraphEntry], Optional[str]]:
        if state.backend != "QNN_NPU":
            return None, None
        graph = self._explicit_graph(state, request)
        if graph is None:
            if state.graph_id:
                return None, "graph_missing_explicit_usable_kv_slots"
            if not self.artifact.graphs:
                return None, None
            return None, "graph_missing"
        required = graph.required_kv(request)
        if required > (graph.usable_kv_slots or -1):
            return None, "graph_capacity_unsafe"
        return graph, None

    def _explicit_graph(self, state: PlannerState, request: PlannerRequest) -> Optional[GraphEntry]:
        workpoint = state.npu_workpoint
        usable = [
            graph
            for graph in self.artifact.graphs
            if graph.phase == state.phase
            and graph.has_explicit_capacity
            and graph.supports_workpoint(workpoint)
            and (not state.graph_id or graph.graph_id == state.graph_id)
        ]
        if not usable:
            return None
        capacity_safe = [graph for graph in usable if graph.required_kv(request) <= (graph.usable_kv_slots or -1)]
        if not capacity_safe:
            return min(usable, key=lambda graph: graph.usable_kv_slots or 0)
        return min(
            capacity_safe,
            key=lambda graph: (
                graph.profiled_energy_mj is None,
                graph.profiled_energy_mj if graph.profiled_energy_mj is not None else float("inf"),
                graph.profiled_exec_us if graph.profiled_exec_us is not None else float("inf"),
                graph.profiled_load_us if graph.profiled_load_us is not None else float("inf"),
                graph.usable_kv_slots or 0,
                graph.graph_id,
            ),
        )

    def _transition_for_candidate(
        self,
        decode: PlannerState,
        request: PlannerRequest,
    ) -> Tuple[Optional[TransitionEdge], bool, bool]:
        if not request.current_state_id or request.current_state_id == decode.state_id:
            return None, False, False
        edges = self._transitions.get((request.current_state_id, decode.state_id), [])
        edge = self._edge_for_context(edges, request.context_len)
        if edge is None:
            return None, True, False
        if not edge.supported:
            return edge, False, True
        current_tbt = self._current_state_tbt(request.current_state_id, request.context_len)
        if current_tbt is not None and decode.tbt_us is not None and edge.total_blocking_us is not None:
            saved = request.predicted_output_mean * max(0.0, current_tbt - decode.tbt_us)
            if saved <= edge.total_blocking_us:
                return edge, False, True
        return edge, False, False

    def _edge_for_context(self, edges: Iterable[TransitionEdge], context_len: int) -> Optional[TransitionEdge]:
        exact: Optional[TransitionEdge] = None
        nearest: Optional[TransitionEdge] = None
        for edge in edges:
            if edge.context_len == context_len:
                exact = edge
                break
            if edge.context_len is None:
                nearest = edge
            elif nearest is None or abs(edge.context_len - context_len) < abs((nearest.context_len or 0) - context_len):
                nearest = edge
        return exact or nearest

    def _current_state_tbt(self, state_id: str, context_len: int) -> Optional[float]:
        states = [
            state
            for state in self.artifact.states
            if state.phase == "decode" and state.state_id == state_id and state.tbt_us is not None
        ]
        exact = [state for state in states if state.length_value == context_len]
        if exact:
            return exact[0].tbt_us
        if not states:
            return None
        return min(states, key=lambda state: abs(state.length_value - context_len)).tbt_us

    def _energy_for_candidate(
        self,
        prefill: PlannerState,
        decode: PlannerState,
        request: PlannerRequest,
        transition: Optional[TransitionEdge],
        prefill_graph: Optional[GraphEntry],
        decode_graph: Optional[GraphEntry],
    ) -> Tuple[bool, Optional[float], List[str]]:
        total = 0.0
        missing: List[str] = []
        complete = True

        prefill_energy = prefill.energy_mj_per_request
        if prefill_energy is None and prefill.active_power_mw is not None and prefill.ttft_ms is not None:
            prefill_energy = prefill.active_power_mw * prefill.ttft_ms / 1000.0
            complete = False
        if prefill_energy is None:
            missing.append("prefill_energy_mj")
            complete = False
        else:
            total += prefill_energy
            if not prefill.energy_complete:
                complete = False

        if decode.energy_mj_per_token is None:
            missing.append("decode_energy_mj_per_token")
            complete = False
        else:
            total += request.predicted_output_mean * decode.energy_mj_per_token
            if not decode.energy_complete:
                complete = False

        if transition is not None:
            if transition.transition_energy_mj is None or not transition.transition_energy_complete:
                missing.append("transition_energy_mj")
                complete = False
            else:
                total += transition.transition_energy_mj

        for label, graph in (("prefill_graph_energy_mj", prefill_graph), ("decode_graph_energy_mj", decode_graph)):
            if graph is None:
                continue
            if graph.profiled_energy_mj is None:
                missing.append(label)
                complete = False
            else:
                total += graph.profiled_energy_mj

        if missing and total == 0.0:
            return False, None, sorted(set(missing))
        return complete and not missing, total, sorted(set(missing))

    def _graph_capacity_trace(
        self,
        prefill_graph: Optional[GraphEntry],
        decode_graph: Optional[GraphEntry],
        request: PlannerRequest,
    ) -> Tuple[Optional[int], Optional[int]]:
        graph = decode_graph or prefill_graph
        if graph is None:
            return None, None
        return graph.required_kv(request), graph.usable_kv_slots

    def _score(self, candidate: CandidatePlan) -> Tuple[object, ...]:
        decode = candidate.decode
        if candidate.energy_complete and candidate.estimated_energy_mj is not None:
            return (0, candidate.estimated_energy_mj, decode.latency_value, decode.state_id)
        if candidate.estimated_energy_mj is not None:
            return (1, candidate.estimated_energy_mj, candidate.transition.transition_energy_mj if candidate.transition else 0.0, decode.latency_value, decode.state_id)
        duration_ms = 0.0
        if candidate.prefill.ttft_ms is not None:
            duration_ms += candidate.prefill.ttft_ms
        if decode.tbt_us is not None:
            duration_ms += decode.tbt_us / 1000.0
        power = decode.active_power_mw if decode.active_power_mw is not None else float("inf")
        transition = candidate.transition.total_blocking_us if candidate.transition and candidate.transition.total_blocking_us is not None else 0.0
        return (2, power * duration_ms, power, transition, decode.latency_value, decode.state_id)

    def _best_effort(
        self,
        prefill_candidates: List[PlannerState],
        decode_candidates: List[PlannerState],
        request: PlannerRequest,
        reject_counts: Counter[str],
    ) -> Optional[CandidatePlan]:
        if not prefill_candidates:
            prefill_candidates = self._phase_candidates("prefill", request.prompt_tokens, request, reject_counts, enforce_slo=False)
        if not decode_candidates:
            decode_candidates = self._phase_candidates("decode", request.context_len, request, reject_counts, enforce_slo=False)
        best: Optional[CandidatePlan] = None
        for prefill in prefill_candidates:
            for decode in decode_candidates:
                candidate = self._compose_candidate(
                    prefill,
                    decode,
                    request,
                    reject_counts,
                    enforce_transition=True,
                    allow_missing_transition=True,
                )
                if candidate is None:
                    continue
                if best is None or (prefill.latency_value + decode.latency_value / 1000.0, decode.state_id) < (
                    best.prefill.latency_value + best.decode.latency_value / 1000.0,
                    best.decode.state_id,
                ):
                    best = candidate
        return best

    def _result_from_candidate(
        self,
        request: PlannerRequest,
        candidate: CandidatePlan,
        status: str,
        feasible_count: int,
        rejected_count: int,
        reject_counts: Counter[str],
        forced_selected_by: Optional[str] = None,
        extra_reject_reasons: Optional[List[str]] = None,
    ) -> PlanResult:
        if forced_selected_by:
            selected_by = forced_selected_by
        elif candidate.energy_complete:
            selected_by = "measured_energy_under_slo"
        elif candidate.estimated_energy_mj is not None:
            selected_by = "estimated_energy_incomplete"
        else:
            selected_by = "latency_power_fallback"

        transition_energy_complete = True
        if candidate.transition is not None:
            transition_energy_complete = candidate.transition.transition_energy_complete
        elif request.current_state_id and request.current_state_id != candidate.decode.state_id:
            transition_energy_complete = False

        reject_reasons = sorted(set(reject_counts) | set(extra_reject_reasons or []))
        return PlanResult(
            request=request,
            status=status,
            selected_by=selected_by,
            chosen_prefill_state=candidate.prefill.state_id,
            chosen_decode_state=candidate.decode.state_id,
            chosen_prefill_graph=candidate.prefill_graph.graph_id if candidate.prefill_graph else None,
            chosen_decode_graph=candidate.decode_graph.graph_id if candidate.decode_graph else None,
            feasible_plan_count=feasible_count,
            rejected_plan_count=rejected_count + sum(reject_counts.values()),
            estimated_ttft_ms=candidate.prefill.ttft_ms,
            estimated_tbt_us=candidate.decode.tbt_us,
            estimated_energy_mj=candidate.estimated_energy_mj,
            energy_complete=candidate.energy_complete,
            missing_energy_terms=candidate.missing_energy_terms,
            slo_check_basis=f"{candidate.prefill.slo_check_basis}+{candidate.decode.slo_check_basis}",
            latency_quantile=_combined_quantile(candidate.prefill.latency_quantile, candidate.decode.latency_quantile),
            prefill_length_match=_length_match(candidate.prefill, request.prompt_tokens),
            decode_length_match=_length_match(candidate.decode, request.context_len),
            prefill_latency_source=candidate.prefill.latency_source,
            decode_latency_source=candidate.decode.latency_source,
            tbt_source=candidate.decode.tbt_source,
            ttft_source=candidate.prefill.ttft_source,
            ttft_complete=candidate.prefill.latency_complete,
            power_basis=_combined_power_basis(candidate.prefill, candidate.decode),
            transition_used=candidate.transition_used,
            transition_type=_transition_type(request, candidate),
            transition_total_blocking_us=candidate.transition.total_blocking_us if candidate.transition else None,
            transition_energy_complete=transition_energy_complete,
            transition_not_amortized_but_best_effort=False,
            graph_required_kv=candidate.graph_required_kv,
            graph_usable_kv_slots=candidate.graph_usable_kv_slots,
            reject_reasons=reject_reasons,
            reject_counts=dict(reject_counts),
            artifact_caveats=list(self.artifact.caveats),
        )


def _combined_quantile(prefill: str, decode: str) -> str:
    if prefill == decode:
        return prefill
    return f"{prefill}+{decode}"


def _length_match(state: PlannerState, requested_length: int) -> str:
    if state.length_value == requested_length:
        return "exact"
    return "nearest_bucket_demo"


def _combined_power_basis(prefill: PlannerState, decode: PlannerState) -> str:
    if prefill.power_basis == decode.power_basis:
        return prefill.power_basis
    return f"{prefill.power_basis}+{decode.power_basis}"


def _transition_type(request: PlannerRequest, candidate: CandidatePlan) -> str:
    if not request.current_state_id or request.current_state_id == candidate.decode.state_id:
        return "none"
    if candidate.transition_missing:
        return "current_to_decode_missing"
    if candidate.transition_used:
        return "current_to_decode"
    return "current_to_decode_untracked"
