#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="${ROOT_DIR}/scripts/run_insightB_transition_overhead.sh"

source_script() {
    # shellcheck source=/dev/null
    INSIGHTB_TRANSITION_OVERHEAD_LIB_ONLY=1 source "${SCRIPT_PATH}"
}

test_transition_header_schema() {
    local csv
    csv="$(mktemp)"

    ensure_transition_header "${csv}"

    local header
    header="$(sed -n '1p' "${csv}")"
    rm -f "${csv}"

    [[ "${header}" == "${TRANSITION_HEADER}" ]] || {
        printf 'unexpected transition CSV header:\n%s\n' "${header}" >&2
        return 1
    }
}

test_state_aliases_map_to_routes_and_controls() {
    [[ "$(state_route npu_burst)" == "qnn-npu" ]] || {
        printf 'npu_burst route mismatch\n' >&2
        return 1
    }
    [[ "$(state_route gpu_734)" == "opencl" ]] || {
        printf 'gpu_734 route mismatch\n' >&2
        return 1
    }
    [[ "$(state_route cpu_big2_2649)" == "cpu" ]] || {
        printf 'cpu_big2_2649 route mismatch\n' >&2
        return 1
    }
    [[ "$(state_npu_workpoint npu_low_balanced)" == "low_balanced" ]] || {
        printf 'npu_low_balanced workpoint mismatch\n' >&2
        return 1
    }
    [[ "$(state_gpu_freq_hz gpu_967)" == "967000000" ]] || {
        printf 'gpu_967 frequency mismatch\n' >&2
        return 1
    }
    is_qnn_state npu_burst || {
        printf 'npu_burst should be a QNN state\n' >&2
        return 1
    }
}

test_same_route_control_kind_classifies_gpu_and_qnn_control_only_transitions() {
    [[ "$(same_route_control_kind gpu_734 gpu_967)" == "gpu_freq" ]] || {
        printf 'gpu_734->gpu_967 should be classified as gpu_freq control transition\n' >&2
        return 1
    }
    [[ "$(same_route_control_kind gpu_967 gpu_734)" == "gpu_freq" ]] || {
        printf 'gpu_967->gpu_734 should be classified as gpu_freq control transition\n' >&2
        return 1
    }
    [[ "$(same_route_control_kind npu_burst npu_low_balanced)" == "qnn_workpoint" ]] || {
        printf 'npu_burst->npu_low_balanced should be classified as qnn_workpoint control transition\n' >&2
        return 1
    }
    [[ "$(same_route_control_kind npu_burst gpu_734)" == "none" ]] || {
        printf 'npu_burst->gpu_734 should not be a same-route control transition\n' >&2
        return 1
    }
    [[ "$(same_route_control_kind gpu_734 gpu_734)" == "none" ]] || {
        printf 'gpu_734->gpu_734 should not be a transition\n' >&2
        return 1
    }
}

test_qnn_transition_cache_guard_checks_from_and_to_segments() {
    local status
    status="$(transition_qnn_support_status npu_burst gpu_734 1792 16 64 32 1920)"
    [[ "${status}" == "ok,1840" ]] || {
        printf 'unexpected supported from-QNN guard: %s\n' "${status}" >&2
        return 1
    }

    status="$(transition_qnn_support_status gpu_734 npu_low_balanced 1810 16 64 32 1920)"
    [[ "${status}" == "unsupported_by_current_aot_cache_size,1922" ]] || {
        printf 'unexpected unsupported to-QNN guard: %s\n' "${status}" >&2
        return 1
    }

    status="$(transition_qnn_support_status gpu_734 gpu_967 4096 16 64 32 '')"
    [[ "${status}" == "ok," ]] || {
        printf 'non-QNN transition should not require cache size: %s\n' "${status}" >&2
        return 1
    }
}

test_control_transition_trace_parser_maps_gpu_freq_only_fields() {
    local log_file
    log_file="$(mktemp)"
    cat > "${log_file}" <<'EOF'
CONTROL_TRANSITION_TRACE from=gpu_734 to=gpu_967 gpu_freq_apply_us=123 total_blocking_us=123 transition_energy_mj= transition_energy_source=unavailable success=1 fallback=0 support_status=ok requested_gpu_freq_hz=967000000 actual_gpu_freq_hz=967000000
EOF

    local parsed
    parsed="$(parse_round_logs "${log_file}" opencl 0)"
    rm -f "${log_file}"

    local decision_us route_apply_us policy_apply_us qnn_workpoint_apply_us gpu_freq_apply_us sched_reserve_us
    local kv_handoff_us graph_rebuild_us decode_entry_us total_blocking_us first_token_gap_us post_switch_tbt_us
    local transition_energy_mj transition_energy_source switch_success fallback_used support_status
    IFS=',' read -r \
        decision_us route_apply_us policy_apply_us qnn_workpoint_apply_us gpu_freq_apply_us sched_reserve_us \
        kv_handoff_us graph_rebuild_us decode_entry_us total_blocking_us first_token_gap_us post_switch_tbt_us \
        transition_energy_mj transition_energy_source switch_success fallback_used support_status <<< "${parsed}"

    [[ "${gpu_freq_apply_us}" == "123" && "${total_blocking_us}" == "123" && "${transition_energy_source}" == "unavailable" && "${switch_success}" == "1" && "${support_status}" == "ok" ]] || {
        printf 'unexpected control transition parse output: %s\n' "${parsed}" >&2
        return 1
    }
}

test_gpu_control_apply_command_records_target_frequency_and_trace() {
    GPU_MIN_FREQ_PATH="/sys/test_gpu/min_freq"
    GPU_MAX_FREQ_PATH="/sys/test_gpu/max_freq"
    GPU_CUR_FREQ_PATH="/sys/test_gpu/cur_freq"
    GPU_GOVERNOR_PATH=""
    GPU_PIN_GOVERNOR=""

    local command
    command="$(build_gpu_control_apply_command gpu_734 gpu_967)"

    [[ "${command}" == *"CONTROL_TRANSITION_TRACE from=gpu_734 to=gpu_967"* ]] || {
        printf 'GPU control command should emit CONTROL_TRANSITION_TRACE: %s\n' "${command}" >&2
        return 1
    }
    [[ "${command}" == *"requested_gpu_freq_hz=967000000"* ]] || {
        printf 'GPU control command should record requested target frequency: %s\n' "${command}" >&2
        return 1
    }
    [[ "${command}" == *"date +%s%N"* && "${command}" != *"awk"* ]] || {
        printf 'GPU control command should use date-based timestamps to avoid nested awk quoting on Android shell: %s\n' "${command}" >&2
        return 1
    }
    [[ "${command}" == *"echo 734000000 > /sys/test_gpu/min_freq"* && "${command}" == *"echo 967000000 > /sys/test_gpu/max_freq"* ]] || {
        printf 'GPU control command should lower min before setting target max: %s\n' "${command}" >&2
        return 1
    }
}

test_qnn_same_route_bench_command_exports_decode_workpoint() {
    MODEL_PATH="/data/local/tmp/model.gguf"
    BENCH_DIR="/data/local/tmp/bench"
    QNN_ACTIVE_CONFIG="/data/local/tmp/qnn/config.json"
    QNN_ACTIVE_MODEL_DIR="/data/local/tmp/qnn"
    QNN_ENABLE_HEXAGON="1"
    QNN_AOT_WRITE_GENERIC_KV="1"
    QNN_AOT_DISABLE_SEED_KV="0"
    NGL="99"
    LLAMA_THREADS="4"
    CONTEXT_LEN="512"
    DECODE_TOKENS_BEFORE_SWITCH="0"
    DECODE_TOKENS_AFTER_SWITCH="16"
    QNN_CACHE_SAFETY_MARGIN="32"
    DEFAULT_CONTEXT_TOKENS="2048"
    BATCH_TOKENS="512"
    UBATCH_TOKENS="512"
    MMAP=""
    TASKSET_MASK="ff"
    LLAMA_BENCH_FAST_EXIT_VALUE="1"
    LLAMA_BENCH_USE_PG_WORKLOAD="1"
    EXPERIMENT_PHASE="phase_boundary"
    OPENCL_QNN_DIRECT_HOST_PTR="0"
    OPENCL_QNN_DIRECT_HOST_PTR_SKIP_UPLOAD="0"

    local command
    command="$(build_remote_bench_command npu_burst npu_low_balanced qnn-npu qnn-npu)"

    [[ "${command}" == *"export GGML_QNN_HTP_WORKPOINT=burst"* ]] || {
        printf 'QNN same-route command should initialize from-state workpoint: %s\n' "${command}" >&2
        return 1
    }
    [[ "${command}" == *"export GGML_HETERO_DYNAMIC_DECODE_QNN_WORKPOINT=low_balanced"* ]] || {
        printf 'QNN same-route command should export decode target workpoint: %s\n' "${command}" >&2
        return 1
    }
    [[ "${command}" == *"export GGML_HETERO_DYNAMIC_PREFILL_ROUTE=qnn-npu"* &&
       "${command}" == *"export GGML_HETERO_DYNAMIC_DECODE_ROUTE=qnn-npu"* ]] || {
        printf 'QNN same-route command should keep both phase routes on qnn-npu: %s\n' "${command}" >&2
        return 1
    }
}

test_dynamic_timing_parser_maps_existing_fields() {
    local line
    line='llama_context: timing phase=decode n_tokens=1 total_wall_us=32480 decide_us=18 apply_us=32000 reserve_us=1900 memory_update_us=42 kv_migration_us=30490 process_ubatch_us=0 bootstrap_sync_us=180 bootstrap_sched_rebuild_us=220 ubatches=0 graph_runs_reused=0 graph_runs_rebuilt=0 route_applied=true route_noop=false bootstrap_ran=true label=decode reason=phase target=opencl'

    local parsed
    parsed="$(parse_transition_timing_line "${line}")"

    [[ "${parsed}" == "18,32000,1900,30490,220,32480,1" ]] || {
        printf 'unexpected timing parse output: %s\n' "${parsed}" >&2
        return 1
    }
}

test_phase_boundary_success_ignores_stage_route_target_string() {
    local log_file
    log_file="$(mktemp)"
    cat > "${log_file}" <<'EOF'
synchronize: timing phase=decode n_tokens=1 total_wall_us=2624 decide_us=2 apply_us=14 reserve_us=2554 memory_update_us=0 kv_migration_us=0 process_ubatch_us=0 bootstrap_sync_us=0 bootstrap_sched_rebuild_us=0 ubatches=0 graph_runs_reused=0 graph_runs_rebuilt=0 route_applied=true route_noop=false bootstrap_ran=false label=decode reason=phase-decode-route target=attn=opencl,ffn=opencl,output=opencl
EOF

    local parsed
    parsed="$(parse_round_logs "${log_file}" cpu 0)"
    rm -f "${log_file}"

    [[ "${parsed}" == "2,14,,,,2554,0,0,,2624,,,,unavailable,1,0,ok" ]] || {
        printf 'phase-boundary parse should ignore target stage string, got: %s\n' "${parsed}" >&2
        return 1
    }
}

test_fallback_detector_ignores_library_and_host_warning_messages() {
    local log_file
    log_file="$(mktemp)"
    cat > "${log_file}" <<'EOF'
failed to load /data/local/tmp/libQnnSystem.so, fallback to libQnnSystem.so
llama_context: keeping CPU compute buffers on CPU memory because qnn-npu host fallback can corrupt or slow mixed qnn/cpu/opencl contexts
EOF

    local detected
    detected="$(detect_fallback_used "${log_file}")"
    rm -f "${log_file}"

    [[ "${detected}" == "0" ]] || {
        printf 'library/host-warning fallback messages should not count as runtime fallback\n' >&2
        return 1
    }
}

test_support_classifier_does_not_treat_normal_kv_contract_logs_as_failure_cause() {
    local log_file
    log_file="$(mktemp)"
    cat > "${log_file}" <<'EOF'
operator(): promoting allocated attn KV contract for dynamic qnn-prefill/opencl-decode to layout=stage-shared transfer=qnn-rpcmem
llama_kv_cache: attn KV contract layout=stage-shared transfer=qnn-rpcmem producer=qnn-npu consumer=opencl
EOF

    local status
    status="$(classify_transition_support_status 1 0 "${log_file}")"
    rm -f "${log_file}"

    [[ "${status}" == "failed" ]] || {
        printf 'normal KV contract logs should not be classified as KV contract failure, got: %s\n' "${status}" >&2
        return 1
    }
}

test_unified_transition_trace_parser_takes_precedence_fields() {
    local line
    line='TRANSITION_TRACE from=npu_burst to=gpu_734 decision_us=11 route_apply_us=22 policy_apply_us=3 qnn_workpoint_apply_us=4 gpu_freq_apply_us=5 sched_reserve_us=6 kv_handoff_us=7 graph_rebuild_us=8 decode_entry_us=9 total_blocking_us=10 first_token_gap_us=11 post_switch_tbt_us=12 transition_energy_mj=13.5 transition_energy_source=estimated success=1 fallback=0 support_status=ok'

    local parsed
    parsed="$(parse_transition_trace_line "${line}")"

    [[ "${parsed}" == "11,22,3,4,5,6,7,8,9,10,11,12,13.5,estimated,1,0,ok" ]] || {
        printf 'unexpected unified trace parse output: %s\n' "${parsed}" >&2
        return 1
    }
}

test_runtime_transition_trace_maps_blocking_and_decode_entry_separately() {
    local line
    line='TRANSITION_TRACE phase=prefill_to_decode decision_us=2 route_apply_us=14 policy_apply_us= qnn_workpoint_apply_us= gpu_freq_apply_us= sched_reserve_us=6576 kv_handoff_us=0 graph_rebuild_us=0 decode_entry_us=58052 total_blocking_us=6734 first_token_gap_us= post_switch_tbt_us= transition_energy_mj= transition_energy_source=unavailable success=1 fallback=0 support_status=ok process_ubatch_us=51318 total_wall_us=58052'

    local parsed
    parsed="$(parse_transition_trace_line "${line}")"

    [[ "${parsed}" == "2,14,,,,6576,0,0,58052,6734,,,,unavailable,1,0,ok" ]] || {
        printf 'runtime transition trace should separate blocking and decode entry, got: %s\n' "${parsed}" >&2
        return 1
    }
}

test_runtime_transition_trace_maps_qnn_workpoint_apply_time() {
    local line
    line='TRANSITION_TRACE phase=prefill_to_decode decision_us=2 route_apply_us=0 policy_apply_us= qnn_workpoint_apply_us=321 gpu_freq_apply_us= sched_reserve_us=0 kv_handoff_us=0 graph_rebuild_us=0 decode_entry_us=60000 total_blocking_us=1000 first_token_gap_us= post_switch_tbt_us= transition_energy_mj= transition_energy_source=unavailable success=1 fallback=0 support_status=ok process_ubatch_us=59000 total_wall_us=60000'

    local parsed
    parsed="$(parse_transition_trace_line "${line}")"

    [[ "${parsed}" == "2,0,,321,,0,0,0,60000,1000,,,,unavailable,1,0,ok" ]] || {
        printf 'runtime transition trace should preserve qnn workpoint apply time, got: %s\n' "${parsed}" >&2
        return 1
    }
}

test_transition_trace_success_is_overridden_by_process_failure() {
    local log_file
    log_file="$(mktemp)"
    cat > "${log_file}" <<'EOF'
TRANSITION_TRACE phase=prefill_to_decode decision_us=2 route_apply_us=14 sched_reserve_us=6576 kv_handoff_us=0 graph_rebuild_us=0 decode_entry_us=58052 total_blocking_us=6734 first_token_gap_us= post_switch_tbt_us= transition_energy_mj= transition_energy_source=unavailable success=1 fallback=0 support_status=ok process_ubatch_us=51318 total_wall_us=58052
EOF

    local parsed
    parsed="$(parse_round_logs "${log_file}" qnn-npu 9)"
    rm -f "${log_file}"

    [[ "${parsed}" == "2,14,,,,6576,0,0,58052,6734,,,,unavailable,0,0,failed" ]] || {
        printf 'process failure should override runtime transition success, got: %s\n' "${parsed}" >&2
        return 1
    }
}

test_token_trace_derives_post_switch_tbt_from_tokens_after_transition() {
    local log_file
    log_file="$(mktemp)"
    cat > "${log_file}" <<'EOF'
TRANSITION_TRACE phase=prefill_to_decode decision_us=2 route_apply_us=14 sched_reserve_us=6576 kv_handoff_us=0 graph_rebuild_us=0 decode_entry_us=58052 total_blocking_us=6734 first_token_gap_us= post_switch_tbt_us= transition_energy_mj= transition_energy_source=unavailable success=1 fallback=0 support_status=ok process_ubatch_us=51318 total_wall_us=58052
DECODE_TOKEN_TRACE phase=decode token_index=1 done_us=1000000 route_applied=1 total_wall_us=900
DECODE_TOKEN_TRACE phase=decode token_index=2 done_us=1000400 route_applied=0 total_wall_us=400
DECODE_TOKEN_TRACE phase=decode token_index=3 done_us=1000900 route_applied=0 total_wall_us=500
DECODE_TOKEN_TRACE phase=decode token_index=4 done_us=1001500 route_applied=0 total_wall_us=600
EOF

    local parsed
    parsed="$(parse_round_logs "${log_file}" opencl 0)"
    rm -f "${log_file}"

    [[ "${parsed}" == "2,14,,,,6576,0,0,58052,6734,,500.00,,unavailable,1,0,ok" ]] || {
        printf 'token traces should derive post-switch TBT without inventing first-token gap, got: %s\n' "${parsed}" >&2
        return 1
    }
}

test_token_trace_does_not_invent_first_token_gap_from_prior_log_tokens() {
    local log_file
    log_file="$(mktemp)"
    cat > "${log_file}" <<'EOF'
DECODE_TOKEN_TRACE phase=decode token_index=7 done_us=2000000 route_applied=0 total_wall_us=450
TRANSITION_TRACE phase=prefill_to_decode decision_us=3 route_apply_us=15 sched_reserve_us=1234 kv_handoff_us=5 graph_rebuild_us=6 decode_entry_us=900 total_blocking_us=100 first_token_gap_us= post_switch_tbt_us= transition_energy_mj= transition_energy_source=unavailable success=1 fallback=0 support_status=ok process_ubatch_us=800 total_wall_us=900
DECODE_TOKEN_TRACE phase=decode token_index=8 done_us=2000750 route_applied=1 total_wall_us=900
DECODE_TOKEN_TRACE phase=decode token_index=9 done_us=2001250 route_applied=0 total_wall_us=500
EOF

    local parsed
    parsed="$(parse_round_logs "${log_file}" qnn-npu 0)"
    rm -f "${log_file}"

    [[ "${parsed}" == "3,15,,,,1234,5,6,900,100,,500.00,,unavailable,1,0,ok" ]] || {
        printf 'token traces should not invent first-token gap from earlier log segments, got: %s\n' "${parsed}" >&2
        return 1
    }
}

test_token_trace_filters_pg_prefill_gap_from_post_switch_tbt() {
    local log_file
    log_file="$(mktemp)"
    cat > "${log_file}" <<'EOF'
TRANSITION_TRACE phase=prefill_to_decode decision_us=2 route_apply_us=12 sched_reserve_us=2658 kv_handoff_us=0 graph_rebuild_us=0 decode_entry_us=2732 total_blocking_us=2732 first_token_gap_us= post_switch_tbt_us= transition_energy_mj= transition_energy_source=unavailable success=1 fallback=0 support_status=ok process_ubatch_us=0 total_wall_us=2732
DECODE_TOKEN_TRACE phase=decode token_index=1 done_us=55521075243 route_applied=1 total_wall_us=2732 process_ubatch_us=0
DECODE_TOKEN_TRACE phase=decode token_index=2 done_us=55589379615 route_applied=0 total_wall_us=68227926 process_ubatch_us=6950
DECODE_TOKEN_TRACE phase=decode token_index=3 done_us=55589430516 route_applied=0 total_wall_us=50737 process_ubatch_us=8329
DECODE_TOKEN_TRACE phase=decode token_index=4 done_us=55589481902 route_applied=0 total_wall_us=51308 process_ubatch_us=7397
DECODE_TOKEN_TRACE phase=decode token_index=5 done_us=55589531794 route_applied=0 total_wall_us=49755 process_ubatch_us=6854
DECODE_TOKEN_TRACE phase=decode token_index=6 done_us=55589583206 route_applied=0 total_wall_us=51327 process_ubatch_us=6683
EOF

    local parsed
    parsed="$(parse_round_logs "${log_file}" opencl 0)"
    rm -f "${log_file}"

    [[ "${parsed}" == "2,12,,,,2658,0,0,2732,2732,,50781.75,,unavailable,1,0,ok" ]] || {
        printf 'pg prefill gap should be filtered from post-switch TBT, got: %s\n' "${parsed}" >&2
        return 1
    }
}

test_parser_uses_last_transition_trace_when_raw_log_has_multiple_segments() {
    local log_file
    log_file="$(mktemp)"
    cat > "${log_file}" <<'EOF'
TRANSITION_TRACE phase=prefill_to_decode decision_us=99 route_apply_us=99 sched_reserve_us=99 kv_handoff_us=99 graph_rebuild_us=99 decode_entry_us=99 total_blocking_us=99 first_token_gap_us= post_switch_tbt_us= transition_energy_mj= transition_energy_source=unavailable success=1 fallback=0 support_status=ok process_ubatch_us=0 total_wall_us=99
DECODE_TOKEN_TRACE phase=decode token_index=1 done_us=1000000 route_applied=1 total_wall_us=99
DECODE_TOKEN_TRACE phase=decode token_index=2 done_us=3000000 route_applied=0 total_wall_us=2000000
TRANSITION_TRACE phase=prefill_to_decode decision_us=4 route_apply_us=16 sched_reserve_us=32 kv_handoff_us=0 graph_rebuild_us=0 decode_entry_us=64 total_blocking_us=64 first_token_gap_us= post_switch_tbt_us= transition_energy_mj= transition_energy_source=unavailable success=1 fallback=0 support_status=ok process_ubatch_us=0 total_wall_us=64
DECODE_TOKEN_TRACE phase=decode token_index=1 done_us=4000000 route_applied=1 total_wall_us=64
DECODE_TOKEN_TRACE phase=decode token_index=2 done_us=4000500 route_applied=0 total_wall_us=500
DECODE_TOKEN_TRACE phase=decode token_index=3 done_us=4001100 route_applied=0 total_wall_us=600
EOF

    local parsed
    parsed="$(parse_round_logs "${log_file}" opencl 0)"
    rm -f "${log_file}"

    [[ "${parsed}" == "4,16,,,,32,0,0,64,64,,550.00,,unavailable,1,0,ok" ]] || {
        printf 'parser should use the last complete transition segment, got: %s\n' "${parsed}" >&2
        return 1
    }
}

test_aggregate_rounds_uses_means_and_success_rate() {
    local rounds_csv
    rounds_csv="$(mktemp)"
    cat > "${rounds_csv}" <<'EOF'
round,decision_us,route_apply_us,policy_apply_us,qnn_workpoint_apply_us,gpu_freq_apply_us,sched_reserve_us,kv_handoff_us,graph_rebuild_us,decode_entry_us,total_blocking_us,first_token_gap_us,post_switch_tbt_us,transition_energy_mj,transition_energy_source,switch_success,fallback_used,support_status,raw_log_path,exit_code,measurement_note
1,10,20,,4,,6,8,10,,100,120,1000,,unavailable,1,0,ok,/tmp/a.log,0,
2,20,40,,6,,8,12,14,,200,240,1100,,unavailable,1,1,ok,/tmp/b.log,0,
EOF

    local summary
    summary="$(aggregate_transition_rounds "${rounds_csv}")"
    rm -f "${rounds_csv}"

    [[ "${summary}" == "15.00,30.00,,5.00,,7.00,10.00,12.00,,150.00,180.00,1050.00,,unavailable,1.000000,1,ok,/tmp/a.log;/tmp/b.log" ]] || {
        printf 'unexpected aggregate output: %s\n' "${summary}" >&2
        return 1
    }
}

main() {
    source_script
    test_transition_header_schema
    test_state_aliases_map_to_routes_and_controls
    test_same_route_control_kind_classifies_gpu_and_qnn_control_only_transitions
    test_qnn_transition_cache_guard_checks_from_and_to_segments
    test_control_transition_trace_parser_maps_gpu_freq_only_fields
    test_gpu_control_apply_command_records_target_frequency_and_trace
    test_qnn_same_route_bench_command_exports_decode_workpoint
    test_dynamic_timing_parser_maps_existing_fields
    test_phase_boundary_success_ignores_stage_route_target_string
    test_fallback_detector_ignores_library_and_host_warning_messages
    test_support_classifier_does_not_treat_normal_kv_contract_logs_as_failure_cause
    test_unified_transition_trace_parser_takes_precedence_fields
    test_runtime_transition_trace_maps_blocking_and_decode_entry_separately
    test_runtime_transition_trace_maps_qnn_workpoint_apply_time
    test_transition_trace_success_is_overridden_by_process_failure
    test_token_trace_derives_post_switch_tbt_from_tokens_after_transition
    test_token_trace_does_not_invent_first_token_gap_from_prior_log_tokens
    test_token_trace_filters_pg_prefill_gap_from_post_switch_tbt
    test_parser_uses_last_transition_trace_when_raw_log_has_multiple_segments
    test_aggregate_rounds_uses_means_and_success_rate
    printf 'Insight B transition overhead helper tests ok\n'
}

main "$@"
