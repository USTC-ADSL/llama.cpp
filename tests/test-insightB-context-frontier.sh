#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="${ROOT_DIR}/scripts/run_insightB_context_frontier.sh"

source_script() {
    # shellcheck source=/dev/null
    INSIGHTB_CONTEXT_FRONTIER_LIB_ONLY=1 source "${SCRIPT_PATH}"
}

test_llama_bench_tg_parser_handles_context_depth() {
    local bench_log
    bench_log="$(mktemp)"
    cat > "${bench_log}" <<'EOF'
| model                  | size     | params | backend | ngl | test        | t/s             |
| ---------------------- | -------: | -----: | ------- | --: | ----------: | --------------: |
| qwen2 7B Q4_K - Medium | 4.36 GiB | 7.62 B | CUDA    |  99 | tg64 @ d512 | 116.71 +/- 0.60 |
EOF

    local parsed
    parsed="$(parse_llama_bench_tg "${bench_log}" 64 512)"
    rm -f "${bench_log}"

    [[ "${parsed}" == "116.71,0.60,1,tg64 @ d512" ]] || {
        printf 'unexpected tg parser output: %s\n' "${parsed}" >&2
        return 1
    }
}

test_qnn_config_parser_selects_relevant_batch_sizes() {
    local config
    config="$(mktemp)"
    cat > "${config}" <<'EOF'
{
  "graphs": [
    {"type": "transformers", "batch_size": 1, "cache_size": 1920, "context_size": 2048},
    {"type": "transformers", "batch_size": 128, "cache_size": 1920, "context_size": 2048},
    {"type": "transformers", "batch_size": 256, "cache_size": 1792, "context_size": 2048}
  ]
}
EOF

    local parsed
    parsed="$(parse_qnn_aot_sizes_from_file "${config}" "1 128")"
    rm -f "${config}"

    [[ "${parsed}" == "1920,2048" ]] || {
        printf 'unexpected qnn size parse output: %s\n' "${parsed}" >&2
        return 1
    }
}

test_qnn_cache_guard_applies_default_margin() {
    qnn_cache_guard_status 1792 64 32 1920 >/dev/null
    local ok_status=$?

    set +e
    qnn_cache_guard_status 1856 64 32 1920 >/dev/null
    local unsupported_status=$?
    set -e

    [[ "${ok_status}" == "0" ]] || {
        printf 'expected 1792+64+32 <= 1920 to be supported\n' >&2
        return 1
    }
    [[ "${unsupported_status}" == "1" ]] || {
        printf 'expected 1856+64+32 > 1920 to be unsupported, got %s\n' "${unsupported_status}" >&2
        return 1
    }
}

test_fallback_detector_ignores_library_and_disabled_messages() {
    local log_file
    log_file="$(mktemp)"
    cat > "${log_file}" <<'EOF'
failed to load /data/local/tmp/libQnnSystem.so, fallback to libQnnSystem.so
llama_context: CPU fallback to GPUOpenCL host buffers is disabled by default
EOF

    local detected
    detected="$(detect_fallback_used "${log_file}")"

    printf 'runtime fallback: falling back to CPU\n' >> "${log_file}"
    local detected_runtime
    detected_runtime="$(detect_fallback_used "${log_file}")"
    rm -f "${log_file}"

    [[ "${detected}" == "0" ]] || {
        printf 'library/disabled fallback messages should not count as runtime fallback\n' >&2
        return 1
    }
    [[ "${detected_runtime}" == "1" ]] || {
        printf 'runtime fallback message was not detected\n' >&2
        return 1
    }
}

test_support_classifier_does_not_treat_generic_cache_logs_as_cache_guard_failures() {
    local log_file
    log_file="$(mktemp)"
    cat > "${log_file}" <<'EOF'
load: special tokens cache size = 23
llama_kv_cache: size =   72.00 MiB (  2048 cells,  36 layers,  1/1 seqs)
adb: device offline
EOF

    local status
    status="$(classify_support_status npu_low_balanced bench_failed 255 '' "${log_file}")"
    rm -f "${log_file}"

    [[ "${status}" == "failed" ]] || {
        printf 'generic cache log text should not become cache-size unsupported, got: %s\n' "${status}" >&2
        return 1
    }
}

test_support_classifier_detects_explicit_cache_capacity_errors() {
    local log_file
    log_file="$(mktemp)"
    cat > "${log_file}" <<'EOF'
qnn-npu KV cache footprint exceeds qnn_aot_cache_size
EOF

    local status
    status="$(classify_support_status npu_low_balanced bench_failed 1 '' "${log_file}")"
    rm -f "${log_file}"

    [[ "${status}" == "unsupported_by_current_aot_cache_size" ]] || {
        printf 'explicit cache capacity error was not detected, got: %s\n' "${status}" >&2
        return 1
    }
}

test_optional_display_env_is_recorded_and_forwarded() {
    local -a env_args=(env "DEVICE=device-id")
    SCREEN_BRIGHTNESS_OVERRIDE=128 \
        KEEP_SCREEN_ON_TIMEOUT_MS=1800000 \
        BENCH_DIR=/data/local/tmp/bench \
        QNN_BIN=/data/local/tmp/qnn-bin \
        SAMPLE_INTERVAL_S=1 \
        TASKSET_MASK=80 \
        LLAMA_BENCH_FAST_EXIT_VALUE=1 \
        LLAMA_BENCH_QNN_PREWARM_DECODE=1 \
        append_optional_env env_args \
        SCREEN_BRIGHTNESS_OVERRIDE \
        KEEP_SCREEN_ON_TIMEOUT_MS \
        BENCH_DIR \
        QNN_BIN \
        SAMPLE_INTERVAL_S \
        TASKSET_MASK \
        LLAMA_BENCH_FAST_EXIT_VALUE \
        LLAMA_BENCH_QNN_PREWARM_DECODE

    local joined
    printf -v joined '%s\n' "${env_args[@]}"

    grep -qx 'SCREEN_BRIGHTNESS_OVERRIDE=128' <<< "${joined}" || {
        printf 'missing SCREEN_BRIGHTNESS_OVERRIDE in env args:\n%s\n' "${joined}" >&2
        return 1
    }
    grep -qx 'KEEP_SCREEN_ON_TIMEOUT_MS=1800000' <<< "${joined}" || {
        printf 'missing KEEP_SCREEN_ON_TIMEOUT_MS in env args:\n%s\n' "${joined}" >&2
        return 1
    }
    grep -qx 'BENCH_DIR=/data/local/tmp/bench' <<< "${joined}" || {
        printf 'missing BENCH_DIR in env args:\n%s\n' "${joined}" >&2
        return 1
    }
    grep -qx 'QNN_BIN=/data/local/tmp/qnn-bin' <<< "${joined}" || {
        printf 'missing QNN_BIN in env args:\n%s\n' "${joined}" >&2
        return 1
    }
    grep -qx 'SAMPLE_INTERVAL_S=1' <<< "${joined}" || {
        printf 'missing SAMPLE_INTERVAL_S in env args:\n%s\n' "${joined}" >&2
        return 1
    }
    grep -qx 'TASKSET_MASK=80' <<< "${joined}" || {
        printf 'missing TASKSET_MASK in env args:\n%s\n' "${joined}" >&2
        return 1
    }
    grep -qx 'LLAMA_BENCH_FAST_EXIT_VALUE=1' <<< "${joined}" || {
        printf 'missing LLAMA_BENCH_FAST_EXIT_VALUE in env args:\n%s\n' "${joined}" >&2
        return 1
    }
    grep -qx 'LLAMA_BENCH_QNN_PREWARM_DECODE=1' <<< "${joined}" || {
        printf 'missing LLAMA_BENCH_QNN_PREWARM_DECODE in env args:\n%s\n' "${joined}" >&2
        return 1
    }
}

test_power_cv_is_computed_from_active_power_and_stddev() {
    local cv
    cv="$(compute_power_cv_pct 2000 100)"
    [[ "${cv}" == "5.00" ]] || {
        printf 'unexpected power CV: %s\n' "${cv}" >&2
        return 1
    }

    local blank
    blank="$(compute_power_cv_pct 0 100)"
    [[ -z "${blank}" ]] || {
        printf 'expected blank power CV for zero active power, got: %s\n' "${blank}" >&2
        return 1
    }
}

test_data_quality_classification() {
    [[ "$(classify_data_quality ok 1 0.50)" == "smoke_only" ]] || {
        printf 'ROUNDS=1 should be smoke_only\n' >&2
        return 1
    }
    [[ "$(classify_data_quality unsupported_by_current_aot_cache_size 3 '')" == "unsupported" ]] || {
        printf 'unsupported rows should be unsupported data quality\n' >&2
        return 1
    }
    [[ "$(classify_data_quality failed 3 '')" == "failed" ]] || {
        printf 'failed rows should be failed data quality\n' >&2
        return 1
    }
    [[ "$(classify_data_quality ok 3 10.01)" == "unstable_power_window" ]] || {
        printf 'power CV > 10%% should be unstable_power_window\n' >&2
        return 1
    }
    [[ "$(classify_data_quality ok 3 10.00)" == "paper_ready" ]] || {
        printf 'power CV <= 10%% with >=3 rounds should be paper_ready\n' >&2
        return 1
    }
    [[ "$(ACTIVE_WINDOW_SAMPLES=4 classify_data_quality ok 3 2.00 3)" == "unstable_power_window" ]] || {
        printf 'short active windows should stay unstable even when CV is low\n' >&2
        return 1
    }
}

test_power_parser_uses_active_plateau_not_startup_window() {
    local samples
    samples="$(mktemp)"
    cat > "${samples}" <<'EOF'
1000,4000000,-100,24000,734000000
1003,4000000,-300,24000,734000000
1006,4000000,-320,24000,734000000
1009,4000000,-330,24000,734000000
1012,4000000,-340,24000,734000000
1015,4000000,-800,24100,734000000
1018,4000000,-790,24100,734000000
1021,4000000,-810,24100,734000000
1024,4000000,-805,24100,734000000
EOF

    local summary
    summary="$(ACTIVE_WINDOW_SAMPLES=4 summarize_power_samples "${samples}")"
    rm -f "${samples}"

    local active_power_mw power_std
    IFS=',' read -r active_power_mw power_std _ <<< "${summary}"
    [[ "${active_power_mw}" == "3205.00" ]] || {
        printf 'expected highest active plateau average 3205.00, got summary: %s\n' "${summary}" >&2
        return 1
    }
    [[ "${power_std}" == "34.16" ]] || {
        printf 'expected active plateau stddev 34.16, got summary: %s\n' "${summary}" >&2
        return 1
    }
}

test_power_parser_shortens_ramp_contaminated_active_window() {
    local samples
    samples="$(mktemp)"
    cat > "${samples}" <<'EOF'
1000,1000000,-400,24000,2649600
1003,1000000,-1200,24000,2649600
1006,1000000,-1300,24000,2649600
1009,1000000,-1500,24100,2649600
1012,1000000,-1700,24100,2649600
1015,1000000,-4200,24100,2649600
1018,1000000,-4300,24100,2649600
1021,1000000,-4250,24100,2649600
EOF

    local summary
    summary="$(ACTIVE_WINDOW_SAMPLES=4 ACTIVE_MIN_WINDOW_SAMPLES=2 ACTIVE_WINDOW_MAX_RANGE_PCT=10 summarize_power_samples "${samples}")"
    rm -f "${samples}"

    local active_power_mw power_std temperature_avg_c temp_max_c stable_range_pct avg_freq min_freq max_freq sample_count
    IFS=',' read -r active_power_mw power_std temperature_avg_c temp_max_c stable_range_pct avg_freq min_freq max_freq sample_count <<< "${summary}"
    [[ "${active_power_mw}" == "4250.00" ]] || {
        printf 'expected shortened plateau average 4250.00, got summary: %s\n' "${summary}" >&2
        return 1
    }
    [[ "${stable_range_pct}" == "2.35" ]] || {
        printf 'expected shortened plateau range 2.35, got summary: %s\n' "${summary}" >&2
        return 1
    }
    [[ "${sample_count}" == "3" ]] || {
        printf 'expected shortened plateau to use 3 samples, got summary: %s\n' "${summary}" >&2
        return 1
    }
}

test_power_parser_prefers_stable_full_window_before_shortening() {
    local samples
    samples="$(mktemp)"
    cat > "${samples}" <<'EOF'
1000,1000000,-400,24000,2649600
1003,1000000,-1200,24000,2649600
1006,1000000,-1300,24000,2649600
1009,1000000,-5000,24100,2649600
1012,1000000,-4200,24100,2649600
1015,1000000,-4150,24100,2649600
1018,1000000,-4250,24100,2649600
1021,1000000,-4200,24100,2649600
1024,1000000,-3900,24100,2649600
EOF

    local summary
    summary="$(ACTIVE_WINDOW_SAMPLES=4 ACTIVE_MIN_WINDOW_SAMPLES=2 ACTIVE_WINDOW_MAX_RANGE_PCT=10 summarize_power_samples "${samples}")"
    rm -f "${samples}"

    local active_power_mw power_std temperature_avg_c temp_max_c stable_range_pct avg_freq min_freq max_freq sample_count
    IFS=',' read -r active_power_mw power_std temperature_avg_c temp_max_c stable_range_pct avg_freq min_freq max_freq sample_count <<< "${summary}"
    [[ "${active_power_mw}" == "4200.00" ]] || {
        printf 'expected stable full-window average 4200.00, got summary: %s\n' "${summary}" >&2
        return 1
    }
    [[ "${sample_count}" == "4" ]] || {
        printf 'expected stable full-window to keep 4 samples, got summary: %s\n' "${summary}" >&2
        return 1
    }
}

test_profile_header_matches_required_schema() {
    local csv
    csv="$(mktemp)"

    ensure_profile_header "${csv}"

    local header
    header="$(sed -n '1p' "${csv}")"
    rm -f "${csv}"

    [[ "${header}" == "${PROFILE_HEADER}" ]] || {
        printf 'profile header does not match required schema\n%s\n' "${header}" >&2
        return 1
    }
}

main() {
    source_script
    test_llama_bench_tg_parser_handles_context_depth
    test_qnn_config_parser_selects_relevant_batch_sizes
    test_qnn_cache_guard_applies_default_margin
    test_fallback_detector_ignores_library_and_disabled_messages
    test_support_classifier_does_not_treat_generic_cache_logs_as_cache_guard_failures
    test_support_classifier_detects_explicit_cache_capacity_errors
    test_optional_display_env_is_recorded_and_forwarded
    test_power_cv_is_computed_from_active_power_and_stddev
    test_data_quality_classification
    test_power_parser_uses_active_plateau_not_startup_window
    test_power_parser_shortens_ramp_contaminated_active_window
    test_power_parser_prefers_stable_full_window_before_shortening
    test_profile_header_matches_required_schema
    printf 'Insight B context frontier helper tests ok\n'
}

main "$@"
