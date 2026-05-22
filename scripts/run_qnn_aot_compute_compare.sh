#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

log() {
    printf '[qnn-aot-compute-compare] %s\n' "$*"
}

die() {
    printf '[qnn-aot-compute-compare] ERROR: %s\n' "$*" >&2
    exit 1
}

require_env() {
    local name="$1"
    [[ -n "${!name:-}" ]] || die "${name} must be set"
}

adb_shell() {
    adb -s "${DEVICE}" shell "$@" | tr -d '\r'
}

adb_su_cat() {
    local path="$1"
    adb -s "${DEVICE}" shell "su -c 'cat ${path} 2>/dev/null || true'" | sed 's/[[:space:]]//g'
}

read_setting() {
    local namespace="$1"
    local key="$2"
    adb_shell "settings get ${namespace} ${key} 2>/dev/null || true" | tr -d '[:space:]'
}

restore_setting() {
    local namespace="$1"
    local key="$2"
    local value="$3"
    if [[ -z "${value}" || "${value}" == "null" ]]; then
        adb -s "${DEVICE}" shell "settings delete ${namespace} ${key}" >/dev/null 2>&1 || true
    else
        adb -s "${DEVICE}" shell "settings put ${namespace} ${key} ${value}" >/dev/null 2>&1 || true
    fi
}

save_display_state() {
    ORIG_SCREEN_OFF_TIMEOUT="$(read_setting system screen_off_timeout)"
    ORIG_SCREEN_BRIGHTNESS="$(read_setting system screen_brightness)"
    ORIG_SCREEN_BRIGHTNESS_MODE="$(read_setting system screen_brightness_mode)"
    ORIG_STAY_ON_WHILE_PLUGGED_IN="$(read_setting global stay_on_while_plugged_in)"

    {
        printf 'screen_off_timeout=%s\n' "${ORIG_SCREEN_OFF_TIMEOUT}"
        printf 'screen_brightness=%s\n' "${ORIG_SCREEN_BRIGHTNESS}"
        printf 'screen_brightness_mode=%s\n' "${ORIG_SCREEN_BRIGHTNESS_MODE}"
        printf 'stay_on_while_plugged_in=%s\n' "${ORIG_STAY_ON_WHILE_PLUGGED_IN}"
    } > "${OUTPUT_DIR}/display_state_before.txt"
}

restore_display_state() {
    restore_setting system screen_off_timeout "${ORIG_SCREEN_OFF_TIMEOUT:-}"
    restore_setting system screen_brightness "${ORIG_SCREEN_BRIGHTNESS:-}"
    restore_setting system screen_brightness_mode "${ORIG_SCREEN_BRIGHTNESS_MODE:-}"
    restore_setting global stay_on_while_plugged_in "${ORIG_STAY_ON_WHILE_PLUGGED_IN:-}"
}

ensure_screen_on() {
    adb -s "${DEVICE}" shell "input keyevent KEYCODE_WAKEUP" >/dev/null 2>&1 || true
    adb -s "${DEVICE}" shell "wm dismiss-keyguard" >/dev/null 2>&1 || true
    adb -s "${DEVICE}" shell "settings put system screen_off_timeout ${KEEP_SCREEN_ON_TIMEOUT_MS}" >/dev/null 2>&1 || true
    adb -s "${DEVICE}" shell "settings put global stay_on_while_plugged_in 7" >/dev/null 2>&1 || true
}

infer_temp_scale_to_mc() {
    local raw="$1"
    if [[ -z "${raw}" || ! "${raw}" =~ ^-?[0-9]+$ ]]; then
        printf '1000\n'
    elif (( raw >= 10000 || raw <= -10000 )); then
        printf '1\n'
    elif (( raw >= 100 || raw <= -100 )); then
        printf '100\n'
    else
        printf '1000\n'
    fi
}

temp_raw_to_c() {
    local raw="$1"
    if [[ -z "${raw}" || ! "${raw}" =~ ^-?[0-9]+$ ]]; then
        printf 'NA\n'
        return 0
    fi
    awk -v raw="${raw}" -v scale="${TEMP_SCALE_TO_MC}" 'BEGIN { printf "%.2f\n", raw * scale / 1000.0 }'
}

temp_c_now() {
    temp_raw_to_c "$(adb_su_cat "${TEMP_PATH}")"
}

wait_for_cooldown() {
    local start_ts
    start_ts="$(date +%s)"
    while true; do
        local temp_c
        temp_c="$(temp_c_now)"
        if [[ "${temp_c}" == "NA" ]] || awk -v t="${temp_c}" -v limit="${COOLDOWN_TEMP_C}" 'BEGIN { exit !(t <= limit) }'; then
            return 0
        fi
        if (( $(date +%s) - start_ts >= COOLDOWN_TIMEOUT_S )); then
            die "cooldown timeout: current temperature ${temp_c}C still exceeds ${COOLDOWN_TEMP_C}C"
        fi
        log "cooling down: ${temp_c}C > ${COOLDOWN_TEMP_C}C"
        sleep "${COOLDOWN_POLL_S}"
    done
}

sample_once() {
    local timestamp_ms voltage current temp_raw temp_mc temp_c power_index
    timestamp_ms="$(date +%s%3N)"
    voltage="$(adb_su_cat "${VOLTAGE_PATH}")"
    current="$(adb_su_cat "${CURRENT_PATH}")"
    temp_raw="$(adb_su_cat "${TEMP_PATH}")"

    [[ -n "${voltage}" ]] || voltage="NA"
    [[ -n "${current}" ]] || current="NA"
    [[ -n "${temp_raw}" ]] || temp_raw="NA"

    if [[ "${temp_raw}" =~ ^-?[0-9]+$ ]]; then
        temp_mc="$(awk -v raw="${temp_raw}" -v scale="${TEMP_SCALE_TO_MC}" 'BEGIN { printf "%.0f", raw * scale }')"
        temp_c="$(awk -v mc="${temp_mc}" 'BEGIN { printf "%.2f", mc / 1000.0 }')"
    else
        temp_mc="NA"
        temp_c="NA"
    fi

    if [[ "${voltage}" =~ ^-?[0-9]+$ && "${current}" =~ ^-?[0-9]+$ ]]; then
        power_index="$(awk -v v="${voltage}" -v i="${current}" 'BEGIN { if (i < 0) i = -i; printf "%.6f", v * i / 1000000000.0 }')"
    else
        power_index="NA"
    fi

    printf '%s,%s,%s,%s,%s,%s,%s\n' "${timestamp_ms}" "${voltage}" "${current}" "${power_index}" "${temp_raw}" "${temp_mc}" "${temp_c}"
}

sample_baseline() {
    local path="$1"
    printf 'timestamp_ms,voltage_uv,current_raw,power_index_mw,temp_raw,temp_mc,temp_c\n' > "${path}"
    local _
    for _ in $(seq 1 "${BASELINE_SAMPLES}"); do
        sample_once >> "${path}"
        sleep "${SAMPLE_INTERVAL_S}"
    done
}

sample_while_running() {
    local pid="$1"
    local sample_path="$2"
    local meta_path="$3"

    printf 'timestamp_ms,voltage_uv,current_raw,power_index_mw,temp_raw,temp_mc,temp_c\n' > "${sample_path}"
    : > "${meta_path}"

    while kill -0 "${pid}" >/dev/null 2>&1; do
        sample_once >> "${sample_path}"
        local temp_c
        temp_c="$(awk -F, 'END { print $7 }' "${sample_path}")"
        if [[ "${temp_c}" =~ ^-?[0-9]+([.][0-9]+)?$ ]] && awk -v t="${temp_c}" -v limit="${TEMP_LIMIT_C}" 'BEGIN { exit !(t >= limit) }'; then
            printf 'THERMAL_ABORT,%s,%s\n' "$(date -u +%FT%TZ)" "${temp_c}" >> "${meta_path}"
            kill "${pid}" >/dev/null 2>&1 || true
            adb -s "${DEVICE}" shell "pkill -INT llama-bench 2>/dev/null || true" >/dev/null 2>&1 || true
            return 0
        fi
        sleep "${SAMPLE_INTERVAL_S}"
    done

    printf 'SAMPLER_EXIT,%s\n' "$(date -u +%FT%TZ)" >> "${meta_path}"
}

make_remote_cmd() {
    local qnn_dir="$1"
    local context_tokens="$2"

    printf 'cd %s && env LD_LIBRARY_PATH=%s:%s:$LD_LIBRARY_PATH ADSP_LIBRARY_PATH=%s:%s GGML_HEXAGON_EXPERIMENTAL=1 GGML_QNN_HTP_WORKPOINT=%s GGML_QNN_AOT_CONFIG=%s/config.json GGML_QNN_AOT_MODEL_DIR=%s GGML_QNN_AOT_WRITE_GENERIC_KV=1 GGML_QNN_AOT_DISABLE_SEED_KV=1 GGML_HETERO_DYNAMIC_MODE=phase GGML_HETERO_DYNAMIC_PREFILL_ROUTE=qnn-npu GGML_HETERO_DYNAMIC_DECODE_ROUTE=qnn-npu GGML_HETERO_DYNAMIC_TRACE_TIMING=1 LLAMA_BENCH_FAST_EXIT=1 LLAMA_BENCH_QNN_PREWARM_DECODE=1 taskset %s ./llama-bench -v -m %s -ngl 99 -dev qnn-npu -t 1 -p 0 -n %s -d 0 -c %s -b 1 -ub 1 -r %s --no-warmup --mmap 0' \
        "${BENCH_DIR}" \
        "${BENCH_DIR}" \
        "${qnn_dir}" \
        "${BENCH_DIR}" \
        "${qnn_dir}" \
        "${QNN_WORKPOINT}" \
        "${qnn_dir}" \
        "${qnn_dir}" \
        "${TASKSET_MASK}" \
        "${MODEL_PATH}" \
        "${DECODE_TOKENS}" \
        "${context_tokens}" \
        "${ROUNDS}"
}

run_condition() {
    local condition="$1"
    local qnn_dir="$2"
    local artifact_context_size="$3"
    local artifact_cache_size="$4"
    local bench_context="$5"
    local notes="$6"

    local raw_dir="${OUTPUT_DIR}/raw/${condition}"
    local remote_cmd bench_log sample_log meta_log status_log bench_pid sampler_pid exit_code
    mkdir -p "${raw_dir}"

    remote_cmd="$(make_remote_cmd "${qnn_dir}" "${bench_context}")"
    printf 'adb -s %q shell %q\n' "${DEVICE}" "${remote_cmd}" > "${raw_dir}/command.sh"
    {
        printf 'condition=%s\n' "${condition}"
        printf 'artifact_context_size=%s\n' "${artifact_context_size}"
        printf 'artifact_cache_size=%s\n' "${artifact_cache_size}"
        printf 'bench_context=%s\n' "${bench_context}"
        printf 'decode_tokens=%s\n' "${DECODE_TOKENS}"
        printf 'rounds=%s\n' "${ROUNDS}"
        printf 'qnn_dir=%s\n' "${qnn_dir}"
        printf 'notes=%s\n' "${notes}"
    } > "${raw_dir}/meta.txt"

    bench_log="${raw_dir}/bench.log"
    sample_log="${raw_dir}/power_samples.csv"
    meta_log="${raw_dir}/sample_meta.txt"
    status_log="${raw_dir}/status.txt"

    wait_for_cooldown
    ensure_screen_on

    printf 'START,%s,%s\n' "$(date -u +%FT%TZ)" "${condition}" > "${status_log}"
    log "running ${condition}: bench_context=${bench_context} qnn_dir=${qnn_dir}"
    adb -s "${DEVICE}" shell "${remote_cmd}" > "${bench_log}" 2>&1 &
    bench_pid=$!

    sample_while_running "${bench_pid}" "${sample_log}" "${meta_log}" &
    sampler_pid=$!

    exit_code=0
    if wait "${bench_pid}"; then
        exit_code=0
    else
        exit_code=$?
    fi
    wait "${sampler_pid}" >/dev/null 2>&1 || true
    printf 'EXIT,%s,%s\n' "${exit_code}" "$(date -u +%FT%TZ)" >> "${status_log}"
}

write_metadata() {
    git rev-parse HEAD > "${OUTPUT_DIR}/git_commit.txt" 2>/dev/null || true
    git status --short > "${OUTPUT_DIR}/git_status.txt" 2>/dev/null || true
    if [[ -d "${ROOT_DIR}/ref/PowerServe/.git" ]]; then
        git -C "${ROOT_DIR}/ref/PowerServe" rev-parse HEAD > "${OUTPUT_DIR}/powerserve_git_commit.txt" 2>/dev/null || true
    fi
    adb -s "${DEVICE}" devices > "${OUTPUT_DIR}/device.txt"
    {
        printf 'DEVICE=%s\n' "${DEVICE}"
        printf 'BENCH_DIR=%s\n' "${BENCH_DIR}"
        printf 'MODEL_PATH=%s\n' "${MODEL_PATH}"
        printf 'QNN_2K=%s\n' "${QNN_2K}"
        printf 'QNN_4K=%s\n' "${QNN_4K}"
        printf 'DECODE_TOKENS=%s\n' "${DECODE_TOKENS}"
        printf 'ROUNDS=%s\n' "${ROUNDS}"
        printf 'TEMP_LIMIT_C=%s\n' "${TEMP_LIMIT_C}"
        printf 'COOLDOWN_TEMP_C=%s\n' "${COOLDOWN_TEMP_C}"
    } > "${OUTPUT_DIR}/paths.txt"
    {
        printf 'DEVICE=%q MODEL_PATH=%q BENCH_DIR=%q QNN_2K=%q QNN_4K=%q OUTPUT_DIR=%q DECODE_TOKENS=%q ROUNDS=%q TEMP_LIMIT_C=%q COOLDOWN_TEMP_C=%q bash scripts/run_qnn_aot_compute_compare.sh\n' \
            "${DEVICE}" "${MODEL_PATH}" "${BENCH_DIR}" "${QNN_2K}" "${QNN_4K}" "${OUTPUT_DIR}" "${DECODE_TOKENS}" "${ROUNDS}" "${TEMP_LIMIT_C}" "${COOLDOWN_TEMP_C}"
    } > "${OUTPUT_DIR}/command_line.txt"
}

discover_sampling_paths() {
    VOLTAGE_PATH="${VOLTAGE_PATH:-/sys/class/power_supply/battery/voltage_now}"
    CURRENT_PATH="${CURRENT_PATH:-/sys/class/power_supply/battery/current_now}"
    TEMP_PATH="${TEMP_PATH:-/sys/class/power_supply/battery/temp}"

    local temp_raw
    temp_raw="$(adb_su_cat "${TEMP_PATH}")"
    if [[ -z "${temp_raw}" || ! "${temp_raw}" =~ ^-?[0-9]+$ ]]; then
        TEMP_PATH="/sys/class/thermal/thermal_zone1/temp"
        temp_raw="$(adb_su_cat "${TEMP_PATH}")"
    fi

    TEMP_SCALE_TO_MC="$(infer_temp_scale_to_mc "${temp_raw}")"
    {
        printf 'VOLTAGE_PATH=%s\n' "${VOLTAGE_PATH}"
        printf 'CURRENT_PATH=%s\n' "${CURRENT_PATH}"
        printf 'TEMP_PATH=%s\n' "${TEMP_PATH}"
        printf 'TEMP_SCALE_TO_MC=%s\n' "${TEMP_SCALE_TO_MC}"
        printf 'INITIAL_TEMP_RAW=%s\n' "${temp_raw}"
        printf 'INITIAL_TEMP_C=%s\n' "$(temp_raw_to_c "${temp_raw}")"
    } > "${OUTPUT_DIR}/sampling_paths.txt"
}

main() {
    require_env DEVICE
    require_env MODEL_PATH
    require_env BENCH_DIR
    require_env QNN_2K
    require_env QNN_4K

    OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/results/insightB/qnn-aot-compute-compare-$(date -u +%Y%m%d-%H%M%S)}"
    DECODE_TOKENS="${DECODE_TOKENS:-1024}"
    ROUNDS="${ROUNDS:-3}"
    TEMP_LIMIT_C="${TEMP_LIMIT_C:-38.0}"
    COOLDOWN_TEMP_C="${COOLDOWN_TEMP_C:-37.0}"
    COOLDOWN_TIMEOUT_S="${COOLDOWN_TIMEOUT_S:-900}"
    COOLDOWN_POLL_S="${COOLDOWN_POLL_S:-5}"
    SAMPLE_INTERVAL_S="${SAMPLE_INTERVAL_S:-1}"
    BASELINE_SAMPLES="${BASELINE_SAMPLES:-12}"
    KEEP_SCREEN_ON_TIMEOUT_MS="${KEEP_SCREEN_ON_TIMEOUT_MS:-1800000}"
    QNN_WORKPOINT="${QNN_WORKPOINT:-burst}"
    TASKSET_MASK="${TASKSET_MASK:-80}"

    mkdir -p "${OUTPUT_DIR}/raw"
    printf '%s\n' "${OUTPUT_DIR}" > "${OUTPUT_DIR}/output_dir.txt"

    write_metadata
    save_display_state
    trap restore_display_state EXIT
    ensure_screen_on
    discover_sampling_paths

    log "output_dir=${OUTPUT_DIR}"
    local initial_temp_c
    initial_temp_c="$(awk -F= '$1 == "INITIAL_TEMP_C" { print $2 }' "${OUTPUT_DIR}/sampling_paths.txt")"
    log "initial_temp=${initial_temp_c}C"
    log "sampling baseline"
    wait_for_cooldown
    sample_baseline "${OUTPUT_DIR}/baseline_power_samples.csv"

    run_condition "2k_ctx2048" "${QNN_2K}" "2048" "1920" "2048" "2k_reference_graph_tg1024_r3"
    run_condition "4k_ctx2048" "${QNN_4K}" "4096" "3968" "2048" "4k_qnn231_same_bench_context_as_2k"
    run_condition "4k_ctx4096" "${QNN_4K}" "4096" "3968" "4096" "4k_qnn231_4096_bench_context_repeat"

    log "raw runs complete"
}

main "$@"
