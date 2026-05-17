#!/usr/bin/env bash

set -euo pipefail

DEVICE="${DEVICE:-}"
MODEL_PATH="${MODEL_PATH:-${MODEL:-}}"
QNN_DIR="${QNN_DIR:-}"
QNN_BIN="${QNN_BIN:-/data/local/tmp/acom-qnn-phase-materializer/bin}"
WORKPOINT_LIST="${WORKPOINT_LIST:-burst high_performance balanced low_balanced high_power_saver power_saver low_power_saver extreme_power_saver}"

TASKSET_MASK="${TASKSET_MASK:-80}"
LLAMA_THREADS="${LLAMA_THREADS:-1}"
NGL="${NGL:-99}"
PROMPT_TOKENS="${PROMPT_TOKENS:-0}"
GEN_TOKENS="${GEN_TOKENS:-128}"
CONTEXT_TOKENS="${CONTEXT_TOKENS:-2048}"
BATCH_TOKENS="${BATCH_TOKENS:-1}"
UBATCH_TOKENS="${UBATCH_TOKENS:-1}"
DEPTH_TOKENS="${DEPTH_TOKENS:-0}"
BENCH_REPEATS="${BENCH_REPEATS:-3}"
MMAP="${MMAP:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/out/npu-workpoint-sweep-$(date -u +%Y%m%d-%H%M%S)}"

TEMP_LIMIT_C="${TEMP_LIMIT_C:-38.0}"
COOLDOWN_TEMP_C="${COOLDOWN_TEMP_C:-37.0}"
COOLDOWN_TIMEOUT_S="${COOLDOWN_TIMEOUT_S:-900}"
COOLDOWN_POLL_S="${COOLDOWN_POLL_S:-5}"

SAMPLE_INTERVAL_S="${SAMPLE_INTERVAL_S:-1}"
POWER_SETTLE_SECONDS="${POWER_SETTLE_SECONDS:-8}"
STABLE_WINDOW_SAMPLES="${STABLE_WINDOW_SAMPLES:-8}"
STABLE_RANGE_PCT="${STABLE_RANGE_PCT:-4.0}"

KEEP_SCREEN_ON_TIMEOUT_MS="${KEEP_SCREEN_ON_TIMEOUT_MS:-1800000}"
SCREEN_BRIGHTNESS_OVERRIDE="${SCREEN_BRIGHTNESS_OVERRIDE:-}"

QNN_AOT_CONFIG="${QNN_AOT_CONFIG:-${GGML_QNN_AOT_CONFIG:-${QNN_DIR:+${QNN_DIR}/config.json}}}"
QNN_AOT_MODEL_DIR="${QNN_AOT_MODEL_DIR:-${GGML_QNN_AOT_MODEL_DIR:-${QNN_DIR}}}"
QNN_AOT_WRITE_GENERIC_KV="${QNN_AOT_WRITE_GENERIC_KV:-1}"
QNN_AOT_DISABLE_SEED_KV="${QNN_AOT_DISABLE_SEED_KV:-1}"
QNN_ENABLE_HEXAGON="${QNN_ENABLE_HEXAGON:-1}"
LLAMA_BENCH_FAST_EXIT_VALUE="${LLAMA_BENCH_FAST_EXIT_VALUE:-1}"
LLAMA_BENCH_QNN_PREWARM_DECODE="${LLAMA_BENCH_QNN_PREWARM_DECODE:-0}"

NPU_SWEEP_SUMMARIZE_ONLY="${NPU_SWEEP_SUMMARIZE_ONLY:-0}"
NPU_SWEEP_SAMPLE_LOG="${NPU_SWEEP_SAMPLE_LOG:-}"

declare -a WORKPOINTS=()

RESULTS_CSV=""
BASELINE_CSV=""
BASELINE_AVG_POWER_MW=""
BASELINE_AVG_TEMP_C=""
TEMP_SCALE_TO_MC=""
TEMP_LIMIT_MC=""
COOLDOWN_TEMP_MC=""

BATTERY_VOLTAGE_PATH="${BATTERY_VOLTAGE_PATH:-}"
BATTERY_CURRENT_PATH="${BATTERY_CURRENT_PATH:-}"
TEMP_PATH="${TEMP_PATH:-}"

ORIG_SCREEN_OFF_TIMEOUT=""
ORIG_SCREEN_BRIGHTNESS=""
ORIG_SCREEN_BRIGHTNESS_MODE=""
ORIG_STAY_ON_WHILE_PLUGGED_IN=""

LOCAL_BENCH_PID=""

log() {
    printf '[npu-workpoint-sweep] %s\n' "$*"
}

die() {
    printf '[npu-workpoint-sweep] ERROR: %s\n' "$*" >&2
    exit 1
}

adb_shell() {
    adb -s "${DEVICE}" shell "$@" | tr -d '\r'
}

adb_shell_raw() {
    adb -s "${DEVICE}" shell "$@"
}

adb_test() {
    adb -s "${DEVICE}" shell "su -c '$1'" >/dev/null 2>&1
}

adb_root_shell() {
    adb -s "${DEVICE}" shell "su -c '$1'" >/dev/null 2>&1
}

adb_root_capture() {
    adb -s "${DEVICE}" shell "su -c '$1'" | tr -d '\r'
}

read_android_setting() {
    local namespace="$1"
    local key="$2"
    adb_shell "settings get ${namespace} ${key} 2>/dev/null || true" | tr -d '[:space:]'
}

restore_android_setting() {
    local namespace="$1"
    local key="$2"
    local value="$3"

    if [[ -z "${value}" || "${value}" == "null" ]]; then
        adb -s "${DEVICE}" shell "settings delete ${namespace} ${key}" >/dev/null 2>&1 || true
    else
        adb -s "${DEVICE}" shell "settings put ${namespace} ${key} ${value}" >/dev/null 2>&1 || true
    fi
}

read_remote_value() {
    local path="$1"
    adb_root_capture "cat ${path} 2>/dev/null || true" | tr -d '[:space:]'
}

find_first_readable_path() {
    local candidate
    for candidate in "$@"; do
        if [[ -z "${candidate}" ]]; then
            continue
        fi
        if adb_test "[ -r ${candidate} ]"; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    return 1
}

float_to_mc() {
    awk -v x="$1" 'BEGIN { printf "%.0f\n", x * 1000.0 }'
}

format_mc() {
    awk -v x="$1" 'BEGIN { printf "%.2f", x / 1000.0 }'
}

infer_temp_scale_to_mc() {
    local raw="$1"
    if [[ -z "${raw}" || ! "${raw}" =~ ^-?[0-9]+$ ]]; then
        die "invalid temperature sample while inferring scale: ${raw}"
    fi

    if (( raw >= 10000 || raw <= -10000 )); then
        printf '1\n'
    elif (( raw >= 100 || raw <= -100 )); then
        printf '100\n'
    else
        printf '1000\n'
    fi
}

normalize_temp_to_mc() {
    local raw="$1"
    local scale="$2"
    if [[ -z "${raw}" || ! "${raw}" =~ ^-?[0-9]+$ ]]; then
        die "invalid temperature sample: ${raw}"
    fi

    case "${scale}" in
        1)
            printf '%s\n' "${raw}"
            ;;
        100)
            printf '%s\n' "$(( raw * 100 ))"
            ;;
        1000)
            printf '%s\n' "$(( raw * 1000 ))"
            ;;
        *)
            die "unsupported temperature scale: ${scale}"
            ;;
    esac
}

check_device_online() {
    adb -s "${DEVICE}" get-state >/dev/null 2>&1 || die "device ${DEVICE} is offline"
}

require_runtime_inputs() {
    [[ -n "${DEVICE}" ]] || die "DEVICE must be set"
    [[ -n "${MODEL_PATH}" ]] || die "MODEL_PATH must be set"
    [[ -n "${QNN_AOT_CONFIG}" ]] || die "QNN_AOT_CONFIG or QNN_DIR must be set"
    [[ -n "${QNN_AOT_MODEL_DIR}" ]] || die "QNN_AOT_MODEL_DIR or QNN_DIR must be set"
}

save_display_state() {
    ORIG_SCREEN_OFF_TIMEOUT="$(read_android_setting system screen_off_timeout)"
    ORIG_SCREEN_BRIGHTNESS="$(read_android_setting system screen_brightness)"
    ORIG_SCREEN_BRIGHTNESS_MODE="$(read_android_setting system screen_brightness_mode)"
    ORIG_STAY_ON_WHILE_PLUGGED_IN="$(read_android_setting global stay_on_while_plugged_in)"
}

ensure_screen_on() {
    adb -s "${DEVICE}" shell "input keyevent KEYCODE_WAKEUP" >/dev/null 2>&1 || true
    adb -s "${DEVICE}" shell "wm dismiss-keyguard" >/dev/null 2>&1 || true
    adb -s "${DEVICE}" shell "settings put system screen_off_timeout ${KEEP_SCREEN_ON_TIMEOUT_MS}" >/dev/null 2>&1 || true
    adb -s "${DEVICE}" shell "settings put global stay_on_while_plugged_in 7" >/dev/null 2>&1 || true

    if [[ -n "${SCREEN_BRIGHTNESS_OVERRIDE}" ]]; then
        adb -s "${DEVICE}" shell "settings put system screen_brightness_mode 0" >/dev/null 2>&1 || true
        adb -s "${DEVICE}" shell "settings put system screen_brightness ${SCREEN_BRIGHTNESS_OVERRIDE}" >/dev/null 2>&1 || true
    fi
}

verify_screen_on() {
    adb -s "${DEVICE}" shell "dumpsys display 2>/dev/null | grep -Eq 'mScreenState=ON|state ON, committedState ON'" >/dev/null 2>&1
}

restore_display_state() {
    restore_android_setting system screen_off_timeout "${ORIG_SCREEN_OFF_TIMEOUT}"
    restore_android_setting system screen_brightness "${ORIG_SCREEN_BRIGHTNESS}"
    restore_android_setting system screen_brightness_mode "${ORIG_SCREEN_BRIGHTNESS_MODE}"
    restore_android_setting global stay_on_while_plugged_in "${ORIG_STAY_ON_WHILE_PLUGGED_IN}"
}

discover_sysfs_paths() {
    BATTERY_VOLTAGE_PATH="$(find_first_readable_path \
        "${BATTERY_VOLTAGE_PATH}" \
        "/sys/class/power_supply/battery/voltage_now")" || true

    BATTERY_CURRENT_PATH="$(find_first_readable_path \
        "${BATTERY_CURRENT_PATH}" \
        "/sys/class/power_supply/battery/current_now")" || true

    TEMP_PATH="$(find_first_readable_path \
        "${TEMP_PATH}" \
        "/sys/class/power_supply/battery/temp" \
        "/sys/class/thermal/thermal_zone0/temp" \
        "/sys/class/thermal/thermal_zone1/temp")" || true

    [[ -n "${BATTERY_VOLTAGE_PATH}" ]] || die "failed to discover battery voltage path"
    [[ -n "${BATTERY_CURRENT_PATH}" ]] || die "failed to discover battery current path"
    [[ -n "${TEMP_PATH}" ]] || die "failed to discover device temperature path"

    TEMP_SCALE_TO_MC="$(infer_temp_scale_to_mc "$(read_remote_value "${TEMP_PATH}")")"
    TEMP_LIMIT_MC="$(float_to_mc "${TEMP_LIMIT_C}")"
    COOLDOWN_TEMP_MC="$(float_to_mc "${COOLDOWN_TEMP_C}")"

    log "temp path: ${TEMP_PATH} (scale=${TEMP_SCALE_TO_MC})"
    log "power paths: V=${BATTERY_VOLTAGE_PATH} I=${BATTERY_CURRENT_PATH}"
}

discover_workpoint_list() {
    mapfile -t WORKPOINTS < <(printf '%s\n' "${WORKPOINT_LIST}" | tr ', ' '\n\n' | awk 'NF')
    (( ${#WORKPOINTS[@]} > 0 )) || die "no workpoints found in WORKPOINT_LIST"
}

read_temp_mc_now() {
    normalize_temp_to_mc "$(read_remote_value "${TEMP_PATH}")" "${TEMP_SCALE_TO_MC}"
}

wait_for_cooldown() {
    local start_ts
    start_ts="$(date +%s)"
    while true; do
        local temp_mc
        temp_mc="$(read_temp_mc_now)"
        if (( temp_mc <= COOLDOWN_TEMP_MC )); then
            return 0
        fi
        if (( $(date +%s) - start_ts >= COOLDOWN_TIMEOUT_S )); then
            die "cooldown timeout: current temperature $(format_mc "${temp_mc}")C still exceeds ${COOLDOWN_TEMP_C}C"
        fi
        log "cooling down: $(format_mc "${temp_mc}")C > ${COOLDOWN_TEMP_C}C"
        sleep "${COOLDOWN_POLL_S}"
    done
}

sample_baseline() {
    local baseline_csv="$1"
    : > "${baseline_csv}"

    local _
    for _ in $(seq 1 12); do
        local ts
        local voltage
        local current
        local temp_raw
        local temp_mc

        ts="$(date +%s)"
        voltage="$(read_remote_value "${BATTERY_VOLTAGE_PATH}")"
        current="$(read_remote_value "${BATTERY_CURRENT_PATH}")"
        temp_raw="$(read_remote_value "${TEMP_PATH}")"
        temp_mc="$(normalize_temp_to_mc "${temp_raw}" "${TEMP_SCALE_TO_MC}")"

        printf '%s,%s,%s,%s\n' "${ts}" "${voltage}" "${current}" "${temp_mc}" >> "${baseline_csv}"
        sleep 1
    done
}

summarize_baseline() {
    local baseline_csv="$1"
    awk -F, '
        function abs(x) { return x < 0 ? -x : x }
        {
            power = (($2 + 0) * abs($3 + 0)) / 1000000.0
            sum_power += power
            sum_temp += $4 + 0
            n++
        }
        END {
            if (n == 0) {
                print "NA,NA"
                exit 0
            }
            printf "%.2f,%.2f\n", sum_power / n, (sum_temp / n) / 1000.0
        }' "${baseline_csv}"
}

thermal_abort_happened() {
    local meta_file="$1"
    grep -q '^THERMAL_ABORT,' "${meta_file}" 2>/dev/null
}

run_bench_with_local_sampling() {
    local run_name="$1"
    local workpoint="$2"
    local bench_log="$3"
    local sample_log="$4"
    local meta_log="$5"

    : > "${sample_log}"
    : > "${meta_log}"
    adb -s "${DEVICE}" shell "cd ${QNN_BIN} && \
export LD_LIBRARY_PATH=${QNN_BIN} && \
export ADSP_LIBRARY_PATH=${QNN_BIN} && \
export GGML_HEXAGON_EXPERIMENTAL=${QNN_ENABLE_HEXAGON} && \
export GGML_QNN_HTP_WORKPOINT=${workpoint} && \
export GGML_QNN_AOT_CONFIG=${QNN_AOT_CONFIG} && \
export GGML_QNN_AOT_MODEL_DIR=${QNN_AOT_MODEL_DIR} && \
export GGML_QNN_AOT_WRITE_GENERIC_KV=${QNN_AOT_WRITE_GENERIC_KV} && \
export GGML_QNN_AOT_DISABLE_SEED_KV=${QNN_AOT_DISABLE_SEED_KV} && \
export LLAMA_BENCH_FAST_EXIT=${LLAMA_BENCH_FAST_EXIT_VALUE} && \
export LLAMA_BENCH_QNN_PREWARM_DECODE=${LLAMA_BENCH_QNN_PREWARM_DECODE} && \
taskset ${TASKSET_MASK} ./llama-bench -v \
  -m ${MODEL_PATH} \
  -ngl ${NGL} -dev qnn-npu -t ${LLAMA_THREADS} \
  -p ${PROMPT_TOKENS} -n ${GEN_TOKENS} \
  -d ${DEPTH_TOKENS} \
  -c ${CONTEXT_TOKENS} -b ${BATCH_TOKENS} -ub ${UBATCH_TOKENS} \
  -r ${BENCH_REPEATS} \
  --no-warmup --mmap ${MMAP}" > "${bench_log}" 2>&1 &
    LOCAL_BENCH_PID=$!

    local status="ok"
    while kill -0 "${LOCAL_BENCH_PID}" >/dev/null 2>&1; do
        local ts
        local voltage
        local current
        local temp_raw
        local temp_mc

        ts="$(date +%s)"
        voltage="$(read_remote_value "${BATTERY_VOLTAGE_PATH}")"
        current="$(read_remote_value "${BATTERY_CURRENT_PATH}")"
        temp_raw="$(read_remote_value "${TEMP_PATH}")"
        temp_mc="$(normalize_temp_to_mc "${temp_raw}" "${TEMP_SCALE_TO_MC}")"

        printf '%s,%s,%s,%s\n' "${ts}" "${voltage}" "${current}" "${temp_mc}" >> "${sample_log}"

        if (( temp_mc >= TEMP_LIMIT_MC )); then
            printf 'THERMAL_ABORT,%s,%s\n' "${ts}" "${temp_mc}" > "${meta_log}"
            kill "${LOCAL_BENCH_PID}" >/dev/null 2>&1 || true
            status="thermal_abort"
            break
        fi

        sleep "${SAMPLE_INTERVAL_S}"
    done

    local bench_exit_code=0
    if wait "${LOCAL_BENCH_PID}"; then
        bench_exit_code=0
    else
        bench_exit_code=$?
    fi

    if [[ ! -s "${meta_log}" ]]; then
        printf 'SAMPLER_EXIT,%s\n' "$(date +%s)" >> "${meta_log}"
    fi

    if thermal_abort_happened "${meta_log}"; then
        status="thermal_abort"
    fi

    LOCAL_BENCH_PID=""
    printf '%s,%s\n' "${bench_exit_code}" "${status}"
}

extract_throughput() {
    local bench_log="$1"
    local throughput
    local label
    local table_line

    table_line="$(awk -F'|' '/\|[[:space:]]*(pp|tg)[0-9]+[[:space:]]*\|/ { line=$0 } END { print line }' "${bench_log}")"
    if [[ -n "${table_line}" ]]; then
        label="$(awk -F'|' '/\|[[:space:]]*(pp|tg)[0-9]+[[:space:]]*\|/ { label=$(NF-2) } END { gsub(/^[[:space:]]+|[[:space:]]+$/, "", label); print label }' "${bench_log}")"
        throughput="$(awk -F'|' '/\|[[:space:]]*(pp|tg)[0-9]+[[:space:]]*\|/ { value=$(NF-1) } END { gsub(/^[[:space:]]+|[[:space:]]+$/, "", value); split(value, parts, / ± /); print parts[1] }' "${bench_log}")"
    else
        throughput="$(grep -Eo '[0-9]+([.][0-9]+)?[[:space:]]*(tok/s|t/s)' "${bench_log}" | tail -n 1 | awk '{ print $1 }' || true)"
        label="$(grep -Eo '\b(pp|tg)[0-9]+\b' "${bench_log}" | tail -n 1 || true)"
    fi

    [[ -n "${throughput}" ]] || throughput="NA"
    [[ -n "${label}" ]] || label="NA"
    printf '%s,%s\n' "${throughput}" "${label}"
}

summarize_samples() {
    local sample_log="$1"
    if [[ ! -s "${sample_log}" ]]; then
        printf 'NA,NA,NA,NA,NA,0,0,0,NA\n'
        return 0
    fi

    awk -F, \
        -v settle_s="${POWER_SETTLE_SECONDS}" \
        -v stable_n="${STABLE_WINDOW_SAMPLES}" \
        -v stable_pct="${STABLE_RANGE_PCT}" '
        function abs(x) { return x < 0 ? -x : x }
        NR == 1 {
            t0 = $1 + 0
        }
        {
            rel = ($1 + 0) - t0
            if (rel < settle_s) {
                next
            }

            n++
            power[n] = (($2 + 0) * abs($3 + 0)) / 1000000.0
            temp[n] = $4 + 0
        }
        END {
            if (n == 0) {
                print "NA,NA,NA,NA,NA,0,0,0,NA"
                exit 0
            }

            stable_start = 1
            stable_end = n
            stable_range = "NA"
            found_any_window = 0
            found_within_threshold = 0
            best_any_range = -1
            best_any_start = 1
            best_any_end = n
            if (stable_n > 1 && n >= stable_n) {
                for (i = 1; i <= n - stable_n + 1; ++i) {
                    min = power[i]
                    max = power[i]
                    sum = 0
                    for (j = i; j < i + stable_n; ++j) {
                        if (power[j] < min) {
                            min = power[j]
                        }
                        if (power[j] > max) {
                            max = power[j]
                        }
                        sum += power[j]
                    }
                    avg = sum / stable_n
                    range_pct = avg > 0 ? ((max - min) / avg * 100.0) : 0

                    if (!found_any_window || range_pct < best_any_range) {
                        found_any_window = 1
                        best_any_range = range_pct
                        best_any_start = i
                        best_any_end = i + stable_n - 1
                    }

                    if (avg > 0 && range_pct <= stable_pct) {
                        if (!found_within_threshold || range_pct < stable_range) {
                            found_within_threshold = 1
                            stable_start = i
                            stable_end = i + stable_n - 1
                            stable_range = range_pct
                        }
                    }
                }
            }

            if (!found_within_threshold && found_any_window) {
                stable_start = best_any_start
                stable_end = best_any_end
                stable_range = best_any_range
            }

            sum_power = 0
            sum_temp = 0
            max_temp = temp[stable_start]
            count = 0
            for (i = stable_start; i <= stable_end; ++i) {
                sum_power += power[i]
                sum_temp += temp[i]
                if (temp[i] > max_temp) {
                    max_temp = temp[i]
                }
                count++
            }

            printf "%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d,%.2f\n", \
                sum_power / count, \
                (sum_temp / count) / 1000.0, \
                max_temp / 1000.0, \
                temp[stable_start] / 1000.0, \
                temp[stable_end] / 1000.0, \
                count, \
                stable_start, \
                stable_end, \
                stable_range
        }' "${sample_log}"
}

compute_delta_vs_baseline() {
    local avg_power_mw="$1"
    if [[ "${avg_power_mw}" == "NA" || -z "${BASELINE_AVG_POWER_MW}" || "${BASELINE_AVG_POWER_MW}" == "NA" ]]; then
        printf 'NA\n'
        return 0
    fi

    awk -v power="${avg_power_mw}" -v base="${BASELINE_AVG_POWER_MW}" 'BEGIN { printf "%.2f\n", power - base }'
}

append_result() {
    local workpoint="$1"
    local status="$2"
    local exit_code="$3"
    local avg_power_mw="$4"
    local delta_vs_baseline_mw="$5"
    local avg_temp_c="$6"
    local max_temp_c="$7"
    local start_temp_c="$8"
    local end_temp_c="$9"
    local throughput_tok_s="${10}"
    local throughput_label="${11}"
    local sample_count="${12}"
    local stable_start_index="${13}"
    local stable_end_index="${14}"
    local stable_range_pct="${15}"
    local bench_log="${16}"
    local sample_log="${17}"

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${workpoint}" \
        "${status}" \
        "${exit_code}" \
        "${avg_power_mw}" \
        "${delta_vs_baseline_mw}" \
        "${avg_temp_c}" \
        "${max_temp_c}" \
        "${start_temp_c}" \
        "${end_temp_c}" \
        "${throughput_tok_s}" \
        "${throughput_label}" \
        "${sample_count}" \
        "${stable_start_index}" \
        "${stable_end_index}" \
        "${stable_range_pct}" \
        "${bench_log}" \
        "${sample_log}" >> "${RESULTS_CSV}"
}

cleanup() {
    if [[ -n "${LOCAL_BENCH_PID}" && "${LOCAL_BENCH_PID}" =~ ^[0-9]+$ ]]; then
        kill -TERM "${LOCAL_BENCH_PID}" >/dev/null 2>&1 || true
    fi
    restore_display_state
}

main() {
    mkdir -p "${OUTPUT_DIR}"
    RESULTS_CSV="${OUTPUT_DIR}/results.csv"
    BASELINE_CSV="${OUTPUT_DIR}/baseline.samples.csv"

    printf '%s\n' 'workpoint,status,bench_exit_code,avg_power_mw,delta_vs_baseline_mw,avg_temp_c,max_temp_c,start_temp_c,end_temp_c,throughput_tok_s,throughput_label,sample_count,stable_start_index,stable_end_index,stable_range_pct,bench_log,sample_log' > "${RESULTS_CSV}"

    require_runtime_inputs
    check_device_online
    save_display_state
    ensure_screen_on
    verify_screen_on || die "failed to keep the device screen ON before the sweep starts"
    discover_sysfs_paths
    discover_workpoint_list

    log "run build-npu-opencl.sh before formal measurements when you need binary/config consistency with the current experiment setup"
    log "output dir: ${OUTPUT_DIR}"
    log "workpoints: ${WORKPOINTS[*]}"
    log "decode bench: -ngl ${NGL} -p ${PROMPT_TOKENS} -n ${GEN_TOKENS} -d ${DEPTH_TOKENS} -c ${CONTEXT_TOKENS} -b ${BATCH_TOKENS} -ub ${UBATCH_TOKENS} -r ${BENCH_REPEATS}"
    log "temperature limit: ${TEMP_LIMIT_C}C"

    log "waiting for cooldown before baseline sampling"
    wait_for_cooldown
    ensure_screen_on
    verify_screen_on || die "screen turned OFF before baseline sampling"

    log "sampling baseline idle power"
    sample_baseline "${BASELINE_CSV}"
    IFS=',' read -r BASELINE_AVG_POWER_MW BASELINE_AVG_TEMP_C <<< "$(summarize_baseline "${BASELINE_CSV}")"
    log "baseline: avg_power_mw=${BASELINE_AVG_POWER_MW} avg_temp_c=${BASELINE_AVG_TEMP_C} sample_log=${BASELINE_CSV}"

    local workpoint
    for workpoint in "${WORKPOINTS[@]}"; do
        local run_name
        local bench_log
        local sample_log
        local meta_log
        local run_result
        local bench_exit_code
        local status
        local sample_summary
        local throughput_summary
        local avg_power_mw
        local delta_vs_baseline_mw
        local avg_temp_c
        local max_temp_c
        local start_temp_c
        local end_temp_c
        local sample_count
        local stable_start_index
        local stable_end_index
        local stable_range_pct
        local throughput_tok_s
        local throughput_label

        run_name="npu_${workpoint}"
        bench_log="${OUTPUT_DIR}/${run_name}.bench.log"
        sample_log="${OUTPUT_DIR}/${run_name}.samples.csv"
        meta_log="${OUTPUT_DIR}/${run_name}.meta"

        log "waiting for cooldown before ${run_name}"
        wait_for_cooldown

        ensure_screen_on
        verify_screen_on || die "screen turned OFF before ${run_name}"

        log "starting benchmark for ${run_name}"
        run_result="$(run_bench_with_local_sampling "${run_name}" "${workpoint}" "${bench_log}" "${sample_log}" "${meta_log}")"
        IFS=',' read -r bench_exit_code status <<< "${run_result}"

        if [[ "${status}" == "ok" && "${bench_exit_code}" != "0" ]]; then
            status="bench_failed"
        fi

        sample_summary="$(summarize_samples "${sample_log}")"
        IFS=',' read -r avg_power_mw avg_temp_c max_temp_c start_temp_c end_temp_c sample_count stable_start_index stable_end_index stable_range_pct <<< "${sample_summary}"

        throughput_summary="$(extract_throughput "${bench_log}")"
        IFS=',' read -r throughput_tok_s throughput_label <<< "${throughput_summary}"

        delta_vs_baseline_mw="$(compute_delta_vs_baseline "${avg_power_mw}")"

        append_result \
            "${workpoint}" \
            "${status}" \
            "${bench_exit_code}" \
            "${avg_power_mw}" \
            "${delta_vs_baseline_mw}" \
            "${avg_temp_c}" \
            "${max_temp_c}" \
            "${start_temp_c}" \
            "${end_temp_c}" \
            "${throughput_tok_s}" \
            "${throughput_label}" \
            "${sample_count}" \
            "${stable_start_index}" \
            "${stable_end_index}" \
            "${stable_range_pct}" \
            "${bench_log}" \
            "${sample_log}"

        log "finished ${run_name}: status=${status} avg_power_mw=${avg_power_mw} delta_vs_baseline_mw=${delta_vs_baseline_mw} stable_window=${stable_start_index}-${stable_end_index} stable_range_pct=${stable_range_pct} throughput=${throughput_tok_s} label=${throughput_label} max_temp=${max_temp_c}C"
    done

    log "results written to ${RESULTS_CSV}"
}

if [[ "${NPU_SWEEP_SUMMARIZE_ONLY}" == "1" ]]; then
    [[ -n "${NPU_SWEEP_SAMPLE_LOG}" ]] || die "NPU_SWEEP_SAMPLE_LOG is required when NPU_SWEEP_SUMMARIZE_ONLY=1"
    summarize_samples "${NPU_SWEEP_SAMPLE_LOG}"
    exit 0
fi

trap cleanup EXIT
main "$@"
