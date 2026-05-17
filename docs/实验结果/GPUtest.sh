#!/usr/bin/env bash

set -euo pipefail

DEVICE="${DEVICE:-}"
MODEL_PATH="${MODEL_PATH:-${MODEL:-}}"
QNN_DIR="${QNN_DIR:-}"
QNN_BIN="${QNN_BIN:-/data/local/tmp/acom-qnn-phase-materializer/bin}"
HTP_WORKPOINT="${HTP_WORKPOINT:-burst}"

LLAMA_DEV="${LLAMA_DEV:-GPUOpenCL}"
TASKSET_MASK="${TASKSET_MASK:-80}"
LLAMA_THREADS="${LLAMA_THREADS:-1}"
NGL="${NGL:-99}"
PROMPT_TOKENS="${PROMPT_TOKENS:-32}"
GEN_TOKENS="${GEN_TOKENS:-0}"
CONTEXT_TOKENS="${CONTEXT_TOKENS:-1024}"
DEPTH_TOKENS="${DEPTH_TOKENS:-0}"
BENCH_REPEATS="${BENCH_REPEATS:-100}"
MMAP="${MMAP:-0}"

OUTPUT_DIR="${OUTPUT_DIR:-docs/out/gpu-freq-sweep-$(date -u +%Y%m%d-%H%M%S)}"
REMOTE_WORKDIR="${REMOTE_WORKDIR:-/data/local/tmp/gpu_freq_sweep}"
REMOTE_SAMPLER_PATH="${REMOTE_SAMPLER_PATH:-${REMOTE_WORKDIR}/sampler.sh}"

GPU_FREQ_LIST="${GPU_FREQ_LIST:-}"
TEMP_LIMIT_C="${TEMP_LIMIT_C:-38.0}"
COOLDOWN_TEMP_C="${COOLDOWN_TEMP_C:-37.0}"
COOLDOWN_TIMEOUT_S="${COOLDOWN_TIMEOUT_S:-900}"
COOLDOWN_POLL_S="${COOLDOWN_POLL_S:-5}"

SAMPLE_INTERVAL_S="${SAMPLE_INTERVAL_S:-1}"
POWER_SETTLE_SECONDS="${POWER_SETTLE_SECONDS:-8}"
STABLE_WINDOW_SAMPLES="${STABLE_WINDOW_SAMPLES:-8}"
STABLE_RANGE_PCT="${STABLE_RANGE_PCT:-4.0}"

GPU_SWEEP_SUMMARIZE_ONLY="${GPU_SWEEP_SUMMARIZE_ONLY:-0}"
GPU_SWEEP_SAMPLE_LOG="${GPU_SWEEP_SAMPLE_LOG:-}"

GPU_AVAILABLE_FREQ_PATH="${GPU_AVAILABLE_FREQ_PATH:-}"
GPU_MIN_FREQ_PATH="${GPU_MIN_FREQ_PATH:-}"
GPU_MAX_FREQ_PATH="${GPU_MAX_FREQ_PATH:-}"
GPU_CUR_FREQ_PATH="${GPU_CUR_FREQ_PATH:-}"
GPU_GOVERNOR_PATH="${GPU_GOVERNOR_PATH:-}"
GPU_PIN_GOVERNOR="${GPU_PIN_GOVERNOR:-}"

QNN_AOT_CONFIG="${QNN_AOT_CONFIG:-${GGML_QNN_AOT_CONFIG:-${QNN_DIR:+${QNN_DIR}/config.json}}}"
QNN_AOT_MODEL_DIR="${QNN_AOT_MODEL_DIR:-${GGML_QNN_AOT_MODEL_DIR:-${QNN_DIR}}}"
QNN_AOT_WRITE_GENERIC_KV="${QNN_AOT_WRITE_GENERIC_KV:-1}"
QNN_AOT_DISABLE_SEED_KV="${QNN_AOT_DISABLE_SEED_KV:-1}"

BATTERY_VOLTAGE_PATH="${BATTERY_VOLTAGE_PATH:-}"
BATTERY_CURRENT_PATH="${BATTERY_CURRENT_PATH:-}"
TEMP_PATH="${TEMP_PATH:-}"
KEEP_SCREEN_ON_TIMEOUT_MS="${KEEP_SCREEN_ON_TIMEOUT_MS:-1800000}"
SCREEN_BRIGHTNESS_OVERRIDE="${SCREEN_BRIGHTNESS_OVERRIDE:-}"

declare -a GPU_FREQS=()

RESULTS_CSV=""
TEMP_SCALE_TO_MC=""
TEMP_LIMIT_MC=""
COOLDOWN_TEMP_MC=""

REMOTE_BENCH_LOG=""
REMOTE_SAMPLE_LOG=""
REMOTE_META_LOG=""
REMOTE_STATUS_FILE=""
REMOTE_BENCH_PID_FILE=""
REMOTE_BENCH_PID=""

ORIG_GPU_MIN_FREQ=""
ORIG_GPU_MAX_FREQ=""
ORIG_GPU_GOVERNOR=""
GPU_SWEEP_LOWEST_FREQ=""
ORIG_SCREEN_OFF_TIMEOUT=""
ORIG_SCREEN_BRIGHTNESS=""
ORIG_SCREEN_BRIGHTNESS_MODE=""
ORIG_STAY_ON_WHILE_PLUGGED_IN=""

log() {
    printf '[gpu-freq-sweep] %s\n' "$*"
}

die() {
    printf '[gpu-freq-sweep] ERROR: %s\n' "$*" >&2
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

format_mhz() {
    awk -v x="$1" 'BEGIN { printf "%.0f", x / 1000000.0 }'
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
}

build_remote_qnn_env() {
    local out=""
    if [[ -n "${QNN_AOT_CONFIG}" ]]; then
        out="${out}export GGML_QNN_AOT_CONFIG=${QNN_AOT_CONFIG} && "
    fi
    if [[ -n "${QNN_AOT_MODEL_DIR}" ]]; then
        out="${out}export GGML_QNN_AOT_MODEL_DIR=${QNN_AOT_MODEL_DIR} && "
    fi
    out="${out}export GGML_QNN_AOT_WRITE_GENERIC_KV=${QNN_AOT_WRITE_GENERIC_KV} && "
    out="${out}export GGML_QNN_AOT_DISABLE_SEED_KV=${QNN_AOT_DISABLE_SEED_KV} && "
    printf '%s' "${out}"
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
    GPU_AVAILABLE_FREQ_PATH="$(find_first_readable_path \
        "${GPU_AVAILABLE_FREQ_PATH}" \
        "/sys/class/kgsl/kgsl-3d0/devfreq/available_frequencies" \
        "/sys/class/kgsl/kgsl-3d0/gpu_available_frequencies" \
        "/sys/class/devfreq/kgsl-3d0/available_frequencies")" || true

    GPU_MIN_FREQ_PATH="$(find_first_readable_path \
        "${GPU_MIN_FREQ_PATH}" \
        "/sys/class/kgsl/kgsl-3d0/devfreq/min_freq" \
        "/sys/class/devfreq/kgsl-3d0/min_freq")" || true

    GPU_MAX_FREQ_PATH="$(find_first_readable_path \
        "${GPU_MAX_FREQ_PATH}" \
        "/sys/class/kgsl/kgsl-3d0/devfreq/max_freq" \
        "/sys/class/devfreq/kgsl-3d0/max_freq")" || true

    GPU_CUR_FREQ_PATH="$(find_first_readable_path \
        "${GPU_CUR_FREQ_PATH}" \
        "/sys/class/kgsl/kgsl-3d0/devfreq/cur_freq" \
        "/sys/class/kgsl/kgsl-3d0/gpuclk" \
        "/sys/class/devfreq/kgsl-3d0/cur_freq")" || true

    GPU_GOVERNOR_PATH="$(find_first_readable_path \
        "${GPU_GOVERNOR_PATH}" \
        "/sys/class/kgsl/kgsl-3d0/devfreq/governor" \
        "/sys/class/devfreq/kgsl-3d0/governor")" || true

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

    [[ -n "${GPU_MIN_FREQ_PATH}" ]] || die "failed to discover GPU min_freq path"
    [[ -n "${GPU_MAX_FREQ_PATH}" ]] || die "failed to discover GPU max_freq path"
    [[ -n "${BATTERY_VOLTAGE_PATH}" ]] || die "failed to discover battery voltage path"
    [[ -n "${BATTERY_CURRENT_PATH}" ]] || die "failed to discover battery current path"
    [[ -n "${TEMP_PATH}" ]] || die "failed to discover device temperature path"

    TEMP_SCALE_TO_MC="$(infer_temp_scale_to_mc "$(read_remote_value "${TEMP_PATH}")")"
    TEMP_LIMIT_MC="$(float_to_mc "${TEMP_LIMIT_C}")"
    COOLDOWN_TEMP_MC="$(float_to_mc "${COOLDOWN_TEMP_C}")"

    log "temp path: ${TEMP_PATH} (scale=${TEMP_SCALE_TO_MC})"
    log "power paths: V=${BATTERY_VOLTAGE_PATH} I=${BATTERY_CURRENT_PATH}"
    log "gpu freq paths: min=${GPU_MIN_FREQ_PATH} max=${GPU_MAX_FREQ_PATH} cur=${GPU_CUR_FREQ_PATH:-unavailable}"
}

discover_freq_list() {
    local raw=""
    if [[ -n "${GPU_FREQ_LIST}" ]]; then
        raw="${GPU_FREQ_LIST}"
    else
        [[ -n "${GPU_AVAILABLE_FREQ_PATH}" ]] || die "GPU_FREQ_LIST is empty and available_frequencies path was not found"
        raw="$(adb_shell "cat ${GPU_AVAILABLE_FREQ_PATH}")"
    fi

    mapfile -t GPU_FREQS < <(printf '%s\n' "${raw}" | tr ' ' '\n' | awk '/^[0-9]+$/' | sort -nr)
    (( ${#GPU_FREQS[@]} > 0 )) || die "no GPU frequencies found"
    GPU_SWEEP_LOWEST_FREQ="${GPU_FREQS[$(( ${#GPU_FREQS[@]} - 1 ))]}"
}

save_original_gpu_state() {
    ORIG_GPU_MIN_FREQ="$(read_remote_value "${GPU_MIN_FREQ_PATH}")"
    ORIG_GPU_MAX_FREQ="$(read_remote_value "${GPU_MAX_FREQ_PATH}")"
    if [[ -n "${GPU_GOVERNOR_PATH}" ]]; then
        ORIG_GPU_GOVERNOR="$(read_remote_value "${GPU_GOVERNOR_PATH}")"
    fi
}

restore_original_gpu_state() {
    if [[ -n "${ORIG_GPU_MAX_FREQ}" && -n "${ORIG_GPU_MIN_FREQ}" ]]; then
        adb_root_shell "echo ${ORIG_GPU_MIN_FREQ} > ${GPU_MIN_FREQ_PATH} && echo ${ORIG_GPU_MAX_FREQ} > ${GPU_MAX_FREQ_PATH}" || true
    fi
    if [[ -n "${GPU_GOVERNOR_PATH}" && -n "${ORIG_GPU_GOVERNOR}" ]]; then
        adb_root_shell "echo ${ORIG_GPU_GOVERNOR} > ${GPU_GOVERNOR_PATH}" || true
    fi
}

pin_gpu_freq() {
    local freq="$1"
    if [[ -n "${GPU_GOVERNOR_PATH}" && -n "${GPU_PIN_GOVERNOR}" ]]; then
        adb_root_shell "echo ${GPU_PIN_GOVERNOR} > ${GPU_GOVERNOR_PATH}" || die "failed to set GPU governor to ${GPU_PIN_GOVERNOR}"
    fi

    adb_root_shell "echo ${GPU_SWEEP_LOWEST_FREQ} > ${GPU_MIN_FREQ_PATH} && echo ${freq} > ${GPU_MAX_FREQ_PATH} && echo ${freq} > ${GPU_MIN_FREQ_PATH}" \
        || die "failed to pin GPU frequency to ${freq}"
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

run_bench_with_local_sampling() {
    local bench_log="$1"
    local sample_log="$2"
    local meta_log="$3"

    local gpu_freq_value="0"
    if [[ -n "${GPU_CUR_FREQ_PATH}" ]]; then
        gpu_freq_value="$(read_remote_value "${GPU_CUR_FREQ_PATH}")"
        [[ -n "${gpu_freq_value}" ]] || gpu_freq_value="0"
    fi

    : > "${sample_log}"
    : > "${meta_log}"

    adb -s "${DEVICE}" shell "cd ${QNN_BIN} && \
export LD_LIBRARY_PATH=${QNN_BIN} && \
export ADSP_LIBRARY_PATH=${QNN_BIN} && \
export GGML_HEXAGON_EXPERIMENTAL=1 && \
export GGML_QNN_HTP_WORKPOINT=${HTP_WORKPOINT} && \
$(build_remote_qnn_env)\
export LLAMA_BENCH_FAST_EXIT=1 && \
taskset ${TASKSET_MASK} ./llama-bench -v \
  -m ${MODEL_PATH} \
  -ngl ${NGL} -dev ${LLAMA_DEV} -t ${LLAMA_THREADS} \
  -p ${PROMPT_TOKENS} -n ${GEN_TOKENS} \
  -d ${DEPTH_TOKENS} \
  -c ${CONTEXT_TOKENS} -r ${BENCH_REPEATS} \
  --no-warmup --mmap ${MMAP}" > "${bench_log}" 2>&1 &
    REMOTE_BENCH_PID=$!

    local status="ok"
    while kill -0 "${REMOTE_BENCH_PID}" >/dev/null 2>&1; do
        local ts
        local voltage
        local current
        local temp_raw
        local temp_mc
        local gpu_freq

        ts="$(date +%s)"
        voltage="$(read_remote_value "${BATTERY_VOLTAGE_PATH}")"
        current="$(read_remote_value "${BATTERY_CURRENT_PATH}")"
        temp_raw="$(read_remote_value "${TEMP_PATH}")"
        temp_mc="$(normalize_temp_to_mc "${temp_raw}" "${TEMP_SCALE_TO_MC}")"
        gpu_freq="${gpu_freq_value}"
        if [[ -n "${GPU_CUR_FREQ_PATH}" ]]; then
            gpu_freq="$(read_remote_value "${GPU_CUR_FREQ_PATH}")"
            [[ -n "${gpu_freq}" ]] || gpu_freq="${gpu_freq_value}"
        fi

        printf '%s,%s,%s,%s,%s\n' "${ts}" "${voltage}" "${current}" "${temp_mc}" "${gpu_freq}" >> "${sample_log}"

        if (( temp_mc >= TEMP_LIMIT_MC )); then
            printf 'THERMAL_ABORT,%s,%s\n' "${ts}" "${temp_mc}" > "${meta_log}"
            kill "${REMOTE_BENCH_PID}" >/dev/null 2>&1 || true
            status="thermal_abort"
            break
        fi

        sleep "${SAMPLE_INTERVAL_S}"
    done

    local bench_exit_code=0
    if wait "${REMOTE_BENCH_PID}"; then
        bench_exit_code=0
    else
        bench_exit_code=$?
    fi
    REMOTE_BENCH_PID=""

    if [[ ! -s "${meta_log}" ]]; then
        printf 'SAMPLER_EXIT,%s\n' "$(date +%s)" >> "${meta_log}"
    fi

    printf '%s,%s\n' "${bench_exit_code}" "${status}"
}

install_remote_sampler() {
    adb_shell_raw "mkdir -p ${REMOTE_WORKDIR}" >/dev/null
    cat <<'EOF' | adb -s "${DEVICE}" shell "cat > ${REMOTE_SAMPLER_PATH} && chmod 755 ${REMOTE_SAMPLER_PATH}"
#!/system/bin/sh

pid="$1"
out="$2"
meta="$3"
status_file="$4"
volt_path="$5"
curr_path="$6"
temp_path="$7"
freq_path="$8"
temp_scale="$9"
sample_interval="${10}"
temp_limit_mc="${11}"

normalize_temp_mc() {
    raw="$1"
    case "$temp_scale" in
        1)
            echo "$raw"
            ;;
        100)
            echo $(( raw * 100 ))
            ;;
        1000)
            echo $(( raw * 1000 ))
            ;;
        *)
            echo "$raw"
            ;;
    esac
}

while [ ! -f "$status_file" ] || [ -d "/proc/$pid" ]; do
    ts="$(date +%s)"
    voltage="$(cat "$volt_path" 2>/dev/null || echo 0)"
    current="$(cat "$curr_path" 2>/dev/null || echo 0)"
    temp_raw="$(cat "$temp_path" 2>/dev/null || echo 0)"
    temp_mc="$(normalize_temp_mc "$temp_raw")"
    if [ -n "$freq_path" ] && [ -r "$freq_path" ]; then
        gpu_freq="$(cat "$freq_path" 2>/dev/null || echo 0)"
    else
        gpu_freq=0
    fi

    printf '%s,%s,%s,%s,%s\n' "$ts" "$voltage" "$current" "$temp_mc" "$gpu_freq" >> "$out"

    if [ "$temp_mc" -ge "$temp_limit_mc" ]; then
        printf 'THERMAL_ABORT,%s,%s\n' "$ts" "$temp_mc" > "$meta"
        kill -INT "$pid" 2>/dev/null || true
        sleep 1
        kill -TERM "$pid" 2>/dev/null || true
        sleep 1
        kill -KILL "$pid" 2>/dev/null || true
        exit 2
    fi

    sleep "$sample_interval"
done

printf 'SAMPLER_EXIT,%s\n' "$(date +%s)" >> "$meta"
EOF
}

start_remote_bench() {
    local run_name="$1"

    REMOTE_BENCH_LOG="${REMOTE_WORKDIR}/${run_name}.bench.log"
    REMOTE_SAMPLE_LOG="${REMOTE_WORKDIR}/${run_name}.samples.csv"
    REMOTE_META_LOG="${REMOTE_WORKDIR}/${run_name}.meta"
    REMOTE_STATUS_FILE="${REMOTE_WORKDIR}/${run_name}.status"
    REMOTE_BENCH_PID_FILE="${REMOTE_WORKDIR}/${run_name}.pid"
    REMOTE_BENCH_PID=""

    local cmd
    cmd=$(cat <<EOF
rm -f ${REMOTE_BENCH_LOG} ${REMOTE_SAMPLE_LOG} ${REMOTE_META_LOG} ${REMOTE_STATUS_FILE} ${REMOTE_BENCH_PID_FILE}
(
cd ${QNN_BIN} &&
export LD_LIBRARY_PATH=${QNN_BIN} &&
export ADSP_LIBRARY_PATH=${QNN_BIN} &&
export GGML_HEXAGON_EXPERIMENTAL=1 &&
export GGML_QNN_HTP_WORKPOINT=${HTP_WORKPOINT} &&
$(build_remote_qnn_env)
export LLAMA_BENCH_FAST_EXIT=1 &&
taskset ${TASKSET_MASK} ./llama-bench -v \
    -m ${MODEL_PATH} \
    -ngl ${NGL} -dev ${LLAMA_DEV} -t ${LLAMA_THREADS} \
    -p ${PROMPT_TOKENS} -n ${GEN_TOKENS} \
    -d ${DEPTH_TOKENS} \
    -c ${CONTEXT_TOKENS} -r ${BENCH_REPEATS} \
    --no-warmup --mmap ${MMAP} > ${REMOTE_BENCH_LOG} 2>&1 &
bench_pid=\$!
echo \$bench_pid > ${REMOTE_BENCH_PID_FILE}
wait \$bench_pid
echo \$? > ${REMOTE_STATUS_FILE}
) >/dev/null 2>&1 &
echo \$!
EOF
)

    adb_shell "${cmd}" >/dev/null

    local attempt
    for attempt in $(seq 1 30); do
        local llama_pid
        llama_pid="$(adb_shell "pidof llama-bench 2>/dev/null | awk '{ print \$1 }' || true" | tr -d '[:space:]')"
        if [[ "${llama_pid}" =~ ^[0-9]+$ ]]; then
            REMOTE_BENCH_PID="${llama_pid}"
            return 0
        fi

        REMOTE_BENCH_PID="$(adb_shell "cat ${REMOTE_BENCH_PID_FILE} 2>/dev/null || true" | tr -d '[:space:]')"
        if [[ "${REMOTE_BENCH_PID}" =~ ^[0-9]+$ ]]; then
            return 0
        fi
        sleep 1
    done

    die "failed to obtain remote llama-bench pid for ${run_name}"
}

start_remote_sampler() {
    local freq_path_arg="/dev/null"
    if [[ -n "${GPU_CUR_FREQ_PATH}" ]]; then
        freq_path_arg="${GPU_CUR_FREQ_PATH}"
    fi

    adb -s "${DEVICE}" shell "su -c 'nohup sh ${REMOTE_SAMPLER_PATH} ${REMOTE_BENCH_PID} ${REMOTE_SAMPLE_LOG} ${REMOTE_META_LOG} ${REMOTE_STATUS_FILE} ${BATTERY_VOLTAGE_PATH} ${BATTERY_CURRENT_PATH} ${TEMP_PATH} ${freq_path_arg} ${TEMP_SCALE_TO_MC} ${SAMPLE_INTERVAL_S} ${TEMP_LIMIT_MC} >/dev/null 2>&1 &'" >/dev/null 2>&1
}

wait_for_remote_completion() {
    local missing_status_count=0

    while true; do
        local status
        status="$(adb_shell "cat ${REMOTE_STATUS_FILE} 2>/dev/null || true" | tr -d '[:space:]')"
        if [[ "${status}" =~ ^-?[0-9]+$ ]]; then
            printf '%s\n' "${status}"
            return 0
        fi

        if [[ -n "${REMOTE_BENCH_PID}" ]] && ! adb_test "[ -d /proc/${REMOTE_BENCH_PID} ]"; then
            missing_status_count=$((missing_status_count + 1))
            if (( missing_status_count >= 10 )); then
                printf '255\n'
                return 0
            fi
        else
            missing_status_count=0
        fi

        sleep 1
    done
}

pull_remote_file() {
    local remote_path="$1"
    local local_path="$2"
    adb -s "${DEVICE}" shell "cat ${remote_path} 2>/dev/null || true" | tr -d '\r' > "${local_path}"
}

thermal_abort_happened() {
    local meta_file="$1"
    grep -q '^THERMAL_ABORT,' "${meta_file}" 2>/dev/null
}

extract_throughput() {
    local bench_log="$1"
    local throughput
    local label

    local table_line
    table_line="$(awk -F'|' '/\|[[:space:]]*(pp|tg)[0-9]+[[:space:]]*\|/ { line=$0 } END { print line }' "${bench_log}")"
    if [[ -n "${table_line}" ]]; then
        label="$(awk -F'|' '/\|[[:space:]]*(pp|tg)[0-9]+[[:space:]]*\|/ { line=$10 } END { gsub(/^[[:space:]]+|[[:space:]]+$/, "", line); print line }' "${bench_log}")"
        throughput="$(awk -F'|' '/\|[[:space:]]*(pp|tg)[0-9]+[[:space:]]*\|/ { line=$11 } END { gsub(/^[[:space:]]+|[[:space:]]+$/, "", line); split(line, parts, / ± /); print parts[1] }' "${bench_log}")"
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
        printf 'NA,NA,NA,NA,NA,0,0,NA,0,NA\n'
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
            freq[n] = $5 + 0
        }
        END {
            if (n == 0) {
                print "NA,NA,NA,NA,NA,0,0,NA,0,NA"
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
            sum_freq = 0
            max_temp = temp[stable_start]
            count = 0
            for (i = stable_start; i <= stable_end; ++i) {
                sum_power += power[i]
                sum_temp += temp[i]
                sum_freq += freq[i]
                if (temp[i] > max_temp) {
                    max_temp = temp[i]
                }
                count++
            }

            printf "%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%.0f,%d,%.2f\n", \
                sum_power / count, \
                (sum_temp / count) / 1000.0, \
                max_temp / 1000.0, \
                temp[stable_start] / 1000.0, \
                temp[stable_end] / 1000.0, \
                count, \
                stable_start, \
                sum_freq / count, \
                stable_end, \
                stable_range
        }' "${sample_log}"
}

append_result() {
    local freq_hz="$1"
    local status="$2"
    local exit_code="$3"
    local avg_power_mw="$4"
    local avg_temp_c="$5"
    local max_temp_c="$6"
    local start_temp_c="$7"
    local end_temp_c="$8"
    local throughput_tok_s="$9"
    local throughput_label="${10}"
    local sample_count="${11}"
    local stable_start_index="${12}"
    local avg_gpu_freq_hz="${13}"
    local stable_end_index="${14}"
    local stable_range_pct="${15}"
    local bench_log="${16}"
    local sample_log="${17}"

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${freq_hz}" \
        "$(format_mhz "${freq_hz}")" \
        "${status}" \
        "${exit_code}" \
        "${avg_power_mw}" \
        "${avg_temp_c}" \
        "${max_temp_c}" \
        "${start_temp_c}" \
        "${end_temp_c}" \
        "${throughput_tok_s}" \
        "${throughput_label}" \
        "${sample_count}" \
        "${stable_start_index}" \
        "${avg_gpu_freq_hz}" \
        "${stable_end_index}" \
        "${stable_range_pct}" \
        "${bench_log}" \
        "${sample_log}" >> "${RESULTS_CSV}"
}

cleanup() {
    if [[ -n "${REMOTE_BENCH_PID}" && "${REMOTE_BENCH_PID}" =~ ^[0-9]+$ ]]; then
        kill -TERM "${REMOTE_BENCH_PID}" >/dev/null 2>&1 || true
    fi
    restore_display_state
    restore_original_gpu_state
}

main() {
    mkdir -p "${OUTPUT_DIR}"
    RESULTS_CSV="${OUTPUT_DIR}/results.csv"
    printf '%s\n' 'freq_hz,freq_mhz,status,bench_exit_code,avg_power_mw,avg_temp_c,max_temp_c,start_temp_c,end_temp_c,throughput_tok_s,throughput_label,sample_count,stable_start_index,avg_gpu_freq_hz,stable_end_index,stable_range_pct,bench_log,sample_log' > "${RESULTS_CSV}"

    require_runtime_inputs
    check_device_online
    save_display_state
    ensure_screen_on
    verify_screen_on || die "failed to keep the device screen ON before the sweep starts"
    discover_sysfs_paths
    discover_freq_list
    save_original_gpu_state

    log "run build-npu-opencl.sh before real measurements to keep the binary consistent with the experiment setup"
    log "output dir: ${OUTPUT_DIR}"
    log "frequencies: ${GPU_FREQS[*]}"
    log "decode bench: -ngl ${NGL} -dev ${LLAMA_DEV} -p ${PROMPT_TOKENS} -n ${GEN_TOKENS} -d ${DEPTH_TOKENS} -c ${CONTEXT_TOKENS} -r ${BENCH_REPEATS}"
    log "temperature limit: ${TEMP_LIMIT_C}C"

    local freq_hz
    for freq_hz in "${GPU_FREQS[@]}"; do
        local run_name
        local bench_log
        local sample_log
        local meta_log
        local bench_exit_code
        local status
        local run_result
        local sample_summary
        local throughput_summary
        local avg_power_mw
        local avg_temp_c
        local max_temp_c
        local start_temp_c
        local end_temp_c
        local sample_count
        local stable_start_index
        local avg_gpu_freq_hz
        local stable_end_index
        local stable_range_pct
        local throughput_tok_s
        local throughput_label

        run_name="gpu_$(format_mhz "${freq_hz}")mhz"
        bench_log="${OUTPUT_DIR}/${run_name}.bench.log"
        sample_log="${OUTPUT_DIR}/${run_name}.samples.csv"
        meta_log="${OUTPUT_DIR}/${run_name}.meta"

        log "waiting for cooldown before ${run_name}"
        wait_for_cooldown

        ensure_screen_on
        verify_screen_on || die "screen turned OFF before ${run_name}"

        log "pinning GPU frequency to ${freq_hz} Hz ($(format_mhz "${freq_hz}") MHz)"
        pin_gpu_freq "${freq_hz}"
        sleep 1

        log "starting benchmark for ${run_name}"
        run_result="$(run_bench_with_local_sampling "${bench_log}" "${sample_log}" "${meta_log}")"
        IFS=',' read -r bench_exit_code status <<< "${run_result}"

        if [[ "${status}" == "ok" && "${bench_exit_code}" != "0" ]]; then
            status="bench_failed"
        fi

        sample_summary="$(summarize_samples "${sample_log}")"
        IFS=',' read -r avg_power_mw avg_temp_c max_temp_c start_temp_c end_temp_c sample_count stable_start_index avg_gpu_freq_hz stable_end_index stable_range_pct <<< "${sample_summary}"

        throughput_summary="$(extract_throughput "${bench_log}")"
        IFS=',' read -r throughput_tok_s throughput_label <<< "${throughput_summary}"

        append_result \
            "${freq_hz}" \
            "${status}" \
            "${bench_exit_code}" \
            "${avg_power_mw}" \
            "${avg_temp_c}" \
            "${max_temp_c}" \
            "${start_temp_c}" \
            "${end_temp_c}" \
            "${throughput_tok_s}" \
            "${throughput_label}" \
            "${sample_count}" \
            "${stable_start_index}" \
            "${avg_gpu_freq_hz}" \
            "${stable_end_index}" \
            "${stable_range_pct}" \
            "${bench_log}" \
            "${sample_log}"

        log "finished ${run_name}: status=${status} avg_power_mw=${avg_power_mw} stable_window=${stable_start_index}-${stable_end_index} stable_range_pct=${stable_range_pct} throughput=${throughput_tok_s} label=${throughput_label} max_temp=${max_temp_c}C"
        REMOTE_BENCH_PID=""
    done

    log "results written to ${RESULTS_CSV}"
}

if [[ "${GPU_SWEEP_SUMMARIZE_ONLY}" == "1" ]]; then
    [[ -n "${GPU_SWEEP_SAMPLE_LOG}" ]] || die "GPU_SWEEP_SAMPLE_LOG is required when GPU_SWEEP_SUMMARIZE_ONLY=1"
    summarize_samples "${GPU_SWEEP_SAMPLE_LOG}"
    exit 0
fi

trap cleanup EXIT
main "$@"
