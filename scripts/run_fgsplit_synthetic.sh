#!/usr/bin/env bash

set -euo pipefail

DEVICE="${DEVICE:-}"
MODEL_PATH="${MODEL_PATH:-${MODEL:-}}"
MODE="${MODE:-synthetic}"
BACKEND_POLICY="${BACKEND_POLICY:-fine_grained_qnn_gpu}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-results/fgsplit/fgsplit-synthetic-${RUN_ID}}"
RESULTS_CSV="${RESULTS_CSV:-results/fgsplit/fgsplit_power_profile.csv}"
SUMMARY_MD="${SUMMARY_MD:-docs/实验结果/FGSplit_${BACKEND_POLICY}_${RUN_ID}.md}"

REMOTE_BIN="${REMOTE_BIN:-${BENCH_DIR:-/data/local/tmp/acom-qnn-phase-materializer/bin}}"
REMOTE_OUTPUT_DIR="${REMOTE_OUTPUT_DIR:-/data/local/tmp/fgsplit_synthetic_${RUN_ID}}"
QNN_STAGE_DIR="${QNN_STAGE_DIR:-${STAGE_QNN_DIR:-${QNN_DIR:-${GGML_QNN_AOT_MODEL_DIR:-}}}}"
QNN_AOT_CONFIG="${QNN_AOT_CONFIG:-${GGML_QNN_AOT_CONFIG:-${QNN_STAGE_DIR:+${QNN_STAGE_DIR}/config.json}}}"
QNN_AOT_MODEL_DIR="${QNN_AOT_MODEL_DIR:-${GGML_QNN_AOT_MODEL_DIR:-${QNN_STAGE_DIR}}}"

LAYERS="${LAYERS:-28}"
FG_MAX_LAYERS="${FG_MAX_LAYERS:-${LAYERS}}"
ROUNDS="${ROUNDS:-20}"
CONTEXT_LEN="${CONTEXT_LEN:-512}"
PROMPT_TOKENS="${PROMPT_TOKENS:-${CONTEXT_LEN}}"
BENCH_PROMPT_TOKENS="${BENCH_PROMPT_TOKENS:-0}"
DECODE_TOKENS="${DECODE_TOKENS:-64}"
CONTEXT_SIZE="${CONTEXT_SIZE:-${CTX_SIZE:-2048}}"
BATCH_TOKENS="${BATCH_TOKENS:-${PROMPT_TOKENS}}"
UBATCH_TOKENS="${UBATCH_TOKENS:-${BATCH_TOKENS}}"
TASKSET_MASK="${TASKSET_MASK:-80}"
LLAMA_THREADS="${LLAMA_THREADS:-1}"
MMAP="${MMAP:-0}"

QNN_WORKPOINT="${QNN_WORKPOINT:-${HTP_WORKPOINT:-burst}}"
QNN_AOT_EVICT_STATELESS_STAGE_GRAPHS="${QNN_AOT_EVICT_STATELESS_STAGE_GRAPHS:-0}"
QNN_AOT_TRACE_LOAD_TIMING="${QNN_AOT_TRACE_LOAD_TIMING:-1}"
LLAMA_BENCH_WARMUP="${LLAMA_BENCH_WARMUP:-1}"
LLAMA_BENCH_QNN_PREWARM_DECODE="${LLAMA_BENCH_QNN_PREWARM_DECODE:-1}"
GPU_FREQ_MHZ="${GPU_FREQ_MHZ:-967}"
GPU_PIN_GOVERNOR="${GPU_PIN_GOVERNOR:-}"
OPENCL_SIM_BUSY="${OPENCL_SIM_BUSY:-0}"
OPENCL_SIM_BUSY_GLOBAL="${OPENCL_SIM_BUSY_GLOBAL:-262144}"
OPENCL_SIM_BUSY_ITERS="${OPENCL_SIM_BUSY_ITERS:-8192}"
OPENCL_SIM_BUSY_BUFFER_ELEMS="${OPENCL_SIM_BUSY_BUFFER_ELEMS:-1048576}"
OPENCL_SIM_BUSY_SLEEP_US="${OPENCL_SIM_BUSY_SLEEP_US:-0}"
if [[ -z "${FGSPLIT_REQUIRE_SUPPORT_OK+x}" ]]; then
    if [[ "${BACKEND_POLICY}" == "fine_grained_qnn_gpu" ]]; then
        FGSPLIT_REQUIRE_SUPPORT_OK=1
    else
        FGSPLIT_REQUIRE_SUPPORT_OK=0
    fi
fi
TEMP_LIMIT_C="${TEMP_LIMIT_C:-38.0}"
COOLDOWN_TEMP_C="${COOLDOWN_TEMP_C:-37.0}"
COOLDOWN_TIMEOUT_S="${COOLDOWN_TIMEOUT_S:-900}"
COOLDOWN_POLL_S="${COOLDOWN_POLL_S:-5}"
SAMPLE_INTERVAL_S="${SAMPLE_INTERVAL_S:-1}"
POWER_SAMPLER="${POWER_SAMPLER:-remote}"
KEEP_SCREEN_ON_TIMEOUT_MS="${KEEP_SCREEN_ON_TIMEOUT_MS:-1800000}"

case "${BACKEND_POLICY}" in
    fine_grained_qnn_gpu)
        DEFAULT_FG_ROUTE="attn_proj=qnn-npu,attn_core=opencl,attn_out=cpu,ffn=qnn-npu,output=cpu"
        DEFAULT_FG_DEVICES="qnn-npu/GPUOpenCL"
        ;;
    single_gpu_opencl)
        DEFAULT_FG_ROUTE="opencl"
        DEFAULT_FG_DEVICES="GPUOpenCL"
        ;;
    single_qnn_npu)
        DEFAULT_FG_ROUTE="qnn-npu"
        DEFAULT_FG_DEVICES="qnn-npu"
        ;;
    *)
        printf '[fgsplit-synthetic] ERROR: unsupported BACKEND_POLICY=%s\n' "${BACKEND_POLICY}" >&2
        exit 1
        ;;
esac

FG_ROUTE="${FG_ROUTE:-${DEFAULT_FG_ROUTE}}"
FG_DEVICES="${FG_DEVICES:-${DEFAULT_FG_DEVICES}}"

BATTERY_VOLTAGE_PATH="${BATTERY_VOLTAGE_PATH:-}"
BATTERY_CURRENT_PATH="${BATTERY_CURRENT_PATH:-}"
TEMP_PATH="${TEMP_PATH:-}"
GPU_BUSY_PATH="${GPU_BUSY_PATH:-}"
GPU_CLOCK_PATH="${GPU_CLOCK_PATH:-}"
GPU_MIN_FREQ_PATH="${GPU_MIN_FREQ_PATH:-}"
GPU_MAX_FREQ_PATH="${GPU_MAX_FREQ_PATH:-}"
GPU_CUR_FREQ_PATH="${GPU_CUR_FREQ_PATH:-}"
GPU_GOVERNOR_PATH="${GPU_GOVERNOR_PATH:-}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_LOG="${OUTPUT_DIR}/raw/bench.log"
SAMPLE_LOG="${OUTPUT_DIR}/raw/power_samples.csv"
RUN_CSV="${OUTPUT_DIR}/fgsplit_power_profile.csv"
COMMAND_FILE="${OUTPUT_DIR}/command.sh"
LOCAL_COMMAND_FILE="${OUTPUT_DIR}/local_command.sh"

TEMP_SCALE_TO_MC=""
TEMP_LIMIT_MC=""
COOLDOWN_TEMP_MC=""
CURRENT_SCALE_TO_UA=""

ORIG_SCREEN_OFF_TIMEOUT=""
ORIG_SCREEN_BRIGHTNESS=""
ORIG_SCREEN_BRIGHTNESS_MODE=""
ORIG_STAY_ON_WHILE_PLUGGED_IN=""
DISPLAY_STATE_SAVED=0

ORIG_GPU_MIN_FREQ=""
ORIG_GPU_MAX_FREQ=""
ORIG_GPU_GOVERNOR=""
GPU_STATE_SAVED=0

log() {
    printf '[fgsplit-synthetic] %s\n' "$*"
}

die() {
    printf '[fgsplit-synthetic] ERROR: %s\n' "$*" >&2
    exit 1
}

adb_shell() {
    adb -s "${DEVICE}" shell "$@" | tr -d '\r'
}

adb_root_capture() {
    adb -s "${DEVICE}" shell "su -c '$1'" | tr -d '\r'
}

adb_root_shell() {
    adb -s "${DEVICE}" shell "su -c '$1'" >/dev/null 2>&1
}

adb_test() {
    adb -s "${DEVICE}" shell "su -c '$1'" >/dev/null 2>&1
}

backend_uses_qnn() {
    [[ "${BACKEND_POLICY}" == "fine_grained_qnn_gpu" || "${BACKEND_POLICY}" == "single_qnn_npu" ]]
}

backend_uses_opencl() {
    [[ "${BACKEND_POLICY}" == "fine_grained_qnn_gpu" || "${BACKEND_POLICY}" == "single_gpu_opencl" ]]
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

read_remote_value() {
    local path="$1"
    if [[ -z "${path}" ]]; then
        return 0
    fi
    adb_root_capture "cat ${path} 2>/dev/null || true" | tr -d '[:space:]'
}

find_first_readable_path() {
    local candidate
    for candidate in "$@"; do
        if [[ -n "${candidate}" ]] && adb_test "[ -r ${candidate} ]"; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    return 1
}

float_to_mc() {
    awk -v x="$1" 'BEGIN { printf "%.0f\n", x * 1000.0 }'
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

infer_current_scale_to_ua() {
    local raw="$1"
    if [[ -z "${raw}" || ! "${raw}" =~ ^-?[0-9]+$ ]]; then
        die "invalid current sample while inferring scale: ${raw}"
    fi
    local abs_raw="${raw#-}"
    if (( abs_raw >= 10000 )); then
        printf '1\n'
    else
        # Some Android vendor battery drivers expose current_now in mA even
        # though the power_supply convention and filename imply microamps.
        printf '1000\n'
    fi
}

normalize_temp_to_mc() {
    local raw="$1"
    local scale="$2"
    if [[ -z "${raw}" || ! "${raw}" =~ ^-?[0-9]+$ ]]; then
        printf '\n'
        return 0
    fi
    case "${scale}" in
        1) printf '%s\n' "${raw}" ;;
        100) printf '%s\n' "$(( raw * 100 ))" ;;
        1000) printf '%s\n' "$(( raw * 1000 ))" ;;
        *) die "unsupported temperature scale: ${scale}" ;;
    esac
}

normalize_current_to_ua() {
    local raw="$1"
    local scale="$2"
    if [[ -z "${raw}" || ! "${raw}" =~ ^-?[0-9]+$ ]]; then
        printf '\n'
        return 0
    fi
    case "${scale}" in
        1) printf '%s\n' "${raw}" ;;
        1000) printf '%s\n' "$(( raw * 1000 ))" ;;
        *) die "unsupported current scale: ${scale}" ;;
    esac
}

format_mc() {
    awk -v x="$1" 'BEGIN { printf "%.2f", x / 1000.0 }'
}

remote_quote() {
    local value="$1"
    printf "'%s'" "${value//\'/\'\\\'\'}"
}

shell_quote() {
    local value="$1"
    printf "'%s'" "${value//\'/\'\\\'\'}"
}

require_inputs() {
    [[ "${MODE}" == "synthetic" ]] || die "only MODE=synthetic is supported in this script"
    [[ -n "${DEVICE}" ]] || die "DEVICE must be set"
    [[ -n "${MODEL_PATH}" ]] || die "MODEL_PATH must be set"
    [[ -n "${REMOTE_BIN}" ]] || die "REMOTE_BIN or BENCH_DIR must be set"
    if backend_uses_qnn; then
        [[ -n "${QNN_AOT_CONFIG}" ]] || die "QNN_AOT_CONFIG, GGML_QNN_AOT_CONFIG, QNN_STAGE_DIR, or QNN_DIR must be set"
        [[ -n "${QNN_AOT_MODEL_DIR}" ]] || die "QNN_AOT_MODEL_DIR, GGML_QNN_AOT_MODEL_DIR, QNN_STAGE_DIR, or QNN_DIR must be set"
    fi
    adb -s "${DEVICE}" get-state >/dev/null 2>&1 || die "device ${DEVICE} is offline"
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
    GPU_BUSY_PATH="$(find_first_readable_path \
        "${GPU_BUSY_PATH}" \
        "/sys/kernel/gpu/gpu_busy" \
        "/sys/class/kgsl/kgsl-3d0/gpubusy")" || true
    GPU_CLOCK_PATH="$(find_first_readable_path \
        "${GPU_CLOCK_PATH}" \
        "/sys/kernel/gpu/gpu_clock" \
        "/sys/class/kgsl/kgsl-3d0/gpuclk" \
        "/sys/class/devfreq/kgsl-3d0/cur_freq" \
        "/sys/class/kgsl/kgsl-3d0/devfreq/cur_freq")" || true
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
        "/sys/class/devfreq/kgsl-3d0/cur_freq")" || true
    GPU_GOVERNOR_PATH="$(find_first_readable_path \
        "${GPU_GOVERNOR_PATH}" \
        "/sys/class/kgsl/kgsl-3d0/devfreq/governor" \
        "/sys/class/devfreq/kgsl-3d0/governor")" || true

    [[ -n "${BATTERY_VOLTAGE_PATH}" ]] || die "failed to discover battery voltage path"
    [[ -n "${BATTERY_CURRENT_PATH}" ]] || die "failed to discover battery current path"
    [[ -n "${TEMP_PATH}" ]] || die "failed to discover device temperature path"

    TEMP_SCALE_TO_MC="$(infer_temp_scale_to_mc "$(read_remote_value "${TEMP_PATH}")")"
    CURRENT_SCALE_TO_UA="$(infer_current_scale_to_ua "$(read_remote_value "${BATTERY_CURRENT_PATH}")")"
    TEMP_LIMIT_MC="$(float_to_mc "${TEMP_LIMIT_C}")"
    COOLDOWN_TEMP_MC="$(float_to_mc "${COOLDOWN_TEMP_C}")"
}

save_display_state() {
    ORIG_SCREEN_OFF_TIMEOUT="$(read_setting system screen_off_timeout)"
    ORIG_SCREEN_BRIGHTNESS="$(read_setting system screen_brightness)"
    ORIG_SCREEN_BRIGHTNESS_MODE="$(read_setting system screen_brightness_mode)"
    ORIG_STAY_ON_WHILE_PLUGGED_IN="$(read_setting global stay_on_while_plugged_in)"
    DISPLAY_STATE_SAVED=1
}

ensure_screen_on() {
    adb -s "${DEVICE}" shell "input keyevent KEYCODE_WAKEUP" >/dev/null 2>&1 || true
    adb -s "${DEVICE}" shell "wm dismiss-keyguard" >/dev/null 2>&1 || true
    adb -s "${DEVICE}" shell "settings put system screen_off_timeout ${KEEP_SCREEN_ON_TIMEOUT_MS}" >/dev/null 2>&1 || true
    adb -s "${DEVICE}" shell "settings put global stay_on_while_plugged_in 7" >/dev/null 2>&1 || true
}

restore_display_state() {
    if [[ "${DISPLAY_STATE_SAVED}" != "1" ]]; then
        return 0
    fi
    restore_setting system screen_off_timeout "${ORIG_SCREEN_OFF_TIMEOUT}"
    restore_setting system screen_brightness "${ORIG_SCREEN_BRIGHTNESS}"
    restore_setting system screen_brightness_mode "${ORIG_SCREEN_BRIGHTNESS_MODE}"
    restore_setting global stay_on_while_plugged_in "${ORIG_STAY_ON_WHILE_PLUGGED_IN}"
}

gpu_freq_hz() {
    awk -v mhz="${GPU_FREQ_MHZ}" 'BEGIN {
        if (mhz == "" || mhz == "0") {
            print "";
        } else if (mhz > 100000) {
            printf "%.0f\n", mhz;
        } else {
            printf "%.0f\n", mhz * 1000000.0;
        }
    }'
}

save_and_pin_gpu_freq() {
    local freq_hz
    freq_hz="$(gpu_freq_hz)"
    if [[ -z "${freq_hz}" || "${freq_hz}" == "0" ]]; then
        return 0
    fi
    if [[ -z "${GPU_MIN_FREQ_PATH}" || -z "${GPU_MAX_FREQ_PATH}" ]]; then
        log "GPU frequency paths not found; requested GPU_FREQ_MHZ=${GPU_FREQ_MHZ} will be recorded but not pinned"
        return 0
    fi

    ORIG_GPU_MIN_FREQ="$(read_remote_value "${GPU_MIN_FREQ_PATH}")"
    ORIG_GPU_MAX_FREQ="$(read_remote_value "${GPU_MAX_FREQ_PATH}")"
    if [[ -n "${GPU_GOVERNOR_PATH}" ]]; then
        ORIG_GPU_GOVERNOR="$(read_remote_value "${GPU_GOVERNOR_PATH}")"
    fi
    GPU_STATE_SAVED=1

    if [[ -n "${GPU_PIN_GOVERNOR}" && -n "${GPU_GOVERNOR_PATH}" ]]; then
        adb_root_shell "echo ${GPU_PIN_GOVERNOR} > ${GPU_GOVERNOR_PATH}" || true
    fi
    adb_root_shell "echo ${freq_hz} > ${GPU_MIN_FREQ_PATH}" || true
    adb_root_shell "echo ${freq_hz} > ${GPU_MAX_FREQ_PATH}" || true
}

restore_gpu_freq() {
    if [[ "${GPU_STATE_SAVED}" != "1" ]]; then
        return 0
    fi
    if [[ -n "${ORIG_GPU_MIN_FREQ}" && -n "${GPU_MIN_FREQ_PATH}" ]]; then
        adb_root_shell "echo ${ORIG_GPU_MIN_FREQ} > ${GPU_MIN_FREQ_PATH}" || true
    fi
    if [[ -n "${ORIG_GPU_MAX_FREQ}" && -n "${GPU_MAX_FREQ_PATH}" ]]; then
        adb_root_shell "echo ${ORIG_GPU_MAX_FREQ} > ${GPU_MAX_FREQ_PATH}" || true
    fi
    if [[ -n "${ORIG_GPU_GOVERNOR}" && -n "${GPU_GOVERNOR_PATH}" ]]; then
        adb_root_shell "echo ${ORIG_GPU_GOVERNOR} > ${GPU_GOVERNOR_PATH}" || true
    fi
}

cleanup() {
    restore_gpu_freq
    restore_display_state
}

current_temp_mc() {
    normalize_temp_to_mc "$(read_remote_value "${TEMP_PATH}")" "${TEMP_SCALE_TO_MC}"
}

wait_for_cooldown() {
    local start_ts now_ts temp_mc
    start_ts="$(date +%s)"
    while true; do
        temp_mc="$(current_temp_mc)"
        if [[ -n "${temp_mc}" ]] && (( temp_mc <= COOLDOWN_TEMP_MC )); then
            log "cooldown ok: $(format_mc "${temp_mc}") C"
            return 0
        fi
        now_ts="$(date +%s)"
        if (( now_ts - start_ts >= COOLDOWN_TIMEOUT_S )); then
            die "cooldown timed out; last temp=$(format_mc "${temp_mc:-0}") C"
        fi
        log "cooling: temp=$(format_mc "${temp_mc:-0}") C target=${COOLDOWN_TEMP_C} C"
        sleep "${COOLDOWN_POLL_S}"
    done
}

write_remote_command() {
    local remote_out_q remote_bin_q model_q qnn_config_q qnn_model_q route_q devices_q
    local voltage_path_q current_path_q temp_path_q gpu_busy_path_q gpu_clock_path_q
    local qnn_env_block fg_env_block opencl_env_block route_env_block warmup_arg
    remote_out_q="$(remote_quote "${REMOTE_OUTPUT_DIR}")"
    remote_bin_q="$(remote_quote "${REMOTE_BIN}")"
    model_q="$(remote_quote "${MODEL_PATH}")"
    qnn_config_q="$(remote_quote "${QNN_AOT_CONFIG}")"
    qnn_model_q="$(remote_quote "${QNN_AOT_MODEL_DIR}")"
    route_q="$(remote_quote "${FG_ROUTE}")"
    devices_q="$(remote_quote "${FG_DEVICES}")"
    voltage_path_q="$(remote_quote "${BATTERY_VOLTAGE_PATH}")"
    current_path_q="$(remote_quote "${BATTERY_CURRENT_PATH}")"
    temp_path_q="$(remote_quote "${TEMP_PATH}")"
    gpu_busy_path_q="$(remote_quote "${GPU_BUSY_PATH}")"
    gpu_clock_path_q="$(remote_quote "${GPU_CLOCK_PATH:-${GPU_CUR_FREQ_PATH}}")"

    if backend_uses_qnn; then
        qnn_env_block="export GGML_HEXAGON_EXPERIMENTAL=1 &&
export GGML_QNN_HTP_WORKPOINT=$(remote_quote "${QNN_WORKPOINT}") &&
export GGML_QNN_AOT_CONFIG=${qnn_config_q} &&
export GGML_QNN_AOT_MODEL_DIR=${qnn_model_q} &&
export GGML_QNN_AOT_WRITE_GENERIC_KV=1 &&
export GGML_QNN_AOT_DISABLE_SEED_KV=1 &&
export GGML_QNN_AOT_EVICT_STATELESS_STAGE_GRAPHS=$(remote_quote "${QNN_AOT_EVICT_STATELESS_STAGE_GRAPHS}") &&
export GGML_QNN_AOT_TRACE_LOAD_TIMING=$(remote_quote "${QNN_AOT_TRACE_LOAD_TIMING}") &&
export GGML_QNN_AOT_TRACE_MATCH=1 &&
export GGML_QNN_AOT_TRACE_BIND=1 &&
export GGML_QNN_AOT_TRACE_ASSIGN=1 &&
export GGML_QNN_AOT_FG_TRACE=1 &&
unset GGML_QNN_AOT_ALLOW_JIT_FALLBACK &&"
    else
        qnn_env_block="unset GGML_HEXAGON_EXPERIMENTAL &&
unset GGML_QNN_HTP_WORKPOINT &&
unset GGML_QNN_AOT_CONFIG &&
unset GGML_QNN_AOT_MODEL_DIR &&
unset GGML_QNN_AOT_WRITE_GENERIC_KV &&
unset GGML_QNN_AOT_DISABLE_SEED_KV &&
unset GGML_QNN_AOT_EVICT_STATELESS_STAGE_GRAPHS &&
unset GGML_QNN_AOT_TRACE_LOAD_TIMING &&
unset GGML_QNN_AOT_TRACE_MATCH &&
unset GGML_QNN_AOT_TRACE_BIND &&
unset GGML_QNN_AOT_TRACE_ASSIGN &&
unset GGML_QNN_AOT_FG_TRACE &&
unset GGML_QNN_AOT_ALLOW_JIT_FALLBACK &&"
    fi

    if [[ "${BACKEND_POLICY}" == "fine_grained_qnn_gpu" ]]; then
        fg_env_block="export GGML_HETERO_FG_SPLIT=1 &&
export GGML_HETERO_FG_SYNC_TRACE=1 &&
export GGML_HETERO_FG_MAX_LAYERS=$(remote_quote "${FG_MAX_LAYERS}") &&"
    else
        fg_env_block="unset GGML_HETERO_FG_SPLIT &&
unset GGML_HETERO_FG_SYNC_TRACE &&
unset GGML_HETERO_FG_MAX_LAYERS &&"
    fi

    if backend_uses_opencl; then
        if [[ "${OPENCL_SIM_BUSY}" != "0" ]]; then
            opencl_env_block="export GGML_OPENCL_KERNEL_TRACE=1 &&
export GGML_OPENCL_KERNEL_TRACE_CSV=${remote_out_q}/opencl_kernel_trace.csv &&
export GGML_OPENCL_SIM_BUSY=1 &&
export GGML_OPENCL_SIM_BUSY_GLOBAL=$(remote_quote "${OPENCL_SIM_BUSY_GLOBAL}") &&
export GGML_OPENCL_SIM_BUSY_ITERS=$(remote_quote "${OPENCL_SIM_BUSY_ITERS}") &&
export GGML_OPENCL_SIM_BUSY_BUFFER_ELEMS=$(remote_quote "${OPENCL_SIM_BUSY_BUFFER_ELEMS}") &&
export GGML_OPENCL_SIM_BUSY_SLEEP_US=$(remote_quote "${OPENCL_SIM_BUSY_SLEEP_US}") &&"
        else
            opencl_env_block="export GGML_OPENCL_KERNEL_TRACE=1 &&
export GGML_OPENCL_KERNEL_TRACE_CSV=${remote_out_q}/opencl_kernel_trace.csv &&
unset GGML_OPENCL_SIM_BUSY &&
unset GGML_OPENCL_SIM_BUSY_ENABLE &&
unset GGML_OPENCL_SIM_BUSY_GLOBAL &&
unset GGML_OPENCL_SIM_BUSY_ITERS &&
unset GGML_OPENCL_SIM_BUSY_BUFFER_ELEMS &&
unset GGML_OPENCL_SIM_BUSY_SLEEP_US &&"
        fi
    else
        opencl_env_block="unset GGML_OPENCL_KERNEL_TRACE &&
unset GGML_OPENCL_KERNEL_TRACE_CSV &&
unset GGML_OPENCL_SIM_BUSY &&
unset GGML_OPENCL_SIM_BUSY_ENABLE &&
unset GGML_OPENCL_SIM_BUSY_GLOBAL &&
unset GGML_OPENCL_SIM_BUSY_ITERS &&
unset GGML_OPENCL_SIM_BUSY_BUFFER_ELEMS &&
unset GGML_OPENCL_SIM_BUSY_SLEEP_US &&"
    fi

    if [[ -n "${FG_ROUTE}" ]]; then
        route_env_block="export GGML_HETERO_PHASE_ROUTE=${route_q} &&"
    else
        route_env_block="unset GGML_HETERO_PHASE_ROUTE &&"
    fi

    if [[ "${LLAMA_BENCH_WARMUP}" == "0" ]]; then
        warmup_arg="--no-warmup"
    else
        warmup_arg=""
    fi

cat > "${COMMAND_FILE}" <<EOF
mkdir -p ${remote_out_q} &&
cd ${remote_bin_q} &&
rm -f cl_profiling.csv cl_stage_profiling.csv cl_trace.json &&
cat > ${remote_out_q}/sample_power.sh <<'FGSPLIT_POWER_SAMPLER'
#!/system/bin/sh
sample_log="\${POWER_SAMPLE_LOG}"
stop_file="\${POWER_STOP_FILE}"
temp_limit_file="\${POWER_TEMP_LIMIT_FILE}"
voltage_path="\${POWER_VOLTAGE_PATH}"
current_path="\${POWER_CURRENT_PATH}"
temp_path="\${POWER_TEMP_PATH}"
gpu_busy_path="\${POWER_GPU_BUSY_PATH}"
gpu_clock_path="\${POWER_GPU_CLOCK_PATH}"
current_scale="\${POWER_CURRENT_SCALE_TO_UA}"
temp_scale="\${POWER_TEMP_SCALE_TO_MC}"
temp_limit_mc="\${POWER_TEMP_LIMIT_MC}"
sample_interval="\${POWER_SAMPLE_INTERVAL_S}"
qnn_workpoint="\${POWER_QNN_WORKPOINT}"

printf 'timestamp_ms,voltage_uv,current_ua,power_mw,temp_raw,temp_c,gpu_busy,gpu_clock_hz,qnn_workpoint\n' > "\${sample_log}"
while [ ! -f "\${stop_file}" ]; do
    timestamp_ms="\$(awk '{ printf "%.0f", \$1 * 1000.0 }' /proc/uptime 2>/dev/null)"
    voltage="\$(cat "\${voltage_path}" 2>/dev/null | tr -d '\r\n[:space:]')"
    current_raw="\$(cat "\${current_path}" 2>/dev/null | tr -d '\r\n[:space:]')"
    temp_raw="\$(cat "\${temp_path}" 2>/dev/null | tr -d '\r\n[:space:]')"
    gpu_busy="\$(cat "\${gpu_busy_path}" 2>/dev/null | tr -d '\r\n')"
    gpu_clock="\$(cat "\${gpu_clock_path}" 2>/dev/null | tr -d '\r\n[:space:]')"

    current_ua=""
    case "\${current_raw}" in
        ''|*[!0-9-]*) ;;
        *) current_ua="\$(( current_raw * current_scale ))" ;;
    esac

    temp_mc=""
    temp_c=""
    case "\${temp_raw}" in
        ''|*[!0-9-]*) ;;
        *)
            temp_mc="\$(( temp_raw * temp_scale ))"
            temp_c="\$(awk -v x="\${temp_mc}" 'BEGIN { printf "%.2f", x / 1000.0 }')"
            if [ -n "\${temp_limit_mc}" ] && [ "\${temp_mc}" -ge "\${temp_limit_mc}" ]; then
                echo 1 > "\${temp_limit_file}"
                pkill -INT llama-bench 2>/dev/null || true
            fi
            ;;
    esac

    power_mw=""
    if [ -n "\${voltage}" ] && [ -n "\${current_ua}" ]; then
        power_mw="\$(awk -v v="\${voltage}" -v i="\${current_ua}" 'BEGIN { if (i < 0) i = -i; printf "%.3f", v * i / 1000000000.0 }')"
    fi

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "\${timestamp_ms}" "\${voltage}" "\${current_ua}" "\${power_mw}" "\${temp_raw}" "\${temp_c}" "\${gpu_busy}" "\${gpu_clock}" "\${qnn_workpoint}" >> "\${sample_log}"
    sleep "\${sample_interval}"
done
FGSPLIT_POWER_SAMPLER
chmod 755 ${remote_out_q}/sample_power.sh &&
rm -f ${remote_out_q}/power_sampler.stop ${remote_out_q}/temperature_limit_hit &&
su -c "POWER_SAMPLE_LOG=${remote_out_q}/power_samples.csv \
POWER_STOP_FILE=${remote_out_q}/power_sampler.stop \
POWER_TEMP_LIMIT_FILE=${remote_out_q}/temperature_limit_hit \
POWER_VOLTAGE_PATH=${voltage_path_q} \
POWER_CURRENT_PATH=${current_path_q} \
POWER_TEMP_PATH=${temp_path_q} \
POWER_GPU_BUSY_PATH=${gpu_busy_path_q} \
POWER_GPU_CLOCK_PATH=${gpu_clock_path_q} \
POWER_CURRENT_SCALE_TO_UA=$(remote_quote "${CURRENT_SCALE_TO_UA}") \
POWER_TEMP_SCALE_TO_MC=$(remote_quote "${TEMP_SCALE_TO_MC}") \
POWER_TEMP_LIMIT_MC=$(remote_quote "${TEMP_LIMIT_MC}") \
POWER_SAMPLE_INTERVAL_S=$(remote_quote "${SAMPLE_INTERVAL_S}") \
POWER_QNN_WORKPOINT=$(remote_quote "${QNN_WORKPOINT}") \
sh ${remote_out_q}/sample_power.sh" &
power_sampler_pid=\$!
export LD_LIBRARY_PATH=${remote_bin_q}:${qnn_model_q}:\$LD_LIBRARY_PATH &&
export ADSP_LIBRARY_PATH=${remote_bin_q}:${qnn_model_q} &&
${qnn_env_block}
${fg_env_block}
${opencl_env_block}
${route_env_block}
export LLAMA_BENCH_FAST_EXIT=1 &&
export LLAMA_BENCH_QNN_PREWARM_DECODE=$(remote_quote "${LLAMA_BENCH_QNN_PREWARM_DECODE}") &&
taskset ${TASKSET_MASK} ./llama-bench -v -o jsonl \\
  -m ${model_q} \\
  -ngl 99 -dev ${devices_q} -t ${LLAMA_THREADS} \\
  -p ${BENCH_PROMPT_TOKENS} -n ${DECODE_TOKENS} -d ${CONTEXT_LEN} -c ${CONTEXT_SIZE} \\
  -b ${BATCH_TOKENS} -ub ${UBATCH_TOKENS} -r ${ROUNDS} \\
  ${warmup_arg} --mmap ${MMAP}
rc=\$?
touch ${remote_out_q}/power_sampler.stop
wait \${power_sampler_pid} >/dev/null 2>&1 || true
if [ -f ${remote_out_q}/temperature_limit_hit ]; then
  echo 'FG_TEMPERATURE_LIMIT_HIT=1'
fi
cp -f cl_profiling.csv cl_stage_profiling.csv cl_trace.json ${remote_out_q}/ 2>/dev/null || true
exit \${rc}
EOF
}

sample_once() {
    local timestamp_ms voltage current_raw current_ua temp_raw temp_mc temp_c power_mw gpu_busy gpu_clock
    timestamp_ms="$(date +%s%3N)"
    voltage="$(read_remote_value "${BATTERY_VOLTAGE_PATH}")"
    current_raw="$(read_remote_value "${BATTERY_CURRENT_PATH}")"
    current_ua="$(normalize_current_to_ua "${current_raw}" "${CURRENT_SCALE_TO_UA}")"
    temp_raw="$(read_remote_value "${TEMP_PATH}")"
    temp_mc="$(normalize_temp_to_mc "${temp_raw}" "${TEMP_SCALE_TO_MC}")"
    temp_c=""
    if [[ -n "${temp_mc}" ]]; then
        temp_c="$(format_mc "${temp_mc}")"
    fi
    power_mw=""
    if [[ -n "${voltage}" && -n "${current_ua}" ]]; then
        power_mw="$(awk -v v="${voltage}" -v i="${current_ua}" 'BEGIN { if (i < 0) i = -i; printf "%.3f", v * i / 1000000000.0 }')"
    fi
    gpu_busy="$(read_remote_value "${GPU_BUSY_PATH}")"
    gpu_clock="$(read_remote_value "${GPU_CLOCK_PATH:-${GPU_CUR_FREQ_PATH}}")"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${timestamp_ms}" "${voltage}" "${current_ua}" "${power_mw}" "${temp_raw}" "${temp_c}" "${gpu_busy}" "${gpu_clock}" "${QNN_WORKPOINT}"
}

sample_while_running() {
    local pid="$1"
    local overheated=0
    printf 'timestamp_ms,voltage_uv,current_ua,power_mw,temp_raw,temp_c,gpu_busy,gpu_clock_hz,qnn_workpoint\n' > "${SAMPLE_LOG}"
    while kill -0 "${pid}" >/dev/null 2>&1; do
        sample_once >> "${SAMPLE_LOG}" || true
        local temp_mc
        temp_mc="$(current_temp_mc)"
        if [[ -n "${temp_mc}" ]] && (( temp_mc >= TEMP_LIMIT_MC )); then
            overheated=1
            log "temperature limit hit: $(format_mc "${temp_mc}") C >= ${TEMP_LIMIT_C} C; stopping llama-bench"
            adb -s "${DEVICE}" shell "pkill -INT llama-bench 2>/dev/null || true" >/dev/null 2>&1 || true
            break
        fi
        sleep "${SAMPLE_INTERVAL_S}"
    done
    if [[ "${overheated}" == "1" ]]; then
        printf 'FG_TEMPERATURE_LIMIT_HIT=1\n' >> "${RAW_LOG}"
    fi
}

run_benchmark() {
    mkdir -p "${OUTPUT_DIR}/raw"
    adb -s "${DEVICE}" shell "rm -rf $(remote_quote "${REMOTE_OUTPUT_DIR}") && mkdir -p $(remote_quote "${REMOTE_OUTPUT_DIR}")" >/dev/null
    adb -s "${DEVICE}" push "${COMMAND_FILE}" "${REMOTE_OUTPUT_DIR}/command.sh" >/dev/null
    adb -s "${DEVICE}" shell "chmod 755 $(remote_quote "${REMOTE_OUTPUT_DIR}/command.sh")" >/dev/null

    log "running remote benchmark"
    adb -s "${DEVICE}" shell "sh $(remote_quote "${REMOTE_OUTPUT_DIR}/command.sh")" > "${RAW_LOG}" 2>&1 &
    local bench_pid=$!

    local exit_code=0
    if [[ "${POWER_SAMPLER}" == "remote" ]]; then
        if wait "${bench_pid}"; then
            exit_code=0
        else
            exit_code=$?
        fi
    else
        sample_while_running "${bench_pid}" &
        local sampler_pid=$!
        if wait "${bench_pid}"; then
            exit_code=0
        else
            exit_code=$?
        fi
        wait "${sampler_pid}" >/dev/null 2>&1 || true
    fi
    printf '\nFG_RUN_EXIT_CODE=%s\n' "${exit_code}" >> "${RAW_LOG}"

    mkdir -p "${OUTPUT_DIR}/remote"
    adb -s "${DEVICE}" pull "${REMOTE_OUTPUT_DIR}/." "${OUTPUT_DIR}/remote/" >/dev/null 2>&1 || true
    if [[ "${POWER_SAMPLER}" == "remote" ]]; then
        if [[ -f "${OUTPUT_DIR}/remote/power_samples.csv" ]]; then
            cp -f "${OUTPUT_DIR}/remote/power_samples.csv" "${SAMPLE_LOG}"
        else
            log "remote power sampler did not produce ${REMOTE_OUTPUT_DIR}/power_samples.csv"
            printf 'timestamp_ms,voltage_uv,current_ua,power_mw,temp_raw,temp_c,gpu_busy,gpu_clock_hz,qnn_workpoint\n' > "${SAMPLE_LOG}"
        fi
    fi
    return "${exit_code}"
}

find_pulled_file() {
    local name="$1"
    find "${OUTPUT_DIR}/remote" -name "${name}" -type f 2>/dev/null | head -n 1
}

refresh_runtime_metadata() {
    local env_file="${OUTPUT_DIR}/env.txt"
    if [[ ! -f "${env_file}" ]]; then
        return 0
    fi

    local tmp_file="${env_file}.tmp"
    awk -v current_scale="${CURRENT_SCALE_TO_UA}" '
        BEGIN { seen_current_scale = 0 }
        /^CURRENT_SCALE_TO_UA=/ {
            print "CURRENT_SCALE_TO_UA=" current_scale
            seen_current_scale = 1
            next
        }
        { print }
        END {
            if (!seen_current_scale) {
                print "CURRENT_SCALE_TO_UA=" current_scale
            }
        }
    ' "${env_file}" > "${tmp_file}"
    mv "${tmp_file}" "${env_file}"
}

read_profile_status() {
    local csv_path="$1"
    python3 - "${csv_path}" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="") as handle:
    rows = list(csv.DictReader(handle))
row = rows[-1] if rows else {}
print(f"{row.get('support_status', '')},{row.get('fallback_used', '')}")
PY
}

write_metadata() {
    mkdir -p "${OUTPUT_DIR}" "$(dirname "${SUMMARY_MD}")" "$(dirname "${RESULTS_CSV}")"
    printf '%s\n' "$0 $*" > "${OUTPUT_DIR}/invocation.txt"
    git -C "${ROOT_DIR}" rev-parse HEAD > "${OUTPUT_DIR}/git_commit.txt" 2>/dev/null || true
    {
        printf 'DEVICE=%s\n' "${DEVICE}"
        printf 'MODEL_PATH=%s\n' "${MODEL_PATH}"
        printf 'MODE=%s\n' "${MODE}"
        printf 'BACKEND_POLICY=%s\n' "${BACKEND_POLICY}"
        printf 'REMOTE_BIN=%s\n' "${REMOTE_BIN}"
        printf 'QNN_AOT_CONFIG=%s\n' "${QNN_AOT_CONFIG}"
        printf 'QNN_AOT_MODEL_DIR=%s\n' "${QNN_AOT_MODEL_DIR}"
        printf 'OUTPUT_DIR=%s\n' "${OUTPUT_DIR}"
        printf 'LAYERS=%s\n' "${LAYERS}"
        printf 'FG_MAX_LAYERS=%s\n' "${FG_MAX_LAYERS}"
        printf 'ROUNDS=%s\n' "${ROUNDS}"
        printf 'CONTEXT_LEN=%s\n' "${CONTEXT_LEN}"
        printf 'PROMPT_TOKENS=%s\n' "${PROMPT_TOKENS}"
        printf 'BENCH_PROMPT_TOKENS=%s\n' "${BENCH_PROMPT_TOKENS}"
        printf 'DECODE_TOKENS=%s\n' "${DECODE_TOKENS}"
        printf 'QNN_WORKPOINT=%s\n' "${QNN_WORKPOINT}"
        printf 'QNN_AOT_EVICT_STATELESS_STAGE_GRAPHS=%s\n' "${QNN_AOT_EVICT_STATELESS_STAGE_GRAPHS}"
        printf 'QNN_AOT_TRACE_LOAD_TIMING=%s\n' "${QNN_AOT_TRACE_LOAD_TIMING}"
        printf 'LLAMA_BENCH_WARMUP=%s\n' "${LLAMA_BENCH_WARMUP}"
        printf 'LLAMA_BENCH_QNN_PREWARM_DECODE=%s\n' "${LLAMA_BENCH_QNN_PREWARM_DECODE}"
        printf 'FGSPLIT_REQUIRE_SUPPORT_OK=%s\n' "${FGSPLIT_REQUIRE_SUPPORT_OK}"
        printf 'CURRENT_SCALE_TO_UA=%s\n' "${CURRENT_SCALE_TO_UA}"
        printf 'GPU_FREQ_MHZ=%s\n' "${GPU_FREQ_MHZ}"
        printf 'OPENCL_SIM_BUSY=%s\n' "${OPENCL_SIM_BUSY}"
        printf 'OPENCL_SIM_BUSY_GLOBAL=%s\n' "${OPENCL_SIM_BUSY_GLOBAL}"
        printf 'OPENCL_SIM_BUSY_ITERS=%s\n' "${OPENCL_SIM_BUSY_ITERS}"
        printf 'OPENCL_SIM_BUSY_BUFFER_ELEMS=%s\n' "${OPENCL_SIM_BUSY_BUFFER_ELEMS}"
        printf 'OPENCL_SIM_BUSY_SLEEP_US=%s\n' "${OPENCL_SIM_BUSY_SLEEP_US}"
        printf 'POWER_SAMPLER=%s\n' "${POWER_SAMPLER}"
        printf 'SAMPLE_INTERVAL_S=%s\n' "${SAMPLE_INTERVAL_S}"
        printf 'TEMP_LIMIT_C=%s\n' "${TEMP_LIMIT_C}"
        printf 'COOLDOWN_TEMP_C=%s\n' "${COOLDOWN_TEMP_C}"
        printf 'FG_ROUTE=%s\n' "${FG_ROUTE}"
        printf 'FG_DEVICES=%s\n' "${FG_DEVICES}"
    } > "${OUTPUT_DIR}/env.txt"

    {
        printf 'DEVICE=%s \\\n' "$(shell_quote "${DEVICE}")"
        printf 'MODEL_PATH=%s \\\n' "$(shell_quote "${MODEL_PATH}")"
        printf 'REMOTE_BIN=%s \\\n' "$(shell_quote "${REMOTE_BIN}")"
        printf 'QNN_AOT_CONFIG=%s \\\n' "$(shell_quote "${QNN_AOT_CONFIG}")"
        printf 'QNN_AOT_MODEL_DIR=%s \\\n' "$(shell_quote "${QNN_AOT_MODEL_DIR}")"
        printf 'OUTPUT_DIR=%s \\\n' "$(shell_quote "${OUTPUT_DIR}")"
        printf 'MODE=%s \\\n' "$(shell_quote "${MODE}")"
        printf 'BACKEND_POLICY=%s \\\n' "$(shell_quote "${BACKEND_POLICY}")"
        printf 'LAYERS=%s \\\n' "$(shell_quote "${LAYERS}")"
        printf 'FG_MAX_LAYERS=%s \\\n' "$(shell_quote "${FG_MAX_LAYERS}")"
        printf 'ROUNDS=%s \\\n' "$(shell_quote "${ROUNDS}")"
        printf 'CONTEXT_LEN=%s \\\n' "$(shell_quote "${CONTEXT_LEN}")"
        printf 'PROMPT_TOKENS=%s \\\n' "$(shell_quote "${PROMPT_TOKENS}")"
        printf 'BENCH_PROMPT_TOKENS=%s \\\n' "$(shell_quote "${BENCH_PROMPT_TOKENS}")"
        printf 'DECODE_TOKENS=%s \\\n' "$(shell_quote "${DECODE_TOKENS}")"
        printf 'CONTEXT_SIZE=%s \\\n' "$(shell_quote "${CONTEXT_SIZE}")"
        printf 'BATCH_TOKENS=%s \\\n' "$(shell_quote "${BATCH_TOKENS}")"
        printf 'UBATCH_TOKENS=%s \\\n' "$(shell_quote "${UBATCH_TOKENS}")"
        printf 'QNN_WORKPOINT=%s \\\n' "$(shell_quote "${QNN_WORKPOINT}")"
        printf 'QNN_AOT_EVICT_STATELESS_STAGE_GRAPHS=%s \\\n' "$(shell_quote "${QNN_AOT_EVICT_STATELESS_STAGE_GRAPHS}")"
        printf 'QNN_AOT_TRACE_LOAD_TIMING=%s \\\n' "$(shell_quote "${QNN_AOT_TRACE_LOAD_TIMING}")"
        printf 'LLAMA_BENCH_WARMUP=%s \\\n' "$(shell_quote "${LLAMA_BENCH_WARMUP}")"
        printf 'LLAMA_BENCH_QNN_PREWARM_DECODE=%s \\\n' "$(shell_quote "${LLAMA_BENCH_QNN_PREWARM_DECODE}")"
        printf 'FGSPLIT_REQUIRE_SUPPORT_OK=%s \\\n' "$(shell_quote "${FGSPLIT_REQUIRE_SUPPORT_OK}")"
        printf 'GPU_FREQ_MHZ=%s \\\n' "$(shell_quote "${GPU_FREQ_MHZ}")"
        printf 'OPENCL_SIM_BUSY=%s \\\n' "$(shell_quote "${OPENCL_SIM_BUSY}")"
        printf 'OPENCL_SIM_BUSY_GLOBAL=%s \\\n' "$(shell_quote "${OPENCL_SIM_BUSY_GLOBAL}")"
        printf 'OPENCL_SIM_BUSY_ITERS=%s \\\n' "$(shell_quote "${OPENCL_SIM_BUSY_ITERS}")"
        printf 'OPENCL_SIM_BUSY_BUFFER_ELEMS=%s \\\n' "$(shell_quote "${OPENCL_SIM_BUSY_BUFFER_ELEMS}")"
        printf 'OPENCL_SIM_BUSY_SLEEP_US=%s \\\n' "$(shell_quote "${OPENCL_SIM_BUSY_SLEEP_US}")"
        printf 'POWER_SAMPLER=%s \\\n' "$(shell_quote "${POWER_SAMPLER}")"
        printf 'SAMPLE_INTERVAL_S=%s \\\n' "$(shell_quote "${SAMPLE_INTERVAL_S}")"
        printf 'TEMP_LIMIT_C=%s \\\n' "$(shell_quote "${TEMP_LIMIT_C}")"
        printf 'COOLDOWN_TEMP_C=%s \\\n' "$(shell_quote "${COOLDOWN_TEMP_C}")"
        printf 'FG_ROUTE=%s \\\n' "$(shell_quote "${FG_ROUTE}")"
        printf 'FG_DEVICES=%s \\\n' "$(shell_quote "${FG_DEVICES}")"
        printf 'bash scripts/run_fgsplit_synthetic.sh\n'
    } > "${LOCAL_COMMAND_FILE}"
}

summarize() {
    local opencl_stage opencl_kernel state_id
    opencl_stage="$(find_pulled_file cl_stage_profiling.csv)"
    opencl_kernel="$(find_pulled_file opencl_kernel_trace.csv)"
    case "${BACKEND_POLICY}" in
        single_gpu_opencl)
            state_id="single_gpu_${GPU_FREQ_MHZ}"
            ;;
        single_qnn_npu)
            state_id="single_qnn_${QNN_WORKPOINT}"
            ;;
        *)
            state_id="fg_qnn_${QNN_WORKPOINT}_gpu_${GPU_FREQ_MHZ}"
            ;;
    esac

    python3 "${ROOT_DIR}/tools/parse_fgsplit_trace.py" \
        --bench-log "${RAW_LOG}" \
        --sample-log "${SAMPLE_LOG}" \
        --opencl-stage-profile "${opencl_stage:-/nonexistent}" \
        --opencl-kernel-trace "${opencl_kernel:-/nonexistent}" \
        --command "${COMMAND_FILE}" \
        --local-command "${LOCAL_COMMAND_FILE}" \
        --output-csv "${RUN_CSV}" \
        --summary-md "${SUMMARY_MD}" \
        --device "${DEVICE}" \
        --model-path "${MODEL_PATH}" \
        --git-commit "$(cat "${OUTPUT_DIR}/git_commit.txt" 2>/dev/null || true)" \
        --output-dir "${OUTPUT_DIR}" \
        --remote-output-dir "${REMOTE_OUTPUT_DIR}" \
        --mode "${MODE}" \
        --backend-policy "${BACKEND_POLICY}" \
        --fg-route "${FG_ROUTE}" \
        --state-id "${state_id}" \
        --workload-type "decode_like" \
        --context-len "${CONTEXT_LEN}" \
        --prompt-tokens "${PROMPT_TOKENS}" \
        --decode-tokens "${DECODE_TOKENS}" \
        --layers "${LAYERS}" \
        --rounds "${ROUNDS}" \
        --semantic-correctness-required "0" \
        --semantic-correctness-status "not_required" \
        --gpu-freq-mhz "${GPU_FREQ_MHZ}" \
        --qnn-workpoint "${QNN_WORKPOINT}" \
        --temp-limit-c "${TEMP_LIMIT_C}" \
        --cooldown-temp-c "${COOLDOWN_TEMP_C}" \
        --raw-log-path "${RAW_LOG}" \
        --sample-path "${SAMPLE_LOG}"

    python3 "${ROOT_DIR}/tools/parse_fgsplit_trace.py" \
        --bench-log "${RAW_LOG}" \
        --sample-log "${SAMPLE_LOG}" \
        --opencl-stage-profile "${opencl_stage:-/nonexistent}" \
        --opencl-kernel-trace "${opencl_kernel:-/nonexistent}" \
        --command "${COMMAND_FILE}" \
        --local-command "${LOCAL_COMMAND_FILE}" \
        --output-csv "${RESULTS_CSV}" \
        --append \
        --device "${DEVICE}" \
        --model-path "${MODEL_PATH}" \
        --git-commit "$(cat "${OUTPUT_DIR}/git_commit.txt" 2>/dev/null || true)" \
        --output-dir "${OUTPUT_DIR}" \
        --remote-output-dir "${REMOTE_OUTPUT_DIR}" \
        --mode "${MODE}" \
        --backend-policy "${BACKEND_POLICY}" \
        --fg-route "${FG_ROUTE}" \
        --state-id "${state_id}" \
        --workload-type "decode_like" \
        --context-len "${CONTEXT_LEN}" \
        --prompt-tokens "${PROMPT_TOKENS}" \
        --decode-tokens "${DECODE_TOKENS}" \
        --layers "${LAYERS}" \
        --rounds "${ROUNDS}" \
        --semantic-correctness-required "0" \
        --semantic-correctness-status "not_required" \
        --gpu-freq-mhz "${GPU_FREQ_MHZ}" \
        --qnn-workpoint "${QNN_WORKPOINT}" \
        --temp-limit-c "${TEMP_LIMIT_C}" \
        --cooldown-temp-c "${COOLDOWN_TEMP_C}" \
        --raw-log-path "${RAW_LOG}" \
        --sample-path "${SAMPLE_LOG}"

    cp "${SUMMARY_MD}" "${OUTPUT_DIR}/summary.md"
}

main() {
    require_inputs
    discover_sysfs_paths
    write_metadata "$@"
    write_remote_command

    refresh_runtime_metadata
    save_display_state
    trap cleanup EXIT
    ensure_screen_on
    save_and_pin_gpu_freq
    wait_for_cooldown

    local run_exit=0
    if run_benchmark; then
        run_exit=0
    else
        run_exit=$?
        log "benchmark exited with ${run_exit}; preserving logs and writing failed/unsupported row"
    fi

    summarize

    local parsed_status parsed_support_status parsed_fallback_used
    parsed_status="$(read_profile_status "${RUN_CSV}")"
    IFS=, read -r parsed_support_status parsed_fallback_used <<< "${parsed_status}"
    if [[ "${FGSPLIT_REQUIRE_SUPPORT_OK}" != "0" ]] &&
       [[ "${parsed_support_status}" != "ok" || "${parsed_fallback_used}" != "0" ]]; then
        log "support gate failed: support_status=${parsed_support_status:-unknown} fallback_used=${parsed_fallback_used:-unknown}"
        run_exit=1
    fi

    log "wrote run csv: ${RUN_CSV}"
    log "updated aggregate csv: ${RESULTS_CSV}"
    log "wrote summary: ${SUMMARY_MD}"
    log "output directory: ${OUTPUT_DIR}"
    return "${run_exit}"
}

main "$@"
