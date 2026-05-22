#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TRANSITION_HEADER='date,model,context_len,effective_prefill_tokens,from_state,to_state,decode_tokens_before_switch,decode_tokens_after_switch,rounds,decision_us,route_apply_us,policy_apply_us,qnn_workpoint_apply_us,gpu_freq_apply_us,sched_reserve_us,kv_handoff_us,graph_rebuild_us,decode_entry_us,total_blocking_us,first_token_gap_us,post_switch_tbt_us,transition_energy_mj,transition_energy_source,switch_success_rate,fallback_count,qnn_aot_cache_size,qnn_aot_context_size,support_status,raw_log_path'
TRANSITION_ROUND_HEADER='round,decision_us,route_apply_us,policy_apply_us,qnn_workpoint_apply_us,gpu_freq_apply_us,sched_reserve_us,kv_handoff_us,graph_rebuild_us,decode_entry_us,total_blocking_us,first_token_gap_us,post_switch_tbt_us,transition_energy_mj,transition_energy_source,switch_success,fallback_used,support_status,raw_log_path,exit_code,measurement_note'

log() {
    printf '[insightB-transition-overhead] %s\n' "$*"
}

die() {
    printf '[insightB-transition-overhead] ERROR: %s\n' "$*" >&2
    exit 1
}

csv_cell() {
    local value="${1:-}"
    value="${value//\"/\"\"}"
    if [[ "${value}" == *","* || "${value}" == *$'\n'* || "${value}" == *$'\r'* || "${value}" == *'"'* ]]; then
        printf '"%s"' "${value}"
    else
        printf '%s' "${value}"
    fi
}

csv_row() {
    local first=1
    local field
    for field in "$@"; do
        if (( first )); then
            first=0
        else
            printf ','
        fi
        csv_cell "${field}"
    done
    printf '\n'
}

ensure_transition_header() {
    local path="$1"
    mkdir -p "$(dirname "${path}")"
    if [[ ! -s "${path}" ]]; then
        printf '%s\n' "${TRANSITION_HEADER}" > "${path}"
        return 0
    fi

    local current_header
    current_header="$(sed -n '1p' "${path}")"
    [[ "${current_header}" == "${TRANSITION_HEADER}" ]] || die "unexpected CSV header in ${path}"
}

ensure_transition_round_header() {
    local path="$1"
    mkdir -p "$(dirname "${path}")"
    if [[ ! -s "${path}" ]]; then
        printf '%s\n' "${TRANSITION_ROUND_HEADER}" > "${path}"
        return 0
    fi

    local current_header
    current_header="$(sed -n '1p' "${path}")"
    [[ "${current_header}" == "${TRANSITION_ROUND_HEADER}" ]] || die "unexpected round CSV header in ${path}"
}

is_number() {
    [[ "${1:-}" =~ ^-?[0-9]+([.][0-9]+)?$ ]]
}

mean_field() {
    local values="$1"
    awk -v values="${values}" '
        BEGIN {
            n = split(values, a, /[[:space:]]+/)
            for (i = 1; i <= n; ++i) {
                if (a[i] != "" && a[i] ~ /^-?[0-9]+([.][0-9]+)?$/) {
                    sum += a[i]
                    count++
                }
            }
            if (count > 0) {
                printf "%.2f", sum / count
            }
        }'
}

state_backend() {
    case "$1" in
        npu_low_balanced|npu_burst)
            printf 'qnn-npu'
            ;;
        gpu_734|gpu_967|gpu_1100)
            printf 'GPUOpenCL'
            ;;
        cpu_big2_2649)
            printf 'CPU'
            ;;
        *)
            return 1
            ;;
    esac
}

state_route() {
    case "$1" in
        npu_low_balanced|npu_burst)
            printf 'qnn-npu'
            ;;
        gpu_734|gpu_967|gpu_1100)
            printf 'opencl'
            ;;
        cpu_big2_2649)
            printf 'cpu'
            ;;
        *)
            return 1
            ;;
    esac
}

state_gpu_freq_hz() {
    case "$1" in
        gpu_734) printf '734000000' ;;
        gpu_967) printf '967000000' ;;
        gpu_1100) printf '1100000000' ;;
        *) return 1 ;;
    esac
}

state_npu_workpoint() {
    case "$1" in
        npu_low_balanced) printf 'low_balanced' ;;
        npu_burst) printf 'burst' ;;
        *) return 1 ;;
    esac
}

is_cpu_state() {
    [[ "$1" == cpu_big2_2649 ]]
}

state_cpu_taskset_mask() {
    case "$1" in
        cpu_big2_2649) printf 'C0' ;;
        *) return 1 ;;
    esac
}

state_cpu_threads() {
    case "$1" in
        cpu_big2_2649) printf '2' ;;
        *) return 1 ;;
    esac
}

state_cpu_freq_khz() {
    case "$1" in
        cpu_big2_2649) printf '2649600' ;;
        *) return 1 ;;
    esac
}

same_route_control_kind() {
    local from_state="$1"
    local to_state="$2"

    if [[ "${from_state}" == "${to_state}" ]]; then
        printf 'none'
        return 0
    fi

    local from_route to_route
    from_route="$(state_route "${from_state}")" || return 1
    to_route="$(state_route "${to_state}")" || return 1

    if [[ "${from_route}" != "${to_route}" ]]; then
        printf 'none'
        return 0
    fi

    case "${from_route}" in
        opencl) printf 'gpu_freq' ;;
        qnn-npu) printf 'qnn_workpoint' ;;
        *) printf 'unsupported' ;;
    esac
}

gpu_control_floor_freq() {
    local from_state="$1"
    local to_state="$2"
    local from_freq to_freq

    from_freq="$(state_gpu_freq_hz "${from_state}")" || return 1
    to_freq="$(state_gpu_freq_hz "${to_state}")" || return 1

    if (( from_freq < to_freq )); then
        printf '%s' "${from_freq}"
    else
        printf '%s' "${to_freq}"
    fi
}

is_qnn_state() {
    [[ "$1" == npu_* ]]
}

transition_uses_cpu_state() {
    is_cpu_state "$1" || is_cpu_state "$2"
}

transition_slug() {
    local from_state="$1"
    local to_state="$2"
    printf '%s_to_%s' "${from_state}" "${to_state}"
}

parse_transition_spec() {
    local spec="$1"
    spec="${spec//->/ }"
    spec="${spec//:/ }"
    awk '{ if (NF == 2) printf "%s,%s", $1, $2; else exit 1 }' <<< "${spec}"
}

extract_kv_value() {
    local line="$1"
    local key="$2"
    awk -v key="${key}" '
        {
            for (i = 1; i <= NF; ++i) {
                split($i, kv, "=")
                if (kv[1] == key) {
                    print substr($i, length(key) + 2)
                    exit
                }
            }
        }' <<< "${line}"
}

parse_transition_timing_line() {
    local line="$1"
    local decision_us route_apply_us sched_reserve_us kv_handoff_us graph_rebuild_us total_blocking_us route_applied
    decision_us="$(extract_kv_value "${line}" "decide_us")"
    route_apply_us="$(extract_kv_value "${line}" "apply_us")"
    sched_reserve_us="$(extract_kv_value "${line}" "reserve_us")"
    kv_handoff_us="$(extract_kv_value "${line}" "kv_migration_us")"
    graph_rebuild_us="$(extract_kv_value "${line}" "bootstrap_sched_rebuild_us")"
    total_blocking_us="$(extract_kv_value "${line}" "total_wall_us")"
    route_applied="$(extract_kv_value "${line}" "route_applied")"

    local success=0
    [[ "${route_applied}" == "true" ]] && success=1

    printf '%s,%s,%s,%s,%s,%s,%s' \
        "${decision_us}" \
        "${route_apply_us}" \
        "${sched_reserve_us}" \
        "${kv_handoff_us}" \
        "${graph_rebuild_us}" \
        "${total_blocking_us}" \
        "${success}"
}

parse_transition_trace_line() {
    local line="$1"
    local decision_us route_apply_us policy_apply_us qnn_workpoint_apply_us gpu_freq_apply_us
    local sched_reserve_us kv_handoff_us graph_rebuild_us decode_entry_us total_blocking_us
    local first_token_gap_us post_switch_tbt_us transition_energy_mj transition_energy_source success fallback support_status

    decision_us="$(extract_kv_value "${line}" "decision_us")"
    route_apply_us="$(extract_kv_value "${line}" "route_apply_us")"
    policy_apply_us="$(extract_kv_value "${line}" "policy_apply_us")"
    qnn_workpoint_apply_us="$(extract_kv_value "${line}" "qnn_workpoint_apply_us")"
    gpu_freq_apply_us="$(extract_kv_value "${line}" "gpu_freq_apply_us")"
    sched_reserve_us="$(extract_kv_value "${line}" "sched_reserve_us")"
    kv_handoff_us="$(extract_kv_value "${line}" "kv_handoff_us")"
    graph_rebuild_us="$(extract_kv_value "${line}" "graph_rebuild_us")"
    decode_entry_us="$(extract_kv_value "${line}" "decode_entry_us")"
    total_blocking_us="$(extract_kv_value "${line}" "total_blocking_us")"
    first_token_gap_us="$(extract_kv_value "${line}" "first_token_gap_us")"
    post_switch_tbt_us="$(extract_kv_value "${line}" "post_switch_tbt_us")"
    transition_energy_mj="$(extract_kv_value "${line}" "transition_energy_mj")"
    transition_energy_source="$(extract_kv_value "${line}" "transition_energy_source")"
    success="$(extract_kv_value "${line}" "success")"
    fallback="$(extract_kv_value "${line}" "fallback")"
    support_status="$(extract_kv_value "${line}" "support_status")"

    [[ -n "${transition_energy_source}" ]] || transition_energy_source="unavailable"
    [[ -n "${success}" ]] || success=0
    [[ -n "${fallback}" ]] || fallback=0
    [[ -n "${support_status}" ]] || support_status="ok"

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s' \
        "${decision_us}" \
        "${route_apply_us}" \
        "${policy_apply_us}" \
        "${qnn_workpoint_apply_us}" \
        "${gpu_freq_apply_us}" \
        "${sched_reserve_us}" \
        "${kv_handoff_us}" \
        "${graph_rebuild_us}" \
        "${decode_entry_us}" \
        "${total_blocking_us}" \
        "${first_token_gap_us}" \
        "${post_switch_tbt_us}" \
        "${transition_energy_mj}" \
        "${transition_energy_source}" \
        "${success}" \
        "${fallback}" \
        "${support_status}"
}

derive_token_timing_from_log() {
    local raw_log="$1"

    awk '
        function kv_value(line, key,    n, fields, i, pos, item) {
            n = split(line, fields, /[[:space:]]+/)
            for (i = 1; i <= n; ++i) {
                item = fields[i]
                pos = index(item, "=")
                if (pos > 0 && substr(item, 1, pos - 1) == key) {
                    return substr(item, pos + 1)
                }
            }
            return ""
        }
        function is_uint(x) {
            return x ~ /^[0-9]+$/
        }
        /TRANSITION_TRACE/ {
            seen_transition = 1
            post_count = 0
            post_sum = 0
            post_done_count = 0
            delete post_wall
            delete post_done
            next
        }
        /DECODE_TOKEN_TRACE/ {
            done = kv_value($0, "done_us")
            total_wall = kv_value($0, "total_wall_us")
            route_applied = kv_value($0, "route_applied")
            if (!seen_transition) {
                next
            }
            if (route_applied == "1") {
                next
            }
            if (is_uint(total_wall)) {
                post_count++
                post_wall[post_count] = total_wall + 0
                post_sum += post_wall[post_count]
            } else if (is_uint(done)) {
                post_done_count++
                post_done[post_done_count] = done + 0
            }
        }
        END {
            first_gap = ""
            post_tbt = ""
            if (post_count > 0) {
                for (i = 1; i <= post_count; ++i) {
                    sorted[i] = post_wall[i]
                }
                for (i = 1; i <= post_count; ++i) {
                    for (j = i + 1; j <= post_count; ++j) {
                        if (sorted[j] < sorted[i]) {
                            tmp = sorted[i]
                            sorted[i] = sorted[j]
                            sorted[j] = tmp
                        }
                    }
                }
                mid = int((post_count + 1) / 2)
                if (post_count % 2 == 0) {
                    median = (sorted[mid] + sorted[mid + 1]) / 2.0
                } else {
                    median = sorted[mid]
                }

                filtered_sum = 0
                filtered_count = 0
                for (i = 1; i <= post_count; ++i) {
                    if (post_count >= 3 && median > 0 && post_wall[i] > median * 5.0) {
                        continue
                    }
                    filtered_sum += post_wall[i]
                    filtered_count++
                }
                if (filtered_count > 0) {
                    post_tbt = sprintf("%.2f", filtered_sum / filtered_count)
                } else {
                    post_tbt = sprintf("%.2f", post_sum / post_count)
                }
            } else if (post_done_count >= 2) {
                for (i = 2; i <= post_done_count; ++i) {
                    if (post_done[i] >= post_done[i - 1]) {
                        sum += post_done[i] - post_done[i - 1]
                        intervals++
                    }
                }
                if (intervals > 0) {
                    post_tbt = sprintf("%.2f", sum / intervals)
                }
            }

            printf "%s,%s", first_gap, post_tbt
        }' "${raw_log}"
}

detect_fallback_used() {
    local raw_log="$1"
    if [[ -s "${raw_log}" ]] && \
        grep -Eiv 'fallback to libQnn|fallback .*disabled|host fallback can corrupt' "${raw_log}" | \
        grep -Eiq 'fallback_used=1|runtime fallback|falling back|fallback to (CPU|GPU|OpenCL|qnn-npu)|replay fallback'; then
        printf '1'
    else
        printf '0'
    fi
}

classify_transition_support_status() {
    local exit_code="$1"
    local switch_success="$2"
    local raw_log="$3"

    if [[ "${exit_code}" == "0" && "${switch_success}" == "1" ]]; then
        printf 'ok'
        return 0
    fi

    if [[ -s "${raw_log}" ]]; then
        if grep -Eiq 'qnn_aot_cache_size|aot.*cache.*exceed|cache_size.*exceed|exceed.*cache_size|KV cache.*exceed|exceed.*KV cache|KV.*footprint.*exceed|footprint.*exceed.*KV|not enough.*KV|requested.*KV.*greater' "${raw_log}"; then
            printf 'unsupported_by_current_aot_cache_size'
            return 0
        fi
        if grep -Eiq 'KV.*(contract|layout|dtype|handoff).*(incompat|unsupported|reject|fail|error)|incompat.*KV|unsupported.*KV|handoff.*(fail|unsupported|reject)' "${raw_log}"; then
            printf 'unsupported_by_kv_contract'
            return 0
        fi
        if grep -Eiq 'invalid device|failed to create context|failed to initialize|init.*failed|cannot initialize|runtime.*reject|backend .* was not initialized' "${raw_log}"; then
            printf 'unsupported_by_runtime'
            return 0
        fi
    fi

    printf 'failed'
}

parse_qnn_aot_sizes_from_file() {
    local config_path="$1"
    local relevant_batches="${2:-1}"

    python3 - "${config_path}" "${relevant_batches}" <<'PY'
import json
import sys

path = sys.argv[1]
target_batches = {
    int(item)
    for item in sys.argv[2].replace(",", " ").split()
    if item.strip().isdigit()
}

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

graphs = [
    g for g in data.get("graphs", [])
    if g.get("type") == "transformers"
    and "cache_size" in g
    and "context_size" in g
]
selected = [
    g for g in graphs
    if not target_batches or int(g.get("batch_size", -1)) in target_batches
]
if not selected:
    selected = graphs

if not selected:
    sys.exit("no transformer graph cache/context sizes found")

cache_size = min(int(g["cache_size"]) for g in selected)
context_size = min(int(g["context_size"]) for g in selected)
print(f"{cache_size},{context_size}")
PY
}

copy_or_pull_qnn_config() {
    local config_path="$1"
    local destination="$2"

    if [[ -f "${config_path}" ]]; then
        cp "${config_path}" "${destination}"
        return 0
    fi

    [[ -n "${DEVICE:-}" ]] || return 1
    adb -s "${DEVICE}" shell "cat ${config_path} 2>/dev/null" | tr -d '\r' > "${destination}"
    [[ -s "${destination}" ]]
}

transition_qnn_support_status() {
    local from_state="$1"
    local to_state="$2"
    local effective_prefill_tokens="$3"
    local decode_tokens_before_switch="$4"
    local decode_tokens_after_switch="$5"
    local safety_margin="$6"
    local cache_size="$7"

    local needs_qnn=0
    local max_footprint=""
    if is_qnn_state "${from_state}"; then
        needs_qnn=1
        max_footprint=$(( effective_prefill_tokens + decode_tokens_before_switch + safety_margin ))
    fi
    if is_qnn_state "${to_state}"; then
        needs_qnn=1
        local to_footprint=$(( effective_prefill_tokens + decode_tokens_before_switch + decode_tokens_after_switch + safety_margin ))
        if [[ -z "${max_footprint}" || "${to_footprint}" -gt "${max_footprint}" ]]; then
            max_footprint="${to_footprint}"
        fi
    fi

    if (( ! needs_qnn )); then
        printf 'ok,'
        return 0
    fi

    if [[ -z "${cache_size}" || ! "${cache_size}" =~ ^[0-9]+$ ]]; then
        printf 'unsupported_by_runtime,%s' "${max_footprint}"
        return 0
    fi

    if (( max_footprint > cache_size )); then
        printf 'unsupported_by_current_aot_cache_size,%s' "${max_footprint}"
    else
        printf 'ok,%s' "${max_footprint}"
    fi
}

aggregate_transition_rounds() {
    local rounds_csv="$1"

    awk -F, '
        function isnum(x) { return x ~ /^-?[0-9]+([.][0-9]+)?$/ }
        function add(col) {
            if (isnum($col)) {
                sum[col] += $col
                count[col]++
            }
        }
        NR > 1 {
            rows++
            for (col = 2; col <= 14; ++col) {
                add(col)
            }
            if ($15 != "") {
                energy_source[$15]++
            }
            if ($16 ~ /^[0-9.]+$/) {
                success_sum += $16
                success_count++
            }
            if ($17 ~ /^[0-9]+$/) {
                fallback_count += $17
            }
            status_count[$18]++
            if ($18 != "ok") {
                non_ok_count++
                if (first_non_ok == "") {
                    first_non_ok = $18
                }
            }
            if ($19 != "") {
                raw_paths = raw_paths (raw_paths == "" ? "" : ";") $19
            }
        }
        END {
            for (col = 2; col <= 14; ++col) {
                out[col] = count[col] > 0 ? sprintf("%.2f", sum[col] / count[col]) : ""
            }
            source = "unavailable"
            if (length(energy_source) == 1) {
                for (key in energy_source) {
                    source = key
                }
            } else if (energy_source["measured"] > 0) {
                source = "measured"
            } else if (energy_source["estimated"] > 0) {
                source = "estimated"
            }
            success_rate = success_count > 0 ? sprintf("%.6f", success_sum / success_count) : ""
            support = "failed"
            if (rows > 0 && non_ok_count == 0) {
                support = "ok"
            } else if (rows > 0 && non_ok_count == rows && first_non_ok != "") {
                support = first_non_ok
            }
            printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%d,%s,%s", \
                out[2], out[3], out[4], out[5], out[6], out[7], out[8], out[9], out[10], \
                out[11], out[12], out[13], out[14], source, success_rate, fallback_count, support, raw_paths
        }' "${rounds_csv}"
}

append_transition_row() {
    local run_csv="$1"
    local global_csv="$2"
    shift 2

    ensure_transition_header "${run_csv}"
    ensure_transition_header "${global_csv}"
    csv_row "$@" >> "${run_csv}"
    csv_row "$@" >> "${global_csv}"
}

append_round_row() {
    local rounds_csv="$1"
    shift

    ensure_transition_round_header "${rounds_csv}"
    csv_row "$@" >> "${rounds_csv}"
}

write_command_file() {
    local path="$1"
    shift
    {
        printf 'cd %q\n' "${ROOT_DIR}"
        printf '%q ' "$@"
        printf '\n'
    } > "${path}"
}

shell_quote() {
    sed "s/'/'\\\\''/g; 1s/^/'/; \$s/\$/'/" <<< "$1"
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

save_display_state() {
    ORIG_SCREEN_OFF_TIMEOUT="$(read_android_setting system screen_off_timeout)"
    ORIG_SCREEN_BRIGHTNESS="$(read_android_setting system screen_brightness)"
    ORIG_SCREEN_BRIGHTNESS_MODE="$(read_android_setting system screen_brightness_mode)"
    ORIG_STAY_ON_WHILE_PLUGGED_IN="$(read_android_setting global stay_on_while_plugged_in)"
    DISPLAY_STATE_SAVED=1
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

restore_display_state() {
    if [[ "${DISPLAY_STATE_SAVED:-0}" != "1" ]]; then
        return 0
    fi
    restore_android_setting system screen_off_timeout "${ORIG_SCREEN_OFF_TIMEOUT:-}"
    restore_android_setting system screen_brightness "${ORIG_SCREEN_BRIGHTNESS:-}"
    restore_android_setting system screen_brightness_mode "${ORIG_SCREEN_BRIGHTNESS_MODE:-}"
    restore_android_setting global stay_on_while_plugged_in "${ORIG_STAY_ON_WHILE_PLUGGED_IN:-}"
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

transition_list_needs_gpu_control() {
    local spec parsed from_state to_state
    for spec in ${TRANSITION_LIST:-}; do
        parsed="$(parse_transition_spec "${spec}")" || continue
        IFS=',' read -r from_state to_state <<< "${parsed}"
        if [[ "$(same_route_control_kind "${from_state}" "${to_state}")" == "gpu_freq" ]]; then
            return 0
        fi
    done
    return 1
}

discover_gpu_control_paths() {
    GPU_AVAILABLE_FREQ_PATH="$(find_first_readable_path \
        "${GPU_AVAILABLE_FREQ_PATH:-}" \
        "/sys/class/kgsl/kgsl-3d0/devfreq/available_frequencies" \
        "/sys/class/kgsl/kgsl-3d0/gpu_available_frequencies" \
        "/sys/class/devfreq/kgsl-3d0/available_frequencies")" || true

    GPU_MIN_FREQ_PATH="$(find_first_readable_path \
        "${GPU_MIN_FREQ_PATH:-}" \
        "/sys/class/kgsl/kgsl-3d0/devfreq/min_freq" \
        "/sys/class/devfreq/kgsl-3d0/min_freq")" || true

    GPU_MAX_FREQ_PATH="$(find_first_readable_path \
        "${GPU_MAX_FREQ_PATH:-}" \
        "/sys/class/kgsl/kgsl-3d0/devfreq/max_freq" \
        "/sys/class/devfreq/kgsl-3d0/max_freq")" || true

    GPU_CUR_FREQ_PATH="$(find_first_readable_path \
        "${GPU_CUR_FREQ_PATH:-}" \
        "/sys/class/kgsl/kgsl-3d0/devfreq/cur_freq" \
        "/sys/class/kgsl/kgsl-3d0/gpuclk" \
        "/sys/class/devfreq/kgsl-3d0/cur_freq")" || true

    GPU_GOVERNOR_PATH="$(find_first_readable_path \
        "${GPU_GOVERNOR_PATH:-}" \
        "/sys/class/kgsl/kgsl-3d0/devfreq/governor" \
        "/sys/class/devfreq/kgsl-3d0/governor")" || true

    if [[ -z "${GPU_MIN_FREQ_PATH}" || -z "${GPU_MAX_FREQ_PATH}" ]]; then
        GPU_CONTROL_AVAILABLE=0
        log "GPU control sysfs paths unavailable; GPU frequency-only transitions will be marked unsupported_by_runtime"
        return 0
    fi

    GPU_CONTROL_AVAILABLE=1
    log "GPU control paths: min=${GPU_MIN_FREQ_PATH} max=${GPU_MAX_FREQ_PATH} cur=${GPU_CUR_FREQ_PATH:-unavailable}"
}

save_original_gpu_control_state() {
    if [[ "${GPU_CONTROL_AVAILABLE:-0}" != "1" || "${GPU_CONTROL_STATE_SAVED:-0}" == "1" ]]; then
        return 0
    fi

    ORIG_GPU_MIN_FREQ="$(read_remote_value "${GPU_MIN_FREQ_PATH}")"
    ORIG_GPU_MAX_FREQ="$(read_remote_value "${GPU_MAX_FREQ_PATH}")"
    if [[ -n "${GPU_GOVERNOR_PATH:-}" ]]; then
        ORIG_GPU_GOVERNOR="$(read_remote_value "${GPU_GOVERNOR_PATH}")"
    fi
    GPU_CONTROL_STATE_SAVED=1
}

restore_gpu_control_state() {
    if [[ "${GPU_CONTROL_STATE_SAVED:-0}" != "1" ]]; then
        return 0
    fi

    if [[ -n "${ORIG_GPU_MIN_FREQ:-}" && -n "${ORIG_GPU_MAX_FREQ:-}" && -n "${GPU_MIN_FREQ_PATH:-}" && -n "${GPU_MAX_FREQ_PATH:-}" ]]; then
        adb_root_shell "echo ${ORIG_GPU_MIN_FREQ} > ${GPU_MIN_FREQ_PATH} && echo ${ORIG_GPU_MAX_FREQ} > ${GPU_MAX_FREQ_PATH}" || true
    fi
    if [[ -n "${GPU_GOVERNOR_PATH:-}" && -n "${ORIG_GPU_GOVERNOR:-}" ]]; then
        adb_root_shell "echo ${ORIG_GPU_GOVERNOR} > ${GPU_GOVERNOR_PATH}" || true
    fi
}

cpu_mask_to_list() {
    local mask="$1"
    mask="${mask#0x}"
    mask="${mask#0X}"
    [[ "${mask}" =~ ^[0-9a-fA-F]+$ ]] || return 1

    local value=$(( 16#${mask} ))
    local cpu=0
    local out=""
    while (( value > 0 )); do
        if (( value & 1 )); then
            out="${out}${out:+ }${cpu}"
        fi
        value=$(( value >> 1 ))
        cpu=$(( cpu + 1 ))
    done
    [[ -n "${out}" ]] || return 1
    printf '%s\n' "${out}"
}

find_cpu_policy_for_cpu() {
    local cpu_id="$1"
    adb_root_capture "
        for p in /sys/devices/system/cpu/cpufreq/policy*; do
            [ -d \"\$p\" ] || continue
            if [ -r \"\$p/related_cpus\" ] && grep -qw ${cpu_id} \"\$p/related_cpus\"; then
                echo \"\$p\"
                exit 0
            fi
        done
        if [ -e /sys/devices/system/cpu/cpu${cpu_id}/cpufreq ]; then
            readlink -f /sys/devices/system/cpu/cpu${cpu_id}/cpufreq 2>/dev/null || true
        fi
    " | sed -n '1p'
}

cpu_policy_list_for_state() {
    local state="$1"
    local mask cpus cpu policy
    local seen=""
    local out=""

    mask="$(state_cpu_taskset_mask "${state}")" || return 1
    cpus="$(cpu_mask_to_list "${mask}")" || return 1

    for cpu in ${cpus}; do
        policy="$(find_cpu_policy_for_cpu "${cpu}")"
        [[ -n "${policy}" ]] || return 1
        if [[ " ${seen} " != *" ${policy} "* ]]; then
            seen="${seen} ${policy}"
            out="${out}${out:+ }${policy}"
        fi
    done

    [[ -n "${out}" ]] || return 1
    printf '%s\n' "${out}"
}

select_cpu_policy_freq() {
    local policy="$1"
    local requested="$2"
    local raw

    raw="$(adb_root_capture "cat ${policy}/scaling_available_frequencies 2>/dev/null || true")"
    if [[ -z "${raw}" ]]; then
        printf '%s\n' "${requested}"
        return 0
    fi

    awk -v requested="${requested}" '
        {
            for (i = 1; i <= NF; ++i) {
                if ($i ~ /^[0-9]+$/) {
                    diff = $i - requested
                    if (diff < 0) diff = -diff
                    if (!seen || diff < best_diff) {
                        seen = 1
                        best = $i
                        best_diff = diff
                    }
                }
            }
        }
        END {
            if (seen) print best
        }' <<< "${raw}"
}

save_cpu_policy_state() {
    local policy="$1"
    if [[ -n "${ORIG_CPU_STATE_SAVED_BY_POLICY[${policy}]+x}" ]]; then
        return 0
    fi

    ORIG_CPU_MIN_FREQ["${policy}"]="$(read_remote_value "${policy}/scaling_min_freq")"
    ORIG_CPU_MAX_FREQ["${policy}"]="$(read_remote_value "${policy}/scaling_max_freq")"
    ORIG_CPU_GOVERNOR["${policy}"]="$(read_remote_value "${policy}/scaling_governor")"
    ORIG_CPU_STATE_SAVED_BY_POLICY["${policy}"]=1
}

restore_cpu_control_state() {
    local policy
    for policy in "${!ORIG_CPU_STATE_SAVED_BY_POLICY[@]}"; do
        if [[ -n "${ORIG_CPU_MIN_FREQ[${policy}]:-}" && -n "${ORIG_CPU_MAX_FREQ[${policy}]:-}" ]]; then
            adb_root_shell "echo ${ORIG_CPU_MIN_FREQ[${policy}]} > ${policy}/scaling_min_freq && echo ${ORIG_CPU_MAX_FREQ[${policy}]} > ${policy}/scaling_max_freq" || true
        fi
        if [[ -n "${ORIG_CPU_GOVERNOR[${policy}]:-}" ]]; then
            adb_root_shell "echo ${ORIG_CPU_GOVERNOR[${policy}]} > ${policy}/scaling_governor" || true
        fi
    done
}

pin_cpu_policy_freq() {
    local policy="$1"
    local requested="$2"
    local selected cpuinfo_min readback_gov readback_min readback_max readback_cur

    selected="$(select_cpu_policy_freq "${policy}" "${requested}")"
    [[ -n "${selected}" ]] || return 1

    save_cpu_policy_state "${policy}"

    cpuinfo_min="$(read_remote_value "${policy}/cpuinfo_min_freq")"
    [[ -n "${cpuinfo_min}" ]] || cpuinfo_min="${selected}"

    adb_root_shell "echo ${CPU_PIN_GOVERNOR} > ${policy}/scaling_governor" || return 1
    adb_root_shell "echo ${cpuinfo_min} > ${policy}/scaling_min_freq" || true
    adb_root_shell "echo ${selected} > ${policy}/scaling_max_freq" || return 1
    adb_root_shell "echo ${selected} > ${policy}/scaling_min_freq" || return 1
    sleep 1

    readback_gov="$(read_remote_value "${policy}/scaling_governor")"
    readback_min="$(read_remote_value "${policy}/scaling_min_freq")"
    readback_max="$(read_remote_value "${policy}/scaling_max_freq")"
    readback_cur="$(read_remote_value "${policy}/scaling_cur_freq")"
    if [[ "${readback_gov}" != "${CPU_PIN_GOVERNOR}" || "${readback_max}" != "${selected}" || "${readback_cur}" != "${selected}" ]]; then
        printf 'failed to pin CPU policy %s: requested=%s selected=%s readback_gov=%s readback_min=%s readback_max=%s readback_cur=%s\n' \
            "${policy}" \
            "${requested}" \
            "${selected}" \
            "${readback_gov:-NA}" \
            "${readback_min:-NA}" \
            "${readback_max:-NA}" \
            "${readback_cur:-NA}" >&2
        return 1
    fi

    PIN_CPU_POLICY_RESULT="${policy}=${selected}"
}

pin_cpu_state_for_transition() {
    local from_state="$1"
    local to_state="$2"
    local cpu_state=""
    local requested policies policy pinned=""

    if is_cpu_state "${from_state}"; then
        cpu_state="${from_state}"
    fi
    if is_cpu_state "${to_state}"; then
        if [[ -n "${cpu_state}" && "${cpu_state}" != "${to_state}" ]]; then
            die "multiple CPU states in one transition are not supported: ${from_state}->${to_state}"
        fi
        cpu_state="${to_state}"
    fi

    [[ -n "${cpu_state}" ]] || return 0
    requested="$(state_cpu_freq_khz "${cpu_state}")" || return 1
    policies="$(cpu_policy_list_for_state "${cpu_state}")" || die "failed to resolve cpufreq policy for ${cpu_state}"

    for policy in ${policies}; do
        local selected
        PIN_CPU_POLICY_RESULT=""
        pin_cpu_policy_freq "${policy}" "${requested}" || die "failed to pin ${policy} for ${cpu_state}"
        selected="${PIN_CPU_POLICY_RESULT}"
        pinned="${pinned}${pinned:+ }${selected}"
    done

    log "CPU state ${cpu_state}: pinned ${pinned} taskset=$(state_cpu_taskset_mask "${cpu_state}") threads=$(state_cpu_threads "${cpu_state}")"
}

prepare_gpu_control_if_needed() {
    if ! transition_list_needs_gpu_control; then
        return 0
    fi

    discover_gpu_control_paths
    save_original_gpu_control_state
}

pin_gpu_freq_for_control_transition() {
    local from_state="$1"
    local to_state="$2"
    local pin_state="$3"
    local pin_freq floor_freq

    [[ "${GPU_CONTROL_AVAILABLE:-0}" == "1" ]] || return 1
    pin_freq="$(state_gpu_freq_hz "${pin_state}")" || return 1
    floor_freq="$(gpu_control_floor_freq "${from_state}" "${to_state}")" || return 1

    local command=""
    if [[ -n "${GPU_GOVERNOR_PATH:-}" && -n "${GPU_PIN_GOVERNOR:-}" ]]; then
        command="echo ${GPU_PIN_GOVERNOR} > ${GPU_GOVERNOR_PATH} && "
    fi
    command="${command}echo ${floor_freq} > ${GPU_MIN_FREQ_PATH} && echo ${pin_freq} > ${GPU_MAX_FREQ_PATH} && echo ${pin_freq} > ${GPU_MIN_FREQ_PATH}"
    adb_root_shell "${command}"
}

build_gpu_control_apply_command() {
    local from_state="$1"
    local to_state="$2"
    local target_freq floor_freq

    target_freq="$(state_gpu_freq_hz "${to_state}")" || return 1
    floor_freq="$(gpu_control_floor_freq "${from_state}" "${to_state}")" || return 1
    [[ -n "${GPU_MIN_FREQ_PATH:-}" && -n "${GPU_MAX_FREQ_PATH:-}" ]] || return 1

    printf 'now_ns() { date +%%s%%N; }; '
    printf 'start=$(now_ns); success=1; '
    if [[ -n "${GPU_GOVERNOR_PATH:-}" && -n "${GPU_PIN_GOVERNOR:-}" ]]; then
        printf 'echo %s > %s || success=0; ' "${GPU_PIN_GOVERNOR}" "${GPU_GOVERNOR_PATH}"
    fi
    printf 'echo %s > %s || success=0; ' "${floor_freq}" "${GPU_MIN_FREQ_PATH}"
    printf 'echo %s > %s || success=0; ' "${target_freq}" "${GPU_MAX_FREQ_PATH}"
    printf 'echo %s > %s || success=0; ' "${target_freq}" "${GPU_MIN_FREQ_PATH}"
    printf 'end=$(now_ns); delta=$(((end - start + 999) / 1000)); '
    if [[ -n "${GPU_CUR_FREQ_PATH:-}" ]]; then
        printf 'actual=$(cat %s 2>/dev/null || true); ' "${GPU_CUR_FREQ_PATH}"
    else
        printf 'actual=; '
    fi
    printf 'status=ok; [ "$success" = "1" ] || status=unsupported_by_runtime; '
    printf 'echo CONTROL_TRANSITION_TRACE from=%s to=%s gpu_freq_apply_us=${delta} total_blocking_us=${delta} transition_energy_mj= transition_energy_source=unavailable success=${success} fallback=0 support_status=${status} requested_gpu_freq_hz=%s actual_gpu_freq_hz=${actual}' \
        "${from_state}" \
        "${to_state}" \
        "${target_freq}"
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
        return 1
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
        return 1
    fi

    case "${scale}" in
        1) printf '%s\n' "${raw}" ;;
        100) printf '%s\n' "$(( raw * 100 ))" ;;
        1000) printf '%s\n' "$(( raw * 1000 ))" ;;
        *) return 1 ;;
    esac
}

discover_power_paths() {
    BATTERY_VOLTAGE_PATH="$(find_first_readable_path \
        "${BATTERY_VOLTAGE_PATH:-}" \
        "/sys/class/power_supply/battery/voltage_now")" || true

    BATTERY_CURRENT_PATH="$(find_first_readable_path \
        "${BATTERY_CURRENT_PATH:-}" \
        "/sys/class/power_supply/battery/current_now")" || true

    TEMP_PATH="$(find_first_readable_path \
        "${TEMP_PATH:-}" \
        "/sys/class/power_supply/battery/temp" \
        "/sys/class/thermal/thermal_zone0/temp" \
        "/sys/class/thermal/thermal_zone1/temp")" || true

    if [[ -z "${BATTERY_VOLTAGE_PATH}" || -z "${BATTERY_CURRENT_PATH}" || -z "${TEMP_PATH}" ]]; then
        POWER_SAMPLING_AVAILABLE=0
        log "power sampling paths unavailable; sample logs will record unavailable"
        return 0
    fi

    local temp_raw
    temp_raw="$(read_remote_value "${TEMP_PATH}")"
    if ! TEMP_SCALE_TO_MC="$(infer_temp_scale_to_mc "${temp_raw}")"; then
        POWER_SAMPLING_AVAILABLE=0
        log "temperature scale unavailable; sample logs will record unavailable"
        return 0
    fi
    POWER_SAMPLING_AVAILABLE=1
}

read_temp_mc() {
    if [[ "${POWER_SAMPLING_AVAILABLE:-0}" != "1" ]]; then
        return 1
    fi
    local temp_raw
    temp_raw="$(read_remote_value "${TEMP_PATH}")"
    normalize_temp_to_mc "${temp_raw}" "${TEMP_SCALE_TO_MC}"
}

wait_for_cooldown_if_needed() {
    if [[ "${POWER_SAMPLING_AVAILABLE:-0}" != "1" ]]; then
        return 0
    fi

    local temp_mc
    if ! temp_mc="$(read_temp_mc)"; then
        return 0
    fi

    if (( temp_mc < TEMP_LIMIT_MC )); then
        return 0
    fi

    log "temperature $(format_mc "${temp_mc}")C >= limit ${TEMP_LIMIT_C}C; cooling to ${COOLDOWN_TEMP_C}C"
    local deadline=$(( $(date +%s) + COOLDOWN_TIMEOUT_S ))
    while (( $(date +%s) < deadline )); do
        sleep "${COOLDOWN_POLL_S}"
        if ! temp_mc="$(read_temp_mc)"; then
            return 0
        fi
        if (( temp_mc <= COOLDOWN_TEMP_MC )); then
            log "cooldown complete at $(format_mc "${temp_mc}")C"
            return 0
        fi
    done

    die "cooldown timed out above ${COOLDOWN_TEMP_C}C"
}

start_power_sampler() {
    local sample_log="$1"
    : > "${sample_log}"
    if [[ "${POWER_SAMPLING_AVAILABLE:-0}" != "1" ]]; then
        printf 'unavailable: power sysfs paths were not readable\n' > "${sample_log}"
        POWER_SAMPLER_PID=""
        return 0
    fi

    (
        while true; do
            local ts voltage current temp_raw temp_mc
            ts="$(date +%s)"
            voltage="$(read_remote_value "${BATTERY_VOLTAGE_PATH}")"
            current="$(read_remote_value "${BATTERY_CURRENT_PATH}")"
            temp_raw="$(read_remote_value "${TEMP_PATH}")"
            temp_mc="$(normalize_temp_to_mc "${temp_raw}" "${TEMP_SCALE_TO_MC}" || printf '')"
            printf '%s,%s,%s,%s\n' "${ts}" "${voltage}" "${current}" "${temp_mc}" >> "${sample_log}"
            sleep "${SAMPLE_INTERVAL_S}"
        done
    ) &
    POWER_SAMPLER_PID=$!
}

stop_power_sampler() {
    if [[ -n "${POWER_SAMPLER_PID:-}" ]]; then
        kill "${POWER_SAMPLER_PID}" >/dev/null 2>&1 || true
        wait "${POWER_SAMPLER_PID}" >/dev/null 2>&1 || true
        POWER_SAMPLER_PID=""
    fi
}

build_remote_bench_command() {
    local from_state="$1"
    local to_state="$2"
    local from_route="$3"
    local to_route="$4"

    local primary_dev=""
    local ngl="${NGL:-99}"
    local taskset_mask="${TASKSET_MASK}"
    local llama_threads="${LLAMA_THREADS}"
    if [[ "${from_route}" == "qnn-npu" || "${to_route}" == "qnn-npu" ]]; then
        primary_dev="qnn-npu"
    elif [[ "${from_route}" == "opencl" || "${to_route}" == "opencl" ]]; then
        primary_dev="GPUOpenCL"
    else
        primary_dev=""
        ngl="0"
    fi

    if transition_uses_cpu_state "${from_state}" "${to_state}"; then
        local cpu_state=""
        if is_cpu_state "${from_state}"; then
            cpu_state="${from_state}"
        else
            cpu_state="${to_state}"
        fi
        taskset_mask="$(state_cpu_taskset_mask "${cpu_state}")"
        llama_threads="$(state_cpu_threads "${cpu_state}")"
    fi

    local qnn_workpoint=""
    if is_qnn_state "${from_state}"; then
        qnn_workpoint="$(state_npu_workpoint "${from_state}")"
    elif is_qnn_state "${to_state}"; then
        qnn_workpoint="$(state_npu_workpoint "${to_state}")"
    fi
    local qnn_decode_workpoint=""
    if [[ "${from_route}" == "qnn-npu" && "${to_route}" == "qnn-npu" && "${from_state}" != "${to_state}" ]] &&
       is_qnn_state "${from_state}" && is_qnn_state "${to_state}"; then
        qnn_decode_workpoint="$(state_npu_workpoint "${to_state}")"
    fi

    local context_tokens
    context_tokens=$(( CONTEXT_LEN + DECODE_TOKENS_BEFORE_SWITCH + DECODE_TOKENS_AFTER_SWITCH + QNN_CACHE_SAFETY_MARGIN ))
    if (( context_tokens < DEFAULT_CONTEXT_TOKENS )); then
        context_tokens="${DEFAULT_CONTEXT_TOKENS}"
    fi

    local bench_workload_args
    if [[ "${LLAMA_BENCH_USE_PG_WORKLOAD}" == "1" && "${EXPERIMENT_PHASE}" == "phase_boundary" && "${DECODE_TOKENS_BEFORE_SWITCH}" == "0" ]]; then
        bench_workload_args="-pg ${CONTEXT_LEN},${DECODE_TOKENS_AFTER_SWITCH}"
    else
        bench_workload_args="-p ${CONTEXT_LEN} -n ${DECODE_TOKENS_AFTER_SWITCH}"
    fi

    local model_q bench_dir_q qnn_config_q qnn_model_dir_q
    model_q="$(shell_quote "${MODEL_PATH}")"
    bench_dir_q="$(shell_quote "${BENCH_DIR}")"
    qnn_config_q="$(shell_quote "${QNN_ACTIVE_CONFIG:-}")"
    qnn_model_dir_q="$(shell_quote "${QNN_ACTIVE_MODEL_DIR:-}")"

    local dev_args="-ngl ${ngl} -t ${llama_threads}"
    if [[ -n "${primary_dev}" ]]; then
        dev_args="-ngl ${ngl} -dev ${primary_dev} -t ${llama_threads}"
    fi

    {
        printf 'cd %s && ' "${bench_dir_q}"
        printf 'export LD_LIBRARY_PATH=%s:$LD_LIBRARY_PATH && ' "${bench_dir_q}"
        printf 'export ADSP_LIBRARY_PATH=%s && ' "${bench_dir_q}"
        printf 'export GGML_HEXAGON_EXPERIMENTAL=%s && ' "${QNN_ENABLE_HEXAGON}"
        if [[ -n "${qnn_workpoint}" ]]; then
            printf 'export GGML_QNN_HTP_WORKPOINT=%s && ' "${qnn_workpoint}"
        fi
        if [[ -n "${qnn_decode_workpoint}" ]]; then
            printf 'export GGML_HETERO_DYNAMIC_DECODE_QNN_WORKPOINT=%s && ' "${qnn_decode_workpoint}"
        fi
        if [[ -n "${QNN_ACTIVE_CONFIG:-}" ]]; then
            printf 'export GGML_QNN_AOT_CONFIG=%s && ' "${qnn_config_q}"
        fi
        if [[ -n "${QNN_ACTIVE_MODEL_DIR:-}" ]]; then
            printf 'export GGML_QNN_AOT_MODEL_DIR=%s && ' "${qnn_model_dir_q}"
        fi
        printf 'export GGML_QNN_AOT_WRITE_GENERIC_KV=%s && ' "${QNN_AOT_WRITE_GENERIC_KV}"
        printf 'export GGML_QNN_AOT_DISABLE_SEED_KV=%s && ' "${QNN_AOT_DISABLE_SEED_KV}"
        printf 'export GGML_HETERO_DYNAMIC_MODE=phase && '
        printf 'export GGML_HETERO_DYNAMIC_PREFILL_ROUTE=%s && ' "${from_route}"
        printf 'export GGML_HETERO_DYNAMIC_DECODE_ROUTE=%s && ' "${to_route}"
        printf 'export GGML_HETERO_DYNAMIC_TRACE_TIMING=1 && '
        if [[ "${to_route}" == "opencl" ]]; then
            printf 'export GGML_HETERO_DYNAMIC_DECODE_TG_ONLY_RESERVE=1 && '
            printf 'export GGML_OPENCL_EXPERIMENTAL_QNN_DIRECT_HOST_PTR=%s && ' "${OPENCL_QNN_DIRECT_HOST_PTR}"
            printf 'export GGML_OPENCL_EXPERIMENTAL_QNN_DIRECT_HOST_PTR_SKIP_UPLOAD=%s && ' "${OPENCL_QNN_DIRECT_HOST_PTR_SKIP_UPLOAD}"
        fi
        printf 'export LLAMA_BENCH_FAST_EXIT=%s && ' "${LLAMA_BENCH_FAST_EXIT_VALUE}"
        printf 'taskset %s ./llama-bench -v -m %s %s %s -d 0 -c %s -b %s -ub %s -r 1 --no-warmup --mmap %s' \
            "${taskset_mask}" \
            "${model_q}" \
            "${dev_args}" \
            "${bench_workload_args}" \
            "${context_tokens}" \
            "${BATCH_TOKENS}" \
            "${UBATCH_TOKENS}" \
            "${MMAP}"
    }
}

write_unsupported_transition_row() {
    local from_state="$1"
    local to_state="$2"
    local support_status="$3"
    local reason="$4"
    local raw_log="$5"

    mkdir -p "$(dirname "${raw_log}")"
    printf '%s\n' "${reason}" > "${raw_log}"

    append_transition_row "${RUN_CSV}" "${GLOBAL_CSV}" \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "${MODEL_PATH}" \
        "${CONTEXT_LEN}" \
        "${CONTEXT_LEN}" \
        "${from_state}" \
        "${to_state}" \
        "${DECODE_TOKENS_BEFORE_SWITCH}" \
        "${DECODE_TOKENS_AFTER_SWITCH}" \
        "${ROUNDS}" \
        '' '' '' '' '' '' '' '' '' '' '' '' '' \
        'unavailable' \
        '' \
        '0' \
        "${QNN_AOT_CACHE_SIZE}" \
        "${QNN_AOT_CONTEXT_SIZE}" \
        "${support_status}" \
        "${raw_log}"
}

parse_round_logs() {
    local raw_log="$1"
    local _phase_boundary_target_alias="$2"
    local exit_code="$3"

    local trace_line=""
    if [[ -s "${raw_log}" ]]; then
        trace_line="$(grep 'TRANSITION_TRACE' "${raw_log}" | tail -n 1 || true)"
    fi

    if [[ -n "${trace_line}" ]]; then
        local parsed
        parsed="$(parse_transition_trace_line "${trace_line}")"

        local decision_us route_apply_us policy_apply_us qnn_workpoint_apply_us gpu_freq_apply_us sched_reserve_us
        local kv_handoff_us graph_rebuild_us decode_entry_us total_blocking_us first_token_gap_us post_switch_tbt_us
        local transition_energy_mj transition_energy_source switch_success fallback_used support_status
        IFS=',' read -r \
            decision_us route_apply_us policy_apply_us qnn_workpoint_apply_us gpu_freq_apply_us sched_reserve_us \
            kv_handoff_us graph_rebuild_us decode_entry_us total_blocking_us first_token_gap_us post_switch_tbt_us \
            transition_energy_mj transition_energy_source switch_success fallback_used support_status <<< "${parsed}"

        if [[ -z "${first_token_gap_us}" || -z "${post_switch_tbt_us}" ]]; then
            local derived_token_timing derived_first_token_gap_us derived_post_switch_tbt_us
            derived_token_timing="$(derive_token_timing_from_log "${raw_log}")"
            IFS=',' read -r derived_first_token_gap_us derived_post_switch_tbt_us <<< "${derived_token_timing}"
            if [[ -z "${first_token_gap_us}" && -n "${derived_first_token_gap_us}" ]]; then
                first_token_gap_us="${derived_first_token_gap_us}"
            fi
            if [[ -z "${post_switch_tbt_us}" && -n "${derived_post_switch_tbt_us}" ]]; then
                post_switch_tbt_us="${derived_post_switch_tbt_us}"
            fi
        fi

        local detected_fallback
        detected_fallback="$(detect_fallback_used "${raw_log}")"
        if [[ "${detected_fallback}" == "1" ]]; then
            fallback_used=1
            switch_success=0
        fi
        if [[ "${exit_code}" != "0" ]]; then
            switch_success=0
        fi
        if [[ "${switch_success}" != "1" ]]; then
            support_status="$(classify_transition_support_status "${exit_code}" "${switch_success}" "${raw_log}")"
        fi

        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s' \
            "${decision_us}" \
            "${route_apply_us}" \
            "${policy_apply_us}" \
            "${qnn_workpoint_apply_us}" \
            "${gpu_freq_apply_us}" \
            "${sched_reserve_us}" \
            "${kv_handoff_us}" \
            "${graph_rebuild_us}" \
            "${decode_entry_us}" \
            "${total_blocking_us}" \
            "${first_token_gap_us}" \
            "${post_switch_tbt_us}" \
            "${transition_energy_mj}" \
            "${transition_energy_source}" \
            "${switch_success}" \
            "${fallback_used}" \
            "${support_status}"
        return 0
    fi

    local timing_line=""
    if [[ -s "${raw_log}" ]]; then
        timing_line="$(grep -m 1 -E 'timing phase=.*total_wall_us=.*route_applied=true' "${raw_log}" || true)"
    fi

    if [[ -z "${timing_line}" ]]; then
        local support_status
        support_status="$(classify_transition_support_status "${exit_code}" "0" "${raw_log}")"
        printf ',,,,,,,,,,,,,unavailable,0,%s,%s' "$(detect_fallback_used "${raw_log}")" "${support_status}"
        return 0
    fi

    local decision_us route_apply_us sched_reserve_us kv_handoff_us graph_rebuild_us total_blocking_us switch_success
    IFS=',' read -r decision_us route_apply_us sched_reserve_us kv_handoff_us graph_rebuild_us total_blocking_us switch_success <<< "$(parse_transition_timing_line "${timing_line}")"

    local fallback_used
    fallback_used="$(detect_fallback_used "${raw_log}")"
    if [[ "${exit_code}" != "0" ]]; then
        switch_success=0
    fi

    local support_status
    support_status="$(classify_transition_support_status "${exit_code}" "${switch_success}" "${raw_log}")"

    printf '%s,%s,,,,%s,%s,%s,,%s,,,,unavailable,%s,%s,%s' \
        "${decision_us}" \
        "${route_apply_us}" \
        "${sched_reserve_us}" \
        "${kv_handoff_us}" \
        "${graph_rebuild_us}" \
        "${total_blocking_us}" \
        "${switch_success}" \
        "${fallback_used}" \
        "${support_status}"
}

run_transition_round() {
    local from_state="$1"
    local to_state="$2"
    local round="$3"
    local transition_dir="$4"
    local rounds_csv="$5"
    local from_route="$6"
    local to_route="$7"

    local round_dir="${transition_dir}/round_${round}"
    local stdout_log="${round_dir}/bench.stdout.log"
    local stderr_log="${round_dir}/bench.stderr.log"
    local raw_log="${round_dir}/bench.raw.log"
    local sample_log="${round_dir}/power_samples.csv"
    local command_path="${round_dir}/command.sh"
    mkdir -p "${round_dir}"

    local remote_command
    remote_command="$(build_remote_bench_command "${from_state}" "${to_state}" "${from_route}" "${to_route}")"
    {
        printf '#!/usr/bin/env bash\n'
        printf 'adb -s %q shell %q\n' "${DEVICE}" "${remote_command}"
    } > "${command_path}"
    chmod +x "${command_path}"

    wait_for_cooldown_if_needed
    start_power_sampler "${sample_log}"

    set +e
    adb -s "${DEVICE}" shell "${remote_command}" > "${stdout_log}" 2> "${stderr_log}"
    local exit_code=$?
    set -e

    stop_power_sampler
    {
        printf '### stdout\n'
        cat "${stdout_log}"
        printf '\n### stderr\n'
        cat "${stderr_log}"
        printf '\n### power_samples\n'
        printf '%s\n' "${sample_log}"
    } > "${raw_log}"

    local parsed
    parsed="$(parse_round_logs "${raw_log}" "${to_route}" "${exit_code}")"

    local decision_us route_apply_us policy_apply_us qnn_workpoint_apply_us gpu_freq_apply_us sched_reserve_us
    local kv_handoff_us graph_rebuild_us decode_entry_us total_blocking_us first_token_gap_us post_switch_tbt_us
    local transition_energy_mj transition_energy_source switch_success fallback_used support_status
    IFS=',' read -r \
        decision_us route_apply_us policy_apply_us qnn_workpoint_apply_us gpu_freq_apply_us sched_reserve_us \
        kv_handoff_us graph_rebuild_us decode_entry_us total_blocking_us first_token_gap_us post_switch_tbt_us \
        transition_energy_mj transition_energy_source switch_success fallback_used support_status <<< "${parsed}"

    local note=""
    if [[ "${EXPERIMENT_PHASE}" == "phase_boundary" && "${DECODE_TOKENS_BEFORE_SWITCH}" != "0" ]]; then
        note="phase_boundary_run_before_switch_tokens_not_executed_mid_decode"
    fi
    if [[ "${transition_energy_source}" == "unavailable" ]]; then
        note="${note}${note:+;}transition_energy_unavailable"
    fi
    if [[ -z "${first_token_gap_us}" ]]; then
        note="${note}${note:+;}first_token_gap_unavailable"
    fi
    if [[ -z "${post_switch_tbt_us}" ]]; then
        note="${note}${note:+;}post_switch_tbt_unavailable"
    fi

    append_round_row "${rounds_csv}" \
        "${round}" \
        "${decision_us}" \
        "${route_apply_us}" \
        "${policy_apply_us}" \
        "${qnn_workpoint_apply_us}" \
        "${gpu_freq_apply_us}" \
        "${sched_reserve_us}" \
        "${kv_handoff_us}" \
        "${graph_rebuild_us}" \
        "${decode_entry_us}" \
        "${total_blocking_us}" \
        "${first_token_gap_us}" \
        "${post_switch_tbt_us}" \
        "${transition_energy_mj}" \
        "${transition_energy_source}" \
        "${switch_success}" \
        "${fallback_used}" \
        "${support_status}" \
        "${raw_log}" \
        "${exit_code}" \
        "${note}"

    if [[ "${support_status}" != "ok" ]]; then
        log "round ${round} ${from_state}->${to_state} recorded as ${support_status}; raw_log=${raw_log}"
    fi
}

run_gpu_control_transition_round() {
    local from_state="$1"
    local to_state="$2"
    local round="$3"
    local transition_dir="$4"
    local rounds_csv="$5"

    local round_dir="${transition_dir}/round_${round}"
    local stdout_log="${round_dir}/bench.stdout.log"
    local stderr_log="${round_dir}/bench.stderr.log"
    local raw_log="${round_dir}/bench.raw.log"
    local sample_log="${round_dir}/power_samples.csv"
    local command_path="${round_dir}/command.sh"
    local setup_log="${round_dir}/gpu_setup.log"
    mkdir -p "${round_dir}"

    local remote_command
    remote_command="$(build_gpu_control_apply_command "${from_state}" "${to_state}")"
    {
        printf '#!/usr/bin/env bash\n'
        printf 'adb -s %q shell %q\n' "${DEVICE}" "su -c '${remote_command}'"
    } > "${command_path}"
    chmod +x "${command_path}"

    wait_for_cooldown_if_needed

    set +e
    pin_gpu_freq_for_control_transition "${from_state}" "${to_state}" "${from_state}" > "${setup_log}" 2>&1
    local setup_exit_code=$?
    set -e

    local exit_code=0
    if [[ "${setup_exit_code}" != "0" ]]; then
        printf 'failed to set initial GPU frequency for %s before control transition\n' "${from_state}" > "${stderr_log}"
        : > "${stdout_log}"
        exit_code="${setup_exit_code}"
    else
        start_power_sampler "${sample_log}"
        set +e
        adb -s "${DEVICE}" shell "su -c '${remote_command}'" > "${stdout_log}" 2> "${stderr_log}"
        exit_code=$?
        set -e
        sleep "${CONTROL_SAMPLE_TAIL_S}"
        stop_power_sampler
    fi

    if [[ ! -s "${sample_log}" ]]; then
        printf 'unavailable: GPU control sample window ended before first sample\n' > "${sample_log}"
    fi

    {
        printf '### control_setup\n'
        cat "${setup_log}" 2>/dev/null || true
        printf '\n### stdout\n'
        cat "${stdout_log}"
        printf '\n### stderr\n'
        cat "${stderr_log}"
        printf '\n### power_samples\n'
        printf '%s\n' "${sample_log}"
    } > "${raw_log}"

    local parsed
    parsed="$(parse_round_logs "${raw_log}" "opencl" "${exit_code}")"

    local decision_us route_apply_us policy_apply_us qnn_workpoint_apply_us gpu_freq_apply_us sched_reserve_us
    local kv_handoff_us graph_rebuild_us decode_entry_us total_blocking_us first_token_gap_us post_switch_tbt_us
    local transition_energy_mj transition_energy_source switch_success fallback_used support_status
    IFS=',' read -r \
        decision_us route_apply_us policy_apply_us qnn_workpoint_apply_us gpu_freq_apply_us sched_reserve_us \
        kv_handoff_us graph_rebuild_us decode_entry_us total_blocking_us first_token_gap_us post_switch_tbt_us \
        transition_energy_mj transition_energy_source switch_success fallback_used support_status <<< "${parsed}"

    local note="gpu_frequency_control_only"
    if [[ "${transition_energy_source}" == "unavailable" ]]; then
        note="${note};transition_energy_unavailable"
    fi
    if [[ -z "${first_token_gap_us}" ]]; then
        note="${note};first_token_gap_unavailable"
    fi
    if [[ -z "${post_switch_tbt_us}" ]]; then
        note="${note};post_switch_tbt_unavailable"
    fi

    append_round_row "${rounds_csv}" \
        "${round}" \
        "${decision_us}" \
        "${route_apply_us}" \
        "${policy_apply_us}" \
        "${qnn_workpoint_apply_us}" \
        "${gpu_freq_apply_us}" \
        "${sched_reserve_us}" \
        "${kv_handoff_us}" \
        "${graph_rebuild_us}" \
        "${decode_entry_us}" \
        "${total_blocking_us}" \
        "${first_token_gap_us}" \
        "${post_switch_tbt_us}" \
        "${transition_energy_mj}" \
        "${transition_energy_source}" \
        "${switch_success}" \
        "${fallback_used}" \
        "${support_status}" \
        "${raw_log}" \
        "${exit_code}" \
        "${note}"

    if [[ "${support_status}" != "ok" ]]; then
        log "round ${round} ${from_state}->${to_state} recorded as ${support_status}; raw_log=${raw_log}"
    fi
}

append_aggregated_transition_summary() {
    local from_state="$1"
    local to_state="$2"
    local summary="$3"

    local decision_us route_apply_us policy_apply_us qnn_workpoint_apply_us gpu_freq_apply_us sched_reserve_us
    local kv_handoff_us graph_rebuild_us decode_entry_us total_blocking_us first_token_gap_us post_switch_tbt_us
    local transition_energy_mj transition_energy_source switch_success_rate fallback_count support_status raw_log_path
    IFS=',' read -r \
        decision_us route_apply_us policy_apply_us qnn_workpoint_apply_us gpu_freq_apply_us sched_reserve_us \
        kv_handoff_us graph_rebuild_us decode_entry_us total_blocking_us first_token_gap_us post_switch_tbt_us \
        transition_energy_mj transition_energy_source switch_success_rate fallback_count support_status raw_log_path <<< "${summary}"

    append_transition_row "${RUN_CSV}" "${GLOBAL_CSV}" \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "${MODEL_PATH}" \
        "${CONTEXT_LEN}" \
        "${CONTEXT_LEN}" \
        "${from_state}" \
        "${to_state}" \
        "${DECODE_TOKENS_BEFORE_SWITCH}" \
        "${DECODE_TOKENS_AFTER_SWITCH}" \
        "${ROUNDS}" \
        "${decision_us}" \
        "${route_apply_us}" \
        "${policy_apply_us}" \
        "${qnn_workpoint_apply_us}" \
        "${gpu_freq_apply_us}" \
        "${sched_reserve_us}" \
        "${kv_handoff_us}" \
        "${graph_rebuild_us}" \
        "${decode_entry_us}" \
        "${total_blocking_us}" \
        "${first_token_gap_us}" \
        "${post_switch_tbt_us}" \
        "${transition_energy_mj}" \
        "${transition_energy_source}" \
        "${switch_success_rate}" \
        "${fallback_count}" \
        "${QNN_AOT_CACHE_SIZE}" \
        "${QNN_AOT_CONTEXT_SIZE}" \
        "${support_status}" \
        "${raw_log_path}"
}

run_transition() {
    local from_state="$1"
    local to_state="$2"

    local from_route to_route
    from_route="$(state_route "${from_state}")" || die "unknown from_state: ${from_state}"
    to_route="$(state_route "${to_state}")" || die "unknown to_state: ${to_state}"

    local slug
    slug="$(transition_slug "${from_state}" "${to_state}")"
    local transition_dir="${OUTPUT_DIR}/raw/${slug}"
    local rounds_csv="${transition_dir}/rounds.csv"
    mkdir -p "${transition_dir}"
    ensure_transition_round_header "${rounds_csv}"

    local guard_status guard_footprint
    IFS=',' read -r guard_status guard_footprint <<< "$(transition_qnn_support_status \
        "${from_state}" \
        "${to_state}" \
        "${CONTEXT_LEN}" \
        "${DECODE_TOKENS_BEFORE_SWITCH}" \
        "${DECODE_TOKENS_AFTER_SWITCH}" \
        "${QNN_CACHE_SAFETY_MARGIN}" \
        "${QNN_AOT_CACHE_SIZE}")"

    if [[ "${guard_status}" != "ok" ]]; then
        write_unsupported_transition_row \
            "${from_state}" \
            "${to_state}" \
            "${guard_status}" \
            "Skipped: QNN footprint ${guard_footprint:-unknown} with safety margin ${QNN_CACHE_SAFETY_MARGIN} is unsupported by active qnn_aot_cache_size ${QNN_AOT_CACHE_SIZE:-unavailable}." \
            "${transition_dir}/unsupported.log"
        return 0
    fi

    local cpu_state_pinned=0
    if transition_uses_cpu_state "${from_state}" "${to_state}"; then
        pin_cpu_state_for_transition "${from_state}" "${to_state}"
        cpu_state_pinned=1
    fi

    local control_kind
    control_kind="$(same_route_control_kind "${from_state}" "${to_state}")"
    if [[ "${control_kind}" == "gpu_freq" ]]; then
        if [[ "${GPU_CONTROL_AVAILABLE:-0}" != "1" ]]; then
            write_unsupported_transition_row \
                "${from_state}" \
                "${to_state}" \
                "unsupported_by_runtime" \
                "Skipped: GPU frequency-only switching requires writable GPU min/max sysfs paths, but they were not available." \
                "${transition_dir}/unsupported.log"
            return 0
        fi

        log "running ${from_state}->${to_state} GPU frequency-control transition"
        local round
        for (( round = 1; round <= ROUNDS; ++round )); do
            run_gpu_control_transition_round "${from_state}" "${to_state}" "${round}" "${transition_dir}" "${rounds_csv}"
        done

        local summary
        summary="$(aggregate_transition_rounds "${rounds_csv}")"
        append_aggregated_transition_summary "${from_state}" "${to_state}" "${summary}"
        if (( cpu_state_pinned )); then
            restore_cpu_control_state
        fi
        return 0
    fi

    if [[ "${control_kind}" == "qnn_workpoint" ]]; then
        log "running ${from_state}->${to_state} QNN workpoint-control transition"
        local round
        for (( round = 1; round <= ROUNDS; ++round )); do
            run_transition_round "${from_state}" "${to_state}" "${round}" "${transition_dir}" "${rounds_csv}" "${from_route}" "${to_route}"
        done

        local summary
        summary="$(aggregate_transition_rounds "${rounds_csv}")"
        append_aggregated_transition_summary "${from_state}" "${to_state}" "${summary}"
        if (( cpu_state_pinned )); then
            restore_cpu_control_state
        fi
        return 0
    fi

    if [[ "${from_route}" == "${to_route}" && "${from_state}" != "${to_state}" ]]; then
        write_unsupported_transition_row \
            "${from_state}" \
            "${to_state}" \
            "unsupported_by_runtime" \
            "Skipped: ${EXPERIMENT_PHASE} dynamic route can measure backend route changes, but ${from_state}->${to_state} maps to the same route (${from_route}). Workpoint/frequency-only switching needs separate instrumentation." \
            "${transition_dir}/unsupported.log"
        if (( cpu_state_pinned )); then
            restore_cpu_control_state
        fi
        return 0
    fi

    log "running ${from_state}->${to_state} context_len=${CONTEXT_LEN} phase=${EXPERIMENT_PHASE}"
    local round
    for (( round = 1; round <= ROUNDS; ++round )); do
        run_transition_round "${from_state}" "${to_state}" "${round}" "${transition_dir}" "${rounds_csv}" "${from_route}" "${to_route}"
    done

    local summary
    summary="$(aggregate_transition_rounds "${rounds_csv}")"
    append_aggregated_transition_summary "${from_state}" "${to_state}" "${summary}"
    if (( cpu_state_pinned )); then
        restore_cpu_control_state
    fi
}

validate_transition_list() {
    local spec from_state to_state parsed
    for spec in ${TRANSITION_LIST}; do
        parsed="$(parse_transition_spec "${spec}")" || die "invalid TRANSITION_LIST entry: ${spec}; use from->to"
        IFS=',' read -r from_state to_state <<< "${parsed}"
        state_route "${from_state}" >/dev/null || die "unsupported from_state: ${from_state}"
        state_route "${to_state}" >/dev/null || die "unsupported to_state: ${to_state}"
    done

    [[ "${CONTEXT_LEN}" =~ ^[0-9]+$ ]] || die "CONTEXT_LEN must be numeric"
    [[ "${DECODE_TOKENS_BEFORE_SWITCH}" =~ ^[0-9]+$ ]] || die "DECODE_TOKENS_BEFORE_SWITCH must be numeric"
    [[ "${DECODE_TOKENS_AFTER_SWITCH}" =~ ^[0-9]+$ ]] || die "DECODE_TOKENS_AFTER_SWITCH must be numeric"
    [[ "${ROUNDS}" =~ ^[0-9]+$ ]] || die "ROUNDS must be numeric"
}

require_runtime_inputs() {
    [[ -n "${DEVICE}" ]] || die "DEVICE must be set"
    [[ -n "${MODEL_PATH}" ]] || die "MODEL_PATH must be set"
    adb -s "${DEVICE}" get-state >/dev/null 2>&1 || die "device ${DEVICE} is offline"
}

prepare_output_dir() {
    if [[ -e "${OUTPUT_DIR}" && "${INSIGHTB_ALLOW_EXISTING_OUTPUT_DIR:-0}" != "1" ]]; then
        if find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
            die "OUTPUT_DIR exists and is non-empty: ${OUTPUT_DIR}"
        fi
    fi
    mkdir -p "${OUTPUT_DIR}/commands" "${RESULTS_DIR}" "$(dirname "${SUMMARY_MD}")"
    ensure_transition_header "${RUN_CSV}"
    ensure_transition_header "${GLOBAL_CSV}"
}

record_run_metadata() {
    local -a top_cmd=(
        env
        "DEVICE=${DEVICE}"
        "MODEL_PATH=${MODEL_PATH}"
        "OUTPUT_DIR=${OUTPUT_DIR}"
        "BENCH_DIR=${BENCH_DIR}"
        "CONTEXT_LEN=${CONTEXT_LEN}"
        "DECODE_TOKENS_BEFORE_SWITCH=${DECODE_TOKENS_BEFORE_SWITCH}"
        "DECODE_TOKENS_AFTER_SWITCH=${DECODE_TOKENS_AFTER_SWITCH}"
        "TRANSITION_LIST=${TRANSITION_LIST}"
        "ROUNDS=${ROUNDS}"
        "TEMP_LIMIT_C=${TEMP_LIMIT_C}"
        "COOLDOWN_TEMP_C=${COOLDOWN_TEMP_C}"
        "SAMPLE_INTERVAL_S=${SAMPLE_INTERVAL_S}"
        "QNN_CACHE_SAFETY_MARGIN=${QNN_CACHE_SAFETY_MARGIN}"
        "EXPERIMENT_PHASE=${EXPERIMENT_PHASE}"
        "CONTROL_SAMPLE_TAIL_S=${CONTROL_SAMPLE_TAIL_S}"
        "TASKSET_MASK=${TASKSET_MASK}"
        "LLAMA_THREADS=${LLAMA_THREADS}"
        "CPU_PIN_GOVERNOR=${CPU_PIN_GOVERNOR}"
        "NGL=${NGL}"
        "BATCH_TOKENS=${BATCH_TOKENS}"
        "UBATCH_TOKENS=${UBATCH_TOKENS}"
        "CONTEXT_TOKENS=${DEFAULT_CONTEXT_TOKENS}"
        "MMAP=${MMAP}"
        "LLAMA_BENCH_FAST_EXIT_VALUE=${LLAMA_BENCH_FAST_EXIT_VALUE}"
        "LLAMA_BENCH_USE_PG_WORKLOAD=${LLAMA_BENCH_USE_PG_WORKLOAD}"
    )
    if [[ -n "${QNN_ACTIVE_CONFIG}" ]]; then
        top_cmd+=("GGML_QNN_AOT_CONFIG=${QNN_ACTIVE_CONFIG}")
    fi
    if [[ -n "${QNN_ACTIVE_MODEL_DIR}" ]]; then
        top_cmd+=("GGML_QNN_AOT_MODEL_DIR=${QNN_ACTIVE_MODEL_DIR}")
    fi
    top_cmd+=(bash "${ROOT_DIR}/scripts/run_insightB_transition_overhead.sh")
    write_command_file "${COMMAND_FILE}" "${top_cmd[@]}"

    if git -C "${ROOT_DIR}" rev-parse HEAD >/dev/null 2>&1; then
        git -C "${ROOT_DIR}" rev-parse HEAD > "${GIT_COMMIT_FILE}"
        git -C "${ROOT_DIR}" status --short > "${GIT_STATUS_FILE}"
    else
        printf 'unavailable\n' > "${GIT_COMMIT_FILE}"
        printf 'unavailable\n' > "${GIT_STATUS_FILE}"
    fi
}

load_qnn_aot_sizes_if_needed() {
    local needs_qnn=0
    local spec parsed from_state to_state
    for spec in ${TRANSITION_LIST}; do
        parsed="$(parse_transition_spec "${spec}")"
        IFS=',' read -r from_state to_state <<< "${parsed}"
        if is_qnn_state "${from_state}" || is_qnn_state "${to_state}"; then
            needs_qnn=1
        fi
    done
    (( needs_qnn )) || return 0

    if [[ -z "${QNN_ACTIVE_CONFIG}" ]]; then
        log "no GGML_QNN_AOT_CONFIG/QNN_AOT_CONFIG/QNN_DIR found; QNN transitions will be marked unsupported_by_runtime"
        return 0
    fi

    if ! copy_or_pull_qnn_config "${QNN_ACTIVE_CONFIG}" "${QNN_CONFIG_COPY}"; then
        log "failed to read QNN AoT config ${QNN_ACTIVE_CONFIG}; QNN transitions will be marked unsupported_by_runtime"
        return 0
    fi

    local batches sizes
    batches="1 ${BATCH_TOKENS:-1} ${UBATCH_TOKENS:-1}"
    if ! sizes="$(parse_qnn_aot_sizes_from_file "${QNN_CONFIG_COPY}" "${batches}")"; then
        log "failed to parse QNN AoT cache/context sizes from ${QNN_ACTIVE_CONFIG}"
        return 0
    fi
    IFS=',' read -r QNN_AOT_CACHE_SIZE QNN_AOT_CONTEXT_SIZE <<< "${sizes}"
    log "QNN AoT config: ${QNN_ACTIVE_CONFIG} cache_size=${QNN_AOT_CACHE_SIZE} context_size=${QNN_AOT_CONTEXT_SIZE}"
}

write_summary_markdown() {
    local summary_path="$1"
    local run_id="$2"
    local run_csv="$3"
    local command_file="$4"
    local output_dir="$5"

    local temp_range
    temp_range="$(find "${output_dir}/raw" -name 'power_samples.csv' -type f 2>/dev/null | xargs -r awk -F, '$4 ~ /^-?[0-9]+$/ {
            temp = $4 / 1000.0
            if (!seen || temp < min) min = temp
            if (!seen || temp > max) max = temp
            seen = 1
        } END {
            if (seen) printf "%.2fC to %.2fC", min, max
            else printf "unavailable"
        }')"
    [[ -n "${temp_range}" ]] || temp_range="unavailable"

    local unsupported
    unsupported="$(awk -F, 'NR > 1 && $28 != "ok" {
            printf "- %s -> %s support_status=%s raw_log=%s\n", $5, $6, $28, $29
        }' "${run_csv}")"
    [[ -n "${unsupported}" ]] || unsupported="- none"

    local missing_fields
    missing_fields="$(awk -F, 'NR > 1 && $28 == "ok" && ($20 == "" || $21 == "") {
            printf "- %s -> %s lacks direct first_token_gap_us or post_switch_tbt_us; raw_log=%s\n", $5, $6, $29
        }' "${run_csv}")"
    [[ -n "${missing_fields}" ]] || missing_fields="- none"

    local paper_ready="needs rerun/review"
    if ! awk -F, 'NR > 1 && ($28 != "ok" || $20 == "" || $21 == "" || $23 == "unavailable") { bad = 1 } END { exit bad ? 0 : 1 }' "${run_csv}"; then
        paper_ready="candidate, pending manual data-quality review"
    fi

    local git_commit="unavailable"
    local git_dirty_status="unavailable"
    local git_status_text=""
    if [[ -s "${output_dir}/git_commit.txt" ]]; then
        git_commit="$(sed -n '1p' "${output_dir}/git_commit.txt")"
    fi
    if [[ -f "${output_dir}/git_status.txt" ]]; then
        if [[ -s "${output_dir}/git_status.txt" ]]; then
            git_dirty_status="dirty"
            git_status_text="$(sed -n '1,80p' "${output_dir}/git_status.txt")"
        else
            git_dirty_status="clean"
        fi
    fi

    {
        printf '# Insight B Transition Cost %s\n\n' "${run_id}"
        printf '## Experiment goal\n\n'
        printf 'Measure phase-boundary transition overhead between representative heterogeneous execution states. This script uses existing dynamic route timing first; mid-decode switching still needs separate runtime instrumentation.\n\n'
        printf '## Exact commands\n\n'
        printf 'Top-level command:\n\n'
        printf '```bash\n'
        sed -n '1,20p' "${command_file}"
        printf '```\n\n'
        printf 'Per-round command files are under `%s/raw/*/round_*/command.sh`.\n\n' "${output_dir}"
        printf '## Git state\n\n'
        printf '%s\n' "- Commit: \`${git_commit}\`"
        printf '%s\n\n' "- Dirty status: \`${git_dirty_status}\`"
        if [[ -n "${git_status_text}" ]]; then
            printf '```text\n'
            printf '%s\n' "${git_status_text}"
            printf '```\n\n'
        fi
        printf '## Temperature range\n\n'
        printf '%s\n\n' "${temp_range}"
        printf '## Main result table\n\n'
        printf '| from_state | to_state | total_blocking_us | kv_handoff_us | first_token_gap_us | post_switch_tbt_us | success_rate | fallback_count | support_status |\n'
        printf '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n'
        awk -F, 'NR > 1 {
            printf "| %s | %s | %s | %s | %s | %s | %s | %s | %s |\n", $5, $6, $19, $16, $20, $21, $24, $25, $28
        }' "${run_csv}"
        printf '\n## Anomalies\n\n'
        printf '%s\n\n' "${unsupported}"
        printf '## Missing direct trace fields\n\n'
        printf '%s\n\n' "${missing_fields}"
        printf '## Raw output directories\n\n'
        printf '%s\n\n' "- \`${output_dir}\`"
        printf '## Paper readiness\n\n'
        printf '%s\n\n' "${paper_ready}"
        printf '## Unsupported conditions\n\n'
        printf '%s\n' "${unsupported}"
    } > "${summary_path}"
}

cleanup() {
    stop_power_sampler
    restore_cpu_control_state
    restore_gpu_control_state
    restore_display_state
}

if [[ "${INSIGHTB_TRANSITION_OVERHEAD_LIB_ONLY:-0}" == "1" ]]; then
    return 0 2>/dev/null || exit 0
fi

DEVICE="${DEVICE:-}"
MODEL_PATH="${MODEL_PATH:-${MODEL:-}}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/results/insightB/transition-overhead-$(date -u +%Y%m%d-%H%M%S)}"
CONTEXT_LEN="${CONTEXT_LEN:-512}"
DECODE_TOKENS_BEFORE_SWITCH="${DECODE_TOKENS_BEFORE_SWITCH:-16}"
DECODE_TOKENS_AFTER_SWITCH="${DECODE_TOKENS_AFTER_SWITCH:-64}"
TRANSITION_LIST="${TRANSITION_LIST:-npu_burst->gpu_734 gpu_734->npu_low_balanced npu_burst->cpu_big2_2649 gpu_734->gpu_967}"
ROUNDS="${ROUNDS:-5}"
TEMP_LIMIT_C="${TEMP_LIMIT_C:-38.0}"
COOLDOWN_TEMP_C="${COOLDOWN_TEMP_C:-37.0}"
COOLDOWN_TIMEOUT_S="${COOLDOWN_TIMEOUT_S:-900}"
COOLDOWN_POLL_S="${COOLDOWN_POLL_S:-5}"
QNN_CACHE_SAFETY_MARGIN="${QNN_CACHE_SAFETY_MARGIN:-32}"
EXPERIMENT_PHASE="${EXPERIMENT_PHASE:-phase_boundary}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/results/insightB}"
GLOBAL_CSV="${RESULTS_DIR}/transition_cost.csv"
RUN_ID="$(date -u +%Y%m%d-%H%M%S)"
RUN_CSV="${OUTPUT_DIR}/transition_cost.csv"
COMMAND_FILE="${OUTPUT_DIR}/command.txt"
GIT_COMMIT_FILE="${OUTPUT_DIR}/git_commit.txt"
GIT_STATUS_FILE="${OUTPUT_DIR}/git_status.txt"
SUMMARY_MD="${ROOT_DIR}/docs/实验结果/InsightB_Transition_Cost_${RUN_ID}.md"

BENCH_DIR="${BENCH_DIR:-${QNN_BIN:-/data/local/tmp/acom-qnn-phase-materializer/bin}}"
TASKSET_MASK="${TASKSET_MASK:-80}"
LLAMA_THREADS="${LLAMA_THREADS:-1}"
NGL="${NGL:-99}"
BATCH_TOKENS="${BATCH_TOKENS:-1}"
UBATCH_TOKENS="${UBATCH_TOKENS:-1}"
DEFAULT_CONTEXT_TOKENS="${CONTEXT_TOKENS:-2048}"
MMAP="${MMAP:-0}"
LLAMA_BENCH_FAST_EXIT_VALUE="${LLAMA_BENCH_FAST_EXIT_VALUE:-1}"
LLAMA_BENCH_USE_PG_WORKLOAD="${LLAMA_BENCH_USE_PG_WORKLOAD:-1}"

QNN_ACTIVE_CONFIG="${GGML_QNN_AOT_CONFIG:-${QNN_AOT_CONFIG:-${QNN_DIR:+${QNN_DIR}/config.json}}}"
QNN_ACTIVE_MODEL_DIR="${GGML_QNN_AOT_MODEL_DIR:-${QNN_AOT_MODEL_DIR:-${QNN_DIR:-}}}"
QNN_CONFIG_COPY="${OUTPUT_DIR}/qnn_aot_config.json"
QNN_AOT_CACHE_SIZE=""
QNN_AOT_CONTEXT_SIZE=""
QNN_AOT_WRITE_GENERIC_KV="${QNN_AOT_WRITE_GENERIC_KV:-1}"
QNN_AOT_DISABLE_SEED_KV="${QNN_AOT_DISABLE_SEED_KV:-1}"
QNN_ENABLE_HEXAGON="${QNN_ENABLE_HEXAGON:-1}"
OPENCL_QNN_DIRECT_HOST_PTR="${OPENCL_QNN_DIRECT_HOST_PTR:-1}"
OPENCL_QNN_DIRECT_HOST_PTR_SKIP_UPLOAD="${OPENCL_QNN_DIRECT_HOST_PTR_SKIP_UPLOAD:-1}"
GPU_AVAILABLE_FREQ_PATH="${GPU_AVAILABLE_FREQ_PATH:-}"
GPU_MIN_FREQ_PATH="${GPU_MIN_FREQ_PATH:-}"
GPU_MAX_FREQ_PATH="${GPU_MAX_FREQ_PATH:-}"
GPU_CUR_FREQ_PATH="${GPU_CUR_FREQ_PATH:-}"
GPU_GOVERNOR_PATH="${GPU_GOVERNOR_PATH:-}"
GPU_PIN_GOVERNOR="${GPU_PIN_GOVERNOR:-}"
GPU_CONTROL_AVAILABLE=0
GPU_CONTROL_STATE_SAVED=0
ORIG_GPU_MIN_FREQ=""
ORIG_GPU_MAX_FREQ=""
ORIG_GPU_GOVERNOR=""
CPU_PIN_GOVERNOR="${CPU_PIN_GOVERNOR:-powersave}"
declare -A ORIG_CPU_MIN_FREQ=()
declare -A ORIG_CPU_MAX_FREQ=()
declare -A ORIG_CPU_GOVERNOR=()
declare -A ORIG_CPU_STATE_SAVED_BY_POLICY=()
PIN_CPU_POLICY_RESULT=""

KEEP_SCREEN_ON_TIMEOUT_MS="${KEEP_SCREEN_ON_TIMEOUT_MS:-1800000}"
SCREEN_BRIGHTNESS_OVERRIDE="${SCREEN_BRIGHTNESS_OVERRIDE:-}"
SAMPLE_INTERVAL_S="${SAMPLE_INTERVAL_S:-1}"
CONTROL_SAMPLE_TAIL_S="${CONTROL_SAMPLE_TAIL_S:-2.0}"
BATTERY_VOLTAGE_PATH="${BATTERY_VOLTAGE_PATH:-}"
BATTERY_CURRENT_PATH="${BATTERY_CURRENT_PATH:-}"
TEMP_PATH="${TEMP_PATH:-}"
TEMP_SCALE_TO_MC=""
TEMP_LIMIT_MC="$(float_to_mc "${TEMP_LIMIT_C}")"
COOLDOWN_TEMP_MC="$(float_to_mc "${COOLDOWN_TEMP_C}")"
POWER_SAMPLING_AVAILABLE=0
POWER_SAMPLER_PID=""
DISPLAY_STATE_SAVED=0

trap cleanup EXIT

main() {
    validate_transition_list
    require_runtime_inputs
    prepare_output_dir
    record_run_metadata
    save_display_state
    ensure_screen_on
    discover_power_paths
    prepare_gpu_control_if_needed
    load_qnn_aot_sizes_if_needed

    local spec parsed from_state to_state
    for spec in ${TRANSITION_LIST}; do
        parsed="$(parse_transition_spec "${spec}")"
        IFS=',' read -r from_state to_state <<< "${parsed}"
        run_transition "${from_state}" "${to_state}"
    done

    write_summary_markdown "${SUMMARY_MD}" "${RUN_ID}" "${RUN_CSV}" "${COMMAND_FILE}" "${OUTPUT_DIR}"
    log "run CSV written to ${RUN_CSV}"
    log "aggregate CSV updated at ${GLOBAL_CSV}"
    log "summary Markdown written to ${SUMMARY_MD}"
}

main "$@"
