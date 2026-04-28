#!/usr/bin/env bash

set -euo pipefail

DEVICE="${DEVICE:-}"
MODEL_PATH="${MODEL_PATH:-${MODEL:-}}"
BENCH_DIR="${BENCH_DIR:-/data/local/tmp/acom-qnn-phase-materializer/bin}"

NGL="${NGL:-0}"
PROMPT_TOKENS="${PROMPT_TOKENS:-0}"
GEN_TOKENS="${GEN_TOKENS:-128}"
CONTEXT_TOKENS="${CONTEXT_TOKENS:-2048}"
BATCH_TOKENS="${BATCH_TOKENS:-1}"
UBATCH_TOKENS="${UBATCH_TOKENS:-1}"
BENCH_REPEATS="${BENCH_REPEATS:-3}"
MMAP="${MMAP:-0}"
LLAMA_BENCH_FAST_EXIT_VALUE="${LLAMA_BENCH_FAST_EXIT_VALUE:-1}"

OUTPUT_DIR="${OUTPUT_DIR:-docs/out/cpu-core-freq-sweep-$(date -u +%Y%m%d-%H%M%S)}"

# CPU_CASE_LIST entry format:
#   name:MASK:THREADS
#   name:cpus=CPU_LIST:THREADS
#
# Examples:
#   CPU_CASE_LIST="big1:80:1 big2:C0:2 small4:cpus=0,1,2,3:4"
#
# When empty, the script discovers online CPUs, groups them by cpuinfo_max_freq,
# then creates little1, littleN, big1, and big2 cases where possible.
CPU_CASE_LIST="${CPU_CASE_LIST:-}"

# CPU_FREQ_LIST is in kHz. When empty, each case uses min/mid/max from the first
# touched cpufreq policy's scaling_available_frequencies.
CPU_FREQ_LIST="${CPU_FREQ_LIST:-}"
CPU_FREQ_POINTS="${CPU_FREQ_POINTS:-min mid max}"
CPU_PIN_GOVERNOR="${CPU_PIN_GOVERNOR:-powersave}"
CPU_PIN_POLICY_FILTER="${CPU_PIN_POLICY_FILTER:-}"

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

CPU_SWEEP_SUMMARIZE_ONLY="${CPU_SWEEP_SUMMARIZE_ONLY:-0}"
CPU_SWEEP_SAMPLE_LOG="${CPU_SWEEP_SAMPLE_LOG:-}"

BATTERY_VOLTAGE_PATH="${BATTERY_VOLTAGE_PATH:-}"
BATTERY_CURRENT_PATH="${BATTERY_CURRENT_PATH:-}"
TEMP_PATH="${TEMP_PATH:-}"

declare -a CPU_IDS=()
declare -a CPU_POLICY_PATHS=()
declare -a CPU_CASE_NAMES=()
declare -a CPU_CASE_MASKS=()
declare -a CPU_CASE_THREADS=()
declare -a CPU_CASE_CPUS=()
declare -A CPU_MAX_FREQ_BY_ID=()
declare -A CPU_POLICY_BY_ID=()
declare -A ORIG_CPU_MIN_FREQ=()
declare -A ORIG_CPU_MAX_FREQ=()
declare -A ORIG_CPU_GOVERNOR=()

RESULTS_CSV=""
BASELINE_CSV=""
BASELINE_AVG_POWER_MW=""
BASELINE_AVG_TEMP_C=""
TEMP_SCALE_TO_MC=""
TEMP_LIMIT_MC=""
COOLDOWN_TEMP_MC=""
LOCAL_BENCH_PID=""

ORIG_SCREEN_OFF_TIMEOUT=""
ORIG_SCREEN_BRIGHTNESS=""
ORIG_SCREEN_BRIGHTNESS_MODE=""
ORIG_STAY_ON_WHILE_PLUGGED_IN=""
DISPLAY_STATE_SAVED=0
CPU_STATE_SAVED=0

log() {
    printf '[cpu-core-freq-sweep] %s\n' "$*"
}

die() {
    printf '[cpu-core-freq-sweep] ERROR: %s\n' "$*" >&2
    exit 1
}

adb_shell() {
    adb -s "${DEVICE}" shell "$@" | tr -d '\r'
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

verify_screen_on() {
    adb -s "${DEVICE}" shell "dumpsys display 2>/dev/null | grep -Eq 'mScreenState=ON|state ON, committedState ON'" >/dev/null 2>&1
}

restore_display_state() {
    if [[ "${DISPLAY_STATE_SAVED}" != "1" ]]; then
        return 0
    fi

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

cpu_list_to_mask() {
    local cpu_list="$1"
    local mask=0
    local item

    IFS=',' read -ra items <<< "${cpu_list}"
    for item in "${items[@]}"; do
        if [[ "${item}" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            local start="${BASH_REMATCH[1]}"
            local end="${BASH_REMATCH[2]}"
            local cpu
            for (( cpu = start; cpu <= end; cpu++ )); do
                mask=$(( mask | (1 << cpu) ))
            done
        elif [[ "${item}" =~ ^[0-9]+$ ]]; then
            mask=$(( mask | (1 << item) ))
        else
            die "invalid CPU list item: ${item}"
        fi
    done

    printf '%X\n' "${mask}"
}

mask_to_cpu_list() {
    local mask_hex="$1"
    mask_hex="${mask_hex#0x}"
    mask_hex="${mask_hex#0X}"
    [[ "${mask_hex}" =~ ^[0-9a-fA-F]+$ ]] || die "invalid taskset mask: ${mask_hex}"

    local mask=$(( 16#${mask_hex} ))
    local cpu
    local sep=""
    for (( cpu = 0; cpu < 32; cpu++ )); do
        if (( mask & (1 << cpu) )); then
            printf '%s%s' "${sep}" "${cpu}"
            sep=","
        fi
    done
    printf '\n'
}

count_cpu_list() {
    local cpu_list="$1"
    awk -F, '{
        n = 0
        for (i = 1; i <= NF; ++i) {
            if ($i != "") {
                n++
            }
        }
        print n
    }' <<< "${cpu_list}"
}

first_n_cpus() {
    local cpu_list="$1"
    local want="$2"
    awk -F, -v want="${want}" '{
        out = ""
        count = 0
        for (i = 1; i <= NF && count < want; ++i) {
            if ($i == "") {
                continue
            }
            out = out == "" ? $i : out "," $i
            count++
        }
        print out
    }' <<< "${cpu_list}"
}

join_policy_paths_for_cpus() {
    local cpu_list="$1"
    local cpu
    local policy
    local seen=" "
    local out=""

    IFS=',' read -ra cpus <<< "${cpu_list}"
    for cpu in "${cpus[@]}"; do
        policy="${CPU_POLICY_BY_ID[${cpu}]:-}"
        [[ -n "${policy}" ]] || die "failed to map cpu${cpu} to a cpufreq policy"
        if [[ "${seen}" != *" ${policy} "* ]]; then
            seen+="${policy} "
            out="${out}${out:+;}${policy}"
        fi
    done

    printf '%s\n' "${out}"
}

filter_policy_paths() {
    local policy_paths="$1"
    local filters="${CPU_PIN_POLICY_FILTER}"
    local policy
    local base
    local out=""

    if [[ -z "${filters}" ]]; then
        printf '%s\n' "${policy_paths}"
        return 0
    fi

    IFS=';' read -ra policies <<< "${policy_paths}"
    for policy in "${policies[@]}"; do
        [[ -n "${policy}" ]] || continue
        base="${policy##*/}"
        if [[ ",${filters}," == *",${base},"* || ",${filters}," == *",${policy},"* ]]; then
            out="${out}${out:+;}${policy}"
        fi
    done

    [[ -n "${out}" ]] || die "CPU_PIN_POLICY_FILTER=${CPU_PIN_POLICY_FILTER} filtered out all policy paths from ${policy_paths}"
    printf '%s\n' "${out}"
}

add_cpu_case() {
    local name="$1"
    local cpu_list="$2"
    local threads="$3"
    local mask

    [[ -n "${name}" ]] || die "CPU case name is empty"
    [[ -n "${cpu_list}" ]] || die "CPU case ${name} has empty CPU list"
    [[ "${threads}" =~ ^[0-9]+$ && "${threads}" -gt 0 ]] || die "CPU case ${name} has invalid thread count: ${threads}"

    mask="$(cpu_list_to_mask "${cpu_list}")"
    CPU_CASE_NAMES+=("${name}")
    CPU_CASE_MASKS+=("${mask}")
    CPU_CASE_THREADS+=("${threads}")
    CPU_CASE_CPUS+=("${cpu_list}")
}

discover_cpu_topology() {
    local policies
    policies="$(adb_root_capture 'for p in /sys/devices/system/cpu/cpufreq/policy*; do [ -d "$p" ] && echo "$p"; done')"
    mapfile -t CPU_POLICY_PATHS < <(printf '%s\n' "${policies}" | awk 'NF')
    (( ${#CPU_POLICY_PATHS[@]} > 0 )) || die "failed to discover cpufreq policy paths"

    local rows
    rows="$(adb_root_capture '
for d in /sys/devices/system/cpu/cpu[0-9]*; do
    id="${d##*cpu}"
    case "$id" in
        ""|*[!0-9]*) continue ;;
    esac
    online=1
    if [ -r "$d/online" ]; then
        online="$(cat "$d/online" 2>/dev/null || echo 0)"
    fi
    [ "$online" = "1" ] || continue

    max_freq=""
    if [ -r "$d/cpufreq/cpuinfo_max_freq" ]; then
        max_freq="$(cat "$d/cpufreq/cpuinfo_max_freq" 2>/dev/null || true)"
    elif [ -r "$d/cpufreq/scaling_max_freq" ]; then
        max_freq="$(cat "$d/cpufreq/scaling_max_freq" 2>/dev/null || true)"
    fi
    case "$max_freq" in
        ""|*[!0-9]*) continue ;;
    esac

    policy=""
    for p in /sys/devices/system/cpu/cpufreq/policy*; do
        [ -d "$p" ] || continue
        related="$(cat "$p/related_cpus" 2>/dev/null || cat "$p/affected_cpus" 2>/dev/null || true)"
        for c in $related; do
            case "$c" in
                *-*)
                    start="${c%-*}"
                    end="${c#*-}"
                    if [ "$id" -ge "$start" ] 2>/dev/null && [ "$id" -le "$end" ] 2>/dev/null; then
                        policy="$p"
                        break
                    fi
                    ;;
                *)
                    if [ "$c" = "$id" ]; then
                        policy="$p"
                        break
                    fi
                    ;;
            esac
        done
        [ -n "$policy" ] && break
    done
    if [ -z "$policy" ] && [ -d "$d/cpufreq" ]; then
        policy="$d/cpufreq"
    fi
    [ -n "$policy" ] || continue
    printf "%s,%s,%s\n" "$id" "$max_freq" "$policy"
done')"

    local row
    while IFS=, read -r cpu max_freq policy; do
        [[ -n "${cpu}" ]] || continue
        CPU_IDS+=("${cpu}")
        CPU_MAX_FREQ_BY_ID["${cpu}"]="${max_freq}"
        CPU_POLICY_BY_ID["${cpu}"]="${policy}"
    done <<< "${rows}"

    (( ${#CPU_IDS[@]} > 0 )) || die "failed to discover online CPUs with cpufreq"
}

discover_case_list() {
    if [[ -n "${CPU_CASE_LIST}" ]]; then
        local entry
        for entry in ${CPU_CASE_LIST}; do
            local name selector threads cpu_list mask
            IFS=: read -r name selector threads <<< "${entry}"
            [[ -n "${name}" && -n "${selector}" && -n "${threads}" ]] || die "invalid CPU_CASE_LIST entry: ${entry}"

            if [[ "${selector}" == cpus=* ]]; then
                cpu_list="${selector#cpus=}"
            else
                mask="${selector}"
                cpu_list="$(mask_to_cpu_list "${mask}")"
            fi
            add_cpu_case "${name}" "${cpu_list}" "${threads}"
        done
        return 0
    fi

    local unique_freqs
    unique_freqs="$(for cpu in "${CPU_IDS[@]}"; do printf '%s\n' "${CPU_MAX_FREQ_BY_ID[${cpu}]}"; done | sort -n | uniq)"

    local low_freq high_freq
    low_freq="$(printf '%s\n' "${unique_freqs}" | head -n 1)"
    high_freq="$(printf '%s\n' "${unique_freqs}" | tail -n 1)"

    local little_cpus=""
    local big_cpus=""
    local cpu
    for cpu in "${CPU_IDS[@]}"; do
        if [[ "${CPU_MAX_FREQ_BY_ID[${cpu}]}" == "${low_freq}" ]]; then
            little_cpus="${little_cpus}${little_cpus:+,}${cpu}"
        fi
        if [[ "${CPU_MAX_FREQ_BY_ID[${cpu}]}" == "${high_freq}" ]]; then
            big_cpus="${big_cpus}${big_cpus:+,}${cpu}"
        fi
    done

    local little_count
    little_count="$(count_cpu_list "${little_cpus}")"
    if (( little_count > 0 )); then
        add_cpu_case "little1" "$(first_n_cpus "${little_cpus}" 1)" 1
        if (( little_count > 1 )); then
            add_cpu_case "little${little_count}" "${little_cpus}" "${little_count}"
        fi
    fi

    if [[ "${high_freq}" != "${low_freq}" ]]; then
        local big_count
        big_count="$(count_cpu_list "${big_cpus}")"
        if (( big_count > 0 )); then
            add_cpu_case "big1" "$(first_n_cpus "${big_cpus}" 1)" 1
            if (( big_count > 1 )); then
                add_cpu_case "big2" "$(first_n_cpus "${big_cpus}" 2)" 2
            fi
        fi
    fi

    (( ${#CPU_CASE_NAMES[@]} > 0 )) || die "no CPU cases discovered"
}

validate_cpu_governor() {
    if [[ "${CPU_PIN_GOVERNOR}" == "walt" ]]; then
        die "CPU_PIN_GOVERNOR=walt is forbidden for frequency pinning; use powersave"
    fi
    if [[ "${CPU_PIN_GOVERNOR}" != "powersave" ]]; then
        die "CPU_PIN_GOVERNOR must be powersave for this experiment; got ${CPU_PIN_GOVERNOR}"
    fi
}

save_original_cpu_state() {
    local policy
    for policy in "${CPU_POLICY_PATHS[@]}"; do
        ORIG_CPU_MIN_FREQ["${policy}"]="$(read_remote_value "${policy}/scaling_min_freq")"
        ORIG_CPU_MAX_FREQ["${policy}"]="$(read_remote_value "${policy}/scaling_max_freq")"
        ORIG_CPU_GOVERNOR["${policy}"]="$(read_remote_value "${policy}/scaling_governor")"
    done
    CPU_STATE_SAVED=1
}

restore_original_cpu_state() {
    if [[ "${CPU_STATE_SAVED}" != "1" ]]; then
        return 0
    fi

    local policy
    for policy in "${CPU_POLICY_PATHS[@]:-}"; do
        local orig_min="${ORIG_CPU_MIN_FREQ[${policy}]:-}"
        local orig_max="${ORIG_CPU_MAX_FREQ[${policy}]:-}"
        local orig_gov="${ORIG_CPU_GOVERNOR[${policy}]:-}"

        if [[ -n "${orig_min}" && -n "${orig_max}" ]]; then
            adb_root_shell "echo ${orig_min} > ${policy}/scaling_min_freq && echo ${orig_max} > ${policy}/scaling_max_freq" || true
        fi
        if [[ -n "${orig_gov}" ]]; then
            adb_root_shell "echo ${orig_gov} > ${policy}/scaling_governor" || true
        fi
    done
}

policy_available_freqs() {
    local policy="$1"
    local raw
    raw="$(adb_root_capture "cat ${policy}/scaling_available_frequencies 2>/dev/null || true")"
    if [[ -z "${raw}" ]]; then
        local min_freq max_freq
        min_freq="$(read_remote_value "${policy}/cpuinfo_min_freq")"
        max_freq="$(read_remote_value "${policy}/cpuinfo_max_freq")"
        [[ -n "${min_freq}" && -n "${max_freq}" ]] || die "failed to read available frequencies for ${policy}"
        raw="${min_freq} ${max_freq}"
    fi
    printf '%s\n' "${raw}" | tr ' ' '\n' | awk '/^[0-9]+$/' | sort -n | uniq
}

derive_freqs_for_policy() {
    local policy="$1"
    if [[ -n "${CPU_FREQ_LIST}" ]]; then
        printf '%s\n' "${CPU_FREQ_LIST}" | tr ', ' '\n\n' | awk '/^[0-9]+$/' | sort -n | uniq
        return 0
    fi

    mapfile -t freqs < <(policy_available_freqs "${policy}")
    (( ${#freqs[@]} > 0 )) || die "no available frequencies for ${policy}"

    local point
    local selected=()
    for point in ${CPU_FREQ_POINTS}; do
        case "${point}" in
            min)
                selected+=("${freqs[0]}")
                ;;
            mid)
                selected+=("${freqs[$(( ${#freqs[@]} / 2 ))]}")
                ;;
            max)
                selected+=("${freqs[$(( ${#freqs[@]} - 1 ))]}")
                ;;
            *)
                if [[ "${point}" =~ ^[0-9]+$ ]]; then
                    selected+=("${point}")
                else
                    die "unsupported CPU_FREQ_POINTS item: ${point}"
                fi
                ;;
        esac
    done

    printf '%s\n' "${selected[@]}" | awk '/^[0-9]+$/' | sort -n | uniq
}

select_policy_freq() {
    local policy="$1"
    local requested="$2"
    policy_available_freqs "${policy}" | awk -v target="${requested}" '
        function abs(x) { return x < 0 ? -x : x }
        {
            diff = abs(($1 + 0) - target)
            if (best == "" || diff < best_diff) {
                best = $1
                best_diff = diff
            }
        }
        END {
            if (best != "") {
                print best
            }
        }'
}

pin_cpu_policy_freq() {
    local policy="$1"
    local requested="$2"
    local selected
    local cpuinfo_min
    local can_write_min=0
    local can_write_max=0
    local readback_gov
    local readback_min
    local readback_max
    local readback_cur
    selected="$(select_policy_freq "${policy}" "${requested}")"
    [[ -n "${selected}" ]] || {
        printf 'failed to select CPU frequency for %s\n' "${policy}" >&2
        return 1
    }
    cpuinfo_min="$(read_remote_value "${policy}/cpuinfo_min_freq")"
    [[ -n "${cpuinfo_min}" ]] || cpuinfo_min="${selected}"

    if adb_test "[ -w ${policy}/scaling_min_freq ]"; then
        can_write_min=1
    fi
    if adb_test "[ -w ${policy}/scaling_max_freq ]"; then
        can_write_max=1
    fi

    adb_root_shell "echo ${CPU_PIN_GOVERNOR} > ${policy}/scaling_governor" \
        || {
            printf 'failed to set governor %s for %s\n' "${CPU_PIN_GOVERNOR}" "${policy}" >&2
            return 1
        }

    if (( can_write_min == 1 )); then
        adb_root_shell "echo ${cpuinfo_min} > ${policy}/scaling_min_freq" || true
    fi
    if (( can_write_max == 1 )); then
        adb_root_shell "echo ${selected} > ${policy}/scaling_max_freq" || true
    fi
    if (( can_write_min == 1 )); then
        adb_root_shell "echo ${selected} > ${policy}/scaling_min_freq" || true
    fi

    readback_gov="$(read_remote_value "${policy}/scaling_governor")"
    readback_min="$(read_remote_value "${policy}/scaling_min_freq")"
    readback_max="$(read_remote_value "${policy}/scaling_max_freq")"
    readback_cur="$(read_remote_value "${policy}/scaling_cur_freq")"
    if [[ "${readback_gov}" != "${CPU_PIN_GOVERNOR}" || "${readback_max}" != "${selected}" || "${readback_cur}" != "${selected}" ]]; then
        printf 'pin verification failed for %s: requested=%s readback_gov=%s readback_min=%s readback_max=%s readback_cur=%s writable_min=%s writable_max=%s\n' \
            "${policy}" \
            "${selected}" \
            "${readback_gov:-NA}" \
            "${readback_min:-NA}" \
            "${readback_max:-NA}" \
            "${readback_cur:-NA}" \
            "${can_write_min}" \
            "${can_write_max}" >&2
        return 1
    fi

    printf '%s\n' "${selected}"
}

pin_cpu_case_freq() {
    local policy_paths="$1"
    local requested="$2"
    local policy
    local selected
    local pinned=""

    IFS=';' read -ra policies <<< "${policy_paths}"
    for policy in "${policies[@]}"; do
        [[ -n "${policy}" ]] || continue
        if ! selected="$(pin_cpu_policy_freq "${policy}" "${requested}")"; then
            return 1
        fi
        pinned="${pinned}${pinned:+;}${policy}=${selected}"
    done

    printf '%s\n' "${pinned}"
}

primary_policy_for_case() {
    local cpu_list="$1"
    local first_cpu="${cpu_list%%,*}"
    local policy="${CPU_POLICY_BY_ID[${first_cpu}]:-}"
    [[ -n "${policy}" ]] || die "failed to find primary policy for CPU case ${cpu_list}"
    printf '%s\n' "${policy}"
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
    local taskset_mask="$1"
    local threads="$2"
    local primary_policy="$3"
    local bench_log="$4"
    local sample_log="$5"
    local meta_log="$6"

    : > "${sample_log}"
    : > "${meta_log}"

    adb -s "${DEVICE}" shell "cd ${BENCH_DIR} && \
export LD_LIBRARY_PATH=${BENCH_DIR} && \
export ADSP_LIBRARY_PATH=${BENCH_DIR} && \
export LLAMA_BENCH_FAST_EXIT=${LLAMA_BENCH_FAST_EXIT_VALUE} && \
taskset ${taskset_mask} ./llama-bench -v \
  -m ${MODEL_PATH} \
  -ngl ${NGL} -t ${threads} \
  -p ${PROMPT_TOKENS} -n ${GEN_TOKENS} \
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
        local cpu_freq

        ts="$(date +%s)"
        voltage="$(read_remote_value "${BATTERY_VOLTAGE_PATH}")"
        current="$(read_remote_value "${BATTERY_CURRENT_PATH}")"
        temp_raw="$(read_remote_value "${TEMP_PATH}")"
        temp_mc="$(normalize_temp_to_mc "${temp_raw}" "${TEMP_SCALE_TO_MC}")"
        cpu_freq="$(read_remote_value "${primary_policy}/scaling_cur_freq")"
        [[ -n "${cpu_freq}" ]] || cpu_freq="0"

        printf '%s,%s,%s,%s,%s\n' "${ts}" "${voltage}" "${current}" "${temp_mc}" "${cpu_freq}" >> "${sample_log}"

        if (( temp_mc >= TEMP_LIMIT_MC )); then
            printf 'THERMAL_ABORT,%s,%s\n' "${ts}" "${temp_mc}" > "${meta_log}"
            kill "${LOCAL_BENCH_PID}" >/dev/null 2>&1 || true
            adb_root_shell "pkill -INT llama-bench 2>/dev/null || true" || true
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
        printf 'NA,NA,NA,NA,NA,0,0,0,NA,NA\n'
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
                print "NA,NA,NA,NA,NA,0,0,0,NA,NA"
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

            printf "%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d,%.2f,%.0f\n", \
                sum_power / count, \
                (sum_temp / count) / 1000.0, \
                max_temp / 1000.0, \
                temp[stable_start] / 1000.0, \
                temp[stable_end] / 1000.0, \
                count, \
                stable_start, \
                stable_end, \
                stable_range, \
                sum_freq / count
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
    local case_name="$1"
    local cpu_list="$2"
    local taskset_mask="$3"
    local threads="$4"
    local requested_freq_khz="$5"
    local pinned_policy_freqs="$6"
    local status="$7"
    local exit_code="$8"
    local avg_power_mw="$9"
    local delta_vs_baseline_mw="${10}"
    local avg_temp_c="${11}"
    local max_temp_c="${12}"
    local start_temp_c="${13}"
    local end_temp_c="${14}"
    local throughput_tok_s="${15}"
    local throughput_label="${16}"
    local sample_count="${17}"
    local stable_start_index="${18}"
    local stable_end_index="${19}"
    local stable_range_pct="${20}"
    local avg_cpu_freq_khz="${21}"
    local bench_log="${22}"
    local sample_log="${23}"
    local csv_cpu_list="${cpu_list//,/+}"

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${case_name}" \
        "${csv_cpu_list}" \
        "${taskset_mask}" \
        "${threads}" \
        "${requested_freq_khz}" \
        "${pinned_policy_freqs}" \
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
        "${avg_cpu_freq_khz}" \
        "${bench_log}" \
        "${sample_log}" >> "${RESULTS_CSV}"
}

cleanup() {
    if [[ -n "${LOCAL_BENCH_PID}" && "${LOCAL_BENCH_PID}" =~ ^[0-9]+$ ]]; then
        kill -TERM "${LOCAL_BENCH_PID}" >/dev/null 2>&1 || true
    fi
    restore_display_state
    restore_original_cpu_state
}

main() {
    validate_cpu_governor

    mkdir -p "${OUTPUT_DIR}"
    RESULTS_CSV="${OUTPUT_DIR}/results.csv"
    BASELINE_CSV="${OUTPUT_DIR}/baseline.samples.csv"

    printf '%s\n' 'case_name,cpu_list,taskset_mask,threads,requested_freq_khz,pinned_policy_freqs,status,bench_exit_code,avg_power_mw,delta_vs_baseline_mw,avg_temp_c,max_temp_c,start_temp_c,end_temp_c,throughput_tok_s,throughput_label,sample_count,stable_start_index,stable_end_index,stable_range_pct,avg_cpu_freq_khz,bench_log,sample_log' > "${RESULTS_CSV}"

    require_runtime_inputs
    check_device_online
    save_display_state
    ensure_screen_on
    verify_screen_on || die "failed to keep the device screen ON before the sweep starts"
    discover_sysfs_paths
    discover_cpu_topology
    discover_case_list
    save_original_cpu_state

    log "run build-npu-opencl.sh before formal measurements when binary freshness matters"
    log "output dir: ${OUTPUT_DIR}"
    log "decode bench: -ngl ${NGL} -p ${PROMPT_TOKENS} -n ${GEN_TOKENS} -c ${CONTEXT_TOKENS} -b ${BATCH_TOKENS} -ub ${UBATCH_TOKENS} -r ${BENCH_REPEATS}"
    log "cpu governor for frequency pinning: ${CPU_PIN_GOVERNOR}"
    log "temperature limit: ${TEMP_LIMIT_C}C"

    local case_index
    for case_index in "${!CPU_CASE_NAMES[@]}"; do
        log "case ${CPU_CASE_NAMES[${case_index}]}: cpus=${CPU_CASE_CPUS[${case_index}]} taskset=${CPU_CASE_MASKS[${case_index}]} threads=${CPU_CASE_THREADS[${case_index}]}"
    done

    log "waiting for cooldown before baseline sampling"
    wait_for_cooldown
    ensure_screen_on
    verify_screen_on || die "screen turned OFF before baseline sampling"

    log "sampling baseline idle power"
    sample_baseline "${BASELINE_CSV}"
    IFS=',' read -r BASELINE_AVG_POWER_MW BASELINE_AVG_TEMP_C <<< "$(summarize_baseline "${BASELINE_CSV}")"
    log "baseline: avg_power_mw=${BASELINE_AVG_POWER_MW} avg_temp_c=${BASELINE_AVG_TEMP_C} sample_log=${BASELINE_CSV}"

    for case_index in "${!CPU_CASE_NAMES[@]}"; do
        local case_name="${CPU_CASE_NAMES[${case_index}]}"
        local taskset_mask="${CPU_CASE_MASKS[${case_index}]}"
        local threads="${CPU_CASE_THREADS[${case_index}]}"
        local cpu_list="${CPU_CASE_CPUS[${case_index}]}"
        local primary_policy
        local policy_paths
        primary_policy="$(primary_policy_for_case "${cpu_list}")"
        policy_paths="$(join_policy_paths_for_cpus "${cpu_list}")"
        policy_paths="$(filter_policy_paths "${policy_paths}")"

        local -a case_freqs=()
        mapfile -t case_freqs < <(derive_freqs_for_policy "${primary_policy}")
        (( ${#case_freqs[@]} > 0 )) || die "no CPU frequencies for case ${case_name}"

        local requested_freq
        for requested_freq in "${case_freqs[@]}"; do
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
            local avg_cpu_freq_khz
            local throughput_tok_s
            local throughput_label
            local pinned_policy_freqs

            run_name="cpu_${case_name}_${requested_freq}khz"
            bench_log="${OUTPUT_DIR}/${run_name}.bench.log"
            sample_log="${OUTPUT_DIR}/${run_name}.samples.csv"
            meta_log="${OUTPUT_DIR}/${run_name}.meta"

            log "waiting for cooldown before ${run_name}"
            wait_for_cooldown

            ensure_screen_on
            verify_screen_on || die "screen turned OFF before ${run_name}"

            log "pinning ${policy_paths} to requested ${requested_freq} kHz with governor ${CPU_PIN_GOVERNOR}"
            if ! pinned_policy_freqs="$(pin_cpu_case_freq "${policy_paths}" "${requested_freq}")"; then
                die "failed to pin ${policy_paths} to requested ${requested_freq} kHz"
            fi
            sleep 1

            log "starting benchmark for ${run_name}"
            run_result="$(run_bench_with_local_sampling "${taskset_mask}" "${threads}" "${primary_policy}" "${bench_log}" "${sample_log}" "${meta_log}")"
            IFS=',' read -r bench_exit_code status <<< "${run_result}"

            if [[ "${status}" == "ok" && "${bench_exit_code}" != "0" ]]; then
                status="bench_failed"
            fi

            sample_summary="$(summarize_samples "${sample_log}")"
            IFS=',' read -r avg_power_mw avg_temp_c max_temp_c start_temp_c end_temp_c sample_count stable_start_index stable_end_index stable_range_pct avg_cpu_freq_khz <<< "${sample_summary}"

            throughput_summary="$(extract_throughput "${bench_log}")"
            IFS=',' read -r throughput_tok_s throughput_label <<< "${throughput_summary}"

            delta_vs_baseline_mw="$(compute_delta_vs_baseline "${avg_power_mw}")"

            append_result \
                "${case_name}" \
                "${cpu_list}" \
                "${taskset_mask}" \
                "${threads}" \
                "${requested_freq}" \
                "${pinned_policy_freqs}" \
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
                "${avg_cpu_freq_khz}" \
                "${bench_log}" \
                "${sample_log}"

            log "finished ${run_name}: status=${status} avg_power_mw=${avg_power_mw} delta_vs_baseline_mw=${delta_vs_baseline_mw} avg_cpu_freq_khz=${avg_cpu_freq_khz} stable_window=${stable_start_index}-${stable_end_index} stable_range_pct=${stable_range_pct} throughput=${throughput_tok_s} label=${throughput_label} max_temp=${max_temp_c}C"
        done
    done

    log "results written to ${RESULTS_CSV}"
}

if [[ "${CPU_SWEEP_SUMMARIZE_ONLY}" == "1" ]]; then
    [[ -n "${CPU_SWEEP_SAMPLE_LOG}" ]] || die "CPU_SWEEP_SAMPLE_LOG is required when CPU_SWEEP_SUMMARIZE_ONLY=1"
    summarize_samples "${CPU_SWEEP_SAMPLE_LOG}"
    exit 0
fi

trap cleanup EXIT
main "$@"
