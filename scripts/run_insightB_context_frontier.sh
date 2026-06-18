#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PROFILE_HEADER='date,model,backend,state_id,context_len,effective_prefill_tokens,decode_tokens,rounds,phase_isolated,throughput_tps,throughput_std,active_power_mw,power_std,energy_mj_per_token,tbt_us,temperature_avg_c,temp_max_c,stable_range_pct,freq_stable,actual_gpu_freq_mhz,actual_cpu_freq_khz,qnn_aot_cache_size,qnn_aot_context_size,support_status,fallback_used,raw_log_path,sample_path'

log() {
    printf '[insightB-context-frontier] %s\n' "$*"
}

die() {
    printf '[insightB-context-frontier] ERROR: %s\n' "$*" >&2
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

ensure_profile_header() {
    local path="$1"
    mkdir -p "$(dirname "${path}")"
    if [[ ! -s "${path}" ]]; then
        printf '%s\n' "${PROFILE_HEADER}" > "${path}"
        return 0
    fi

    local current_header
    current_header="$(sed -n '1p' "${path}")"
    if [[ "${current_header}" == "${PROFILE_HEADER}" ]]; then
        return 0
    fi

    die "unexpected CSV header in ${path}"
}

blank_na() {
    local value="${1:-}"
    if [[ -z "${value}" || "${value}" == "NA" ]]; then
        printf ''
    else
        printf '%s' "${value}"
    fi
}

is_number() {
    [[ "${1:-}" =~ ^-?[0-9]+([.][0-9]+)?$ ]]
}

compute_energy_mj_per_token() {
    local active_power_mw="$1"
    local throughput_tps="$2"
    if ! is_number "${active_power_mw}" || ! is_number "${throughput_tps}"; then
        printf ''
        return 0
    fi
    awk -v p="${active_power_mw}" -v t="${throughput_tps}" 'BEGIN { if (t > 0) printf "%.6f", p / t }'
}

compute_tbt_us() {
    local throughput_tps="$1"
    if ! is_number "${throughput_tps}"; then
        printf ''
        return 0
    fi
    awk -v t="${throughput_tps}" 'BEGIN { if (t > 0) printf "%.2f", 1000000.0 / t }'
}

compute_power_cv_pct() {
    local active_power_mw="$1"
    local power_std="$2"
    if ! is_number "${active_power_mw}" || ! is_number "${power_std}"; then
        printf ''
        return 0
    fi
    awk -v p="${active_power_mw}" -v s="${power_std}" 'BEGIN { if (p > 0) printf "%.2f", 100.0 * s / p }'
}

classify_data_quality() {
    local support_status="$1"
    local rounds="$2"
    local power_cv_pct="${3:-}"
    local sample_count="${4:-}"
    local unstable_threshold="${POWER_CV_UNSTABLE_PCT:-10.0}"
    local expected_window_samples="${ACTIVE_WINDOW_SAMPLES:-4}"

    case "${support_status}" in
        unsupported_*)
            printf 'unsupported'
            return 0
            ;;
        failed|'')
            printf 'failed'
            return 0
            ;;
    esac

    if [[ "${support_status}" != "ok" ]]; then
        printf 'failed'
        return 0
    fi

    if [[ "${rounds}" =~ ^[0-9]+$ ]] && (( rounds == 1 )); then
        printf 'smoke_only'
        return 0
    fi

    if [[ "${sample_count}" =~ ^[0-9]+$ && "${expected_window_samples}" =~ ^[0-9]+$ ]] && (( sample_count < expected_window_samples )); then
        printf 'unstable_power_window'
        return 0
    fi

    if ! is_number "${power_cv_pct}"; then
        printf 'failed'
        return 0
    fi

    if awk -v cv="${power_cv_pct}" -v threshold="${unstable_threshold}" 'BEGIN { exit !(cv > threshold) }'; then
        printf 'unstable_power_window'
        return 0
    fi

    printf 'paper_ready'
}

format_mhz_from_hz() {
    local hz="$1"
    if ! is_number "${hz}"; then
        printf ''
        return 0
    fi
    awk -v x="${hz}" 'BEGIN { printf "%.0f", x / 1000000.0 }'
}

max_int() {
    local a="$1"
    local b="$2"
    if (( a > b )); then
        printf '%s\n' "${a}"
    else
        printf '%s\n' "${b}"
    fi
}

append_optional_env() {
    local array_name="$1"
    shift

    local env_name
    local env_value
    for env_name in "$@"; do
        env_value="${!env_name:-}"
        if [[ -n "${env_value}" ]]; then
            eval "${array_name}+=(\"\${env_name}=\${env_value}\")"
        fi
    done
}

parse_llama_bench_tg() {
    local bench_log="$1"
    local decode_tokens="$2"
    local context_len="$3"

    if [[ ! -s "${bench_log}" ]]; then
        printf ',,,'
        return 0
    fi

    awk -F'|' -v decode_tokens="${decode_tokens}" -v context_len="${context_len}" '
        function trim(s) {
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", s)
            return s
        }
        function norm(s) {
            s = trim(s)
            gsub(/[[:space:]]+/, " ", s)
            return s
        }
        BEGIN {
            wanted = "tg" decode_tokens
            if (context_len > 0) {
                wanted = wanted " @ d" context_len
            }
        }
        /\|/ {
            for (i = 1; i < NF; ++i) {
                label = norm($i)
                if (label == wanted) {
                    value = norm($(i + 1))
                    gsub(/tok\/s|t\/s/, "", value)
                    gsub(/±/, "+/-", value)
                    split(value, parts, /\+\/-/)
                    tps = norm(parts[1])
                    std = norm(parts[2])
                    found_tps = tps
                    found_std = std
                    found_label = label
                }
            }
        }
        END {
            if (found_tps != "") {
                printf "%s,%s,1,%s", found_tps, found_std, found_label
            } else {
                printf ",,,"
            }
        }' "${bench_log}"
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

graphs_with_sizes = [
    g for g in data.get("graphs", [])
    if "cache_size" in g
    and "context_size" in g
]
graphs = [
    g for g in graphs_with_sizes
    if g.get("type") == "transformers"
]
if not graphs:
    graphs = graphs_with_sizes
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

qnn_cache_guard_status() {
    local effective_prefill_tokens="$1"
    local decode_tokens="$2"
    local safety_margin="$3"
    local cache_size="$4"

    local footprint=$(( effective_prefill_tokens + decode_tokens + safety_margin ))
    printf '%s\n' "${footprint}"
    (( footprint <= cache_size ))
}

summarize_power_samples() {
    local sample_log="$1"
    local settle_seconds="${POWER_SETTLE_SECONDS:-8}"
    local active_window_samples="${ACTIVE_WINDOW_SAMPLES:-4}"
    local active_min_window_samples="${ACTIVE_MIN_WINDOW_SAMPLES:-2}"
    local active_window_max_range_pct="${ACTIVE_WINDOW_MAX_RANGE_PCT:-10.0}"

    if [[ ! -s "${sample_log}" ]]; then
        printf ',,,,,,,,'
        return 0
    fi

    awk -F, \
        -v settle_s="${settle_seconds}" \
        -v active_n="${active_window_samples}" \
        -v min_active_n="${active_min_window_samples}" \
        -v max_range_pct="${active_window_max_range_pct}" '
        function abs(x) { return x < 0 ? -x : x }
        function eval_window(start, len,    j, min_power, max_power, sum_power, avg_power, range_pct) {
            min_power = power[start]
            max_power = power[start]
            sum_power = 0
            for (j = start; j < start + len; ++j) {
                if (power[j] < min_power) {
                    min_power = power[j]
                }
                if (power[j] > max_power) {
                    max_power = power[j]
                }
                sum_power += power[j]
            }
            avg_power = sum_power / len
            range_pct = avg_power > 0 ? ((max_power - min_power) / avg_power * 100.0) : 0
            candidate_avg = avg_power
            candidate_range = range_pct
        }
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
            if (NF >= 5 && $5 != "") {
                has_freq = 1
                freq[n] = $5 + 0
            }
        }
        END {
            if (n == 0) {
                printf ",,,,,,,,"
                exit 0
            }

            window_n = active_n + 0
            if (window_n < 1) {
                window_n = 1
            }
            if (min_active_n < 1) {
                min_active_n = 1
            }
            if (window_n > n) {
                window_n = n
            }
            if (min_active_n > window_n) {
                min_active_n = window_n
            }

            stable_start = 1
            stable_end = window_n
            stable_range = 0
            best_any_avg = -1
            best_any_start = 1
            best_any_range = 0
            best_stable_avg = -1
            best_stable_start = 1
            best_stable_range = 0
            for (i = 1; i <= n - window_n + 1; ++i) {
                eval_window(i, window_n)
                if (candidate_avg > best_any_avg) {
                    best_any_avg = candidate_avg
                    best_any_start = i
                    best_any_range = candidate_range
                }
                if (candidate_range <= max_range_pct && candidate_avg > best_stable_avg) {
                    best_stable_avg = candidate_avg
                    best_stable_start = i
                    best_stable_range = candidate_range
                }
            }

            used_stable_full_window = 0
            if (best_stable_avg >= 0) {
                stable_start = best_stable_start
                stable_end = best_stable_start + window_n - 1
                stable_range = best_stable_range
                used_stable_full_window = 1
            } else {
                stable_start = best_any_start
                stable_end = best_any_start + window_n - 1
                stable_range = best_any_range
            }

            if (!used_stable_full_window && stable_range > max_range_pct && window_n > min_active_n) {
                found_shorter_plateau = 0
                for (len = window_n - 1; len >= min_active_n; --len) {
                    best_len_avg = -1
                    best_len_start = 1
                    best_len_range = 0
                    for (i = 1; i <= n - len + 1; ++i) {
                        eval_window(i, len)
                        if (candidate_range <= max_range_pct && candidate_avg > best_len_avg) {
                            best_len_avg = candidate_avg
                            best_len_start = i
                            best_len_range = candidate_range
                        }
                    }
                    if (best_len_avg >= 0) {
                        stable_start = best_len_start
                        stable_end = best_len_start + len - 1
                        stable_range = best_len_range
                        found_shorter_plateau = 1
                        break
                    }
                }
            }

            sum_power = 0
            sumsq_power = 0
            sum_temp = 0
            max_temp = temp[stable_start]
            count = 0
            if (has_freq) {
                sum_freq = 0
                min_freq = freq[stable_start]
                max_freq = freq[stable_start]
            }
            for (i = stable_start; i <= stable_end; ++i) {
                sum_power += power[i]
                sumsq_power += power[i] * power[i]
                sum_temp += temp[i]
                if (temp[i] > max_temp) {
                    max_temp = temp[i]
                }
                if (has_freq) {
                    sum_freq += freq[i]
                    if (freq[i] < min_freq) {
                        min_freq = freq[i]
                    }
                    if (freq[i] > max_freq) {
                        max_freq = freq[i]
                    }
                }
                count++
            }

            avg = sum_power / count
            if (count > 1) {
                variance = (sumsq_power - (sum_power * sum_power / count)) / (count - 1)
                if (variance < 0) {
                    variance = 0
                }
                std = sqrt(variance)
            } else {
                std = 0
            }

            printf "%.2f,%.2f,%.2f,%.2f,%.2f", \
                avg, \
                std, \
                (sum_temp / count) / 1000.0, \
                max_temp / 1000.0, \
                stable_range
            if (has_freq) {
                printf ",%.0f,%.0f,%.0f,%d", sum_freq / count, min_freq, max_freq, count
            } else {
                printf ",,,,%d", count
            }
        }' "${sample_log}"
}

freq_stable_flag() {
    local requested="$1"
    local min_actual="$2"
    local max_actual="$3"
    local tolerance_pct="${FREQ_STABLE_TOLERANCE_PCT:-2.0}"

    if ! is_number "${requested}" || ! is_number "${min_actual}" || ! is_number "${max_actual}"; then
        printf ''
        return 0
    fi

    awk -v req="${requested}" -v minv="${min_actual}" -v maxv="${max_actual}" -v tol="${tolerance_pct}" '
        BEGIN {
            lo = req * (1.0 - tol / 100.0)
            hi = req * (1.0 + tol / 100.0)
            print (minv >= lo && maxv <= hi) ? 1 : 0
        }'
}

detect_fallback_used() {
    local raw_log="$1"
    if [[ -s "${raw_log}" ]] && \
        grep -Eiv 'fallback to libQnn|fallback .*disabled' "${raw_log}" | \
        grep -Eiq 'fallback_used=1|runtime fallback|falling back|fallback to (CPU|GPU|OpenCL|qnn-npu)|replay fallback'; then
        printf '1'
    else
        printf '0'
    fi
}

classify_support_status() {
    local state_id="$1"
    local bench_status="$2"
    local bench_exit_code="$3"
    local throughput_tps="$4"
    local raw_log="$5"

    if [[ "${bench_status}" == "ok" && "${bench_exit_code}" == "0" && -n "${throughput_tps}" ]]; then
        printf 'ok'
        return 0
    fi

    if [[ "${state_id}" == npu_* && -s "${raw_log}" ]]; then
        if grep -Eiq 'qnn_aot_cache_size|aot.*cache.*exceed|cache_size.*exceed|exceed.*cache_size|KV cache.*exceed|exceed.*KV cache|KV.*footprint.*exceed|footprint.*exceed.*KV|not enough.*KV|requested.*KV.*greater' "${raw_log}"; then
            printf 'unsupported_by_current_aot_cache_size'
            return 0
        fi
        if grep -Eiq 'KV.*contract|KV.*layout|KV.*dtype|handoff' "${raw_log}"; then
            printf 'unsupported_by_kv_contract'
            return 0
        fi
    fi

    if [[ -s "${raw_log}" ]] && grep -Eiq 'invalid device|failed to create context|failed to initialize|init.*failed|cannot initialize|runtime.*reject' "${raw_log}"; then
        printf 'unsupported_by_runtime'
        return 0
    fi

    printf 'failed'
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

is_qnn_state() {
    [[ "$1" == npu_* ]]
}

script_for_state() {
    case "$1" in
        npu_low_balanced|npu_burst)
            printf '%s/docs/实验结果/NPUtest.sh' "${ROOT_DIR}"
            ;;
        gpu_734|gpu_967|gpu_1100)
            printf '%s/docs/实验结果/GPUtest.sh' "${ROOT_DIR}"
            ;;
        cpu_big2_2649)
            printf '%s/docs/实验结果/CPUtest.sh' "${ROOT_DIR}"
            ;;
        *)
            return 1
            ;;
    esac
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

append_profile_row() {
    local run_csv="$1"
    local global_csv="$2"
    shift 2

    ensure_profile_header "${run_csv}"
    ensure_profile_header "${global_csv}"
    csv_row "$@" >> "${run_csv}"
    csv_row "$@" >> "${global_csv}"
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

read_first_result_row() {
    local results_csv="$1"
    if [[ -s "${results_csv}" ]]; then
        awk 'NR == 2 { print; exit }' "${results_csv}"
    fi
}

field_from_csv_line() {
    local line="$1"
    local index="$2"
    awk -F, -v idx="${index}" '{ print $idx }' <<< "${line}"
}

write_summary_markdown() {
    local summary_path="$1"
    local run_id="$2"
    local run_csv="$3"
    local command_file="$4"
    local output_dir="$5"
    local rounds="$6"

    local temp_range
    temp_range="$(awk -F, 'NR > 1 && $16 != "" {
            if (!seen || $16 + 0 < min) min = $16 + 0
            if (!seen || $17 + 0 > max) max = $17 + 0
            seen = 1
        } END {
            if (seen) printf "%.2fC to %.2fC", min, max
            else printf "unavailable"
        }' "${run_csv}")"

    local unsupported
    unsupported="$(awk -F, 'NR > 1 && $24 != "ok" {
            printf "- %s context_len=%s support_status=%s fallback_used=%s\n", $4, $5, $24, $25
        }' "${run_csv}")"
    [[ -n "${unsupported}" ]] || unsupported="- none"

    local data_quality_distribution
    data_quality_distribution="$(awk -F, -v rounds="${rounds}" -v threshold="${POWER_CV_UNSTABLE_PCT:-10.0}" '
        function isnum(x) { return x ~ /^-?[0-9]+([.][0-9]+)?$/ }
        function quality(    cv) {
            if ($24 ~ /^unsupported_/) return "unsupported"
            if ($24 != "ok") return "failed"
            if (rounds == 1) return "smoke_only"
            if (!isnum($12) || !isnum($13) || $12 + 0 <= 0) return "failed"
            cv = 100.0 * ($13 + 0) / ($12 + 0)
            if (cv > threshold) return "unstable_power_window"
            return "paper_ready"
        }
        NR > 1 {
            key = quality()
            count[key]++
        } END {
            for (key in count) {
                printf "- %s: %d\n", key, count[key]
            }
        }' "${run_csv}" | sort)"
    [[ -n "${data_quality_distribution}" ]] || data_quality_distribution="- none"

    local paper_ready="needs rerun/review"
    if (( rounds >= 3 )) && ! awk -F, -v threshold="${POWER_CV_UNSTABLE_PCT:-10.0}" '
        function isnum(x) { return x ~ /^-?[0-9]+([.][0-9]+)?$/ }
        NR > 1 {
            if ($24 == "ok") {
                if ($9 != "1" || !isnum($12) || !isnum($13) || $12 + 0 <= 0 || (100.0 * ($13 + 0) / ($12 + 0)) > threshold) bad = 1
            } else if ($24 !~ /^unsupported_/) {
                bad = 1
            }
        }
        END { exit bad ? 0 : 1 }' "${run_csv}"; then
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
        printf '# Insight B Context Frontier %s\n\n' "${run_id}"
        printf '## Experiment goal\n\n'
        printf 'Measure whether decode throughput and energy frontier changes with effective context length using `llama-bench -d context_len -p 0 -n decode_tokens`.\n\n'
        printf '## Exact commands\n\n'
        printf 'Top-level command:\n\n'
        printf '```bash\n'
        sed -n '1,20p' "${command_file}"
        printf '```\n\n'
        printf 'Per-condition command files are in `%s/commands/`.\n\n' "${output_dir}"
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
        printf '| state_id | context_len | phase_isolated | throughput_tps | active_power_mw | power_cv_pct | energy_mj_per_token | support_status | fallback_used | data_quality |\n'
        printf '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |\n'
        awk -F, -v rounds="${rounds}" -v threshold="${POWER_CV_UNSTABLE_PCT:-10.0}" '
        function isnum(x) { return x ~ /^-?[0-9]+([.][0-9]+)?$/ }
        function quality(    cv) {
            if ($24 ~ /^unsupported_/) return "unsupported"
            if ($24 != "ok") return "failed"
            if (rounds == 1) return "smoke_only"
            if (!isnum($12) || !isnum($13) || $12 + 0 <= 0) return "failed"
            cv = 100.0 * ($13 + 0) / ($12 + 0)
            if (cv > threshold) return "unstable_power_window"
            return "paper_ready"
        }
        NR > 1 {
        power_cv = (isnum($12) && isnum($13) && $12 + 0 > 0) ? sprintf("%.2f", 100.0 * ($13 + 0) / ($12 + 0)) : ""
        printf "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n", $4, $5, $9, $10, $12, power_cv, $14, $24, $25, quality()
        }' "${run_csv}"
        printf '\n## Data quality distribution\n\n'
        printf '%s\n\n' "${data_quality_distribution}"
        printf '\n## Anomalies\n\n'
        printf '%s\n\n' "${unsupported}"
        printf '## Raw output directories\n\n'
        printf '%s\n\n' "- \`${output_dir}\`"
        printf '## Paper readiness\n\n'
        printf '%s\n\n' "${paper_ready}"
        printf '## Unsupported conditions\n\n'
        printf '%s\n' "${unsupported}"
    } > "${summary_path}"
}

if [[ "${INSIGHTB_CONTEXT_FRONTIER_LIB_ONLY:-0}" == "1" ]]; then
    return 0 2>/dev/null || exit 0
fi

DEVICE="${DEVICE:-}"
MODEL_PATH="${MODEL_PATH:-${MODEL:-}}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/results/insightB/context-frontier-$(date -u +%Y%m%d-%H%M%S)}"
CONTEXT_LIST="${CONTEXT_LIST:-0 512 1024 1536 1792}"
DECODE_TOKENS="${DECODE_TOKENS:-64}"
ROUNDS="${ROUNDS:-5}"
TEMP_LIMIT_C="${TEMP_LIMIT_C:-38.0}"
COOLDOWN_TEMP_C="${COOLDOWN_TEMP_C:-37.0}"
KEEP_SCREEN_ON_TIMEOUT_MS="${KEEP_SCREEN_ON_TIMEOUT_MS:-1800000}"
SCREEN_BRIGHTNESS_OVERRIDE="${SCREEN_BRIGHTNESS_OVERRIDE:-}"
STATES="${STATES:-npu_low_balanced npu_burst gpu_734 gpu_967 gpu_1100 cpu_big2_2649}"
QNN_CACHE_SAFETY_MARGIN="${QNN_CACHE_SAFETY_MARGIN:-32}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/results/insightB}"
GLOBAL_CSV="${RESULTS_DIR}/context_decode_profile.csv"
RUN_ID="$(date -u +%Y%m%d-%H%M%S)"
RUN_CSV="${OUTPUT_DIR}/context_decode_profile.csv"
COMMAND_FILE="${OUTPUT_DIR}/command.txt"
GIT_COMMIT_FILE="${OUTPUT_DIR}/git_commit.txt"
GIT_STATUS_FILE="${OUTPUT_DIR}/git_status.txt"
SUMMARY_MD="${ROOT_DIR}/docs/实验结果/InsightB_Context_Frontier_${RUN_ID}.md"
QNN_ACTIVE_CONFIG="${GGML_QNN_AOT_CONFIG:-${QNN_AOT_CONFIG:-${QNN_DIR:+${QNN_DIR}/config.json}}}"
QNN_ACTIVE_MODEL_DIR="${GGML_QNN_AOT_MODEL_DIR:-${QNN_AOT_MODEL_DIR:-${QNN_DIR:-}}}"
LLAMA_BENCH_QNN_PREWARM_DECODE="${LLAMA_BENCH_QNN_PREWARM_DECODE:-1}"
QNN_CONFIG_COPY="${OUTPUT_DIR}/qnn_aot_config.json"
QNN_AOT_CACHE_SIZE=""
QNN_AOT_CONTEXT_SIZE=""
DEFAULT_CONTEXT_TOKENS="${CONTEXT_TOKENS:-2048}"

require_runtime_inputs() {
    [[ -n "${DEVICE}" ]] || die "DEVICE must be set"
    [[ -n "${MODEL_PATH}" ]] || die "MODEL_PATH must be set"
    adb -s "${DEVICE}" get-state >/dev/null 2>&1 || die "device ${DEVICE} is offline"
}

prepare_output_dir() {
    if [[ -e "${OUTPUT_DIR}" ]] && [[ "${INSIGHTB_ALLOW_EXISTING_OUTPUT_DIR:-0}" != "1" ]]; then
        if find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
            die "OUTPUT_DIR exists and is non-empty: ${OUTPUT_DIR}"
        fi
    fi
    mkdir -p "${OUTPUT_DIR}/commands" "${RESULTS_DIR}" "$(dirname "${SUMMARY_MD}")"
    ensure_profile_header "${RUN_CSV}"
    ensure_profile_header "${GLOBAL_CSV}"
}

record_run_metadata() {
    local -a top_cmd=(
        env \
        "DEVICE=${DEVICE}" \
        "MODEL_PATH=${MODEL_PATH}" \
        "OUTPUT_DIR=${OUTPUT_DIR}" \
        "CONTEXT_LIST=${CONTEXT_LIST}" \
        "DECODE_TOKENS=${DECODE_TOKENS}" \
        "ROUNDS=${ROUNDS}" \
        "TEMP_LIMIT_C=${TEMP_LIMIT_C}" \
        "COOLDOWN_TEMP_C=${COOLDOWN_TEMP_C}" \
        "STATES=${STATES}" \
        "QNN_CACHE_SAFETY_MARGIN=${QNN_CACHE_SAFETY_MARGIN}"
    )
    append_optional_env top_cmd \
        KEEP_SCREEN_ON_TIMEOUT_MS \
        SCREEN_BRIGHTNESS_OVERRIDE \
        BENCH_DIR \
        QNN_BIN \
        SAMPLE_INTERVAL_S \
        POWER_SETTLE_SECONDS \
        ACTIVE_WINDOW_SAMPLES \
        ACTIVE_MIN_WINDOW_SAMPLES \
        ACTIVE_WINDOW_MAX_RANGE_PCT \
        STABLE_WINDOW_SAMPLES \
        STABLE_RANGE_PCT \
        TASKSET_MASK \
        LLAMA_THREADS \
        NGL \
        BATCH_TOKENS \
        UBATCH_TOKENS \
        MMAP \
        LLAMA_BENCH_FAST_EXIT_VALUE \
        GPU_PIN_GOVERNOR \
        CPU_PIN_GOVERNOR \
        CPU_PIN_POLICY_FILTER \
        LLAMA_BENCH_QNN_PREWARM_DECODE \
        REMOTE_WORKDIR
    if [[ -n "${QNN_ACTIVE_CONFIG}" ]]; then
        top_cmd+=("GGML_QNN_AOT_CONFIG=${QNN_ACTIVE_CONFIG}")
    fi
    if [[ -n "${QNN_ACTIVE_MODEL_DIR}" ]]; then
        top_cmd+=("GGML_QNN_AOT_MODEL_DIR=${QNN_ACTIVE_MODEL_DIR}")
    fi
    top_cmd+=(bash "${ROOT_DIR}/scripts/run_insightB_context_frontier.sh")
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
    local state
    for state in ${STATES}; do
        if is_qnn_state "${state}"; then
            needs_qnn=1
        fi
    done
    (( needs_qnn )) || return 0

    if [[ -z "${QNN_ACTIVE_CONFIG}" ]]; then
        log "no GGML_QNN_AOT_CONFIG/QNN_AOT_CONFIG/QNN_DIR found; QNN rows will be marked unsupported_by_runtime"
        return 0
    fi

    if ! copy_or_pull_qnn_config "${QNN_ACTIVE_CONFIG}" "${QNN_CONFIG_COPY}"; then
        log "failed to read QNN AoT config ${QNN_ACTIVE_CONFIG}; QNN rows will be marked unsupported_by_runtime"
        return 0
    fi

    local batches
    batches="1 ${BATCH_TOKENS:-1} ${UBATCH_TOKENS:-1}"
    local sizes
    if ! sizes="$(parse_qnn_aot_sizes_from_file "${QNN_CONFIG_COPY}" "${batches}")"; then
        log "failed to parse QNN AoT cache/context sizes from ${QNN_ACTIVE_CONFIG}"
        return 0
    fi
    IFS=',' read -r QNN_AOT_CACHE_SIZE QNN_AOT_CONTEXT_SIZE <<< "${sizes}"
    log "QNN AoT config: ${QNN_ACTIVE_CONFIG} cache_size=${QNN_AOT_CACHE_SIZE} context_size=${QNN_AOT_CONTEXT_SIZE}"
}

append_skipped_qnn_row() {
    local state_id="$1"
    local context_len="$2"
    local run_dir="$3"
    local reason="$4"
    local support_status="$5"
    local raw_log="${run_dir}/unsupported.log"
    local backend
    backend="$(state_backend "${state_id}")"

    mkdir -p "${run_dir}"
    printf '%s\n' "${reason}" > "${raw_log}"

    append_profile_row "${RUN_CSV}" "${GLOBAL_CSV}" \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "${MODEL_PATH}" \
        "${backend}" \
        "${state_id}" \
        "${context_len}" \
        "${context_len}" \
        "${DECODE_TOKENS}" \
        "${ROUNDS}" \
        '' \
        '' \
        '' \
        '' \
        '' \
        '' \
        '' \
        '' \
        '' \
        '' \
        '' \
        '' \
        '' \
        "${QNN_AOT_CACHE_SIZE}" \
        "${QNN_AOT_CONTEXT_SIZE}" \
        "${support_status}" \
        '0' \
        "${raw_log}" \
        ''
}

run_condition() {
    local state_id="$1"
    local context_len="$2"
    local backend
    backend="$(state_backend "${state_id}")" || die "unknown state: ${state_id}"

    local run_dir="${OUTPUT_DIR}/raw/${state_id}/context_${context_len}"
    local sweep_stdout="${run_dir}/sweep.stdout.log"
    local command_path="${OUTPUT_DIR}/commands/${state_id}_context_${context_len}.sh"
    mkdir -p "${run_dir}" "$(dirname "${command_path}")"

    if is_qnn_state "${state_id}"; then
        if [[ -z "${QNN_AOT_CACHE_SIZE}" || -z "${QNN_AOT_CONTEXT_SIZE}" ]]; then
            append_skipped_qnn_row \
                "${state_id}" \
                "${context_len}" \
                "${run_dir}" \
                "Skipped: missing or unreadable active QNN AoT config (${QNN_ACTIVE_CONFIG:-unset})." \
                "unsupported_by_runtime"
            return 0
        fi

        local footprint
        set +e
        footprint="$(qnn_cache_guard_status "${context_len}" "${DECODE_TOKENS}" "${QNN_CACHE_SAFETY_MARGIN}" "${QNN_AOT_CACHE_SIZE}")"
        local guard_status=$?
        set -e
        if (( guard_status != 0 )); then
            append_skipped_qnn_row \
                "${state_id}" \
                "${context_len}" \
                "${run_dir}" \
                "Skipped: effective_prefill_tokens(${context_len}) + decode_tokens(${DECODE_TOKENS}) + QNN_CACHE_SAFETY_MARGIN(${QNN_CACHE_SAFETY_MARGIN}) = ${footprint} exceeds qnn_aot_cache_size(${QNN_AOT_CACHE_SIZE})." \
                "unsupported_by_current_aot_cache_size"
            return 0
        fi
    fi

    local min_context_tokens
    min_context_tokens=$(( context_len + DECODE_TOKENS + QNN_CACHE_SAFETY_MARGIN ))
    local bench_context_tokens
    bench_context_tokens="$(max_int "${DEFAULT_CONTEXT_TOKENS}" "${min_context_tokens}")"

    local script_path
    script_path="$(script_for_state "${state_id}")"

    local -a cmd_env=(
        env
        "DEVICE=${DEVICE}"
        "MODEL_PATH=${MODEL_PATH}"
        "OUTPUT_DIR=${run_dir}"
        "TEMP_LIMIT_C=${TEMP_LIMIT_C}"
        "COOLDOWN_TEMP_C=${COOLDOWN_TEMP_C}"
        "PROMPT_TOKENS=0"
        "GEN_TOKENS=${DECODE_TOKENS}"
        "DEPTH_TOKENS=${context_len}"
        "BENCH_REPEATS=${ROUNDS}"
        "CONTEXT_TOKENS=${bench_context_tokens}"
    )
    append_optional_env cmd_env \
        KEEP_SCREEN_ON_TIMEOUT_MS \
        SCREEN_BRIGHTNESS_OVERRIDE \
        BENCH_DIR \
        QNN_BIN \
        SAMPLE_INTERVAL_S \
        POWER_SETTLE_SECONDS \
        ACTIVE_WINDOW_SAMPLES \
        ACTIVE_MIN_WINDOW_SAMPLES \
        ACTIVE_WINDOW_MAX_RANGE_PCT \
        STABLE_WINDOW_SAMPLES \
        STABLE_RANGE_PCT \
        TASKSET_MASK \
        LLAMA_THREADS \
        NGL \
        BATCH_TOKENS \
        UBATCH_TOKENS \
        MMAP \
        LLAMA_BENCH_FAST_EXIT_VALUE \
        GPU_PIN_GOVERNOR \
        CPU_PIN_GOVERNOR \
        CPU_PIN_POLICY_FILTER \
        LLAMA_BENCH_QNN_PREWARM_DECODE \
        REMOTE_WORKDIR

    case "${state_id}" in
        npu_low_balanced|npu_burst)
            cmd_env+=(
                "WORKPOINT_LIST=$(state_npu_workpoint "${state_id}")"
                "QNN_AOT_CONFIG=${QNN_ACTIVE_CONFIG}"
                "GGML_QNN_AOT_CONFIG=${QNN_ACTIVE_CONFIG}"
            )
            if [[ -n "${QNN_ACTIVE_MODEL_DIR}" ]]; then
                cmd_env+=(
                    "QNN_AOT_MODEL_DIR=${QNN_ACTIVE_MODEL_DIR}"
                    "GGML_QNN_AOT_MODEL_DIR=${QNN_ACTIVE_MODEL_DIR}"
                    "QNN_DIR=${QNN_ACTIVE_MODEL_DIR}"
                )
            fi
            ;;
        gpu_734|gpu_967|gpu_1100)
            cmd_env+=(
                "GPU_FREQ_LIST=$(state_gpu_freq_hz "${state_id}")"
            )
            if [[ -n "${QNN_ACTIVE_CONFIG}" ]]; then
                cmd_env+=(
                    "QNN_AOT_CONFIG=${QNN_ACTIVE_CONFIG}"
                    "GGML_QNN_AOT_CONFIG=${QNN_ACTIVE_CONFIG}"
                )
            fi
            if [[ -n "${QNN_ACTIVE_MODEL_DIR}" ]]; then
                cmd_env+=(
                    "QNN_AOT_MODEL_DIR=${QNN_ACTIVE_MODEL_DIR}"
                    "GGML_QNN_AOT_MODEL_DIR=${QNN_ACTIVE_MODEL_DIR}"
                    "QNN_DIR=${QNN_ACTIVE_MODEL_DIR}"
                )
            fi
            ;;
        cpu_big2_2649)
            cmd_env+=(
                "CPU_CASE_LIST=big2:C0:2"
                "CPU_FREQ_LIST=2649600"
                "CPU_FREQ_POINTS=2649600"
            )
            ;;
    esac

    write_command_file "${command_path}" "${cmd_env[@]}" bash "${script_path}"

    log "running ${state_id} context_len=${context_len}"
    set +e
    "${cmd_env[@]}" bash "${script_path}" > "${sweep_stdout}" 2>&1
    local sweep_exit=$?
    set -e

    local results_csv="${run_dir}/results.csv"
    local result_line
    result_line="$(read_first_result_row "${results_csv}")"

    local bench_status=""
    local bench_exit_code="${sweep_exit}"
    local raw_log="${sweep_stdout}"
    local sample_path=""
    local actual_gpu_freq_mhz=""
    local actual_cpu_freq_khz=""
    local freq_stable=""
    local requested_freq=""

    case "${state_id}" in
        npu_low_balanced|npu_burst)
            if [[ -n "${result_line}" ]]; then
                bench_status="$(field_from_csv_line "${result_line}" 2)"
                bench_exit_code="$(field_from_csv_line "${result_line}" 3)"
                raw_log="$(field_from_csv_line "${result_line}" 16)"
                sample_path="$(field_from_csv_line "${result_line}" 17)"
            fi
            ;;
        gpu_734|gpu_967|gpu_1100)
            requested_freq="$(state_gpu_freq_hz "${state_id}")"
            if [[ -n "${result_line}" ]]; then
                bench_status="$(field_from_csv_line "${result_line}" 3)"
                bench_exit_code="$(field_from_csv_line "${result_line}" 4)"
                raw_log="$(field_from_csv_line "${result_line}" 17)"
                sample_path="$(field_from_csv_line "${result_line}" 18)"
            fi
            ;;
        cpu_big2_2649)
            requested_freq="2649600"
            if [[ -n "${result_line}" ]]; then
                bench_status="$(field_from_csv_line "${result_line}" 7)"
                bench_exit_code="$(field_from_csv_line "${result_line}" 8)"
                raw_log="$(field_from_csv_line "${result_line}" 22)"
                sample_path="$(field_from_csv_line "${result_line}" 23)"
            fi
            ;;
    esac

    [[ -n "${bench_status}" ]] || bench_status="script_failed"
    [[ -n "${raw_log}" && -f "${raw_log}" ]] || raw_log="${sweep_stdout}"

    local throughput_tps throughput_std phase_isolated throughput_label
    IFS=',' read -r throughput_tps throughput_std phase_isolated throughput_label <<< "$(parse_llama_bench_tg "${raw_log}" "${DECODE_TOKENS}" "${context_len}")"
    throughput_tps="$(blank_na "${throughput_tps}")"
    throughput_std="$(blank_na "${throughput_std}")"
    phase_isolated="$(blank_na "${phase_isolated}")"

    local active_power_mw power_std temperature_avg_c temp_max_c stable_range_pct avg_freq min_freq max_freq sample_count
    IFS=',' read -r active_power_mw power_std temperature_avg_c temp_max_c stable_range_pct avg_freq min_freq max_freq sample_count <<< "$(summarize_power_samples "${sample_path}")"
    active_power_mw="$(blank_na "${active_power_mw}")"
    power_std="$(blank_na "${power_std}")"
    temperature_avg_c="$(blank_na "${temperature_avg_c}")"
    temp_max_c="$(blank_na "${temp_max_c}")"
    stable_range_pct="$(blank_na "${stable_range_pct}")"

    case "${state_id}" in
        gpu_734|gpu_967|gpu_1100)
            actual_gpu_freq_mhz="$(format_mhz_from_hz "${avg_freq}")"
            freq_stable="$(freq_stable_flag "${requested_freq}" "${min_freq}" "${max_freq}")"
            ;;
        cpu_big2_2649)
            actual_cpu_freq_khz="$(blank_na "${avg_freq}")"
            freq_stable="$(freq_stable_flag "${requested_freq}" "${min_freq}" "${max_freq}")"
            ;;
    esac

    local support_status fallback_used energy_mj_per_token tbt_us power_cv_pct
    support_status="$(classify_support_status "${state_id}" "${bench_status}" "${bench_exit_code}" "${throughput_tps}" "${raw_log}")"
    fallback_used="$(detect_fallback_used "${raw_log}")"
    if [[ "${support_status}" != "ok" ]]; then
        throughput_tps=""
        throughput_std=""
        phase_isolated="$(blank_na "${phase_isolated}")"
        active_power_mw=""
        power_std=""
        temperature_avg_c="$(blank_na "${temperature_avg_c}")"
        temp_max_c="$(blank_na "${temp_max_c}")"
        stable_range_pct="$(blank_na "${stable_range_pct}")"
        freq_stable="$(blank_na "${freq_stable}")"
        actual_gpu_freq_mhz="$(blank_na "${actual_gpu_freq_mhz}")"
        actual_cpu_freq_khz="$(blank_na "${actual_cpu_freq_khz}")"
    fi
    energy_mj_per_token="$(compute_energy_mj_per_token "${active_power_mw}" "${throughput_tps}")"
    tbt_us="$(compute_tbt_us "${throughput_tps}")"
    power_cv_pct="$(compute_power_cv_pct "${active_power_mw}" "${power_std}")"
    classify_data_quality "${support_status}" "${ROUNDS}" "${power_cv_pct}" "${sample_count}" >/dev/null

    append_profile_row "${RUN_CSV}" "${GLOBAL_CSV}" \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "${MODEL_PATH}" \
        "${backend}" \
        "${state_id}" \
        "${context_len}" \
        "${context_len}" \
        "${DECODE_TOKENS}" \
        "${ROUNDS}" \
        "${phase_isolated}" \
        "${throughput_tps}" \
        "${throughput_std}" \
        "${active_power_mw}" \
        "${power_std}" \
        "${energy_mj_per_token}" \
        "${tbt_us}" \
        "${temperature_avg_c}" \
        "${temp_max_c}" \
        "${stable_range_pct}" \
        "${freq_stable}" \
        "${actual_gpu_freq_mhz}" \
        "${actual_cpu_freq_khz}" \
        "${QNN_AOT_CACHE_SIZE}" \
        "${QNN_AOT_CONTEXT_SIZE}" \
        "${support_status}" \
        "${fallback_used}" \
        "${raw_log}" \
        "${sample_path}"

    if [[ "${support_status}" != "ok" ]]; then
        log "condition ${state_id} context_len=${context_len} recorded as ${support_status}; raw_log=${raw_log}"
    fi
}

validate_lists() {
    local state
    for state in ${STATES}; do
        state_backend "${state}" >/dev/null || die "unsupported STATES entry: ${state}"
    done

    local context_len
    for context_len in ${CONTEXT_LIST}; do
        [[ "${context_len}" =~ ^[0-9]+$ ]] || die "invalid CONTEXT_LIST entry: ${context_len}"
    done
    [[ "${DECODE_TOKENS}" =~ ^[0-9]+$ ]] || die "DECODE_TOKENS must be numeric"
    [[ "${ROUNDS}" =~ ^[0-9]+$ ]] || die "ROUNDS must be numeric"
}

main() {
    validate_lists
    require_runtime_inputs
    prepare_output_dir
    record_run_metadata
    load_qnn_aot_sizes_if_needed

    local state context_len
    for context_len in ${CONTEXT_LIST}; do
        for state in ${STATES}; do
            run_condition "${state}" "${context_len}"
        done
    done

    write_summary_markdown "${SUMMARY_MD}" "${RUN_ID}" "${RUN_CSV}" "${COMMAND_FILE}" "${OUTPUT_DIR}" "${ROUNDS}"
    log "run CSV written to ${RUN_CSV}"
    log "aggregate CSV updated at ${GLOBAL_CSV}"
    log "summary Markdown written to ${SUMMARY_MD}"
}

main "$@"
