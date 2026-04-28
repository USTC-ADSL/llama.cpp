#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="${ROOT_DIR}/docs/实验结果/NPUtest.sh"

sample_log="$(mktemp)"
trap 'rm -f "${sample_log}"' EXIT

cat > "${sample_log}" <<'EOF'
0,1000000,1,27600
1,1000000,1,27600
2,1000000,1,27600
3,1000000,1,27600
4,1000000,1,27600
5,1000000,1,27600
6,1000000,1,27600
7,1000000,1,27600
8,1000000,6908,27600
9,1000000,7047,27600
10,1000000,7276,27600
11,1000000,7359,27600
12,1000000,7482,27600
13,1000000,7526,27600
14,1000000,7403,27600
15,1000000,7248,27600
16,1000000,7239,27700
17,1000000,7281,27700
18,1000000,7113,27700
19,1000000,6956,27700
20,1000000,6838,27700
21,1000000,6883,27700
22,1000000,6808,27800
23,1000000,6736,27800
24,1000000,6750,27800
25,1000000,6602,27800
26,1000000,6629,27900
27,1000000,6661,27900
28,1000000,6599,28000
29,1000000,6565,28000
30,1000000,6535,28000
31,1000000,6589,28100
32,1000000,6530,28100
33,1000000,6631,28200
34,1000000,6573,28200
35,1000000,6503,28300
36,1000000,6545,28300
EOF

summary="$(
    NPU_SWEEP_SUMMARIZE_ONLY=1 \
    NPU_SWEEP_SAMPLE_LOG="${sample_log}" \
    POWER_SETTLE_SECONDS=8 \
    STABLE_WINDOW_SAMPLES=8 \
    STABLE_RANGE_PCT=4 \
    bash "${SCRIPT_PATH}"
)"

IFS=',' read -r avg_power_mw avg_temp_c max_temp_c start_temp_c end_temp_c sample_count stable_start_index stable_end_index stable_range_pct <<< "${summary}"

awk -v x="${avg_power_mw}" 'BEGIN { exit !(x >= 6565.0 && x <= 6566.5) }' || {
    printf 'unexpected steady-state power: %s\n' "${avg_power_mw}" >&2
    exit 1
}

[[ "${stable_start_index}" == "21" ]] || {
    printf 'unexpected stable_start_index: %s\n' "${stable_start_index}" >&2
    exit 1
}

[[ "${stable_end_index}" == "28" ]] || {
    printf 'unexpected stable_end_index: %s\n' "${stable_end_index}" >&2
    exit 1
}

[[ "${stable_range_pct}" == "1.95" ]] || {
    printf 'unexpected stable_range_pct: %s\n' "${stable_range_pct}" >&2
    exit 1
}

printf 'steady-state summary ok: power=%s start=%s end=%s range=%s\n' \
    "${avg_power_mw}" \
    "${stable_start_index}" \
    "${stable_end_index}" \
    "${stable_range_pct}"
