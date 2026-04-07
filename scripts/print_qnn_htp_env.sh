#!/bin/bash

set -euo pipefail

WORKPOINT=""
DCVS_ENABLE=""
DCVS_POWER_MODE=""
BUS_CORNER=""
CORE_CORNER=""
BUS_MIN=""
BUS_TARGET=""
BUS_MAX=""
CORE_MIN=""
CORE_TARGET=""
CORE_MAX=""
SLEEP_DISABLE=""
SLEEP_LATENCY_US=""
RPC_POLLING_US=""
RPC_CONTROL_LATENCY_US=""

print_help() {
    cat <<'EOF'
Usage:
  eval "$($(basename "$0") --workpoint burst [options])"

Description:
  Print export commands for the QNN HTP workpoint sweep environment knobs.
  This is intended for static per-run workpoint sweeps on qnn-npu decode runs.

Options:
  --workpoint <name>           high-level preset: native, burst, high_performance,
                               balanced, low_balanced, high_power_saver,
                               power_saver, low_power_saver, extreme_power_saver
  --dcvs-enable <0|1>          override GGML_QNN_HTP_DCVS_ENABLE
  --dcvs-power-mode <name>     raw QNN DCVS power mode:
                               adjust_up_down, adjust_only_up, performance,
                               power_saver, power_saver_aggressive, duty_cycle
  --bus-corner <name>          set GGML_QNN_HTP_BUS_VCORNER
  --core-corner <name>         set GGML_QNN_HTP_CORE_VCORNER
  --bus-min <name>             set GGML_QNN_HTP_BUS_VCORNER_MIN
  --bus-target <name>          set GGML_QNN_HTP_BUS_VCORNER_TARGET
  --bus-max <name>             set GGML_QNN_HTP_BUS_VCORNER_MAX
  --core-min <name>            set GGML_QNN_HTP_CORE_VCORNER_MIN
  --core-target <name>         set GGML_QNN_HTP_CORE_VCORNER_TARGET
  --core-max <name>            set GGML_QNN_HTP_CORE_VCORNER_MAX
  --sleep-disable <0|1>        set GGML_QNN_HTP_SLEEP_DISABLE
  --sleep-latency-us <int>     set GGML_QNN_HTP_SLEEP_LATENCY_US
  --rpc-polling-us <int>       set GGML_QNN_HTP_RPC_POLLING_US
  --rpc-control-latency-us <int>
                               set GGML_QNN_HTP_RPC_CONTROL_LATENCY_US
  -h, --help                   show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workpoint)
            WORKPOINT="$2"
            shift 2
            ;;
        --dcvs-enable)
            DCVS_ENABLE="$2"
            shift 2
            ;;
        --dcvs-power-mode)
            DCVS_POWER_MODE="$2"
            shift 2
            ;;
        --bus-corner)
            BUS_CORNER="$2"
            shift 2
            ;;
        --core-corner)
            CORE_CORNER="$2"
            shift 2
            ;;
        --bus-min)
            BUS_MIN="$2"
            shift 2
            ;;
        --bus-target)
            BUS_TARGET="$2"
            shift 2
            ;;
        --bus-max)
            BUS_MAX="$2"
            shift 2
            ;;
        --core-min)
            CORE_MIN="$2"
            shift 2
            ;;
        --core-target)
            CORE_TARGET="$2"
            shift 2
            ;;
        --core-max)
            CORE_MAX="$2"
            shift 2
            ;;
        --sleep-disable)
            SLEEP_DISABLE="$2"
            shift 2
            ;;
        --sleep-latency-us)
            SLEEP_LATENCY_US="$2"
            shift 2
            ;;
        --rpc-polling-us)
            RPC_POLLING_US="$2"
            shift 2
            ;;
        --rpc-control-latency-us)
            RPC_CONTROL_LATENCY_US="$2"
            shift 2
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            print_help >&2
            exit 1
            ;;
    esac
done

emit_export() {
    local key="$1"
    local value="$2"
    if [[ -n "$value" ]]; then
        printf "export %s='%s'\n" "$key" "$value"
    fi
}

emit_export "GGML_QNN_HTP_WORKPOINT" "$WORKPOINT"
emit_export "GGML_QNN_HTP_DCVS_ENABLE" "$DCVS_ENABLE"
emit_export "GGML_QNN_HTP_DCVS_POWER_MODE" "$DCVS_POWER_MODE"
emit_export "GGML_QNN_HTP_BUS_VCORNER" "$BUS_CORNER"
emit_export "GGML_QNN_HTP_CORE_VCORNER" "$CORE_CORNER"
emit_export "GGML_QNN_HTP_BUS_VCORNER_MIN" "$BUS_MIN"
emit_export "GGML_QNN_HTP_BUS_VCORNER_TARGET" "$BUS_TARGET"
emit_export "GGML_QNN_HTP_BUS_VCORNER_MAX" "$BUS_MAX"
emit_export "GGML_QNN_HTP_CORE_VCORNER_MIN" "$CORE_MIN"
emit_export "GGML_QNN_HTP_CORE_VCORNER_TARGET" "$CORE_TARGET"
emit_export "GGML_QNN_HTP_CORE_VCORNER_MAX" "$CORE_MAX"
emit_export "GGML_QNN_HTP_SLEEP_DISABLE" "$SLEEP_DISABLE"
emit_export "GGML_QNN_HTP_SLEEP_LATENCY_US" "$SLEEP_LATENCY_US"
emit_export "GGML_QNN_HTP_RPC_POLLING_US" "$RPC_POLLING_US"
emit_export "GGML_QNN_HTP_RPC_CONTROL_LATENCY_US" "$RPC_CONTROL_LATENCY_US"
