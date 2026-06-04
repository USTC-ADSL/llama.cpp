#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="${ROOT_DIR}/src/llama-context.cpp"
SYNC_BODY="$(mktemp)"
RECORD_SIG="$(mktemp)"

awk '
    /^void llama_context::synchronize\(\)/ { in_sync = 1 }
    /^const llama_model & llama_context::get_model\(\) const/ { in_sync = 0 }
    in_sync { print }
' "${SOURCE}" > "${SYNC_BODY}"

if grep -n 'LLAMA_LOG_INFO("DECODE_TOKEN_TRACE' "${SYNC_BODY}" >/tmp/low-overhead-token-trace.matches; then
    cat /tmp/low-overhead-token-trace.matches >&2
    rm -f /tmp/low-overhead-token-trace.matches "${SYNC_BODY}" "${RECORD_SIG}"
    printf 'DECODE_TOKEN_TRACE must be buffered and dumped outside the per-token hot path\n' >&2
    exit 1
fi

if grep -n 'trace_timing = hetero_dynamic_trace_timing_enabled' "${SOURCE}" >/tmp/low-overhead-token-trace.matches; then
    cat /tmp/low-overhead-token-trace.matches >&2
    rm -f /tmp/low-overhead-token-trace.matches "${SYNC_BODY}" "${RECORD_SIG}"
    printf 'GGML_HETERO_DYNAMIC_TRACE_TIMING=1 must not enable detailed phase timers\n' >&2
    exit 1
fi

if grep -n 'hetero_dynamic_trace_timing_enabled() && hetero_phase_trace.active' "${SOURCE}" >/tmp/low-overhead-token-trace.matches; then
    cat /tmp/low-overhead-token-trace.matches >&2
    rm -f /tmp/low-overhead-token-trace.matches "${SYNC_BODY}" "${RECORD_SIG}"
    printf 'phase trace accounting must be guarded by GGML_HETERO_DYNAMIC_TRACE_TIMING_DETAIL=1\n' >&2
    exit 1
fi

awk '
    /^void llama_context::hetero_decode_token_trace_record\(/ { in_sig = 1 }
    in_sig { print }
    in_sig && /\) \{/ { in_sig = 0 }
' "${SOURCE}" > "${RECORD_SIG}"

if grep -n ',' "${RECORD_SIG}" >/tmp/low-overhead-token-trace.matches; then
    cat /tmp/low-overhead-token-trace.matches >&2
    rm -f /tmp/low-overhead-token-trace.matches "${SYNC_BODY}" "${RECORD_SIG}"
    printf 'token trace record must accept only the done timestamp in the hot path\n' >&2
    exit 1
fi

awk '
    /^void llama_context::hetero_decode_token_trace_record\(/ { in_record = 1 }
    in_record { print }
    in_record && /^}/ { in_record = 0 }
' "${SOURCE}" > "${RECORD_SIG}"

if grep -n 'reserve' "${RECORD_SIG}" >/tmp/low-overhead-token-trace.matches; then
    cat /tmp/low-overhead-token-trace.matches >&2
    rm -f /tmp/low-overhead-token-trace.matches "${SYNC_BODY}" "${RECORD_SIG}"
    printf 'token trace buffer must be reserved outside the per-token record path\n' >&2
    exit 1
fi

rm -f /tmp/low-overhead-token-trace.matches "${SYNC_BODY}" "${RECORD_SIG}"
