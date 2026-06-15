#include "llama-context.h"

#include "llama-arch.h"
#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-kv-cache-iswa.h"
#include "llama-memory.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-mmap.h"
#include "llama-model.h"
#include "llama-hetero-route.h"
#include "ggml-profiler.h"
#include "llama-ext.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cinttypes>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <system_error>

#if defined(__linux__)
#include <sched.h>
#include <sys/types.h>
#endif

bool llama_context_qnn_accel_backend_requested(
        const std::vector<std::string> & device_names,
        const llama_hetero_route_spec & hetero_route,
        const llama_hetero_route_spec & dynamic_prefill_route,
        const llama_hetero_route_spec & dynamic_decode_route,
        const llama_hetero_route_spec & dynamic_fallback_route);

bool llama_context_should_disable_cpu_qnn_host_fallback(
        bool first_device_is_qnn,
        bool routes_use_opencl,
        bool routes_use_cpu);

llama_hetero_kv_contract llama_dynamic_phase_migration_kv_contract(
        const std::string & producer_backend,
        const std::string & consumer_backend,
        const char * reason);

bool llama_context_should_attempt_qnn_phase_kv_migration(
        const std::string & current_attn_backend,
        const std::string & target_attn_backend,
        uint32_t            n_tokens,
        bool                generic_kv_enabled);

bool llama_context_should_sync_opencl_before_qnn_direct_import(
        const std::string & current_attn_backend,
        const std::string & target_attn_backend,
        uint32_t            n_tokens,
        bool                generic_kv_enabled);

llama_opencl_external_host_sync_scope llama_context_opencl_sync_scope_for_qnn_direct_import(
        const std::string & current_attn_backend,
        const std::string & target_attn_backend,
        uint32_t            n_tokens,
        bool                generic_kv_enabled);

bool llama_context_should_use_qnn_written_generic_kv_for_cpu(
        const std::string & current_attn_backend,
        const std::string & target_attn_backend,
        uint32_t            n_tokens,
        bool                generic_kv_enabled,
        bool                qnn_writeback_ready,
        bool                live_kv_cpu_accessible,
        bool                qnn_writeback_flushed);

bool llama_context_kv_buft_is_cpu_accessible(ggml_backend_buffer_type_t buft);

bool llama_context_should_try_qnn_written_generic_kv_for_opencl(
        const std::string & current_attn_backend,
        const std::string & target_attn_backend,
        uint32_t            n_tokens,
        bool                generic_kv_enabled,
        bool                qnn_writeback_ready);

llama_hetero_kv_contract llama_dynamic_phase_shared_qnn_kv_contract(
        const std::string & prefill_attn_backend,
        const std::string & decode_attn_backend,
        bool                qnn_host_buffer_available,
        bool                opencl_can_alias_qnn_host);

bool llama_context_should_use_qnn_shared_phase_kv(
        const std::string & current_attn_backend,
        const std::string & target_attn_backend,
        uint32_t            n_tokens,
        bool                generic_kv_enabled,
        const llama_hetero_kv_contract & allocated_kv_contract);

llama_opencl_external_host_sync_scope llama_context_opencl_sync_scope_for_qnn_shared_phase_kv(
        const std::string & current_attn_backend,
        const std::string & target_attn_backend,
        uint32_t            n_tokens,
        bool                generic_kv_enabled,
        const llama_hetero_kv_contract & allocated_kv_contract);

bool llama_context_should_try_qnn_opencl_direct_host_ptr_visibility(
        const std::string & current_attn_backend,
        const std::string & target_attn_backend,
        uint32_t            n_tokens,
        bool                generic_kv_enabled,
        const llama_hetero_kv_contract & allocated_kv_contract,
        bool                experimental_enabled);

bool llama_context_should_use_dynamic_decode_tg_only_sched_reserve(
        bool     dynamic_route_enabled,
        uint32_t n_tokens,
        bool     experimental_enabled);

bool llama_context_should_prewarm_dynamic_qnn_opencl_kv_aliases(
        const std::string & prefill_attn_backend,
        const std::string & decode_attn_backend,
        bool                generic_kv_enabled,
        const llama_hetero_kv_contract & allocated_kv_contract,
        bool                experimental_enabled);

bool llama_context_should_preload_dynamic_qnn_decode_graphs(
        bool dynamic_route_enabled,
        bool dynamic_decode_uses_qnn,
        bool preload_enabled);

bool llama_context_should_try_cpu_opencl_uma_kv_handoff(
        const std::string & current_attn_backend,
        const std::string & target_attn_backend,
        uint32_t            n_tokens,
        bool                disabled,
        bool                allow_opencl_to_cpu);

const llama_dynamic_route_candidate * llama_context_initial_dynamic_decode_candidate(
        const llama_dynamic_route_runtime_config & config);

int64_t llama_context_transition_blocking_us(
        int64_t total_wall_us,
        int64_t process_ubatch_us);

int64_t llama_context_decode_token_gap_us(
        int64_t previous_decode_done_us,
        int64_t current_decode_done_us);

bool llama_context_should_apply_qnn_workpoint_switch(
        const llama_hetero_route_spec & current_route,
        const llama_hetero_route_spec & target_route,
        uint32_t                        n_tokens,
        const std::string &             current_workpoint,
        const char *                    target_workpoint);

bool llama_context_should_apply_gpu_freq_switch(
        const llama_hetero_route_spec & current_route,
        const llama_hetero_route_spec & target_route,
        uint32_t                        n_tokens,
        uint64_t                        current_freq_hz,
        uint64_t                        target_freq_hz);

bool llama_context_should_apply_cpu_state_switch(
        const llama_hetero_route_spec & current_route,
        const llama_hetero_route_spec & target_route,
        uint32_t                        n_tokens,
        const std::string &             current_affinity_mask,
        const char *                    target_affinity_mask,
        int32_t                         current_threads,
        int32_t                         target_threads);

bool llama_context_should_apply_cpu_freq_switch(
        const llama_hetero_route_spec & current_route,
        const llama_hetero_route_spec & target_route,
        uint32_t                        n_tokens,
        uint64_t                        current_freq_khz,
        uint64_t                        target_freq_khz);

llama_hetero_kv_contract llama_dynamic_phase_migration_kv_contract(
        const std::string & producer_backend,
        const std::string & consumer_backend,
        const char * reason) {
    const std::string producer = llama_hetero_canonical_backend(producer_backend);
    const std::string consumer = llama_hetero_canonical_backend(consumer_backend);

    llama_hetero_kv_contract contract;
    contract.producer_backend = producer;
    contract.consumer_backend = consumer;
    contract.implemented = true;
    contract.reason = reason != nullptr ? reason : "dynamic-phase-migration";

    if (llama_hetero_is_qnn_backend(producer) &&
        (consumer == "cpu" || consumer == "opencl")) {
        contract.layout = llama_hetero_kv_layout_kind::STAGE_SHARED;
        contract.transfer = llama_hetero_kv_transfer_mode::QNN_RPCMEM;
        contract.storage_backend = "qnn-npu-host";
        contract.shared_buffer_required = true;
        contract.buffer_available = false;
        contract.zero_copy = false;
        return contract;
    }

    contract.storage_backend = consumer;
    contract.transfer = llama_hetero_kv_transfer_mode::NONE;
    contract.shared_buffer_required = false;
    contract.buffer_available = true;
    contract.zero_copy = false;
    return contract;
}

bool llama_context_should_attempt_qnn_phase_kv_migration(
        const std::string & current_attn_backend,
        const std::string & target_attn_backend,
        uint32_t            n_tokens,
        bool                generic_kv_enabled) {
    if (!generic_kv_enabled || n_tokens != 1) {
        return false;
    }

    const std::string current = llama_hetero_canonical_backend(current_attn_backend);
    const std::string target  = llama_hetero_canonical_backend(target_attn_backend);
    const bool current_is_qnn = llama_hetero_is_qnn_backend(current);
    const bool target_is_qnn  = llama_hetero_is_qnn_backend(target);

    if (current_is_qnn) {
        return target == "cpu" || target == "opencl";
    }

    return target_is_qnn && (current == "cpu" || current == "opencl");
}

bool llama_context_should_sync_opencl_before_qnn_direct_import(
        const std::string & current_attn_backend,
        const std::string & target_attn_backend,
        uint32_t            n_tokens,
        bool                generic_kv_enabled) {
    if (!llama_context_should_attempt_qnn_phase_kv_migration(
                current_attn_backend,
                target_attn_backend,
                n_tokens,
                generic_kv_enabled)) {
        return false;
    }

    const std::string current = llama_hetero_canonical_backend(current_attn_backend);
    const std::string target  = llama_hetero_canonical_backend(target_attn_backend);
    return current == "opencl" && llama_hetero_is_qnn_backend(target);
}

llama_opencl_external_host_sync_scope llama_context_opencl_sync_scope_for_qnn_direct_import(
        const std::string & current_attn_backend,
        const std::string & target_attn_backend,
        uint32_t            n_tokens,
        bool                generic_kv_enabled) {
    return llama_context_should_sync_opencl_before_qnn_direct_import(
                current_attn_backend,
                target_attn_backend,
                n_tokens,
                generic_kv_enabled)
            ? llama_opencl_external_host_sync_scope::ACTIVE_KV_PREFIX
            : llama_opencl_external_host_sync_scope::FULL_BUFFER;
}

bool llama_context_should_use_qnn_written_generic_kv_for_cpu(
        const std::string & current_attn_backend,
        const std::string & target_attn_backend,
        uint32_t            n_tokens,
        bool                generic_kv_enabled,
        bool                qnn_writeback_ready,
        bool                live_kv_cpu_accessible,
        bool                qnn_writeback_flushed) {
    GGML_UNUSED(qnn_writeback_flushed);

    if (!generic_kv_enabled || !qnn_writeback_ready || !live_kv_cpu_accessible || n_tokens != 1) {
        return false;
    }

    const std::string current = llama_hetero_canonical_backend(current_attn_backend);
    const std::string target  = llama_hetero_canonical_backend(target_attn_backend);
    return llama_hetero_is_qnn_backend(current) && target == "cpu";
}

bool llama_context_should_try_qnn_written_generic_kv_for_opencl(
        const std::string & current_attn_backend,
        const std::string & target_attn_backend,
        uint32_t            n_tokens,
        bool                generic_kv_enabled,
        bool                qnn_writeback_ready) {
    if (!generic_kv_enabled || !qnn_writeback_ready || n_tokens != 1) {
        return false;
    }

    const std::string current = llama_hetero_canonical_backend(current_attn_backend);
    const std::string target  = llama_hetero_canonical_backend(target_attn_backend);
    return llama_hetero_is_qnn_backend(current) && target == "opencl";
}

bool llama_context_kv_buft_is_cpu_accessible(ggml_backend_buffer_type_t buft) {
    if (buft == nullptr) {
        return false;
    }

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu_dev != nullptr && ggml_backend_dev_supports_buft(cpu_dev, buft)) {
        return true;
    }

    return ggml_backend_buft_is_host(buft);
}

static bool llama_context_live_kv_is_cpu_accessible(const llama_memory_i * memory) {
    if (memory == nullptr) {
        return false;
    }

    const auto breakdown = memory->memory_breakdown();
    if (breakdown.empty()) {
        return false;
    }

    bool has_cpu_accessible_kv = false;
    for (const auto & [buft, bytes] : breakdown) {
        if (bytes == 0) {
            continue;
        }

        if (!llama_context_kv_buft_is_cpu_accessible(buft)) {
            return false;
        }
        has_cpu_accessible_kv = true;
    }

    return has_cpu_accessible_kv;
}

llama_hetero_kv_contract llama_dynamic_phase_shared_qnn_kv_contract(
        const std::string & prefill_attn_backend,
        const std::string & decode_attn_backend,
        bool                qnn_host_buffer_available,
        bool                opencl_can_alias_qnn_host) {
    const std::string prefill = llama_hetero_canonical_backend(prefill_attn_backend);
    const std::string decode  = llama_hetero_canonical_backend(decode_attn_backend);

    llama_hetero_kv_contract contract;

    if (!llama_hetero_is_qnn_backend(prefill) || decode != "opencl") {
        contract.reason = "dynamic-phase-shared-qnn-kv-not-applicable";
        return contract;
    }

    if (!qnn_host_buffer_available || !opencl_can_alias_qnn_host) {
        contract.reason = !qnn_host_buffer_available
            ? "qnn-host-buffer-unavailable"
            : "opencl-cannot-alias-qnn-host";
        return contract;
    }

    contract.producer_backend = prefill;
    contract.consumer_backend = decode;
    contract.storage_backend = "qnn-npu-host";
    contract.layout = llama_hetero_kv_layout_kind::STAGE_SHARED;
    contract.transfer = llama_hetero_kv_transfer_mode::QNN_RPCMEM;
    contract.shared_buffer_required = true;
    contract.implemented = true;
    contract.buffer_available = true;
    contract.zero_copy = true;
    contract.reason = "dynamic-qnn-prefill-opencl-decode-shared-kv";
    return contract;
}

bool llama_context_should_use_qnn_shared_phase_kv(
        const std::string & current_attn_backend,
        const std::string & target_attn_backend,
        uint32_t            n_tokens,
        bool                generic_kv_enabled,
        const llama_hetero_kv_contract & allocated_kv_contract) {
    if (!llama_context_should_attempt_qnn_phase_kv_migration(
                current_attn_backend,
                target_attn_backend,
                n_tokens,
                generic_kv_enabled)) {
        return false;
    }

    if (llama_hetero_canonical_backend(target_attn_backend) != "opencl") {
        return false;
    }

    return allocated_kv_contract.stage_boundary_active() &&
           llama_hetero_is_qnn_backend(allocated_kv_contract.producer_backend) &&
           llama_hetero_canonical_backend(allocated_kv_contract.consumer_backend) == "opencl" &&
           allocated_kv_contract.storage_backend == "qnn-npu-host" &&
           allocated_kv_contract.layout == llama_hetero_kv_layout_kind::STAGE_SHARED &&
           allocated_kv_contract.transfer == llama_hetero_kv_transfer_mode::QNN_RPCMEM &&
           allocated_kv_contract.shared_buffer_required &&
           allocated_kv_contract.implemented &&
           allocated_kv_contract.buffer_available &&
           allocated_kv_contract.zero_copy;
}

llama_opencl_external_host_sync_scope llama_context_opencl_sync_scope_for_qnn_shared_phase_kv(
        const std::string & current_attn_backend,
        const std::string & target_attn_backend,
        uint32_t            n_tokens,
        bool                generic_kv_enabled,
        const llama_hetero_kv_contract & allocated_kv_contract) {
    return llama_context_should_use_qnn_shared_phase_kv(
                current_attn_backend,
                target_attn_backend,
                n_tokens,
                generic_kv_enabled,
                allocated_kv_contract)
            ? llama_opencl_external_host_sync_scope::ACTIVE_KV_PREFIX
            : llama_opencl_external_host_sync_scope::FULL_BUFFER;
}

bool llama_context_should_try_qnn_opencl_direct_host_ptr_visibility(
        const std::string & current_attn_backend,
        const std::string & target_attn_backend,
        uint32_t            n_tokens,
        bool                generic_kv_enabled,
        const llama_hetero_kv_contract & allocated_kv_contract,
        bool                experimental_enabled) {
    return experimental_enabled &&
           llama_context_should_use_qnn_shared_phase_kv(
                   current_attn_backend,
                   target_attn_backend,
                   n_tokens,
                   generic_kv_enabled,
                   allocated_kv_contract);
}

bool llama_context_should_use_dynamic_decode_tg_only_sched_reserve(
        bool     dynamic_route_enabled,
        uint32_t n_tokens,
        bool     experimental_enabled) {
    return experimental_enabled && dynamic_route_enabled && n_tokens == 1;
}

bool llama_context_should_prewarm_dynamic_qnn_opencl_kv_aliases(
        const std::string & prefill_attn_backend,
        const std::string & decode_attn_backend,
        bool                generic_kv_enabled,
        const llama_hetero_kv_contract & allocated_kv_contract,
        bool                experimental_enabled) {
    (void) prefill_attn_backend;
    (void) decode_attn_backend;
    (void) generic_kv_enabled;
    (void) allocated_kv_contract;
    (void) experimental_enabled;

    // The existing sync API creates the alias and publishes host contents in one
    // call. Running it during context construction is too early for qnn->opencl:
    // QNN has not written the prefill KV yet, so a correct prewarm needs a
    // separate alias-only path.
    return false;
}

bool llama_context_should_preload_dynamic_qnn_decode_graphs(
        bool dynamic_route_enabled,
        bool dynamic_decode_uses_qnn,
        bool preload_enabled) {
    return preload_enabled && dynamic_route_enabled && dynamic_decode_uses_qnn;
}

bool llama_context_should_try_cpu_opencl_uma_kv_handoff(
        const std::string & current_attn_backend,
        const std::string & target_attn_backend,
        uint32_t            n_tokens,
        bool                disabled,
        bool                allow_opencl_to_cpu) {
    if (disabled || n_tokens != 1) {
        return false;
    }

    const std::string current = llama_hetero_canonical_backend(current_attn_backend);
    const std::string target  = llama_hetero_canonical_backend(target_attn_backend);
    if (current == "cpu" && target == "opencl") {
        return true;
    }

    return allow_opencl_to_cpu && current == "opencl" && target == "cpu";
}

const llama_dynamic_route_candidate * llama_context_initial_dynamic_decode_candidate(
        const llama_dynamic_route_runtime_config & config) {
    if (config.decode.configured) {
        return &config.decode;
    }

    for (const auto & entry : config.decode_schedule) {
        if (entry.start_token == 1 && entry.route.configured) {
            return &entry.route;
        }
        if (entry.start_token > 1) {
            break;
        }
    }

    return nullptr;
}

int64_t llama_context_transition_blocking_us(
        int64_t total_wall_us,
        int64_t process_ubatch_us) {
    return std::max<int64_t>(int64_t(0), total_wall_us - std::max<int64_t>(int64_t(0), process_ubatch_us));
}

int64_t llama_context_decode_token_gap_us(
        int64_t previous_decode_done_us,
        int64_t current_decode_done_us) {
    if (previous_decode_done_us <= 0 || current_decode_done_us <= 0 || current_decode_done_us < previous_decode_done_us) {
        return -1;
    }
    return current_decode_done_us - previous_decode_done_us;
}

static std::string llama_context_canonical_qnn_workpoint(const char * value) {
    std::string normalized;
    if (value == nullptr) {
        return normalized;
    }

    for (const unsigned char c : std::string(value)) {
        if (std::isalnum(c) != 0) {
            normalized.push_back(static_cast<char>(std::tolower(c)));
        }
    }

    if (normalized == "max" || normalized == "performance") {
        return "burst";
    }
    if (normalized == "highperformance" || normalized == "sustainedhighperformance") {
        return "high_performance";
    }
    if (normalized == "default") {
        return "balanced";
    }
    if (normalized == "lowbalanced") {
        return "low_balanced";
    }
    if (normalized == "highpowersaver") {
        return "high_power_saver";
    }
    if (normalized == "powersaver") {
        return "power_saver";
    }
    if (normalized == "lowpowersaver") {
        return "low_power_saver";
    }
    if (normalized == "extremepowersaver") {
        return "extreme_power_saver";
    }

    return normalized;
}

bool llama_context_should_apply_qnn_workpoint_switch(
        const llama_hetero_route_spec & current_route,
        const llama_hetero_route_spec & target_route,
        uint32_t                        n_tokens,
        const std::string &             current_workpoint,
        const char *                    target_workpoint) {
    if (n_tokens != 1 || target_workpoint == nullptr || target_workpoint[0] == '\0') {
        return false;
    }

    const std::string current_backend =
        llama_hetero_canonical_backend(llama_hetero_phase_backend_for_route(current_route));
    const std::string target_backend =
        llama_hetero_canonical_backend(llama_hetero_phase_backend_for_route(target_route));
    if (current_backend != "qnn-npu" || target_backend != "qnn-npu") {
        return false;
    }

    const std::string current = llama_context_canonical_qnn_workpoint(current_workpoint.c_str());
    const std::string target  = llama_context_canonical_qnn_workpoint(target_workpoint);
    return !target.empty() && current != target;
}

bool llama_context_should_apply_gpu_freq_switch(
        const llama_hetero_route_spec & current_route,
        const llama_hetero_route_spec & target_route,
        uint32_t                        n_tokens,
        uint64_t                        current_freq_hz,
        uint64_t                        target_freq_hz) {
    if (n_tokens != 1 || target_freq_hz == 0) {
        return false;
    }

    const std::string current_backend =
        llama_hetero_canonical_backend(llama_hetero_phase_backend_for_route(current_route));
    const std::string target_backend =
        llama_hetero_canonical_backend(llama_hetero_phase_backend_for_route(target_route));
    if (current_backend != "opencl" || target_backend != "opencl") {
        return false;
    }

    return current_freq_hz == 0 || current_freq_hz != target_freq_hz;
}

bool llama_context_should_apply_cpu_state_switch(
        const llama_hetero_route_spec & current_route,
        const llama_hetero_route_spec & target_route,
        uint32_t                        n_tokens,
        const std::string &             current_affinity_mask,
        const char *                    target_affinity_mask,
        int32_t                         current_threads,
        int32_t                         target_threads) {
    if (n_tokens != 1) {
        return false;
    }

    const std::string current_backend =
        llama_hetero_canonical_backend(llama_hetero_phase_backend_for_route(current_route));
    const std::string target_backend =
        llama_hetero_canonical_backend(llama_hetero_phase_backend_for_route(target_route));
    const bool current_is_cpu =
        current_backend.empty() || current_backend == "cpu";
    const bool target_is_cpu = target_backend == "cpu";
    if (!current_is_cpu || !target_is_cpu) {
        return false;
    }

    const bool has_target_mask =
        target_affinity_mask != nullptr && target_affinity_mask[0] != '\0';
    const bool has_target_threads = target_threads > 0;
    if (!has_target_mask && !has_target_threads) {
        return false;
    }

    const bool mask_changed =
        has_target_mask &&
        (current_affinity_mask.empty() || current_affinity_mask != target_affinity_mask);
    const bool threads_changed = has_target_threads && current_threads != target_threads;
    return mask_changed || threads_changed;
}

bool llama_context_should_apply_cpu_freq_switch(
        const llama_hetero_route_spec & current_route,
        const llama_hetero_route_spec & target_route,
        uint32_t                        n_tokens,
        uint64_t                        current_freq_khz,
        uint64_t                        target_freq_khz) {
    if (n_tokens != 1 || target_freq_khz == 0) {
        return false;
    }

    const std::string current_backend =
        llama_hetero_canonical_backend(llama_hetero_phase_backend_for_route(current_route));
    const std::string target_backend =
        llama_hetero_canonical_backend(llama_hetero_phase_backend_for_route(target_route));
    const bool current_is_cpu =
        current_backend.empty() || current_backend == "cpu";
    const bool target_is_cpu = target_backend == "cpu";
    if (!current_is_cpu || !target_is_cpu) {
        return false;
    }

    return current_freq_khz == 0 || current_freq_khz != target_freq_khz;
}

namespace {

using ggml_backend_qnn_aot_has_pending_generic_kv_writeback_t = bool (*)(ggml_backend_t backend);
using ggml_backend_qnn_aot_flush_pending_generic_kv_writeback_t = bool (*)(ggml_backend_t backend);
using ggml_backend_qnn_aot_reset_state_t = bool (*)(ggml_backend_t backend);
using ggml_backend_qnn_aot_preload_decode_graphs_t = bool (*)(ggml_backend_t backend, size_t n_tokens);
using ggml_backend_qnn_set_htp_workpoint_t = bool (*)(ggml_backend_t backend, const char * workpoint);

bool hetero_dynamic_trace_timing_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * value = std::getenv("GGML_HETERO_DYNAMIC_TRACE_TIMING");
        enabled = (value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0) ? 1 : 0;
    }

    return enabled != 0;
}

bool env_flag_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

uint64_t llama_context_effective_decode_gpu_freq_hz(
        const llama_dynamic_route_runtime_config & config,
        const llama_dynamic_route_decision &       decision) {
    return decision.backend_state.has_gpu_freq_hz
        ? decision.backend_state.gpu_freq_hz
        : config.decode_gpu_freq_hz;
}

uint64_t llama_context_effective_decode_cpu_freq_khz(
        const llama_dynamic_route_runtime_config & config,
        const llama_dynamic_route_decision &       decision) {
    return decision.backend_state.has_cpu_freq_khz
        ? decision.backend_state.cpu_freq_khz
        : config.decode_cpu_freq_khz;
}

std::string llama_context_effective_decode_cpu_affinity_mask(
        const llama_dynamic_route_runtime_config & config,
        const llama_dynamic_route_decision &       decision) {
    return decision.backend_state.has_cpu_affinity_mask
        ? decision.backend_state.cpu_affinity_mask
        : config.decode_cpu_affinity_mask;
}

int32_t llama_context_effective_decode_cpu_threads(
        const llama_dynamic_route_runtime_config & config,
        const llama_dynamic_route_decision &       decision) {
    return decision.backend_state.has_cpu_threads
        ? decision.backend_state.cpu_threads
        : config.decode_cpu_threads;
}

const char * llama_context_effective_decode_qnn_workpoint(
        const llama_dynamic_route_decision & decision,
        std::string &                        owned_workpoint) {
    if (decision.backend_state.has_qnn_workpoint) {
        owned_workpoint = decision.backend_state.qnn_workpoint;
        return owned_workpoint.c_str();
    }

    return std::getenv("GGML_HETERO_DYNAMIC_DECODE_QNN_WORKPOINT");
}

std::string llama_context_format_cpu_mask(uint64_t mask) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%" PRIX64, mask);
    return buffer;
}

bool llama_context_parse_cpu_mask(
        const std::string & value,
        uint64_t &          out_mask,
        std::string &       error) {
    std::string trimmed = value;
    trimmed.erase(trimmed.begin(), std::find_if(trimmed.begin(), trimmed.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    trimmed.erase(std::find_if(trimmed.rbegin(), trimmed.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), trimmed.end());

    if (trimmed.empty()) {
        error = "empty CPU affinity mask";
        return false;
    }

    const bool has_hex_alpha =
        std::any_of(trimmed.begin(), trimmed.end(), [](unsigned char ch) {
            return (ch >= 'a' && ch <= 'f') || (ch >= 'A' && ch <= 'F');
        });
    const int base =
        has_hex_alpha ||
        (trimmed.size() > 2 && trimmed[0] == '0' && (trimmed[1] == 'x' || trimmed[1] == 'X'))
            ? 16
            : 0;

    char * end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(trimmed.c_str(), &end, base);
    if (errno != 0 || end == trimmed.c_str() || (end != nullptr && *end != '\0') || parsed == 0) {
        error = "invalid CPU affinity mask: " + value;
        return false;
    }

    out_mask = static_cast<uint64_t>(parsed);
    return true;
}

bool llama_context_cpu_mask_to_threadpool_cpumask(
        const std::string & value,
        bool *              cpumask,
        std::string &       error) {
    if (cpumask == nullptr) {
        error = "null threadpool cpumask";
        return false;
    }

    std::fill_n(cpumask, GGML_MAX_N_THREADS, false);
    if (value.empty()) {
        return true;
    }

    uint64_t parsed_mask = 0;
    if (!llama_context_parse_cpu_mask(value, parsed_mask, error)) {
        return false;
    }

    for (int cpu = 0; cpu < std::min<int>(64, GGML_MAX_N_THREADS); ++cpu) {
        if ((parsed_mask & (uint64_t(1) << cpu)) != 0) {
            cpumask[cpu] = true;
        }
    }

    return true;
}

#if defined(__linux__)
std::string llama_context_read_current_cpu_affinity_mask() {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0) {
        return {};
    }

    uint64_t mask = 0;
    const int limit = std::min<int>(CPU_SETSIZE, 64);
    for (int cpu = 0; cpu < limit; ++cpu) {
        if (CPU_ISSET(cpu, &set)) {
            mask |= (uint64_t(1) << cpu);
        }
    }

    return mask != 0 ? llama_context_format_cpu_mask(mask) : std::string();
}

bool llama_context_apply_cpu_affinity_mask(
        const std::string & target_mask,
        std::string &       actual_mask,
        std::string &       error) {
    uint64_t parsed_mask = 0;
    if (!llama_context_parse_cpu_mask(target_mask, parsed_mask, error)) {
        return false;
    }

    cpu_set_t set;
    CPU_ZERO(&set);
    for (int cpu = 0; cpu < std::min<int>(CPU_SETSIZE, 64); ++cpu) {
        if ((parsed_mask & (uint64_t(1) << cpu)) != 0) {
            CPU_SET(cpu, &set);
        }
    }

    int applied_count = 0;
    int failure_count = 0;
    int first_errno = 0;

    std::error_code ec;
    for (const auto & entry : std::filesystem::directory_iterator("/proc/self/task", ec)) {
        const std::string name = entry.path().filename().string();
        char * end = nullptr;
        errno = 0;
        const long tid_long = std::strtol(name.c_str(), &end, 10);
        if (errno != 0 || end == name.c_str() || (end != nullptr && *end != '\0') || tid_long <= 0) {
            continue;
        }

        if (sched_setaffinity(static_cast<pid_t>(tid_long), sizeof(set), &set) == 0) {
            ++applied_count;
        } else if (errno != ESRCH) {
            ++failure_count;
            if (first_errno == 0) {
                first_errno = errno;
            }
        }
    }

    if (applied_count == 0) {
        if (sched_setaffinity(0, sizeof(set), &set) == 0) {
            ++applied_count;
        } else {
            first_errno = errno;
            ++failure_count;
        }
    }

    actual_mask = llama_context_read_current_cpu_affinity_mask();
    if (applied_count == 0 || actual_mask.empty()) {
        error = "sched_setaffinity failed";
        if (first_errno != 0) {
            error += ": ";
            error += std::strerror(first_errno);
        }
        return false;
    }

    if (failure_count > 0) {
        LLAMA_LOG_WARN("%s: CPU affinity applied to %d thread(s), %d non-ESRCH failure(s), first_errno=%d\n",
                __func__,
                applied_count,
                failure_count,
                first_errno);
    }

    return true;
}

std::string llama_context_read_task_cpu_affinity_summary() {
    std::map<std::string, int> mask_counts;
    int task_count = 0;

    std::error_code ec;
    for (const auto & entry : std::filesystem::directory_iterator("/proc/self/task", ec)) {
        std::ifstream status_file(entry.path() / "status");
        if (!status_file) {
            continue;
        }

        std::string line;
        std::string mask = "<unknown>";
        while (std::getline(status_file, line)) {
            constexpr const char * prefix = "Cpus_allowed_list:";
            if (line.rfind(prefix, 0) == 0) {
                mask = line.substr(std::strlen(prefix));
                mask.erase(mask.begin(), std::find_if(mask.begin(), mask.end(), [](unsigned char ch) {
                    return !std::isspace(ch);
                }));
                mask.erase(std::find_if(mask.rbegin(), mask.rend(), [](unsigned char ch) {
                    return !std::isspace(ch);
                }).base(), mask.end());
                break;
            }
        }

        ++mask_counts[mask];
        ++task_count;
    }

    std::string summary = "tasks=" + std::to_string(task_count) + " masks=";
    bool first = true;
    for (const auto & [mask, count] : mask_counts) {
        if (!first) {
            summary += ",";
        }
        summary += mask + ":" + std::to_string(count);
        first = false;
    }
    if (first) {
        summary += "<none>";
    }
    return summary;
}
#else
std::string llama_context_read_current_cpu_affinity_mask() {
    return {};
}

bool llama_context_apply_cpu_affinity_mask(
        const std::string & target_mask,
        std::string &       actual_mask,
        std::string &       error) {
    GGML_UNUSED(target_mask);
    GGML_UNUSED(actual_mask);
    error = "CPU affinity switch is only implemented on Linux/Android";
    return false;
}

std::string llama_context_read_task_cpu_affinity_summary() {
    return "tasks=0 masks=unsupported";
}
#endif

bool hetero_dynamic_trace_timing_detail_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        enabled =
            hetero_dynamic_trace_timing_enabled() &&
            env_flag_enabled("GGML_HETERO_DYNAMIC_TRACE_TIMING_DETAIL") ? 1 : 0;
    }

    return enabled != 0;
}

const char * hetero_phase_name(uint32_t n_tokens) {
    return n_tokens > 1 ? "prefill" : "decode";
}

int64_t percentile_nearest_rank_us(std::vector<int64_t> values, double percentile) {
    if (values.empty()) {
        return 0;
    }

    std::sort(values.begin(), values.end());

    const double rank = percentile * double(values.size() - 1);
    const size_t index = static_cast<size_t>(std::ceil(rank));
    return values[std::min(index, values.size() - 1)];
}

const char * canonicalize_hetero_backend_device_name(const char * value) {
    const std::string normalized = llama_hetero_canonical_backend(value != nullptr ? value : "");
    if (normalized.empty() || normalized == "cpu") {
        return nullptr;
    }
    if (normalized == "opencl") {
        return "GPUOpenCL";
    }
    if (normalized == "qnn-npu") {
        return "qnn-npu";
    }
    if (normalized == "qnn-gpu") {
        return "qnn-gpu";
    }
    if (normalized == "qnn-cpu") {
        return "qnn-cpu";
    }

    return value;
}

bool hetero_route_requests_qnn(const llama_hetero_route_spec & route) {
    static constexpr std::array<llama_hetero_route_stage, 5> kStages = {{
        llama_hetero_route_stage::ATTN_PROJ,
        llama_hetero_route_stage::ATTN_CORE,
        llama_hetero_route_stage::ATTN_OUT,
        llama_hetero_route_stage::FFN,
        llama_hetero_route_stage::OUTPUT,
    }};

    for (const auto stage : kStages) {
        if (llama_hetero_is_qnn_backend(route.backend_for(stage))) {
            return true;
        }
    }

    return false;
}

bool llama_context_read_u64_file(const std::string & path, uint64_t & value) {
    if (path.empty()) {
        return false;
    }

    FILE * file = std::fopen(path.c_str(), "r");
    if (file == nullptr) {
        return false;
    }

    char buffer[128] = {};
    const bool ok = std::fgets(buffer, sizeof(buffer), file) != nullptr;
    std::fclose(file);
    if (!ok) {
        return false;
    }

    char * end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(buffer, &end, 10);
    if (errno != 0 || end == buffer) {
        return false;
    }

    value = static_cast<uint64_t>(parsed);
    return true;
}

bool llama_context_write_u64_file(const std::string & path, uint64_t value) {
    if (path.empty()) {
        return false;
    }

    FILE * file = std::fopen(path.c_str(), "w");
    if (file == nullptr) {
        return false;
    }

    const int written = std::fprintf(file, "%" PRIu64 "\n", value);
    const int close_rc = std::fclose(file);
    return written > 0 && close_rc == 0;
}

bool seq0_prefix_tokens_from_memory(const llama_memory_i * memory, size_t & n_tokens) {
    n_tokens = 0;
    if (memory == nullptr) {
        return true;
    }

    const auto seq0_prefix_from_cache = [](const llama_kv_cache * kv_cache, size_t & n_tokens) {
        n_tokens = 0;
        if (kv_cache == nullptr) {
            return false;
        }

        uint32_t cache_tokens = 0;
        if (!kv_cache->seq_is_physical_prefix(0, &cache_tokens)) {
            return false;
        }

        n_tokens = cache_tokens;
        return true;
    };

    if (auto * kv_cache = dynamic_cast<const llama_kv_cache *>(memory)) {
        return seq0_prefix_from_cache(kv_cache, n_tokens);
    }

    if (auto * kv_cache_iswa = dynamic_cast<const llama_kv_cache_iswa *>(memory)) {
        size_t base_tokens = 0;
        size_t swa_tokens = 0;
        if (!seq0_prefix_from_cache(kv_cache_iswa->get_base(), base_tokens) ||
            !seq0_prefix_from_cache(kv_cache_iswa->get_swa(), swa_tokens) ||
            base_tokens != swa_tokens) {
            return false;
        }

        n_tokens = base_tokens;
        return true;
    }

    if (auto * hybrid_memory = dynamic_cast<const llama_memory_hybrid *>(memory)) {
        return seq0_prefix_from_cache(hybrid_memory->get_mem_attn(), n_tokens);
    }

    if (auto * hybrid_iswa_memory = dynamic_cast<const llama_memory_hybrid_iswa *>(memory)) {
        auto * attn_cache = hybrid_iswa_memory->get_mem_attn();
        size_t base_tokens = 0;
        size_t swa_tokens = 0;
        if (attn_cache == nullptr ||
            !seq0_prefix_from_cache(attn_cache->get_base(), base_tokens) ||
            !seq0_prefix_from_cache(attn_cache->get_swa(), swa_tokens) ||
            base_tokens != swa_tokens) {
            return false;
        }

        n_tokens = base_tokens;
        return true;
    }

    return memory->seq_pos_max(0) < 0;
}

bool batch_extract_appendable_seq0_tokens(
        const llama_batch & batch,
        size_t              expected_first_pos,
        std::vector<llama_token> & out_tokens) {
    if (batch.token == nullptr || batch.n_tokens <= 0) {
        return false;
    }

    out_tokens.clear();
    out_tokens.reserve(batch.n_tokens);

    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        const int32_t n_seq_id = batch.n_seq_id != nullptr ? batch.n_seq_id[i] : 1;
        if (n_seq_id != 1) {
            return false;
        }

        const llama_seq_id seq_id = batch.seq_id != nullptr ? batch.seq_id[i][0] : 0;
        if (seq_id != 0) {
            return false;
        }

        if (batch.pos != nullptr) {
            const llama_pos expected_pos = static_cast<llama_pos>(expected_first_pos + static_cast<size_t>(i));
            if (batch.pos[i] != expected_pos) {
                return false;
            }
        }

        out_tokens.push_back(batch.token[i]);
    }

    return true;
}

} // namespace

bool llama_context_qnn_accel_backend_requested(
        const std::vector<std::string> & device_names,
        const llama_hetero_route_spec & hetero_route,
        const llama_hetero_route_spec & dynamic_prefill_route,
        const llama_hetero_route_spec & dynamic_decode_route,
        const llama_hetero_route_spec & dynamic_fallback_route) {
    const bool model_requests_qnn_accel = std::any_of(device_names.begin(), device_names.end(), [](const std::string & name) {
        const std::string normalized = llama_hetero_canonical_backend(name);
        return normalized == "qnn-npu" || normalized == "qnn-cpu";
    });

    return model_requests_qnn_accel ||
           hetero_route_requests_qnn(hetero_route) ||
           hetero_route_requests_qnn(dynamic_prefill_route) ||
           hetero_route_requests_qnn(dynamic_decode_route) ||
           hetero_route_requests_qnn(dynamic_fallback_route);
}

bool llama_context_should_disable_cpu_qnn_host_fallback(
        bool first_device_is_qnn,
        bool routes_use_opencl,
        bool routes_use_cpu) {
    return first_device_is_qnn && (routes_use_opencl || routes_use_cpu);
}

//
// llama_context
//

llama_context::llama_context(
        const llama_model & model,
              llama_context_params params) :
    model(model),
    kv_type_k(params.type_k),
    kv_type_v(params.type_v),
    kv_swa_full(params.swa_full),
    cvec(std::make_unique<llama_adapter_cvec>()),
    loras(std::make_unique<llama_adapter_loras>()),
    balloc(std::make_unique<llama_batch_allocr>(model.hparams.n_pos_per_embd())) {
    // TODO warning when creating llama_context with awkward ctx size that is not a power of 2,
    //     may need to be backend-dependent
    LLAMA_LOG_INFO("%s: constructing llama_context\n", __func__);

    t_start_us = model.t_start_us;
    t_load_us  = model.t_load_us;

    const auto & hparams = model.hparams;

    cparams.n_seq_max = std::max(1u, params.n_seq_max);
    if (cparams.n_seq_max > LLAMA_MAX_SEQ) {
        throw std::runtime_error("n_seq_max must be <= " + std::to_string(LLAMA_MAX_SEQ));
    }

    cparams.n_threads        = params.n_threads;
    cparams.n_threads_batch  = params.n_threads_batch;
    cparams.yarn_ext_factor  = params.yarn_ext_factor  >= 0.0f ? params.yarn_ext_factor  : hparams.yarn_ext_factor;
    cparams.yarn_attn_factor = params.yarn_attn_factor >= 0.0f ? params.yarn_attn_factor : hparams.yarn_attn_factor;
    cparams.yarn_beta_fast   = params.yarn_beta_fast   >= 0.0f ? params.yarn_beta_fast   : hparams.yarn_beta_fast;
    cparams.yarn_beta_slow   = params.yarn_beta_slow   >= 0.0f ? params.yarn_beta_slow   : hparams.yarn_beta_slow;
    cparams.embeddings       = params.embeddings;
    cparams.offload_kqv      = params.offload_kqv;
    cparams.no_perf          = params.no_perf;
    cparams.pooling_type     = params.pooling_type;
    cparams.warmup           = false;

    cparams.n_ctx            = params.n_ctx           == 0    ? hparams.n_ctx_train           : params.n_ctx;
    cparams.rope_freq_base   = params.rope_freq_base  == 0.0f ? hparams.rope_freq_base_train  : params.rope_freq_base;
    cparams.rope_freq_scale  = params.rope_freq_scale == 0.0f ? hparams.rope_freq_scale_train : params.rope_freq_scale;

    cparams.n_ctx_orig_yarn  = params.yarn_orig_ctx    != 0 ? params.yarn_orig_ctx    :
                               hparams.n_ctx_orig_yarn != 0 ? hparams.n_ctx_orig_yarn :
                                                              hparams.n_ctx_train;

    cparams.cb_eval           = params.cb_eval;
    cparams.cb_eval_user_data = params.cb_eval_user_data;

    // Initialize backend samplers here so they are part of the sampling graph
    // before the reserve passes run later in this function. This avoids a later
    // re-reserve when graph nodes change.
    if (params.samplers != nullptr && params.n_samplers > 0) {
        for (size_t i = 0; i < params.n_samplers; ++i) {
            const auto & config = params.samplers[i];

            if (llama_sampler_chain_get(config.sampler, -1) == nullptr) {
                throw std::runtime_error("the backend samplers must be of type llama_sampler_chain");
            }

            if (set_sampler(config.seq_id, config.sampler)) {
                const int n_samplers = llama_sampler_chain_n(config.sampler);

                LLAMA_LOG_INFO("%s: setting backend sampler for seq_id %d (n = %d)\n", __func__, config.seq_id, n_samplers);
            }
        }
    }

    auto rope_scaling_type = params.rope_scaling_type;
    if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED) {
        rope_scaling_type = hparams.rope_scaling_type_train;
    }

    if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_NONE) {
        cparams.rope_freq_scale = 1.0f; // never scale if scaling type is none
    }

    if (cparams.yarn_ext_factor < 0.0f) { // negative indicates 'not set'
        cparams.yarn_ext_factor = rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN ? 1.0f : 0.0f;
    }

    if (cparams.yarn_ext_factor != 0) {
        static auto get_mscale = [](float scale, float mscale) {
            return scale <= 1.0f ? 1.0f : (0.1f * mscale * logf(scale) + 1.0f);
        };

        const float factor = 1.0f / cparams.rope_freq_scale;

        // ref: https://github.com/huggingface/transformers/blob/6d00f6b0a5679c36510f203e4226e36f517c3032/src/transformers/modeling_rope_utils.py#L336-L348
        if (hparams.rope_yarn_log_mul != 0.0f) {
            // note: here we assume `mscale == 1.0f`
            // TODO: start reading the actual value of mscale and handle the case where it is not 1.0f
                  float mscale          = 1.0f;
            const float mscale_all_dims = hparams.rope_yarn_log_mul;

            // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
            // special-case DEEPSEEK v2:
            // https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/blob/main/config.json#L42-L43
            if (model.arch == LLM_ARCH_DEEPSEEK2 && mscale_all_dims != 1.0f) {
                mscale = mscale_all_dims;
            }

            cparams.yarn_attn_factor = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dims);

            LLAMA_LOG_WARN("%s: setting new yarn_attn_factor = %.4f (mscale == %.1f, mscale_all_dim = %.1f)\n",
                    __func__, cparams.yarn_attn_factor, mscale, mscale_all_dims);
        } else {
            cparams.yarn_attn_factor = get_mscale(factor, 1.0f);
        }

        // when YARN is applied with yarn_ext_factor != 0.0f, we need to cancel this factor:
        // https://github.com/ggml-org/llama.cpp/blob/a81a569577cc38b32558958b048228150be63eae/ggml/src/ggml-cpu/ops.cpp#L5541-L5544
        //
        // ref: https://github.com/ggml-org/llama.cpp/discussions/7416
        //      https://github.com/ggml-org/llama.cpp/pull/17945
        cparams.yarn_attn_factor *= 1.0f / (1.0f + 0.1f * logf(factor));
    }

    cparams.yarn_attn_factor *= hparams.rope_attn_factor;

    if (cparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
        if (hparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
            cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
        } else {
            cparams.pooling_type = hparams.pooling_type;
        }
    }

    if (params.attention_type == LLAMA_ATTENTION_TYPE_UNSPECIFIED) {
        cparams.causal_attn = hparams.causal_attn;
    } else {
        cparams.causal_attn = params.attention_type == LLAMA_ATTENTION_TYPE_CAUSAL;
    }

    cparams.flash_attn = params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.auto_fa    = params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO;

    cparams.fused_gdn_ar = true;
    cparams.fused_gdn_ch = true;
    cparams.auto_fgdn    = true;

    // with causal attention, the batch size is limited by the context size
    cparams.n_batch = cparams.causal_attn ? std::min(cparams.n_ctx, params.n_batch) : params.n_batch;

    cparams.n_ubatch = std::min(cparams.n_batch, params.n_ubatch == 0 ? params.n_batch : params.n_ubatch);

    cparams.op_offload = params.op_offload;
    cparams.kv_unified = params.kv_unified;

    // initialized later
    cparams.pipeline_parallel = false;

    {
        const char * LLAMA_GRAPH_REUSE_DISABLE = getenv("LLAMA_GRAPH_REUSE_DISABLE");
        graph_reuse_disable = LLAMA_GRAPH_REUSE_DISABLE ? (atoi(LLAMA_GRAPH_REUSE_DISABLE) != 0) : graph_reuse_disable;

        if (graph_reuse_disable) {
            LLAMA_LOG_WARN("%s: graph reuse disabled\n", __func__);
        }
    }

    // ref: https://github.com/ggml-org/llama.cpp/pull/17046#discussion_r2503085732
    cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256);

    if (cparams.kv_unified) {
        cparams.n_ctx_seq = cparams.n_ctx;
    } else {
        cparams.n_ctx_seq = cparams.n_ctx / cparams.n_seq_max;
        cparams.n_ctx_seq = GGML_PAD(cparams.n_ctx_seq, 256);

        if (cparams.n_ctx_seq == 0) {
            throw std::runtime_error("n_ctx_seq == 0");
        }

        if (cparams.n_ctx != cparams.n_ctx_seq * cparams.n_seq_max) {
            cparams.n_ctx =  cparams.n_ctx_seq * cparams.n_seq_max;
            LLAMA_LOG_WARN("%s: n_ctx is not divisible by n_seq_max - rounding down to %u\n", __func__, cparams.n_ctx);
        }
    }

    LLAMA_LOG_INFO("%s: n_seq_max     = %u\n",   __func__, cparams.n_seq_max);
    LLAMA_LOG_INFO("%s: n_ctx         = %u\n",   __func__, cparams.n_ctx);
    LLAMA_LOG_INFO("%s: n_ctx_seq     = %u\n",   __func__, cparams.n_ctx_seq);
    LLAMA_LOG_INFO("%s: n_batch       = %u\n",   __func__, cparams.n_batch);
    LLAMA_LOG_INFO("%s: n_ubatch      = %u\n",   __func__, cparams.n_ubatch);
    LLAMA_LOG_INFO("%s: causal_attn   = %d\n",   __func__, cparams.causal_attn);
    LLAMA_LOG_INFO("%s: flash_attn    = %s\n",   __func__, llama_flash_attn_type_name(params.flash_attn_type));
    LLAMA_LOG_INFO("%s: kv_unified    = %s\n",   __func__, cparams.kv_unified ? "true" : "false");
    LLAMA_LOG_INFO("%s: freq_base     = %.1f\n", __func__, cparams.rope_freq_base);
    LLAMA_LOG_INFO("%s: freq_scale    = %g\n",   __func__, cparams.rope_freq_scale);

    if (cparams.n_ctx_seq < hparams.n_ctx_train) {
        LLAMA_LOG_WARN("%s: n_ctx_seq (%u) < n_ctx_train (%u) -- the full capacity of the model will not be utilized\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (cparams.n_ctx_seq > hparams.n_ctx_train) {
        LLAMA_LOG_WARN("%s: n_ctx_seq (%u) > n_ctx_train (%u) -- possible training context overflow\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    const bool hetero_plan_from_params =
        params.hetero_phase_route != nullptr || params.hetero_kv_layout != nullptr;

    hetero_plan = hetero_plan_from_params
        ? llama_hetero_build_execution_plan(params.hetero_phase_route, params.hetero_kv_layout)
        : model.get_hetero_plan();
    hetero_plan_base   = hetero_plan;
    aot_active_route_requests_qnn = hetero_route_requests_qnn(hetero_plan.route);
    dynamic_route_config = llama_dynamic_route_config_from_env();

    if (hetero_dynamic_trace_timing_enabled()) {
        hetero_decode_token_trace_records.reserve(cparams.n_ctx);
    }

    if (hetero_plan_from_params && !llama_hetero_execution_plan_equals(hetero_plan, model.get_hetero_plan())) {
        const std::string ctx_route = llama_hetero_format_route_spec(hetero_plan.route);
        const std::string model_route = llama_hetero_format_route_spec(model.get_hetero_plan().route);
        LLAMA_LOG_WARN("%s: context hetero route overrides the model-load plan. Graph routing will use route=%s, but tensor residency was chosen with model route=%s. For lower switching / staging overhead and future QNN integration, prefer setting llama_model_params.hetero_phase_route / hetero_kv_layout before model load.\n",
                __func__,
                ctx_route.empty() ? "<default>" : ctx_route.c_str(),
                model_route.empty() ? "<default>" : model_route.c_str());
    }

    const auto & hetero_route = hetero_plan.route;
    const auto dynamic_schedule_uses_qnn = [&]() {
        for (const auto & entry : dynamic_route_config.decode_schedule) {
            if (llama_dynamic_route_uses_qnn(entry.route.plan)) {
                return true;
            }
        }
        return false;
    };
    const auto dynamic_schedule_uses_opencl = [&]() {
        for (const auto & entry : dynamic_route_config.decode_schedule) {
            if (llama_dynamic_route_uses_opencl(entry.route.plan)) {
                return true;
            }
        }
        return false;
    };
    const auto dynamic_schedule_uses_cpu = [&]() {
        for (const auto & entry : dynamic_route_config.decode_schedule) {
            if (entry.route.plan.has_any_route() &&
                llama_hetero_is_cpu_backend(llama_hetero_phase_backend_for_route(entry.route.plan.route))) {
                return true;
            }
        }
        return false;
    };
    const bool dynamic_cpu_opencl_zero_copy =
        llama_hetero_route_has_cpu_opencl_adjacent_boundary(dynamic_route_config.prefill.plan.route) ||
        llama_hetero_route_has_cpu_opencl_adjacent_boundary(dynamic_route_config.decode.plan.route) ||
        llama_hetero_route_has_cpu_opencl_adjacent_boundary(dynamic_route_config.fallback.plan.route);
    const bool dynamic_qnn_shared_host =
        llama_hetero_route_has_qnn_adjacent_boundary(dynamic_route_config.prefill.plan.route) ||
        llama_hetero_route_has_qnn_adjacent_boundary(dynamic_route_config.decode.plan.route) ||
        llama_hetero_route_has_qnn_adjacent_boundary(dynamic_route_config.fallback.plan.route);
    const bool hetero_cpu_opencl_zero_copy =
        llama_hetero_route_has_cpu_opencl_adjacent_boundary(hetero_route) ||
        dynamic_cpu_opencl_zero_copy;
    const auto env_flag_enabled = [](const char * name) {
        const char * value = std::getenv(name);
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    };
    const bool enable_cpu_opencl_shared_host_experimental =
        env_flag_enabled("GGML_HETERO_ENABLE_CPU_OPENCL_SHARED_HOST");
    const bool disable_cpu_opencl_shared_host =
        env_flag_enabled("GGML_HETERO_DISABLE_CPU_OPENCL_SHARED_HOST");
    const bool qnn_shared_host_experimental =
        env_flag_enabled("GGML_HETERO_QNN_SHARED_HOST");
    const bool hetero_trace_share =
        env_flag_enabled("GGML_HETERO_TRACE_SHARE");
    const bool hetero_qnn_shared_host_requested =
        qnn_shared_host_experimental && (
            llama_hetero_route_has_qnn_adjacent_boundary(hetero_route) ||
            dynamic_qnn_shared_host);
    bool hetero_qnn_shared_host_compute = false;
    const bool enable_cpu_opencl_shared_host =
        hetero_cpu_opencl_zero_copy &&
        enable_cpu_opencl_shared_host_experimental &&
        !disable_cpu_opencl_shared_host;
    bool hetero_shared_host_compute = enable_cpu_opencl_shared_host;
    ggml_backend_buffer_type_t shared_host_buft = nullptr;
    std::vector<std::string> model_device_names;
    model_device_names.reserve(model.devices.size());
    for (auto * dev : model.devices) {
        model_device_names.emplace_back(dev != nullptr ? ggml_backend_dev_name(dev) : "");
    }
    const bool qnn_backend_requested = llama_context_qnn_accel_backend_requested(
        model_device_names,
        hetero_route,
        dynamic_route_config.prefill.plan.route,
        dynamic_route_config.decode.plan.route,
        dynamic_route_config.fallback.plan.route) ||
        dynamic_schedule_uses_qnn();

    if (!hparams.vocab_only) {
        // GPU backends
        const auto backend_device_already_added = [&](ggml_backend_dev_t dev) {
            return std::any_of(backends.begin(), backends.end(), [&](const ggml_backend_ptr & backend) {
                return ggml_backend_get_device(backend.get()) == dev;
            });
        };

        for (auto * dev : model.devices) {
            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            if (backend == nullptr) {
                throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev)));
            }
            backends.emplace_back(backend);
        }

        // add ACCEL backends (such as BLAS)
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                if (!qnn_backend_requested && llama_hetero_is_qnn_backend(ggml_backend_dev_name(dev))) {
                    continue;
                }
                if (backend_device_already_added(dev)) {
                    continue;
                }
                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                if (backend == nullptr) {
                    throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev)));
                }
                backends.emplace_back(backend);
            }
        }

        // add CPU backend
        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (backend_cpu == nullptr) {
            throw std::runtime_error("failed to initialize CPU backend");
        }
        backends.emplace_back(backend_cpu);

        ensure_hetero_backends_for_route(hetero_route, "hetero");
        ensure_dynamic_route_backends_ready(dynamic_route_config);

        const auto find_opencl_host_buft = [&]() -> ggml_backend_buffer_type_t {
            for (const auto & backend : backends) {
                ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
                if (dev != nullptr && std::strcmp(ggml_backend_dev_name(dev), "GPUOpenCL") == 0) {
                    return ggml_backend_dev_host_buffer_type(dev);
                }
            }

            return nullptr;
        };

        const auto find_qnn_host_buft = [&]() -> ggml_backend_buffer_type_t {
            for (const auto & backend : backends) {
                ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
                if (dev != nullptr && std::strcmp(ggml_backend_dev_name(dev), "qnn-npu") == 0) {
                    return ggml_backend_dev_host_buffer_type(dev);
                }
            }

            return nullptr;
        };

        ggml_backend_buffer_type_t opencl_shared_host_buft = find_opencl_host_buft();
        ggml_backend_buffer_type_t qnn_shared_host_buft = find_qnn_host_buft();
        const bool opencl_host_buffer_available = opencl_shared_host_buft != nullptr;
        const bool qnn_host_buffer_available = qnn_shared_host_buft != nullptr;
        const auto opencl_supports_buft = [&](ggml_backend_buffer_type_t buft) -> bool {
            if (buft == nullptr) {
                return false;
            }

            ggml_backend_dev_t opencl_dev = ggml_backend_dev_by_name("GPUOpenCL");
            return opencl_dev != nullptr && ggml_backend_dev_supports_buft(opencl_dev, buft);
        };

        const bool opencl_can_alias_qnn_host = opencl_supports_buft(qnn_shared_host_buft);

        if (hetero_qnn_shared_host_requested) {
            if (enable_cpu_opencl_shared_host) {
                if (qnn_host_buffer_available && opencl_can_alias_qnn_host) {
                    shared_host_buft = qnn_shared_host_buft;
                    hetero_qnn_shared_host_compute = true;
                    LLAMA_LOG_INFO("%s: enabling unified CPU/QNN/OpenCL shared-host compute buffers with %s for hetero decode stages\n",
                            __func__,
                            ggml_backend_buft_name(shared_host_buft));
                } else {
                    shared_host_buft = opencl_shared_host_buft;
                    hetero_qnn_shared_host_compute = false;
                    LLAMA_LOG_WARN("%s: requested unified CPU/QNN/OpenCL shared-host compute buffers, but qnn-npu-host is unavailable or not OpenCL-compatible in this context. Falling back to %s for compute tensors; the attn KV contract may still allocate qnn-npu-host separately.\n",
                            __func__,
                            shared_host_buft ? ggml_backend_buft_name(shared_host_buft) : "<null>");
                }
            } else {
                shared_host_buft = qnn_shared_host_buft;
                hetero_qnn_shared_host_compute = qnn_host_buffer_available;
            }
        } else if (enable_cpu_opencl_shared_host) {
            shared_host_buft = opencl_shared_host_buft;
        }

        hetero_shared_host_compute =
            shared_host_buft != nullptr &&
            (enable_cpu_opencl_shared_host || hetero_qnn_shared_host_compute);

        if (hetero_cpu_opencl_zero_copy && !enable_cpu_opencl_shared_host) {
            if (disable_cpu_opencl_shared_host) {
                LLAMA_LOG_INFO("%s: disabling CPU/OpenCL shared-host compute buffers via GGML_HETERO_DISABLE_CPU_OPENCL_SHARED_HOST\n",
                               __func__);
            } else {
                LLAMA_LOG_WARN("%s: CPU/OpenCL shared-host compute buffers are disabled by default because the current OpenCL shared-host path can corrupt decode semantics. Set GGML_HETERO_ENABLE_CPU_OPENCL_SHARED_HOST=1 to re-enable it experimentally.\n",
                               __func__);
            }
        }

        hetero_kv_contract_allocated = llama_hetero_finalize_kv_contract(
                hetero_plan.attn_kv,
                opencl_host_buffer_available,
                qnn_host_buffer_available);

        const auto maybe_promote_allocated_kv = [&](const llama_dynamic_route_candidate & candidate) {
            if (!candidate.configured ||
                llama_hetero_kv_contract_can_satisfy(hetero_kv_contract_allocated, candidate.plan.attn_kv)) {
                return;
            }

            llama_hetero_kv_contract upgraded = llama_hetero_finalize_kv_contract(
                    candidate.plan.attn_kv,
                    opencl_host_buffer_available,
                    qnn_host_buffer_available);

            if (llama_hetero_kv_contract_can_satisfy(upgraded, hetero_kv_contract_allocated)) {
                LLAMA_LOG_INFO("%s: promoting allocated attn KV contract for %s to layout=%s transfer=%s\n",
                        __func__,
                        candidate.label.empty() ? "<unnamed>" : candidate.label.c_str(),
                        llama_hetero_kv_layout_name(upgraded.layout),
                        llama_hetero_kv_transfer_mode_name(upgraded.transfer));
                hetero_kv_contract_allocated = std::move(upgraded);
                return;
            }

            LLAMA_LOG_WARN("%s: dynamic route %s requests attn KV contract layout=%s transfer=%s, which is incompatible with the current allocated contract layout=%s transfer=%s. This candidate will remain runtime-rejected until the context is rebuilt with a compatible contract.\n",
                    __func__,
                    candidate.label.empty() ? "<unnamed>" : candidate.label.c_str(),
                    llama_hetero_kv_layout_name(candidate.plan.attn_kv.layout),
                    llama_hetero_kv_transfer_mode_name(candidate.plan.attn_kv.transfer),
                    llama_hetero_kv_layout_name(hetero_kv_contract_allocated.layout),
                    llama_hetero_kv_transfer_mode_name(hetero_kv_contract_allocated.transfer));
        };

        maybe_promote_allocated_kv(dynamic_route_config.prefill);
        maybe_promote_allocated_kv(dynamic_route_config.decode);
        maybe_promote_allocated_kv(dynamic_route_config.fallback);

        const auto maybe_promote_dynamic_phase_shared_qnn_kv = [&]() {
            const llama_dynamic_route_candidate * initial_decode =
                llama_context_initial_dynamic_decode_candidate(dynamic_route_config);
            if (!dynamic_route_config.prefill.configured || initial_decode == nullptr) {
                return;
            }

            if (!llama_hetero_route_is_phase_homogeneous(dynamic_route_config.prefill.plan.route) ||
                !llama_hetero_route_is_phase_homogeneous(initial_decode->plan.route)) {
                return;
            }

            llama_hetero_kv_contract upgraded = llama_dynamic_phase_shared_qnn_kv_contract(
                    llama_hetero_phase_backend_for_route(dynamic_route_config.prefill.plan.route),
                    llama_hetero_phase_backend_for_route(initial_decode->plan.route),
                    qnn_host_buffer_available,
                    opencl_can_alias_qnn_host);
            if (!upgraded.stage_boundary_active() ||
                llama_hetero_kv_contract_can_satisfy(hetero_kv_contract_allocated, upgraded)) {
                return;
            }

            if (llama_hetero_kv_contract_can_satisfy(upgraded, hetero_kv_contract_allocated)) {
                LLAMA_LOG_INFO("%s: promoting allocated attn KV contract for dynamic qnn-prefill/opencl-decode to layout=%s transfer=%s\n",
                        __func__,
                        llama_hetero_kv_layout_name(upgraded.layout),
                        llama_hetero_kv_transfer_mode_name(upgraded.transfer));
                hetero_kv_contract_allocated = std::move(upgraded);
                return;
            }

            LLAMA_LOG_WARN("%s: dynamic qnn-prefill/opencl-decode shared KV requires layout=%s transfer=%s, which is incompatible with the current allocated contract layout=%s transfer=%s. The direct shared-KV fast path will stay disabled until the context is rebuilt with a compatible contract.\n",
                    __func__,
                    llama_hetero_kv_layout_name(upgraded.layout),
                    llama_hetero_kv_transfer_mode_name(upgraded.transfer),
                    llama_hetero_kv_layout_name(hetero_kv_contract_allocated.layout),
                    llama_hetero_kv_transfer_mode_name(hetero_kv_contract_allocated.transfer));
        };

        maybe_promote_dynamic_phase_shared_qnn_kv();

        const bool first_device_is_opencl =
            !model.devices.empty() &&
            model.devices[0] != nullptr &&
            std::strcmp(ggml_backend_dev_name(model.devices[0]), "GPUOpenCL") == 0;
        const bool allow_cpu_opencl_host_fallback =
            enable_cpu_opencl_shared_host_experimental && !disable_cpu_opencl_shared_host;

        if (first_device_is_opencl && !allow_cpu_opencl_host_fallback) {
            LLAMA_LOG_WARN("%s: CPU fallback to GPUOpenCL host buffers is disabled by default because the current OpenCL host-buffer path can corrupt decode semantics. Set GGML_HETERO_ENABLE_CPU_OPENCL_SHARED_HOST=1 to re-enable it experimentally.\n",
                           __func__);
        }

        if (hetero_plan.attn_kv.stage_boundary_active()) {
            LLAMA_LOG_INFO("%s: hetero attn KV contract requested layout=%s transfer=%s producer=%s consumer=%s storage=%s reason=%s\n",
                    __func__,
                    llama_hetero_kv_layout_name(hetero_plan.attn_kv.layout),
                    llama_hetero_kv_transfer_mode_name(hetero_plan.attn_kv.transfer),
                    hetero_plan.attn_kv.producer_backend.c_str(),
                    hetero_plan.attn_kv.consumer_backend.c_str(),
                    hetero_plan.attn_kv.storage_backend.empty() ? "<unset>" : hetero_plan.attn_kv.storage_backend.c_str(),
                    hetero_plan.attn_kv.reason.empty() ? "<none>" : hetero_plan.attn_kv.reason.c_str());
            LLAMA_LOG_INFO("%s: hetero attn KV contract allocated layout=%s transfer=%s zero_copy=%s available=%s reason=%s\n",
                    __func__,
                    llama_hetero_kv_layout_name(hetero_kv_contract_allocated.layout),
                    llama_hetero_kv_transfer_mode_name(hetero_kv_contract_allocated.transfer),
                    hetero_kv_contract_allocated.zero_copy ? "true" : "false",
                    hetero_kv_contract_allocated.buffer_available ? "true" : "false",
                    hetero_kv_contract_allocated.reason.empty() ? "<none>" : hetero_kv_contract_allocated.reason.c_str());
        }

        if (dynamic_route_config.enabled()) {
            const auto route_string_or = [](const llama_dynamic_route_candidate & candidate) {
                const std::string route = llama_hetero_format_route_spec(candidate.plan.route);
                return route.empty() ? std::string("<unset>") : route;
            };
            const auto schedule_string = [&]() {
                std::string result;
                for (const auto & entry : dynamic_route_config.decode_schedule) {
                    if (!result.empty()) {
                        result += ";";
                    }
                    const std::string route = llama_hetero_format_route_spec(entry.route.plan.route);
                    result += std::to_string(entry.start_token) + ":" +
                        (route.empty() ? std::string("<unset>") : route);
                }
                return result.empty() ? std::string("<unset>") : result;
            };

            LLAMA_LOG_INFO("%s: dynamic route mode=%s prefill=%s decode=%s fallback=%s decode_schedule=%s slo_us=%" PRId64 " allow_qnn=%s decode_switch_after=%" PRIu64 "\n",
                    __func__,
                    llama_dynamic_route_mode_name(dynamic_route_config.mode),
                    route_string_or(dynamic_route_config.prefill).c_str(),
                    route_string_or(dynamic_route_config.decode).c_str(),
                    route_string_or(dynamic_route_config.fallback).c_str(),
                    schedule_string().c_str(),
                    dynamic_route_config.slo_us,
                    dynamic_route_config.allow_qnn ? "true" : "false",
                    dynamic_route_config.decode_switch_after);
        }

        // create a list of the set_n_threads functions in the backends
        for (auto & backend : backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            if (reg) {
                auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
                if (ggml_backend_set_n_threads_fn) {
                    set_n_threads_fns.emplace_back(backend.get(), ggml_backend_set_n_threads_fn);
                }
            }
        }

        llama_set_abort_callback(this, params.abort_callback, params.abort_callback_data);

        // graph outputs buffer
        {
            if (output_reserve(params.n_seq_max) < params.n_seq_max) {
                throw std::runtime_error("failed to reserve initial output buffer");
            }

            LLAMA_LOG_INFO("%s: %10s  output buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buffer_name    (buf_output.get()),
                    ggml_backend_buffer_get_size(buf_output.get()) / 1024.0 / 1024.0);
        }
    }

    // init the memory module
    if (!hparams.vocab_only) {
        kv_attn_v_trans = !cparams.flash_attn;

        llama_memory_params params_mem = {
            /*.type_k   =*/ params.type_k,
            /*.type_v   =*/ params.type_v,
            /*.swa_full =*/ params.swa_full,
            /*.attn_v_trans =*/ kv_attn_v_trans,
            /*.attn_v_trans_pinned =*/ true,
            /*.kv_contract =*/ hetero_kv_contract_allocated,
        };

        memory.reset(model.create_memory(params_mem, cparams));
    }

    // init backends
    if (!hparams.vocab_only) {
        LLAMA_LOG_DEBUG("%s: enumerating backends\n", __func__);

        backend_buft.clear();
        backend_ptrs.clear();
        backend_buf_exp_size.clear();

        ggml_backend_buffer_type_t shared_host_compute_buft = nullptr;
        const bool first_device_is_opencl_local =
            !model.devices.empty() &&
            model.devices[0] != nullptr &&
            std::strcmp(ggml_backend_dev_name(model.devices[0]), "GPUOpenCL") == 0;
        const bool first_device_is_qnn_local =
            !model.devices.empty() &&
            model.devices[0] != nullptr &&
            llama_hetero_is_qnn_backend(ggml_backend_dev_name(model.devices[0]));
        const bool allow_cpu_opencl_host_fallback_local =
            enable_cpu_opencl_shared_host_experimental && !disable_cpu_opencl_shared_host;
        const bool routes_use_opencl_local =
            llama_hetero_is_opencl_backend(llama_hetero_phase_backend_for_route(hetero_route)) ||
            llama_dynamic_route_uses_opencl(dynamic_route_config.prefill.plan) ||
            llama_dynamic_route_uses_opencl(dynamic_route_config.decode.plan) ||
            llama_dynamic_route_uses_opencl(dynamic_route_config.fallback.plan) ||
            dynamic_schedule_uses_opencl();
        const auto plan_uses_cpu = [](const llama_hetero_execution_plan & plan) {
            return plan.has_any_route() &&
                   llama_hetero_is_cpu_backend(llama_hetero_phase_backend_for_route(plan.route));
        };
        const bool routes_use_cpu_local =
            llama_hetero_is_cpu_backend(llama_hetero_phase_backend_for_route(hetero_route)) ||
            plan_uses_cpu(dynamic_route_config.prefill.plan) ||
            plan_uses_cpu(dynamic_route_config.decode.plan) ||
            plan_uses_cpu(dynamic_route_config.fallback.plan) ||
            dynamic_schedule_uses_cpu();
        const bool disable_cpu_qnn_host_fallback_local =
            llama_context_should_disable_cpu_qnn_host_fallback(
                    first_device_is_qnn_local,
                    routes_use_opencl_local,
                    routes_use_cpu_local);
        const auto backend_dev_type_name = [](enum ggml_backend_dev_type type) -> const char * {
            switch (type) {
                case GGML_BACKEND_DEVICE_TYPE_CPU:   return "CPU";
                case GGML_BACKEND_DEVICE_TYPE_GPU:   return "GPU";
                case GGML_BACKEND_DEVICE_TYPE_IGPU:  return "IGPU";
                case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "ACCEL";
            }
            return "UNKNOWN";
        };

        if (hetero_shared_host_compute) {
            shared_host_compute_buft = shared_host_buft;
            if (shared_host_compute_buft != nullptr) {
                LLAMA_LOG_INFO("%s: enabling shared host compute buffers with %s for hetero decode stages (cpu/opencl=%s, qnn-mixed=%s)\n",
                               __func__,
                               ggml_backend_buft_name(shared_host_compute_buft),
                               enable_cpu_opencl_shared_host ? "true" : "false",
                               hetero_qnn_shared_host_compute ? "true" : "false");
                if (hetero_trace_share) {
                    std::fprintf(stderr,
                                 "ggml_hetero_buft: shared_host=%s cpu_opencl=%d qnn_mixed=%d\n",
                                 ggml_backend_buft_name(shared_host_compute_buft),
                                 (int) enable_cpu_opencl_shared_host,
                                 (int) hetero_qnn_shared_host_compute);
                }
            } else {
                LLAMA_LOG_WARN("%s: requested hetero shared-host compute buffers (cpu/opencl=%s, qnn-mixed=%s), but the selected host buffer type is unavailable\n",
                               __func__,
                               enable_cpu_opencl_shared_host ? "true" : "false",
                               hetero_qnn_shared_host_requested ? "true" : "false");
                if (hetero_trace_share) {
                    std::fprintf(stderr,
                                 "ggml_hetero_buft: shared_host=<null> cpu_opencl=%d qnn_mixed=%d requested_qnn=%d\n",
                                 (int) enable_cpu_opencl_shared_host,
                                 (int) hetero_qnn_shared_host_compute,
                                 (int) hetero_qnn_shared_host_requested);
                }
            }
        }

        for (auto & backend : backends) {
            auto * buft = ggml_backend_get_default_buffer_type(backend.get());
            ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
            auto backend_type = ggml_backend_dev_type(dev);

            if (hetero_shared_host_compute && shared_host_compute_buft != nullptr) {
                const char * backend_name = dev ? ggml_backend_dev_name(dev) : nullptr;
                const bool is_opencl_backend =
                    backend_name != nullptr && std::strcmp(backend_name, "GPUOpenCL") == 0;
                const bool is_qnn_backend =
                    backend_name != nullptr && llama_hetero_is_qnn_backend(backend_name);
                const bool use_shared_host_buft =
                    (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU &&
                     (enable_cpu_opencl_shared_host || hetero_qnn_shared_host_compute)) ||
                    (is_opencl_backend && (enable_cpu_opencl_shared_host || hetero_qnn_shared_host_compute)) ||
                    (is_qnn_backend && hetero_qnn_shared_host_compute);

                if (use_shared_host_buft) {
                    buft = shared_host_compute_buft;
                }
            }

            const bool allow_cpu_device_host_fallback =
                (!first_device_is_opencl_local || allow_cpu_opencl_host_fallback_local) &&
                !disable_cpu_qnn_host_fallback_local;

            if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU &&
                !(hetero_shared_host_compute && shared_host_compute_buft != nullptr) &&
                allow_cpu_device_host_fallback &&
                !model.devices.empty()) {
                // use the host buffer of the first device CPU for faster transfer of the intermediate state
                auto * dev = model.devices[0];
                auto * host_buft = ggml_backend_dev_host_buffer_type(dev);
                if (host_buft) {
                    buft = host_buft;
                }
            }

            if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU &&
                disable_cpu_qnn_host_fallback_local) {
                LLAMA_LOG_INFO("%s: keeping CPU compute buffers on CPU memory because qnn-npu host fallback can corrupt or slow mixed qnn/cpu/opencl contexts\n",
                        __func__);
            }

            if (hetero_shared_host_compute) {
                const char * backend_name = dev ? ggml_backend_dev_name(dev) : ggml_backend_name(backend.get());
                LLAMA_LOG_INFO("%s: hetero compute buft backend=%s type=%s buft=%s\n",
                        __func__,
                        backend_name ? backend_name : "<null>",
                        backend_dev_type_name(backend_type),
                        buft ? ggml_backend_buft_name(buft) : "<null>");
                if (hetero_trace_share) {
                    std::fprintf(stderr,
                                 "ggml_hetero_buft: backend=%s type=%s buft=%s\n",
                                 backend_name ? backend_name : "<null>",
                                 backend_dev_type_name(backend_type),
                                 buft ? ggml_backend_buft_name(buft) : "<null>");
                }
            }

            backend_buft.push_back(buft);
            backend_ptrs.push_back(backend.get());
            backend_buf_exp_size.push_back(0);
        }

        LLAMA_LOG_DEBUG("%s: backend_ptrs.size() = %zu\n", __func__, backend_ptrs.size());

        // TODO: move these checks to ggml_backend_sched
        // enabling pipeline parallelism in the scheduler increases memory usage, so it is only done when necessary
        bool pipeline_parallel =
            model.n_devices() > 1 &&
            model.n_gpu_layers() > model.hparams.n_layer &&
            model.split_mode() == LLAMA_SPLIT_MODE_LAYER &&
            cparams.offload_kqv &&
            !model.has_tensor_overrides();

        // pipeline parallelism requires support for async compute and events in all devices
        if (pipeline_parallel) {
            for (auto & backend : backends) {
                auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));
                if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    // ignore CPU backend
                    // TODO: should we ignore ACCEL types too?
                    continue;
                }
                auto * dev = ggml_backend_get_device(backend.get());
                ggml_backend_dev_props props;
                ggml_backend_dev_get_props(dev, &props);
                if (!props.caps.async || !props.caps.events) {
                    // device does not support async compute or events
                    pipeline_parallel = false;
                    break;
                }
            }
        }

        cparams.pipeline_parallel = pipeline_parallel;

        if (cparams.pipeline_parallel) {
            LLAMA_LOG_INFO("%s: pipeline parallelism enabled\n", __func__);

            if (!graph_reuse_disable) {
                // TODO: figure out a way to make graph reuse work with pipeline parallelism
                // ref: https://github.com/ggml-org/llama.cpp/pull/20463
                LLAMA_LOG_WARN("%s: graph reuse is currently not compatible with pipeline parallelism - disabling\n", __func__);

                graph_reuse_disable = true;
            }
        }

        sched_reserve();
        maybe_prewarm_dynamic_qnn_opencl_kv_aliases();
        maybe_preload_dynamic_qnn_decode_graphs();

        if (!cparams.flash_attn) {
            if (ggml_is_quantized(params.type_v)) {
                throw std::runtime_error("quantized V cache was requested, but this requires Flash Attention");
            }
        }
    }

    // Initialize the full vocabulary token ids for backend samplers.
    {
        const int n_vocab = model.vocab.n_tokens();

        sampling.token_ids_full_vocab.resize(n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            sampling.token_ids_full_vocab[i] = i;
        }
    }
}

bool llama_context::ensure_hetero_backend_ready(const std::string & backend_name, const char * route_name) {
    const char * requested_device_name = canonicalize_hetero_backend_device_name(backend_name.c_str());
    if (requested_device_name == nullptr) {
        return true;
    }

    for (const auto & backend : backends) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
        if (dev != nullptr && std::strcmp(ggml_backend_dev_name(dev), requested_device_name) == 0) {
            return true;
        }
    }

    ggml_backend_dev_t dev = ggml_backend_dev_by_name(requested_device_name);
    if (dev == nullptr) {
        LLAMA_LOG_WARN("%s: requested hetero backend %s via %s is unavailable\n",
                __func__,
                requested_device_name,
                route_name != nullptr ? route_name : "<unknown>");
        return false;
    }

    if (!backend_ptrs.empty()) {
        LLAMA_LOG_WARN("%s: backend %s via %s was not initialized at context creation time. Rebuild the context (or preload it via model/context hetero routes or GGML_HETERO_DYNAMIC_* env) so scheduler buffers can include it.\n",
                __func__,
                requested_device_name,
                route_name != nullptr ? route_name : "<unknown>");
        return false;
    }

    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (backend == nullptr) {
        throw std::runtime_error(format("failed to initialize auxiliary hetero backend %s", requested_device_name));
    }

    LLAMA_LOG_INFO("%s: initialized auxiliary hetero backend %s via %s\n",
            __func__,
            requested_device_name,
            route_name != nullptr ? route_name : "<unknown>");

    if (backend_cpu != nullptr) {
        auto cpu_it = std::find_if(backends.begin(), backends.end(), [&](const auto & candidate) {
            return candidate.get() == backend_cpu;
        });
        if (cpu_it != backends.end()) {
            backends.insert(cpu_it, ggml_backend_ptr(backend));
            return true;
        }
    }

    backends.emplace_back(backend);
    return true;
}

bool llama_context::ensure_hetero_backends_for_route(const llama_hetero_route_spec & route, const char * label_prefix) {
    static constexpr std::array<std::pair<llama_hetero_route_stage, const char *>, 5> kStages = {{
        { llama_hetero_route_stage::ATTN_PROJ, "attn_proj" },
        { llama_hetero_route_stage::ATTN_CORE, "attn_core" },
        { llama_hetero_route_stage::ATTN_OUT,  "attn_out"  },
        { llama_hetero_route_stage::FFN,       "ffn"       },
        { llama_hetero_route_stage::OUTPUT,    "output"    },
    }};

    bool ok = true;
    for (const auto & [stage, suffix] : kStages) {
        const std::string route_name = format("%s.%s",
                label_prefix != nullptr ? label_prefix : "hetero",
                suffix);
        ok = ensure_hetero_backend_ready(route.backend_for(stage), route_name.c_str()) && ok;
    }

    return ok;
}

bool llama_context::ensure_dynamic_route_backends_ready(const llama_dynamic_route_runtime_config & config) {
    bool ok = true;
    if (config.prefill.configured) {
        ok = ensure_hetero_backends_for_route(config.prefill.plan.route, "dynamic.prefill") && ok;
    }
    if (config.decode.configured) {
        ok = ensure_hetero_backends_for_route(config.decode.plan.route, "dynamic.decode") && ok;
    }
    if (config.fallback.configured) {
        ok = ensure_hetero_backends_for_route(config.fallback.plan.route, "dynamic.fallback") && ok;
    }
    for (const auto & entry : config.decode_schedule) {
        const std::string label = "dynamic." + entry.route.label;
        ok = ensure_hetero_backends_for_route(entry.route.plan.route, label.c_str()) && ok;
    }

    return ok;
}

bool llama_context::backend_available_for_route(const std::string & backend_name) const {
    const std::string canonical = llama_hetero_canonical_backend(backend_name);
    if (canonical.empty()) {
        return true;
    }
    if (canonical == "cpu") {
        return backend_cpu != nullptr;
    }

    const char * requested_device_name = canonicalize_hetero_backend_device_name(canonical.c_str());
    if (requested_device_name == nullptr) {
        return false;
    }

    for (const auto & backend : backends) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
        if (dev != nullptr && std::strcmp(ggml_backend_dev_name(dev), requested_device_name) == 0) {
            return true;
        }
    }

    return false;
}

ggml_backend_t llama_context::find_backend_for_route(const std::string & backend_name) const {
    const std::string canonical = llama_hetero_canonical_backend(backend_name);
    if (canonical.empty() || canonical == "cpu") {
        return backend_cpu;
    }

    const char * requested_device_name = canonicalize_hetero_backend_device_name(canonical.c_str());
    if (requested_device_name == nullptr) {
        return nullptr;
    }

    for (const auto & backend : backends) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
        if (dev != nullptr && std::strcmp(ggml_backend_dev_name(dev), requested_device_name) == 0) {
            return backend.get();
        }
    }

    return nullptr;
}

void llama_context::maybe_prewarm_dynamic_qnn_opencl_kv_aliases() {
    if (memory == nullptr || !dynamic_route_config.enabled()) {
        return;
    }

    const std::string prefill_attn_backend =
        llama_hetero_canonical_backend(
                dynamic_route_config.prefill.plan.route.backend_for(llama_hetero_route_stage::ATTN_CORE));
    const std::string decode_attn_backend =
        llama_hetero_canonical_backend(
                dynamic_route_config.decode.plan.route.backend_for(llama_hetero_route_stage::ATTN_CORE));
    const bool generic_qnn_kv_enabled = env_flag_enabled("GGML_QNN_AOT_WRITE_GENERIC_KV");
    const bool experimental_enabled = env_flag_enabled("GGML_OPENCL_EXPERIMENTAL_QNN_DIRECT_HOST_PTR");

    if (!llama_context_should_prewarm_dynamic_qnn_opencl_kv_aliases(
                prefill_attn_backend,
                decode_attn_backend,
                generic_qnn_kv_enabled,
                hetero_kv_contract_allocated,
                experimental_enabled)) {
        return;
    }

    ggml_backend_t opencl_backend = find_backend_for_route("opencl");
    if (opencl_backend == nullptr) {
        LLAMA_LOG_WARN("%s: skipping eager qnn->opencl KV alias prewarm because the OpenCL backend is unavailable\n",
                __func__);
        return;
    }

    llama_opencl_external_host_sync_timing timing;
    if (!sync_dynamic_cpu_opencl_kv(/* host_to_device = */ true, &timing)) {
        LLAMA_LOG_WARN("%s: eager qnn->opencl KV alias prewarm failed; falling back to on-demand alias creation during the first decode switch\n",
                __func__);
        return;
    }

    LLAMA_LOG_INFO("%s: prewarmed experimental qnn->opencl KV alias path alias_us=%" PRId64 " backend_sync_us=%" PRId64 " transfer_us=%" PRId64 "\n",
            __func__,
            timing.alias_us,
            timing.backend_sync_us,
            timing.transfer_us);
}

void llama_context::maybe_preload_dynamic_qnn_decode_graphs() {
    if (!dynamic_route_config.enabled()) {
        return;
    }

    bool decode_uses_qnn =
        llama_dynamic_route_uses_qnn(dynamic_route_config.decode.plan) ||
        llama_dynamic_route_uses_qnn(dynamic_route_config.fallback.plan);
    for (const auto & entry : dynamic_route_config.decode_schedule) {
        decode_uses_qnn = decode_uses_qnn || llama_dynamic_route_uses_qnn(entry.route.plan);
    }

    if (!llama_context_should_preload_dynamic_qnn_decode_graphs(
                dynamic_route_config.enabled(),
                decode_uses_qnn,
                env_flag_enabled("GGML_HETERO_DYNAMIC_PRELOAD_QNN_DECODE"))) {
        return;
    }

    ggml_backend_t qnn_backend = find_backend_for_route("qnn-npu");
    if (qnn_backend == nullptr) {
        LLAMA_LOG_WARN("%s: skipping QNN decode graph preload because the qnn-npu backend is unavailable\n",
                __func__);
        return;
    }

    ggml_backend_dev_t qnn_dev = ggml_backend_get_device(qnn_backend);
    ggml_backend_reg_t qnn_reg = qnn_dev != nullptr ? ggml_backend_dev_backend_reg(qnn_dev) : nullptr;
    auto * preload_fn =
        qnn_reg != nullptr
            ? (ggml_backend_qnn_aot_preload_decode_graphs_t)
                  ggml_backend_reg_get_proc_address(qnn_reg, "ggml_backend_qnn_aot_preload_decode_graphs")
            : nullptr;
    if (preload_fn == nullptr) {
        LLAMA_LOG_WARN("%s: qnn backend does not expose AoT decode graph preload support\n", __func__);
        return;
    }

    const int64_t t_start_us = ggml_time_us();
    const bool preloaded = preload_fn(qnn_backend, 1);
    const int64_t preload_us = ggml_time_us() - t_start_us;
    if (!preloaded) {
        LLAMA_LOG_WARN("%s: QNN decode graph preload failed preload_us=%" PRId64 "\n",
                __func__,
                preload_us);
        return;
    }

    LLAMA_LOG_INFO("%s: preloaded QNN decode AoT graphs for batch=1 preload_us=%" PRId64 "\n",
            __func__,
            preload_us);
}

void llama_context::validate_dynamic_seq0_token_history() {
    if (qnn_prefix_replay_active) {
        return;
    }

    size_t prefix_tokens = 0;
    const bool prefix_valid = seq0_prefix_tokens_from_memory(memory.get(), prefix_tokens);
    if (!prefix_valid) {
        if (!dynamic_seq0_token_history.empty()) {
            LLAMA_LOG_WARN("%s: clearing tracked seq0 token history because memory is not a physical seq0 prefix\n",
                    __func__);
        }
        dynamic_seq0_token_history.clear();
        return;
    }

    if (prefix_tokens == dynamic_seq0_token_history.size()) {
        return;
    }

    if (!dynamic_seq0_token_history.empty() || prefix_tokens != 0) {
        LLAMA_LOG_WARN("%s: clearing tracked seq0 token history due to state mismatch (tracked=%zu, memory=%zu)\n",
                __func__,
                dynamic_seq0_token_history.size(),
                prefix_tokens);
    }

    dynamic_seq0_token_history.clear();
}

const std::vector<llama_token> & llama_context::get_dynamic_seq0_token_history() const {
    return dynamic_seq0_token_history;
}

void llama_context::set_dynamic_seq0_token_history(const std::vector<llama_token> & tokens) {
    dynamic_seq0_token_history = tokens;
}

void llama_context::clear_dynamic_seq0_token_history() {
    dynamic_seq0_token_history.clear();
}

void llama_context::record_dynamic_seq0_token_history(const llama_batch & batch_inp, size_t prefix_tokens_before_decode) {
    if (qnn_prefix_replay_active) {
        return;
    }

    std::vector<llama_token> new_tokens;
    if (!batch_extract_appendable_seq0_tokens(batch_inp, prefix_tokens_before_decode, new_tokens)) {
        if (!dynamic_seq0_token_history.empty() || prefix_tokens_before_decode != 0) {
            LLAMA_LOG_WARN("%s: disabling seq0 token-history tracking because the batch is not a contiguous seq0 token append\n",
                    __func__);
        }
        dynamic_seq0_token_history.clear();
        return;
    }

    if (dynamic_seq0_token_history.size() != prefix_tokens_before_decode) {
        if (!dynamic_seq0_token_history.empty()) {
            LLAMA_LOG_WARN("%s: dropping seq0 token-history tracking because the tracked prefix no longer matches the context state\n",
                    __func__);
        }
        dynamic_seq0_token_history.clear();
        if (prefix_tokens_before_decode != 0) {
            return;
        }
    }

    dynamic_seq0_token_history.insert(
        dynamic_seq0_token_history.end(),
        new_tokens.begin(),
        new_tokens.end());
}

bool llama_context::sync_dynamic_cpu_opencl_kv(
        bool host_to_device,
        llama_opencl_external_host_sync_timing * timing,
        llama_opencl_external_host_sync_scope sync_scope) {
    if (memory == nullptr) {
        return true;
    }

    ggml_backend_t opencl_backend = nullptr;
    for (const auto & backend : backends) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
        if (dev != nullptr && std::strcmp(ggml_backend_dev_name(dev), "GPUOpenCL") == 0) {
            opencl_backend = backend.get();
            break;
        }
    }

    if (opencl_backend == nullptr) {
        LLAMA_LOG_ERROR("%s: dynamic CPU/OpenCL KV sync requested, but GPUOpenCL backend is unavailable\n", __func__);
        return false;
    }

    std::vector<llama_kv_cache *> kv_caches;
    auto append_kv_cache = [&](llama_kv_cache * kv_cache) {
        if (kv_cache != nullptr) {
            kv_caches.push_back(kv_cache);
        }
    };

    if (auto * kv_cache = dynamic_cast<llama_kv_cache *>(memory.get())) {
        append_kv_cache(kv_cache);
    } else if (auto * kv_cache_iswa = dynamic_cast<llama_kv_cache_iswa *>(memory.get())) {
        append_kv_cache(kv_cache_iswa->get_base());
        append_kv_cache(kv_cache_iswa->get_swa());
    } else if (auto * hybrid_memory = dynamic_cast<llama_memory_hybrid *>(memory.get())) {
        append_kv_cache(hybrid_memory->get_mem_attn());
    } else if (auto * hybrid_iswa_memory = dynamic_cast<llama_memory_hybrid_iswa *>(memory.get())) {
        auto * attn_cache = hybrid_iswa_memory->get_mem_attn();
        append_kv_cache(attn_cache != nullptr ? attn_cache->get_base() : nullptr);
        append_kv_cache(attn_cache != nullptr ? attn_cache->get_swa() : nullptr);
    }

    if (timing != nullptr) {
        timing->clear();
    }

    for (llama_kv_cache * kv_cache : kv_caches) {
        llama_opencl_external_host_sync_timing cache_timing;
        if (!kv_cache->sync_external_opencl_host_aliases(opencl_backend, host_to_device, &cache_timing, sync_scope)) {
            return false;
        }
        if (timing != nullptr) {
            timing->accumulate(cache_timing);
        }
    }

    return true;
}

void llama_context::maybe_debug_dump_powerserve_prefix_before_qnn_switch() {
    const char * dump_dir_env = std::getenv("GGML_DEBUG_DUMP_POWERSERVE_SEED_KV_DIR");
    if (dump_dir_env == nullptr || dump_dir_env[0] == '\0' || memory == nullptr) {
        return;
    }

    uint32_t dump_tokens = 0;
    if (const char * dump_tokens_env = std::getenv("GGML_DEBUG_DUMP_POWERSERVE_SEED_KV_TOKENS");
        dump_tokens_env != nullptr && dump_tokens_env[0] != '\0') {
        char * end = nullptr;
        errno = 0;
        const unsigned long parsed = std::strtoul(dump_tokens_env, &end, 10);
        if (errno == 0 && end != dump_tokens_env && *end == '\0' &&
            parsed <= std::numeric_limits<uint32_t>::max()) {
            dump_tokens = static_cast<uint32_t>(parsed);
        } else {
            LLAMA_LOG_WARN("%s: ignoring invalid GGML_DEBUG_DUMP_POWERSERVE_SEED_KV_TOKENS=%s\n",
                    __func__, dump_tokens_env);
        }
    }

    if (dump_tokens == 0) {
        const llama_pos seq_max = memory->seq_pos_max(0);
        if (seq_max < 0) {
            LLAMA_LOG_WARN("%s: skip prefix dump before qnn switch because seq 0 has no KV state yet\n", __func__);
            return;
        }
        dump_tokens = static_cast<uint32_t>(seq_max + 1);
    }

    std::vector<std::pair<std::string, llama_kv_cache *>> kv_caches;
    auto append_kv_cache = [&](const char * label, llama_kv_cache * kv_cache) {
        if (kv_cache != nullptr) {
            kv_caches.emplace_back(label != nullptr ? label : "kv", kv_cache);
        }
    };

    if (auto * kv_cache = dynamic_cast<llama_kv_cache *>(memory.get())) {
        append_kv_cache("attn", kv_cache);
    } else if (auto * kv_cache_iswa = dynamic_cast<llama_kv_cache_iswa *>(memory.get())) {
        append_kv_cache("attn", kv_cache_iswa->get_base());
        append_kv_cache("swa", kv_cache_iswa->get_swa());
    } else if (auto * hybrid_memory = dynamic_cast<llama_memory_hybrid *>(memory.get())) {
        append_kv_cache("attn", hybrid_memory->get_mem_attn());
    } else if (auto * hybrid_iswa_memory = dynamic_cast<llama_memory_hybrid_iswa *>(memory.get())) {
        auto * attn_cache = hybrid_iswa_memory->get_mem_attn();
        append_kv_cache("attn", attn_cache != nullptr ? attn_cache->get_base() : nullptr);
        append_kv_cache("swa", attn_cache != nullptr ? attn_cache->get_swa() : nullptr);
    }

    if (kv_caches.empty()) {
        LLAMA_LOG_WARN("%s: skip prefix dump before qnn switch because no llama_kv_cache is available\n", __func__);
        return;
    }

    const std::filesystem::path base_dir(dump_dir_env);
    for (size_t i = 0; i < kv_caches.size(); ++i) {
        const auto & [label, kv_cache] = kv_caches[i];
        const std::filesystem::path dump_dir =
            kv_caches.size() == 1 ? base_dir : (base_dir / label);
        LLAMA_LOG_INFO("%s: dumping %u token(s) of %s KV state to %s before non-QNN -> qnn switch\n",
                __func__,
                dump_tokens,
                label.c_str(),
                dump_dir.string().c_str());
        if (!kv_cache->dump_powerserve_seed_kv(dump_dir.string(), dump_tokens)) {
            LLAMA_LOG_WARN("%s: failed to dump %s KV state to %s\n",
                    __func__,
                    label.c_str(),
                    dump_dir.string().c_str());
        }
    }
}

bool llama_context::apply_hetero_plan(llama_hetero_execution_plan plan, bool update_base_plan, const char * source) {
    if (!llama_hetero_kv_contract_can_satisfy(hetero_kv_contract_allocated, plan.attn_kv)) {
        LLAMA_LOG_WARN("%s: rejecting hetero plan update from %s: requested attn KV contract layout=%s transfer=%s producer=%s consumer=%s, but the current context was allocated with layout=%s transfer=%s zero_copy=%s. Rebuild the context or add KV migration for this transition.\n",
                __func__,
                source != nullptr ? source : "unknown",
                llama_hetero_kv_layout_name(plan.attn_kv.layout),
                llama_hetero_kv_transfer_mode_name(plan.attn_kv.transfer),
                plan.attn_kv.producer_backend.empty() ? "<unset>" : plan.attn_kv.producer_backend.c_str(),
                plan.attn_kv.consumer_backend.empty() ? "<unset>" : plan.attn_kv.consumer_backend.c_str(),
                llama_hetero_kv_layout_name(hetero_kv_contract_allocated.layout),
                llama_hetero_kv_transfer_mode_name(hetero_kv_contract_allocated.transfer),
                hetero_kv_contract_allocated.zero_copy ? "true" : "false");
        return false;
    }

    if (llama_hetero_execution_plan_equals(hetero_plan, plan)) {
        if (update_base_plan) {
            hetero_plan_base = plan;
        }
        aot_active_route_requests_qnn = hetero_route_requests_qnn(hetero_plan.route);
        if (hetero_dynamic_trace_timing_detail_enabled()) {
            LLAMA_LOG_INFO("%s: skipped no-op hetero plan update via %s\n",
                    __func__,
                    source != nullptr ? source : "unknown");
        }
        return true;
    }

    hetero_plan = std::move(plan);
    aot_active_route_requests_qnn = hetero_route_requests_qnn(hetero_plan.route);
    if (update_base_plan) {
        hetero_plan_base = hetero_plan;
    }

    const bool source_is_dynamic_candidate =
        source != nullptr &&
        (std::strcmp(source, "prefill") == 0 ||
         std::strcmp(source, "decode") == 0 ||
         std::strcmp(source, "fallback") == 0 ||
         std::strcmp(source, "base") == 0);
    const bool target_plan_pre_reserved =
        source_is_dynamic_candidate &&
        std::any_of(
            hetero_dynamic_pre_reserved_plans.begin(),
            hetero_dynamic_pre_reserved_plans.end(),
            [&](const llama_hetero_execution_plan & candidate) {
                return llama_hetero_execution_plan_equals(candidate, hetero_plan);
            });

    sched_need_reserve = !target_plan_pre_reserved;

    LLAMA_LOG_INFO("%s: updated hetero plan via %s: backend=%s route=%s\n",
            __func__,
            source != nullptr ? source : "unknown",
            llama_hetero_phase_backend_for_route(hetero_plan.route).c_str(),
            llama_hetero_format_route_spec(hetero_plan.route).c_str());

    return true;
}

bool llama_context::set_dynamic_route_config(const llama_dynamic_route_config & config) {
    llama_dynamic_route_runtime_config runtime_config;
    std::string error;
    if (!llama_dynamic_route_build_runtime_config(config, runtime_config, &error)) {
        LLAMA_LOG_WARN("%s: failed to parse dynamic route config: %s\n",
                __func__,
                error.empty() ? "<unknown>" : error.c_str());
        return false;
    }

    if (!ensure_dynamic_route_backends_ready(runtime_config)) {
        return false;
    }
    dynamic_route_config = std::move(runtime_config);
    hetero_dynamic_pre_reserved_plans.clear();

    const std::string prefill_route  = llama_hetero_format_route_spec(dynamic_route_config.prefill.plan.route);
    const std::string decode_route   = llama_hetero_format_route_spec(dynamic_route_config.decode.plan.route);
    const std::string fallback_route = llama_hetero_format_route_spec(dynamic_route_config.fallback.plan.route);
    std::string decode_schedule;
    for (const auto & entry : dynamic_route_config.decode_schedule) {
        if (!decode_schedule.empty()) {
            decode_schedule += ";";
        }
        const std::string route = llama_hetero_format_route_spec(entry.route.plan.route);
        decode_schedule += std::to_string(entry.start_token) + ":" +
            (route.empty() ? std::string("<unset>") : route);
    }

    LLAMA_LOG_INFO("%s: dynamic route mode=%s prefill=%s decode=%s fallback=%s decode_schedule=%s slo_us=%" PRId64 " allow_qnn=%s decode_switch_after=%" PRIu64 "\n",
            __func__,
            llama_dynamic_route_mode_name(dynamic_route_config.mode),
            prefill_route.empty()  ? "<unset>" : prefill_route.c_str(),
            decode_route.empty()   ? "<unset>" : decode_route.c_str(),
            fallback_route.empty() ? "<unset>" : fallback_route.c_str(),
            decode_schedule.empty() ? "<unset>" : decode_schedule.c_str(),
            dynamic_route_config.slo_us,
            dynamic_route_config.allow_qnn ? "true" : "false",
            dynamic_route_config.decode_switch_after);

    return true;
}

std::string llama_context::get_dynamic_route_mode() const {
    return llama_dynamic_route_mode_name(dynamic_route_config.mode);
}

bool llama_context::should_sync_before_dynamic_gpu_freq_switch(uint32_t n_tokens) const {
    if (!dynamic_route_config.enabled() ||
        !dynamic_route_config.decode_gpu_freq_sync_before_apply ||
        n_tokens != 1) {
        return false;
    }

    const uint64_t decode_token_index = dynamic_route_state.decode_calls + 1;
    if (dynamic_route_config.decode_switch_after > 0 &&
        decode_token_index <= dynamic_route_config.decode_switch_after) {
        return false;
    }

    const bool qnn_available =
        backend_available_for_route("qnn-npu") ||
        backend_available_for_route("qnn-gpu") ||
        backend_available_for_route("qnn-cpu");
    const llama_dynamic_route_request request = {
        /*.n_tokens =*/ n_tokens,
        /*.decode_token_index =*/ decode_token_index,
        /*.opencl_backend_available =*/ backend_available_for_route("opencl"),
        /*.qnn_backend_available =*/ qnn_available,
        /*.current_plan =*/ &hetero_plan,
        /*.base_plan =*/ &hetero_plan_base,
        /*.allocated_kv_contract =*/ &hetero_kv_contract_allocated,
    };
    const llama_dynamic_route_decision decision = llama_dynamic_route_decide(dynamic_route_config, request);
    if (decision.should_apply || decision.reason == "decode-switch-wait") {
        return false;
    }

    const uint64_t target_gpu_freq_hz =
        llama_context_effective_decode_gpu_freq_hz(dynamic_route_config, decision);
    return llama_context_should_apply_gpu_freq_switch(
            hetero_plan.route,
            decision.plan.route,
            n_tokens,
            gpu_current_freq_hz,
            target_gpu_freq_hz);
}

void llama_context::maybe_apply_dynamic_route(uint32_t n_tokens) {
    if (n_tokens > 1) {
        dynamic_route_state.prefill_calls++;
    } else {
        dynamic_route_state.decode_calls++;
    }

    if (!dynamic_route_config.enabled()) {
        return;
    }

    const uint64_t decode_token_index = n_tokens > 1 ? 0 : dynamic_route_state.decode_calls;
    const bool decode_schedule_active =
        n_tokens == 1 &&
        !dynamic_route_config.decode_schedule.empty() &&
        decode_token_index > 0;
    const bool decode_switch_after_active =
        n_tokens == 1 &&
        !decode_schedule_active &&
        dynamic_route_config.decode_switch_after > 0 &&
        decode_token_index > 0;
    const bool decode_switch_boundary_reached =
        decode_switch_after_active &&
        decode_token_index > dynamic_route_config.decode_switch_after;

    const bool qnn_available =
        backend_available_for_route("qnn-npu") ||
        backend_available_for_route("qnn-gpu") ||
        backend_available_for_route("qnn-cpu");

    const llama_dynamic_route_request request = {
        /*.n_tokens =*/ n_tokens,
        /*.decode_token_index =*/ decode_token_index,
        /*.opencl_backend_available =*/ backend_available_for_route("opencl"),
        /*.qnn_backend_available =*/ qnn_available,
        /*.current_plan =*/ &hetero_plan,
        /*.base_plan =*/ &hetero_plan_base,
        /*.allocated_kv_contract =*/ &hetero_kv_contract_allocated,
    };

    const bool trace_timing = hetero_dynamic_trace_timing_detail_enabled();
    const int64_t t_decide_start_us = trace_timing ? ggml_time_us() : 0;
    llama_dynamic_route_decision decision = llama_dynamic_route_decide(dynamic_route_config, request);
    const int64_t t_decide_end_us = trace_timing ? ggml_time_us() : 0;
    const bool decode_schedule_boundary_reached =
        decision.decode_schedule_active &&
        decision.decode_schedule_switch_after > 0 &&
        decode_token_index == decision.decode_schedule_start_token;
    const uint64_t active_switch_after_tokens =
        decision.decode_schedule_active
            ? decision.decode_schedule_switch_after
            : dynamic_route_config.decode_switch_after;

    if (trace_timing && hetero_phase_trace.active) {
        hetero_phase_trace.route_decide_us = t_decide_end_us - t_decide_start_us;
        hetero_phase_trace.route_reason = decision.reason;
        hetero_phase_trace.decode_token_index = decode_token_index;
        hetero_phase_trace.switch_after_tokens = active_switch_after_tokens;
        if (decode_switch_boundary_reached || decode_schedule_boundary_reached) {
            hetero_phase_trace.transition_phase = "decode_midway";
        }
    }

    const std::string source_route = llama_hetero_format_route_spec(hetero_plan.route);
    const std::string target_route = llama_hetero_format_route_spec(decision.plan.route);
    std::string decode_qnn_workpoint_storage;
    const char * decode_qnn_workpoint =
        llama_context_effective_decode_qnn_workpoint(decision, decode_qnn_workpoint_storage);
    const uint64_t target_gpu_freq_hz =
        llama_context_effective_decode_gpu_freq_hz(dynamic_route_config, decision);
    const uint64_t target_cpu_freq_khz =
        llama_context_effective_decode_cpu_freq_khz(dynamic_route_config, decision);
    const std::string configured_cpu_affinity_mask =
        llama_context_effective_decode_cpu_affinity_mask(dynamic_route_config, decision);
    const int32_t target_cpu_threads =
        llama_context_effective_decode_cpu_threads(dynamic_route_config, decision);
    const std::string current_phase_backend =
        llama_hetero_canonical_backend(llama_hetero_phase_backend_for_route(hetero_plan.route));
    const std::string target_phase_backend =
        llama_hetero_canonical_backend(llama_hetero_phase_backend_for_route(decision.plan.route));
    const bool schedule_route_switch_pending =
        decision.should_apply && decision.decode_schedule_active;
    if (qnn_htp_current_workpoint.empty()) {
        const char * initial_workpoint = std::getenv("GGML_QNN_HTP_WORKPOINT");
        if (initial_workpoint == nullptr || initial_workpoint[0] == '\0') {
            initial_workpoint = std::getenv("GGML_QNN_HTP_POWER_MODE");
        }
        qnn_htp_current_workpoint = llama_context_canonical_qnn_workpoint(initial_workpoint);
    }

    const bool should_apply_qnn_workpoint_state_only =
        !decision.should_apply &&
        decision.reason != "decode-switch-wait" &&
        llama_context_should_apply_qnn_workpoint_switch(
                hetero_plan.route,
                decision.plan.route,
                n_tokens,
                qnn_htp_current_workpoint,
                decode_qnn_workpoint);
    const bool should_apply_qnn_workpoint_before_route =
        schedule_route_switch_pending &&
        target_phase_backend == "qnn-npu" &&
        decode_qnn_workpoint != nullptr &&
        decode_qnn_workpoint[0] != '\0' &&
        llama_context_canonical_qnn_workpoint(qnn_htp_current_workpoint.c_str()) !=
            llama_context_canonical_qnn_workpoint(decode_qnn_workpoint);

    if (should_apply_qnn_workpoint_state_only || should_apply_qnn_workpoint_before_route) {
        ggml_backend_t qnn_backend = find_backend_for_route("qnn-npu");
        bool applied = false;
        const int64_t t_qnn_workpoint_start_us = trace_timing ? ggml_time_us() : 0;
        if (qnn_backend != nullptr) {
            auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(qnn_backend));
            auto * set_workpoint_fn =
                reg != nullptr
                    ? (ggml_backend_qnn_set_htp_workpoint_t)
                          ggml_backend_reg_get_proc_address(reg, "ggml_backend_qnn_set_htp_workpoint")
                    : nullptr;
            if (set_workpoint_fn != nullptr) {
                applied = set_workpoint_fn(qnn_backend, decode_qnn_workpoint);
            } else {
                LLAMA_LOG_ERROR("%s: qnn backend does not expose ggml_backend_qnn_set_htp_workpoint\n", __func__);
            }
        } else {
            LLAMA_LOG_ERROR("%s: qnn-npu backend unavailable for runtime HTP workpoint switch\n", __func__);
        }
        const int64_t t_qnn_workpoint_end_us = trace_timing ? ggml_time_us() : 0;
        const int64_t qnn_workpoint_apply_us =
            trace_timing ? t_qnn_workpoint_end_us - t_qnn_workpoint_start_us : 0;

        if (trace_timing && hetero_phase_trace.active) {
            if (should_apply_qnn_workpoint_state_only) {
                hetero_phase_trace.route_applied = applied;
                hetero_phase_trace.route_noop = true;
                hetero_phase_trace.route_apply_us = 0;
                hetero_phase_trace.route_label = decision.plan_label.empty() ? "decode" : decision.plan_label;
                hetero_phase_trace.route_reason = applied ? "qnn-workpoint-only" : "qnn-workpoint-apply-failed";
                hetero_phase_trace.source_route = source_route;
                hetero_phase_trace.target_route = target_route;
            }
            hetero_phase_trace.qnn_workpoint_apply_us = qnn_workpoint_apply_us;
        }

        if (applied) {
            qnn_htp_current_workpoint = llama_context_canonical_qnn_workpoint(decode_qnn_workpoint);
            if (should_apply_qnn_workpoint_state_only) {
                dynamic_route_state.route_switches++;
            }
            LLAMA_LOG_INFO("%s: applied qnn HTP workpoint switch to %s\n", __func__, decode_qnn_workpoint);
        } else {
            LLAMA_LOG_ERROR("%s: failed to apply qnn HTP workpoint switch to %s\n", __func__, decode_qnn_workpoint);
        }

        if (trace_timing && hetero_dynamic_trace_timing_detail_enabled()) {
            LLAMA_LOG_INFO("%s: timing phase=%s n_tokens=%u qnn_workpoint_apply=%s target_workpoint=%s "
                    "decide_us=%" PRId64 " qnn_workpoint_apply_us=%" PRId64 " target=%s\n",
                    __func__,
                    hetero_phase_name(n_tokens),
                    n_tokens,
                    applied ? "true" : "false",
                    decode_qnn_workpoint,
                    t_decide_end_us - t_decide_start_us,
                    qnn_workpoint_apply_us,
                    target_route.empty() ? "<default>" : target_route.c_str());
        }
        if (should_apply_qnn_workpoint_state_only || !applied) {
            return;
        }
    }

    if (gpu_current_freq_hz == 0 && target_gpu_freq_hz > 0) {
        uint64_t observed_gpu_freq_hz = 0;
        if (llama_context_read_u64_file(dynamic_route_config.gpu_cur_freq_path, observed_gpu_freq_hz) ||
            llama_context_read_u64_file(dynamic_route_config.gpu_min_freq_path, observed_gpu_freq_hz)) {
            gpu_current_freq_hz = observed_gpu_freq_hz;
        }
    }

    const bool should_apply_gpu_freq_state_only =
        !decision.should_apply &&
        decision.reason != "decode-switch-wait" &&
        llama_context_should_apply_gpu_freq_switch(
                hetero_plan.route,
                decision.plan.route,
                n_tokens,
                gpu_current_freq_hz,
                target_gpu_freq_hz);
    const bool should_apply_gpu_freq_before_route =
        schedule_route_switch_pending &&
        target_phase_backend == "opencl" &&
        target_gpu_freq_hz > 0 &&
        (gpu_current_freq_hz == 0 || gpu_current_freq_hz != target_gpu_freq_hz);

    if (should_apply_gpu_freq_state_only || should_apply_gpu_freq_before_route) {
        const uint64_t target_freq_hz = target_gpu_freq_hz;
        const uint64_t floor_freq_hz =
            gpu_current_freq_hz > 0 ? std::min(gpu_current_freq_hz, target_freq_hz) : target_freq_hz;

        bool applied = false;
        uint64_t actual_freq_hz = 0;
        const int64_t t_gpu_freq_start_us = trace_timing ? ggml_time_us() : 0;

        const bool have_paths =
            !dynamic_route_config.gpu_min_freq_path.empty() &&
            !dynamic_route_config.gpu_max_freq_path.empty();
        const bool wrote =
            have_paths &&
            llama_context_write_u64_file(dynamic_route_config.gpu_min_freq_path, floor_freq_hz) &&
            llama_context_write_u64_file(dynamic_route_config.gpu_max_freq_path, target_freq_hz) &&
            llama_context_write_u64_file(dynamic_route_config.gpu_min_freq_path, target_freq_hz);
        const bool have_actual =
            llama_context_read_u64_file(dynamic_route_config.gpu_cur_freq_path, actual_freq_hz) ||
            llama_context_read_u64_file(dynamic_route_config.gpu_min_freq_path, actual_freq_hz);
        applied = wrote && (!have_actual || actual_freq_hz == target_freq_hz);

        const int64_t t_gpu_freq_end_us = trace_timing ? ggml_time_us() : 0;
        const int64_t gpu_freq_apply_us =
            trace_timing ? t_gpu_freq_end_us - t_gpu_freq_start_us : 0;

        if (trace_timing && hetero_phase_trace.active) {
            if (should_apply_gpu_freq_state_only) {
                hetero_phase_trace.route_applied = applied;
                hetero_phase_trace.route_noop = true;
                hetero_phase_trace.route_apply_us = 0;
                hetero_phase_trace.route_label = decision.plan_label.empty() ? "decode" : decision.plan_label;
                hetero_phase_trace.route_reason = applied ? "gpu-freq-only" : "gpu-freq-apply-failed";
                hetero_phase_trace.source_route = source_route;
                hetero_phase_trace.target_route = target_route;
            }
            hetero_phase_trace.gpu_freq_apply_us = gpu_freq_apply_us;
            hetero_phase_trace.requested_gpu_freq_hz = target_freq_hz;
            hetero_phase_trace.actual_gpu_freq_hz = actual_freq_hz;
        }

        if (applied) {
            gpu_current_freq_hz = target_freq_hz;
            if (should_apply_gpu_freq_state_only) {
                dynamic_route_state.route_switches++;
            }
            LLAMA_LOG_INFO("%s: applied GPU frequency switch to %" PRIu64 " Hz actual=%" PRIu64 "\n",
                    __func__,
                    target_freq_hz,
                    actual_freq_hz);
        } else {
            LLAMA_LOG_ERROR("%s: failed to apply GPU frequency switch to %" PRIu64
                    " Hz have_paths=%s wrote=%s actual=%" PRIu64 "\n",
                    __func__,
                    target_freq_hz,
                    have_paths ? "true" : "false",
                    wrote ? "true" : "false",
                    actual_freq_hz);
        }

        if (trace_timing && hetero_dynamic_trace_timing_detail_enabled()) {
            LLAMA_LOG_INFO("%s: timing phase=%s n_tokens=%u gpu_freq_apply=%s requested_gpu_freq_hz=%" PRIu64
                    " actual_gpu_freq_hz=%" PRIu64 " decide_us=%" PRId64
                    " gpu_freq_pre_sync_us=%" PRId64 " gpu_freq_apply_us=%" PRId64 " target=%s\n",
                    __func__,
                    hetero_phase_name(n_tokens),
                    n_tokens,
                    applied ? "true" : "false",
                    target_freq_hz,
                    actual_freq_hz,
                    t_decide_end_us - t_decide_start_us,
                    hetero_phase_trace.active ? hetero_phase_trace.gpu_freq_pre_sync_us : int64_t(0),
                    gpu_freq_apply_us,
                    target_route.empty() ? "<default>" : target_route.c_str());
        }
        if (trace_timing && hetero_phase_trace.active) {
            hetero_transition_trace_log(
                    (t_decide_end_us - t_decide_start_us) +
                        hetero_phase_trace.gpu_freq_pre_sync_us +
                        gpu_freq_apply_us,
                    0,
                    0,
                    false);
        }
        if (should_apply_gpu_freq_state_only || !applied) {
            return;
        }
    }

    if (cpu_current_freq_khz == 0 && target_cpu_freq_khz > 0) {
        uint64_t observed_cpu_freq_khz = 0;
        if (llama_context_read_u64_file(dynamic_route_config.cpu_cur_freq_path, observed_cpu_freq_khz) ||
            llama_context_read_u64_file(dynamic_route_config.cpu_min_freq_path, observed_cpu_freq_khz)) {
            cpu_current_freq_khz = observed_cpu_freq_khz;
        }
    }

    const bool should_apply_cpu_freq_state_only =
        !decision.should_apply &&
        decision.reason != "decode-switch-wait" &&
        llama_context_should_apply_cpu_freq_switch(
                hetero_plan.route,
                decision.plan.route,
                n_tokens,
                cpu_current_freq_khz,
                target_cpu_freq_khz);
    const bool should_apply_cpu_freq_before_route =
        schedule_route_switch_pending &&
        target_phase_backend == "cpu" &&
        target_cpu_freq_khz > 0 &&
        (cpu_current_freq_khz == 0 || cpu_current_freq_khz != target_cpu_freq_khz);

    if (should_apply_cpu_freq_state_only || should_apply_cpu_freq_before_route) {
        const uint64_t target_freq_khz = target_cpu_freq_khz;
        const uint64_t floor_freq_khz =
            cpu_current_freq_khz > 0 ? std::min(cpu_current_freq_khz, target_freq_khz) : target_freq_khz;

        bool applied = false;
        uint64_t actual_freq_khz = 0;
        uint64_t readback_min_khz = 0;
        uint64_t readback_max_khz = 0;
        const int64_t t_cpu_freq_start_us = trace_timing ? ggml_time_us() : 0;

        const bool have_paths =
            !dynamic_route_config.cpu_min_freq_path.empty() &&
            !dynamic_route_config.cpu_max_freq_path.empty();
        const bool wrote =
            have_paths &&
            llama_context_write_u64_file(dynamic_route_config.cpu_min_freq_path, floor_freq_khz) &&
            llama_context_write_u64_file(dynamic_route_config.cpu_max_freq_path, target_freq_khz) &&
            llama_context_write_u64_file(dynamic_route_config.cpu_min_freq_path, target_freq_khz);
        const bool have_min =
            llama_context_read_u64_file(dynamic_route_config.cpu_min_freq_path, readback_min_khz);
        const bool have_max =
            llama_context_read_u64_file(dynamic_route_config.cpu_max_freq_path, readback_max_khz);
        const bool have_cur =
            llama_context_read_u64_file(dynamic_route_config.cpu_cur_freq_path, actual_freq_khz);
        if (!have_cur && have_min) {
            actual_freq_khz = readback_min_khz;
        }
        applied =
            wrote &&
            have_min &&
            have_max &&
            readback_min_khz == target_freq_khz &&
            readback_max_khz == target_freq_khz;

        const int64_t t_cpu_freq_end_us = trace_timing ? ggml_time_us() : 0;
        const int64_t cpu_freq_apply_us =
            trace_timing ? t_cpu_freq_end_us - t_cpu_freq_start_us : 0;

        if (trace_timing && hetero_phase_trace.active) {
            if (should_apply_cpu_freq_state_only) {
                hetero_phase_trace.route_applied = applied;
                hetero_phase_trace.route_noop = true;
                hetero_phase_trace.route_apply_us = 0;
                hetero_phase_trace.route_label = decision.plan_label.empty() ? "decode" : decision.plan_label;
                hetero_phase_trace.route_reason = applied ? "cpu-freq-only" : "cpu-freq-apply-failed";
                hetero_phase_trace.source_route = source_route;
                hetero_phase_trace.target_route = target_route;
            }
            hetero_phase_trace.cpu_freq_apply_us = cpu_freq_apply_us;
            hetero_phase_trace.requested_cpu_freq_khz = target_freq_khz;
            hetero_phase_trace.actual_cpu_freq_khz = actual_freq_khz;
        }

        if (applied) {
            cpu_current_freq_khz = target_freq_khz;
            if (should_apply_cpu_freq_state_only) {
                dynamic_route_state.route_switches++;
            }
            LLAMA_LOG_INFO("%s: applied CPU frequency switch to %" PRIu64
                    " kHz actual=%" PRIu64 " min=%" PRIu64 " max=%" PRIu64 "\n",
                    __func__,
                    target_freq_khz,
                    actual_freq_khz,
                    readback_min_khz,
                    readback_max_khz);
        } else {
            LLAMA_LOG_ERROR("%s: failed to apply CPU frequency switch to %" PRIu64
                    " kHz have_paths=%s wrote=%s actual=%" PRIu64
                    " min=%" PRIu64 " max=%" PRIu64 "\n",
                    __func__,
                    target_freq_khz,
                    have_paths ? "true" : "false",
                    wrote ? "true" : "false",
                    actual_freq_khz,
                    readback_min_khz,
                    readback_max_khz);
        }

        if (trace_timing && hetero_dynamic_trace_timing_detail_enabled()) {
            LLAMA_LOG_INFO("%s: timing phase=%s n_tokens=%u cpu_freq_apply=%s requested_cpu_freq_khz=%" PRIu64
                    " actual_cpu_freq_khz=%" PRIu64 " decide_us=%" PRId64
                    " cpu_freq_apply_us=%" PRId64 " target=%s\n",
                    __func__,
                    hetero_phase_name(n_tokens),
                    n_tokens,
                    applied ? "true" : "false",
                    target_freq_khz,
                    actual_freq_khz,
                    t_decide_end_us - t_decide_start_us,
                    cpu_freq_apply_us,
                    target_route.empty() ? "<default>" : target_route.c_str());
        }
        if (trace_timing && hetero_phase_trace.active) {
            hetero_transition_trace_log(
                    (t_decide_end_us - t_decide_start_us) + cpu_freq_apply_us,
                    0,
                    0,
                    false);
        }
        if (should_apply_cpu_freq_state_only || !applied) {
            return;
        }
    }

    std::string target_cpu_affinity_mask;
    if (!configured_cpu_affinity_mask.empty()) {
        uint64_t parsed_cpu_mask = 0;
        std::string cpu_mask_error;
        if (llama_context_parse_cpu_mask(
                    configured_cpu_affinity_mask,
                    parsed_cpu_mask,
                    cpu_mask_error)) {
            target_cpu_affinity_mask = llama_context_format_cpu_mask(parsed_cpu_mask);
        } else {
            target_cpu_affinity_mask = configured_cpu_affinity_mask;
        }
    }

    if (cpu_current_affinity_mask.empty() && !target_cpu_affinity_mask.empty()) {
        cpu_current_affinity_mask = llama_context_read_current_cpu_affinity_mask();
    }

    const std::string current_phase_backend_for_cpu_state = current_phase_backend;
    const std::string target_phase_backend_for_cpu_state = target_phase_backend;
    const bool cpu_state_would_change =
        llama_context_should_apply_cpu_state_switch(
                hetero_plan.route,
                decision.plan.route,
                n_tokens,
                cpu_current_affinity_mask,
                target_cpu_affinity_mask.empty() ? nullptr : target_cpu_affinity_mask.c_str(),
                cparams.n_threads,
                target_cpu_threads);
    const bool current_phase_is_cpu_for_cpu_state =
        current_phase_backend_for_cpu_state.empty() || current_phase_backend_for_cpu_state == "cpu";
    const bool should_apply_cpu_state_before_route =
        schedule_route_switch_pending &&
        target_phase_backend_for_cpu_state == "cpu" &&
        (!target_cpu_affinity_mask.empty() || target_cpu_threads > 0) &&
        (!current_phase_is_cpu_for_cpu_state || cpu_state_would_change);
    const bool should_apply_cpu_state_switch =
        decision.reason != "decode-switch-wait" &&
        (cpu_state_would_change || should_apply_cpu_state_before_route);
    const bool cpu_route_metadata_only =
        should_apply_cpu_state_switch &&
        decision.should_apply &&
        (current_phase_backend_for_cpu_state.empty() || current_phase_backend_for_cpu_state == "cpu") &&
        target_phase_backend_for_cpu_state == "cpu";

    const char * fn = __func__;
    const auto apply_cpu_state_switch = [&](bool state_only_trace) -> bool {
        bool affinity_applied = true;
        bool threads_applied = true;
        std::string actual_cpu_affinity_mask = cpu_current_affinity_mask;
        std::string cpu_affinity_error;

        const int64_t t_cpu_affinity_start_us = trace_timing ? ggml_time_us() : 0;
        if (!target_cpu_affinity_mask.empty()) {
            affinity_applied = llama_context_apply_cpu_affinity_mask(
                    target_cpu_affinity_mask,
                    actual_cpu_affinity_mask,
                    cpu_affinity_error);
            if (affinity_applied) {
                cpu_current_affinity_mask = actual_cpu_affinity_mask;
            }
        }
        const int64_t t_cpu_affinity_end_us = trace_timing ? ggml_time_us() : 0;
        int64_t cpu_affinity_apply_us =
            trace_timing ? t_cpu_affinity_end_us - t_cpu_affinity_start_us : 0;

        const int64_t t_cpu_threads_start_us = trace_timing ? ggml_time_us() : 0;
        std::string cpu_threads_error;
        if (target_cpu_threads > 0) {
            const int32_t target_threads = target_cpu_threads;
            if (cparams.n_threads != target_threads) {
                struct ggml_threadpool_params tpp = ggml_threadpool_params_default(target_threads);
                tpp.strict_cpu = env_flag_enabled("GGML_HETERO_DYNAMIC_DECODE_CPU_STRICT");
                if (llama_context_cpu_mask_to_threadpool_cpumask(
                            target_cpu_affinity_mask,
                            tpp.cpumask,
                            cpu_threads_error)) {
                    ggml_threadpool_t new_threadpool = ggml_threadpool_new(&tpp);
                    if (new_threadpool != nullptr) {
                        ggml_threadpool_t old_owned_threadpool = owned_dynamic_decode_threadpool;

                        threadpool = new_threadpool;
                        if (backend_cpu != nullptr) {
                            auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
                            auto * set_threadpool_fn =
                                (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(
                                        reg,
                                        "ggml_backend_cpu_set_threadpool");
                            if (set_threadpool_fn) {
                                set_threadpool_fn(backend_cpu, new_threadpool);
                            }

                            auto * set_n_threads_fn =
                                (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(
                                        reg,
                                        "ggml_backend_set_n_threads");
                            if (set_n_threads_fn) {
                                set_n_threads_fn(backend_cpu, target_threads);
                            }
                        }

                        set_n_threads(target_threads, cparams.n_threads_batch);

                        if (!target_cpu_affinity_mask.empty()) {
                            std::string post_threadpool_affinity_mask;
                            std::string post_threadpool_affinity_error;
                            const int64_t t_cpu_affinity_reapply_start_us =
                                trace_timing ? ggml_time_us() : 0;
                            const bool post_threadpool_affinity_applied =
                                llama_context_apply_cpu_affinity_mask(
                                        target_cpu_affinity_mask,
                                        post_threadpool_affinity_mask,
                                        post_threadpool_affinity_error);
                            const int64_t t_cpu_affinity_reapply_end_us =
                                trace_timing ? ggml_time_us() : 0;
                            if (trace_timing) {
                                cpu_affinity_apply_us +=
                                    t_cpu_affinity_reapply_end_us - t_cpu_affinity_reapply_start_us;
                            }
                            if (post_threadpool_affinity_applied) {
                                actual_cpu_affinity_mask = post_threadpool_affinity_mask;
                                cpu_current_affinity_mask = actual_cpu_affinity_mask;
                            } else {
                                affinity_applied = false;
                                cpu_affinity_error =
                                    post_threadpool_affinity_error.empty()
                                        ? "post-threadpool affinity apply failed"
                                        : post_threadpool_affinity_error;
                            }
                        }

                        owned_dynamic_decode_threadpool = new_threadpool;
                        if (old_owned_threadpool != nullptr && old_owned_threadpool != new_threadpool) {
                            ggml_threadpool_free(old_owned_threadpool);
                        }

                        threads_applied =
                            cparams.n_threads == target_threads;
                    } else {
                        threads_applied = false;
                        cpu_threads_error = "ggml_threadpool_new failed";
                    }
                } else {
                    threads_applied = false;
                }
            } else {
                threads_applied = true;
            }
        }
        const int64_t t_cpu_threads_end_us = trace_timing ? ggml_time_us() : 0;
        const int64_t cpu_threads_apply_us =
            trace_timing ? t_cpu_threads_end_us - t_cpu_threads_start_us : 0;

        const bool applied = affinity_applied && threads_applied;
        if (trace_timing && hetero_phase_trace.active) {
            if (state_only_trace) {
                hetero_phase_trace.route_applied = applied;
                hetero_phase_trace.route_noop = true;
                hetero_phase_trace.route_apply_us = 0;
                hetero_phase_trace.route_label = decision.plan_label.empty() ? "decode" : decision.plan_label;
                hetero_phase_trace.route_reason = applied ? "cpu-state-only" : "cpu-state-apply-failed";
                hetero_phase_trace.source_route = source_route;
                hetero_phase_trace.target_route = target_route;
            }
            hetero_phase_trace.cpu_affinity_apply_us = cpu_affinity_apply_us;
            hetero_phase_trace.cpu_threads_apply_us = cpu_threads_apply_us;
            hetero_phase_trace.requested_cpu_affinity_mask = target_cpu_affinity_mask;
            hetero_phase_trace.actual_cpu_affinity_mask = actual_cpu_affinity_mask;
            hetero_phase_trace.requested_cpu_threads = target_cpu_threads;
            hetero_phase_trace.actual_cpu_threads = cparams.n_threads;
        }

        if (applied) {
            if (state_only_trace || cpu_route_metadata_only) {
                dynamic_route_state.route_switches++;
            }
            LLAMA_LOG_INFO("%s: applied CPU state switch affinity=%s actual_affinity=%s threads=%d\n",
                    fn,
                    target_cpu_affinity_mask.empty() ? "<unchanged>" : target_cpu_affinity_mask.c_str(),
                    actual_cpu_affinity_mask.empty() ? "<unknown>" : actual_cpu_affinity_mask.c_str(),
                    cparams.n_threads);
        } else {
            LLAMA_LOG_ERROR("%s: failed to apply CPU state switch affinity=%s actual_affinity=%s "
                    "threads=%d target_threads=%d affinity_error=%s threads_error=%s\n",
                    fn,
                    target_cpu_affinity_mask.empty() ? "<unchanged>" : target_cpu_affinity_mask.c_str(),
                    actual_cpu_affinity_mask.empty() ? "<unknown>" : actual_cpu_affinity_mask.c_str(),
                    cparams.n_threads,
                    target_cpu_threads,
                    cpu_affinity_error.empty() ? "<none>" : cpu_affinity_error.c_str(),
                    cpu_threads_error.empty() ? "<none>" : cpu_threads_error.c_str());
        }

        if (trace_timing && hetero_dynamic_trace_timing_detail_enabled()) {
            LLAMA_LOG_INFO("%s: timing phase=%s n_tokens=%u cpu_state_apply=%s "
                    "requested_cpu_affinity_mask=%s actual_cpu_affinity_mask=%s "
                    "requested_cpu_threads=%d actual_cpu_threads=%d decide_us=%" PRId64
                    " cpu_affinity_apply_us=%" PRId64 " cpu_threads_apply_us=%" PRId64
                    " target=%s\n",
                    fn,
                    hetero_phase_name(n_tokens),
                    n_tokens,
                    applied ? "true" : "false",
                    target_cpu_affinity_mask.empty() ? "<unchanged>" : target_cpu_affinity_mask.c_str(),
                    actual_cpu_affinity_mask.empty() ? "<unknown>" : actual_cpu_affinity_mask.c_str(),
                    target_cpu_threads,
                    cparams.n_threads,
                    t_decide_end_us - t_decide_start_us,
                    cpu_affinity_apply_us,
                    cpu_threads_apply_us,
                    target_route.empty() ? "<default>" : target_route.c_str());
        }
        return applied;
    };

    if (should_apply_cpu_state_switch &&
            (!decision.should_apply || !env_flag_enabled("GGML_HETERO_DYNAMIC_CPU_STATE_AFTER_ROUTE"))) {
        const bool cpu_state_applied = apply_cpu_state_switch(!decision.should_apply);
        if (trace_timing && hetero_phase_trace.active) {
            if (!decision.should_apply) {
                hetero_transition_trace_log(
                        (t_decide_end_us - t_decide_start_us) +
                            hetero_phase_trace.cpu_affinity_apply_us +
                            hetero_phase_trace.cpu_threads_apply_us,
                        0,
                        0,
                        false);
            }
        }
        if (!decision.should_apply || !cpu_state_applied) {
            return;
        }
        if (cpu_route_metadata_only) {
            hetero_plan = std::move(decision.plan);
            aot_active_route_requests_qnn = hetero_route_requests_qnn(hetero_plan.route);
            if (trace_timing && hetero_phase_trace.active) {
                hetero_phase_trace.route_applied = true;
                hetero_phase_trace.route_noop = true;
                hetero_phase_trace.route_apply_us = 0;
                hetero_phase_trace.route_label = decision.plan_label.empty() ? "decode" : decision.plan_label;
                hetero_phase_trace.route_reason = "cpu-route-metadata-only";
                hetero_phase_trace.source_route = source_route;
                hetero_phase_trace.target_route = target_route;
            }
            if (dynamic_route_config.trace_enabled) {
                LLAMA_LOG_INFO("%s: dynamic CPU route metadata-only update target=%s\n",
                        __func__,
                        target_route.empty() ? "<default>" : target_route.c_str());
            }
            if (trace_timing && hetero_dynamic_trace_timing_detail_enabled()) {
                LLAMA_LOG_INFO("%s: timing phase=%s n_tokens=%u route_apply=true label=%s reason=cpu-route-metadata-only decide_us=%" PRId64 " apply_us=0 target=%s\n",
                        __func__,
                        hetero_phase_name(n_tokens),
                        n_tokens,
                        decision.plan_label.empty() ? "<none>" : decision.plan_label.c_str(),
                        t_decide_end_us - t_decide_start_us,
                        target_route.empty() ? "<default>" : target_route.c_str());
            }
            return;
        }
    }

    if (!decision.should_apply) {
        if (dynamic_route_config.trace_enabled) {
            LLAMA_LOG_INFO("%s: dynamic route skip phase=%s reason=%s\n",
                    __func__,
                    n_tokens > 1 ? "prefill" : "decode",
                    decision.reason.empty() ? "<none>" : decision.reason.c_str());
        }
        if (trace_timing && hetero_dynamic_trace_timing_detail_enabled()) {
            LLAMA_LOG_INFO("%s: timing phase=%s n_tokens=%u route_apply=false reason=%s decide_us=%" PRId64 "\n",
                    __func__,
                    hetero_phase_name(n_tokens),
                    n_tokens,
                    decision.reason.empty() ? "<none>" : decision.reason.c_str(),
                    t_decide_end_us - t_decide_start_us);
        }
        return;
    }

    const std::string current_attn_backend =
        llama_hetero_canonical_backend(hetero_plan.route.backend_for(llama_hetero_route_stage::ATTN_CORE));
    const std::string target_attn_backend =
        llama_hetero_canonical_backend(decision.plan.route.backend_for(llama_hetero_route_stage::ATTN_CORE));
    const bool qnn_aot_enabled = std::getenv("GGML_QNN_AOT_CONFIG") != nullptr;
    const bool switching_into_qnn_decode =
        n_tokens == 1 &&
        !hetero_route_requests_qnn(hetero_plan.route) &&
        hetero_route_requests_qnn(decision.plan.route);
    const bool switching_out_of_qnn_decode =
        n_tokens == 1 &&
        hetero_route_requests_qnn(hetero_plan.route) &&
        !hetero_route_requests_qnn(decision.plan.route);
    const char * generic_qnn_kv_env = std::getenv("GGML_QNN_AOT_WRITE_GENERIC_KV");
    const bool generic_qnn_kv_enabled =
        generic_qnn_kv_env != nullptr &&
        generic_qnn_kv_env[0] != '\0' &&
        std::strcmp(generic_qnn_kv_env, "0") != 0;
    const bool should_attempt_qnn_kv_migration =
        llama_context_should_attempt_qnn_phase_kv_migration(
                current_attn_backend,
                target_attn_backend,
                n_tokens,
                generic_qnn_kv_enabled);
    const bool should_prepare_qnn_direct_generic_kv_import =
        switching_into_qnn_decode &&
        should_attempt_qnn_kv_migration &&
        !llama_hetero_is_qnn_backend(current_attn_backend) &&
        llama_hetero_is_qnn_backend(target_attn_backend);
    const bool should_use_qnn_shared_phase_kv =
        llama_context_should_use_qnn_shared_phase_kv(
                current_attn_backend,
                target_attn_backend,
                n_tokens,
                generic_qnn_kv_enabled,
                hetero_kv_contract_allocated);
    const char * direct_qnn_opencl_host_ptr_env = std::getenv("GGML_OPENCL_EXPERIMENTAL_QNN_DIRECT_HOST_PTR");
    const bool direct_qnn_opencl_host_ptr_enabled =
        direct_qnn_opencl_host_ptr_env != nullptr &&
        direct_qnn_opencl_host_ptr_env[0] != '\0' &&
        std::strcmp(direct_qnn_opencl_host_ptr_env, "0") != 0;
    const bool should_try_qnn_opencl_direct_host_ptr_visibility =
        llama_context_should_try_qnn_opencl_direct_host_ptr_visibility(
                current_attn_backend,
                target_attn_backend,
                n_tokens,
                generic_qnn_kv_enabled,
                hetero_kv_contract_allocated,
                direct_qnn_opencl_host_ptr_enabled);
    const bool should_flush_pending_qnn_kv = switching_out_of_qnn_decode;
    const bool should_migrate_cpu_opencl_kv =
        n_tokens == 1 &&
        ((current_attn_backend == "cpu" && target_attn_backend == "opencl") ||
         (current_attn_backend == "opencl" && target_attn_backend == "cpu"));
    const llama_hetero_execution_plan previous_plan = hetero_plan;
    bool migrated_qnn_kv = false;
    bool prepared_qnn_direct_generic_kv_import = false;
    bool qnn_generic_kv_writeback_ready = !switching_out_of_qnn_decode;
    bool qnn_generic_kv_writeback_flushed = false;

    if (switching_into_qnn_decode) {
        size_t prefix_tokens = 0;
        if (!seq0_prefix_tokens_from_memory(memory.get(), prefix_tokens)) {
            LLAMA_LOG_WARN("%s: refusing non-QNN -> qnn decode switch because memory is not a physical seq0 prefix\n",
                    __func__);
            return;
        }
        if (prefix_tokens > 0 && dynamic_seq0_token_history.size() != prefix_tokens) {
            LLAMA_LOG_WARN("%s: refusing non-QNN -> qnn decode switch because seq0 token history is unavailable (tracked=%zu, memory=%zu)\n",
                    __func__,
                    dynamic_seq0_token_history.size(),
                    prefix_tokens);
            return;
        }
    }

    if (switching_into_qnn_decode) {
        maybe_debug_dump_powerserve_prefix_before_qnn_switch();
    }

    if (should_flush_pending_qnn_kv) {
        qnn_generic_kv_writeback_ready = false;
        ggml_backend_t qnn_backend = nullptr;
        for (const auto & backend : backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
            if (dev != nullptr && std::strcmp(ggml_backend_dev_name(dev), "qnn-npu") == 0) {
                qnn_backend = backend.get();
                break;
            }
        }

        if (qnn_backend != nullptr) {
            auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(qnn_backend));
            auto * has_pending_fn =
                (ggml_backend_qnn_aot_has_pending_generic_kv_writeback_t)
                    ggml_backend_reg_get_proc_address(reg, "ggml_backend_qnn_aot_has_pending_generic_kv_writeback");
            auto * flush_pending_fn =
                (ggml_backend_qnn_aot_flush_pending_generic_kv_writeback_t)
                    ggml_backend_reg_get_proc_address(reg, "ggml_backend_qnn_aot_flush_pending_generic_kv_writeback");

            if (has_pending_fn != nullptr && flush_pending_fn != nullptr && has_pending_fn(qnn_backend)) {
                LLAMA_LOG_INFO("%s: starting KV migration after prefill before decode route switch\n", __func__);
                const int64_t t_kv_migration_start_us = trace_timing ? ggml_time_us() : 0;
                const bool flushed = flush_pending_fn(qnn_backend);
                const int64_t t_kv_migration_end_us = trace_timing ? ggml_time_us() : 0;
                if (trace_timing && hetero_phase_trace.active) {
                    hetero_phase_trace.kv_migration_us += t_kv_migration_end_us - t_kv_migration_start_us;
                }
                if (!flushed) {
                    LLAMA_LOG_ERROR("%s: deferred QNN KV migration failed; keeping existing route and skipping decode backend switch\n",
                            __func__);
                    return;
                }
                qnn_generic_kv_writeback_ready = true;
                qnn_generic_kv_writeback_flushed = true;
            } else if (has_pending_fn != nullptr && flush_pending_fn != nullptr) {
                qnn_generic_kv_writeback_ready = true;
            } else if (generic_qnn_kv_enabled) {
                LLAMA_LOG_WARN("%s: qnn backend does not expose generic KV writeback flush hooks; keeping explicit state migration for %s -> %s\n",
                        __func__,
                        current_attn_backend.c_str(),
                        target_attn_backend.c_str());
            }
        } else if (generic_qnn_kv_enabled) {
            LLAMA_LOG_WARN("%s: qnn backend unavailable for generic KV writeback flush; keeping explicit state migration for %s -> %s\n",
                    __func__,
                    current_attn_backend.c_str(),
                    target_attn_backend.c_str());
        }
    }

    if (should_migrate_cpu_opencl_kv) {
        bool migrated = false;
        const bool try_cpu_opencl_uma_kv =
            llama_context_should_try_cpu_opencl_uma_kv_handoff(
                    current_attn_backend,
                    target_attn_backend,
                    n_tokens,
                    env_flag_enabled("GGML_HETERO_DISABLE_CPU_OPENCL_UMA_KV_HANDOFF"),
                    env_flag_enabled("GGML_HETERO_ENABLE_OPENCL_CPU_UMA_KV_HANDOFF"));

        if (try_cpu_opencl_uma_kv) {
            LLAMA_LOG_INFO("%s: trying CPU/OpenCL UMA KV handoff before decode route switch (%s -> %s)\n",
                    __func__,
                    current_attn_backend.c_str(),
                    target_attn_backend.c_str());
            const int64_t t_kv_sync_start_us = trace_timing ? ggml_time_us() : 0;
            llama_opencl_external_host_sync_timing opencl_sync_timing;
            const bool host_to_device = current_attn_backend == "cpu" && target_attn_backend == "opencl";
            const bool synced = sync_dynamic_cpu_opencl_kv(
                    host_to_device,
                    &opencl_sync_timing,
                    llama_opencl_external_host_sync_scope::ACTIVE_KV_PREFIX);
            const int64_t t_kv_sync_end_us = trace_timing ? ggml_time_us() : 0;
            migrated = synced && opencl_sync_timing.synced_buffers > 0;
            if (trace_timing && hetero_phase_trace.active) {
                hetero_phase_trace.kv_migration_us += t_kv_sync_end_us - t_kv_sync_start_us;
                hetero_phase_trace.kv_alias_us += opencl_sync_timing.alias_us;
                hetero_phase_trace.kv_backend_sync_us += opencl_sync_timing.backend_sync_us;
                hetero_phase_trace.kv_transfer_us += opencl_sync_timing.transfer_us;
            }
            if (migrated) {
                LLAMA_LOG_INFO("%s: completed CPU/OpenCL UMA KV handoff using %zu host-visible KV buffer(s), %zu range(s), total %.2f MiB alias_us=%" PRId64 " backend_sync_us=%" PRId64 " transfer_us=%" PRId64 "\n",
                        __func__,
                        opencl_sync_timing.synced_buffers,
                        opencl_sync_timing.synced_ranges,
                        opencl_sync_timing.synced_bytes / 1024.0 / 1024.0,
                        opencl_sync_timing.alias_us,
                        opencl_sync_timing.backend_sync_us,
                        opencl_sync_timing.transfer_us);
            } else if (synced) {
                LLAMA_LOG_WARN("%s: CPU/OpenCL UMA KV handoff found no host-visible KV buffers; falling back to state rebuild\n",
                        __func__);
            } else {
                LLAMA_LOG_WARN("%s: CPU/OpenCL UMA KV handoff sync failed; falling back to state rebuild\n",
                        __func__);
            }
        }

        if (!migrated) {
            LLAMA_LOG_INFO("%s: starting CPU/OpenCL KV migration before decode route switch (%s -> %s)\n",
                    __func__,
                    current_attn_backend.c_str(),
                    target_attn_backend.c_str());
            const int64_t t_kv_sync_start_us = trace_timing ? ggml_time_us() : 0;
            migrated = migrate_dynamic_cpu_opencl_kv(current_attn_backend, target_attn_backend);
            const int64_t t_kv_sync_end_us = trace_timing ? ggml_time_us() : 0;
            if (trace_timing && hetero_phase_trace.active) {
                hetero_phase_trace.kv_migration_us += t_kv_sync_end_us - t_kv_sync_start_us;
            }
            if (!migrated) {
                LLAMA_LOG_ERROR("%s: CPU/OpenCL KV migration failed; keeping existing route and skipping backend switch\n",
                        __func__);
                return;
            }
        }
    }

    if (llama_context_should_use_qnn_written_generic_kv_for_cpu(
                current_attn_backend,
                target_attn_backend,
                n_tokens,
                generic_qnn_kv_enabled,
                qnn_generic_kv_writeback_ready,
                llama_context_live_kv_is_cpu_accessible(memory.get()),
                qnn_generic_kv_writeback_flushed)) {
        migrated_qnn_kv = true;
        LLAMA_LOG_INFO("%s: reusing QNN-written live generic KV directly for decode route switch (%s -> %s); state rebuild skipped\n",
                __func__,
                current_attn_backend.c_str(),
                target_attn_backend.c_str());
    }

    if (should_prepare_qnn_direct_generic_kv_import) {
        LLAMA_LOG_INFO("%s: preparing direct generic KV import before decode route switch (%s -> %s)\n",
                __func__,
                current_attn_backend.c_str(),
                target_attn_backend.c_str());

        const int64_t t_kv_sync_start_us = trace_timing ? ggml_time_us() : 0;
        bool prepared = true;
        llama_opencl_external_host_sync_timing opencl_sync_timing;

        if (llama_context_should_sync_opencl_before_qnn_direct_import(
                    current_attn_backend,
                    target_attn_backend,
                    n_tokens,
                    generic_qnn_kv_enabled)) {
            ggml_backend_t opencl_backend = find_backend_for_route("opencl");
            if (opencl_backend == nullptr) {
                LLAMA_LOG_WARN("%s: OpenCL backend unavailable before QNN direct generic KV import\n",
                        __func__);
                prepared = false;
            } else {
                ggml_backend_synchronize(opencl_backend);
                LLAMA_LOG_INFO("%s: synchronized OpenCL backend before QNN direct generic KV import\n",
                        __func__);
            }
        }

        if (prepared && current_attn_backend == "opencl") {
            const auto sync_scope = llama_context_opencl_sync_scope_for_qnn_direct_import(
                    current_attn_backend,
                    target_attn_backend,
                    n_tokens,
                    generic_qnn_kv_enabled);
            prepared = sync_dynamic_cpu_opencl_kv(
                    /* host_to_device = */ false,
                    &opencl_sync_timing,
                    sync_scope);
            if (!prepared) {
                LLAMA_LOG_WARN("%s: failed to synchronize OpenCL KV back to host before QNN direct import\n",
                        __func__);
            } else {
                LLAMA_LOG_INFO("%s: synchronized OpenCL KV back to host before QNN direct import using %zu range(s), total %.2f MiB alias_us=%" PRId64 " backend_sync_us=%" PRId64 " transfer_us=%" PRId64 "\n",
                        __func__,
                        opencl_sync_timing.synced_ranges,
                        opencl_sync_timing.synced_bytes / 1024.0 / 1024.0,
                        opencl_sync_timing.alias_us,
                        opencl_sync_timing.backend_sync_us,
                        opencl_sync_timing.transfer_us);
            }
        }

        ggml_backend_t qnn_backend = prepared ? find_backend_for_route(target_attn_backend) : nullptr;
        if (prepared && qnn_backend == nullptr) {
            LLAMA_LOG_WARN("%s: qnn backend unavailable for direct generic KV import preparation\n", __func__);
            prepared = false;
        }

        ggml_backend_qnn_aot_reset_state_t reset_state_fn = nullptr;
        if (prepared) {
            ggml_backend_dev_t qnn_dev = ggml_backend_get_device(qnn_backend);
            ggml_backend_reg_t qnn_reg = qnn_dev != nullptr ? ggml_backend_dev_backend_reg(qnn_dev) : nullptr;
            reset_state_fn =
                qnn_reg != nullptr
                    ? (ggml_backend_qnn_aot_reset_state_t)
                          ggml_backend_reg_get_proc_address(qnn_reg, "ggml_backend_qnn_aot_reset_state")
                    : nullptr;
            if (reset_state_fn == nullptr) {
                LLAMA_LOG_WARN("%s: qnn backend does not expose AoT reset_state support for direct generic KV import\n",
                        __func__);
                prepared = false;
            }
        }

        if (prepared && !reset_state_fn(qnn_backend)) {
            LLAMA_LOG_WARN("%s: failed to reset QNN AoT state before direct generic KV import\n", __func__);
            prepared = false;
        }

        const int64_t t_kv_sync_end_us = trace_timing ? ggml_time_us() : 0;
        if (trace_timing && hetero_phase_trace.active) {
            hetero_phase_trace.kv_migration_us += t_kv_sync_end_us - t_kv_sync_start_us;
            hetero_phase_trace.kv_alias_us += opencl_sync_timing.alias_us;
            hetero_phase_trace.kv_backend_sync_us += opencl_sync_timing.backend_sync_us;
            hetero_phase_trace.kv_transfer_us += opencl_sync_timing.transfer_us;
        }

        if (prepared) {
            prepared_qnn_direct_generic_kv_import = true;
            migrated_qnn_kv = true;
            LLAMA_LOG_INFO("%s: prepared direct generic KV import for %zu prefix token(s) before non-QNN -> qnn decode switch\n",
                    __func__,
                    dynamic_seq0_token_history.size());
        } else {
            LLAMA_LOG_WARN("%s: direct generic KV import preparation failed; falling back to QNN prefix replay for %s -> %s\n",
                    __func__,
                    current_attn_backend.c_str(),
                    target_attn_backend.c_str());
        }
    }

    if (should_use_qnn_shared_phase_kv) {
        LLAMA_LOG_INFO("%s: reusing shared QNN KV directly for decode route switch (%s -> %s)%s\n",
                __func__,
                current_attn_backend.c_str(),
                target_attn_backend.c_str(),
                should_try_qnn_opencl_direct_host_ptr_visibility
                    ? " with experimental OpenCL direct-host-ptr visibility"
                    : "");
        const int64_t t_kv_sync_start_us = trace_timing ? ggml_time_us() : 0;
        llama_opencl_external_host_sync_timing opencl_sync_timing;
        migrated_qnn_kv =
            target_attn_backend == "opencl"
                ? sync_dynamic_cpu_opencl_kv(
                        /* host_to_device = */ true,
                        &opencl_sync_timing,
                        llama_context_opencl_sync_scope_for_qnn_shared_phase_kv(
                                current_attn_backend,
                                target_attn_backend,
                                n_tokens,
                                generic_qnn_kv_enabled,
                                hetero_kv_contract_allocated))
                : true;
        const int64_t t_kv_sync_end_us = trace_timing ? ggml_time_us() : 0;
        if (trace_timing && hetero_phase_trace.active) {
            hetero_phase_trace.kv_migration_us += t_kv_sync_end_us - t_kv_sync_start_us;
            hetero_phase_trace.kv_alias_us += opencl_sync_timing.alias_us;
            hetero_phase_trace.kv_backend_sync_us += opencl_sync_timing.backend_sync_us;
            hetero_phase_trace.kv_transfer_us += opencl_sync_timing.transfer_us;
        }
        if (!migrated_qnn_kv) {
            LLAMA_LOG_WARN("%s: direct shared QNN KV handoff failed; falling back to state rebuild for %s -> %s\n",
                    __func__,
                    current_attn_backend.c_str(),
                    target_attn_backend.c_str());
        } else if (target_attn_backend == "opencl") {
            LLAMA_LOG_INFO("%s: completed direct shared QNN/OpenCL KV handoff using %zu host-visible KV buffer(s), %zu range(s), total %.2f MiB alias_us=%" PRId64 " backend_sync_us=%" PRId64 " transfer_us=%" PRId64 "\n",
                    __func__,
                    opencl_sync_timing.synced_buffers,
                    opencl_sync_timing.synced_ranges,
                    opencl_sync_timing.synced_bytes / 1024.0 / 1024.0,
                    opencl_sync_timing.alias_us,
                    opencl_sync_timing.backend_sync_us,
                    opencl_sync_timing.transfer_us);
        }
    }

    if (!migrated_qnn_kv &&
        llama_context_should_try_qnn_written_generic_kv_for_opencl(
                current_attn_backend,
                target_attn_backend,
                n_tokens,
                generic_qnn_kv_enabled,
                qnn_generic_kv_writeback_ready)) {
        LLAMA_LOG_INFO("%s: trying QNN-written generic KV handoff before decode route switch (%s -> %s)\n",
                __func__,
                current_attn_backend.c_str(),
                target_attn_backend.c_str());
        const int64_t t_kv_sync_start_us = trace_timing ? ggml_time_us() : 0;
        llama_opencl_external_host_sync_timing opencl_sync_timing;
        const bool synced = sync_dynamic_cpu_opencl_kv(
                /* host_to_device = */ true,
                &opencl_sync_timing,
                llama_opencl_external_host_sync_scope::ACTIVE_KV_PREFIX);
        const int64_t t_kv_sync_end_us = trace_timing ? ggml_time_us() : 0;
        migrated_qnn_kv = synced && opencl_sync_timing.synced_buffers > 0;
        if (trace_timing && hetero_phase_trace.active) {
            hetero_phase_trace.kv_migration_us += t_kv_sync_end_us - t_kv_sync_start_us;
            hetero_phase_trace.kv_alias_us += opencl_sync_timing.alias_us;
            hetero_phase_trace.kv_backend_sync_us += opencl_sync_timing.backend_sync_us;
            hetero_phase_trace.kv_transfer_us += opencl_sync_timing.transfer_us;
        }
        if (migrated_qnn_kv) {
            LLAMA_LOG_INFO("%s: completed QNN-written generic KV handoff to OpenCL using %zu host-visible KV buffer(s), %zu range(s), total %.2f MiB alias_us=%" PRId64 " backend_sync_us=%" PRId64 " transfer_us=%" PRId64 "\n",
                    __func__,
                    opencl_sync_timing.synced_buffers,
                    opencl_sync_timing.synced_ranges,
                    opencl_sync_timing.synced_bytes / 1024.0 / 1024.0,
                    opencl_sync_timing.alias_us,
                    opencl_sync_timing.backend_sync_us,
                    opencl_sync_timing.transfer_us);
        } else if (synced) {
            LLAMA_LOG_WARN("%s: QNN-written generic KV handoff to OpenCL found no host-visible KV buffers; falling back to state rebuild\n",
                    __func__);
        } else {
            LLAMA_LOG_WARN("%s: QNN-written generic KV handoff to OpenCL sync failed; falling back to state rebuild\n",
                    __func__);
        }
    }

    if (should_attempt_qnn_kv_migration &&
        !should_prepare_qnn_direct_generic_kv_import &&
        !migrated_qnn_kv) {
        LLAMA_LOG_INFO("%s: starting QNN KV migration before decode route switch (%s -> %s)\n",
                __func__,
                current_attn_backend.c_str(),
                target_attn_backend.c_str());
        const int64_t t_kv_sync_start_us = trace_timing ? ggml_time_us() : 0;
        migrated_qnn_kv = rebuild_dynamic_consumer_kv_from_state(
                current_attn_backend,
                target_attn_backend,
                "qnn-phase-state-migration");
        const int64_t t_kv_sync_end_us = trace_timing ? ggml_time_us() : 0;
        if (trace_timing && hetero_phase_trace.active) {
            hetero_phase_trace.kv_migration_us += t_kv_sync_end_us - t_kv_sync_start_us;
        }
        if (!migrated_qnn_kv) {
            LLAMA_LOG_WARN("%s: direct QNN KV migration failed; falling back to prefix replay for %s -> %s\n",
                    __func__,
                    current_attn_backend.c_str(),
                    target_attn_backend.c_str());
        }
    }

    const int64_t t_apply_start_us = trace_timing ? ggml_time_us() : 0;
    const bool applied = apply_hetero_plan(std::move(decision.plan), /* update_base_plan = */ false, decision.plan_label.c_str());
    const int64_t t_apply_end_us = trace_timing ? ggml_time_us() : 0;

    if (applied && switching_into_qnn_decode) {
        aot_skip_bootstrap_for_next_decode = true;
        qnn_prefix_replay_pending = false;
        qnn_prefix_replay_restore_plan_valid = false;
        qnn_prefix_replay_rebuild_live_memory = false;
        if (prepared_qnn_direct_generic_kv_import) {
            LLAMA_LOG_INFO("%s: using direct generic KV import for non-QNN -> qnn decode switch; prefix replay skipped\n",
                    __func__);
        } else if (!dynamic_seq0_token_history.empty()) {
            qnn_prefix_replay_restore_plan = previous_plan;
            qnn_prefix_replay_restore_plan_valid = true;
            qnn_prefix_replay_pending = true;
            LLAMA_LOG_INFO("%s: queued QNN prefix replay for %zu token(s) after non-QNN -> qnn decode switch\n",
                    __func__,
                    dynamic_seq0_token_history.size());
        }
    } else if (applied &&
               switching_out_of_qnn_decode &&
               qnn_aot_enabled &&
               !target_attn_backend.empty() &&
               !migrated_qnn_kv) {
        qnn_prefix_replay_pending = false;
        qnn_prefix_replay_restore_plan_valid = false;
        qnn_prefix_replay_rebuild_live_memory = false;
        if (!dynamic_seq0_token_history.empty()) {
            qnn_prefix_replay_restore_plan = previous_plan;
            qnn_prefix_replay_restore_plan_valid = true;
            qnn_prefix_replay_pending = true;
            qnn_prefix_replay_rebuild_live_memory = true;
            LLAMA_LOG_INFO("%s: queued %s prefix replay for %zu token(s) after qnn -> %s decode switch\n",
                    __func__,
                    target_attn_backend.c_str(),
                    dynamic_seq0_token_history.size(),
                    target_attn_backend.c_str());
        }
    }

    if (trace_timing && hetero_phase_trace.active) {
        hetero_phase_trace.route_applied = applied;
        hetero_phase_trace.route_noop = !sched_need_reserve;
        hetero_phase_trace.route_apply_us = t_apply_end_us - t_apply_start_us;
        hetero_phase_trace.route_label = decision.plan_label;
        hetero_phase_trace.source_route = source_route;
        hetero_phase_trace.target_route = target_route;
    }

    if (applied) {
        dynamic_route_state.route_switches++;
    }

    if (trace_timing && hetero_dynamic_trace_timing_detail_enabled()) {
        LLAMA_LOG_INFO("%s: timing phase=%s n_tokens=%u route_apply=%s label=%s reason=%s decide_us=%" PRId64 " apply_us=%" PRId64 " target=%s\n",
                __func__,
                hetero_phase_name(n_tokens),
                n_tokens,
                applied ? "true" : "false",
                decision.plan_label.empty() ? "<none>" : decision.plan_label.c_str(),
                decision.reason.empty() ? "<none>" : decision.reason.c_str(),
                t_decide_end_us - t_decide_start_us,
                t_apply_end_us - t_apply_start_us,
                target_route.empty() ? "<default>" : target_route.c_str());
    }
}

llama_context::~llama_context() {
    hetero_decode_token_trace_dump();

    if (owned_dynamic_decode_threadpool != nullptr) {
        if (backend_cpu != nullptr) {
            auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
            auto * set_threadpool_fn =
                (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(
                        reg,
                        "ggml_backend_cpu_set_threadpool");
            if (set_threadpool_fn) {
                set_threadpool_fn(backend_cpu, nullptr);
            }
        }
        if (threadpool == owned_dynamic_decode_threadpool) {
            threadpool = nullptr;
        }
        if (threadpool_batch == owned_dynamic_decode_threadpool) {
            threadpool_batch = nullptr;
        }
        ggml_threadpool_free(owned_dynamic_decode_threadpool);
        owned_dynamic_decode_threadpool = nullptr;
    }

    if (!model.hparams.no_alloc) {
        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            ggml_backend_t             backend = backend_ptrs[i];
            ggml_backend_buffer_type_t buft    = backend_buft[i];

            const size_t size_exp = backend_buf_exp_size[i];
            const size_t size_act = ggml_backend_sched_get_buffer_size(sched.get(), backend);
            if (size_exp == size_act) {
                LLAMA_LOG_DEBUG("%s: %10s compute buffer size is %8.4f MiB, matches expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            } else {
                LLAMA_LOG_WARN("%s: %10s compute buffer size of %8.4f MiB, does not match expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            }
        }
    }
    ggml_opt_free(opt_ctx);
}

void llama_context::hetero_decode_token_trace_record(int64_t done_us) {
    hetero_decode_token_trace_records.push_back(done_us);
}

void llama_context::hetero_decode_token_trace_dump() {
    if (hetero_decode_token_trace_records.empty()) {
        return;
    }

    std::vector<int64_t> tbt_us;
    tbt_us.reserve(hetero_decode_token_trace_records.size() > 1 ? hetero_decode_token_trace_records.size() - 1 : 0);

    int64_t prev_done_us = 0;
    int64_t tbt_sum_us = 0;
    for (size_t i = 0; i < hetero_decode_token_trace_records.size(); ++i) {
        const int64_t done_us = hetero_decode_token_trace_records[i];
        LLAMA_LOG_INFO("DECODE_TOKEN_TRACE phase=decode token_index=%zu done_us=%" PRId64 "\n",
                i + 1,
                done_us);

        if (prev_done_us > 0 && done_us >= prev_done_us) {
            const int64_t delta_us = done_us - prev_done_us;
            tbt_us.push_back(delta_us);
            tbt_sum_us += delta_us;
        }
        prev_done_us = done_us;
    }

    if (!tbt_us.empty()) {
        const double mean_us = double(tbt_sum_us) / double(tbt_us.size());
        LLAMA_LOG_INFO("DECODE_TOKEN_TBT_SUMMARY count=%zu mean_us=%.2f p50_us=%" PRId64 " p95_us=%" PRId64 " p99_us=%" PRId64 "\n",
                tbt_us.size(),
                mean_us,
                percentile_nearest_rank_us(tbt_us, 0.50),
                percentile_nearest_rank_us(tbt_us, 0.95),
                percentile_nearest_rank_us(tbt_us, 0.99));
    }

    hetero_decode_token_trace_records.clear();
}

void llama_context::hetero_transition_trace_log(
        int64_t total_us,
        int64_t process_ubatch_us,
        int64_t sync_done_us,
        bool    include_first_token_gap) {
    if (!hetero_phase_trace.route_applied ||
        hetero_phase_trace.n_tokens != 1 ||
        hetero_phase_trace.transition_trace_emitted) {
        return;
    }

    const int64_t transition_blocking_us =
        llama_context_transition_blocking_us(total_us, process_ubatch_us);
    const int64_t first_token_gap_us =
        include_first_token_gap
            ? llama_context_decode_token_gap_us(hetero_last_decode_token_done_us, sync_done_us)
            : -1;
    const std::string first_token_gap =
        first_token_gap_us >= 0 ? std::to_string(first_token_gap_us) : "";
    const char * transition_phase =
        hetero_phase_trace.transition_phase.empty()
            ? "prefill_to_decode"
            : hetero_phase_trace.transition_phase.c_str();

    LLAMA_LOG_INFO("TRANSITION_TRACE phase=%s "
            "decision_us=%" PRId64 " route_apply_us=%" PRId64 " "
            "policy_apply_us= qnn_workpoint_apply_us=%" PRId64 " gpu_freq_pre_sync_us=%" PRId64 " gpu_freq_apply_us=%" PRId64 " "
            "cpu_freq_apply_us=%" PRId64 " cpu_affinity_apply_us=%" PRId64 " cpu_threads_apply_us=%" PRId64 " "
            "sched_reserve_us=%" PRId64 " kv_handoff_us=%" PRId64 " "
            "graph_rebuild_us=%" PRId64 " decode_entry_us=%" PRId64 " "
            "total_blocking_us=%" PRId64 " first_token_gap_us=%s post_switch_tbt_us=%s "
            "transition_energy_mj= transition_energy_source=unavailable "
            "success=1 fallback=0 support_status=ok "
            "decode_token_index=%" PRIu64 " switch_after_tokens=%" PRIu64 " "
            "source_route=%s target_route=%s "
            "requested_gpu_freq_hz=%" PRIu64 " actual_gpu_freq_hz=%" PRIu64 " "
            "requested_cpu_freq_khz=%" PRIu64 " actual_cpu_freq_khz=%" PRIu64 " "
            "requested_cpu_affinity_mask=%s actual_cpu_affinity_mask=%s "
            "requested_cpu_threads=%d actual_cpu_threads=%d "
            "process_ubatch_us=%" PRId64 " total_wall_us=%" PRId64 " "
            "graph_runs_reused=%d graph_runs_rebuilt=%d\n",
            transition_phase,
            hetero_phase_trace.route_decide_us,
            hetero_phase_trace.route_apply_us,
            hetero_phase_trace.qnn_workpoint_apply_us,
            hetero_phase_trace.gpu_freq_pre_sync_us,
            hetero_phase_trace.gpu_freq_apply_us,
            hetero_phase_trace.cpu_freq_apply_us,
            hetero_phase_trace.cpu_affinity_apply_us,
            hetero_phase_trace.cpu_threads_apply_us,
            hetero_phase_trace.reserve_us,
            hetero_phase_trace.kv_migration_us,
            hetero_phase_trace.bootstrap_sched_rebuild_us,
            total_us,
            transition_blocking_us,
            first_token_gap.c_str(),
            first_token_gap.c_str(),
            hetero_phase_trace.decode_token_index,
            hetero_phase_trace.switch_after_tokens,
            hetero_phase_trace.source_route.empty() ? "<default>" : hetero_phase_trace.source_route.c_str(),
            hetero_phase_trace.target_route.empty() ? "<default>" : hetero_phase_trace.target_route.c_str(),
            hetero_phase_trace.requested_gpu_freq_hz,
            hetero_phase_trace.actual_gpu_freq_hz,
            hetero_phase_trace.requested_cpu_freq_khz,
            hetero_phase_trace.actual_cpu_freq_khz,
            hetero_phase_trace.requested_cpu_affinity_mask.c_str(),
            hetero_phase_trace.actual_cpu_affinity_mask.c_str(),
            hetero_phase_trace.requested_cpu_threads,
            hetero_phase_trace.actual_cpu_threads,
            process_ubatch_us,
            total_us,
            hetero_phase_trace.graph_runs_reused,
            hetero_phase_trace.graph_runs_rebuilt);

    hetero_phase_trace.transition_trace_emitted = true;
}

void llama_context::sched_reserve() {
    const uint32_t reserve_request_tokens = sched_reserve_request_tokens;
    sched_reserve_request_tokens = 0;

    if (!sched_need_reserve) {
        return;
    }

    sched_need_reserve = false;
    hetero_dynamic_pre_reserved_plans.clear();

    const int64_t pending_batch_compute_start_us =
        (n_queued_tokens == 0) ? t_compute_start_us : 0;

    LLAMA_LOG_INFO("%s: reserving ...\n", __func__);

    hetero_phase_trace_suppress_sync_log = true;
    synchronize();
    hetero_phase_trace_suppress_sync_log = false;

    if (pending_batch_compute_start_us != 0 && t_compute_start_us == 0 && n_queued_tokens == 0) {
        // sched_reserve() synchronizes before the current batch is formally queued.
        // Preserve its start timestamp so subsequent perf accounting measures the
        // batch that triggered the reserve instead of time since process start.
        t_compute_start_us = pending_batch_compute_start_us;
    }

    const int64_t t_start_us = ggml_time_us();
    llama_sched_reserve_timing reserve_timing;

    const uint32_t n_seqs = cparams.n_seq_max;
    const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

    const int64_t t_sched_new_start_us = ggml_time_us();
    const size_t max_nodes = this->graph_max_nodes(n_tokens);

    LLAMA_LOG_DEBUG("%s: max_nodes = %zu\n", __func__, max_nodes);

    gf_res_prev.reset(new llm_graph_result(max_nodes));
    gf_res_reserve.reset(new llm_graph_result(max_nodes));

    sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, cparams.pipeline_parallel, cparams.op_offload));
    reserve_timing.sched_new_us += ggml_time_us() - t_sched_new_start_us;

    llama_memory_context_ptr mctx;
    const int64_t t_memory_init_start_us = ggml_time_us();
    if (memory) {
        LLAMA_LOG_DEBUG("%s: reserving full memory module\n", __func__);
        mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory module");
        }
    }
    reserve_timing.memory_init_us += ggml_time_us() - t_memory_init_start_us;

    // avoid reserving graphs with zero outputs - assume one output per sequence
    const int n_outputs = n_seqs;

    LLAMA_LOG_DEBUG("%s: worst-case: n_tokens = %d, n_seqs = %d, n_outputs = %d\n", __func__, n_tokens, n_seqs, n_outputs);

    const int64_t t_feature_probe_start_us = ggml_time_us();
    // resolve automatic Flash Attention use
    if (cparams.auto_fa) {
        auto * gf = graph_reserve(1, n_seqs, n_outputs, mctx.get(), true);
        if (!gf) {
            throw std::runtime_error("failed to reserve graph for Flash Attention check");
        }

        const size_t prefix_len = strlen(LLAMA_TENSOR_NAME_FATTN) + 1;
        bool fa_device_mismatch = false;
        for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
            ggml_tensor * n = ggml_graph_node(gf, i);
            if (n->op != GGML_OP_FLASH_ATTN_EXT) {
                continue;
            }
            ggml_backend_dev_t device_fa = ggml_backend_get_device(ggml_backend_sched_get_tensor_backend(sched.get(), n));

            // TODO: instead of the tensor names, use a map to keep track of which (FA) tensors belong to which layer
            GGML_ASSERT(strncmp(n->name, LLAMA_TENSOR_NAME_FATTN "-", prefix_len) == 0);
            const int il = std::stoi(n->name + prefix_len);
            ggml_backend_dev_t device_kv = model.dev_layer(il);
            if (device_fa != device_kv) {
                LLAMA_LOG_WARN("%s: layer %d is assigned to device %s but the Flash Attention tensor "
                        "is assigned to device %s (usually due to missing support)\n",
                        __func__, il, ggml_backend_dev_name(device_kv), ggml_backend_dev_name(device_fa));
                // FIXME: fa_device_mismatch logic is wrong for --no-kv-offload, but this is broken anyways
                fa_device_mismatch = true;
                break;
            }
        }

        if (fa_device_mismatch) {
            cparams.flash_attn = false;
            LLAMA_LOG_WARN("%s: Flash Attention was auto, set to disabled\n", __func__);
        } else {
            cparams.flash_attn = true;
            LLAMA_LOG_INFO("%s: Flash Attention was auto, set to enabled\n", __func__);
        }

        cparams.auto_fa = false;
    }

    if (cparams.auto_fgdn) {
        LLAMA_LOG_INFO("%s: resolving fused Gated Delta Net support:\n", __func__);

        if (cparams.fused_gdn_ar) {
            auto * gf = graph_reserve(1, n_seqs, n_outputs, mctx.get(), true);
            if (!gf) {
                throw std::runtime_error("failed to reserve graph for fused Gated Delta Net check (autoregressive)");
            }

            const size_t prefix_len = strlen(LLAMA_TENSOR_NAME_FGDN_AR) + 1;
            bool gdn_device_mismatch = false;
            for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
                ggml_tensor * n = ggml_graph_node(gf, i);
                if (n->op != GGML_OP_GATED_DELTA_NET) {
                    continue;
                }
                ggml_backend_dev_t device_gdn = ggml_backend_get_device(ggml_backend_sched_get_tensor_backend(sched.get(), n));

                GGML_ASSERT(strncmp(n->name, LLAMA_TENSOR_NAME_FGDN_AR "-", prefix_len) == 0);
                const int il = std::stoi(n->name + prefix_len);
                ggml_backend_dev_t device_kv = model.dev_layer(il);
                if (device_gdn != device_kv) {
                    LLAMA_LOG_WARN("%s: layer %d is assigned to device %s but the fused Gated Delta Net tensor "
                            "is assigned to device %s (usually due to missing support)\n",
                            __func__, il, ggml_backend_dev_name(device_kv), ggml_backend_dev_name(device_gdn));
                    gdn_device_mismatch = true;
                    break;
                }
            }

            if (gdn_device_mismatch) {
                cparams.fused_gdn_ar = false;
                LLAMA_LOG_WARN("%s: fused Gated Delta Net (autoregressive) not supported, set to disabled\n", __func__);
            } else {
                LLAMA_LOG_INFO("%s: fused Gated Delta Net (autoregressive) enabled\n", __func__);
            }
        }

        if (cparams.fused_gdn_ch) {
            // more than one token in the batch per sequence in order to take the chunked path
            // note: n_outputs must match n_tokens for embedding models with mean/rank pooling,
            // because build_pooling creates inp_mean with shape [n_tokens, n_seqs] and multiplies
            // it with t_embd which is reduced to [n_outputs, ...] via out_ids. if n_outputs != n_tokens,
            // the ggml_mul_mat assertion fails. this matches the pp reservation below (line ~553).
            const uint32_t n_tokens_ch = 16*n_seqs;
            auto * gf = graph_reserve(n_tokens_ch, n_seqs, n_tokens_ch, mctx.get(), true);
            if (!gf) {
                throw std::runtime_error("failed to reserve graph for fused Gated Delta Net check (chunked)");
            }

            const size_t prefix_len = strlen(LLAMA_TENSOR_NAME_FGDN_CH) + 1;
            bool gdn_device_mismatch = false;
            for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
                ggml_tensor * n = ggml_graph_node(gf, i);
                if (n->op != GGML_OP_GATED_DELTA_NET) {
                    continue;
                }
                ggml_backend_dev_t device_gdn = ggml_backend_get_device(ggml_backend_sched_get_tensor_backend(sched.get(), n));

                GGML_ASSERT(strncmp(n->name, LLAMA_TENSOR_NAME_FGDN_CH "-", prefix_len) == 0);
                const int il = std::stoi(n->name + prefix_len);
                ggml_backend_dev_t device_kv = model.dev_layer(il);
                if (device_gdn != device_kv) {
                    LLAMA_LOG_WARN("%s: layer %d is assigned to device %s but the fused Gated Delta Net tensor "
                            "is assigned to device %s (usually due to missing support)\n",
                            __func__, il, ggml_backend_dev_name(device_kv), ggml_backend_dev_name(device_gdn));
                    gdn_device_mismatch = true;
                    break;
                }
            }

            if (gdn_device_mismatch) {
                cparams.fused_gdn_ch = false;
                LLAMA_LOG_WARN("%s: fused Gated Delta Net (chunked) not supported, set to disabled\n", __func__);
            } else {
                LLAMA_LOG_INFO("%s: fused Gated Delta Net (chunked) enabled\n", __func__);
            }
        }

        cparams.auto_fgdn = false;
    }
    reserve_timing.feature_probe_us += ggml_time_us() - t_feature_probe_start_us;

    // reserve worst-case graph
    int n_splits_pp = -1;
    int n_nodes_pp  = -1;

    int n_splits_tg = -1;
    int n_nodes_tg  = -1;

    const bool use_decode_tg_only_reserve =
        llama_context_should_use_dynamic_decode_tg_only_sched_reserve(
                dynamic_route_config.enabled(),
                reserve_request_tokens,
                env_flag_enabled("GGML_HETERO_DYNAMIC_DECODE_TG_ONLY_RESERVE"));

    const auto reserve_plan_buffers = [&](const llama_hetero_execution_plan & plan,
                                          bool capture_stats,
                                          bool decode_tg_only) {
        const auto saved_plan = hetero_plan;
        const bool saved_qnn_active = aot_active_route_requests_qnn;
        hetero_plan = plan;
        aot_active_route_requests_qnn = hetero_route_requests_qnn(hetero_plan.route);

        if (!decode_tg_only) {
            // reserve pp (prompt processing) graph first so that buffers are only allocated once
            {
                auto * gf = graph_reserve(n_tokens, n_seqs, n_tokens, mctx.get(),
                        model.hparams.no_alloc, model.hparams.no_alloc ? backend_buf_exp_size.data() : nullptr);
                if (!gf) {
                    if (cparams.pipeline_parallel) {
                        LLAMA_LOG_WARN("%s: compute buffer allocation failed, retrying without pipeline parallelism\n", __func__);
                        cparams.pipeline_parallel = false;
                        sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, false, cparams.op_offload));
                        gf = graph_reserve(n_tokens, n_seqs, n_tokens, mctx.get());
                    }
                    if (!gf) {
                        hetero_plan = saved_plan;
                        aot_active_route_requests_qnn = saved_qnn_active;
                        throw std::runtime_error("failed to allocate compute pp buffers");
                    }
                }

                if (capture_stats) {
                    n_splits_pp = ggml_backend_sched_get_n_splits(sched.get());
                    n_nodes_pp  = ggml_graph_n_nodes(gf);
                }
            }
        }

        // reserve with tg (token generation) graph to get the number of splits and nodes
        {
            auto * gf = graph_reserve(n_seqs, n_seqs, n_seqs, mctx.get(), model.hparams.no_alloc);
            if (!gf) {
                hetero_plan = saved_plan;
                aot_active_route_requests_qnn = saved_qnn_active;
                throw std::runtime_error("failed to allocate compute tg buffers");
            }

            if (capture_stats) {
                n_splits_tg = ggml_backend_sched_get_n_splits(sched.get());
                n_nodes_tg  = ggml_graph_n_nodes(gf);
            }
        }

        if (decode_tg_only) {
            if (capture_stats) {
                n_splits_pp = n_splits_tg;
                n_nodes_pp  = n_nodes_tg;
            }
        } else {
            // reserve again with pp graph to avoid ggml-alloc reallocations during inference
            {
                auto * gf = graph_reserve(n_tokens, n_seqs, n_tokens, mctx.get(), model.hparams.no_alloc);
                if (!gf) {
                    hetero_plan = saved_plan;
                    aot_active_route_requests_qnn = saved_qnn_active;
                    throw std::runtime_error("failed to allocate compute pp buffers");
                }
                (void) gf;
            }
        }

        hetero_plan = saved_plan;
        aot_active_route_requests_qnn = saved_qnn_active;
    };

    const int64_t t_plan_reserve_start_us = ggml_time_us();
    if (use_decode_tg_only_reserve) {
        LLAMA_LOG_INFO("%s: using experimental decode tg-only reserve for current dynamic phase switch\n", __func__);
    }
    reserve_plan_buffers(hetero_plan, /* capture_stats = */ true, /* decode_tg_only = */ use_decode_tg_only_reserve);

    if (dynamic_route_config.enabled() && env_flag_enabled("GGML_HETERO_DYNAMIC_PRERESERVE")) {
        const bool include_fallback = env_flag_enabled("GGML_HETERO_DYNAMIC_PRERESERVE_FALLBACK");
        std::vector<llama_hetero_execution_plan> reserve_candidates;
        const auto plan_is_pre_reserved = [&](const llama_hetero_execution_plan & plan) {
            return std::any_of(
                hetero_dynamic_pre_reserved_plans.begin(),
                hetero_dynamic_pre_reserved_plans.end(),
                [&](const llama_hetero_execution_plan & candidate) {
                    return llama_hetero_execution_plan_equals(candidate, plan);
                });
        };
        const auto append_reserved_plan = [&](const llama_hetero_execution_plan & plan) {
            if (!plan.route.has_any_route()) {
                return;
            }
            if (plan_is_pre_reserved(plan)) {
                return;
            }
            hetero_dynamic_pre_reserved_plans.push_back(plan);
        };
        const auto append_unique_plan = [&](const llama_hetero_execution_plan & plan) {
            if (!plan.route.has_any_route()) {
                return;
            }
            if (plan_is_pre_reserved(plan)) {
                return;
            }
            for (const auto & existing : reserve_candidates) {
                if (llama_hetero_execution_plan_equals(existing, plan)) {
                    return;
                }
            }
            reserve_candidates.push_back(plan);
        };

        append_reserved_plan(hetero_plan);
        if (dynamic_route_config.prefill.configured) {
            append_unique_plan(dynamic_route_config.prefill.plan);
        }
        if (dynamic_route_config.decode.configured) {
            append_unique_plan(dynamic_route_config.decode.plan);
        }
        for (const auto & entry : dynamic_route_config.decode_schedule) {
            append_unique_plan(entry.route.plan);
        }
        if (include_fallback && dynamic_route_config.fallback.configured) {
            append_unique_plan(dynamic_route_config.fallback.plan);
        }

        for (const auto & plan : reserve_candidates) {
            if (llama_hetero_execution_plan_equals(plan, hetero_plan)) {
                continue;
            }
            reserve_plan_buffers(plan, /* capture_stats = */ false, /* decode_tg_only = */ false);
            append_reserved_plan(plan);
        }

        LLAMA_LOG_INFO("%s: pre-reserved %zu dynamic hot plans%s\n",
                __func__,
                hetero_dynamic_pre_reserved_plans.size(),
                include_fallback ? " (including fallback)" : "");
    }
    reserve_timing.plan_reserve_us += ggml_time_us() - t_plan_reserve_start_us;

    const int64_t t_finalize_start_us = ggml_time_us();
    for (size_t i = 0; i < backend_ptrs.size(); ++i) {
        ggml_backend_t             backend = backend_ptrs[i];
        ggml_backend_buffer_type_t buft    = backend_buft[i];
        if (!model.hparams.no_alloc) {
            backend_buf_exp_size[i] = ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
        if (backend_buf_exp_size[i] > 1) {
            LLAMA_LOG_INFO("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buft_name(buft),
                    backend_buf_exp_size[i] / 1024.0 / 1024.0);
        }
    }

    if (n_nodes_pp == n_nodes_tg) {
        LLAMA_LOG_INFO("%s: graph nodes  = %d\n", __func__, n_nodes_pp);
    } else {
        LLAMA_LOG_INFO("%s: graph nodes  = %d (with bs=%d), %d (with bs=1)\n", __func__, n_nodes_pp, n_tokens, n_nodes_tg);
    }

    if (n_splits_pp == n_splits_tg) {
        LLAMA_LOG_INFO("%s: graph splits = %d\n", __func__, n_splits_pp);
    } else {
        LLAMA_LOG_INFO("%s: graph splits = %d (with bs=%d), %d (with bs=1)\n", __func__, n_splits_pp, n_tokens, n_splits_tg);
    }
    reserve_timing.finalize_us += ggml_time_us() - t_finalize_start_us;

    const int64_t t_end_us = ggml_time_us();
    const int64_t reserve_us = t_end_us - t_start_us;

    if (hetero_dynamic_trace_timing_detail_enabled() && hetero_phase_trace.active) {
        hetero_phase_trace.reserve_us += reserve_us;
        hetero_phase_trace.reserve_sched_new_us += reserve_timing.sched_new_us;
        hetero_phase_trace.reserve_memory_init_us += reserve_timing.memory_init_us;
        hetero_phase_trace.reserve_feature_probe_us += reserve_timing.feature_probe_us;
        hetero_phase_trace.reserve_plan_reserve_us += reserve_timing.plan_reserve_us;
        hetero_phase_trace.reserve_finalize_us += reserve_timing.finalize_us;
    }

    LLAMA_LOG_INFO("%s: reserve took %.2f ms, sched copies = %d\n",
            __func__, reserve_us/1000.0, ggml_backend_sched_get_n_copies(sched.get()));
}

void llama_context::synchronize() {
    if (!sched) {
        return;
    }

    ggml_backend_sched_synchronize(sched.get());

    int64_t sync_done_us = 0;
    const auto get_sync_done_us = [&]() {
        if (sync_done_us == 0) {
            sync_done_us = ggml_time_us();
        }
        return sync_done_us;
    };

    if (!hetero_phase_trace_suppress_sync_log &&
        hetero_dynamic_trace_timing_enabled() &&
        n_queued_tokens == 1) {
        hetero_decode_token_trace_record(get_sync_done_us());
    }

    if (!hetero_phase_trace_suppress_sync_log &&
        hetero_dynamic_trace_timing_detail_enabled() &&
        hetero_phase_trace.active &&
        hetero_phase_trace.batch_start_us > 0) {
        const int64_t detail_sync_done_us = get_sync_done_us();
        const int64_t total_us = detail_sync_done_us - hetero_phase_trace.batch_start_us;
        const int64_t reserve_accounted_us =
            hetero_phase_trace.reserve_sched_new_us +
            hetero_phase_trace.reserve_memory_init_us +
            hetero_phase_trace.reserve_feature_probe_us +
            hetero_phase_trace.reserve_plan_reserve_us +
            hetero_phase_trace.reserve_finalize_us;
        const int64_t reserve_unattributed_us =
            std::max<int64_t>(int64_t(0), hetero_phase_trace.reserve_us - reserve_accounted_us);
        const int64_t kv_accounted_us =
            hetero_phase_trace.kv_alias_us +
            hetero_phase_trace.kv_backend_sync_us +
            hetero_phase_trace.kv_transfer_us;
        const int64_t kv_unattributed_us =
            std::max<int64_t>(int64_t(0), hetero_phase_trace.kv_migration_us - kv_accounted_us);
        LLAMA_LOG_INFO("%s: timing phase=%s n_tokens=%u total_wall_us=%" PRId64 " decide_us=%" PRId64 " apply_us=%" PRId64 " qnn_workpoint_apply_us=%" PRId64 " gpu_freq_pre_sync_us=%" PRId64 " gpu_freq_apply_us=%" PRId64 " cpu_freq_apply_us=%" PRId64 " cpu_affinity_apply_us=%" PRId64 " cpu_threads_apply_us=%" PRId64 " reserve_us=%" PRId64 " memory_update_us=%" PRId64 " kv_migration_us=%" PRId64 " process_ubatch_us=%" PRId64 " bootstrap_sync_us=%" PRId64 " bootstrap_sched_rebuild_us=%" PRId64 " ubatches=%d graph_runs_reused=%d graph_runs_rebuilt=%d route_applied=%s route_noop=%s bootstrap_ran=%s label=%s reason=%s target=%s\n",
                __func__,
                hetero_phase_name(hetero_phase_trace.n_tokens),
                hetero_phase_trace.n_tokens,
                total_us,
                hetero_phase_trace.route_decide_us,
                hetero_phase_trace.route_apply_us,
                hetero_phase_trace.qnn_workpoint_apply_us,
                hetero_phase_trace.gpu_freq_pre_sync_us,
                hetero_phase_trace.gpu_freq_apply_us,
                hetero_phase_trace.cpu_freq_apply_us,
                hetero_phase_trace.cpu_affinity_apply_us,
                hetero_phase_trace.cpu_threads_apply_us,
                hetero_phase_trace.reserve_us,
                hetero_phase_trace.memory_update_us,
                hetero_phase_trace.kv_migration_us,
                hetero_phase_trace.process_ubatch_us,
                hetero_phase_trace.bootstrap_sync_us,
                hetero_phase_trace.bootstrap_sched_rebuild_us,
                hetero_phase_trace.process_ubatches,
                hetero_phase_trace.graph_runs_reused,
                hetero_phase_trace.graph_runs_rebuilt,
                hetero_phase_trace.route_applied ? "true" : "false",
                hetero_phase_trace.route_noop ? "true" : "false",
                hetero_phase_trace.bootstrap_ran ? "true" : "false",
                hetero_phase_trace.route_label.empty() ? "<none>" : hetero_phase_trace.route_label.c_str(),
                hetero_phase_trace.route_reason.empty() ? "<none>" : hetero_phase_trace.route_reason.c_str(),
                hetero_phase_trace.target_route.empty() ? "<default>" : hetero_phase_trace.target_route.c_str());
        LLAMA_LOG_INFO("%s: timing reserve_breakdown sched_new_us=%" PRId64 " memory_init_us=%" PRId64 " feature_probe_us=%" PRId64 " plan_reserve_us=%" PRId64 " finalize_us=%" PRId64 " unattributed_us=%" PRId64 "\n",
                __func__,
                hetero_phase_trace.reserve_sched_new_us,
                hetero_phase_trace.reserve_memory_init_us,
                hetero_phase_trace.reserve_feature_probe_us,
                hetero_phase_trace.reserve_plan_reserve_us,
                hetero_phase_trace.reserve_finalize_us,
                reserve_unattributed_us);
        LLAMA_LOG_INFO("%s: timing kv_breakdown alias_us=%" PRId64 " backend_sync_us=%" PRId64 " transfer_us=%" PRId64 " unattributed_us=%" PRId64 "\n",
                __func__,
                hetero_phase_trace.kv_alias_us,
                hetero_phase_trace.kv_backend_sync_us,
                hetero_phase_trace.kv_transfer_us,
                kv_unattributed_us);
        hetero_transition_trace_log(
                total_us,
                hetero_phase_trace.process_ubatch_us,
                detail_sync_done_us,
                true);
        if (hetero_phase_trace.n_tokens == 1) {
            hetero_last_decode_token_done_us = detail_sync_done_us;
        } else {
            hetero_last_decode_token_done_us = 0;
        }
        hetero_phase_trace.reset();
    }

    // FIXME: if multiple single tokens are evaluated without a synchronization,
    // the stats will be added to the prompt evaluation stats
    // this should only happen when using batch size 1 to evaluate a batch

    // add the evaluation to the stats
    if (n_queued_tokens == 1) {
        if (!cparams.no_perf) {
            t_eval_us += get_sync_done_us() - t_compute_start_us;
        }
        n_eval++;
    } else if (n_queued_tokens > 1) {
        if (!cparams.no_perf) {
            t_p_eval_us += get_sync_done_us() - t_compute_start_us;
        }
        n_p_eval += n_queued_tokens;
    }

    // get a more accurate load time, upon first eval
    if (n_queued_tokens > 0 && !has_evaluated_once) {
        t_load_us = ggml_time_us() - t_start_us;
        has_evaluated_once = true;
    }

    n_queued_tokens = 0;
    t_compute_start_us = 0;

    if (aot_bootstrap_cpu_sched_active && aot_saved_sched) {
        LLAMA_LOG_DEBUG("%s: restoring steady-state scheduler after AoT bootstrap CPU correction\n", __func__);
        sched = std::move(aot_saved_sched);
        aot_bootstrap_cpu_sched_active = false;
    }
}

const llama_model & llama_context::get_model() const {
    return model;
}

const llama_cparams & llama_context::get_cparams() const {
    return cparams;
}

ggml_backend_sched_t llama_context::get_sched() const {
    return sched.get();
}

const std::vector<ggml_backend_t> & llama_context::get_backend_ptrs() const {
    return backend_ptrs;
}

uint32_t llama_context::n_ctx() const {
    return cparams.n_ctx;
}

uint32_t llama_context::n_ctx_seq() const {
    return cparams.n_ctx_seq;
}

uint32_t llama_context::n_batch() const {
    return cparams.n_batch;
}

uint32_t llama_context::n_ubatch() const {
    return cparams.n_ubatch;
}

uint32_t llama_context::n_seq_max() const {
    return cparams.n_seq_max;
}

uint32_t llama_context::n_threads() const {
    return cparams.n_threads;
}

uint32_t llama_context::n_threads_batch() const {
    return cparams.n_threads_batch;
}

llama_memory_t llama_context::get_memory() const {
    return memory.get();
}

bool llama_context::memory_update(bool optimize) {
    if (!memory) {
        return false;
    }

    {
        const auto mctx = memory->init_update(this, optimize);
        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                    // noop
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    // no updates need to be performed
                    return false;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: failed to prepare memory update\n", __func__);
                    return false;
                }
        }

        // reset the previous graph result to make sure that it won't be reused
        // TODO: change the mctx->apply() to return information if a graph reserve is needed
        //       reset the graph result only if the memory module did reset the scheduler
        gf_res_prev->reset();

        if (!mctx->apply()) {
            LLAMA_LOG_ERROR("%s: failed to apply memory update\n", __func__);
        }
    }

    // if the memory module did any computation, we have to reserve a new worst-case graph
    {
        const auto mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory context");
        }

        const uint32_t n_seqs = cparams.n_seq_max;
        const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

        auto * gf = graph_reserve(n_tokens, n_seqs, n_tokens, mctx.get());
        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to reserve graph after the memory update\n", __func__);
        }
    }

    return true;
}

enum llama_pooling_type llama_context::pooling_type() const {
    return cparams.pooling_type;
}

float * llama_context::get_logits() {
    output_reorder();

    return logits.data;
}

int64_t llama_context::output_resolve_row(int32_t i) const {
    int64_t j = -1;

    // support negative indices (last output row)
    if (i < 0) {
        j = n_outputs + i;
        if (j < 0) {
            throw std::runtime_error(format("negative index out of range [0, %d)", n_outputs));
        }
    } else if ((size_t) i >= output_ids.size()) {
        throw std::runtime_error(format("out of range [0, %zu)", output_ids.size()));
    } else {
        // use output_ids to translate the batch token index into a row number
        // that holds this token's data.
        j = output_ids[i];
    }

    if (j < 0) {
        // the batch token was not configured to output anything
        throw std::runtime_error(format("batch.logits[%d] != true", i));
    }

    if (j >= n_outputs) {
        throw std::runtime_error(format("corrupt output buffer (j=%" PRId64 ", n_outputs=%d)", j, n_outputs));
    }

    return j;
}

float * llama_context::get_logits_ith(int32_t i) {
    output_reorder();

    try {
        if (logits.data == nullptr) {
            throw std::runtime_error("no logits");
        }

        const int64_t j = output_resolve_row(i);
        return logits.data + j*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid logits id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings() {
    output_reorder();

    return embd.data;
}

llama_token * llama_context::get_sampled_tokens()  const{
    return sampling.sampled.data;
}

float * llama_context::get_embeddings_ith(int32_t i) {
    output_reorder();

    try {
        if (embd.data == nullptr) {
            throw std::runtime_error("no embeddings");
        }

        const int64_t j = output_resolve_row(i);
        const uint32_t n_embd_out = model.hparams.n_embd_out();
        return embd.data + j*n_embd_out;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_seq(llama_seq_id seq_id) {
    auto it = embd_seq.find(seq_id);
    if (it == embd_seq.end()) {
        return nullptr;
    }

    return it->second.data();
}

llama_token llama_context::get_sampled_token_ith(int32_t idx) {
    output_reorder();

    if (!sampling.sampled.has_data()) {
        return LLAMA_TOKEN_NULL;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        GGML_ASSERT(row < (int64_t) sampling.sampled.size);
        return sampling.sampled.data[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled token id %d, reason: %s\n", __func__, idx, err.what());
        return LLAMA_TOKEN_NULL;
    }
}

float * llama_context::get_sampled_probs_ith(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size() || sampling.probs_count[row] == 0) {
            return nullptr;
        }
        return sampling.probs.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

float * llama_context::get_sampled_logits_ith(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size() || sampling.logits_count[row] == 0) {
            return nullptr;
        }
        return sampling.logits.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

const llama_token * llama_context::get_sampled_candidates_ith(int32_t idx) {
    output_reorder();

    try {
        const int64_t row = output_resolve_row(idx);
        if (sampling.candidates.has_data() &&
            (size_t) row < sampling.candidates_count.size() &&
            sampling.candidates_count[row] > 0) {
            return sampling.candidates.data + row*model.vocab.n_tokens();
        }
    } catch (const std::exception & err) {
        // fallback to full vocab list
        GGML_UNUSED(err);
    }

    return sampling.token_ids_full_vocab.data();
}

size_t llama_context::get_sampled_candidates_count(int32_t idx) {
    output_reorder();

    if (!sampling.candidates.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.candidates_count.size()) {
            return 0;
        }
        return sampling.candidates_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled candidates count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_logits_count(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return model.vocab.n_tokens();
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size()) {
            return 0;
        }
        return sampling.logits_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_probs_count(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size()) {
            return 0;
        }
        return sampling.probs_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}


void llama_context::attach_threadpool(
           ggml_threadpool_t threadpool,
           ggml_threadpool_t threadpool_batch) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = threadpool;
    this->threadpool_batch = threadpool_batch ? threadpool_batch : threadpool;
}

void llama_context::detach_threadpool() {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = nullptr;
    this->threadpool_batch = nullptr;
}

void llama_context::set_n_threads(int32_t n_threads, int32_t n_threads_batch) {
    LLAMA_LOG_DEBUG("%s: n_threads = %d, n_threads_batch = %d\n", __func__, n_threads, n_threads_batch);

    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads_batch;
}

void llama_context::set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->abort_callback      = abort_callback;
    this->abort_callback_data = abort_callback_data;

    for (auto & backend : backends) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        auto * set_abort_callback_fn = (ggml_backend_set_abort_callback_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_abort_callback");
        if (set_abort_callback_fn) {
            set_abort_callback_fn(backend.get(), this->abort_callback, this->abort_callback_data);
        }
    }
}

void llama_context::set_embeddings(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    cparams.embeddings = value;

    // TODO: not sure yet if we want to reserve here
    //sched_need_reserve = true;
}

void llama_context::set_causal_attn(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.causal_attn == value) {
        return;
    }

    cparams.causal_attn = value;

    sched_need_reserve = true;
}

void llama_context::set_warmup(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.warmup == value) {
        return;
    }

    cparams.warmup = value;

    // warmups are usually with small batches, so no need to reserve
    //sched_need_reserve = true;
}

bool llama_context::set_sampler(llama_seq_id seq_id, llama_sampler * sampler) {
    if (!sampler && sampling.samplers.count(seq_id) == 0) {
        return true;
    }

    LLAMA_LOG_DEBUG("%s: seq_id = %d, sampler = %p\n", __func__, (int) seq_id, (void *) sampler);

    const bool can_offload =
        sampler &&
        sampler->iface->backend_init &&
        sampler->iface->backend_apply &&
        llama_sampler_chain_n(sampler) > 0;

    if (sampler && can_offload) {
        auto * buft = ggml_backend_dev_buffer_type(model.dev_output());

        sampler->iface->backend_init(sampler, buft);

        sampling.samplers[seq_id] = sampler;

        sched_need_reserve = true;

        return true;
    }

    if (sampler && !can_offload) {
        LLAMA_LOG_WARN("%s: sampler '%s' for seq_id = %d, cannot be offloaded to the backend\n", __func__, llama_sampler_name(sampler), seq_id);

        if (sampling.samplers.count(seq_id) > 0) {
            sched_need_reserve = true;
        }

        sampling.samplers.erase(seq_id);

        return false;
    }

    sampling.samplers.erase(seq_id);

    sched_need_reserve = true;

    return true;
}

bool llama_context::set_hetero_plan(llama_hetero_execution_plan plan) {
    if (!ensure_hetero_backends_for_route(plan.route, "manual")) {
        return false;
    }
    return apply_hetero_plan(std::move(plan), /* update_base_plan = */ true, "manual");
}

const llama_hetero_execution_plan & llama_context::get_hetero_plan() const {
    return hetero_plan;
}

void llama_context::set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    if (adapters_lora_are_same(adapters, n_adapters, scales)) {
        return;
    }

    loras.reset(new llama_adapter_loras());

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] != 0.0f) {
            loras->insert({adapters[i], scales[i]});
        }
    }

    sched_need_reserve = true;
}

bool llama_context::adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    // Adapters with a zero scale are never added to `loras`, so also ignore them for the comparison.
    size_t n_non_zero = 0;

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] == 0.0f) {
            continue;
        }
        n_non_zero++;

        auto it = loras->find(adapters[i]);

        if (it == loras->end() || it->second != scales[i]) {
            return false;
        }
    }

    if (n_non_zero != loras->size()) {
        return false;
    }

    return true;
}

bool llama_context::set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end) {
    LLAMA_LOG_DEBUG("%s: il_start = %d, il_end = %d\n", __func__, il_start, il_end);

    // TODO: should we reserve?

    return cvec->apply(model, data, len, n_embd, il_start, il_end);
}

llm_graph_result * llama_context::process_ubatch(const llama_ubatch & ubatch, llm_graph_type gtype, llama_memory_context_i * mctx, ggml_status & ret) {
    const bool trace_timing = hetero_dynamic_trace_timing_detail_enabled() && hetero_phase_trace.active;
    const int64_t t_process_start_us = trace_timing ? ggml_time_us() : 0;

    if (aot_bootstrap_cpu_sched_active) {
        // Keep the bootstrap CPU-only scheduler alive until any async output fetches
        // from the correction graph have been synchronized, then restore the main one.
        hetero_phase_trace_suppress_sync_log = true;
        synchronize();
        hetero_phase_trace_suppress_sync_log = false;
    }

    if (mctx && !mctx->apply()) {
        LLAMA_LOG_ERROR("%s: failed to apply memory context\n", __func__);
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }

    const auto run_graph_once = [&](llm_graph_result * res, bool allow_reuse, bool force_cpu_graph) -> ggml_status {
        aot_force_cpu_graph = force_cpu_graph;

        auto restore_force_cpu = [this]() {
            aot_force_cpu_graph = false;
        };

        auto * gf = res->get_gf();
        bool reused_graph = false;

        // the new graph parameters
        // in order to correctly reuse a graph, its full topology has to be uniquely determined by these parameters
        const auto gparams = graph_params(res, ubatch, mctx, gtype);

        if (!graph_reuse_disable && allow_reuse && res->can_reuse(gparams)) {
            n_reused++;
            reused_graph = true;
        } else {
            res->reset();

            ggml_backend_sched_reset(sched.get());
            ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);

            gf = model.build_graph(gparams);

            if (!gf) {
                restore_force_cpu();
                LLAMA_LOG_ERROR("%s: failed to initialize graph\n", __func__);
                return GGML_STATUS_FAILED;
            }

            if (!ggml_backend_sched_alloc_graph(sched.get(), gf)) {
                restore_force_cpu();
                LLAMA_LOG_ERROR("%s: failed to allocate graph\n", __func__);
                return GGML_STATUS_ALLOC_FAILED;
            }
        }

        // FIXME this call causes a crash if any model inputs were not used in the graph and were therefore not allocated
        res->set_inputs(&ubatch);

        restore_force_cpu();
        if (trace_timing) {
            if (reused_graph) {
                hetero_phase_trace.graph_runs_reused++;
            } else {
                hetero_phase_trace.graph_runs_rebuilt++;
            }
        }
        return graph_compute(res->get_gf(), ubatch, ubatch.n_tokens > 1);
    };

    const bool aot_single_token_pos0 =
        std::getenv("GGML_QNN_AOT_CONFIG") != nullptr &&
        aot_active_route_requests_qnn &&
        !aot_skip_bootstrap_for_next_decode &&
        ubatch.n_tokens == 1 &&
        ubatch.n_pos > 0 &&
        ubatch.pos != nullptr &&
        ubatch.pos[0] == 0;

    auto * res = gf_res_prev.get();
    auto status = run_graph_once(res, /* allow_reuse = */ true, /* force_cpu_graph = */ false);
    if (ubatch.n_tokens == 1) {
        aot_skip_bootstrap_for_next_decode = false;
    }
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: failed to compute graph, compute status: %d\n", __func__, status);
        ret = status;
        return nullptr;
    }

    if (aot_single_token_pos0) {
        LLAMA_LOG_INFO("%s: running AoT bootstrap CPU correction for initial decode token\n", __func__);
        if (trace_timing) {
            hetero_phase_trace.bootstrap_ran = true;
        }

        // Finish the seed run before rebuilding the scheduler for the correction graph.
        const int64_t t_bootstrap_sync_start_us = trace_timing ? ggml_time_us() : 0;
        ggml_backend_sched_synchronize(sched.get());
        if (trace_timing) {
            hetero_phase_trace.bootstrap_sync_us += ggml_time_us() - t_bootstrap_sync_start_us;
        }

        // The scheduler will be rebound to the correction graph below, so rebuild the
        // steady-state QNN graph on the next decode rather than reusing stale splits.
        res->invalidate_reuse();

        auto * cpu_res = gf_res_reserve.get();
        const bool correction_requires_preserving_offloaded_weights = model.n_gpu_layers() > 0;

        if (correction_requires_preserving_offloaded_weights) {
            LLAMA_LOG_INFO("%s: bootstrap correction keeps the steady-state scheduler because n_gpu_layers=%d leaves model weights pre-allocated on non-CPU backends\n",
                    __func__, model.n_gpu_layers());
        } else {
            ggml_backend_buffer_type_t cpu_buft = ggml_backend_get_default_buffer_type(backend_cpu);
            for (size_t i = 0; i < backend_ptrs.size(); ++i) {
                if (backend_ptrs[i] == backend_cpu) {
                    cpu_buft = backend_buft[i];
                    break;
                }
            }

            ggml_backend_t cpu_backend_ptrs[] = { backend_cpu };
            ggml_backend_buffer_type_t cpu_backend_bufts[] = { cpu_buft };

            aot_saved_sched = std::move(sched);
            const int64_t t_bootstrap_sched_start_us = trace_timing ? ggml_time_us() : 0;
            sched.reset(ggml_backend_sched_new(
                    cpu_backend_ptrs,
                    cpu_backend_bufts,
                    1,
                    cpu_res->get_max_nodes(),
                    /* parallel = */ false,
                    cparams.op_offload));
            if (trace_timing) {
                hetero_phase_trace.bootstrap_sched_rebuild_us += ggml_time_us() - t_bootstrap_sched_start_us;
            }

            if (!sched) {
                sched = std::move(aot_saved_sched);
                LLAMA_LOG_ERROR("%s: failed to create CPU-only scheduler for AoT bootstrap correction\n", __func__);
                ret = GGML_STATUS_ALLOC_FAILED;
                return nullptr;
            }

            aot_bootstrap_cpu_sched_active = true;
        }

        const char * prev_qnn_disable = std::getenv("GGML_QNN_DISABLE_BACKEND");
        const bool had_prev_qnn_disable = prev_qnn_disable != nullptr;
        const std::string prev_qnn_disable_value = had_prev_qnn_disable ? prev_qnn_disable : "";

        setenv("GGML_QNN_DISABLE_BACKEND", "1", 1);
        status = run_graph_once(cpu_res, /* allow_reuse = */ false, /* force_cpu_graph = */ true);

        if (had_prev_qnn_disable) {
            setenv("GGML_QNN_DISABLE_BACKEND", prev_qnn_disable_value.c_str(), 1);
        } else {
            unsetenv("GGML_QNN_DISABLE_BACKEND");
        }

        if (status != GGML_STATUS_SUCCESS) {
            if (aot_bootstrap_cpu_sched_active) {
                sched = std::move(aot_saved_sched);
                aot_bootstrap_cpu_sched_active = false;
            }
            LLAMA_LOG_ERROR("%s: failed to compute bootstrap CPU correction graph, status: %d\n", __func__, status);
            ret = status;
            return nullptr;
        }

        if (trace_timing) {
            hetero_phase_trace.process_ubatches++;
            hetero_phase_trace.process_ubatch_us += ggml_time_us() - t_process_start_us;
        }

        ret = GGML_STATUS_SUCCESS;
        return cpu_res;
    }

    if (trace_timing) {
        hetero_phase_trace.process_ubatches++;
        hetero_phase_trace.process_ubatch_us += ggml_time_us() - t_process_start_us;
    }

    ret = GGML_STATUS_SUCCESS;

    return res;
}

int llama_context::encode(const llama_batch & batch_inp) {
    GGML_ASSERT((!batch_inp.token && batch_inp.embd) || (batch_inp.token && !batch_inp.embd)); // NOLINT

    if (batch_inp.n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const auto & hparams = model.hparams;

    const int64_t n_embd  = hparams.n_embd_inp();
    const int64_t n_vocab = model.vocab.n_tokens();

    // note: during encode, we always pass the full sequence starting from pos = 0
    if (!balloc->init(batch_inp, model.vocab, nullptr, n_embd, cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max, true)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens = balloc->get_n_tokens();

    // [TAG_NO_CACHE_PAD]
    // TODO: add new split mode where we pad the input sequences so that ubatch.equal_seqs == true
    const llama_ubatch ubatch = balloc->split_simple(n_tokens);

    // micro-batching is not possible for non-causal encoding, so we process the batch in a single shot
    GGML_ASSERT(cparams.n_ubatch >= n_tokens && "encoder requires n_ubatch >= n_tokens");

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }

    // TODO: this clear of the buffer can easily be forgotten - need something better
    embd_seq.clear();

    sched_reserve_request_tokens = n_tokens;
    sched_reserve();

    n_queued_tokens += n_tokens;

    // reserve output buffer
    if (output_reserve(n_tokens) < n_tokens) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %u outputs\n", __func__, n_tokens);
        return -2;
    };

    for (uint32_t i = 0; i < n_tokens; ++i) {
        output_ids[i] = i;
    }

    n_outputs = n_tokens;

    const auto causal_attn_org = cparams.causal_attn;

    // always use non-causal attention for encoder graphs
    // TODO: this is a tmp solution until we have a proper way to support enc-dec models
    //       ref: https://github.com/ggml-org/llama.cpp/pull/12181#issuecomment-2730451223
    cparams.causal_attn = false;

    ggml_status status;
    const auto * res = process_ubatch(ubatch, LLM_GRAPH_TYPE_ENCODER, nullptr, status);

    cparams.causal_attn = causal_attn_org;

    if (!res) {
        switch (status) {
            case GGML_STATUS_ABORTED:      return  2;
            case GGML_STATUS_ALLOC_FAILED: return -2;
            case GGML_STATUS_FAILED:       return -3;
            case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
        }
    }

    auto * t_logits = res->get_logits();
    auto * t_embd = res->get_embd_pooled() ? res->get_embd_pooled() : res->get_embd();

    // extract logits
    if (logits.data && t_logits) {
        ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
        GGML_ASSERT(backend_res != nullptr);
        GGML_ASSERT(logits.data != nullptr);

        ggml_backend_tensor_get_async(backend_res, t_logits, logits.data, 0, n_tokens*n_vocab*sizeof(float));
    }

    // extract embeddings
    if (embd.data && t_embd) {
        ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
        GGML_ASSERT(backend_embd != nullptr);

        switch (cparams.pooling_type) {
            case LLAMA_POOLING_TYPE_NONE:
                {
                    // extract token embeddings
                    GGML_ASSERT(embd.data != nullptr);
                    const uint32_t n_embd_out = hparams.n_embd_out();

                    GGML_ASSERT(n_tokens*n_embd_out <= (int64_t) embd.size);
                    ggml_backend_tensor_get_async(backend_embd, t_embd, embd.data, 0, n_tokens*n_embd_out*sizeof(float));
                } break;
            case LLAMA_POOLING_TYPE_MEAN:
            case LLAMA_POOLING_TYPE_CLS:
            case LLAMA_POOLING_TYPE_LAST:
                {
                    // extract sequence embeddings
                    auto & embd_seq_out = embd_seq;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        embd_seq_out[seq_id].resize(n_embd);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd*seq_idx)*sizeof(float), n_embd*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_RANK:
                {
                    // extract the rerank score - n_cls_out floats per sequence
                    auto & embd_seq_out = embd_seq;

                    const uint32_t n_cls_out = hparams.n_cls_out;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        embd_seq_out[seq_id].resize(n_cls_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_UNSPECIFIED:
                {
                    GGML_ABORT("unknown pooling type");
                }
        }
    }

    // TODO: hacky solution
    if (model.arch == LLM_ARCH_T5 && t_embd) {
        //cross.t_embd = t_embd;

        synchronize();

        cross.n_embd = t_embd->ne[0];
        cross.n_enc  = t_embd->ne[1];
        cross.v_embd.resize(cross.n_embd*cross.n_enc);
        memcpy(cross.v_embd.data(), embd.data, ggml_nbytes(t_embd));

        const auto & batch = balloc->get_batch();

        // remember the sequence ids used during the encoding - needed for cross attention later
        cross.seq_ids_enc.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; i++) {
            cross.seq_ids_enc[i].clear();

            for (int s = 0; s < batch.n_seq_id[i]; s++) {
                const llama_seq_id seq_id = batch.seq_id[i][s];

                cross.seq_ids_enc[i].insert(seq_id);
            }
        }
    }

    return 0;
}

static std::map<llama_seq_id, uint32_t> build_seq_to_output_row(const llama_ubatch & ubatch, uint32_t row_offset) {
    std::map<llama_seq_id, uint32_t> seq_to_row;
    // how many output tokens we have seen so far for this ubatch.
    uint32_t local = 0;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        // skip tokens that are not output.
        if (!ubatch.output[i]) {
            continue;
        }

        const llama_seq_id seq_id = ubatch.seq_id[i][0];
        // row_offset is the number of output tokens before this ubatch.
        seq_to_row[seq_id] = row_offset + local;
        ++local;
    }
    return seq_to_row;
}

static void copy_tensor_async_ints(
    const std::map<llama_seq_id, ggml_tensor*> & tensor_map,
    const buffer_view<llama_token> & sampled,
    const std::map<llama_seq_id, uint32_t> & seq_to_row,
    ggml_backend_sched_t sched) {
    if (!sampled.has_data()) {
        return;
    }

    for (const auto & [seq_id, tensor] : tensor_map) {
        auto it = seq_to_row.find(seq_id);
        if (it == seq_to_row.end()) {
            continue;
        }

        const uint32_t row = it->second;
        GGML_ASSERT(row < sampled.size);

        GGML_ASSERT(ggml_is_contiguous(tensor) && "sampled tokens tensor must be contiguous for async copy");

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        ggml_backend_tensor_get_async(backend, tensor, sampled.data + row, 0, sizeof(sampled.data[row]));
    }
}

static void copy_tensor_async_floats(
    const std::map<llama_seq_id, ggml_tensor*> & tensor_map,
    const buffer_view<float> & dst,
    size_t stride,
    std::vector<uint32_t> & counts,
    const std::map<llama_seq_id, uint32_t> & seq_to_row,
    ggml_backend_sched_t sched) {
    if (!dst.has_data()) {
        return;
    }

    for (const auto & [seq_id, tensor] : tensor_map) {
        auto it = seq_to_row.find(seq_id);
        if (it == seq_to_row.end()) {
            continue;
        }

        const uint32_t row = it->second;
        GGML_ASSERT(row < counts.size());

        GGML_ASSERT(ggml_is_contiguous(tensor) && "logits/probs tensor must be contiguous for async copy");

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        float * row_ptr = dst.data + (size_t) row * stride;
        ggml_backend_tensor_get_async(backend, tensor, row_ptr, 0, ggml_nbytes(tensor));

        // Update the actual number of logits/probabilities that were written for this row.
        counts[row] = ggml_nelements(tensor);
    }
}

static void copy_tensor_async_candidates(
    const std::map<llama_seq_id, ggml_tensor*> & tensor_map,
    const buffer_view<llama_token> & dst,
    size_t stride,
    std::vector<uint32_t> & counts,
    const std::map<llama_seq_id, uint32_t> & seq_to_row,
    ggml_backend_sched_t sched) {
    if (!dst.has_data()) {
        return;
    }

    for (const auto & [seq_id, tensor] : tensor_map) {
        auto it = seq_to_row.find(seq_id);
        if (it == seq_to_row.end()) {
            continue;
        }

        const uint32_t row = it->second;
        GGML_ASSERT(row < counts.size());

        GGML_ASSERT(ggml_is_contiguous(tensor) && "candidates tensor must be contiguous for async copy");

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        llama_token * row_ptr = dst.data + (size_t) row * stride;
        ggml_backend_tensor_get_async(backend, tensor, row_ptr, 0, ggml_nbytes(tensor));

        // Update the actual number of candidates that were written.
        counts[row] = ggml_nelements(tensor);
    }
}

static bool needs_raw_logits(const llama_ubatch & ubatch, const std::map<llama_seq_id, llama_sampler *> & samplers) {
    for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
        if (!ubatch.output[i]) {
            continue;
        }

        // Check if the output token has at least one sequence without a backend sampler.
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
            llama_seq_id seq_id = ubatch.seq_id[i][j];
            if (samplers.find(seq_id) == samplers.end()) {
                return true;
            }
        }
    }
    return false; // all sequences use backend sampling
}

int llama_context::decode(const llama_batch & batch_inp) {
    GGML_ASSERT((!batch_inp.token && batch_inp.embd) || (batch_inp.token && !batch_inp.embd)); // NOLINT

    if (!memory) {
        LLAMA_LOG_DEBUG("%s: cannot decode batches with this context (calling encode() instead)\n", __func__);
        return encode(batch_inp);
    }

    if (batch_inp.n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    // Tag this decode invocation as prefill/decode for backend profiler CSVs.
    // Use the user-visible batch shape (not ubatch splitting) so llama-bench phase
    // runs map to expected labels.
    ggml_profiler_set_inference_phase(batch_inp.n_tokens > 1 ? GGML_INFERENCE_PHASE_PREFILL : GGML_INFERENCE_PHASE_DECODE);

    const auto & vocab   = model.vocab;
    const auto & hparams = model.hparams;
    validate_dynamic_seq0_token_history();
    const size_t seq0_prefix_tokens_before_decode = dynamic_seq0_token_history.size();

    const int64_t n_vocab = vocab.n_tokens();
    const int64_t n_embd  = hparams.n_embd_inp();

    // when computing embeddings, all tokens are output
    const bool output_all   = cparams.embeddings;
    const bool has_samplers = !sampling.samplers.empty();

    const uint32_t n_seq_max = cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max;

    // TODO: avoid this workaround in the future
    if (has_samplers && batch_inp.logits) {
        std::vector<int32_t> seq_output_count(n_seq_max, 0);

        for (int32_t i = 0; i < batch_inp.n_tokens; ++i) {
            if (batch_inp.logits[i] == 0) {
                continue;
            }

            const int ns = batch_inp.n_seq_id ? batch_inp.n_seq_id[i] : 1;

            for (int32_t s = 0; s < ns; ++s) {
                const llama_seq_id seq_id = batch_inp.seq_id ? batch_inp.seq_id[i][s] : 0;

                seq_output_count[seq_id]++;
                if (seq_output_count[seq_id] > 1) {
                    LLAMA_LOG_ERROR("%s: backend sampling requires at most one output token per sequence (seq_id %d had %d)\n",
                            __func__, seq_id, seq_output_count[seq_id]);
                    return -1;
                }
            }
        }
    }

    if (!balloc->init(batch_inp, vocab, memory.get(), n_embd, n_seq_max, output_all)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens_all  = balloc->get_n_tokens();
    const uint32_t n_outputs_all = balloc->get_n_outputs();

    if (output_all) {
        // require that all tokens are output
        if (n_outputs_all != n_tokens_all) {
            LLAMA_LOG_ERROR("%s: pooled embedding requires that all tokens are output (n_outputs_all = %d, n_tokens_all = %d)\n",
                    __func__, n_outputs_all, n_tokens_all);
            return -1;
        }
    }

    GGML_ASSERT(n_tokens_all <= cparams.n_batch);

    GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all) && "non-causal attention requires n_ubatch >= n_tokens");

    if (should_sync_before_dynamic_gpu_freq_switch(n_tokens_all)) {
        const bool trace_timing = hetero_dynamic_trace_timing_detail_enabled();
        const int64_t t_gpu_freq_pre_sync_start_us = trace_timing ? ggml_time_us() : 0;
        synchronize();
        pending_gpu_freq_pre_sync_us =
            trace_timing ? ggml_time_us() - t_gpu_freq_pre_sync_start_us : 0;
        if (trace_timing) {
            LLAMA_LOG_INFO("%s: synchronized before GPU frequency switch pre_sync_us=%" PRId64 "\n",
                    __func__,
                    pending_gpu_freq_pre_sync_us);
        }
    }

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }

    if (hetero_dynamic_trace_timing_detail_enabled()) {
        if (hetero_phase_trace.active) {
            LLAMA_LOG_WARN("%s: overwriting pending hetero phase trace for phase=%s n_tokens=%u before synchronize() completed\n",
                    __func__,
                    hetero_phase_name(hetero_phase_trace.n_tokens),
                    hetero_phase_trace.n_tokens);
        }
        hetero_phase_trace.reset();
        hetero_phase_trace.active = true;
        hetero_phase_trace.n_tokens = n_tokens_all;
        hetero_phase_trace.batch_start_us = t_compute_start_us;
        hetero_phase_trace.gpu_freq_pre_sync_us = pending_gpu_freq_pre_sync_us;
        pending_gpu_freq_pre_sync_us = 0;
    }

    // TODO: this clear of the buffer can easily be forgotten - need something better
    embd_seq.clear();
    output_swaps.clear();

    sched_reserve_request_tokens = n_tokens_all;
    maybe_apply_dynamic_route(n_tokens_all);
    if (qnn_prefix_replay_pending) {
        const bool trace_was_active = hetero_phase_trace.active;
        const int64_t t_replay_start_us =
            (hetero_dynamic_trace_timing_detail_enabled() && trace_was_active) ? ggml_time_us() : 0;
        hetero_phase_trace.active = false;
        const bool replay_ok = replay_dynamic_qnn_prefix();
        hetero_phase_trace.active = trace_was_active;
        if (hetero_dynamic_trace_timing_detail_enabled() && trace_was_active) {
            hetero_phase_trace.kv_migration_us += ggml_time_us() - t_replay_start_us;
        }
        if (!replay_ok) {
            return -3;
        }
    }
    sched_reserve();
    n_queued_tokens += n_tokens_all;

    bool did_optimize = false;

    // handle any pending shifts/copies
    const int64_t t_memory_update_start_us =
        (hetero_dynamic_trace_timing_detail_enabled() && hetero_phase_trace.active) ? ggml_time_us() : 0;
    memory_update(false);
    if (hetero_dynamic_trace_timing_detail_enabled() && hetero_phase_trace.active) {
        hetero_phase_trace.memory_update_us += ggml_time_us() - t_memory_update_start_us;
    }

    llama_memory_context_ptr mctx;

    while (true) {
        mctx = memory->init_batch(*balloc, cparams.n_ubatch, output_all);
        if (!mctx) {
            return -2;
        }

        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    LLAMA_LOG_ERROR("%s: unexpected memory context status: %d\n", __func__, mctx->get_status());

                    return -2;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
                {
                    if (!did_optimize) {
                        did_optimize = true;

                        if (memory_update(true)) {
                            LLAMA_LOG_DEBUG("%s: retrying batch size %d after cache optimization\n", __func__, balloc->get_n_tokens());

                            continue;
                        }
                    }

                    LLAMA_LOG_WARN("%s: failed to find a memory slot for batch of size %d\n", __func__, balloc->get_n_tokens());

                    return 1;
                }
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: compute failed while preparing batch of size %d\n", __func__, balloc->get_n_tokens());

                    return -2;
                }
        }

        break;
    }

    // reserve output buffer
    if (output_reserve(n_outputs_all) < n_outputs_all) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
        return -2;
    };

    int64_t n_outputs_prev = 0;

    do {
        const auto & ubatch = mctx->get_ubatch();

        // count the outputs in this ubatch
        {
            int32_t n_outputs_new = 0;

            if (n_outputs_all == n_tokens_all) {
                n_outputs_new = ubatch.n_tokens;
            } else {
                for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
                    n_outputs_new += (int32_t) (ubatch.output[i] != 0);
                }
            }

            // needs to happen before the graph is built
            n_outputs = n_outputs_new;
        }

        ggml_status status;
        const auto * res = process_ubatch(ubatch, LLM_GRAPH_TYPE_DECODER, mctx.get(), status);

        if (!res) {
            // the last ubatch failed or was aborted -> remove all positions of that ubatch from the memory module
            llama_pos pos_min[LLAMA_MAX_SEQ];
            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                pos_min[s] = std::numeric_limits<llama_pos>::max();
            }

            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                const auto & seq_id = ubatch.seq_id[i][0];

                pos_min[seq_id] = std::min(pos_min[seq_id], ubatch.pos[i]);
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (pos_min[s] == std::numeric_limits<llama_pos>::max()) {
                    continue;
                }

                LLAMA_LOG_WARN("%s: removing memory module entries for seq_id = %d, pos = [%d, +inf)\n", __func__, s, pos_min[s]);

                memory->seq_rm(s, pos_min[s], -1);
            }

            switch (status) {
                case GGML_STATUS_ABORTED:      return  2;
                case GGML_STATUS_ALLOC_FAILED: return -2;
                case GGML_STATUS_FAILED:       return -3;
                case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
            }
        }

        // plot the computation graph in dot format (for debugging purposes)
        //if (n_past%100 == 0) {
        //    ggml_graph_dump_dot(gf, NULL, "llama.dot");
        //}

        auto * t_logits = res->get_logits();
        auto * t_embd   = cparams.embeddings ? res->get_embd() : nullptr;

        if (t_embd && res->get_embd_pooled()) {
            t_embd = res->get_embd_pooled();
        }

        // extract logits
        if (logits.data && t_logits && n_outputs > 0 && needs_raw_logits(ubatch, sampling.samplers)) {
            ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
            GGML_ASSERT(backend_res != nullptr);
            GGML_ASSERT(logits.data != nullptr);

            float * logits_out = logits.data + n_outputs_prev*n_vocab;

            if (n_outputs) {
                GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                GGML_ASSERT((n_outputs_prev + n_outputs)*n_vocab <= (int64_t) logits.size);
                ggml_backend_tensor_get_async(backend_res, t_logits, logits_out, 0, n_outputs*n_vocab*sizeof(float));
            }
        }

        // extract embeddings
        if (embd.data && t_embd && n_outputs > 0) {
            ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
            GGML_ASSERT(backend_embd != nullptr);

            switch (cparams.pooling_type) {
                case LLAMA_POOLING_TYPE_NONE:
                    {
                        // extract token embeddings
                        GGML_ASSERT(embd.data != nullptr);
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        float * embd_out = embd.data + n_outputs_prev*n_embd_out;

                        if (n_outputs) {
                            GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                            GGML_ASSERT((n_outputs_prev + n_outputs)*n_embd_out <= (int64_t) embd.size);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_out, 0, n_outputs*n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_MEAN:
                case LLAMA_POOLING_TYPE_CLS:
                case LLAMA_POOLING_TYPE_LAST:
                    {
                        // extract sequence embeddings (cleared before processing each batch)
                        auto & embd_seq_out = embd_seq;

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_embd);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd*seq_idx)*sizeof(float), n_embd*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_RANK:
                    {
                        // extract the rerank score - n_cls_out floats per sequence
                        auto & embd_seq_out = embd_seq;

                        const uint32_t n_cls_out = hparams.n_cls_out;

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_cls_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_UNSPECIFIED:
                    {
                        GGML_ABORT("unknown pooling type");
                    }
            }
        }

        // Copy backend sampling output if this ubatch produced any sampling tensors.
        if (has_samplers && (!res->t_sampled.empty() || !res->t_sampled_probs.empty() || !res->t_sampled_logits.empty())) {
            const auto seq_to_output_row = build_seq_to_output_row(ubatch, n_outputs_prev);
            const auto stride = n_vocab;

            // async copy the sampling data from the backend to the host
            copy_tensor_async_ints(res->t_sampled, sampling.sampled, seq_to_output_row, sched.get());

            copy_tensor_async_floats    (res->t_sampled_logits, sampling.logits,     stride, sampling.logits_count,     seq_to_output_row, sched.get());
            copy_tensor_async_floats    (res->t_sampled_probs,  sampling.probs,      stride, sampling.probs_count,      seq_to_output_row, sched.get());
            copy_tensor_async_candidates(res->t_candidates,     sampling.candidates, stride, sampling.candidates_count, seq_to_output_row, sched.get());
        }

        n_outputs_prev += n_outputs;
    } while (mctx->next());

    // set to total number of outputs in the batch, for use in llama_get_logits_ith
    n_outputs = n_outputs_all;

    // set output mappings
    if (n_outputs > 0) {
        bool sorted_output = true;

        auto & out_ids = balloc->get_out_ids();

        GGML_ASSERT(out_ids.size() == (size_t) n_outputs);

        for (int64_t i = 0; i < n_outputs; ++i) {
            int64_t out_id = out_ids[i];
            output_ids[out_id] = i;
            if (out_id != i) {
                sorted_output = false;
            }
        }

        // make the outputs have the same order they had in the user-provided batch
        // note: this is mostly relevant for recurrent models atm
        if (!sorted_output && n_outputs > 1) {
            GGML_ASSERT((size_t) n_outputs == out_ids.size());

            // TODO: is there something more efficient which also minimizes swaps?
            // selection sort, to minimize swaps (from https://en.wikipedia.org/wiki/Selection_sort)
            for (uint32_t i = 0; i < n_outputs - 1; ++i) {
                uint32_t j_min = i;
                for (uint32_t j = i + 1; j < n_outputs; ++j) {
                    if (out_ids[j] < out_ids[j_min]) {
                        j_min = j;
                    }
                }
                if (j_min == i) {
                    continue;
                }
                std::swap(out_ids[i], out_ids[j_min]);

                // remember the swaps and apply them lazily upon logits/embeddings access
                output_swaps.push_back({ i, j_min });
            }

            std::fill(output_ids.begin(), output_ids.end(), -1);

            for (uint32_t i = 0; i < n_outputs; ++i) {
                output_ids[out_ids[i]] = i;
            }
        }
    }

    // wait for the computation to finish (automatically done when obtaining the model output)
    //synchronize();

    record_dynamic_seq0_token_history(batch_inp, seq0_prefix_tokens_before_decode);
    return 0;
}

//
// output
//

uint32_t llama_context::output_reserve(int32_t n_outputs) {
    const auto & hparams = model.hparams;
    const auto & vocab   = model.vocab;

    const int64_t n_outputs_max = std::max<int64_t>(n_outputs, n_seq_max());

    const auto n_batch    = cparams.n_batch;
    const auto n_vocab    = vocab.n_tokens();
    const auto n_embd_out = hparams.n_embd_out();

    bool has_logits = true;
    bool has_embd   = cparams.embeddings;

    // TODO: hacky enc-dec support
    if (model.arch == LLM_ARCH_T5) {
        has_logits = true;
        has_embd   = true;
    }


    size_t backend_float_count = 0;
    size_t backend_token_count = 0;

    logits.size = has_logits ? n_vocab*n_outputs_max : 0;
    embd.size   = has_embd ? n_embd_out*n_outputs_max : 0;

    // Allocate backend sampling output buffers if there are backend samplers configured.
    const bool has_sampling = !sampling.samplers.empty();
    if (has_sampling) {
        backend_float_count = 2 * n_vocab * n_outputs_max;      // logits + probs
        backend_token_count = (1 + n_vocab) * n_outputs_max;    // sampled + candidates
    }

    if (output_ids.empty()) {
        // init, never resized afterwards
        output_ids.resize(n_batch);
    }

    const std::string output_backend_name =
        llama_hetero_canonical_backend(hetero_plan.route.backend_for(llama_hetero_route_stage::OUTPUT));
    ggml_backend_buffer_type_t desired_output_buft = ggml_backend_cpu_buffer_type();
    ggml_backend_dev_t desired_output_dev = nullptr;

    if (!output_backend_name.empty()) {
        ggml_backend_t desired_output_backend = find_backend_for_route(output_backend_name);
        desired_output_dev = desired_output_backend != nullptr ? ggml_backend_get_device(desired_output_backend) : nullptr;
    }
    if (desired_output_dev == nullptr) {
        desired_output_dev = model.dev_output();
    }

    if (desired_output_dev != nullptr) {
        ggml_backend_buffer_type_t output_dev_host_buft =
            ggml_backend_dev_host_buffer_type(desired_output_dev);
        if (output_dev_host_buft != nullptr) {
            desired_output_buft = output_dev_host_buft;
        }
    }

    const size_t prev_size = buf_output ? ggml_backend_buffer_get_size(buf_output.get()) : 0;
    const bool output_buft_changed =
        buf_output != nullptr &&
        ggml_backend_buffer_get_type(buf_output.get()) != desired_output_buft;
    const size_t new_size  =
        (logits.size + embd.size + backend_float_count) * sizeof(float) +
        (                          backend_token_count) * sizeof(llama_token);

    // alloc only when more than the current capacity is required
    // TODO: also consider shrinking the buffer
    if (!buf_output || prev_size < new_size || output_buft_changed) {
        if (buf_output) {
#ifndef NDEBUG
            // This doesn't happen often, but may be annoying in some cases (like the HellaSwag benchmark)
            LLAMA_LOG_DEBUG("%s: reallocating output buffer from size %.02f MiB to %.02f MiB\n", __func__, prev_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
#endif
            synchronize();

            // TODO: not needed?
            buf_output = nullptr;
            logits.data = nullptr;
            embd.data = nullptr;
        }

        buf_output.reset(ggml_backend_buft_alloc_buffer(desired_output_buft, new_size));
        if (buf_output == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to allocate output buffer of size %.2f MiB\n", __func__, new_size / (1024.0 * 1024.0));
            return 0;
        }
    }

    float * output_base = (float *) ggml_backend_buffer_get_base(buf_output.get());

    size_t offset = 0;
    uint8_t * base = (uint8_t *) output_base;

    logits = has_logits ? buffer_view<float>{output_base, logits.size} : buffer_view<float>{nullptr, 0};
    offset += logits.size * sizeof(float);

    embd = has_embd ? buffer_view<float>{(float *) (base + offset), embd.size} : buffer_view<float>{nullptr, 0};
    offset += embd.size * sizeof(float);

    if (has_sampling) {
        sampling.logits = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.logits.size * sizeof(float);

        sampling.probs = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.probs.size * sizeof(float);

        sampling.sampled = {(llama_token *) (base + offset), (size_t)n_outputs_max};
        offset += sampling.sampled.size * sizeof(llama_token);

        sampling.candidates = {(llama_token *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.candidates.size * sizeof(llama_token);

        // The count vectors keep track of the actual number of logits/probs/candidates
        // copied from the backend for each output row.

        sampling.logits_count.resize(n_outputs_max);
        sampling.probs_count.resize(n_outputs_max);
        sampling.candidates_count.resize(n_outputs_max);

        std::fill(sampling.logits_count.begin(),     sampling.logits_count.end(),     0);
        std::fill(sampling.probs_count.begin(),      sampling.probs_count.end(),      0);
        std::fill(sampling.candidates_count.begin(), sampling.candidates_count.end(), 0);

        std::fill_n(sampling.sampled.data, sampling.sampled.size, LLAMA_TOKEN_NULL);
    } else {
        sampling.logits     = {nullptr, 0};
        sampling.probs      = {nullptr, 0};
        sampling.sampled    = {nullptr, 0};
        sampling.candidates = {nullptr, 0};

        sampling.logits_count.clear();
        sampling.probs_count.clear();
        sampling.candidates_count.clear();
    }

    // set all ids as invalid (negative)
    std::fill(output_ids.begin(), output_ids.end(), -1);

    this->n_outputs = 0;

    return n_outputs_max;
}

void llama_context::output_reorder() {
    const uint64_t n_vocab = model.vocab.n_tokens();
    const uint64_t n_embd  = model.hparams.n_embd;

    for (size_t s = 0; s < output_swaps.size(); ++s) {
        const uint64_t i0 = output_swaps[s].i0;
        const uint64_t i1 = output_swaps[s].i1;

        if (logits.size > 0) {
            for (uint64_t k = 0; k < n_vocab; k++) {
                std::swap(logits.data[i0*n_vocab + k], logits.data[i1*n_vocab + k]);
            }
        }

        if (embd.size > 0) {
            for (uint64_t k = 0; k < n_embd; k++) {
                std::swap(embd.data[i0*n_embd + k], embd.data[i1*n_embd + k]);
            }
        }

        if (!sampling.samplers.empty()) {
            assert(sampling.logits.size > 0);
            assert(sampling.probs.size > 0);
            assert(sampling.candidates.size > 0);
            assert(sampling.sampled.size > 0);
            assert(sampling.logits_count.size() > 0);
            assert(sampling.probs_count.size() > 0);
            assert(sampling.candidates_count.size() > 0);

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.logits.data[i0*n_vocab + k], sampling.logits.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.probs.data[i0*n_vocab + k], sampling.probs.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.candidates.data[i0*n_vocab + k], sampling.candidates.data[i1*n_vocab + k]);
            }

            std::swap(sampling.sampled.data[i0],     sampling.sampled.data[i1]);
            std::swap(sampling.logits_count[i0],     sampling.logits_count[i1]);
            std::swap(sampling.probs_count[i0],      sampling.probs_count[i1]);
            std::swap(sampling.candidates_count[i0], sampling.candidates_count[i1]);
        }
    }

    output_swaps.clear();
}

//
// graph
//

uint32_t llama_context::graph_max_nodes(uint32_t n_tokens) const {
    if (model.arch == LLM_ARCH_QWEN3NEXT || model.arch == LLM_ARCH_KIMI_LINEAR || model.arch == LLM_ARCH_QWEN35 || model.arch == LLM_ARCH_QWEN35MOE) {
        return std::max<uint32_t>(n_tokens * 40, 32u * model.n_tensors());
    }
    uint32_t res = std::max<uint32_t>(1024u, 8u*model.n_tensors());
    for (const auto & lora : model.loras) {
        res += lora->get_n_nodes();
    }
    return res;
}

llm_graph_result * llama_context::get_gf_res_reserve() const {
    return static_cast<llm_graph_result *>(gf_res_reserve.get());
}

ggml_cgraph * llama_context::graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only, size_t * sizes) {
    LLAMA_LOG_DEBUG("%s: reserving a graph for ubatch with n_tokens = %4u, n_seqs = %2u, n_outputs = %4u\n", __func__, n_tokens, n_seqs, n_outputs);
    GGML_ASSERT(n_outputs >= 1);

    if (n_tokens % n_seqs != 0) {
        n_tokens = ((n_tokens + (n_seqs - 1)) / n_seqs) * n_seqs; // round to next multiple of n_seqs
        n_outputs = std::max(n_outputs, n_tokens);

        LLAMA_LOG_DEBUG("%s: making n_tokens a multiple of n_seqs - n_tokens = %u, n_seqs = %u, n_outputs = %u\n", __func__, n_tokens, n_seqs, n_outputs);
    }

    ggml_backend_sched_reset(sched.get());

    // when the scheduler is reset, we cannot reuse the old graph, so we reset the previous graph result to prevent that
    gf_res_prev->reset();

    // store the n_outputs as it is, and restore it afterwards
    // TODO: not sure if needed, might simplify in the future by removing this
    const auto save_n_outputs = this->n_outputs;

    this->n_outputs = n_outputs;

    llama_batch_allocr balloc(model.hparams.n_pos_per_embd());
    llama_ubatch ubatch = balloc.ubatch_reserve(n_tokens/n_seqs, n_seqs);

    // set one output token per sequence in order to activate all backend samplers
    std::vector<llama_seq_id> seq_ids(n_seqs);
    for (uint32_t i = 0; i < n_seqs; ++i) {
        seq_ids[i] = i;
        ubatch.n_seq_id[i] = 1;
        ubatch.seq_id[i] = &seq_ids[i];
        ubatch.output[i] = true;
    }

    auto * res = gf_res_reserve.get();

    const auto gparams = graph_params(res, ubatch, mctx, LLM_GRAPH_TYPE_DEFAULT);

    res->reset();

    auto * gf = model.build_graph(gparams);

    this->n_outputs = save_n_outputs;

    // initialize scheduler with the specified graph
    if (split_only) {
        if (sizes) {
            ggml_backend_sched_reserve_size(sched.get(), gf, sizes);
        } else {
            ggml_backend_sched_split_graph(sched.get(), gf);
        }
    } else if (!ggml_backend_sched_reserve(sched.get(), gf)) {
        GGML_ASSERT(!sizes);
        LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
        return nullptr;
    }

    return gf;
}

llm_graph_params llama_context::graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const {
    return {
        /*.arch        =*/ model.arch,
        /*.hparams     =*/ model.hparams,
        /*.cparams     =*/ cparams,
        /*.ubatch      =*/ ubatch,
        /*.gtype       =*/ gtype,
        /*.sched       =*/ sched.get(),
        /*.backend_cpu =*/ backend_cpu,
        /*.model       =*/ &model,
        /*.hetero_route =*/ hetero_plan.route,
        /*.cvec        =*/ cvec.get(),
        /*.loras       =*/ loras.get(),
        /*.mctx        =*/ mctx,
        /*.cross       =*/ &cross,
        /*.samplers    =*/ sampling.samplers,
        /*.n_outputs   =*/ n_outputs,
        /*.cb          =*/ graph_get_cb(),
        /*.res         =*/ res,
    };
}

ggml_status llama_context::graph_compute(
            ggml_cgraph * gf,
                   bool   batched) {
    const llama_ubatch empty_ubatch = {};
    return graph_compute(gf, empty_ubatch, batched);
}

ggml_status llama_context::graph_compute(
            ggml_cgraph * gf,
      const llama_ubatch & ubatch,
                   bool   batched) {
    GGML_UNUSED(ubatch);

    int n_threads        = batched ? cparams.n_threads_batch : cparams.n_threads;
    ggml_threadpool_t tp = batched ? threadpool_batch        : threadpool;

    if (backend_cpu != nullptr) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
        auto * set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        if (set_threadpool_fn) {
            set_threadpool_fn(backend_cpu, tp);
        }
    }

    // set the number of threads for all the backends
    for (const auto & set_n_threads_fn : set_n_threads_fns) {
        set_n_threads_fn.second(set_n_threads_fn.first, n_threads);
    }

    if (hetero_phase_trace.active && env_flag_enabled("GGML_HETERO_DYNAMIC_TRACE_CPU_TASKS")) {
        LLAMA_LOG_INFO("%s: cpu_task_affinity phase=%s decode_token_index=%" PRIu64
                " batched=%s n_threads=%d threadpool=%s %s\n",
                __func__,
                hetero_phase_name(ubatch.n_tokens),
                hetero_phase_trace.decode_token_index,
                batched ? "true" : "false",
                n_threads,
                tp != nullptr ? "set" : "null",
                llama_context_read_task_cpu_affinity_summary().c_str());
    }

    auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: ggml_backend_sched_graph_compute_async failed with error %d\n", __func__, status);
    }

    if (status == GGML_STATUS_SUCCESS &&
            hetero_phase_trace.active &&
            ubatch.n_tokens == 1 &&
            !hetero_phase_trace.requested_cpu_affinity_mask.empty() &&
            hetero_phase_trace.cpu_affinity_apply_us > 0) {
        std::string actual_cpu_affinity_mask;
        std::string cpu_affinity_error;
        const int64_t t_cpu_affinity_reapply_start_us = ggml_time_us();
        const bool cpu_affinity_reapplied = llama_context_apply_cpu_affinity_mask(
                hetero_phase_trace.requested_cpu_affinity_mask,
                actual_cpu_affinity_mask,
                cpu_affinity_error);
        const int64_t cpu_affinity_reapply_us = ggml_time_us() - t_cpu_affinity_reapply_start_us;
        hetero_phase_trace.cpu_affinity_apply_us += cpu_affinity_reapply_us;

        if (cpu_affinity_reapplied) {
            cpu_current_affinity_mask = actual_cpu_affinity_mask;
            hetero_phase_trace.actual_cpu_affinity_mask = actual_cpu_affinity_mask;
            if (env_flag_enabled("GGML_HETERO_DYNAMIC_TRACE_CPU_TASKS")) {
                LLAMA_LOG_INFO("%s: cpu_task_affinity_reapplied decode_token_index=%" PRIu64
                        " apply_us=%" PRId64 " actual_affinity=%s %s\n",
                        __func__,
                        hetero_phase_trace.decode_token_index,
                        cpu_affinity_reapply_us,
                        actual_cpu_affinity_mask.empty() ? "<unknown>" : actual_cpu_affinity_mask.c_str(),
                        llama_context_read_task_cpu_affinity_summary().c_str());
            }
        } else {
            LLAMA_LOG_WARN("%s: failed to reapply CPU affinity after compute mask=%s error=%s\n",
                    __func__,
                    hetero_phase_trace.requested_cpu_affinity_mask.c_str(),
                    cpu_affinity_error.empty() ? "<none>" : cpu_affinity_error.c_str());
        }
    }

    return status;
}

llm_graph_cb llama_context::graph_get_cb() const {
    const bool qnn_aot_enabled = std::getenv("GGML_QNN_AOT_CONFIG") != nullptr;
    ggml_backend_t qnn_aot_backend = nullptr;
    ggml_backend_t qnn_gpu_backend = nullptr;
    ggml_backend_t qnn_cpu_backend = nullptr;
    ggml_backend_t opencl_backend = nullptr;

    for (const auto & backend : backends) {
        const char * backend_name = ggml_backend_name(backend.get());
        if (backend_name == nullptr) {
            continue;
        }

        if (qnn_aot_enabled && std::strcmp(backend_name, "qnn-npu") == 0) {
            qnn_aot_backend = backend.get();
        }
        if (std::strcmp(backend_name, "qnn-gpu") == 0) {
            qnn_gpu_backend = backend.get();
        }
        if (std::strcmp(backend_name, "qnn-cpu") == 0) {
            qnn_cpu_backend = backend.get();
        }
        if (std::strcmp(backend_name, "OpenCL") == 0) {
            opencl_backend = backend.get();
        }
    }

    const auto parse_hetero_backend = [&](const std::string & value) -> ggml_backend_t {
        if (value.empty()) {
            return nullptr;
        }

        const std::string normalized = llama_hetero_canonical_backend(value);
        if (normalized == "cpu") {
            return backend_cpu;
        }
        if (normalized == "opencl") {
            return opencl_backend;
        }
        if (normalized == "qnn-npu") {
            return qnn_aot_backend;
        }
        if (normalized == "qnn-gpu") {
            return qnn_gpu_backend;
        }
        if (normalized == "qnn-cpu") {
            return qnn_cpu_backend;
        }
        return nullptr;
    };

    const auto & hetero_route = hetero_plan.route;
    const std::string hetero_phase_backend_name = llama_hetero_phase_backend_for_route(hetero_route);
    const std::string hetero_output_backend_name = llama_hetero_phase_output_tail_backend_for_route(hetero_route);
    const ggml_backend_t hetero_phase_backend = parse_hetero_backend(hetero_phase_backend_name);
    const ggml_backend_t hetero_output_backend = parse_hetero_backend(hetero_output_backend_name);
    const bool hetero_stage_enabled = hetero_route.has_any_route();
    const bool hetero_route_uses_opencl = llama_hetero_is_opencl_backend(hetero_phase_backend_name);

    if (hetero_stage_enabled) {
        const auto value_or = [](const std::string & value) -> const char * {
            return value.empty() ? "<unset>" : value.c_str();
        };

        static bool logged_hetero_route_api = false;
        if (!logged_hetero_route_api) {
            LLAMA_LOG_INFO("%s: phase-only hetero route backend=%s route=%s\n",
                    __func__,
                    value_or(hetero_phase_backend_name),
                    llama_hetero_format_route_spec(hetero_route).c_str());
            logged_hetero_route_api = true;
        }
        static bool logged_hetero_output_route = false;
        if (hetero_output_backend != nullptr && !logged_hetero_output_route) {
            LLAMA_LOG_INFO("%s: routing decode output tail (norm/result_norm/result_output) via %s\n",
                    __func__, ggml_backend_name(hetero_output_backend));
            logged_hetero_output_route = true;
        }
        static bool logged_hetero_output_default_route = false;
        if (hetero_route_uses_opencl && hetero_output_backend == nullptr && !logged_hetero_output_default_route) {
            LLAMA_LOG_INFO("%s: leaving decode output tail (norm/result_norm/result_output) to default scheduler for OpenCL route\n",
                    __func__);
            logged_hetero_output_default_route = true;
        }
        static bool warned_hetero_attn_kv_boundary = false;
        static bool logged_hetero_attn_kv_contract = false;
        if (hetero_plan.attn_kv.stage_boundary_active() && !logged_hetero_attn_kv_contract) {
            LLAMA_LOG_INFO("%s: attn KV contract layout=%s transfer=%s zero_copy=%s available=%s reason=%s\n",
                    __func__,
                    llama_hetero_kv_layout_name(hetero_kv_contract_allocated.layout),
                    llama_hetero_kv_transfer_mode_name(hetero_kv_contract_allocated.transfer),
                    hetero_kv_contract_allocated.zero_copy ? "true" : "false",
                    hetero_kv_contract_allocated.buffer_available ? "true" : "false",
                    hetero_kv_contract_allocated.reason.empty() ? "<none>" : hetero_kv_contract_allocated.reason.c_str());
            logged_hetero_attn_kv_contract = true;
        }
        if (hetero_plan.attn_kv.stage_boundary_active() &&
            !hetero_kv_contract_allocated.is_split_safe() &&
            !warned_hetero_attn_kv_boundary) {
            LLAMA_LOG_WARN("%s: attn_proj and attn_core are split across %s -> %s, but the allocated KV contract is not zero-copy safe (layout=%s transfer=%s reason=%s). Keep attn_proj and attn_core on the same backend or rebuild the context with a compatible shared KV contract.\n",
                    __func__,
                    hetero_plan.attn_kv.producer_backend.c_str(),
                    hetero_plan.attn_kv.consumer_backend.c_str(),
                    llama_hetero_kv_layout_name(hetero_kv_contract_allocated.layout),
                    llama_hetero_kv_transfer_mode_name(hetero_kv_contract_allocated.transfer),
                    hetero_kv_contract_allocated.reason.empty() ? "<none>" : hetero_kv_contract_allocated.reason.c_str());
            warned_hetero_attn_kv_boundary = true;
        }
    }

    return [&, qnn_aot_enabled, qnn_aot_backend, qnn_gpu_backend, qnn_cpu_backend,
            hetero_stage_enabled, hetero_route_uses_opencl,
            hetero_phase_backend, hetero_output_backend](const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }

        const char * tensor_name = ggml_get_name(cur);
        const auto trace_assign_enabled = []() {
            const char * value = std::getenv("GGML_QNN_AOT_TRACE_ASSIGN");
            return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
        };
        const auto trace_tensor = [&](const char * reason, ggml_backend_t backend, bool supported) {
            if (!trace_assign_enabled()) {
                return;
            }

            if (tensor_name == nullptr) {
                return;
            }

            if (!llama_hetero_is_stage_tensor_name(tensor_name)) {
                return;
            }

            std::fprintf(stderr,
                         "[aot-assign] name=%s reason=%s backend=%s supported=%d\n",
                         tensor_name,
                         reason ? reason : "<null>",
                         backend ? ggml_backend_name(backend) : "<null>",
                         (int) supported);
        };
        const auto set_tensor_backend = [&](ggml_backend_t backend, bool pinned) {
            if (pinned) {
                ggml_backend_sched_set_tensor_backend_pinned(sched.get(), cur, backend);
            } else {
                ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend);
            }
        };

        // norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
        // FIXME: fix in ggml_backend_sched
        const bool full_offload = model.n_gpu_layers() > model.hparams.n_layer;
        if (ubatch.n_tokens < 32 || full_offload) {
            if (il != -1 && strcmp(name, "norm") == 0) {
                const auto & dev_layer = model.dev_layer(il);
                for (const auto & backend : backends) {
                    if (ggml_backend_get_device(backend.get()) == dev_layer) {
                        if (ggml_backend_supports_op(backend.get(), cur)) {
                            set_tensor_backend(backend.get(), false);
                        }
                    }
                }
            }
        }

        const auto tensor_has_ffn_input_ancestor = [&](const ggml_tensor * tensor, int depth, const auto & self) -> bool {
            if (tensor == nullptr || depth < 0) {
                return false;
            }

            const char * src_name = ggml_get_name(tensor);
            if (llama_hetero_name_has_prefix(src_name, "ffn_inp-")) {
                return true;
            }

            for (int i = 0; i < GGML_MAX_SRC; ++i) {
                if (self(tensor->src[i], depth - 1, self)) {
                    return true;
                }
            }

            return false;
        };

        const bool generic_norm_tensor = tensor_name != nullptr && (
            llama_hetero_name_has_prefix(tensor_name, "norm-") ||
            llama_hetero_name_has_prefix(tensor_name, "norm_w-"));
        const bool ffn_lineage_norm = generic_norm_tensor && tensor_has_ffn_input_ancestor(cur->src[0], 2, tensor_has_ffn_input_ancestor);

        const bool output_stage = llama_hetero_is_output_tensor_name(tensor_name);
        const bool route_output_to_backend = output_stage && hetero_output_backend != nullptr;

        const bool attn_proj_stage = llama_hetero_is_attn_proj_tensor_name(tensor_name) && !ffn_lineage_norm;
        const bool attn_core_stage = llama_hetero_is_attn_core_tensor_name(tensor_name);
        const bool attn_out_stage  = llama_hetero_is_attn_out_tensor_name(tensor_name);
        const bool attn_stage      = attn_proj_stage || attn_core_stage || attn_out_stage;
        const bool ffn_stage       = llama_hetero_is_ffn_tensor_name(tensor_name) || ffn_lineage_norm;
        const bool aot_transformer_stage = attn_stage || ffn_stage;
        const bool aot_lm_head_stage = tensor_name != nullptr && (
            std::strcmp(tensor_name, "norm") == 0 ||
            std::strcmp(tensor_name, "result_norm") == 0 ||
            std::strcmp(tensor_name, "result_output") == 0);
        const bool defer_opencl_output_tail_to_scheduler =
            hetero_route_uses_opencl &&
            hetero_output_backend == nullptr &&
            output_stage;
        const bool explicit_hetero_stage =
            (route_output_to_backend && hetero_output_backend != nullptr) ||
            ((attn_stage || ffn_stage) && hetero_phase_backend != nullptr);
        const bool preserve_stage_purity_on_cpu =
            hetero_stage_enabled &&
            !hetero_route_uses_opencl &&
            backend_cpu != nullptr &&
            llama_hetero_is_stage_tensor_name(tensor_name);

        auto resolve_stage_backend = [&]() -> ggml_backend_t {
            if (route_output_to_backend && hetero_output_backend != nullptr) {
                return hetero_output_backend;
            }
            if ((attn_stage || ffn_stage) && hetero_phase_backend != nullptr) {
                return hetero_phase_backend;
            }
            if (defer_opencl_output_tail_to_scheduler) {
                return nullptr;
            }
            if (qnn_aot_enabled && qnn_aot_backend != nullptr &&
                (!hetero_stage_enabled || llama_hetero_is_qnn_backend(hetero_phase_backend_name)) &&
                (aot_transformer_stage || aot_lm_head_stage)) {
                return qnn_aot_backend;
            }
            return nullptr;
        };

        const auto stage_backend_is_qnn = [&](ggml_backend_t backend) {
            return backend != nullptr &&
                (backend == qnn_aot_backend ||
                 backend == qnn_gpu_backend ||
                 backend == qnn_cpu_backend);
        };

        const bool correction_force_candidate =
            aot_transformer_stage ||
            aot_lm_head_stage ||
            (tensor_name != nullptr && (
                std::strcmp(tensor_name, "inp_tokens") == 0 ||
                std::strcmp(tensor_name, "embd") == 0));

        if (aot_force_cpu_graph && backend_cpu != nullptr && correction_force_candidate) {
            const ggml_backend_t stage_backend = resolve_stage_backend();
            const bool keep_stage_backend_for_offloaded_weights =
                model.n_gpu_layers() > 0 &&
                stage_backend != nullptr &&
                stage_backend != backend_cpu &&
                !stage_backend_is_qnn(stage_backend);

            if (!keep_stage_backend_for_offloaded_weights) {
                set_tensor_backend(backend_cpu, false);
                trace_tensor(stage_backend_is_qnn(stage_backend) ? "bootstrap-qnn-cpu" : "bootstrap-cpu",
                             backend_cpu,
                             true);
                return;
            }
        }

        const bool hetero_force_cpu = tensor_name != nullptr && (
            std::strcmp(tensor_name, "inp_tokens") == 0 ||
            std::strcmp(tensor_name, "embd") == 0 ||
            (std::strcmp(tensor_name, "norm") == 0 && !defer_opencl_output_tail_to_scheduler) ||
            (!route_output_to_backend && output_stage && !defer_opencl_output_tail_to_scheduler));

        const bool qnn_force_cpu = tensor_name != nullptr && (
            std::strcmp(tensor_name, "inp_tokens") == 0 ||
            std::strcmp(tensor_name, "embd") == 0 ||
            ((qnn_aot_backend == nullptr || !qnn_aot_enabled) && aot_lm_head_stage));

        if (qnn_aot_enabled && qnn_aot_backend != nullptr) {
            if (qnn_force_cpu) {
                set_tensor_backend(backend_cpu, explicit_hetero_stage);
                trace_tensor("force-cpu", backend_cpu, true);
                return;
            }

            ggml_backend_t target_backend = resolve_stage_backend();
            if (target_backend != nullptr) {
                const bool supported = ggml_backend_supports_op(target_backend, cur);
                if (supported) {
                    set_tensor_backend(target_backend, explicit_hetero_stage);
                    trace_tensor(target_backend == qnn_aot_backend ? "aot-qnn" : "hetero-stage", target_backend, true);
                } else if (preserve_stage_purity_on_cpu) {
                    set_tensor_backend(backend_cpu, true);
                    trace_tensor("hetero-unsupported-cpu-fallback", backend_cpu, true);
                } else {
                    trace_tensor(target_backend == qnn_aot_backend ? "aot-unsupported" : "hetero-unsupported", target_backend, false);
                }
                return;
            }

            if (preserve_stage_purity_on_cpu) {
                set_tensor_backend(backend_cpu, true);
                trace_tensor("hetero-purity-cpu", backend_cpu, true);
                return;
            }

            return;
        }

        if (!hetero_stage_enabled) {
            return;
        }

        if (hetero_force_cpu) {
            set_tensor_backend(backend_cpu, true);
            trace_tensor("hetero-force-cpu", backend_cpu, true);
            return;
        }

        ggml_backend_t target_backend = resolve_stage_backend();
        if (target_backend != nullptr) {
            const bool supported = ggml_backend_supports_op(target_backend, cur);
            if (supported) {
                set_tensor_backend(target_backend, true);
                trace_tensor("hetero-stage", target_backend, true);
            } else if (preserve_stage_purity_on_cpu) {
                set_tensor_backend(backend_cpu, true);
                trace_tensor("hetero-unsupported-cpu-fallback", backend_cpu, true);
            } else {
                trace_tensor("hetero-unsupported", target_backend, false);
            }
            return;
        }

        if (preserve_stage_purity_on_cpu) {
            set_tensor_backend(backend_cpu, true);
            trace_tensor("hetero-purity-cpu", backend_cpu, true);
            return;
        }

        trace_tensor("hetero-unmatched", nullptr, false);
    };
}


//
// state save/load
//

class llama_io_write_dummy : public llama_io_write_i {
public:
    llama_io_write_dummy() = default;

    void write(const void * /* src */, size_t size) override {
        size_written += size;
    }

    void write_tensor(const ggml_tensor * /* tensor */, size_t /* offset */, size_t size) override {
        size_written += size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    size_t size_written = 0;
};

class llama_io_write_buffer : public llama_io_write_i {
public:
    llama_io_write_buffer(
            uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(const ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        ggml_backend_tensor_get(tensor, ptr, offset, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;
};

class llama_io_read_buffer : public llama_io_read_i {
public:
    llama_io_read_buffer(const uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    const uint8_t * read(size_t size) override {
        const uint8_t * base_ptr = ptr;
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        ptr += size;
        size_read += size;
        buf_size -= size;
        return base_ptr;
    }

    void read_to(void * dst, size_t size) override {
        memcpy(dst, read(size), size);
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;
};

class llama_io_write_file : public llama_io_write_i {
public:
    llama_io_write_file(llama_file * f) : file(f) {}

    void write(const void * src, size_t size) override {
        file->write_raw(src, size);
        size_written += size;
    }

    void write_tensor(const ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        ggml_backend_tensor_get(tensor, temp_buffer.data(), offset, size);
        write(temp_buffer.data(), temp_buffer.size());
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    llama_file * file;
    size_t size_written = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_read_file : public llama_io_read_i {
public:
    llama_io_read_file(llama_file * f) : file(f) {}

    void read_to(void * dst, size_t size) override {
        file->read_raw(dst, size);
        size_read += size;
    }

    const uint8_t * read(size_t size) override {
        temp_buffer.resize(size);
        read_to(temp_buffer.data(), size);
        return temp_buffer.data();
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    llama_file * file;
    size_t size_read = 0;
    std::vector<uint8_t> temp_buffer;
};

size_t llama_context::state_get_size() {
    llama_io_write_dummy io;
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_get_data(uint8_t * dst, size_t size) {
    llama_io_write_buffer io(dst, size);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_set_data(const uint8_t * src, size_t size) {
    llama_io_read_buffer io(src, size);
    try {
        return state_read_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags) {
    llama_io_write_dummy io;
    try {
        return state_seq_write_data(io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_get_data(llama_seq_id seq_id, uint8_t * dst, size_t size, llama_state_seq_flags flags) {
    llama_io_write_buffer io(dst, size);
    try {
        return state_seq_write_data(io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags) {
    llama_io_read_buffer io(src, size);
    try {
        return state_seq_read_data(io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        return 0;
    }
}

bool llama_context::state_load_file(const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // sanity checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_SESSION_MAGIC || version != LLAMA_SESSION_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
            return false;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in session file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return false;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t n_state_size_cur = file.size() - file.tell();

        llama_io_read_file io( &file);
        const size_t n_read = state_read_data(io);

        if (n_read != n_state_size_cur) {
            LLAMA_LOG_ERROR("%s: did not read all of the session file data! size %zu, got %zu\n", __func__, n_state_size_cur, n_read);
            return false;
        }
    }

    return true;
}

bool llama_context::state_save_file(const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_SESSION_MAGIC);
    file.write_u32(LLAMA_SESSION_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_write_data(io);

    return true;
}

size_t llama_context::state_seq_load_file(llama_seq_id seq_id, const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // version checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_STATE_SEQ_MAGIC || version != LLAMA_STATE_SEQ_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for sequence state file: %08x, %08x\n", __func__, magic, version);
            return 0;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in sequence state file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return 0;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t state_size = file.size() - file.tell();
        llama_io_read_file io(&file);
        const size_t nread = state_seq_read_data(io, seq_id, 0);
        if (!nread) {
            LLAMA_LOG_ERROR("%s: failed to restore sequence state\n", __func__);
            return 0;
        }
        GGML_ASSERT(nread <= state_size);
        GGML_ASSERT(nread + sizeof(uint32_t) * 3 + sizeof(llama_token) * *n_token_count_out == file.tell());
    }

    return file.tell();
}

size_t llama_context::state_seq_save_file(llama_seq_id seq_id, const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_STATE_SEQ_MAGIC);
    file.write_u32(LLAMA_STATE_SEQ_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_seq_write_data(io, seq_id, 0);

    const size_t res = file.tell();
    GGML_ASSERT(res == sizeof(uint32_t) * 3 + sizeof(llama_token) * n_token_count + io.n_bytes());

    return res;
}

size_t llama_context::state_write_data(llama_io_write_i & io) {
    LLAMA_LOG_DEBUG("%s: writing state\n", __func__);

    // write model info
    {
        LLAMA_LOG_DEBUG("%s: - writing model info\n", __func__);

        const std::string arch_str = llm_arch_name(model.arch);
        io.write_string(arch_str);
        // TODO: add more model-specific info which should prevent loading the session file if not identical
    }

    if (memory != nullptr) {
        LLAMA_LOG_DEBUG("%s: - writing memory module\n", __func__);
        memory->state_write(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_read_data(llama_io_read_i & io) {
    LLAMA_LOG_DEBUG("%s: reading state\n", __func__);

    // read model info
    {
        LLAMA_LOG_DEBUG("%s: - reading model info\n", __func__);

        const std::string cur_arch_str = llm_arch_name(model.arch);

        std::string arch_str;
        io.read_string(arch_str);
        if (cur_arch_str != arch_str) {
            throw std::runtime_error(format("wrong model arch: '%s' instead of '%s'", arch_str.c_str(), cur_arch_str.c_str()));
        }
        // TODO: add more info which needs to be identical but which is not verified otherwise
    }

    if (memory) {
        LLAMA_LOG_DEBUG("%s: - reading memory module\n", __func__);

        memory->state_read(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(seq_id);

    if (memory) {
        memory->state_write(io, seq_id, flags);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_read_data(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(seq_id);

    if (memory) {
        memory->state_read(io, seq_id, flags);
    }

    return io.n_bytes();
}

bool llama_context::rebuild_dynamic_consumer_kv_from_state(
        const std::string & producer_backend,
        const std::string & consumer_backend,
        const char * reason) {
    if (memory == nullptr) {
        return true;
    }

    const std::string producer = llama_hetero_canonical_backend(producer_backend);
    const std::string consumer = llama_hetero_canonical_backend(consumer_backend);

    const bool producer_supported =
        producer == "cpu" || producer == "opencl" || llama_hetero_is_qnn_backend(producer);
    const bool consumer_supported = consumer == "cpu" || consumer == "opencl";

    if (!producer_supported || !consumer_supported || producer == consumer) {
        LLAMA_LOG_ERROR("%s: unsupported dynamic KV rebuild request producer=%s consumer=%s\n",
                __func__,
                producer.empty() ? "<unset>" : producer.c_str(),
                consumer.empty() ? "<unset>" : consumer.c_str());
        return false;
    }

    if (producer == "opencl" && !sync_dynamic_cpu_opencl_kv(/* host_to_device = */ false)) {
        LLAMA_LOG_ERROR("%s: failed to synchronize OpenCL-backed KV buffers before migration to %s\n",
                __func__,
                consumer.c_str());
        return false;
    }

    try {
        llama_io_write_dummy io_size;
        state_write_data(io_size);

        std::vector<uint8_t> state(io_size.n_bytes());
        llama_io_write_buffer io_write(state.data(), state.size());
        const size_t n_written = state_write_data(io_write);
        if (n_written != state.size()) {
            throw std::runtime_error(format("unexpected state size during CPU/OpenCL migration: %zu != %zu", n_written, state.size()));
        }

        llama_hetero_kv_contract migration_contract =
            llama_dynamic_phase_migration_kv_contract(producer, consumer, reason);

        llama_memory_params params_mem = {
            /*.type_k =*/ kv_type_k,
            /*.type_v =*/ kv_type_v,
            /*.swa_full =*/ kv_swa_full,
            /*.attn_v_trans =*/ kv_attn_v_trans,
            /*.attn_v_trans_pinned =*/ true,
            /*.kv_contract =*/ migration_contract,
        };

        llama_memory_ptr migrated_memory(model.create_memory(params_mem, cparams));
        if (!migrated_memory) {
            throw std::runtime_error("failed to create migrated memory module");
        }

        llama_memory_ptr old_memory = std::move(memory);
        memory = std::move(migrated_memory);

        try {
            llama_io_read_buffer io_read(state.data(), state.size());
            const size_t n_read = state_read_data(io_read);
            if (n_read != state.size()) {
                throw std::runtime_error(format("unexpected restored state size during CPU/OpenCL migration: %zu != %zu", n_read, state.size()));
            }
        } catch (...) {
            memory = std::move(old_memory);
            throw;
        }

        gf_res_prev.reset();
        gf_res_reserve.reset();
        aot_saved_sched.reset();
        hetero_dynamic_pre_reserved_plans.clear();
        sched_need_reserve = true;

        LLAMA_LOG_INFO("%s: rebuilt KV-backed memory for dynamic phase migration %s -> %s using consumer-owned placement (reason=%s)\n",
                __func__,
                producer.c_str(),
                consumer.c_str(),
                migration_contract.reason.c_str());
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: dynamic KV rebuild failed for %s -> %s: %s\n",
                __func__,
                producer.c_str(),
                consumer.c_str(),
                err.what());
        return false;
    }

    return true;
}

bool llama_context::migrate_dynamic_cpu_opencl_kv(
        const std::string & producer_backend,
        const std::string & consumer_backend) {
    const std::string producer = llama_hetero_canonical_backend(producer_backend);
    const std::string consumer = llama_hetero_canonical_backend(consumer_backend);

    if ((producer != "cpu" && producer != "opencl") ||
        (consumer != "cpu" && consumer != "opencl") ||
        producer == consumer) {
        LLAMA_LOG_ERROR("%s: unsupported CPU/OpenCL KV migration request producer=%s consumer=%s\n",
                __func__,
                producer.empty() ? "<unset>" : producer.c_str(),
                consumer.empty() ? "<unset>" : consumer.c_str());
        return false;
    }

    return rebuild_dynamic_consumer_kv_from_state(producer, consumer, "cpu-opencl-phase-migration");
}

bool llama_context::replay_dynamic_qnn_prefix() {
    if (dynamic_seq0_token_history.empty()) {
        qnn_prefix_replay_pending = false;
        qnn_prefix_replay_restore_plan_valid = false;
        qnn_prefix_replay_rebuild_live_memory = false;
        return true;
    }

    const std::string active_backend_name =
        llama_hetero_canonical_backend(hetero_plan.route.backend_for(llama_hetero_route_stage::ATTN_CORE));
    const bool replaying_into_qnn = hetero_route_requests_qnn(hetero_plan.route);
    const bool opencl_is_primary_backend =
        !model.devices.empty() &&
        model.devices[0] != nullptr &&
        std::strcmp(ggml_backend_dev_name(model.devices[0]), "GPUOpenCL") == 0;
    ggml_backend_t qnn_backend = replaying_into_qnn ? find_backend_for_route(active_backend_name) : nullptr;
    ggml_backend_qnn_aot_reset_state_t reset_state_fn = nullptr;
    llama_batch replay_batch = {};
    llama_memory_ptr saved_memory;
    bool replay_using_scratch_memory = false;
    const char * replay_prefill_route_env = std::getenv("GGML_HETERO_DYNAMIC_PREFILL_ROUTE");
    const char * replay_decode_route_env  = std::getenv("GGML_HETERO_DYNAMIC_DECODE_ROUTE");
    const bool replay_had_prefill_route_env = replay_prefill_route_env != nullptr;
    const bool replay_had_decode_route_env  = replay_decode_route_env != nullptr;
    const std::string replay_saved_prefill_route = replay_had_prefill_route_env ? replay_prefill_route_env : "";
    const std::string replay_saved_decode_route  = replay_had_decode_route_env  ? replay_decode_route_env  : "";

    const auto reset_replay_graph_state = [&]() {
        gf_res_prev.reset();
        gf_res_reserve.reset();
        aot_saved_sched.reset();
        hetero_dynamic_pre_reserved_plans.clear();
        sched_need_reserve = true;
    };

    const auto restore_replay_route_env = [&]() {
        if (replay_had_prefill_route_env) {
            setenv("GGML_HETERO_DYNAMIC_PREFILL_ROUTE", replay_saved_prefill_route.c_str(), 1);
        } else {
            unsetenv("GGML_HETERO_DYNAMIC_PREFILL_ROUTE");
        }

        if (replay_had_decode_route_env) {
            setenv("GGML_HETERO_DYNAMIC_DECODE_ROUTE", replay_saved_decode_route.c_str(), 1);
        } else {
            unsetenv("GGML_HETERO_DYNAMIC_DECODE_ROUTE");
        }
    };

    const auto restore_memory_after_replay = [&]() {
        if (!replay_using_scratch_memory) {
            return;
        }

        memory = std::move(saved_memory);
        replay_using_scratch_memory = false;
        reset_replay_graph_state();
    };

    const auto keep_replayed_memory = [&]() {
        if (!replay_using_scratch_memory) {
            return;
        }

        saved_memory.reset();
        replay_using_scratch_memory = false;
        reset_replay_graph_state();
    };

    auto cleanup_failure = [&](const char * reason) {
        if (reason != nullptr) {
            LLAMA_LOG_ERROR("%s: %s\n", __func__, reason);
        }
        if (replay_batch.token != nullptr || replay_batch.embd != nullptr || replay_batch.pos != nullptr ||
            replay_batch.n_seq_id != nullptr || replay_batch.seq_id != nullptr || replay_batch.logits != nullptr) {
            llama_batch_free(replay_batch);
        }
        dynamic_seq0_token_history.clear();
        qnn_prefix_replay_pending = false;
        qnn_prefix_replay_rebuild_live_memory = false;
        if (reset_state_fn != nullptr && qnn_backend != nullptr) {
            reset_state_fn(qnn_backend);
        }
        restore_replay_route_env();
        restore_memory_after_replay();
        if (qnn_prefix_replay_restore_plan_valid &&
            !apply_hetero_plan(qnn_prefix_replay_restore_plan, /* update_base_plan = */ false, "qnn-prefix-replay-revert")) {
            LLAMA_LOG_ERROR("%s: failed to restore the previous route after QNN replay failure\n", __func__);
        }
        qnn_prefix_replay_restore_plan_valid = false;
        qnn_prefix_replay_active = false;
        return false;
    };

    if (replaying_into_qnn) {
        if (qnn_backend == nullptr) {
            return cleanup_failure("failed to find the active qnn backend for prefix replay");
        }

        ggml_backend_dev_t qnn_dev = ggml_backend_get_device(qnn_backend);
        ggml_backend_reg_t qnn_reg = qnn_dev != nullptr ? ggml_backend_dev_backend_reg(qnn_dev) : nullptr;
        reset_state_fn =
            qnn_reg != nullptr
                ? (ggml_backend_qnn_aot_reset_state_t)
                      ggml_backend_reg_get_proc_address(qnn_reg, "ggml_backend_qnn_aot_reset_state")
                : nullptr;
        if (reset_state_fn == nullptr) {
            return cleanup_failure("qnn backend does not expose AoT reset_state support");
        }
    }

    LLAMA_LOG_INFO("%s: replaying %zu seq0 prefix token(s) on %s before the current decode token%s\n",
            __func__,
            dynamic_seq0_token_history.size(),
            active_backend_name.empty() ? "<active-route>" : active_backend_name.c_str(),
            qnn_prefix_replay_rebuild_live_memory ? " (rebuilding target KV state)" : "");

    if (n_queued_tokens > 0) {
        synchronize();
    }

    if (qnn_prefix_replay_rebuild_live_memory &&
        active_backend_name == "opencl" &&
        opencl_is_primary_backend) {
        LLAMA_LOG_INFO("%s: OpenCL is the primary backend for this context, keeping prefix replay on the decoder graph so OpenCL-native model weights stay in the fast path\n",
                __func__);
    }

    if (replaying_into_qnn && !reset_state_fn(qnn_backend)) {
        return cleanup_failure("failed to reset QNN AoT state before prefix replay");
    }

    try {
        llama_hetero_kv_contract replay_kv_contract = hetero_kv_contract_allocated;
        if (qnn_prefix_replay_rebuild_live_memory && !active_backend_name.empty()) {
            const std::string producer_backend =
                qnn_prefix_replay_restore_plan_valid
                    ? llama_hetero_canonical_backend(
                          qnn_prefix_replay_restore_plan.route.backend_for(llama_hetero_route_stage::ATTN_CORE))
                    : std::string("qnn-npu");
            replay_kv_contract = llama_dynamic_phase_migration_kv_contract(
                    producer_backend,
                    active_backend_name,
                    "qnn-prefix-replay-phase-migration");
        }

        llama_memory_params params_mem = {
            /*.type_k =*/ kv_type_k,
            /*.type_v =*/ kv_type_v,
            /*.swa_full =*/ kv_swa_full,
            /*.attn_v_trans =*/ kv_attn_v_trans,
            /*.attn_v_trans_pinned =*/ true,
            /*.kv_contract =*/ replay_kv_contract,
        };

        llama_memory_ptr scratch_memory(model.create_memory(params_mem, cparams));
        if (!scratch_memory) {
            return cleanup_failure("failed to create scratch memory for QNN prefix replay");
        }

        saved_memory = std::move(memory);
        memory = std::move(scratch_memory);
        replay_using_scratch_memory = true;
        reset_replay_graph_state();

        if (replaying_into_qnn) {
            LLAMA_LOG_INFO("%s: switched to empty scratch memory so QNN prefix replay re-materializes the tracked prefix without reusing the live generic KV state\n",
                    __func__);
        } else {
            LLAMA_LOG_INFO("%s: switched to empty scratch memory so prefix replay rebuilds the target backend KV state from the tracked seq0 history\n",
                    __func__);
        }

        // Replay should rebuild the active route exactly like a static run on
        // the destination backend. Temporarily hide the phase-switch env so the
        // replay does not trigger a second dynamic route decision mid-prefix.
        unsetenv("GGML_HETERO_DYNAMIC_PREFILL_ROUTE");
        unsetenv("GGML_HETERO_DYNAMIC_DECODE_ROUTE");
    } catch (const std::exception & err) {
        return cleanup_failure(err.what());
    }

    replay_batch = llama_batch_init((int32_t) dynamic_seq0_token_history.size(), 0, 1);
    if (replay_batch.token == nullptr || replay_batch.pos == nullptr ||
        replay_batch.n_seq_id == nullptr || replay_batch.seq_id == nullptr || replay_batch.logits == nullptr) {
        return cleanup_failure("failed to allocate replay batch buffers");
    }

    replay_batch.n_tokens = static_cast<int32_t>(dynamic_seq0_token_history.size());
    for (int32_t i = 0; i < replay_batch.n_tokens; ++i) {
        replay_batch.token[i] = dynamic_seq0_token_history[(size_t) i];
        replay_batch.pos[i] = i;
        replay_batch.n_seq_id[i] = 1;
        replay_batch.seq_id[i][0] = 0;
        replay_batch.logits[i] = 0;
    }

    llama_batch_allocr replay_balloc(model.hparams.n_pos_per_embd());
    if (!replay_balloc.init(
            replay_batch,
            model.vocab,
            nullptr,
            model.hparams.n_embd_inp(),
            cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max,
            /* output_all = */ false)) {
        return cleanup_failure("failed to initialize replay batch");
    }

    qnn_prefix_replay_active = true;
    if (replaying_into_qnn) {
        aot_skip_bootstrap_for_next_decode = true;
    }
    sched_reserve_request_tokens = static_cast<uint32_t>(replay_batch.n_tokens);
    sched_reserve();

    bool replay_did_optimize = false;
    llama_memory_context_ptr mctx;

    while (true) {
        mctx = memory->init_batch(replay_balloc, cparams.n_ubatch, /* embd_all = */ false);
        if (!mctx) {
            return cleanup_failure("failed to prepare replay memory context");
        }

        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                return cleanup_failure("unexpected replay memory status LLAMA_MEMORY_STATUS_NO_UPDATE");
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
                if (!replay_did_optimize) {
                    replay_did_optimize = true;
                    if (memory_update(true)) {
                        continue;
                    }
                }
                return cleanup_failure("failed to find a replay memory slot for the tracked prefix");
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                return cleanup_failure("memory computation failed while preparing the replay prefix");
        }

        break;
    }

    do {
        const auto & ubatch = mctx->get_ubatch();
        n_outputs = 0;

        ggml_status status;
        const auto * res = process_ubatch(ubatch, LLM_GRAPH_TYPE_DECODER, mctx.get(), status);
        if (!res) {
            llama_pos pos_min[LLAMA_MAX_SEQ];
            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                pos_min[s] = std::numeric_limits<llama_pos>::max();
            }

            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                const llama_seq_id seq_id = ubatch.seq_id[i][0];
                pos_min[seq_id] = std::min(pos_min[seq_id], ubatch.pos[i]);
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (pos_min[s] != std::numeric_limits<llama_pos>::max()) {
                    memory->seq_rm(s, pos_min[s], -1);
                }
            }

            switch (status) {
                case GGML_STATUS_ABORTED:
                    return cleanup_failure(replaying_into_qnn ? "QNN prefix replay was aborted" : "prefix replay was aborted");
                case GGML_STATUS_ALLOC_FAILED:
                    return cleanup_failure(replaying_into_qnn ? "QNN prefix replay failed during graph allocation" : "prefix replay failed during graph allocation");
                case GGML_STATUS_FAILED:
                    return cleanup_failure(replaying_into_qnn ? "QNN prefix replay graph execution failed" : "prefix replay graph execution failed");
                case GGML_STATUS_SUCCESS:
                    return cleanup_failure(replaying_into_qnn ? "QNN prefix replay returned an unexpected success state" : "prefix replay returned an unexpected success state");
            }
        }
    } while (mctx->next());

    ggml_backend_sched_synchronize(sched.get());
    restore_replay_route_env();
    if (qnn_prefix_replay_rebuild_live_memory) {
        keep_replayed_memory();
    } else {
        restore_memory_after_replay();
    }

    llama_batch_free(replay_batch);
    qnn_prefix_replay_pending = false;
    qnn_prefix_replay_restore_plan_valid = false;
    qnn_prefix_replay_rebuild_live_memory = false;
    qnn_prefix_replay_active = false;
    return true;
}

//
// perf
//

llama_perf_context_data llama_context::perf_get_data() const {
    llama_perf_context_data data = {};

    data.t_start_ms  = 1e-3 * t_start_us;
    data.t_load_ms   = 1e-3 * t_load_us;
    data.t_p_eval_ms = 1e-3 * t_p_eval_us;
    data.t_eval_ms   = 1e-3 * t_eval_us;
    data.n_p_eval    = std::max(1, n_p_eval);
    data.n_eval      = std::max(1, n_eval);
    data.n_reused    = std::max(0, n_reused);

    return data;
}

void llama_context::perf_reset() {
    hetero_decode_token_trace_dump();
    hetero_last_decode_token_done_us = 0;

    t_start_us  = ggml_time_us();
    t_eval_us   = n_eval = 0;
    t_p_eval_us = n_p_eval = 0;
    n_reused    = 0;
}

std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> llama_context::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> ret;
    for (const auto & [buft, size] : model.memory_breakdown()) {
        ret[buft].model += size;
    }
    if (memory) {
        for (const auto & [buft, size] : memory->memory_breakdown()) {
            ret[buft].context += size;
        }
    }
    if (model.hparams.no_alloc) {
        for (size_t i = 0; i < backends.size(); ++i) {
            ggml_backend_t             backend = backends[i].get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += backend_buf_exp_size[i];
        }
    } else {
        for (const auto & backend_ptr : backends) {
            ggml_backend_t             backend = backend_ptr.get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
    }
    return ret;
}

//
// training
//

static void llama_set_param(struct ggml_tensor * tensor, llama_opt_param_filter param_filter, void * userdata) {
    if (!tensor || tensor->type != GGML_TYPE_F32) {
        return;
    }
    if (!param_filter(tensor, userdata)) {
        return;
    }
    if (strcmp(tensor->name, "token_embd.weight") == 0) {
        return; // FIXME
    }
    if (strcmp(tensor->name, "rope_freqs.weight") == 0) {
        return; // FIXME
    }
    ggml_set_param(tensor);
}

void llama_context::opt_init(struct llama_model * model, struct llama_opt_params lopt_params) {
    GGML_ASSERT(!opt_ctx);
    model->hparams.n_ctx_train = lopt_params.n_ctx_train > 0 ? lopt_params.n_ctx_train : n_ctx();
    const uint32_t n_batch     = std::min(this->n_batch(),  model->hparams.n_ctx_train);
    const uint32_t n_ubatch    = std::min(this->n_ubatch(), n_batch);
    GGML_ASSERT(model->hparams.n_ctx_train % n_batch  == 0);
    GGML_ASSERT(n_batch                    % n_ubatch == 0);

    ggml_opt_params opt_params = ggml_opt_default_params(sched.get(), GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
    opt_params.opt_period      = n_batch / n_ubatch;
    opt_params.get_opt_pars    = lopt_params.get_opt_pars;
    opt_params.get_opt_pars_ud = lopt_params.get_opt_pars_ud;
    opt_params.optimizer       = lopt_params.optimizer_type;
    opt_ctx = ggml_opt_init(opt_params);

    llama_opt_param_filter param_filter = lopt_params.param_filter;
    void * param_filter_ud              = lopt_params.param_filter_ud;

  //llama_set_param(model->tok_embd,        param_filter, param_filter_ud); // FIXME
    llama_set_param(model->type_embd,       param_filter, param_filter_ud);
    llama_set_param(model->pos_embd,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm_b,      param_filter, param_filter_ud);
    llama_set_param(model->output_norm,     param_filter, param_filter_ud);
    llama_set_param(model->output_norm_b,   param_filter, param_filter_ud);
    llama_set_param(model->output,          param_filter, param_filter_ud);
    llama_set_param(model->output_b,        param_filter, param_filter_ud);
    llama_set_param(model->output_norm_enc, param_filter, param_filter_ud);
    llama_set_param(model->cls,             param_filter, param_filter_ud);
    llama_set_param(model->cls_b,           param_filter, param_filter_ud);
    llama_set_param(model->cls_out,         param_filter, param_filter_ud);
    llama_set_param(model->cls_out_b,       param_filter, param_filter_ud);
    llama_set_param(model->cls_norm,        param_filter, param_filter_ud);

    for (struct llama_layer & layer : model->layers) {
        for (size_t i = 0; i < sizeof(layer)/sizeof(struct ggml_tensor *); ++i) {
            llama_set_param(reinterpret_cast<struct ggml_tensor **>(&layer)[i], param_filter, param_filter_ud);
        }
    }
}

void llama_context::opt_epoch_iter(
        ggml_opt_dataset_t               dataset,
        ggml_opt_result_t                result,
        const std::vector<llama_token> & tokens,
        const std::vector<llama_token> & labels_sparse,
        llama_batch                    & batch,
        ggml_opt_epoch_callback          callback,
        bool                             train,
        int64_t                          idata_in_loop,
        int64_t                          ndata_in_loop,
        int64_t                          t_loop_start) {
    GGML_ASSERT(opt_ctx);
    const uint32_t n_ctx    = llama_model_n_ctx_train(&model);
    const uint32_t n_batch  = std::min(this->n_batch(),  n_ctx);
    const uint32_t n_ubatch = std::min(this->n_ubatch(), n_batch);

    memory->clear(true);

    for (uint32_t pos_ctx = 0; pos_ctx < n_ctx; pos_ctx += n_batch) {
        batch.n_tokens = n_batch;
        for (uint32_t pos_batch = 0; pos_batch < n_batch; ++pos_batch) {
            batch.token   [pos_batch]    = tokens[pos_ctx + pos_batch];
            batch.pos     [pos_batch]    = pos_ctx + pos_batch;
            batch.n_seq_id[pos_batch]    = 1;
            batch.seq_id  [pos_batch][0] = 0;
            batch.logits  [pos_batch]    = true;
        }

        if (!balloc->init(batch, model.vocab, nullptr, model.hparams.n_embd_inp(), cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max, true)) {
            LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
            return;
        }

        const uint32_t n_tokens_all = balloc->get_n_tokens();

        n_queued_tokens += n_tokens_all;

        embd_seq.clear();

        uint32_t n_outputs_all = n_tokens_all;

        auto mctx = memory->init_batch(*balloc, cparams.n_ubatch, true);
        if (!mctx || mctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: could not initialize batch\n", __func__);
            break;
        }

        // reserve output buffer
        if (output_reserve(n_outputs_all) < n_outputs_all) {
            LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
            GGML_ABORT("TODO: handle this error");
        };

        uint32_t pos_batch = 0;
        do {
            const auto & ubatch = mctx->get_ubatch();

            n_outputs = ubatch.n_tokens;

            if (!mctx->apply()) {
                LLAMA_LOG_ERROR("%s: failed to update the memory context\n", __func__);
                break;
            }

            auto * res = gf_res_prev.get();

            const auto gparams = graph_params(res, ubatch, mctx.get(), LLM_GRAPH_TYPE_DEFAULT);

            res->reset();

            auto * gf = model.build_graph(gparams);

            struct ggml_context * ctx_compute_opt;
            {
                const size_t size_gf = ggml_graph_size(gf);
                const size_t size_meta = 4*size_gf*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(size_gf, /*grads = */ true);
                struct ggml_init_params params = {
                    /*.mem_size   =*/ size_meta,
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ true,
                };
                ctx_compute_opt = ggml_init(params);
            }
            ggml_opt_prepare_alloc(opt_ctx, ctx_compute_opt, gf, res->get_inp_tokens(), res->get_logits());
            ggml_opt_alloc(opt_ctx, train);

            res->set_inputs(&ubatch);
            {
                struct ggml_tensor * labels = ggml_opt_labels(opt_ctx);
                GGML_ASSERT(labels->ne[1] == n_ubatch);
                ggml_set_zero(labels);
                const float onef = 1.0f;
                for (uint32_t pos_ubatch = 0; pos_ubatch < n_ubatch; ++pos_ubatch) {
                    const uint32_t ilabel = pos_ctx + pos_batch + pos_ubatch;
                    GGML_ASSERT(labels_sparse[ilabel] < labels->ne[0]);
                    ggml_backend_tensor_set(labels, &onef, (pos_ubatch*labels->ne[0] + labels_sparse[ilabel])*sizeof(float), sizeof(float));
                }
            }
            ggml_opt_eval(opt_ctx, result);
            if (callback) {
                callback(train, opt_ctx, dataset, result, idata_in_loop + (pos_ctx + pos_batch)/n_ubatch + 1, ndata_in_loop, t_loop_start);
            }
            ggml_free(ctx_compute_opt);

            pos_batch += ubatch.n_tokens;
        } while (mctx->next());
    }
}

void llama_context::opt_epoch(
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    const uint32_t n_ctx    = this->n_ctx();
    const uint32_t n_batch  = std::min(cparams.n_batch,  n_ctx);
    const uint32_t n_ubatch = std::min(cparams.n_ubatch, n_batch);
    const  int64_t ndata    = ggml_opt_dataset_ndata(dataset);

    GGML_ASSERT(idata_split >= 0);
    GGML_ASSERT(idata_split <= ndata);

    const uint32_t ubatch_per_ctx = n_ctx / n_ubatch;

    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);
    std::vector<llama_token>        tokens(n_ctx);
    std::vector<llama_token> labels_sparse(n_ctx);

    int64_t idata = 0;

    int64_t t_loop_start = ggml_time_us();
    int64_t ndata_in_loop = idata_split*ubatch_per_ctx;
    for (; idata < idata_split; ++idata) {
        constexpr bool train = true;
        const int64_t idata_in_loop = idata*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_train, tokens, labels_sparse, batch,
            callback_train, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    t_loop_start = ggml_time_us();
    ndata_in_loop = (ndata - idata_split)*ubatch_per_ctx;
    for (; idata < ndata; ++idata) {
        constexpr bool train = false;
        const int64_t idata_in_loop = (idata - idata_split)*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_eval, tokens, labels_sparse, batch,
            callback_eval, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    llama_batch_free(batch);
}

//
// interface implementation
//

llama_context_params llama_context_default_params() {
    llama_context_params result = {
        /*.n_ctx                       =*/ 512,
        /*.n_batch                     =*/ 2048,
        /*.n_ubatch                    =*/ 512,
        /*.n_seq_max                   =*/ 1,
        /*.n_threads                   =*/ GGML_DEFAULT_N_THREADS, // TODO: better default
        /*.n_threads_batch             =*/ GGML_DEFAULT_N_THREADS,
        /*.rope_scaling_type           =*/ LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
        /*.pooling_type                =*/ LLAMA_POOLING_TYPE_UNSPECIFIED,
        /*.attention_type              =*/ LLAMA_ATTENTION_TYPE_UNSPECIFIED,
        /*.flash_attn_type             =*/ LLAMA_FLASH_ATTN_TYPE_AUTO,
        /*.rope_freq_base              =*/ 0.0f,
        /*.rope_freq_scale             =*/ 0.0f,
        /*.yarn_ext_factor             =*/ -1.0f,
        /*.yarn_attn_factor            =*/ -1.0f,
        /*.yarn_beta_fast              =*/ -1.0f,
        /*.yarn_beta_slow              =*/ -1.0f,
        /*.yarn_orig_ctx               =*/ 0,
        /*.defrag_thold                =*/ -1.0f,
        /*.cb_eval                     =*/ nullptr,
        /*.cb_eval_user_data           =*/ nullptr,
        /*.type_k                      =*/ GGML_TYPE_F16,
        /*.type_v                      =*/ GGML_TYPE_F16,
        /*.abort_callback              =*/ nullptr,
        /*.abort_callback_data         =*/ nullptr,
        /*.hetero_phase_route          =*/ nullptr,
        /*.hetero_kv_layout            =*/ nullptr,
        /*.embeddings                  =*/ false,
        /*.offload_kqv                 =*/ true,
        /*.no_perf                     =*/ true,
        /*.op_offload                  =*/ true,
        /*.swa_full                    =*/ true,
        /*.kv_unified                  =*/ false,
        /*.sampler                     =*/ nullptr,
        /*.n_sampler                   =*/ 0,
    };

    return result;
}

llama_dynamic_route_config llama_dynamic_route_default_config() {
    llama_dynamic_route_config result = {
        /*.mode               =*/ "disabled",
        /*.prefill_route      =*/ nullptr,
        /*.prefill_kv_layout  =*/ nullptr,
        /*.decode_route       =*/ nullptr,
        /*.decode_kv_layout   =*/ nullptr,
        /*.fallback_route     =*/ nullptr,
        /*.fallback_kv_layout =*/ nullptr,
        /*.slo_us             =*/ 0,
        /*.allow_qnn          =*/ true,
        /*.decode_switch_after=*/ 0,
        /*.decode_gpu_freq_hz =*/ 0,
        /*.gpu_min_freq_path  =*/ nullptr,
        /*.gpu_max_freq_path  =*/ nullptr,
        /*.gpu_cur_freq_path  =*/ nullptr,
        /*.decode_cpu_freq_khz =*/ 0,
        /*.cpu_min_freq_path  =*/ nullptr,
        /*.cpu_max_freq_path  =*/ nullptr,
        /*.cpu_cur_freq_path  =*/ nullptr,
        /*.decode_cpu_affinity_mask =*/ nullptr,
        /*.decode_cpu_threads =*/ 0,
    };

    return result;
}

llama_context * llama_init_from_model(
                 llama_model * model,
        llama_context_params   params) {
    if (!model) {
        LLAMA_LOG_ERROR("%s: model cannot be NULL\n", __func__);
        return nullptr;
    }

    if (params.n_batch == 0 && params.n_ubatch == 0) {
        LLAMA_LOG_ERROR("%s: n_batch and n_ubatch cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.n_ctx == 0 && model->hparams.n_ctx_train == 0) {
        LLAMA_LOG_ERROR("%s: n_ctx and model->hparams.n_ctx_train cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && model->arch == LLM_ARCH_GROK) {
        LLAMA_LOG_WARN("%s: flash_attn is not compatible with Grok - forcing off\n", __func__);
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }

    if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO && ggml_is_quantized(params.type_k)) {
        const uint32_t blck_size = ggml_blck_size(params.type_k);
        for (uint32_t il = 0; il < model->hparams.n_layer; ++il) {
            if (model->hparams.n_embd_head_k(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: K cache type %s with block size %u does not divide n_embd_head_k=%u\n",
                    __func__, ggml_type_name(params.type_k), blck_size, model->hparams.n_embd_head_k(il));
                return nullptr;
            }
        }
    }

    if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO && ggml_is_quantized(params.type_v)) {
        const uint32_t blck_size = ggml_blck_size(params.type_v);
        for (uint32_t il = 0; il < model->hparams.n_layer; ++il) {
            if (model->hparams.n_embd_head_v(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: V cache type %s with block size %u does not divide n_embd_head_v=%u\n",
                    __func__, ggml_type_name(params.type_v), blck_size, model->hparams.n_embd_head_v(il));
                return nullptr;
            }
        }
    }

    if (ggml_is_quantized(params.type_v) && params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED) {
        LLAMA_LOG_ERROR("%s: V cache quantization requires flash_attn\n", __func__);
        return nullptr;
    }

    if (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED &&
        params.pooling_type != model->hparams.pooling_type) {
        //user-specified pooling-type is different from the model default
        LLAMA_LOG_WARN("%s: model default pooling_type is [%d], but [%d] was specified\n", __func__,
                       model->hparams.pooling_type, params.pooling_type);
    }

    try {
        auto * ctx = new llama_context(*model, params);
        return ctx;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to initialize the context: %s\n", __func__, err.what());
    }

    return nullptr;
}

// deprecated
llama_context * llama_new_context_with_model(
                 llama_model * model,
        llama_context_params   params) {
    return llama_init_from_model(model, params);
}

void llama_free(llama_context * ctx) {
    delete ctx;
}

uint32_t llama_n_ctx(const llama_context * ctx) {
    return ctx->n_ctx();
}

uint32_t llama_n_ctx_seq(const llama_context * ctx) {
    return ctx->n_ctx_seq();
}

uint32_t llama_n_batch(const llama_context * ctx) {
    return ctx->n_batch();
}

uint32_t llama_n_ubatch(const llama_context * ctx) {
    return ctx->n_ubatch();
}

uint32_t llama_n_seq_max(const llama_context * ctx) {
    return ctx->n_seq_max();
}

const llama_model * llama_get_model(const llama_context * ctx) {
    return &ctx->get_model();
}

enum llama_pooling_type llama_pooling_type(const llama_context * ctx) {
    return ctx->pooling_type();
}

void llama_attach_threadpool(
            llama_context * ctx,
        ggml_threadpool_t   threadpool,
        ggml_threadpool_t   threadpool_batch) {
    ctx->attach_threadpool(threadpool, threadpool_batch);
}

void llama_detach_threadpool(llama_context * ctx) {
    ctx->detach_threadpool();
}

void llama_set_n_threads(llama_context * ctx, int32_t n_threads, int32_t n_threads_batch) {
    ctx->set_n_threads(n_threads, n_threads_batch);
}

int32_t llama_n_threads(llama_context * ctx) {
    return ctx->n_threads();
}

int32_t llama_n_threads_batch(llama_context * ctx) {
    return ctx->n_threads_batch();
}

void llama_set_abort_callback(llama_context * ctx, bool (*abort_callback)(void * data), void * abort_callback_data) {
    ctx->set_abort_callback(abort_callback, abort_callback_data);
}

bool llama_set_hetero_phase_route(
        llama_context * ctx,
        const char * route_spec,
        const char * kv_layout) {
    if (ctx == nullptr) {
        LLAMA_LOG_ERROR("%s: ctx cannot be NULL\n", __func__);
        return false;
    }

    return ctx->set_hetero_plan(llama_hetero_build_execution_plan(route_spec, kv_layout));
}

int32_t llama_get_hetero_phase_route(
        const llama_context * ctx,
        char * buf,
        size_t buf_size) {
    if (ctx == nullptr) {
        if (buf != nullptr && buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }

    const std::string route = llama_hetero_format_route_spec(ctx->get_hetero_plan().route);
    const char * value = route.empty() ? "<default>" : route.c_str();
    if (buf == nullptr) {
        return int32_t(std::strlen(value));
    }
    return snprintf(buf, buf_size, "%s", value);
}

int32_t llama_get_hetero_kv_layout(
        const llama_context * ctx,
        char * buf,
        size_t buf_size) {
    if (ctx == nullptr) {
        if (buf != nullptr && buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }

    const auto & contract = ctx->get_hetero_plan().attn_kv;
    const char * value = nullptr;
    switch (contract.transfer) {
        case llama_hetero_kv_transfer_mode::NONE:
            value = contract.layout == llama_hetero_kv_layout_kind::LEGACY ? "legacy" : "stage-shared";
            break;
        case llama_hetero_kv_transfer_mode::CPU_OPENCL_ZERO_COPY:
            value = "cpu-opencl-zero-copy";
            break;
        case llama_hetero_kv_transfer_mode::QNN_RPCMEM:
            value = "qnn-rpcmem";
            break;
    }

    value = value != nullptr ? value : "unknown";
    if (buf == nullptr) {
        return int32_t(std::strlen(value));
    }
    return snprintf(buf, buf_size, "%s", value);
}

bool llama_set_dynamic_route_config(
        llama_context * ctx,
        llama_dynamic_route_config config) {
    if (ctx == nullptr) {
        LLAMA_LOG_ERROR("%s: ctx cannot be NULL\n", __func__);
        return false;
    }

    return ctx->set_dynamic_route_config(config);
}

int32_t llama_get_dynamic_route_mode(
        const llama_context * ctx,
        char * buf,
        size_t buf_size) {
    if (ctx == nullptr) {
        if (buf != nullptr && buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }

    const std::string mode = ctx->get_dynamic_route_mode();
    if (buf == nullptr) {
        return int32_t(mode.size());
    }
    return snprintf(buf, buf_size, "%s", mode.c_str());
}

void llama_set_embeddings(llama_context * ctx, bool embeddings) {
    ctx->set_embeddings(embeddings);
}

void llama_set_causal_attn(llama_context * ctx, bool causal_attn) {
    ctx->set_causal_attn(causal_attn);
}

void llama_set_warmup(llama_context * ctx, bool warmup) {
    ctx->set_warmup(warmup);
}

void llama_synchronize(llama_context * ctx) {
    ctx->synchronize();
}

float * llama_get_logits(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_logits();
}

float * llama_get_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    float * res = nullptr;

    res = ctx->get_sampled_logits_ith(i);

    if (!res) {
        res = ctx->get_logits_ith(i);
    }

    return res;
}

float * llama_get_embeddings(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings();
}

float * llama_get_embeddings_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_ith(i);
}

float * llama_get_embeddings_seq(llama_context * ctx, llama_seq_id seq_id) {
    ctx->synchronize();

    return ctx->get_embeddings_seq(seq_id);
}

bool llama_set_sampler(llama_context * ctx, llama_seq_id seq_id, llama_sampler * smpl) {
    return ctx->set_sampler(seq_id, smpl);
}

llama_token llama_get_sampled_token_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_token_ith(i);
}

float * llama_get_sampled_probs_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_probs_ith(i);
}

float * llama_get_sampled_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_logits_ith(i);
}

llama_token * llama_get_sampled_candidates_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return const_cast<llama_token *>(ctx->get_sampled_candidates_ith(i));
}

uint32_t llama_get_sampled_candidates_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_candidates_count(i));
}

uint32_t llama_get_sampled_logits_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_logits_count(i));
}

uint32_t llama_get_sampled_probs_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_probs_count(i));
}

struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs) {
    auto * memory = ctx->get_memory();
    llama_memory_context_ptr mctx;
    if (memory) {
        mctx = memory->init_full();
    }
    return ctx->graph_reserve(n_tokens, n_seqs, n_outputs, mctx.get());
}

// llama adapter API

int32_t llama_set_adapters_lora(
            llama_context * ctx,
            llama_adapter_lora ** adapters,
            size_t n_adapters,
            float * scales) {
    if (adapters == nullptr || scales == nullptr) {
        GGML_ASSERT(n_adapters == 0 && "invalid llama_set_adapters_lora call");
    }

    ctx->set_adapters_lora(adapters, n_adapters, scales);

    return 0;
}

int32_t llama_set_adapter_cvec(
        llama_context * ctx,
          const float * data,
               size_t   len,
              int32_t   n_embd,
              int32_t   il_start,
              int32_t   il_end) {
    bool res = ctx->set_adapter_cvec(data, len, n_embd, il_start, il_end);

    return res ? 0 : -1;
}

//
// memory
//

llama_memory_t llama_get_memory(const struct llama_context * ctx) {
    return ctx->get_memory();
}

void llama_memory_clear(llama_memory_t mem, bool data) {
    if (!mem) {
        return;
    }

    mem->clear(data);
}

bool llama_memory_seq_rm(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return true;
    }

    return mem->seq_rm(seq_id, p0, p1);
}

void llama_memory_seq_cp(
        llama_memory_t mem,
          llama_seq_id seq_id_src,
          llama_seq_id seq_id_dst,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return;
    }

    mem->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_seq_keep(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return;
    }

    mem->seq_keep(seq_id);
}

void llama_memory_seq_add(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
             llama_pos delta) {
    if (!mem) {
        return;
    }

    mem->seq_add(seq_id, p0, p1, delta);
}

void llama_memory_seq_div(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
                   int d) {
    if (!mem) {
        return;
    }

    mem->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_seq_pos_min(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_min(seq_id);
}

llama_pos llama_memory_seq_pos_max(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_max(seq_id);
}

bool llama_memory_can_shift(llama_memory_t mem) {
    if (!mem) {
        return false;
    }

    return mem->get_can_shift();
}

// llama state API

// deprecated
size_t llama_get_state_size(llama_context * ctx) {
    return llama_state_get_size(ctx);
}

// deprecated
size_t llama_copy_state_data(llama_context * ctx, uint8_t * dst) {
    return llama_state_get_data(ctx, dst, -1);
}

// deprecated
size_t llama_set_state_data(llama_context * ctx, const uint8_t * src) {
    return llama_state_set_data(ctx, src, -1);
}

// deprecated
bool llama_load_session_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    return llama_state_load_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
}

// deprecated
bool llama_save_session_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    return llama_state_save_file(ctx, path_session, tokens, n_token_count);
}

// Returns the *actual* size of the state.
// Intended to be used when saving to state to a buffer.
size_t llama_state_get_size(llama_context * ctx) {
    return ctx->state_get_size();
}

size_t llama_state_get_data(llama_context * ctx, uint8_t * dst, size_t size) {
    ctx->synchronize();

    return ctx->state_get_data(dst, size);
}

// Sets the state reading from the specified source address
size_t llama_state_set_data(llama_context * ctx, const uint8_t * src, size_t size) {
    ctx->synchronize();

    return ctx->state_set_data(src, size);
}

bool llama_state_load_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_load_file(path_session, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading session file: %s\n", __func__, err.what());
        return false;
    }
}

bool llama_state_save_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_save_file(path_session, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving session file: %s\n", __func__, err.what());
        return false;
    }
}

size_t llama_state_seq_get_size(llama_context * ctx, llama_seq_id seq_id) {
    return llama_state_seq_get_size_ext(ctx, seq_id, 0);
}

size_t llama_state_seq_get_data(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_get_data_ext(ctx, dst, size, seq_id, 0);
}

size_t llama_state_seq_set_data(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_set_data_ext(ctx, src, size, seq_id, 0);
}

size_t llama_state_seq_get_size_ext(llama_context * ctx, llama_seq_id seq_id, llama_state_seq_flags flags) {
    return ctx->state_seq_get_size(seq_id, flags);
}

size_t llama_state_seq_get_data_ext(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_get_data(seq_id, dst, size, flags);
}

size_t llama_state_seq_set_data_ext(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_set_data(seq_id, src, size, flags);
}

size_t llama_state_seq_save_file(llama_context * ctx, const char * filepath, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_seq_save_file(seq_id, filepath, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_state_seq_load_file(llama_context * ctx, const char * filepath, llama_seq_id dest_seq_id, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_seq_load_file(dest_seq_id, filepath, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

///

int32_t llama_encode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->encode(batch);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: failed to encode, ret = %d\n", __func__, ret);
    }

    return ret;
}

int32_t llama_decode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->decode(batch);
    if (ret != 0 && ret != 1) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }

    return ret;
}

//
// perf
//

llama_perf_context_data llama_perf_context(const llama_context * ctx) {
    llama_perf_context_data data = {};

    if (ctx == nullptr) {
        return data;
    }

    data = ctx->perf_get_data();

    return data;
}

void llama_perf_context_print(const llama_context * ctx) {
    const auto data = llama_perf_context(ctx);

    const double t_end_ms = 1e-3 * ggml_time_us();

    LLAMA_LOG_INFO("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
    LLAMA_LOG_INFO("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
    LLAMA_LOG_INFO("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
    LLAMA_LOG_INFO("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
    LLAMA_LOG_INFO("%s:    graphs reused = %10d\n", __func__, data.n_reused);
}

void llama_perf_context_reset(llama_context * ctx) {
    ctx->perf_reset();
}

void llama_memory_breakdown_print(const struct llama_context * ctx) {
    const std::vector<ggml_backend_dev_t> & devices = ctx->get_model().devices;

    std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> memory_breakdown = ctx->memory_breakdown();

    std::vector<std::array<std::string, 9>> table_data;
    table_data.reserve(devices.size());
    const std::string template_header = "%s: | %s | %s   %s    %s   %s   %s   %s    %s |\n";
    const std::string template_gpu    = "%s: | %s | %s = %s + (%s = %s + %s + %s) + %s |\n";
    const std::string template_other  = "%s: | %s | %s   %s    %s = %s + %s + %s    %s |\n";

    table_data.push_back({template_header, "memory breakdown [MiB]", "total", "free", "self", "model", "context", "compute", "unaccounted"});

    constexpr size_t MiB = 1024 * 1024;
    const std::vector<std::string> desc_prefixes_strip = {"NVIDIA ", "GeForce ", "Tesla ", "AMD ", "Radeon ", "Instinct "};

    // track seen buffer types to avoid double counting:
    std::set<ggml_backend_buffer_type_t> seen_buffer_types;

    // accumulative memory breakdown for each device and for host:
    std::vector<llama_memory_breakdown_data> mb_dev(devices.size());
    llama_memory_breakdown_data              mb_host;

    for (const auto & buft_mb : memory_breakdown) {
        ggml_backend_buffer_type_t          buft = buft_mb.first;
        const llama_memory_breakdown_data & mb   = buft_mb.second;
        if (ggml_backend_buft_is_host(buft)) {
            mb_host.model   += mb.model;
            mb_host.context += mb.context;
            mb_host.compute += mb.compute;
            seen_buffer_types.insert(buft);
            continue;
        }
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (dev) {
            int i_dev = -1;
            for (size_t i = 0; i < devices.size(); i++) {
                if (devices[i] == dev) {
                    i_dev = i;
                    break;
                }
            }
            if (i_dev != -1) {
                mb_dev[i_dev].model   += mb.model;
                mb_dev[i_dev].context += mb.context;
                mb_dev[i_dev].compute += mb.compute;
                seen_buffer_types.insert(buft);
                continue;
            }
        }
    }

    // print memory breakdown for each device:
    for (size_t i = 0; i < devices.size(); i++) {
        ggml_backend_dev_t          dev = devices[i];
        llama_memory_breakdown_data mb  = mb_dev[i];

        const std::string name = ggml_backend_dev_name(dev);
        std::string desc = ggml_backend_dev_description(dev);
        for (const std::string & prefix : desc_prefixes_strip) {
            if (desc.length() >= prefix.length() && desc.substr(0, prefix.length()) == prefix) {
                desc = desc.substr(prefix.length());
            }
        }

        size_t free, total;
        ggml_backend_dev_memory(dev, &free, &total);

        const size_t self = mb.model + mb.context + mb.compute;
        const size_t unaccounted = total - self - free;

        table_data.push_back({
            template_gpu,
            "  - " + name + " (" + desc + ")",
            std::to_string(total / MiB),
            std::to_string(free / MiB),
            std::to_string(self / MiB),
            std::to_string(mb.model / MiB),
            std::to_string(mb.context / MiB),
            std::to_string(mb.compute / MiB),
            std::to_string(unaccounted / MiB)});
    }

    // print memory breakdown for host:
    {
        const size_t self = mb_host.model + mb_host.context + mb_host.compute;
        table_data.push_back({
            template_other,
            "  - Host",
            "", // total
            "", // free
            std::to_string(self / MiB),
            std::to_string(mb_host.model / MiB),
            std::to_string(mb_host.context / MiB),
            std::to_string(mb_host.compute / MiB),
            ""}); // unaccounted
    }

    // print memory breakdown for all remaining buffer types:
    for (const auto & buft_mb : memory_breakdown) {
        ggml_backend_buffer_type_t          buft = buft_mb.first;
        const llama_memory_breakdown_data & mb   = buft_mb.second;
        if (seen_buffer_types.count(buft) == 1) {
            continue;
        }
        const std::string name = ggml_backend_buft_name(buft);
        const size_t self = mb.model + mb.context + mb.compute;
        table_data.push_back({
            template_other,
            "  - " + name,
            "", // total
            "", // free
            std::to_string(self / MiB),
            std::to_string(mb.model / MiB),
            std::to_string(mb.context / MiB),
            std::to_string(mb.compute / MiB),
            ""}); // unaccounted
        seen_buffer_types.insert(buft);
    }

    for (size_t j = 1; j < table_data[0].size(); j++) {
        size_t max_len = 0;
        for (const auto & td : table_data) {
            max_len = std::max(max_len, td[j].length());
        }
        for (auto & td : table_data) {
            td[j].insert(j == 1 ? td[j].length() : 0, max_len - td[j].length(), ' ');
        }
    }
    for (const auto & td : table_data) {
        LLAMA_LOG_INFO(td[0].c_str(),
            __func__, td[1].c_str(), td[2].c_str(), td[3].c_str(), td[4].c_str(), td[5].c_str(),
            td[6].c_str(), td[7].c_str(), td[8].c_str());
    }
}

//
// training
//

bool llama_opt_param_filter_all(const struct ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(tensor);
    GGML_UNUSED(userdata);
    return true;
}

void llama_opt_init(struct llama_context * ctx, struct llama_model * model, struct llama_opt_params lopt_params) {
    ctx->opt_init(model, lopt_params);
}

void llama_opt_epoch(
        struct llama_context    * ctx,
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    ctx->opt_epoch(
        dataset,
        result_train,
        result_eval,
        idata_split,
        callback_train,
        callback_eval);
}
