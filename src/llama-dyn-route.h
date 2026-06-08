#pragma once

#include "llama.h"
#include "llama-hetero-route.h"

#include <cstdint>
#include <string>

enum class llama_dynamic_route_mode {
    DISABLED,
    PHASE_HEURISTIC,
    COST_MODEL_RESERVED,
};

struct llama_dynamic_route_candidate {
    std::string                 label;
    llama_hetero_execution_plan plan;
    bool                        configured = false;
};

struct llama_dynamic_route_runtime_config {
    llama_dynamic_route_mode mode = llama_dynamic_route_mode::DISABLED;

    llama_dynamic_route_candidate prefill;
    llama_dynamic_route_candidate decode;
    llama_dynamic_route_candidate fallback;

    int64_t slo_us = 0;
    bool    allow_qnn = true;
    bool    trace_enabled = false;
    uint64_t decode_switch_after = 0;
    uint64_t decode_gpu_freq_hz = 0;
    std::string gpu_min_freq_path;
    std::string gpu_max_freq_path;
    std::string gpu_cur_freq_path;
    uint64_t decode_cpu_freq_khz = 0;
    std::string cpu_min_freq_path;
    std::string cpu_max_freq_path;
    std::string cpu_cur_freq_path;
    std::string decode_cpu_affinity_mask;
    int32_t decode_cpu_threads = 0;

    bool enabled() const {
        return mode != llama_dynamic_route_mode::DISABLED;
    }
};

struct llama_dynamic_route_runtime_state {
    uint64_t prefill_calls  = 0;
    uint64_t decode_calls   = 0;
    uint64_t route_switches = 0;
};

struct llama_dynamic_route_request {
    uint32_t n_tokens = 0;
    uint64_t decode_token_index = 0;

    bool opencl_backend_available = false;
    bool qnn_backend_available    = false;

    const llama_hetero_execution_plan * current_plan           = nullptr;
    const llama_hetero_execution_plan * base_plan              = nullptr;
    const llama_hetero_kv_contract *    allocated_kv_contract  = nullptr;
};

struct llama_dynamic_route_decision {
    bool should_apply = false;

    llama_hetero_execution_plan plan;
    std::string                 plan_label;
    std::string                 reason;
};

const char * llama_dynamic_route_mode_name(llama_dynamic_route_mode mode);

bool llama_dynamic_route_parse_mode(
        const char * value,
        llama_dynamic_route_mode & out_mode);

bool llama_dynamic_route_build_runtime_config(
        const llama_dynamic_route_config & public_config,
        llama_dynamic_route_runtime_config & out_config,
        std::string * error = nullptr);

llama_dynamic_route_runtime_config llama_dynamic_route_config_from_env();

bool llama_dynamic_route_uses_qnn(const llama_hetero_execution_plan & plan);
bool llama_dynamic_route_uses_opencl(const llama_hetero_execution_plan & plan);

llama_dynamic_route_decision llama_dynamic_route_decide(
        const llama_dynamic_route_runtime_config & config,
        const llama_dynamic_route_request & request);
