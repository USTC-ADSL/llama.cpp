#include "llama-dyn-route.h"

#include <array>
#include <cstdlib>

namespace {

std::string to_lower_trimmed(const char * value) {
    return llama_hetero_to_lower(llama_hetero_trim(value != nullptr ? value : ""));
}

bool env_flag_enabled(const char * name, bool default_value) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }

    return std::strcmp(value, "0") != 0;
}

int64_t env_i64_value(const char * name, int64_t default_value) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }

    return std::strtoll(value, nullptr, 10);
}

bool candidate_uses_backend(
        const llama_hetero_execution_plan & plan,
        bool (*predicate)(const std::string &)) {
    static constexpr std::array<llama_hetero_route_stage, 5> kStages = {{
        llama_hetero_route_stage::ATTN_PROJ,
        llama_hetero_route_stage::ATTN_CORE,
        llama_hetero_route_stage::ATTN_OUT,
        llama_hetero_route_stage::FFN,
        llama_hetero_route_stage::OUTPUT,
    }};

    for (const auto stage : kStages) {
        if (predicate(plan.route.backend_for(stage))) {
            return true;
        }
    }

    return predicate(plan.attn_kv.producer_backend) ||
           predicate(plan.attn_kv.consumer_backend) ||
           predicate(plan.attn_kv.storage_backend);
}

void set_candidate(
        llama_dynamic_route_candidate & candidate,
        const char * label,
        const char * route_spec,
        const char * kv_layout) {
    candidate.label = label != nullptr ? label : "";
    candidate.plan = llama_hetero_build_execution_plan(route_spec, kv_layout);
    candidate.configured =
        (route_spec != nullptr && route_spec[0] != '\0') ||
        (kv_layout  != nullptr && kv_layout [0] != '\0');
}

llama_dynamic_route_decision evaluate_plan(
        const llama_dynamic_route_runtime_config & config,
        const llama_dynamic_route_request & request,
        const llama_hetero_execution_plan & candidate_plan,
        const char * label,
        const char * success_reason) {
    llama_dynamic_route_decision decision;
    decision.plan       = candidate_plan;
    decision.plan_label = label != nullptr ? label : "<unnamed>";

    if (request.current_plan != nullptr &&
        llama_hetero_execution_plan_equals(*request.current_plan, candidate_plan)) {
        decision.reason = "already-active";
        return decision;
    }

    if (!config.allow_qnn && llama_dynamic_route_uses_qnn(candidate_plan)) {
        decision.reason = "qnn-disabled-by-config";
        return decision;
    }

    if (llama_dynamic_route_uses_opencl(candidate_plan) && !request.opencl_backend_available) {
        decision.reason = "opencl-backend-unavailable";
        return decision;
    }

    if (llama_dynamic_route_uses_qnn(candidate_plan) && !request.qnn_backend_available) {
        decision.reason = "qnn-backend-unavailable";
        return decision;
    }

    if (request.allocated_kv_contract != nullptr &&
        !llama_hetero_kv_contract_can_satisfy(*request.allocated_kv_contract, candidate_plan.attn_kv)) {
        decision.reason = "kv-contract-incompatible";
        return decision;
    }

    decision.should_apply = true;
    decision.reason       = success_reason != nullptr ? success_reason : "phase-selected";
    return decision;
}

} // namespace

const char * llama_dynamic_route_mode_name(llama_dynamic_route_mode mode) {
    switch (mode) {
        case llama_dynamic_route_mode::DISABLED:
            return "disabled";
        case llama_dynamic_route_mode::PHASE_HEURISTIC:
            return "phase-heuristic";
        case llama_dynamic_route_mode::COST_MODEL_RESERVED:
            return "cost-model";
    }

    return "unknown";
}

bool llama_dynamic_route_parse_mode(
        const char * value,
        llama_dynamic_route_mode & out_mode) {
    const std::string normalized = to_lower_trimmed(value);

    if (normalized.empty() ||
        normalized == "off" ||
        normalized == "disable" ||
        normalized == "disabled" ||
        normalized == "static") {
        out_mode = llama_dynamic_route_mode::DISABLED;
        return true;
    }

    if (normalized == "heuristic" ||
        normalized == "phase" ||
        normalized == "phase-heuristic") {
        out_mode = llama_dynamic_route_mode::PHASE_HEURISTIC;
        return true;
    }

    if (normalized == "cost" ||
        normalized == "cost-model" ||
        normalized == "slo-aware") {
        out_mode = llama_dynamic_route_mode::COST_MODEL_RESERVED;
        return true;
    }

    return false;
}

bool llama_dynamic_route_build_runtime_config(
        const llama_dynamic_route_config & public_config,
        llama_dynamic_route_runtime_config & out_config,
        std::string * error) {
    out_config = {};

    if (!llama_dynamic_route_parse_mode(public_config.mode, out_config.mode)) {
        if (error != nullptr) {
            *error = std::string("unknown dynamic route mode: ") +
                (public_config.mode != nullptr ? public_config.mode : "<null>");
        }
        return false;
    }

    set_candidate(out_config.prefill,  "prefill",  public_config.prefill_route,  public_config.prefill_kv_layout);
    set_candidate(out_config.decode,   "decode",   public_config.decode_route,   public_config.decode_kv_layout);
    set_candidate(out_config.fallback, "fallback", public_config.fallback_route, public_config.fallback_kv_layout);

    out_config.slo_us        = public_config.slo_us;
    out_config.allow_qnn     = public_config.allow_qnn;
    out_config.trace_enabled = false;

    return true;
}

llama_dynamic_route_runtime_config llama_dynamic_route_config_from_env() {
    llama_dynamic_route_runtime_config config;

    const char * mode_env = std::getenv("GGML_HETERO_DYNAMIC_MODE");
    const char * prefill_route_env = std::getenv("GGML_HETERO_DYNAMIC_PREFILL_ROUTE");
    const char * prefill_kv_env    = std::getenv("GGML_HETERO_DYNAMIC_PREFILL_KV");
    const char * decode_route_env  = std::getenv("GGML_HETERO_DYNAMIC_DECODE_ROUTE");
    const char * decode_kv_env     = std::getenv("GGML_HETERO_DYNAMIC_DECODE_KV");
    const char * fallback_route_env = std::getenv("GGML_HETERO_DYNAMIC_FALLBACK_ROUTE");
    const char * fallback_kv_env    = std::getenv("GGML_HETERO_DYNAMIC_FALLBACK_KV");

    if (mode_env == nullptr || mode_env[0] == '\0') {
        if ((prefill_route_env != nullptr && prefill_route_env[0] != '\0') ||
            (prefill_kv_env    != nullptr && prefill_kv_env   [0] != '\0') ||
            (decode_route_env  != nullptr && decode_route_env [0] != '\0') ||
            (decode_kv_env     != nullptr && decode_kv_env    [0] != '\0') ||
            (fallback_route_env != nullptr && fallback_route_env[0] != '\0') ||
            (fallback_kv_env    != nullptr && fallback_kv_env   [0] != '\0')) {
            config.mode = llama_dynamic_route_mode::PHASE_HEURISTIC;
        }
    } else {
        llama_dynamic_route_mode parsed_mode = llama_dynamic_route_mode::DISABLED;
        if (llama_dynamic_route_parse_mode(mode_env, parsed_mode)) {
            config.mode = parsed_mode;
        }
    }

    set_candidate(config.prefill,  "prefill",  prefill_route_env,  prefill_kv_env);
    set_candidate(config.decode,   "decode",   decode_route_env,   decode_kv_env);
    set_candidate(config.fallback, "fallback", fallback_route_env, fallback_kv_env);

    config.slo_us        = env_i64_value("GGML_HETERO_DYNAMIC_SLO_US", 0);
    config.allow_qnn     = env_flag_enabled("GGML_HETERO_DYNAMIC_ALLOW_QNN", true);
    config.trace_enabled = env_flag_enabled("GGML_HETERO_DYNAMIC_TRACE", false);

    return config;
}

bool llama_dynamic_route_uses_qnn(const llama_hetero_execution_plan & plan) {
    return candidate_uses_backend(plan, llama_hetero_is_qnn_backend);
}

bool llama_dynamic_route_uses_opencl(const llama_hetero_execution_plan & plan) {
    return candidate_uses_backend(plan, llama_hetero_is_opencl_backend);
}

llama_dynamic_route_decision llama_dynamic_route_decide(
        const llama_dynamic_route_runtime_config & config,
        const llama_dynamic_route_request & request) {
    llama_dynamic_route_decision decision;

    if (!config.enabled()) {
        decision.reason = "dynamic-routing-disabled";
        return decision;
    }

    if (config.mode == llama_dynamic_route_mode::COST_MODEL_RESERVED) {
        decision.reason = "cost-model-routing-reserved";
        return decision;
    }

    const bool is_prefill = request.n_tokens > 1;
    const auto & primary = is_prefill ? config.prefill : config.decode;

    if (primary.configured) {
        decision = evaluate_plan(
                config,
                request,
                primary.plan,
                primary.label.c_str(),
                is_prefill ? "phase-prefill-route" : "phase-decode-route");
        if (decision.should_apply) {
            return decision;
        }
    }

    if (config.fallback.configured) {
        decision = evaluate_plan(
                config,
                request,
                config.fallback.plan,
                config.fallback.label.c_str(),
                "phase-fallback-route");
        if (decision.should_apply) {
            return decision;
        }
    }

    if (request.base_plan != nullptr) {
        decision = evaluate_plan(
                config,
                request,
                *request.base_plan,
                "base",
                "phase-base-route");
        if (decision.should_apply) {
            return decision;
        }
    }

    if (!primary.configured) {
        decision.reason = is_prefill ? "prefill-route-unconfigured" : "decode-route-unconfigured";
    } else if (decision.reason.empty()) {
        decision.reason = "no-compatible-dynamic-route";
    }

    return decision;
}
