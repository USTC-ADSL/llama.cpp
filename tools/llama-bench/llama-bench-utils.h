#pragma once

#include <cstddef>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

struct llama_bench_round_reset_entry {
    std::string           backend_name;
    bool                  has_qnn_aot_reset = false;
    std::function<bool()> reset_qnn_aot_state;
};

struct llama_bench_round_reset_result {
    size_t                   eligible_backends = 0;
    size_t                   reset_backends    = 0;
    std::vector<std::string> failed_backends;

    bool ok() const {
        return failed_backends.empty();
    }
};

inline llama_bench_round_reset_result llama_bench_reset_qnn_aot_backends(
        const std::vector<llama_bench_round_reset_entry> & entries) {
    llama_bench_round_reset_result result;

    for (const auto & entry : entries) {
        if (!entry.has_qnn_aot_reset) {
            continue;
        }

        ++result.eligible_backends;

        if (entry.reset_qnn_aot_state && entry.reset_qnn_aot_state()) {
            ++result.reset_backends;
            continue;
        }

        result.failed_backends.push_back(entry.backend_name);
    }

    return result;
}

inline bool llama_bench_env_flag_enabled(const char * name) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return false;
    }

    const std::string text(value);
    return text != "0" && text != "false" && text != "FALSE" && text != "off" && text != "OFF" &&
           text != "no" && text != "NO";
}

inline bool llama_bench_qnn_decode_prewarm_enabled() {
    return llama_bench_env_flag_enabled("LLAMA_BENCH_QNN_PREWARM_DECODE");
}

inline bool llama_bench_qnn_depth_prewarm_enabled() {
    return llama_bench_env_flag_enabled("LLAMA_BENCH_QNN_PREWARM_DEPTH");
}

inline bool llama_bench_should_run_qnn_decode_prewarm(
        const std::vector<llama_bench_round_reset_entry> & entries,
        int                                                n_gen,
        bool                                               enabled) {
    if (!enabled || n_gen <= 0) {
        return false;
    }

    for (const auto & entry : entries) {
        if (entry.has_qnn_aot_reset) {
            return true;
        }
    }

    return false;
}

inline bool llama_bench_should_run_qnn_depth_prewarm(
        const std::vector<llama_bench_round_reset_entry> & entries,
        int                                                n_depth,
        bool                                               enabled) {
    if (!enabled || n_depth <= 0) {
        return false;
    }

    for (const auto & entry : entries) {
        if (entry.has_qnn_aot_reset) {
            return true;
        }
    }

    return false;
}

inline std::string llama_bench_format_round_event(
        size_t              benchmark_index,
        size_t              benchmark_count,
        int                 round_index,
        int                 reps,
        const std::string & event) {
    return "llama-bench: benchmark " + std::to_string(benchmark_index) + "/" + std::to_string(benchmark_count) +
           ": round " + std::to_string(round_index) + "/" + std::to_string(reps) + ": " + event;
}
