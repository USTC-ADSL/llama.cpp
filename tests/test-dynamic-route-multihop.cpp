#include "common.h"
#include "ggml.h"
#include "llama.h"
#include "testing.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

struct captured_logs {
    ggml_log_callback previous_callback = nullptr;
    void * previous_user_data = nullptr;
    std::string text;
};

struct top_logit {
    llama_token token;
    float logit;
};

struct semantic_snapshot {
    int decode_step = 0;
    std::vector<top_logit> top;
};

struct route_run_result {
    bool ok = false;
    std::string logs;
    std::string final_route;
    std::vector<semantic_snapshot> snapshots;
};

struct log_expectation {
    std::string needle;
    int min_count;
};

struct multihop_case_config {
    std::string schedule;
    std::string expected_schedule_log;
    std::string prompt;
    int n_ctx;
    int min_prompt_tokens;
    int decode_tokens;
    std::string expected_final_backend;
    int min_transition_count;
    std::vector<int> snapshot_steps;
    std::vector<log_expectation> fast_log_expectations;
};

static constexpr int64_t fast_transition_total_blocking_limit_us = 100000;

static void capture_log_callback(ggml_log_level level, const char * text, void * user_data) {
    auto * logs = static_cast<captured_logs *>(user_data);
    if (logs != nullptr && text != nullptr) {
        logs->text += text;
        if (logs->previous_callback != nullptr) {
            logs->previous_callback(level, text, logs->previous_user_data);
            return;
        }
    }

    if (text != nullptr) {
        std::fputs(text, stderr);
    }
}

static std::string get_model_path(int argc, char ** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            return argv[i + 1];
        }
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            return argv[i + 1];
        }
        if (argv[i][0] != '-') {
            return argv[i];
        }
    }

    const char * env_model = std::getenv("LLAMACPP_TEST_MODELFILE");
    return env_model != nullptr ? env_model : "";
}

static void force_qnn_prefill_multihop_schedule_env(const std::string & schedule, bool fast_kv_paths) {
    setenv("GGML_HETERO_DYNAMIC_MODE", "phase", 1);
    setenv("GGML_HETERO_DYNAMIC_PREFILL_ROUTE", "qnn-npu", 1);
    unsetenv("GGML_HETERO_DYNAMIC_DECODE_ROUTE");
    unsetenv("GGML_HETERO_DYNAMIC_DECODE_KV");
    unsetenv("GGML_HETERO_QNN_SHARED_HOST");
    setenv("GGML_HETERO_DYNAMIC_DECODE_SCHEDULE", schedule.c_str(), 1);
    setenv("GGML_HETERO_DYNAMIC_TRACE", "1", 1);
    setenv("GGML_HETERO_DYNAMIC_TRACE_TIMING", "1", 1);
    setenv("GGML_HETERO_DYNAMIC_TRACE_TIMING_DETAIL", "1", 1);
    if (fast_kv_paths) {
        setenv("GGML_QNN_AOT_WRITE_GENERIC_KV", "1", 1);
        unsetenv("GGML_HETERO_DISABLE_CPU_OPENCL_UMA_KV_HANDOFF");
        setenv("GGML_HETERO_ENABLE_OPENCL_CPU_UMA_KV_HANDOFF", "1", 1);
    } else {
        setenv("GGML_QNN_AOT_WRITE_GENERIC_KV", "0", 1);
        setenv("GGML_HETERO_DISABLE_CPU_OPENCL_UMA_KV_HANDOFF", "1", 1);
        unsetenv("GGML_HETERO_ENABLE_OPENCL_CPU_UMA_KV_HANDOFF");
    }
}

static std::vector<llama_token> tokenize_prompt(const llama_vocab * vocab, const std::string & prompt) {
    const int needed = -llama_tokenize(
            vocab,
            prompt.c_str(),
            static_cast<int32_t>(prompt.size()),
            nullptr,
            0,
            true,
            true);
    if (needed <= 0) {
        return {};
    }

    std::vector<llama_token> tokens(needed);
    const int n_tokens = llama_tokenize(
            vocab,
            prompt.c_str(),
            static_cast<int32_t>(prompt.size()),
            tokens.data(),
            static_cast<int32_t>(tokens.size()),
            true,
            true);
    if (n_tokens <= 0) {
        return {};
    }

    tokens.resize(n_tokens);
    return tokens;
}

static bool decode_tokens(
        llama_context * ctx,
        const std::vector<llama_token> & tokens,
        llama_pos start_pos) {
    llama_batch batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        common_batch_add(
                batch,
                tokens[i],
                start_pos + static_cast<llama_pos>(i),
                { 0 },
                i + 1 == tokens.size());
    }

    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return rc == 0;
}

static std::string current_route(llama_context * ctx) {
    const int32_t needed = llama_get_hetero_phase_route(ctx, nullptr, 0);
    if (needed < 0) {
        return {};
    }

    std::string route(static_cast<size_t>(needed) + 1, '\0');
    llama_get_hetero_phase_route(ctx, route.data(), route.size());
    route.resize(std::strlen(route.c_str()));
    return route;
}

static bool contains(const std::string & haystack, const char * needle) {
    return haystack.find(needle) != std::string::npos;
}

static bool route_matches_backend(const std::string & route, const std::string & backend) {
    if (!contains(route, backend.c_str())) {
        return false;
    }

    if (backend != "cpu" && contains(route, "cpu")) {
        return false;
    }
    if (backend != "opencl" && contains(route, "opencl")) {
        return false;
    }
    if (backend != "qnn" && contains(route, "qnn")) {
        return false;
    }
    return true;
}

static int count_occurrences(const std::string & haystack, const char * needle) {
    int count = 0;
    size_t pos = 0;
    const size_t needle_len = std::strlen(needle);
    while (needle_len > 0 && (pos = haystack.find(needle, pos)) != std::string::npos) {
        ++count;
        pos += needle_len;
    }
    return count;
}

static std::string transition_trace_line(
        const std::string & logs,
        int decode_token_index,
        int switch_after_tokens) {
    const std::string token_marker = "decode_token_index=" + std::to_string(decode_token_index);
    const std::string boundary_marker = "switch_after_tokens=" + std::to_string(switch_after_tokens);
    size_t pos = 0;
    while ((pos = logs.find("TRANSITION_TRACE", pos)) != std::string::npos) {
        const size_t end = logs.find('\n', pos);
        const std::string line = logs.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
        if (contains(line, token_marker.c_str()) && contains(line, boundary_marker.c_str())) {
            return line;
        }
        if (end == std::string::npos) {
            break;
        }
        pos = end + 1;
    }
    return {};
}

static std::string transition_trace_field(const std::string & line, const char * field_name) {
    const std::string prefix = std::string(field_name) + "=";
    const size_t value_start = line.find(prefix);
    if (value_start == std::string::npos) {
        return {};
    }

    const size_t start = value_start + prefix.size();
    const size_t end = line.find(' ', start);
    return line.substr(start, end == std::string::npos ? std::string::npos : end - start);
}

static int64_t transition_trace_field_i64(const std::string & line, const char * field_name) {
    const std::string value = transition_trace_field(line, field_name);
    if (value.empty()) {
        return -1;
    }

    char * end = nullptr;
    const long long parsed = std::strtoll(value.c_str(), &end, 10);
    if (end == value.c_str() || *end != '\0') {
        return -1;
    }

    return static_cast<int64_t>(parsed);
}

static void assert_fast_transition_trace(
        testing & t,
        const std::string & logs,
        int decode_token_index,
        int switch_after_tokens) {
    const std::string line = transition_trace_line(logs, decode_token_index, switch_after_tokens);
    const std::string label = "decode token " + std::to_string(decode_token_index) +
        " switch_after " + std::to_string(switch_after_tokens);
    if (!t.assert_true(label + " transition trace should exist", !line.empty())) {
        return;
    }

    t.assert_true(label + " transition trace should report success", contains(line, "success=1"));
    t.assert_true(label + " transition trace should not fall back", contains(line, "fallback=0"));
    t.assert_true(label + " transition trace should report ok support status", contains(line, "support_status=ok"));
    t.assert_true(label + " transition trace should avoid graph rebuild", contains(line, "graph_rebuild_us=0"));
}

static void assert_fast_transition_zero_kv_handoff(
        testing & t,
        const std::string & logs,
        int decode_token_index,
        int switch_after_tokens) {
    const std::string line = transition_trace_line(logs, decode_token_index, switch_after_tokens);
    const std::string label = "decode token " + std::to_string(decode_token_index) +
        " switch_after " + std::to_string(switch_after_tokens);
    if (!t.assert_true(label + " transition trace should exist", !line.empty())) {
        return;
    }

    t.assert_equal(
            label + " transition trace should report zero KV handoff",
            std::string("0"),
            transition_trace_field(line, "kv_handoff_us"));
}

static void assert_fast_transition_total_blocking_under_us(
        testing & t,
        const std::string & logs,
        int decode_token_index,
        int switch_after_tokens,
        int64_t max_total_blocking_us) {
    const std::string line = transition_trace_line(logs, decode_token_index, switch_after_tokens);
    const std::string label = "decode token " + std::to_string(decode_token_index) +
        " switch_after " + std::to_string(switch_after_tokens);
    if (!t.assert_true(label + " transition trace should exist", !line.empty())) {
        return;
    }

    const int64_t total_blocking_us = transition_trace_field_i64(line, "total_blocking_us");
    if (!t.assert_true(label + " transition trace should report total blocking time",
                total_blocking_us >= 0)) {
        return;
    }

    t.assert_true(
            label + " transition total blocking should stay below " +
            std::to_string(max_total_blocking_us) + " us (actual " +
            std::to_string(total_blocking_us) + " us)",
            total_blocking_us < max_total_blocking_us);
}

static bool should_capture_snapshot(const std::vector<int> & snapshot_steps, int decode_step) {
    return std::find(snapshot_steps.begin(), snapshot_steps.end(), decode_step) != snapshot_steps.end();
}

static multihop_case_config short_multihop_config() {
    return {
        /*.schedule =*/ "1:opencl;3:qnn-npu;5:cpu;7:opencl",
        /*.expected_schedule_log =*/ "decode_schedule=1:attn=opencl,ffn=opencl,output=opencl;3:attn=qnn-npu,ffn=qnn-npu,output=qnn-npu;5:attn=cpu,ffn=cpu,output=cpu;7:attn=opencl,ffn=opencl,output=opencl",
        /*.prompt =*/ "Mira fixed the bridge before sunrise and checked every cable.",
        /*.n_ctx =*/ 160,
        /*.min_prompt_tokens =*/ 8,
        /*.decode_tokens =*/ 7,
        /*.expected_final_backend =*/ "opencl",
        /*.min_transition_count =*/ 4,
        /*.snapshot_steps =*/ { 1, 2, 3, 4, 5, 6, 7 },
        /*.fast_log_expectations =*/ {
            { "completed direct shared QNN/OpenCL KV handoff", 1 },
            { "prepared direct generic KV import", 1 },
            { "using direct generic KV import", 1 },
            { "reusing QNN-written live generic KV directly", 1 },
            { "completed CPU/OpenCL UMA KV handoff", 1 },
        },
    };
}

static multihop_case_config interval32_multihop_config() {
    return {
        /*.schedule =*/ "1:cpu;33:opencl;65:qnn-npu;97:cpu",
        /*.expected_schedule_log =*/ "decode_schedule=1:attn=cpu,ffn=cpu,output=cpu;33:attn=opencl,ffn=opencl,output=opencl;65:attn=qnn-npu,ffn=qnn-npu,output=qnn-npu;97:attn=cpu,ffn=cpu,output=cpu",
        /*.prompt =*/ "Mira fixed the bridge before sunrise and checked every cable.",
        /*.n_ctx =*/ 160,
        /*.min_prompt_tokens =*/ 8,
        /*.decode_tokens =*/ 100,
        /*.expected_final_backend =*/ "cpu",
        /*.min_transition_count =*/ 4,
        /*.snapshot_steps =*/ { 1, 32, 33, 64, 65, 96, 97, 100 },
        /*.fast_log_expectations =*/ {
            { "reusing QNN-written live generic KV directly", 2 },
            { "completed CPU/OpenCL UMA KV handoff", 1 },
            { "prepared direct generic KV import", 1 },
            { "using direct generic KV import", 1 },
        },
    };
}

static multihop_case_config nonuniform_multihop_config(
        const std::string & prompt = "Mira fixed the bridge before sunrise and checked every cable.",
        int n_ctx = 160,
        int min_prompt_tokens = 8) {
    return {
        /*.schedule =*/ "1:cpu;5:opencl;14:qnn-npu;31:cpu",
        /*.expected_schedule_log =*/ "decode_schedule=1:attn=cpu,ffn=cpu,output=cpu;5:attn=opencl,ffn=opencl,output=opencl;14:attn=qnn-npu,ffn=qnn-npu,output=qnn-npu;31:attn=cpu,ffn=cpu,output=cpu",
        /*.prompt =*/ prompt,
        /*.n_ctx =*/ n_ctx,
        /*.min_prompt_tokens =*/ min_prompt_tokens,
        /*.decode_tokens =*/ 34,
        /*.expected_final_backend =*/ "cpu",
        /*.min_transition_count =*/ 4,
        /*.snapshot_steps =*/ { 1, 4, 5, 13, 14, 30, 31, 34 },
        /*.fast_log_expectations =*/ {
            { "reusing QNN-written live generic KV directly", 2 },
            { "completed CPU/OpenCL UMA KV handoff", 1 },
            { "prepared direct generic KV import", 1 },
            { "using direct generic KV import", 1 },
        },
    };
}

static multihop_case_config opencl_cpu_direct_multihop_config() {
    return {
        /*.schedule =*/ "1:opencl;9:cpu;17:opencl",
        /*.expected_schedule_log =*/ "decode_schedule=1:attn=opencl,ffn=opencl,output=opencl;9:attn=cpu,ffn=cpu,output=cpu;17:attn=opencl,ffn=opencl,output=opencl",
        /*.prompt =*/ "Mira fixed the bridge before sunrise and checked every cable.",
        /*.n_ctx =*/ 160,
        /*.min_prompt_tokens =*/ 8,
        /*.decode_tokens =*/ 20,
        /*.expected_final_backend =*/ "opencl",
        /*.min_transition_count =*/ 3,
        /*.snapshot_steps =*/ { 1, 8, 9, 16, 17, 20 },
        /*.fast_log_expectations =*/ {
            { "completed direct shared QNN/OpenCL KV handoff", 1 },
            { "completed CPU/OpenCL UMA KV handoff", 2 },
        },
    };
}

static multihop_case_config repeated_qnn_reentry_multihop_config() {
    return {
        /*.schedule =*/ "1:opencl;5:qnn-npu;9:opencl;13:qnn-npu;17:cpu",
        /*.expected_schedule_log =*/ "decode_schedule=1:attn=opencl,ffn=opencl,output=opencl;5:attn=qnn-npu,ffn=qnn-npu,output=qnn-npu;9:attn=opencl,ffn=opencl,output=opencl;13:attn=qnn-npu,ffn=qnn-npu,output=qnn-npu;17:attn=cpu,ffn=cpu,output=cpu",
        /*.prompt =*/ "Mira fixed the bridge before sunrise and checked every cable.",
        /*.n_ctx =*/ 160,
        /*.min_prompt_tokens =*/ 8,
        /*.decode_tokens =*/ 20,
        /*.expected_final_backend =*/ "cpu",
        /*.min_transition_count =*/ 5,
        /*.snapshot_steps =*/ { 1, 4, 5, 8, 9, 12, 13, 16, 17, 20 },
        /*.fast_log_expectations =*/ {
            { "completed direct shared QNN/OpenCL KV handoff", 2 },
            { "prepared direct generic KV import", 2 },
            { "using direct generic KV import", 2 },
            { "reusing QNN-written live generic KV directly", 1 },
        },
    };
}

static std::vector<top_logit> top_k_logits(const float * logits, int32_t n_vocab, size_t k) {
    if (logits == nullptr || n_vocab <= 0 || k == 0) {
        return {};
    }

    k = std::min(k, static_cast<size_t>(n_vocab));
    std::vector<llama_token> ids(static_cast<size_t>(n_vocab));
    std::iota(ids.begin(), ids.end(), 0);
    const auto better = [&](llama_token lhs, llama_token rhs) {
        return logits[lhs] > logits[rhs];
    };

    std::nth_element(ids.begin(), ids.begin() + static_cast<ptrdiff_t>(k - 1), ids.end(), better);
    ids.resize(k);
    std::sort(ids.begin(), ids.end(), better);

    std::vector<top_logit> top;
    top.reserve(k);
    for (const llama_token token : ids) {
        top.push_back({ token, logits[token] });
    }
    return top;
}

static int rank_of_token(const std::vector<top_logit> & top, llama_token token, size_t limit) {
    limit = std::min(limit, top.size());
    for (size_t i = 0; i < limit; ++i) {
        if (top[i].token == token) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

static int top_overlap(const std::vector<top_logit> & lhs, const std::vector<top_logit> & rhs, size_t limit) {
    limit = std::min({ limit, lhs.size(), rhs.size() });
    int overlap = 0;
    for (size_t i = 0; i < limit; ++i) {
        if (rank_of_token(rhs, lhs[i].token, limit) >= 0) {
            ++overlap;
        }
    }
    return overlap;
}

static route_run_result run_qnn_prefill_multihop_case(
        testing & t,
        captured_logs & logs,
        const std::string & model_path,
        const multihop_case_config & config,
        bool fast_kv_paths,
        bool assert_fast_path_logs) {
    route_run_result result;
    force_qnn_prefill_multihop_schedule_env(config.schedule, fast_kv_paths);
    const size_t run_log_start = logs.text.size();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = -1;
    mparams.hetero_phase_route = "qnn-npu";

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!t.assert_true("model should load", model != nullptr)) {
        return result;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = static_cast<uint32_t>(config.n_ctx);
    cparams.n_batch = 64;
    cparams.n_ubatch = 64;
    cparams.n_threads = 4;
    cparams.n_threads_batch = 4;
    cparams.hetero_phase_route = "qnn-npu";
    cparams.no_perf = true;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!t.assert_true("context should initialize with QNN prefill and scheduled decode routes", ctx != nullptr)) {
        llama_model_free(model);
        return result;
    }
    llama_set_warmup(ctx, false);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> prompt_tokens = tokenize_prompt(
            vocab,
            config.prompt);
    if (!t.assert_true("prompt should tokenize to the configured minimum length",
                prompt_tokens.size() >= static_cast<size_t>(config.min_prompt_tokens))) {
        llama_free(ctx);
        llama_model_free(model);
        return result;
    }

    if (!t.assert_true("QNN prefill should decode", decode_tokens(ctx, prompt_tokens, 0))) {
        llama_free(ctx);
        llama_model_free(model);
        return result;
    }

    for (int i = 0; i < config.decode_tokens; ++i) {
        const size_t token_offset = static_cast<size_t>(i) % prompt_tokens.size();
        const size_t token_index = (prompt_tokens.size() - 1 + prompt_tokens.size() - token_offset) % prompt_tokens.size();
        const llama_token token = prompt_tokens[token_index];
        const std::vector<llama_token> decode_token = { token };
        if (!t.assert_true("scheduled decode token should decode",
                    decode_tokens(ctx, decode_token, static_cast<llama_pos>(prompt_tokens.size() + (size_t) i)))) {
            llama_free(ctx);
            llama_model_free(model);
            return result;
        }

        const float * logits = llama_get_logits_ith(ctx, 0);
        if (!t.assert_true("scheduled decode token should expose logits", logits != nullptr)) {
            llama_free(ctx);
            llama_model_free(model);
            return result;
        }

        const int decode_step = i + 1;
        if (should_capture_snapshot(config.snapshot_steps, decode_step)) {
            result.snapshots.push_back({ decode_step, top_k_logits(logits, n_vocab, 64) });
        }
    }

    result.logs = logs.text.substr(run_log_start);
    result.final_route = current_route(ctx);
    result.ok = true;

    t.assert_true(
            "dynamic schedule should be active",
            contains(result.logs, config.expected_schedule_log.c_str()));
    if (assert_fast_path_logs) {
        for (const log_expectation & expectation : config.fast_log_expectations) {
            t.assert_true(
                    "expected fast-path log: " + expectation.needle,
                    count_occurrences(result.logs, expectation.needle.c_str()) >= expectation.min_count);
        }
    }
    t.assert_true(
            "expected route transitions should be traced",
            count_occurrences(result.logs, "TRANSITION_TRACE") >= config.min_transition_count);
    t.assert_true(
            "all traced transitions should report success",
            count_occurrences(result.logs, "success=1") >= config.min_transition_count);
    t.assert_equal(
            "expected logits snapshots should be captured",
            config.snapshot_steps.size(),
            result.snapshots.size());
    t.assert_true(
            "fast multihop route should not refuse QNN switching",
            !contains(result.logs, "refusing non-QNN -> qnn decode switch"));
    if (assert_fast_path_logs) {
        t.assert_true(
                "fast multihop route should not queue prefix replay",
                !contains(result.logs, "queued QNN prefix replay"));
        t.assert_true(
                "fast multihop route should not rebuild from state",
                !contains(result.logs, "rebuild_dynamic_consumer_kv_from_state"));
    }
    t.assert_true(
            "fast multihop route should not fall back",
            !contains(result.logs, "fallback=1"));
    t.assert_true(
            "fast multihop route should not hit unmatched QNN AoT graphs",
            !contains(result.logs, "unmatched cgraph"));
    t.assert_true(
            "fast multihop route should not hit CPU_REPACK weight readback abort",
            !contains(result.logs, "CPU_REPACK does not implement get_tensor"));
    t.assert_true(
            "fast multihop route should not crash",
            !contains(result.logs, "SIGSEGV") && !contains(result.logs, "Segmentation fault"));
    t.assert_true(
            "final scheduled route should match expected backend",
            route_matches_backend(result.final_route, config.expected_final_backend));

    llama_free(ctx);
    llama_model_free(model);
    return result;
}

static void assert_semantic_alignment(testing & t, const route_run_result & fast, const route_run_result & reference) {
    if (!t.assert_true("fast multihop run should complete", fast.ok) ||
        !t.assert_true("reference multihop run should complete", reference.ok)) {
        return;
    }

    if (!t.assert_equal("fast/reference runs should produce the same number of logits snapshots",
                reference.snapshots.size(), fast.snapshots.size())) {
        return;
    }

    for (size_t i = 0; i < fast.snapshots.size(); ++i) {
        const auto & fast_top = fast.snapshots[i].top;
        const auto & ref_top = reference.snapshots[i].top;
        if (!t.assert_equal("fast/reference snapshots should use the same decode step",
                    reference.snapshots[i].decode_step, fast.snapshots[i].decode_step)) {
            return;
        }
        if (!t.assert_true("fast logits snapshot should contain top candidates", !fast_top.empty()) ||
            !t.assert_true("reference logits snapshot should contain top candidates", !ref_top.empty())) {
            return;
        }

        const int ref_top1_rank_in_fast = rank_of_token(fast_top, ref_top[0].token, 64);
        const int fast_top1_rank_in_ref = rank_of_token(ref_top, fast_top[0].token, 64);
        const int overlap16 = top_overlap(fast_top, ref_top, 16);
        std::fprintf(stderr,
                "semantic alignment decode_step=%zu fast_top1=%d ref_top1=%d "
                "ref_top1_rank_in_fast=%d fast_top1_rank_in_ref=%d top16_overlap=%d\n",
                static_cast<size_t>(fast.snapshots[i].decode_step),
                fast_top[0].token,
                ref_top[0].token,
                ref_top1_rank_in_fast,
                fast_top1_rank_in_ref,
                overlap16);

        t.assert_true("reference top-1 token should remain in fast-path top-64 candidates",
                ref_top1_rank_in_fast >= 0);
        t.assert_true("fast-path top-1 token should remain in reference top-64 candidates",
                fast_top1_rank_in_ref >= 0);
        t.assert_true("fast/reference top-16 candidate sets should overlap",
                overlap16 >= 1);
    }
}

static void assert_semantic_candidate_overlap(testing & t, const route_run_result & fast, const route_run_result & reference) {
    if (!t.assert_true("fast multihop run should complete", fast.ok) ||
        !t.assert_true("reference multihop run should complete", reference.ok)) {
        return;
    }

    if (!t.assert_equal("fast/reference runs should produce the same number of logits snapshots",
                reference.snapshots.size(), fast.snapshots.size())) {
        return;
    }

    for (size_t i = 0; i < fast.snapshots.size(); ++i) {
        const auto & fast_top = fast.snapshots[i].top;
        const auto & ref_top = reference.snapshots[i].top;
        if (!t.assert_equal("fast/reference snapshots should use the same decode step",
                    reference.snapshots[i].decode_step, fast.snapshots[i].decode_step)) {
            return;
        }
        if (!t.assert_true("fast logits snapshot should contain top candidates", !fast_top.empty()) ||
            !t.assert_true("reference logits snapshot should contain top candidates", !ref_top.empty())) {
            return;
        }

        const int ref_top1_rank_in_fast = rank_of_token(fast_top, ref_top[0].token, 64);
        const int fast_top1_rank_in_ref = rank_of_token(ref_top, fast_top[0].token, 64);
        const int overlap16 = top_overlap(fast_top, ref_top, 16);
        const int overlap64 = top_overlap(fast_top, ref_top, 64);
        std::fprintf(stderr,
                "semantic candidate alignment decode_step=%zu fast_top1=%d ref_top1=%d "
                "ref_top1_rank_in_fast=%d fast_top1_rank_in_ref=%d top16_overlap=%d top64_overlap=%d\n",
                static_cast<size_t>(fast.snapshots[i].decode_step),
                fast_top[0].token,
                ref_top[0].token,
                ref_top1_rank_in_fast,
                fast_top1_rank_in_ref,
                overlap16,
                overlap64);

        t.assert_true("fast/reference top-16 candidate sets should overlap",
                overlap16 >= 1);
        t.assert_true("fast/reference top-64 candidate sets should retain semantic overlap",
                overlap64 >= 4);
    }
}

int main(int argc, char ** argv) {
    const std::string model_path = get_model_path(argc, argv);
    if (model_path.empty()) {
        std::fprintf(stderr,
                "\033[33mWARNING: No model file provided. Skipping this test. "
                "Pass -m <model> or set LLAMACPP_TEST_MODELFILE.\n\033[0m");
        return 0;
    }

    if (std::getenv("GGML_QNN_AOT_CONFIG") == nullptr) {
        std::fprintf(stderr,
                "\033[33mWARNING: GGML_QNN_AOT_CONFIG is not set. "
                "Skipping QNN dynamic-route multihop runtime test.\n\033[0m");
        return 0;
    }

    captured_logs logs;
    llama_log_get(&logs.previous_callback, &logs.previous_user_data);
    llama_log_set(capture_log_callback, &logs);

    ggml_backend_load_all();

    testing t;
    t.test("qnn prefill dynamic decode supports opencl qnn cpu opencl multihop", [&](testing & t) {
        const multihop_case_config config = short_multihop_config();
        run_qnn_prefill_multihop_case(
                t,
                logs,
                model_path,
                config,
                /* fast_kv_paths = */ true,
                /* assert_fast_path_logs = */ true);
    });

    t.test("fast multihop logits stay aligned with conservative migration", [&](testing & t) {
        const multihop_case_config config = short_multihop_config();
        const route_run_result fast = run_qnn_prefill_multihop_case(
                t,
                logs,
                model_path,
                config,
                /* fast_kv_paths = */ true,
                /* assert_fast_path_logs = */ true);
        const route_run_result reference = run_qnn_prefill_multihop_case(
                t,
                logs,
                model_path,
                config,
                /* fast_kv_paths = */ false,
                /* assert_fast_path_logs = */ false);
        assert_semantic_alignment(t, fast, reference);
    });

    t.test("qnn prefill dynamic decode supports 32 token cpu opencl qnn cpu intervals", [&](testing & t) {
        const multihop_case_config config = interval32_multihop_config();
        const route_run_result result = run_qnn_prefill_multihop_case(
                t,
                logs,
                model_path,
                config,
                /* fast_kv_paths = */ true,
                /* assert_fast_path_logs = */ true);
        if (!t.assert_true("32-token interval multihop run should complete", result.ok)) {
            return;
        }
        t.assert_true(
                "32-token interval schedule should switch at decode token 33",
                contains(result.logs, "decode_token_index=33 switch_after_tokens=32"));
        t.assert_true(
                "32-token interval schedule should switch at decode token 65",
                contains(result.logs, "decode_token_index=65 switch_after_tokens=64"));
        t.assert_true(
                "32-token interval schedule should switch at decode token 97",
                contains(result.logs, "decode_token_index=97 switch_after_tokens=96"));
        assert_fast_transition_trace(t, result.logs, 33, 32);
        assert_fast_transition_trace(t, result.logs, 65, 64);
        assert_fast_transition_trace(t, result.logs, 97, 96);
        assert_fast_transition_zero_kv_handoff(t, result.logs, 97, 96);
        assert_fast_transition_total_blocking_under_us(t, result.logs, 33, 32, fast_transition_total_blocking_limit_us);
        assert_fast_transition_total_blocking_under_us(t, result.logs, 97, 96, fast_transition_total_blocking_limit_us);
    });

    t.test("fast 32 token interval logits stay aligned with conservative migration", [&](testing & t) {
        const multihop_case_config config = interval32_multihop_config();
        const route_run_result fast = run_qnn_prefill_multihop_case(
                t,
                logs,
                model_path,
                config,
                /* fast_kv_paths = */ true,
                /* assert_fast_path_logs = */ true);
        const route_run_result reference = run_qnn_prefill_multihop_case(
                t,
                logs,
                model_path,
                config,
                /* fast_kv_paths = */ false,
                /* assert_fast_path_logs = */ false);
        assert_semantic_alignment(t, fast, reference);
    });

    t.test("fast nonuniform interval logits stay aligned with conservative migration", [&](testing & t) {
        const multihop_case_config config = nonuniform_multihop_config();
        const route_run_result fast = run_qnn_prefill_multihop_case(
                t,
                logs,
                model_path,
                config,
                /* fast_kv_paths = */ true,
                /* assert_fast_path_logs = */ true);
        const route_run_result reference = run_qnn_prefill_multihop_case(
                t,
                logs,
                model_path,
                config,
                /* fast_kv_paths = */ false,
                /* assert_fast_path_logs = */ false);
        assert_semantic_candidate_overlap(t, fast, reference);
        if (!t.assert_true("nonuniform interval fast run should complete", fast.ok)) {
            return;
        }
        t.assert_true(
                "nonuniform interval schedule should switch at decode token 5",
                contains(fast.logs, "decode_token_index=5 switch_after_tokens=4"));
        t.assert_true(
                "nonuniform interval schedule should switch at decode token 14",
                contains(fast.logs, "decode_token_index=14 switch_after_tokens=13"));
        t.assert_true(
                "nonuniform interval schedule should switch at decode token 31",
                contains(fast.logs, "decode_token_index=31 switch_after_tokens=30"));
        assert_fast_transition_trace(t, fast.logs, 5, 4);
        assert_fast_transition_trace(t, fast.logs, 14, 13);
        assert_fast_transition_trace(t, fast.logs, 31, 30);
        assert_fast_transition_zero_kv_handoff(t, fast.logs, 31, 30);
        assert_fast_transition_total_blocking_under_us(t, fast.logs, 5, 4, fast_transition_total_blocking_limit_us);
        assert_fast_transition_total_blocking_under_us(t, fast.logs, 31, 30, fast_transition_total_blocking_limit_us);
    });

    t.test("fast nonuniform interval logits stay aligned for alternate prompt", [&](testing & t) {
        const multihop_case_config config = nonuniform_multihop_config(
                "Nora logged the sensor readings before sunset and verified every checksum.");
        const route_run_result fast = run_qnn_prefill_multihop_case(
                t,
                logs,
                model_path,
                config,
                /* fast_kv_paths = */ true,
                /* assert_fast_path_logs = */ true);
        const route_run_result reference = run_qnn_prefill_multihop_case(
                t,
                logs,
                model_path,
                config,
                /* fast_kv_paths = */ false,
                /* assert_fast_path_logs = */ false);
        assert_semantic_candidate_overlap(t, fast, reference);
        if (!t.assert_true("alternate-prompt nonuniform interval fast run should complete", fast.ok)) {
            return;
        }
        t.assert_true(
                "alternate-prompt nonuniform interval schedule should switch at decode token 5",
                contains(fast.logs, "decode_token_index=5 switch_after_tokens=4"));
        t.assert_true(
                "alternate-prompt nonuniform interval schedule should switch at decode token 14",
                contains(fast.logs, "decode_token_index=14 switch_after_tokens=13"));
        t.assert_true(
                "alternate-prompt nonuniform interval schedule should switch at decode token 31",
                contains(fast.logs, "decode_token_index=31 switch_after_tokens=30"));
        assert_fast_transition_trace(t, fast.logs, 5, 4);
        assert_fast_transition_trace(t, fast.logs, 14, 13);
        assert_fast_transition_trace(t, fast.logs, 31, 30);
        assert_fast_transition_zero_kv_handoff(t, fast.logs, 31, 30);
        assert_fast_transition_total_blocking_under_us(t, fast.logs, 5, 4, fast_transition_total_blocking_limit_us);
        assert_fast_transition_total_blocking_under_us(t, fast.logs, 31, 30, fast_transition_total_blocking_limit_us);
    });

    t.test("fast nonuniform interval logits stay aligned after longer qnn prefill", [&](testing & t) {
        const multihop_case_config config = nonuniform_multihop_config(
                "Before the maintenance window started, Mira copied the calibration table, "
                "checked the backup controller status, compared the voltage history, "
                "and wrote a concise incident note for the night operator.",
                /* n_ctx = */ 256,
                /* min_prompt_tokens = */ 32);
        const route_run_result fast = run_qnn_prefill_multihop_case(
                t,
                logs,
                model_path,
                config,
                /* fast_kv_paths = */ true,
                /* assert_fast_path_logs = */ true);
        const route_run_result reference = run_qnn_prefill_multihop_case(
                t,
                logs,
                model_path,
                config,
                /* fast_kv_paths = */ false,
                /* assert_fast_path_logs = */ false);
        assert_semantic_candidate_overlap(t, fast, reference);
        if (!t.assert_true("longer-prefill nonuniform interval fast run should complete", fast.ok)) {
            return;
        }
        t.assert_true(
                "longer-prefill nonuniform interval schedule should switch at decode token 5",
                contains(fast.logs, "decode_token_index=5 switch_after_tokens=4"));
        t.assert_true(
                "longer-prefill nonuniform interval schedule should switch at decode token 14",
                contains(fast.logs, "decode_token_index=14 switch_after_tokens=13"));
        t.assert_true(
                "longer-prefill nonuniform interval schedule should switch at decode token 31",
                contains(fast.logs, "decode_token_index=31 switch_after_tokens=30"));
        assert_fast_transition_trace(t, fast.logs, 5, 4);
        assert_fast_transition_trace(t, fast.logs, 14, 13);
        assert_fast_transition_trace(t, fast.logs, 31, 30);
        assert_fast_transition_zero_kv_handoff(t, fast.logs, 31, 30);
        assert_fast_transition_total_blocking_under_us(t, fast.logs, 5, 4, fast_transition_total_blocking_limit_us);
        assert_fast_transition_total_blocking_under_us(t, fast.logs, 31, 30, fast_transition_total_blocking_limit_us);
    });

    t.test("fast direct opencl to cpu logits stay aligned", [&](testing & t) {
        const multihop_case_config config = opencl_cpu_direct_multihop_config();
        const route_run_result fast = run_qnn_prefill_multihop_case(
                t,
                logs,
                model_path,
                config,
                /* fast_kv_paths = */ true,
                /* assert_fast_path_logs = */ true);
        const route_run_result reference = run_qnn_prefill_multihop_case(
                t,
                logs,
                model_path,
                config,
                /* fast_kv_paths = */ false,
                /* assert_fast_path_logs = */ false);
        assert_semantic_candidate_overlap(t, fast, reference);
        if (!t.assert_true("direct OpenCL -> CPU fast run should complete", fast.ok)) {
            return;
        }
        t.assert_true(
                "direct OpenCL -> CPU schedule should switch at decode token 9",
                contains(fast.logs, "decode_token_index=9 switch_after_tokens=8"));
        t.assert_true(
                "direct CPU -> OpenCL schedule should switch back at decode token 17",
                contains(fast.logs, "decode_token_index=17 switch_after_tokens=16"));
        assert_fast_transition_trace(t, fast.logs, 9, 8);
        assert_fast_transition_trace(t, fast.logs, 17, 16);
        assert_fast_transition_total_blocking_under_us(t, fast.logs, 9, 8, fast_transition_total_blocking_limit_us);
        assert_fast_transition_total_blocking_under_us(t, fast.logs, 17, 16, fast_transition_total_blocking_limit_us);
    });

    t.test("fast repeated qnn reentry logits stay aligned after first qnn switch", [&](testing & t) {
        const multihop_case_config config = repeated_qnn_reentry_multihop_config();
        const route_run_result fast = run_qnn_prefill_multihop_case(
                t,
                logs,
                model_path,
                config,
                /* fast_kv_paths = */ true,
                /* assert_fast_path_logs = */ true);
        const route_run_result reference = run_qnn_prefill_multihop_case(
                t,
                logs,
                model_path,
                config,
                /* fast_kv_paths = */ false,
                /* assert_fast_path_logs = */ false);
        assert_semantic_candidate_overlap(t, fast, reference);
        if (!t.assert_true("repeated QNN re-entry fast run should complete", fast.ok)) {
            return;
        }
        t.assert_true(
                "repeated QNN re-entry schedule should first switch into QNN at decode token 5",
                contains(fast.logs, "decode_token_index=5 switch_after_tokens=4"));
        t.assert_true(
                "repeated QNN re-entry schedule should switch back to OpenCL at decode token 9",
                contains(fast.logs, "decode_token_index=9 switch_after_tokens=8"));
        t.assert_true(
                "repeated QNN re-entry schedule should switch into QNN again at decode token 13",
                contains(fast.logs, "decode_token_index=13 switch_after_tokens=12"));
        t.assert_true(
                "repeated QNN re-entry schedule should switch to CPU at decode token 17",
                contains(fast.logs, "decode_token_index=17 switch_after_tokens=16"));
        assert_fast_transition_trace(t, fast.logs, 5, 4);
        assert_fast_transition_trace(t, fast.logs, 9, 8);
        assert_fast_transition_trace(t, fast.logs, 13, 12);
        assert_fast_transition_trace(t, fast.logs, 17, 16);
        assert_fast_transition_total_blocking_under_us(t, fast.logs, 13, 12, fast_transition_total_blocking_limit_us);
        assert_fast_transition_zero_kv_handoff(t, fast.logs, 17, 16);
        assert_fast_transition_total_blocking_under_us(t, fast.logs, 17, 16, fast_transition_total_blocking_limit_us);
    });

    llama_log_set(logs.previous_callback, logs.previous_user_data);
    return t.summary();
}
