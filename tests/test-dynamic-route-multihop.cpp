#include "common.h"
#include "ggml.h"
#include "llama.h"
#include "testing.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct captured_logs {
    ggml_log_callback previous_callback = nullptr;
    void * previous_user_data = nullptr;
    std::string text;
};

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

static void force_qnn_prefill_multihop_schedule_env() {
    setenv("GGML_HETERO_DYNAMIC_MODE", "phase", 1);
    setenv("GGML_HETERO_DYNAMIC_PREFILL_ROUTE", "qnn-npu", 1);
    unsetenv("GGML_HETERO_DYNAMIC_DECODE_ROUTE");
    unsetenv("GGML_HETERO_DYNAMIC_DECODE_KV");
    unsetenv("GGML_HETERO_QNN_SHARED_HOST");
    setenv("GGML_HETERO_DYNAMIC_DECODE_SCHEDULE", "1:opencl;3:qnn-npu;5:cpu;7:opencl", 1);
    setenv("GGML_HETERO_DYNAMIC_TRACE", "1", 1);
    setenv("GGML_HETERO_DYNAMIC_TRACE_TIMING", "1", 1);
    setenv("GGML_HETERO_DYNAMIC_TRACE_TIMING_DETAIL", "1", 1);
    setenv("GGML_QNN_AOT_WRITE_GENERIC_KV", "1", 1);
    setenv("GGML_HETERO_ENABLE_OPENCL_CPU_UMA_KV_HANDOFF", "1", 1);
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

static void run_qnn_prefill_multihop_case(testing & t, captured_logs & logs, const std::string & model_path) {
    force_qnn_prefill_multihop_schedule_env();
    const size_t run_log_start = logs.text.size();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = -1;
    mparams.hetero_phase_route = "qnn-npu";

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!t.assert_true("model should load", model != nullptr)) {
        return;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 160;
    cparams.n_batch = 64;
    cparams.n_ubatch = 64;
    cparams.n_threads = 4;
    cparams.n_threads_batch = 4;
    cparams.hetero_phase_route = "qnn-npu";
    cparams.no_perf = true;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!t.assert_true("context should initialize with QNN prefill and scheduled decode routes", ctx != nullptr)) {
        llama_model_free(model);
        return;
    }
    llama_set_warmup(ctx, false);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> prompt_tokens = tokenize_prompt(
            vocab,
            "Mira fixed the bridge before sunrise and checked every cable.");
    if (!t.assert_true("prompt should tokenize to at least eight tokens", prompt_tokens.size() >= 8)) {
        llama_free(ctx);
        llama_model_free(model);
        return;
    }

    if (!t.assert_true("QNN prefill should decode", decode_tokens(ctx, prompt_tokens, 0))) {
        llama_free(ctx);
        llama_model_free(model);
        return;
    }

    for (int i = 0; i < 7; ++i) {
        const llama_token token = prompt_tokens[(prompt_tokens.size() - 1 - (size_t) i) % prompt_tokens.size()];
        const std::vector<llama_token> decode_token = { token };
        if (!t.assert_true("scheduled decode token should decode",
                    decode_tokens(ctx, decode_token, static_cast<llama_pos>(prompt_tokens.size() + (size_t) i)))) {
            llama_free(ctx);
            llama_model_free(model);
            return;
        }
    }

    const std::string run_logs = logs.text.substr(run_log_start);
    const std::string route_after = current_route(ctx);

    t.assert_true(
            "dynamic schedule should be active",
            contains(run_logs, "decode_schedule=1:attn=opencl,ffn=opencl,output=opencl;3:attn=qnn-npu,ffn=qnn-npu,output=qnn-npu;5:attn=cpu,ffn=cpu,output=cpu;7:attn=opencl,ffn=opencl,output=opencl"));
    t.assert_true(
            "QNN -> OpenCL should use direct shared KV handoff",
            contains(run_logs, "completed direct shared QNN/OpenCL KV handoff"));
    t.assert_true(
            "OpenCL -> QNN should prepare direct generic KV import",
            contains(run_logs, "prepared direct generic KV import"));
    t.assert_true(
            "OpenCL -> QNN should use direct generic KV import",
            contains(run_logs, "using direct generic KV import"));
    t.assert_true(
            "QNN -> CPU should reuse QNN-written live generic KV",
            contains(run_logs, "reusing QNN-written live generic KV directly"));
    t.assert_true(
            "CPU -> OpenCL should use UMA KV handoff",
            contains(run_logs, "completed CPU/OpenCL UMA KV handoff"));
    t.assert_true(
            "four route transitions should be traced",
            count_occurrences(run_logs, "TRANSITION_TRACE") >= 4);
    t.assert_true(
            "all traced transitions should report success",
            count_occurrences(run_logs, "success=1") >= 4);
    t.assert_true(
            "fast multihop route should not refuse QNN switching",
            !contains(run_logs, "refusing non-QNN -> qnn decode switch"));
    t.assert_true(
            "fast multihop route should not queue prefix replay",
            !contains(run_logs, "queued QNN prefix replay"));
    t.assert_true(
            "fast multihop route should not rebuild from state",
            !contains(run_logs, "rebuild_dynamic_consumer_kv_from_state"));
    t.assert_true(
            "fast multihop route should not fall back",
            !contains(run_logs, "fallback=1"));
    t.assert_true(
            "fast multihop route should not hit unmatched QNN AoT graphs",
            !contains(run_logs, "unmatched cgraph"));
    t.assert_true(
            "fast multihop route should not crash",
            !contains(run_logs, "SIGSEGV") && !contains(run_logs, "Segmentation fault"));
    t.assert_true(
            "final scheduled route should be OpenCL",
            contains(route_after, "opencl") && !contains(route_after, "qnn") && !contains(route_after, "cpu"));

    llama_free(ctx);
    llama_model_free(model);
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
        run_qnn_prefill_multihop_case(t, logs, model_path);
    });

    llama_log_set(logs.previous_callback, logs.previous_user_data);
    return t.summary();
}
