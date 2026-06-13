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

enum class invalid_kv_shape {
    REMOVED_FRONT_TOKEN,
    SHARED_SEQ_CELL,
};

struct route_case {
    const char * producer_backend;
    int32_t n_gpu_layers;
    bool kv_unified;
    invalid_kv_shape shape;
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

static void force_producer_to_qnn_schedule_env(const char * producer_backend) {
    setenv("GGML_HETERO_DYNAMIC_MODE", "phase", 1);
    setenv("GGML_HETERO_DYNAMIC_PREFILL_ROUTE", producer_backend, 1);
    unsetenv("GGML_HETERO_DYNAMIC_DECODE_ROUTE");
    unsetenv("GGML_HETERO_DYNAMIC_DECODE_KV");
    const std::string schedule = std::string("1:") + producer_backend + ";2:qnn-npu";
    setenv("GGML_HETERO_DYNAMIC_DECODE_SCHEDULE", schedule.c_str(), 1);
    setenv("GGML_HETERO_DYNAMIC_TRACE", "1", 1);
    setenv("GGML_HETERO_DYNAMIC_TRACE_TIMING", "1", 1);
    setenv("GGML_HETERO_DYNAMIC_TRACE_TIMING_DETAIL", "1", 1);
    setenv("GGML_QNN_AOT_WRITE_GENERIC_KV", "1", 1);
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

static void make_live_kv_nonphysical_prefix(testing & t, llama_memory_t mem, invalid_kv_shape shape) {
    switch (shape) {
        case invalid_kv_shape::REMOVED_FRONT_TOKEN: {
            const bool removed_front = llama_memory_seq_rm(mem, 0, 0, 1);
            t.assert_true("front-token removal should succeed", removed_front);
            t.assert_equal("seq0 should now start after position zero", (llama_pos) 1, llama_memory_seq_pos_min(mem, 0));
            return;
        }
        case invalid_kv_shape::SHARED_SEQ_CELL: {
            llama_memory_seq_cp(mem, 0, 1, 0, 1);
            t.assert_equal("seq0 should still start at position zero", (llama_pos) 0, llama_memory_seq_pos_min(mem, 0));
            t.assert_equal("seq1 should share the first physical token", (llama_pos) 0, llama_memory_seq_pos_min(mem, 1));
            t.assert_equal("seq1 should only share position zero", (llama_pos) 0, llama_memory_seq_pos_max(mem, 1));
            return;
        }
    }
}

static void run_nonphysical_prefix_refusal_case(
        testing & t,
        captured_logs & logs,
        const std::string & model_path,
        const route_case & route) {
    force_producer_to_qnn_schedule_env(route.producer_backend);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = route.n_gpu_layers;
    mparams.hetero_phase_route = route.producer_backend;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!t.assert_true("model should load", model != nullptr)) {
        return;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 128;
    cparams.n_batch = 64;
    cparams.n_ubatch = 64;
    cparams.n_seq_max = 2;
    cparams.kv_unified = route.kv_unified;
    cparams.n_threads = 4;
    cparams.n_threads_batch = 4;
    cparams.hetero_phase_route = route.producer_backend;
    cparams.no_perf = true;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!t.assert_true("context should initialize with producer and scheduled QNN routes", ctx != nullptr)) {
        llama_model_free(model);
        return;
    }
    llama_set_warmup(ctx, false);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> prompt_tokens = tokenize_prompt(
            vocab,
            "Mira fixed the bridge before sunrise.");
    if (!t.assert_true("prompt should tokenize to at least four tokens", prompt_tokens.size() >= 4)) {
        llama_free(ctx);
        llama_model_free(model);
        return;
    }

    if (!t.assert_true("prefill on the producer backend should decode", decode_tokens(ctx, prompt_tokens, 0))) {
        llama_free(ctx);
        llama_model_free(model);
        return;
    }
    std::vector<llama_token> first_decode = { prompt_tokens.back() };
    if (!t.assert_true("first scheduled producer decode should decode", decode_tokens(ctx, first_decode, prompt_tokens.size()))) {
        llama_free(ctx);
        llama_model_free(model);
        return;
    }

    llama_memory_t mem = llama_get_memory(ctx);
    const llama_pos before_min = llama_memory_seq_pos_min(mem, 0);
    const llama_pos before_max = llama_memory_seq_pos_max(mem, 0);
    t.assert_equal("seq0 should start as an append-only prefix", (llama_pos) 0, before_min);
    t.assert_equal("seq0 max should include the first decode token",
            static_cast<llama_pos>(prompt_tokens.size()), before_max);

    make_live_kv_nonphysical_prefix(t, mem, route.shape);

    const size_t second_decode_log_start = logs.text.size();
    std::vector<llama_token> second_decode = { prompt_tokens[prompt_tokens.size() - 2] };
    if (!t.assert_true("second decode should continue on the existing producer route after QNN switch refusal",
                decode_tokens(ctx, second_decode, static_cast<llama_pos>(prompt_tokens.size() + 1)))) {
        llama_free(ctx);
        llama_model_free(model);
        return;
    }

    const std::string second_decode_logs = logs.text.substr(second_decode_log_start);
    const std::string route_after_refusal = current_route(ctx);

    t.assert_true(
            "runtime should reject non-QNN -> QNN switch for non-physical-prefix memory",
            contains(second_decode_logs,
                "refusing non-QNN -> qnn decode switch because memory is not a physical seq0 prefix"));
    t.assert_true(
            "tracked token history should be cleared after the memory is fragmented",
            contains(second_decode_logs,
                "clearing tracked seq0 token history because memory is not a physical seq0 prefix"));
    t.assert_true(
            "direct generic KV import must not be prepared after the guard rejects the switch",
            !contains(second_decode_logs, "prepared direct generic KV import"));
    t.assert_true(
            "direct generic KV import must not be used after the guard rejects the switch",
            !contains(second_decode_logs, "using direct generic KV import"));
    t.assert_true(
            "QNN prefix replay must not be queued after the guard rejects the switch",
            !contains(second_decode_logs, "queued QNN prefix replay"));
    t.assert_true(
            "route should remain on the producer backend after the refused switch",
            contains(route_after_refusal, route.producer_backend) && !contains(route_after_refusal, "qnn"));

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
                "Skipping QNN dynamic-route nonphysical-prefix runtime test.\n\033[0m");
        return 0;
    }

    captured_logs logs;
    llama_log_get(&logs.previous_callback, &logs.previous_user_data);
    llama_log_set(capture_log_callback, &logs);

    ggml_backend_load_all();

    testing t;
    t.test("cpu to qnn switch is refused when live seq0 memory is not a physical prefix", [&](testing & t) {
        run_nonphysical_prefix_refusal_case(t, logs, model_path, {
            /*.producer_backend =*/ "cpu",
            /*.n_gpu_layers =*/ 0,
            /*.kv_unified =*/ false,
            /*.shape =*/ invalid_kv_shape::REMOVED_FRONT_TOKEN,
        });
    });

    t.test("opencl to qnn switch is refused when live seq0 memory is not a physical prefix", [&](testing & t) {
        run_nonphysical_prefix_refusal_case(t, logs, model_path, {
            /*.producer_backend =*/ "opencl",
            /*.n_gpu_layers =*/ -1,
            /*.kv_unified =*/ false,
            /*.shape =*/ invalid_kv_shape::REMOVED_FRONT_TOKEN,
        });
    });

    t.test("cpu to qnn switch is refused when live KV contains a shared sequence cell", [&](testing & t) {
        run_nonphysical_prefix_refusal_case(t, logs, model_path, {
            /*.producer_backend =*/ "cpu",
            /*.n_gpu_layers =*/ 0,
            /*.kv_unified =*/ true,
            /*.shape =*/ invalid_kv_shape::SHARED_SEQ_CELL,
        });
    });

    llama_log_set(logs.previous_callback, logs.previous_user_data);
    return t.summary();
}
