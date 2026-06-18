#include "e2e-bench-utils.h"

#include "common.h"
#include "ggml.h"
#include "llama.h"

#include "../../src/llama-context.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

struct sample_result {
    int     rep           = 0;
    int     sample        = 0;
    int     prompt_tokens = 0;
    int     gen_tokens    = 0;
    int64_t elapsed_us    = 0;
};

static void null_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) text;
    (void) user_data;
}

static void print_usage(int, char ** argv) {
    std::fprintf(stderr,
            "usage: %s -m model.gguf [llama-bench-like options] [--dataset path --limit N]\n"
            "\n"
            "llama-bench-like options:\n"
            "  -t, --threads N          threads for prompt and generation\n"
            "  -p, --n-prompt N         synthetic prompt tokens; with --dataset, N>0 truncates prompt and 0 keeps full prompt\n"
            "  -n, --n-gen N            generated tokens per sample\n"
            "  -d, --n-depth N          synthetic depth tokens before each sample\n"
            "  -pg PP,TG                set prompt/decode token counts\n"
            "  -c, --ctx-size N         context size\n"
            "  -b, --batch-size N       logical batch size\n"
            "  -ub, --ubatch-size N     physical batch size\n"
            "  -r, --repetitions N      repetitions over the prompt set\n"
            "  --no-warmup              skip pre-start warmup\n"
            "  --mmap 0|1               model mmap setting\n"
            "\n"
            "backend/script compatibility:\n"
            "  -ngl, --n-gpu-layers N   GPU/offload layers\n"
            "  -dev, --device DEV       backend device name, slash-separated list, auto, or none\n"
            "  -v, --verbose            verbose llama.cpp logging\n"
            "\n"
            "e2e options:\n"
            "  --dataset PATH           JSONL file or directory; ShareGPT conversations are supported\n"
            "  --limit N, --samples N   max dataset samples; 0 means all\n"
            "  --dataset-output-tokens  generate the same token count as each dataset assistant output\n"
            "  --no-wait-start          do not wait at READY before MEASURE_BEGIN\n",
            argv[0]);
}

static bool fast_exit_requested() {
    const char * value = std::getenv("LLAMA_E2E_BENCH_FAST_EXIT");
    if (value == nullptr) {
        value = std::getenv("LLAMA_BENCH_FAST_EXIT");
    }
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static bool parse_devices_arg(const std::string & value, std::vector<ggml_backend_dev_t> & out, std::string & err) {
    const std::string trimmed = e2e_bench_trim(value);
    if (trimmed.empty() || trimmed == "auto") {
        return true;
    }
    if (trimmed == "none") {
        out.push_back(nullptr);
        return true;
    }

    std::stringstream ss(trimmed);
    std::string name;
    while (std::getline(ss, name, '/')) {
        name = e2e_bench_trim(name);
        if (name.empty()) {
            err = "invalid empty device name in: " + value;
            return false;
        }
        ggml_backend_dev_t dev = ggml_backend_dev_by_name(name.c_str());
        if (dev == nullptr || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            err = "invalid non-CPU backend device: " + name;
            return false;
        }
        out.push_back(dev);
    }
    out.push_back(nullptr);
    return true;
}

static std::string make_synthetic_prompt(int n_prompt) {
    std::string prompt;
    for (int i = 0; i < n_prompt; ++i) {
        if (!prompt.empty()) {
            prompt.push_back(' ');
        }
        prompt += "benchmark";
    }
    return prompt;
}

static std::vector<llama_token> make_depth_tokens(const llama_vocab * vocab, int n_depth) {
    std::vector<llama_token> tokens;
    if (n_depth <= 0) {
        return tokens;
    }
    tokens.reserve((size_t) n_depth);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    for (int i = 0; i < n_depth; ++i) {
        if (i == 0 && llama_vocab_get_add_bos(vocab)) {
            tokens.push_back(llama_vocab_bos(vocab));
        } else {
            tokens.push_back((llama_token) (1 + (i % std::max(1, n_vocab - 1))));
        }
    }
    return tokens;
}

static bool decode_tokens(llama_context * ctx, const std::vector<llama_token> & tokens, int n_batch, int & n_past) {
    if (tokens.empty()) {
        return true;
    }

    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    size_t offset = 0;
    while (offset < tokens.size()) {
        common_batch_clear(batch);
        const int n_cur = std::min<int>(n_batch, (int) (tokens.size() - offset));
        for (int i = 0; i < n_cur; ++i) {
            const bool logits = offset + i + 1 == tokens.size();
            common_batch_add(batch, tokens[offset + i], n_past + i, { 0 }, logits);
        }
        const int res = llama_decode(ctx, batch);
        if (res != 0) {
            std::fprintf(stderr, "llama-e2e-bench: decode failed, res=%d\n", res);
            llama_batch_free(batch);
            return false;
        }
        n_past += n_cur;
        offset += (size_t) n_cur;
    }
    llama_synchronize(ctx);
    llama_batch_free(batch);
    return true;
}

static bool decode_one(llama_context * ctx, llama_token token, int pos) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_clear(batch);
    common_batch_add(batch, token, pos, { 0 }, true);
    const int res = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (res != 0) {
        std::fprintf(stderr, "llama-e2e-bench: single-token decode failed, res=%d\n", res);
        return false;
    }
    llama_synchronize(ctx);
    return true;
}

static bool run_sample(
        llama_context * ctx,
        llama_sampler * sampler,
        const e2e_bench_sample & sample,
        const e2e_bench_params & params,
        bool dataset_prompt,
        int target_gen_tokens,
        sample_result & result) {
    llama_memory_clear(llama_get_memory(ctx), true);
    ctx->clear_dynamic_seq0_token_history();
    ctx->reset_dynamic_route_runtime_state();
    llama_sampler_reset(sampler);

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_past = 0;

    const int64_t start_us = llama_time_us();

    const std::vector<llama_token> depth_tokens = make_depth_tokens(vocab, params.n_depth);
    if (!decode_tokens(ctx, depth_tokens, params.n_batch, n_past)) {
        return false;
    }

    std::vector<llama_token> prompt_tokens = common_tokenize(vocab, sample.prompt, true, true);
    const int max_prompt_tokens = std::max(1, (int) llama_n_ctx(ctx) - params.n_depth - target_gen_tokens);
    int prompt_cap = max_prompt_tokens;
    if (params.n_prompt > 0) {
        prompt_cap = std::min(prompt_cap, params.n_prompt);
    }
    if ((dataset_prompt || params.n_prompt > 0) && (int) prompt_tokens.size() > prompt_cap) {
        prompt_tokens.resize((size_t) prompt_cap);
    }
    if (!dataset_prompt && params.n_prompt == 0) {
        prompt_tokens.clear();
    }

    if (prompt_tokens.empty() && target_gen_tokens > 0) {
        prompt_tokens.push_back(llama_vocab_get_add_bos(vocab) ? llama_vocab_bos(vocab) : (llama_token) 1);
    }

    if (!decode_tokens(ctx, prompt_tokens, params.n_batch, n_past)) {
        return false;
    }

    int generated = 0;
    for (; generated < target_gen_tokens; ++generated) {
        llama_token token = llama_sampler_sample(sampler, ctx, -1);
        llama_sampler_accept(sampler, token);
        if (!decode_one(ctx, token, n_past)) {
            return false;
        }
        ++n_past;
    }

    const int64_t end_us = llama_time_us();
    result.prompt_tokens = (int) prompt_tokens.size();
    result.gen_tokens    = generated;
    result.elapsed_us    = end_us - start_us;
    return true;
}

static int e2e_bench_sample_target_gen_tokens(
        const llama_vocab *      vocab,
        const e2e_bench_sample & sample,
        const e2e_bench_params & params) {
    if (params.dataset_output_tokens && !sample.output.empty()) {
        const std::vector<llama_token> output_tokens = common_tokenize(vocab, sample.output, false, true);
        return (int) output_tokens.size();
    }
    return params.n_gen;
}

static std::vector<int> e2e_bench_resolve_sample_gen_tokens(
        const llama_vocab *                    vocab,
        const std::vector<e2e_bench_sample> &  samples,
        const e2e_bench_params &               params) {
    std::vector<int> targets;
    targets.reserve(samples.size());
    for (const auto & sample : samples) {
        targets.push_back(std::max(0, e2e_bench_sample_target_gen_tokens(vocab, sample, params)));
    }
    return targets;
}

static bool run_warmup(
        llama_context *                       ctx,
        llama_sampler *                       sampler,
        const std::vector<e2e_bench_sample> & samples,
        const std::vector<int> &              sample_gen_tokens,
        const e2e_bench_params &              params,
        bool                                  dataset_prompt) {
    const int warmup_gen_tokens = sample_gen_tokens.empty() ? 0 : std::min(sample_gen_tokens.front(), 1);
    sample_result ignored;
    return run_sample(ctx, sampler, samples.front(), params, dataset_prompt, warmup_gen_tokens, ignored);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    std::setlocale(LC_CTYPE, ".UTF-8");

    e2e_bench_params params;
    std::string err;
    if (!e2e_bench_parse_args(argc, argv, params, err)) {
        std::fprintf(stderr, "llama-e2e-bench: error: %s\n", err.c_str());
        print_usage(argc, argv);
        return 1;
    }
    if (params.help) {
        print_usage(argc, argv);
        return 0;
    }

    ggml_backend_load_all();
    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

    if (!params.verbose) {
        llama_log_set(null_log_callback, nullptr);
    }

    std::vector<ggml_backend_dev_t> devices;
    if (!parse_devices_arg(params.devices, devices, err)) {
        std::fprintf(stderr, "llama-e2e-bench: error: %s\n", err.c_str());
        return 1;
    }

    std::vector<e2e_bench_sample> samples;
    bool dataset_prompt = !params.dataset.empty();
    if (dataset_prompt) {
        samples = e2e_bench_load_samples(params.dataset, params.limit, err);
        if (!err.empty()) {
            std::fprintf(stderr, "llama-e2e-bench: error: %s\n", err.c_str());
            return 1;
        }
    } else {
        e2e_bench_sample sample;
        sample.prompt = make_synthetic_prompt(params.n_prompt);
        samples.push_back(sample);
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = params.n_gpu_layers;
    mparams.use_mmap     = params.use_mmap;
    if (!devices.empty()) {
        mparams.devices = devices.data();
    }

    llama_model * model = llama_model_load_from_file(params.model.c_str(), mparams);
    if (model == nullptr) {
        std::fprintf(stderr, "llama-e2e-bench: error: failed to load model: %s\n", params.model.c_str());
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::vector<int> sample_gen_tokens =
        e2e_bench_resolve_sample_gen_tokens(vocab, samples, params);
    const int max_gen_tokens =
        sample_gen_tokens.empty() ? params.n_gen :
        *std::max_element(sample_gen_tokens.begin(), sample_gen_tokens.end());

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = params.n_ctx > 0 ? params.n_ctx : std::max(512, params.n_depth + params.n_prompt + max_gen_tokens + 8);
    cparams.n_batch         = params.n_batch;
    cparams.n_ubatch        = params.n_ubatch;
    cparams.n_threads       = params.n_threads;
    cparams.n_threads_batch = params.n_threads;
    cparams.no_perf         = false;
    cparams.swa_full        = false;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        std::fprintf(stderr, "llama-e2e-bench: error: failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    if (!params.no_warmup) {
        std::fprintf(stderr, "llama-e2e-bench: warmup begin\n");
        if (!run_warmup(ctx, sampler, samples, sample_gen_tokens, params, dataset_prompt)) {
            std::fprintf(stderr, "llama-e2e-bench: error: warmup failed\n");
            llama_sampler_free(sampler);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        std::fprintf(stderr, "llama-e2e-bench: warmup end\n");
        llama_memory_clear(llama_get_memory(ctx), true);
        ctx->clear_dynamic_seq0_token_history();
        ctx->reset_dynamic_route_runtime_state();
    }

    llama_perf_context_reset(ctx);
    llama_perf_sampler_reset(sampler);

    std::fprintf(stderr,
            "llama-e2e-bench: READY prompts=%zu reps=%d n_gen=%d max_gen=%d dataset_output_tokens=%d n_ctx=%u n_batch=%d n_ubatch=%d dataset=%s\n",
            samples.size(),
            params.reps,
            params.n_gen,
            max_gen_tokens,
            params.dataset_output_tokens ? 1 : 0,
            llama_n_ctx(ctx),
            params.n_batch,
            params.n_ubatch,
            dataset_prompt ? params.dataset.c_str() : "<synthetic>");
    std::fflush(stderr);

    if (params.wait_start) {
        std::fprintf(stderr, "llama-e2e-bench: press ENTER to start measured run\n");
        std::fflush(stderr);
        std::string line;
        if (!std::getline(std::cin, line)) {
            std::fprintf(stderr, "llama-e2e-bench: stdin closed, starting measured run\n");
        }
    }

    std::fprintf(stderr, "llama-e2e-bench: MEASURE_BEGIN\n");
    std::fflush(stderr);
    const int64_t measure_start_us = llama_time_us();

    std::vector<sample_result> results;
    results.reserve((size_t) params.reps * samples.size());

    for (int rep = 0; rep < params.reps; ++rep) {
        for (size_t i = 0; i < samples.size(); ++i) {
            sample_result result;
            result.rep    = rep + 1;
            result.sample = (int) i + 1;
            std::fprintf(stderr, "llama-e2e-bench: SAMPLE_BEGIN rep=%d sample=%zu/%zu\n", rep + 1, i + 1, samples.size());
            std::fflush(stderr);
            if (!run_sample(ctx, sampler, samples[i], params, dataset_prompt, sample_gen_tokens[i], result)) {
                std::fprintf(stderr, "llama-e2e-bench: error: sample failed rep=%d sample=%zu\n", rep + 1, i + 1);
                llama_sampler_free(sampler);
                llama_free(ctx);
                llama_model_free(model);
                return 1;
            }
            results.push_back(result);
            std::printf("sample,rep=%d,index=%d,prompt_tokens=%d,gen_tokens=%d,elapsed_ms=%.3f,tok_s=%.3f\n",
                    result.rep,
                    result.sample,
                    result.prompt_tokens,
                    result.gen_tokens,
                    result.elapsed_us / 1000.0,
                    result.elapsed_us > 0 ? 1000000.0 * result.gen_tokens / result.elapsed_us : 0.0);
            std::fflush(stdout);
            std::fprintf(stderr,
                    "llama-e2e-bench: SAMPLE_END rep=%d sample=%zu/%zu elapsed_ms=%.3f\n",
                    rep + 1,
                    i + 1,
                    samples.size(),
                    result.elapsed_us / 1000.0);
        }
    }

    const int64_t measure_elapsed_us = llama_time_us() - measure_start_us;
    const int total_prompt_tokens = std::accumulate(results.begin(), results.end(), 0, [](int sum, const sample_result & r) {
        return sum + r.prompt_tokens;
    });
    const int total_gen_tokens = std::accumulate(results.begin(), results.end(), 0, [](int sum, const sample_result & r) {
        return sum + r.gen_tokens;
    });

    std::printf("summary,samples=%zu,reps=%d,total_prompt_tokens=%d,total_gen_tokens=%d,elapsed_ms=%.3f,tok_s=%.3f\n",
            samples.size(),
            params.reps,
            total_prompt_tokens,
            total_gen_tokens,
            measure_elapsed_us / 1000.0,
            measure_elapsed_us > 0 ? 1000000.0 * total_gen_tokens / measure_elapsed_us : 0.0);
    std::fflush(stdout);

    std::fprintf(stderr, "llama-e2e-bench: MEASURE_END elapsed_ms=%.3f\n", measure_elapsed_us / 1000.0);
    if (params.verbose) {
        llama_perf_sampler_print(sampler);
        llama_perf_context_print(ctx);
    }

    llama_sampler_free(sampler);
    llama_free(ctx);
    llama_model_free(model);

    if (fast_exit_requested()) {
        std::fflush(nullptr);
        std::_Exit(0);
    }

    llama_backend_free();
    return 0;
}
