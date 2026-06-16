#include "../tools/e2e-bench/e2e-bench-utils.h"

#include "testing.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

static std::string write_temp_file(const std::string & name, const std::string & contents) {
    const std::string path = std::string("/data/local/tmp/") + name;
    std::ofstream out(path);
    out << contents;
    out.close();
    return path;
}

int main() {
    testing t;

    t.test("parse llama-bench compatible knobs and pg override", [](testing & t) {
        const char * argv[] = {
            "llama-e2e-bench", "-m", "model.gguf", "-t", "3", "-p", "64", "-n", "9", "-d", "7",
            "-pg", "128,32", "-c", "256", "-b", "16", "-ub", "4", "-r", "2", "--no-warmup",
            "--mmap", "0", "--dataset", "sharegpt.jsonl", "--limit", "10", "--no-wait-start",
        };

        e2e_bench_params params;
        std::string err;
        t.assert_true("parse should succeed", e2e_bench_parse_args(sizeof(argv) / sizeof(argv[0]), const_cast<char **>(argv), params, err));
        t.assert_equal("model path", std::string("model.gguf"), params.model);
        t.assert_equal("threads", 3, params.n_threads);
        t.assert_equal("pg prompt override", 128, params.n_prompt);
        t.assert_equal("pg gen override", 32, params.n_gen);
        t.assert_equal("depth", 7, params.n_depth);
        t.assert_equal("ctx", 256, params.n_ctx);
        t.assert_equal("batch", 16, params.n_batch);
        t.assert_equal("ubatch", 4, params.n_ubatch);
        t.assert_equal("reps", 2, params.reps);
        t.assert_true("warmup disabled", params.no_warmup);
        t.assert_true("mmap disabled", !params.use_mmap);
        t.assert_equal("dataset path", std::string("sharegpt.jsonl"), params.dataset);
        t.assert_equal("limit", 10, params.limit);
        t.assert_true("wait disabled", !params.wait_start);
    });

    t.test("load sharegpt jsonl prompts from first human message", [](testing & t) {
        const std::string path = write_temp_file(
            "e2e-bench-sharegpt.jsonl",
            "{\"conversations\":[{\"from\":\"system\",\"value\":\"ignore\"},{\"from\":\"human\",\"value\":\"hello\"},{\"from\":\"gpt\",\"value\":\"hi\"}]}\n"
            "{\"conversations\":[{\"from\":\"user\",\"value\":\"second prompt\"},{\"from\":\"assistant\",\"value\":\"ok\"}]}\n");

        std::string err;
        const std::vector<std::string> prompts = e2e_bench_load_prompts(path, 2, err);

        std::remove(path.c_str());

        t.assert_equal("prompt count", 2, (int) prompts.size());
        t.assert_equal("first prompt", std::string("hello"), prompts[0]);
        t.assert_equal("second prompt", std::string("second prompt"), prompts[1]);
    });

    t.test("select default sharegpt file when dataset argument is a directory", [](testing & t) {
        std::string err;
        const std::string selected = e2e_bench_resolve_dataset_path(
                "/data/local/tmp/e2e-bench-dataset-dir",
                {"other.jsonl", "sharegpt_gpt4.jsonl"},
                err);
        t.assert_equal("selected sharegpt_gpt4", std::string("/data/local/tmp/e2e-bench-dataset-dir/sharegpt_gpt4.jsonl"), selected);
    });

    return t.failures == 0 ? 0 : 1;
}
