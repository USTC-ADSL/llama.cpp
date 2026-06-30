#pragma once

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

struct e2e_bench_params {
    std::string model;
    std::string dataset;
    std::string devices = "auto";

    int n_threads    = 1;
    int n_prompt     = 512;
    int n_gen        = 128;
    int n_depth      = 0;
    int n_ctx        = 0;
    int n_batch      = 2048;
    int n_ubatch     = 512;
    int reps         = 1;
    int n_gpu_layers = 99;
    int limit        = 0;

    bool no_warmup  = false;
    bool use_mmap   = true;
    bool wait_start = true;
    bool verbose    = false;
    bool help       = false;
    bool dataset_output_tokens = false;

    std::string planner_prefill_profile;
    std::string planner_decode_profile;
    std::string planner_context_match = "exact";
    std::string planner_initial_decode_state;
    double planner_ttft_slo_ms = -1.0;
    double planner_tbt_slo_ms  = -1.0;
    int planner_input_len  = -1;
    int planner_output_len = -1;
    int planner_bucket_size = 32;
    int planner_max_context = 6144;
};

struct e2e_bench_sample {
    std::string prompt;
    std::string output;
};

static inline std::string e2e_bench_trim(const std::string & s) {
    size_t first = 0;
    while (first < s.size() && std::isspace((unsigned char) s[first])) {
        ++first;
    }
    size_t last = s.size();
    while (last > first && std::isspace((unsigned char) s[last - 1])) {
        --last;
    }
    return s.substr(first, last - first);
}

static inline bool e2e_bench_ends_with(const std::string & s, const std::string & suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static inline std::string e2e_bench_join_path(const std::string & dir, const std::string & name) {
    if (dir.empty() || dir.back() == '/') {
        return dir + name;
    }
    return dir + "/" + name;
}

static inline bool e2e_bench_parse_int(const std::string & text, int & out) {
    std::string value = e2e_bench_trim(text);
    if (value.empty()) {
        return false;
    }
    char * end = nullptr;
    errno = 0;
    long parsed = std::strtol(value.c_str(), &end, 10);
    if (errno != 0 || end == value.c_str() || *end != '\0') {
        return false;
    }
    out = (int) parsed;
    return true;
}

static inline bool e2e_bench_parse_double(const std::string & text, double & out) {
    std::string value = e2e_bench_trim(text);
    if (value.empty()) {
        return false;
    }
    char * end = nullptr;
    errno = 0;
    double parsed = std::strtod(value.c_str(), &end);
    if (errno != 0 || end == value.c_str() || *end != '\0') {
        return false;
    }
    out = parsed;
    return true;
}

static inline bool e2e_bench_parse_bool(const std::string & text, bool & out) {
    std::string value = e2e_bench_trim(text);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return (char) std::tolower(c);
    });
    if (value == "1" || value == "true" || value == "yes" || value == "on") {
        out = true;
        return true;
    }
    if (value == "0" || value == "false" || value == "no" || value == "off") {
        out = false;
        return true;
    }
    return false;
}

static inline bool e2e_bench_take_arg(int argc, char ** argv, int & i, std::string & value, std::string & err) {
    if (++i >= argc) {
        err = std::string("missing value for ") + argv[i - 1];
        return false;
    }
    value = argv[i];
    return true;
}

static inline bool e2e_bench_parse_args(int argc, char ** argv, e2e_bench_params & params, std::string & err) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        std::string value;

        auto parse_int_arg = [&](int & target) {
            if (!e2e_bench_take_arg(argc, argv, i, value, err)) {
                return false;
            }
            if (!e2e_bench_parse_int(value, target)) {
                err = "invalid integer for " + arg + ": " + value;
                return false;
            }
            return true;
        };
        auto parse_double_arg = [&](double & target) {
            if (!e2e_bench_take_arg(argc, argv, i, value, err)) {
                return false;
            }
            if (!e2e_bench_parse_double(value, target)) {
                err = "invalid number for " + arg + ": " + value;
                return false;
            }
            return true;
        };

        if (arg == "-h" || arg == "--help") {
            params.help = true;
        } else if (arg == "-v" || arg == "--verbose") {
            params.verbose = true;
        } else if (arg == "-m" || arg == "--model") {
            if (!e2e_bench_take_arg(argc, argv, i, params.model, err)) {
                return false;
            }
        } else if (arg == "-t" || arg == "--threads") {
            if (!parse_int_arg(params.n_threads)) {
                return false;
            }
        } else if (arg == "-p" || arg == "--n-prompt") {
            if (!parse_int_arg(params.n_prompt)) {
                return false;
            }
        } else if (arg == "-n" || arg == "--n-gen") {
            if (!parse_int_arg(params.n_gen)) {
                return false;
            }
        } else if (arg == "-d" || arg == "--n-depth") {
            if (!parse_int_arg(params.n_depth)) {
                return false;
            }
        } else if (arg == "-c" || arg == "--ctx-size") {
            if (!parse_int_arg(params.n_ctx)) {
                return false;
            }
        } else if (arg == "-b" || arg == "--batch-size") {
            if (!parse_int_arg(params.n_batch)) {
                return false;
            }
        } else if (arg == "-ub" || arg == "--ubatch-size") {
            if (!parse_int_arg(params.n_ubatch)) {
                return false;
            }
        } else if (arg == "-r" || arg == "--repetitions") {
            if (!parse_int_arg(params.reps)) {
                return false;
            }
        } else if (arg == "-ngl" || arg == "--n-gpu-layers") {
            if (!parse_int_arg(params.n_gpu_layers)) {
                return false;
            }
        } else if (arg == "-dev" || arg == "--device") {
            if (!e2e_bench_take_arg(argc, argv, i, params.devices, err)) {
                return false;
            }
        } else if (arg == "-pg") {
            if (!e2e_bench_take_arg(argc, argv, i, value, err)) {
                return false;
            }
            const size_t comma = value.find(',');
            if (comma == std::string::npos) {
                err = "invalid -pg value, expected pp,tg: " + value;
                return false;
            }
            int pp = 0;
            int tg = 0;
            if (!e2e_bench_parse_int(value.substr(0, comma), pp) ||
                !e2e_bench_parse_int(value.substr(comma + 1), tg)) {
                err = "invalid -pg value, expected pp,tg: " + value;
                return false;
            }
            params.n_prompt = pp;
            params.n_gen    = tg;
        } else if (arg == "--no-warmup") {
            params.no_warmup = true;
        } else if (arg == "-mmp" || arg == "--mmap") {
            if (!e2e_bench_take_arg(argc, argv, i, value, err)) {
                return false;
            }
            if (!e2e_bench_parse_bool(value, params.use_mmap)) {
                err = "invalid bool for " + arg + ": " + value;
                return false;
            }
        } else if (arg == "--dataset") {
            if (!e2e_bench_take_arg(argc, argv, i, params.dataset, err)) {
                return false;
            }
        } else if (arg == "--limit" || arg == "--samples") {
            if (!parse_int_arg(params.limit)) {
                return false;
            }
        } else if (arg == "--dataset-output-tokens") {
            params.dataset_output_tokens = true;
        } else if (arg == "--no-wait-start") {
            params.wait_start = false;
        } else if (arg == "--wait-start") {
            params.wait_start = true;
        } else if (arg == "--planner-prefill-profile") {
            if (!e2e_bench_take_arg(argc, argv, i, params.planner_prefill_profile, err)) {
                return false;
            }
        } else if (arg == "--planner-decode-profile" || arg == "--planner-profile") {
            if (!e2e_bench_take_arg(argc, argv, i, params.planner_decode_profile, err)) {
                return false;
            }
        } else if (arg == "--planner-ttft-slo-ms") {
            if (!parse_double_arg(params.planner_ttft_slo_ms)) {
                return false;
            }
        } else if (arg == "--planner-tbt-slo-ms") {
            if (!parse_double_arg(params.planner_tbt_slo_ms)) {
                return false;
            }
        } else if (arg == "--planner-input-len") {
            if (!parse_int_arg(params.planner_input_len)) {
                return false;
            }
        } else if (arg == "--planner-output-len") {
            if (!parse_int_arg(params.planner_output_len)) {
                return false;
            }
        } else if (arg == "--planner-bucket-size") {
            if (!parse_int_arg(params.planner_bucket_size)) {
                return false;
            }
        } else if (arg == "--planner-max-context") {
            if (!parse_int_arg(params.planner_max_context)) {
                return false;
            }
        } else if (arg == "--planner-context-match") {
            if (!e2e_bench_take_arg(argc, argv, i, params.planner_context_match, err)) {
                return false;
            }
        } else if (arg == "--planner-initial-decode-state") {
            if (!e2e_bench_take_arg(argc, argv, i, params.planner_initial_decode_state, err)) {
                return false;
            }
        } else {
            err = "unknown argument: " + arg;
            return false;
        }
    }

    if (params.help) {
        return true;
    }
    if (params.model.empty()) {
        err = "missing required -m/--model";
        return false;
    }
    if (params.n_threads <= 0 || params.n_gen < 0 || params.n_prompt < 0 || params.n_depth < 0 ||
        params.n_batch <= 0 || params.n_ubatch <= 0 || params.reps <= 0 || params.limit < 0) {
        err = "numeric arguments are out of range";
        return false;
    }
    if (params.planner_bucket_size <= 0 || params.planner_max_context <= 0 ||
        params.planner_input_len < -1 || params.planner_output_len < -1) {
        err = "planner numeric arguments are out of range";
        return false;
    }
    if (!params.planner_decode_profile.empty() &&
        params.planner_context_match != "exact" &&
        params.planner_context_match != "floor" &&
        params.planner_context_match != "ceil" &&
        params.planner_context_match != "nearest") {
        err = "--planner-context-match must be exact, floor, ceil, or nearest";
        return false;
    }
    return true;
}

static inline bool e2e_bench_is_directory(const std::string & path) {
    struct stat st {};
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

static inline bool e2e_bench_is_regular_file(const std::string & path) {
    struct stat st {};
    return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

static inline std::string e2e_bench_resolve_dataset_path(
        const std::string & dataset,
        const std::vector<std::string> & names,
        std::string & err) {
    if (dataset.empty()) {
        err = "dataset path is empty";
        return {};
    }

    auto find_name = [&](const std::string & wanted) -> std::string {
        for (const auto & name : names) {
            if (name == wanted) {
                return e2e_bench_join_path(dataset, name);
            }
        }
        return {};
    };

    std::string selected = find_name("sharegpt_gpt4.jsonl");
    if (!selected.empty()) {
        return selected;
    }

    std::vector<std::string> jsonl_names;
    for (const auto & name : names) {
        if (e2e_bench_ends_with(name, ".jsonl")) {
            jsonl_names.push_back(name);
        }
    }
    std::sort(jsonl_names.begin(), jsonl_names.end());
    if (!jsonl_names.empty()) {
        return e2e_bench_join_path(dataset, jsonl_names.front());
    }

    err = "no .jsonl file found in dataset directory: " + dataset;
    return {};
}

static inline std::vector<std::string> e2e_bench_list_dir_names(const std::string & path, std::string & err) {
    std::vector<std::string> names;
    DIR * dir = opendir(path.c_str());
    if (dir == nullptr) {
        err = "failed to open dataset directory " + path + ": " + std::strerror(errno);
        return names;
    }
    while (dirent * entry = readdir(dir)) {
        const std::string name = entry->d_name;
        if (name != "." && name != "..") {
            names.push_back(name);
        }
    }
    closedir(dir);
    return names;
}

static inline std::string e2e_bench_resolve_dataset_path(const std::string & dataset, std::string & err) {
    if (e2e_bench_is_regular_file(dataset)) {
        return dataset;
    }
    if (!e2e_bench_is_directory(dataset)) {
        err = "dataset path is not a file or directory: " + dataset;
        return {};
    }
    const std::vector<std::string> names = e2e_bench_list_dir_names(dataset, err);
    if (!err.empty()) {
        return {};
    }
    return e2e_bench_resolve_dataset_path(dataset, names, err);
}

static inline std::string e2e_bench_first_text_from_array(
        const nlohmann::ordered_json &   arr,
        const std::vector<std::string> & roles) {
    if (!arr.is_array()) {
        return {};
    }
    for (const auto & msg : arr) {
        if (!msg.is_object()) {
            continue;
        }
        const std::string role = msg.value("from", msg.value("role", ""));
        if (std::find(roles.begin(), roles.end(), role) == roles.end()) {
            continue;
        }
        if (msg.contains("value") && msg["value"].is_string()) {
            return msg["value"].get<std::string>();
        }
        if (msg.contains("content") && msg["content"].is_string()) {
            return msg["content"].get<std::string>();
        }
    }
    return {};
}

static inline std::string e2e_bench_prompt_from_json(const nlohmann::ordered_json & item) {
    if (item.contains("conversations")) {
        std::string prompt = e2e_bench_first_text_from_array(item["conversations"], {"human", "user"});
        if (!prompt.empty()) {
            return prompt;
        }
    }
    if (item.contains("messages")) {
        std::string prompt = e2e_bench_first_text_from_array(item["messages"], {"human", "user"});
        if (!prompt.empty()) {
            return prompt;
        }
    }
    if (item.contains("prompt") && item["prompt"].is_string()) {
        return item["prompt"].get<std::string>();
    }
    if (item.contains("text") && item["text"].is_string()) {
        return item["text"].get<std::string>();
    }
    return {};
}

static inline std::string e2e_bench_output_from_json(const nlohmann::ordered_json & item) {
    if (item.contains("conversations")) {
        std::string output = e2e_bench_first_text_from_array(item["conversations"], {"gpt", "assistant"});
        if (!output.empty()) {
            return output;
        }
    }
    if (item.contains("messages")) {
        std::string output = e2e_bench_first_text_from_array(item["messages"], {"assistant", "gpt"});
        if (!output.empty()) {
            return output;
        }
    }
    if (item.contains("output") && item["output"].is_string()) {
        return item["output"].get<std::string>();
    }
    if (item.contains("response") && item["response"].is_string()) {
        return item["response"].get<std::string>();
    }
    if (item.contains("completion") && item["completion"].is_string()) {
        return item["completion"].get<std::string>();
    }
    if (item.contains("prompt") && item["prompt"].is_string() &&
        item.contains("text") && item["text"].is_string()) {
        return item["text"].get<std::string>();
    }
    return {};
}

static inline e2e_bench_sample e2e_bench_sample_from_json(const nlohmann::ordered_json & item) {
    e2e_bench_sample sample;
    sample.prompt = e2e_bench_prompt_from_json(item);
    sample.output = e2e_bench_output_from_json(item);
    return sample;
}

static inline std::vector<e2e_bench_sample> e2e_bench_load_samples(
        const std::string & dataset,
        int limit,
        std::string & err) {
    std::vector<e2e_bench_sample> samples;
    std::string path = e2e_bench_resolve_dataset_path(dataset, err);
    if (!err.empty()) {
        return samples;
    }

    std::ifstream in(path);
    if (!in) {
        err = "failed to open dataset file: " + path;
        return samples;
    }

    std::string line;
    int line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        if (e2e_bench_trim(line).empty()) {
            continue;
        }
        try {
            nlohmann::ordered_json item = nlohmann::ordered_json::parse(line);
            e2e_bench_sample sample = e2e_bench_sample_from_json(item);
            if (!sample.prompt.empty()) {
                samples.push_back(sample);
                if (limit > 0 && (int) samples.size() >= limit) {
                    break;
                }
            }
        } catch (const std::exception & e) {
            err = "failed to parse dataset line " + std::to_string(line_no) + ": " + e.what();
            samples.clear();
            return samples;
        }
    }

    if (samples.empty()) {
        err = "no usable prompts found in dataset: " + path;
    }
    return samples;
}

static inline std::vector<std::string> e2e_bench_load_prompts(
        const std::string & dataset,
        int limit,
        std::string & err) {
    std::vector<std::string> prompts;
    const std::vector<e2e_bench_sample> samples = e2e_bench_load_samples(dataset, limit, err);
    if (!err.empty()) {
        return prompts;
    }
    prompts.reserve(samples.size());
    for (const auto & sample : samples) {
        prompts.push_back(sample.prompt);
    }
    return prompts;
}
