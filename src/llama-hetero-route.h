#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>

enum class llama_hetero_route_stage {
    ATTN,
    ATTN_PROJ,
    ATTN_CORE,
    ATTN_OUT,
    FFN,
    OUTPUT,
};

struct llama_hetero_route_spec {
    std::string attn;
    std::string attn_proj;
    std::string attn_core;
    std::string attn_out;
    std::string ffn;
    std::string output;

    bool has_any_route() const {
        return !attn.empty() ||
               !attn_proj.empty() ||
               !attn_core.empty() ||
               !attn_out.empty() ||
               !ffn.empty() ||
               !output.empty();
    }

    std::string backend_for(llama_hetero_route_stage stage) const {
        switch (stage) {
            case llama_hetero_route_stage::ATTN:
                return attn;
            case llama_hetero_route_stage::ATTN_PROJ:
                return !attn_proj.empty() ? attn_proj : attn;
            case llama_hetero_route_stage::ATTN_CORE:
                return !attn_core.empty() ? attn_core : attn;
            case llama_hetero_route_stage::ATTN_OUT:
                return !attn_out.empty() ? attn_out : attn;
            case llama_hetero_route_stage::FFN:
                return ffn;
            case llama_hetero_route_stage::OUTPUT:
                if (!output.empty()) {
                    return output;
                }
                if (!attn_out.empty()) {
                    return attn_out;
                }
                return attn;
        }

        return {};
    }
};

static inline std::string llama_hetero_trim(std::string_view value) {
    size_t begin = 0;
    size_t end = value.size();

    while (begin < end && std::isspace(static_cast<unsigned char>(value[begin]))) {
        ++begin;
    }
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }

    return std::string(value.substr(begin, end - begin));
}

static inline std::string llama_hetero_to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

static inline std::string llama_hetero_canonical_backend(std::string_view value) {
    std::string normalized = llama_hetero_to_lower(llama_hetero_trim(value));

    if (normalized.empty()) {
        return {};
    }

    if (normalized == "cpu") {
        return "cpu";
    }
    if (normalized == "opencl" || normalized == "gpuopencl" || normalized == "gpu") {
        return "opencl";
    }
    if (normalized == "qnn" || normalized == "qnn-npu" || normalized == "npu" || normalized == "htp0" || normalized == "htp") {
        return "qnn-npu";
    }
    if (normalized == "qnn-gpu") {
        return "qnn-gpu";
    }
    if (normalized == "qnn-cpu") {
        return "qnn-cpu";
    }

    return normalized;
}

static inline int llama_hetero_backend_kind(const std::string & value) {
    if (value.empty()) {
        return 0;
    }

    if (value == "cpu") {
        return 1;
    }
    if (value == "opencl") {
        return 2;
    }
    return 3;
}

static inline bool llama_hetero_name_has_prefix(const char * value, const char * prefix) {
    return value != nullptr && prefix != nullptr && std::strncmp(value, prefix, std::strlen(prefix)) == 0;
}

static inline bool llama_hetero_is_output_tensor_name(const char * tensor_name) {
    return tensor_name != nullptr && (
        std::strcmp(tensor_name, "result_norm") == 0 ||
        std::strcmp(tensor_name, "result_output") == 0);
}

static inline bool llama_hetero_is_attn_proj_tensor_name(const char * tensor_name) {
    return tensor_name != nullptr && (
        llama_hetero_name_has_prefix(tensor_name, "norm-") ||
        llama_hetero_name_has_prefix(tensor_name, "attn_norm-") ||
        llama_hetero_name_has_prefix(tensor_name, "Qcur") ||
        llama_hetero_name_has_prefix(tensor_name, "Kcur") ||
        llama_hetero_name_has_prefix(tensor_name, "Vcur"));
}

static inline bool llama_hetero_is_attn_core_tensor_name(const char * tensor_name) {
    return tensor_name != nullptr && (
        llama_hetero_name_has_prefix(tensor_name, "__fattn__-") ||
        llama_hetero_name_has_prefix(tensor_name, "fattn") ||
        llama_hetero_name_has_prefix(tensor_name, "kq") ||
        llama_hetero_name_has_prefix(tensor_name, "kqv") ||
        llama_hetero_name_has_prefix(tensor_name, "v_cont-"));
}

static inline bool llama_hetero_is_attn_out_tensor_name(const char * tensor_name) {
    return llama_hetero_name_has_prefix(tensor_name, "attn_out-");
}

static inline bool llama_hetero_is_ffn_tensor_name(const char * tensor_name) {
    return tensor_name != nullptr && (
        llama_hetero_name_has_prefix(tensor_name, "ffn") ||
        llama_hetero_name_has_prefix(tensor_name, "l_out-"));
}

static inline bool llama_hetero_is_stage_tensor_name(const char * tensor_name) {
    return llama_hetero_is_attn_proj_tensor_name(tensor_name) ||
           llama_hetero_is_attn_core_tensor_name(tensor_name) ||
           llama_hetero_is_attn_out_tensor_name(tensor_name) ||
           llama_hetero_is_ffn_tensor_name(tensor_name) ||
           (tensor_name != nullptr && (
               std::strcmp(tensor_name, "embd") == 0 ||
               std::strcmp(tensor_name, "norm") == 0)) ||
           llama_hetero_is_output_tensor_name(tensor_name);
}

static inline void llama_hetero_set_route_field(llama_hetero_route_spec & spec, const std::string & key, const std::string & value) {
    if (key == "attn" || key == "attention") {
        spec.attn = value;
    } else if (key == "attn_proj" || key == "attn_projection" || key == "attention_projection" || key == "qkv_proj") {
        spec.attn_proj = value;
    } else if (key == "attn_core" || key == "attention_core") {
        spec.attn_core = value;
    } else if (key == "attn_out" || key == "attn_output" || key == "attention_output" || key == "o_proj") {
        spec.attn_out = value;
    } else if (key == "ffn" || key == "mlp") {
        spec.ffn = value;
    } else if (key == "output" || key == "tail" || key == "decode_output") {
        spec.output = value;
    }
}

static inline const char * llama_hetero_route_env_value() {
    const char * value = std::getenv("GGML_HETERO_STAGE_ROUTE");
    if (value != nullptr && value[0] != '\0') {
        return value;
    }

    value = std::getenv("GGML_HETERO_ROUTE");
    if (value != nullptr && value[0] != '\0') {
        return value;
    }

    return nullptr;
}

static inline llama_hetero_route_spec llama_hetero_parse_route_spec_from_env() {
    llama_hetero_route_spec spec;

    const char * route_env = llama_hetero_route_env_value();
    if (route_env != nullptr) {
        std::string route(route_env);
        size_t begin = 0;

        while (begin < route.size()) {
            size_t end = route.find_first_of(",;", begin);
            if (end == std::string::npos) {
                end = route.size();
            }

            std::string token = llama_hetero_trim(std::string_view(route).substr(begin, end - begin));
            if (!token.empty()) {
                size_t sep = token.find('=');
                if (sep == std::string::npos) {
                    sep = token.find(':');
                }

                if (sep != std::string::npos) {
                    const std::string key = llama_hetero_to_lower(llama_hetero_trim(std::string_view(token).substr(0, sep)));
                    const std::string value = llama_hetero_canonical_backend(std::string_view(token).substr(sep + 1));
                    if (!key.empty() && !value.empty()) {
                        llama_hetero_set_route_field(spec, key, value);
                    }
                }
            }

            begin = end + 1;
        }

        return spec;
    }

    spec.attn = llama_hetero_canonical_backend(std::getenv("GGML_HETERO_ATTN_BACKEND") ? std::getenv("GGML_HETERO_ATTN_BACKEND") : "");
    spec.ffn  = llama_hetero_canonical_backend(std::getenv("GGML_HETERO_FFN_BACKEND")  ? std::getenv("GGML_HETERO_FFN_BACKEND")  : "");

    return spec;
}

static inline bool llama_hetero_route_has_cpu_opencl_mix(const llama_hetero_route_spec & spec) {
    bool has_cpu = false;
    bool has_opencl = false;

    const std::array<llama_hetero_route_stage, 5> stages = {{
        llama_hetero_route_stage::ATTN_PROJ,
        llama_hetero_route_stage::ATTN_CORE,
        llama_hetero_route_stage::ATTN_OUT,
        llama_hetero_route_stage::FFN,
        llama_hetero_route_stage::OUTPUT,
    }};

    for (const auto stage : stages) {
        const int kind = llama_hetero_backend_kind(spec.backend_for(stage));
        if (kind == 1) {
            has_cpu = true;
        } else if (kind == 2) {
            has_opencl = true;
        }
    }

    return has_cpu && has_opencl;
}
