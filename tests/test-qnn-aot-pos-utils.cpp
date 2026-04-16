#include "../ggml/src/ggml-qnn/qnn/aot-pos-utils.hpp"

#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static bool expect_true(bool condition, const char * label) {
    if (condition) {
        return true;
    }

    std::fprintf(stderr, "%s: expected true, got false\n", label);
    return false;
}

static bool expect_false(bool condition, const char * label) {
    if (!condition) {
        return true;
    }

    std::fprintf(stderr, "%s: expected false, got true\n", label);
    return false;
}

static bool expect_eq_size(size_t actual, size_t expected, const char * label) {
    if (actual == expected) {
        return true;
    }

    std::fprintf(stderr, "%s: expected %zu, got %zu\n", label, expected, actual);
    return false;
}

template <typename T>
static ggml_tensor * make_tensor_1d(ggml_context * ctx, ggml_type type, const std::vector<T> & values) {
    ggml_tensor * tensor = ggml_new_tensor_1d(ctx, type, static_cast<int64_t>(values.size()));
    std::memcpy(tensor->data, values.data(), values.size() * sizeof(T));
    return tensor;
}

int main(void) {
    ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };

    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to initialize ggml context\n");
        return 1;
    }

    bool ok = true;

    auto * pos_i32 = make_tensor_1d<int32_t>(ctx, GGML_TYPE_I32, {22, 23});
    auto * slot_i64 = make_tensor_1d<int64_t>(ctx, GGML_TYPE_I64, {22});
    auto * broken_i64 = make_tensor_1d<int64_t>(ctx, GGML_TYPE_I64, {22, 24});

    size_t inferred = 0;
    ok &= expect_true(
        qnn::qnn_aot_try_infer_contiguous_start_pos(pos_i32, 2, inferred),
        "I32 positions should infer a start position");
    ok &= expect_eq_size(inferred, 22, "I32 positions should keep the first slot");

    inferred = 0;
    ok &= expect_true(
        qnn::qnn_aot_try_infer_contiguous_start_pos(slot_i64, 1, inferred),
        "I64 KV slots should infer a start position");
    ok &= expect_eq_size(inferred, 22, "I64 KV slots should expose the first slot");

    inferred = 0;
    ok &= expect_false(
        qnn::qnn_aot_try_infer_contiguous_start_pos(broken_i64, 2, inferred),
        "non-contiguous KV slots must be rejected");

    const std::vector<ggml_tensor *> inputs = {broken_i64, pos_i32};
    ok &= expect_eq_size(
        qnn::qnn_aot_infer_start_pos_from_inputs(inputs, 2, 7),
        22,
        "input scan should ignore invalid slot tensors and keep the best valid start");

    ggml_free(ctx);
    return ok ? 0 : 1;
}
