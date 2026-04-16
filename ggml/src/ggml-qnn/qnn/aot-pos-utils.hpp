#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace qnn {

inline ggml_backend_buffer_t qnn_aot_tensor_buffer(const ggml_tensor * tensor) {
    return tensor == nullptr ? nullptr : (tensor->view_src ? tensor->view_src->buffer : tensor->buffer);
}

inline bool qnn_aot_tensor_has_host_accessible_data(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->data == nullptr) {
        return false;
    }

    ggml_backend_buffer_t buffer = qnn_aot_tensor_buffer(tensor);
    if (buffer == nullptr) {
        return true;
    }

    if (!ggml_backend_buffer_is_host(buffer)) {
        return false;
    }

    void * base = ggml_backend_buffer_get_base(buffer);
    if (base == nullptr) {
        return false;
    }

    const uintptr_t data_addr = reinterpret_cast<uintptr_t>(tensor->data);
    const uintptr_t base_addr = reinterpret_cast<uintptr_t>(base);
    const uintptr_t end_addr  = base_addr + ggml_backend_buffer_get_size(buffer);
    return data_addr >= base_addr && data_addr < end_addr;
}

template <typename T>
inline bool qnn_aot_copy_tensor_prefix(const ggml_tensor * tensor,
                                       size_t              n_values,
                                       std::vector<T> &    out_values) {
    if (tensor == nullptr || tensor->data == nullptr || n_values == 0 || static_cast<size_t>(tensor->ne[0]) < n_values) {
        return false;
    }

    out_values.resize(n_values);
    const size_t bytes = n_values * sizeof(T);

    if (qnn_aot_tensor_has_host_accessible_data(tensor)) {
        std::memcpy(out_values.data(), tensor->data, bytes);
        return true;
    }

    ggml_backend_buffer_t buffer = qnn_aot_tensor_buffer(tensor);
    if (buffer == nullptr) {
        return false;
    }

    ggml_backend_tensor_get(tensor, out_values.data(), 0, bytes);
    return true;
}

inline bool qnn_aot_try_infer_contiguous_start_pos(const ggml_tensor * tensor,
                                                   size_t              n_tokens,
                                                   size_t &            out_start_pos) {
    if (tensor == nullptr || ggml_n_dims(tensor) != 1 || n_tokens == 0) {
        return false;
    }

    if (tensor->type == GGML_TYPE_I32) {
        std::vector<int32_t> values;
        if (!qnn_aot_copy_tensor_prefix(tensor, n_tokens, values)) {
            return false;
        }
        if (values[0] < 0) {
            return false;
        }

        for (size_t i = 1; i < n_tokens; ++i) {
            if (values[i] != values[0] + static_cast<int32_t>(i)) {
                return false;
            }
        }

        out_start_pos = static_cast<size_t>(values[0]);
        return true;
    }

    if (tensor->type == GGML_TYPE_I64) {
        std::vector<int64_t> values;
        if (!qnn_aot_copy_tensor_prefix(tensor, n_tokens, values)) {
            return false;
        }
        if (values[0] < 0) {
            return false;
        }

        for (size_t i = 1; i < n_tokens; ++i) {
            if (values[i] != values[0] + static_cast<int64_t>(i)) {
                return false;
            }
        }

        out_start_pos = static_cast<size_t>(values[0]);
        return true;
    }

    return false;
}

inline size_t qnn_aot_infer_start_pos_from_inputs(const std::vector<ggml_tensor *> & inputs,
                                                  size_t                              n_tokens,
                                                  size_t                              fallback_pos) {
    size_t inferred = fallback_pos;

    for (auto * tensor : inputs) {
        size_t candidate = 0;
        if (qnn_aot_try_infer_contiguous_start_pos(tensor, n_tokens, candidate)) {
            inferred = std::max(inferred, candidate);
        }
    }

    return inferred;
}

} // namespace qnn
