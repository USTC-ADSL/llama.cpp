#pragma once

#include <cctype>
#include <cmath>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace qnn {

inline std::string_view qnn_aot_trim_ascii(std::string_view value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front())) != 0) {
        value.remove_prefix(1);
    }
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back())) != 0) {
        value.remove_suffix(1);
    }
    return value;
}

inline std::string qnn_aot_lower_ascii(std::string_view value) {
    std::string lowered;
    lowered.reserve(value.size());
    for (const char c : value) {
        lowered.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    return lowered;
}

inline bool qnn_aot_backend_is_qnn(std::string_view backend) {
    const std::string normalized = qnn_aot_lower_ascii(qnn_aot_trim_ascii(backend));
    return normalized == "qnn" ||
           normalized == "qnn-npu" ||
           normalized == "qnn-htp" ||
           normalized == "npu" ||
           normalized == "htp";
}

inline bool qnn_aot_backend_is_non_qnn(std::string_view backend) {
    backend = qnn_aot_trim_ascii(backend);
    return !backend.empty() && !qnn_aot_backend_is_qnn(backend);
}

inline bool qnn_aot_route_attention_uses_non_qnn_backend(std::string_view route_spec) {
    route_spec = qnn_aot_trim_ascii(route_spec);
    if (route_spec.empty()) {
        return false;
    }

    if (route_spec.find('=') == std::string_view::npos) {
        return qnn_aot_backend_is_non_qnn(route_spec);
    }

    size_t pos = 0;
    while (pos < route_spec.size()) {
        const size_t next = route_spec.find(',', pos);
        const std::string_view raw_part =
            route_spec.substr(pos, next == std::string_view::npos ? std::string_view::npos : next - pos);
        const std::string_view part = qnn_aot_trim_ascii(raw_part);
        const size_t split = part.find('=');
        if (split != std::string_view::npos) {
            const std::string key = qnn_aot_lower_ascii(qnn_aot_trim_ascii(part.substr(0, split)));
            const std::string_view value = qnn_aot_trim_ascii(part.substr(split + 1));
            if ((key == "attn" ||
                 key == "attn_proj" ||
                 key == "attn_core" ||
                 key == "attn_out") &&
                qnn_aot_backend_is_non_qnn(value)) {
                return true;
            }
        }

        if (next == std::string_view::npos) {
            break;
        }
        pos = next + 1;
    }

    return false;
}

inline bool qnn_aot_decode_schedule_attention_uses_non_qnn_backend(const char * decode_schedule) {
    const std::string_view schedule = decode_schedule != nullptr ? std::string_view(decode_schedule) : std::string_view();
    size_t pos = 0;
    while (pos < schedule.size()) {
        const size_t next = schedule.find(';', pos);
        const std::string_view raw_entry =
            schedule.substr(pos, next == std::string_view::npos ? std::string_view::npos : next - pos);
        const std::string_view entry = qnn_aot_trim_ascii(raw_entry);
        const size_t split = entry.find(':');
        if (split != std::string_view::npos) {
            const std::string_view route_spec = qnn_aot_trim_ascii(entry.substr(split + 1));
            if (qnn_aot_route_attention_uses_non_qnn_backend(route_spec)) {
                return true;
            }
        }

        if (next == std::string_view::npos) {
            break;
        }
        pos = next + 1;
    }

    return false;
}

// QNN AoT transformer graphs expose attention-scaled K rows
// (raw_k / sqrt(head_dim)). Generic CPU/OpenCL KV consumers expect raw K.
inline bool qnn_aot_restore_unscaled_key_rows_for_generic_kv(std::vector<float> & key_rows,
                                                             size_t               n_tokens,
                                                             size_t               token_values,
                                                             size_t               n_kv_heads,
                                                             size_t               head_dim) {
    if (token_values == 0 || n_kv_heads == 0 || head_dim == 0) {
        return false;
    }

    if (token_values != n_kv_heads * head_dim) {
        return false;
    }

    if (key_rows.size() != n_tokens * token_values) {
        return false;
    }

    const float scale = std::sqrt(static_cast<float>(head_dim));
    for (float & value : key_rows) {
        value *= scale;
    }

    return true;
}

// QNN private KV caches store K rows in the attention-scaled form emitted by the
// AoT graph. Generic CPU/OpenCL KV stores raw K rows. Importing generic KV back
// into a QNN graph must therefore undo the generic raw-K representation.
inline bool qnn_aot_apply_attention_scaled_key_rows_for_private_kv(std::vector<float> & key_rows,
                                                                   size_t               n_tokens,
                                                                   size_t               token_values,
                                                                   size_t               n_kv_heads,
                                                                   size_t               head_dim) {
    if (token_values == 0 || n_kv_heads == 0 || head_dim == 0) {
        return false;
    }

    if (token_values != n_kv_heads * head_dim) {
        return false;
    }

    if (key_rows.size() != n_tokens * token_values) {
        return false;
    }

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (float & value : key_rows) {
        value *= scale;
    }

    return true;
}

// Deferred generic-KV staging may span multiple transformer graph shards for the
// same prefill. Only the first shard of a fresh prefill (layer 0 at token offset
// 0) should discard previously staged payloads; later shards must preserve them.
inline bool qnn_aot_should_reset_staged_generic_kv_writeback(size_t token_offset,
                                                             size_t graph_start_layer_id,
                                                             size_t pending_layers) {
    if (token_offset != 0) {
        return false;
    }

    if (graph_start_layer_id == 0) {
        return true;
    }

    // With no staged layers yet, a reset is a no-op and keeps the caller logic
    // simple even if the first observed shard does not start at layer 0.
    return pending_layers == 0;
}

inline bool qnn_aot_should_write_generic_kv(bool   generic_kv_writeback_needed,
                                            size_t n_tokens,
                                            bool   has_kq_mask,
                                            bool   has_cache_k_layers,
                                            bool   has_cache_v_layers) {
    return generic_kv_writeback_needed &&
           n_tokens > 0 &&
           has_kq_mask &&
           has_cache_k_layers &&
           has_cache_v_layers;
}

inline bool qnn_aot_kv_prefix_import_required(size_t graph_kv_position,
                                              size_t required_prefix_tokens) {
    return graph_kv_position < required_prefix_tokens;
}

} // namespace qnn
