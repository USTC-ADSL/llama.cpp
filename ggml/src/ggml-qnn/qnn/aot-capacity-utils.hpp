#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace qnn {

struct qnn_aot_capacity_identity {
    std::string model_path;
    size_t      cache_size = 0;
    size_t      context_size = 0;
};

struct qnn_aot_capacity_request {
    size_t n_tokens = 1;
    size_t required_kv_slots = 0;
    size_t preferred_context_size = 0;
};

struct qnn_aot_graph_capacity_view {
    size_t      batch_size = 0;
    size_t      cache_size = 0;
    size_t      context_size = 0;
    std::string model_path;
};

struct qnn_aot_graph_kv_cursor_view {
    qnn_aot_graph_capacity_view capacity;
    size_t                      start_layer_id = 0;
    size_t                      end_layer_id = 0;
    size_t                      kv_position = 0;
};

inline bool qnn_aot_capacity_identity_matches(
        const qnn_aot_graph_capacity_view & view,
        const qnn_aot_capacity_identity &   identity) {
    return view.model_path == identity.model_path &&
           view.cache_size == identity.cache_size &&
           view.context_size == identity.context_size;
}

inline size_t qnn_aot_select_batch_size(
        const std::vector<qnn_aot_graph_capacity_view> & graphs,
        size_t                                           n_tokens) {
    const size_t target_tokens = n_tokens > 0 ? n_tokens : 1;

    bool   have_ge = false;
    bool   have_lt = false;
    size_t best_ge = 0;
    size_t best_lt = 0;

    for (const auto & graph : graphs) {
        if (graph.batch_size == 0) {
            continue;
        }

        if (graph.batch_size >= target_tokens) {
            if (!have_ge || graph.batch_size < best_ge) {
                best_ge = graph.batch_size;
                have_ge = true;
            }
            continue;
        }

        if (!have_lt || graph.batch_size > best_lt) {
            best_lt = graph.batch_size;
            have_lt = true;
        }
    }

    if (have_ge) {
        return best_ge;
    }

    if (have_lt) {
        return best_lt;
    }

    return 0;
}

inline bool qnn_aot_select_capacity_identity(
        const std::vector<qnn_aot_graph_capacity_view> & graphs,
        const qnn_aot_capacity_request &                 request,
        qnn_aot_capacity_identity &                      out_identity) {
    const size_t batch_size = qnn_aot_select_batch_size(graphs, request.n_tokens);
    if (batch_size == 0) {
        return false;
    }

    const qnn_aot_graph_capacity_view * preferred = nullptr;
    const qnn_aot_graph_capacity_view * fallback = nullptr;
    const auto is_better = [](const qnn_aot_graph_capacity_view & lhs,
                              const qnn_aot_graph_capacity_view & rhs) {
        if (lhs.cache_size != rhs.cache_size) {
            return lhs.cache_size < rhs.cache_size;
        }
        if (lhs.context_size != rhs.context_size) {
            return lhs.context_size < rhs.context_size;
        }
        return lhs.model_path < rhs.model_path;
    };

    for (const auto & graph : graphs) {
        if (graph.batch_size != batch_size) {
            continue;
        }
        if (graph.cache_size < request.required_kv_slots) {
            continue;
        }

        if (request.preferred_context_size > 0 &&
            graph.context_size == request.preferred_context_size &&
            (preferred == nullptr || is_better(graph, *preferred))) {
            preferred = &graph;
        }

        if (fallback == nullptr || is_better(graph, *fallback)) {
            fallback = &graph;
        }
    }

    const qnn_aot_graph_capacity_view * selected = preferred != nullptr ? preferred : fallback;
    if (selected == nullptr) {
        return false;
    }

    out_identity.model_path = selected->model_path;
    out_identity.cache_size = selected->cache_size;
    out_identity.context_size = selected->context_size;
    return true;
}

inline std::vector<qnn_aot_graph_capacity_view> qnn_aot_select_capacity_chain(
        const std::vector<qnn_aot_graph_capacity_view> & graphs,
        const qnn_aot_capacity_request &                 request,
        qnn_aot_capacity_identity *                      out_identity = nullptr) {
    std::vector<qnn_aot_graph_capacity_view> chain;

    qnn_aot_capacity_identity identity;
    if (!qnn_aot_select_capacity_identity(graphs, request, identity)) {
        return chain;
    }

    const size_t batch_size = qnn_aot_select_batch_size(graphs, request.n_tokens);
    if (batch_size == 0) {
        return chain;
    }

    for (const auto & graph : graphs) {
        if (graph.batch_size == batch_size &&
            qnn_aot_capacity_identity_matches(graph, identity)) {
            chain.push_back(graph);
        }
    }

    if (out_identity != nullptr) {
        *out_identity = identity;
    }

    return chain;
}

inline bool qnn_aot_select_active_kv_cursor(
        const std::vector<qnn_aot_graph_kv_cursor_view> & graphs,
        const qnn_aot_capacity_identity &                 identity,
        size_t &                                          out_cursor) {
    bool found = false;
    size_t cursor = 0;

    for (const auto & graph : graphs) {
        if (!qnn_aot_capacity_identity_matches(graph.capacity, identity)) {
            continue;
        }

        if (graph.end_layer_id <= graph.start_layer_id) {
            continue;
        }

        if (!found || graph.kv_position < cursor) {
            cursor = graph.kv_position;
        }
        found = true;
    }

    if (!found) {
        return false;
    }

    out_cursor = cursor;
    return true;
}

inline size_t qnn_aot_kv_cursor_after_prefix_import(
        size_t graph_kv_position,
        size_t required_prefix_tokens) {
    return graph_kv_position < required_prefix_tokens ? required_prefix_tokens : graph_kv_position;
}

} // namespace qnn
