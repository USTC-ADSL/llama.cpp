#include "../ggml/src/ggml-qnn/qnn/aot-capacity-utils.hpp"

#include "ggml-backend.h"

#include <cstdio>
#include <string>
#include <vector>

bool ggml_backend_qnn_aot_set_required_kv_capacity(
        ggml_backend_t backend,
        size_t         required_kv_slots,
        size_t         preferred_context_size);

bool ggml_backend_qnn_aot_preload_decode_graphs_for_capacity(
        ggml_backend_t backend,
        size_t         n_tokens,
        size_t         required_kv_slots,
        size_t         preferred_context_size);

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

static bool expect_eq_string(const std::string & actual, const std::string & expected, const char * label) {
    if (actual == expected) {
        return true;
    }

    std::fprintf(stderr, "%s: expected '%s', got '%s'\n", label, expected.c_str(), actual.c_str());
    return false;
}

static qnn::qnn_aot_graph_capacity_view graph_view(
        size_t      batch_size,
        size_t      cache_size,
        size_t      context_size,
        const char * model_path) {
    qnn::qnn_aot_graph_capacity_view view;
    view.batch_size = batch_size;
    view.cache_size = cache_size;
    view.context_size = context_size;
    view.model_path = model_path;
    return view;
}

static bool expect_identity(
        const qnn::qnn_aot_capacity_identity & identity,
        const char *                          model_path,
        size_t                                cache_size,
        size_t                                context_size,
        const char *                          label) {
    bool ok = true;
    ok &= expect_eq_string(identity.model_path, model_path, label);
    ok &= expect_eq_size(identity.cache_size, cache_size, label);
    ok &= expect_eq_size(identity.context_size, context_size, label);
    return ok;
}

int main(void) {
    bool ok = true;

    ok &= expect_false(
        ggml_backend_qnn_aot_set_required_kv_capacity(nullptr, 2500, 4096),
        "capacity setter backend proc should reject null backend");
    ok &= expect_false(
        ggml_backend_qnn_aot_preload_decode_graphs_for_capacity(nullptr, 1, 2500, 4096),
        "capacity preload backend proc should reject null backend");

    const std::vector<qnn::qnn_aot_graph_capacity_view> graphs = {
        graph_view(1,   1920, 2048, "qnn-2k.bin"),
        graph_view(128, 1920, 2048, "qnn-2k.bin"),
        graph_view(1,   3968, 4096, "qnn-4k.bin"),
        graph_view(128, 3968, 4096, "qnn-4k.bin"),
        graph_view(1,   6016, 6144, "qnn-6k.bin"),
        graph_view(128, 6016, 6144, "qnn-6k.bin"),
    };

    ok &= expect_eq_size(
        qnn::qnn_aot_select_batch_size(graphs, 1),
        1,
        "single-token decode should select batch 1");

    ok &= expect_eq_size(
        qnn::qnn_aot_select_batch_size(graphs, 64),
        128,
        "prefill chunks smaller than 128 should select batch 128");

    ok &= expect_eq_size(
        qnn::qnn_aot_select_batch_size(graphs, 256),
        128,
        "oversized chunks should fall back to the largest available batch");

    {
        qnn::qnn_aot_capacity_identity identity;
        qnn::qnn_aot_capacity_request request;
        request.n_tokens = 1;
        request.required_kv_slots = 2500;
        request.preferred_context_size = 2048;

        ok &= expect_true(
            qnn::qnn_aot_select_capacity_identity(graphs, request, identity),
            "capacity selection should fall back when preferred context is too small");
        ok &= expect_identity(
            identity,
            "qnn-4k.bin",
            3968,
            4096,
            "required KV slots should guard against choosing the 2K graph");
    }

    {
        qnn::qnn_aot_capacity_identity identity;
        qnn::qnn_aot_capacity_request request;
        request.n_tokens = 1;
        request.required_kv_slots = 2500;
        request.preferred_context_size = 4096;

        ok &= expect_true(
            qnn::qnn_aot_select_capacity_identity(graphs, request, identity),
            "preferred safe context should be selectable");
        ok &= expect_identity(
            identity,
            "qnn-4k.bin",
            3968,
            4096,
            "preferred safe context should win exactly");
    }

    {
        qnn::qnn_aot_capacity_identity identity;
        qnn::qnn_aot_capacity_request request;
        request.n_tokens = 1;
        request.required_kv_slots = 2500;
        request.preferred_context_size = 8192;

        ok &= expect_true(
            qnn::qnn_aot_select_capacity_identity(graphs, request, identity),
            "missing preferred context should fall back to smallest sufficient cache");
        ok &= expect_identity(
            identity,
            "qnn-4k.bin",
            3968,
            4096,
            "smallest sufficient cache should win when preferred context is absent");
    }

    {
        qnn::qnn_aot_capacity_identity identity;
        qnn::qnn_aot_capacity_request request;
        request.n_tokens = 1;
        request.required_kv_slots = 7000;
        request.preferred_context_size = 6144;

        ok &= expect_false(
            qnn::qnn_aot_select_capacity_identity(graphs, request, identity),
            "selection should fail when no graph has enough KV capacity");
    }

    {
        qnn::qnn_aot_capacity_identity identity;
        qnn::qnn_aot_capacity_request request;
        request.n_tokens = 1;
        request.required_kv_slots = 2500;
        request.preferred_context_size = 4096;

        ok &= expect_true(
            qnn::qnn_aot_select_capacity_identity(graphs, request, identity),
            "identity should be available for chain filtering");
        ok &= expect_true(
            qnn::qnn_aot_capacity_identity_matches(graph_view(1, 3968, 4096, "qnn-4k.bin"), identity),
            "matching capacity graph should stay in the selected chain");
        ok &= expect_false(
            qnn::qnn_aot_capacity_identity_matches(graph_view(1, 1920, 2048, "qnn-2k.bin"), identity),
            "mixed 2K graph should be rejected from a 4K chain");
        ok &= expect_false(
            qnn::qnn_aot_capacity_identity_matches(graph_view(1, 3968, 4096, "other-4k.bin"), identity),
            "same numeric capacity from a different model path should not share state");
    }

    {
        qnn::qnn_aot_capacity_identity decode_identity;
        qnn::qnn_aot_capacity_request decode_request;
        decode_request.n_tokens = 1;
        decode_request.required_kv_slots = 2500;
        decode_request.preferred_context_size = 4096;

        qnn::qnn_aot_capacity_identity prefill_identity;
        qnn::qnn_aot_capacity_request prefill_request;
        prefill_request.n_tokens = 128;
        prefill_request.required_kv_slots = 2500;
        prefill_request.preferred_context_size = 4096;

        ok &= expect_true(
            qnn::qnn_aot_select_capacity_identity(graphs, decode_request, decode_identity),
            "batch 1 decode should select a capacity identity");
        ok &= expect_true(
            qnn::qnn_aot_select_capacity_identity(graphs, prefill_request, prefill_identity),
            "batch 128 prefill should select a capacity identity");
        ok &= expect_true(
            qnn::qnn_aot_capacity_identity_matches(
                graph_view(
                    128,
                    prefill_identity.cache_size,
                    prefill_identity.context_size,
                    prefill_identity.model_path.c_str()),
                decode_identity),
            "batch 1 and batch 128 should be able to share the same capacity identity");
    }

    {
        const std::vector<qnn::qnn_aot_graph_capacity_view> mixed_bucket = {
            graph_view(1, 1920, 2048, "qnn-2k.bin"),
            graph_view(1, 3968, 4096, "qnn-4k.bin"),
            graph_view(1, 3968, 4096, "qnn-4k.bin"),
        };

        qnn::qnn_aot_capacity_identity identity;
        qnn::qnn_aot_capacity_request request;
        request.n_tokens = 1;
        request.required_kv_slots = 2500;
        request.preferred_context_size = 4096;

        const auto chain = qnn::qnn_aot_select_capacity_chain(mixed_bucket, request, &identity);

        ok &= expect_eq_size(
            chain.size(),
            2,
            "mixed 2K/4K bucket should return only the selected 4K chain");
        for (const auto & view : chain) {
            ok &= expect_true(
                qnn::qnn_aot_capacity_identity_matches(view, identity),
                "selected chain entries must share one capacity identity");
        }
        ok &= expect_identity(
            identity,
            "qnn-4k.bin",
            3968,
            4096,
            "selected chain identity should be the requested 4K graph");
    }

    {
        const std::vector<qnn::qnn_aot_graph_capacity_view> mixed_bucket = {
            graph_view(1, 1920, 2048, "qnn-2k.bin"),
            graph_view(1, 1920, 2048, "qnn-2k.bin"),
            graph_view(1, 3968, 4096, "qnn-4k.bin"),
        };

        qnn::qnn_aot_capacity_identity identity;
        qnn::qnn_aot_capacity_request request;
        request.n_tokens = 1;

        const auto chain = qnn::qnn_aot_select_capacity_chain(mixed_bucket, request, &identity);

        ok &= expect_eq_size(
            chain.size(),
            2,
            "default batch-only compatibility should still choose one internally consistent capacity chain");
        ok &= expect_identity(
            identity,
            "qnn-2k.bin",
            1920,
            2048,
            "default mixed-capacity selection should choose the smallest sufficient capacity");
    }

    {
        const std::vector<qnn::qnn_aot_graph_kv_cursor_view> cursor_views = {
            { graph_view(1, 1920, 2048, "qnn-2k.bin"), 0, 24, 1800 },
            { graph_view(1, 3968, 4096, "qnn-4k.bin"), 0, 12, 0 },
            { graph_view(1, 3968, 4096, "qnn-4k.bin"), 12, 24, 32 },
        };

        qnn::qnn_aot_capacity_identity identity;
        identity.model_path = "qnn-4k.bin";
        identity.cache_size = 3968;
        identity.context_size = 4096;

        size_t active_cursor = 9999;
        ok &= expect_true(
            qnn::qnn_aot_select_active_kv_cursor(cursor_views, identity, active_cursor),
            "selected capacity chain should expose an active private KV cursor");
        ok &= expect_eq_size(
            active_cursor,
            0,
            "selected capacity active cursor should use the minimum cursor across its stateful chain");
        ok &= expect_eq_size(
            qnn::qnn_aot_kv_cursor_after_prefix_import(active_cursor, 1800),
            1800,
            "selected capacity cursor should advance to the imported generic prefix");
    }

    {
        const std::vector<qnn::qnn_aot_graph_capacity_view> mixed_graphs = {
            graph_view(1, 1920, 2048, "qnn-2k.bin"),
            graph_view(1, 1920, 2048, "qnn-2k.bin"),
            graph_view(1, 3968, 4096, "qnn-4k.bin"),
            graph_view(1, 3968, 4096, "qnn-4k.bin"),
            graph_view(1, 6016, 6144, "qnn-6k.bin"),
            graph_view(1, 6016, 6144, "qnn-6k.bin"),
        };
        const std::vector<qnn::qnn_aot_graph_kv_cursor_view> mixed_cursors = {
            { graph_view(1, 1920, 2048, "qnn-2k.bin"), 0, 12, 1792 },
            { graph_view(1, 1920, 2048, "qnn-2k.bin"), 12, 24, 1800 },
            { graph_view(1, 3968, 4096, "qnn-4k.bin"), 0, 12, 0 },
            { graph_view(1, 3968, 4096, "qnn-4k.bin"), 12, 24, 24 },
            { graph_view(1, 6016, 6144, "qnn-6k.bin"), 0, 12, 0 },
            { graph_view(1, 6016, 6144, "qnn-6k.bin"), 12, 24, 0 },
        };

        qnn::qnn_aot_capacity_request request;
        request.n_tokens = 1;
        request.required_kv_slots = 2500;
        request.preferred_context_size = 4096;

        qnn::qnn_aot_capacity_identity selected_identity;
        const auto selected_chain =
            qnn::qnn_aot_select_capacity_chain(mixed_graphs, request, &selected_identity);

        ok &= expect_eq_size(
            selected_chain.size(),
            2,
            "integrated capacity switch should select only the 4K graph shards");
        ok &= expect_identity(
            selected_identity,
            "qnn-4k.bin",
            3968,
            4096,
            "integrated capacity switch should select the requested 4K identity");
        for (const auto & view : selected_chain) {
            ok &= expect_true(
                qnn::qnn_aot_capacity_identity_matches(view, selected_identity),
                "integrated selected graph shard should match the selected capacity identity");
        }

        size_t active_cursor = 9999;
        ok &= expect_true(
            qnn::qnn_aot_select_active_kv_cursor(mixed_cursors, selected_identity, active_cursor),
            "integrated selected capacity should resolve a private KV cursor");
        ok &= expect_eq_size(
            active_cursor,
            0,
            "integrated selected capacity must not reuse the loaded 2K private KV cursor");
        ok &= expect_eq_size(
            qnn::qnn_aot_kv_cursor_after_prefix_import(active_cursor, 1800),
            1800,
            "integrated selected capacity should advance from generic prefix import after switching");
    }

    return ok ? 0 : 1;
}
