#include "../src/llama-hetero-route.h"
#include "../src/llama-context.h"
#include "../src/llama-dyn-route.h"
#include "../src/llama-kv-cache.h"
#include "testing.h"

#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

llama_hetero_kv_contract llama_dynamic_phase_migration_kv_contract(
        const std::string & producer_backend,
        const std::string & consumer_backend,
        const char * reason);

bool llama_context_should_attempt_qnn_phase_kv_migration(
        const std::string & current_attn_backend,
        const std::string & target_attn_backend,
        uint32_t            n_tokens,
        bool                generic_kv_enabled);

bool llama_context_should_sync_opencl_before_qnn_direct_import(
        const std::string & current_attn_backend,
        const std::string & target_attn_backend,
        uint32_t            n_tokens,
        bool                generic_kv_enabled);

llama_opencl_external_host_sync_scope llama_context_opencl_sync_scope_for_qnn_direct_import(
        const std::string & current_attn_backend,
        const std::string & target_attn_backend,
        uint32_t            n_tokens,
        bool                generic_kv_enabled);

bool llama_context_should_use_qnn_written_generic_kv_for_cpu(
        const std::string & current_attn_backend,
        const std::string & target_attn_backend,
        uint32_t            n_tokens,
        bool                generic_kv_enabled,
        bool                qnn_writeback_ready,
        bool                live_kv_cpu_accessible,
        bool                qnn_writeback_flushed);

bool llama_context_kv_buft_is_cpu_accessible(ggml_backend_buffer_type_t buft);

bool llama_context_should_try_qnn_written_generic_kv_for_opencl(
        const std::string & current_attn_backend,
        const std::string & target_attn_backend,
        uint32_t            n_tokens,
        bool                generic_kv_enabled,
        bool                qnn_writeback_ready);

llama_hetero_kv_contract llama_dynamic_phase_shared_qnn_kv_contract(
        const std::string & prefill_attn_backend,
        const std::string & decode_attn_backend,
        bool                qnn_host_buffer_available,
        bool                opencl_can_alias_qnn_host);

bool llama_context_should_use_qnn_shared_phase_kv(
        const std::string & current_attn_backend,
        const std::string & target_attn_backend,
        uint32_t            n_tokens,
        bool                generic_kv_enabled,
        const llama_hetero_kv_contract & allocated_kv_contract);

llama_opencl_external_host_sync_scope llama_context_opencl_sync_scope_for_qnn_shared_phase_kv(
        const std::string & current_attn_backend,
        const std::string & target_attn_backend,
        uint32_t            n_tokens,
        bool                generic_kv_enabled,
        const llama_hetero_kv_contract & allocated_kv_contract);

bool llama_context_should_try_qnn_opencl_direct_host_ptr_visibility(
        const std::string & current_attn_backend,
        const std::string & target_attn_backend,
        uint32_t            n_tokens,
        bool                generic_kv_enabled,
        const llama_hetero_kv_contract & allocated_kv_contract,
        bool                experimental_enabled);

bool llama_context_should_use_dynamic_decode_tg_only_sched_reserve(
        bool     dynamic_route_enabled,
        uint32_t n_tokens,
        bool     experimental_enabled);

bool llama_context_should_prewarm_dynamic_qnn_opencl_kv_aliases(
        const std::string & prefill_attn_backend,
        const std::string & decode_attn_backend,
        bool                generic_kv_enabled,
        const llama_hetero_kv_contract & allocated_kv_contract,
        bool                experimental_enabled);

bool llama_context_should_preload_dynamic_qnn_decode_graphs(
        bool dynamic_route_enabled,
        bool dynamic_route_uses_qnn,
        bool preload_enabled);

std::vector<size_t> llama_context_dynamic_qnn_preload_token_sizes(
        bool   dynamic_route_enabled,
        bool   dynamic_prefill_uses_qnn,
        bool   dynamic_decode_uses_qnn,
        bool   preload_enabled,
        size_t prefill_tokens);

bool llama_context_should_try_cpu_opencl_uma_kv_handoff(
        const std::string & current_attn_backend,
        const std::string & target_attn_backend,
        uint32_t            n_tokens,
        bool                disabled,
        bool                allow_opencl_to_cpu);

const llama_dynamic_route_candidate * llama_context_initial_dynamic_decode_candidate(
        const llama_dynamic_route_runtime_config & config);

llama_hetero_route_spec llama_kv_cache_dynamic_decode_route_for_initial_placement(
        const char * explicit_decode_route,
        const char * decode_schedule);

bool llama_context_should_apply_qnn_workpoint_switch(
        const llama_hetero_route_spec & current_route,
        const llama_hetero_route_spec & target_route,
        uint32_t                        n_tokens,
        const std::string &             current_workpoint,
        const char *                    target_workpoint);

bool llama_context_should_apply_prefill_qnn_workpoint(
        const llama_hetero_route_spec & current_route,
        const llama_hetero_route_spec & target_route,
        uint32_t                        n_tokens,
        const std::string &             current_workpoint,
        const char *                    target_workpoint);

bool llama_context_should_apply_qnn_capacity_switch(
        const llama_hetero_route_spec &      current_route,
        const llama_hetero_route_spec &      target_route,
        uint32_t                             n_tokens,
        uint64_t                             current_required_kv_slots,
        uint64_t                             current_context_size,
        const llama_dynamic_backend_state &  target_state,
        bool                                 generic_kv_enabled);

bool llama_context_is_qnn_request_capacity_boundary(
        uint32_t n_tokens,
        size_t   seq0_prefix_tokens_before_decode,
        bool     qnn_prefix_replay_active);

uint64_t llama_context_qnn_request_required_kv_slots(
        uint32_t input_tokens,
        uint64_t margin);

uint32_t llama_context_qnn_request_input_tokens(
        uint32_t decode_tokens,
        uint32_t declared_request_tokens);

uint64_t llama_context_qnn_request_capacity_margin_from_env(const char * value);

uint64_t llama_context_qnn_request_capacity_margin();

bool llama_context_should_prepare_qnn_request_capacity(
        bool                                      qnn_backend_available,
        bool                                      aot_active_route_requests_qnn,
        const llama_hetero_route_spec &           current_route,
        const llama_hetero_route_spec &           base_route,
        const llama_dynamic_route_runtime_config & dynamic_route_config);

std::vector<std::pair<size_t, size_t>> llama_kv_cache_plan_token_prefix_sync_ranges(
        size_t   tensor_offset,
        size_t   token_bytes,
        uint32_t kv_size,
        uint32_t n_kv_sync,
        uint32_t token_stripes);

int main() {
    testing t;

    t.test("opencl external host sync timing accumulates and clears sub-phases", [](testing & t) {
        llama_opencl_external_host_sync_timing total;
        llama_opencl_external_host_sync_timing part_a;
        llama_opencl_external_host_sync_timing part_b;

        part_a.alias_us = 11;
        part_a.backend_sync_us = 22;
        part_a.transfer_us = 33;
        part_a.synced_buffers = 1;
        part_a.synced_ranges = 4;
        part_a.synced_bytes = 1024;

        part_b.alias_us = 5;
        part_b.backend_sync_us = 7;
        part_b.transfer_us = 9;
        part_b.synced_buffers = 2;
        part_b.synced_ranges = 6;
        part_b.synced_bytes = 2048;

        total.accumulate(part_a);
        total.accumulate(part_b);

        t.assert_equal("alias timing should accumulate across buffers", total.alias_us, (int64_t) 16);
        t.assert_equal("backend sync timing should accumulate across buffers", total.backend_sync_us, (int64_t) 29);
        t.assert_equal("transfer timing should accumulate across buffers", total.transfer_us, (int64_t) 42);
        t.assert_equal("accounted timing should sum all sub-phases", total.accounted_us(), (int64_t) 87);
        t.assert_equal("synced buffer count should accumulate", total.synced_buffers, (size_t) 3);
        t.assert_equal("synced range count should accumulate", total.synced_ranges, (size_t) 10);
        t.assert_equal("synced byte count should accumulate", total.synced_bytes, (size_t) 3072);

        total.clear();

        t.assert_equal("clear should reset alias timing", total.alias_us, (int64_t) 0);
        t.assert_equal("clear should reset backend sync timing", total.backend_sync_us, (int64_t) 0);
        t.assert_equal("clear should reset transfer timing", total.transfer_us, (int64_t) 0);
        t.assert_equal("clear should reset accounted timing", total.accounted_us(), (int64_t) 0);
        t.assert_equal("clear should reset synced buffer count", total.synced_buffers, (size_t) 0);
        t.assert_equal("clear should reset synced range count", total.synced_ranges, (size_t) 0);
        t.assert_equal("clear should reset synced byte count", total.synced_bytes, (size_t) 0);
    });

    t.test("sched reserve timing accumulates and clears sub-phases", [](testing & t) {
        llama_sched_reserve_timing total;
        llama_sched_reserve_timing part_a;
        llama_sched_reserve_timing part_b;

        part_a.sched_new_us = 10;
        part_a.memory_init_us = 20;
        part_a.feature_probe_us = 30;
        part_a.plan_reserve_us = 40;
        part_a.finalize_us = 50;

        part_b.sched_new_us = 1;
        part_b.memory_init_us = 2;
        part_b.feature_probe_us = 3;
        part_b.plan_reserve_us = 4;
        part_b.finalize_us = 5;

        total.accumulate(part_a);
        total.accumulate(part_b);

        t.assert_equal("sched_new timing should accumulate", total.sched_new_us, (int64_t) 11);
        t.assert_equal("memory_init timing should accumulate", total.memory_init_us, (int64_t) 22);
        t.assert_equal("feature_probe timing should accumulate", total.feature_probe_us, (int64_t) 33);
        t.assert_equal("plan_reserve timing should accumulate", total.plan_reserve_us, (int64_t) 44);
        t.assert_equal("finalize timing should accumulate", total.finalize_us, (int64_t) 55);
        t.assert_equal("accounted timing should sum all reserve sub-phases", total.accounted_us(), (int64_t) 165);

        total.clear();

        t.assert_equal("clear should reset sched_new timing", total.sched_new_us, (int64_t) 0);
        t.assert_equal("clear should reset memory_init timing", total.memory_init_us, (int64_t) 0);
        t.assert_equal("clear should reset feature_probe timing", total.feature_probe_us, (int64_t) 0);
        t.assert_equal("clear should reset plan_reserve timing", total.plan_reserve_us, (int64_t) 0);
        t.assert_equal("clear should reset finalize timing", total.finalize_us, (int64_t) 0);
        t.assert_equal("clear should reset accounted timing", total.accounted_us(), (int64_t) 0);
    });

    t.test("opencl KV prefix range planner covers K rows and transposed V stripes", [](testing & t) {
        const auto k_ranges = llama_kv_cache_plan_token_prefix_sync_ranges(
                /* tensor_offset = */ 1024,
                /* token_bytes = */ 64,
                /* kv_size = */ 2048,
                /* n_kv_sync = */ 256,
                /* token_stripes = */ 1);

        t.assert_equal("K prefix should use one contiguous range", k_ranges.size(), (size_t) 1);
        t.assert_equal("K prefix range should start at the tensor offset", k_ranges[0].first, (size_t) 1024);
        t.assert_equal("K prefix range should cover n_kv_sync rows", k_ranges[0].second, (size_t) 256 * 64);

        const auto v_ranges = llama_kv_cache_plan_token_prefix_sync_ranges(
                /* tensor_offset = */ 4096,
                /* token_bytes = */ 2,
                /* kv_size = */ 2048,
                /* n_kv_sync = */ 256,
                /* token_stripes = */ 4);

        t.assert_equal("transposed V prefix should sync one range per embedding stripe", v_ranges.size(), (size_t) 4);
        for (size_t i = 0; i < v_ranges.size(); ++i) {
            t.assert_equal("transposed V stripe offset should advance by one full KV row",
                           v_ranges[i].first,
                           (size_t) 4096 + i * 2048 * 2);
            t.assert_equal("transposed V stripe range should cover n_kv_sync cells",
                           v_ranges[i].second,
                           (size_t) 256 * 2);
        }

        const auto full_v_ranges = llama_kv_cache_plan_token_prefix_sync_ranges(
                /* tensor_offset = */ 4096,
                /* token_bytes = */ 2,
                /* kv_size = */ 2048,
                /* n_kv_sync = */ 4096,
                /* token_stripes = */ 4);

        t.assert_equal("full transposed V ranges should merge into one contiguous tensor range", full_v_ranges.size(), (size_t) 1);
        t.assert_equal("full transposed V merged range should start at the tensor offset", full_v_ranges[0].first, (size_t) 4096);
        t.assert_equal("full transposed V merged range should clamp to kv_size", full_v_ranges[0].second, (size_t) 2048 * 2 * 4);

        const auto empty_ranges = llama_kv_cache_plan_token_prefix_sync_ranges(
                /* tensor_offset = */ 0,
                /* token_bytes = */ 2,
                /* kv_size = */ 2048,
                /* n_kv_sync = */ 0,
                /* token_stripes = */ 4);
        t.assert_equal("empty prefix should produce no sync ranges", empty_ranges.size(), (size_t) 0);
    });

    t.test("qnn same-backend decode can switch HTP workpoints without a route change", [](testing & t) {
        const auto qnn_plan = llama_hetero_build_execution_plan("qnn-npu", nullptr);
        const auto gpu_plan = llama_hetero_build_execution_plan("opencl", nullptr);

        t.assert_true(
                "qnn->qnn single-token decode with a different target workpoint should apply runtime HTP control",
                llama_context_should_apply_qnn_workpoint_switch(
                        qnn_plan.route,
                        qnn_plan.route,
                        1,
                        "burst",
                        "low_balanced"));

        t.assert_true(
                "the same workpoint should not be re-applied every decode token",
                !llama_context_should_apply_qnn_workpoint_switch(
                        qnn_plan.route,
                        qnn_plan.route,
                        1,
                        "low_balanced",
                        "low_balanced"));

        t.assert_true(
                "prefill-sized batches must not trigger decode workpoint-only switching",
                !llama_context_should_apply_qnn_workpoint_switch(
                        qnn_plan.route,
                        qnn_plan.route,
                        512,
                        "burst",
                        "low_balanced"));

        t.assert_true(
                "qnn workpoint-only switching only applies when both phase routes stay on qnn",
                !llama_context_should_apply_qnn_workpoint_switch(
                        qnn_plan.route,
                        gpu_plan.route,
                        1,
                        "burst",
                        "low_balanced"));
    });

    t.test("qnn prefill can apply HTP workpoint before prefill graph execution", [](testing & t) {
        const auto qnn_plan = llama_hetero_build_execution_plan("qnn-npu", nullptr);
        const auto cpu_plan = llama_hetero_build_execution_plan("cpu", nullptr);

        t.assert_true(
                "qnn prefill with a different target workpoint should apply runtime HTP control",
                llama_context_should_apply_prefill_qnn_workpoint(
                        qnn_plan.route,
                        qnn_plan.route,
                        128,
                        "burst",
                        "low_balanced"));

        t.assert_true(
                "prefill route switches to qnn should apply the target workpoint before graph execution",
                llama_context_should_apply_prefill_qnn_workpoint(
                        cpu_plan.route,
                        qnn_plan.route,
                        128,
                        "burst",
                        "low_balanced"));

        t.assert_true(
                "single-token decode should not use the prefill workpoint path",
                !llama_context_should_apply_prefill_qnn_workpoint(
                        qnn_plan.route,
                        qnn_plan.route,
                        1,
                        "burst",
                        "low_balanced"));

        t.assert_true(
                "the same prefill workpoint should not be re-applied",
                !llama_context_should_apply_prefill_qnn_workpoint(
                        qnn_plan.route,
                        qnn_plan.route,
                        128,
                        "low_balanced",
                        "low_balanced"));

        t.assert_true(
                "non-qnn prefill targets should not apply QNN workpoint control",
                !llama_context_should_apply_prefill_qnn_workpoint(
                        qnn_plan.route,
                        cpu_plan.route,
                        128,
                        "burst",
                        "low_balanced"));
    });

    t.test("qnn same-backend decode can switch KV capacity without a route change", [](testing & t) {
        const auto qnn_plan = llama_hetero_build_execution_plan("qnn-npu", nullptr);
        const auto gpu_plan = llama_hetero_build_execution_plan("opencl", nullptr);

        llama_dynamic_backend_state target_4k;
        target_4k.has_qnn_context_size = true;
        target_4k.qnn_context_size = 4096;
        target_4k.has_qnn_required_kv_slots = true;
        target_4k.qnn_required_kv_slots = 2500;

        t.assert_true(
                "qnn->qnn single-token decode with a different target capacity should apply runtime capacity control",
                llama_context_should_apply_qnn_capacity_switch(
                        qnn_plan.route,
                        qnn_plan.route,
                        1,
                        /* current_required_kv_slots = */ 1920,
                        /* current_context_size = */ 2048,
                        target_4k,
                        /* generic_kv_enabled = */ true));

        t.assert_true(
                "the same capacity should not be re-applied every decode token",
                !llama_context_should_apply_qnn_capacity_switch(
                        qnn_plan.route,
                        qnn_plan.route,
                        1,
                        /* current_required_kv_slots = */ 2500,
                        /* current_context_size = */ 4096,
                        target_4k,
                        /* generic_kv_enabled = */ true));

        t.assert_true(
                "prefill-sized batches must not trigger decode capacity-only switching",
                !llama_context_should_apply_qnn_capacity_switch(
                        qnn_plan.route,
                        qnn_plan.route,
                        128,
                        /* current_required_kv_slots = */ 1920,
                        /* current_context_size = */ 2048,
                        target_4k,
                        /* generic_kv_enabled = */ true));

        t.assert_true(
                "qnn capacity-only switching only applies when the target route stays on qnn",
                !llama_context_should_apply_qnn_capacity_switch(
                        qnn_plan.route,
                        gpu_plan.route,
                        1,
                        /* current_required_kv_slots = */ 1920,
                        /* current_context_size = */ 2048,
                        target_4k,
                        /* generic_kv_enabled = */ true));

        t.assert_true(
                "same-qnn private capacity migration must be refused without generic KV writeback",
                !llama_context_should_apply_qnn_capacity_switch(
                        qnn_plan.route,
                        qnn_plan.route,
                        1,
                        /* current_required_kv_slots = */ 1920,
                        /* current_context_size = */ 2048,
                        target_4k,
                        /* generic_kv_enabled = */ false));
    });

    t.test("qnn request capacity boundary starts on empty prefill only", [](testing & t) {
        t.assert_true(
                "prefill on an empty seq0 prefix should start a new request capacity selection",
                llama_context_is_qnn_request_capacity_boundary(
                        /* n_tokens = */ 128,
                        /* seq0_prefix_tokens_before_decode = */ 0,
                        /* qnn_prefix_replay_active = */ false));

        t.assert_true(
                "single-token decode should not start a new request capacity selection",
                !llama_context_is_qnn_request_capacity_boundary(
                        /* n_tokens = */ 1,
                        /* seq0_prefix_tokens_before_decode = */ 0,
                        /* qnn_prefix_replay_active = */ false));

        t.assert_true(
                "prefill with an existing seq0 prefix should not reset request capacity state",
                !llama_context_is_qnn_request_capacity_boundary(
                        /* n_tokens = */ 128,
                        /* seq0_prefix_tokens_before_decode = */ 64,
                        /* qnn_prefix_replay_active = */ false));

        t.assert_true(
                "qnn prefix replay must not recursively start a new request boundary",
                !llama_context_is_qnn_request_capacity_boundary(
                        /* n_tokens = */ 128,
                        /* seq0_prefix_tokens_before_decode = */ 0,
                        /* qnn_prefix_replay_active = */ true));
    });

    t.test("qnn request capacity uses input tokens plus margin", [](testing & t) {
        t.assert_equal(
                "1000 token request with default margin should stay under a 2K artifact guard",
                llama_context_qnn_request_required_kv_slots(
                        /* input_tokens = */ 1000,
                        /* margin = */ 32),
                (uint64_t) 1032);

        t.assert_equal(
                "2500 token request should request enough KV slots to avoid the 2K graph",
                llama_context_qnn_request_required_kv_slots(
                        /* input_tokens = */ 2500,
                        /* margin = */ 32),
                (uint64_t) 2532);
    });

    t.test("qnn request capacity can use declared request length over decode chunk", [](testing & t) {
        t.assert_equal(
                "without a declared request length, uses the current decode token count",
                llama_context_qnn_request_input_tokens(
                        /* decode_tokens = */ 128,
                        /* declared_request_tokens = */ 0),
                (uint32_t) 128);

        t.assert_equal(
                "declared request length carries full prompt size when decode is chunked",
                llama_context_qnn_request_input_tokens(
                        /* decode_tokens = */ 128,
                        /* declared_request_tokens = */ 2501),
                (uint32_t) 2501);

        t.assert_equal(
                "never shrinks below the current decode token count",
                llama_context_qnn_request_input_tokens(
                        /* decode_tokens = */ 256,
                        /* declared_request_tokens = */ 128),
                (uint32_t) 256);
    });

    t.test("qnn request capacity margin defaults to 32 and accepts env override", [](testing & t) {
        t.assert_equal(
                "unset margin env should use the explicit default",
                llama_context_qnn_request_capacity_margin_from_env(nullptr),
                (uint64_t) 32);

        t.assert_equal(
                "empty margin env should use the explicit default",
                llama_context_qnn_request_capacity_margin_from_env(""),
                (uint64_t) 32);

        t.assert_equal(
                "invalid margin env should use the explicit default",
                llama_context_qnn_request_capacity_margin_from_env("not-a-number"),
                (uint64_t) 32);

        t.assert_equal(
                "zero margin env should use the explicit default",
                llama_context_qnn_request_capacity_margin_from_env("0"),
                (uint64_t) 32);

        t.assert_equal(
                "valid margin env should override the default",
                llama_context_qnn_request_capacity_margin_from_env("96"),
                (uint64_t) 96);

        const char * old_margin = std::getenv("GGML_HETERO_DYNAMIC_QNN_REQUEST_KV_MARGIN");
        const std::string old_margin_storage = old_margin != nullptr ? old_margin : "";
        const bool had_old_margin = old_margin != nullptr;

        unsetenv("GGML_HETERO_DYNAMIC_QNN_REQUEST_KV_MARGIN");
        t.assert_equal(
                "runtime margin env should default when unset",
                llama_context_qnn_request_capacity_margin(),
                (uint64_t) 32);

        setenv("GGML_HETERO_DYNAMIC_QNN_REQUEST_KV_MARGIN", "128", 1);
        t.assert_equal(
                "runtime margin env should honor overrides",
                llama_context_qnn_request_capacity_margin(),
                (uint64_t) 128);

        if (had_old_margin) {
            setenv("GGML_HETERO_DYNAMIC_QNN_REQUEST_KV_MARGIN", old_margin_storage.c_str(), 1);
        } else {
            unsetenv("GGML_HETERO_DYNAMIC_QNN_REQUEST_KV_MARGIN");
        }
    });

    t.test("qnn request capacity also prepares for static qnn backend", [](testing & t) {
        const auto empty_plan = llama_hetero_build_execution_plan("", nullptr);
        llama_dynamic_route_runtime_config empty_dynamic;

        t.assert_true(
                "a static qnn-npu backend without a dynamic route should still receive request capacity selection",
                llama_context_should_prepare_qnn_request_capacity(
                        /* qnn_backend_available = */ true,
                        /* aot_active_route_requests_qnn = */ false,
                        empty_plan.route,
                        empty_plan.route,
                        empty_dynamic));

        t.assert_true(
                "no qnn backend and no qnn route should skip request capacity selection",
                !llama_context_should_prepare_qnn_request_capacity(
                        /* qnn_backend_available = */ false,
                        /* aot_active_route_requests_qnn = */ false,
                        empty_plan.route,
                        empty_plan.route,
                        empty_dynamic));

        llama_dynamic_route_runtime_config dynamic_qnn;
        dynamic_qnn.prefill.plan = llama_hetero_build_execution_plan("qnn-npu", nullptr);
        t.assert_true(
                "dynamic qnn routes should still request capacity even before backend availability is checked",
                llama_context_should_prepare_qnn_request_capacity(
                        /* qnn_backend_available = */ false,
                        /* aot_active_route_requests_qnn = */ false,
                        empty_plan.route,
                        empty_plan.route,
                        dynamic_qnn));
    });

    t.test("qnn to opencl phase migration uses qnn shared host contract", [](testing & t) {
        const auto contract = llama_dynamic_phase_migration_kv_contract("qnn-npu", "opencl", "unit-test");

        t.assert_true("qnn->opencl should be a stage boundary contract", contract.stage_boundary_active());
        t.assert_equal("qnn->opencl should request stage-shared layout",
                       (int) contract.layout,
                       (int) llama_hetero_kv_layout_kind::STAGE_SHARED);
        t.assert_equal("qnn->opencl should request QNN RPCMEM transfer",
                       (int) contract.transfer,
                       (int) llama_hetero_kv_transfer_mode::QNN_RPCMEM);
        t.assert_equal("qnn->opencl should keep shared KV on qnn-npu-host storage",
                       contract.storage_backend,
                       std::string("qnn-npu-host"));
        t.assert_true("qnn->opencl should require a shared buffer", contract.shared_buffer_required);
    });

    t.test("qnn to cpu phase migration also uses qnn shared host contract", [](testing & t) {
        const auto contract = llama_dynamic_phase_migration_kv_contract("qnn-npu", "cpu", "unit-test");

        t.assert_true("qnn->cpu should be a stage boundary contract", contract.stage_boundary_active());
        t.assert_equal("qnn->cpu should request stage-shared layout",
                       (int) contract.layout,
                       (int) llama_hetero_kv_layout_kind::STAGE_SHARED);
        t.assert_equal("qnn->cpu should request QNN RPCMEM transfer",
                       (int) contract.transfer,
                       (int) llama_hetero_kv_transfer_mode::QNN_RPCMEM);
        t.assert_equal("qnn->cpu should keep shared KV on qnn-npu-host storage",
                       contract.storage_backend,
                       std::string("qnn-npu-host"));
    });

    t.test("cpu to opencl phase migration stays legacy", [](testing & t) {
        const auto contract = llama_dynamic_phase_migration_kv_contract("cpu", "opencl", "unit-test");

        t.assert_true("cpu->opencl should remain non-shared",
                      !contract.stage_boundary_active() ||
                      contract.transfer == llama_hetero_kv_transfer_mode::NONE);
        t.assert_equal("cpu->opencl should not require shared buffers",
                       contract.shared_buffer_required,
                       false);
    });

    t.test("dynamic qnn prefill and opencl decode can pre-allocate shared qnn kv", [](testing & t) {
        const auto contract = llama_dynamic_phase_shared_qnn_kv_contract(
                "qnn-npu",
                "opencl",
                /* qnn_host_buffer_available = */ true,
                /* opencl_can_alias_qnn_host = */ true);

        t.assert_true("qnn-prefill/opencl-decode should request a stage boundary contract",
                      contract.stage_boundary_active());
        t.assert_equal("qnn-prefill/opencl-decode should use qnn rpcmem transfer",
                       (int) contract.transfer,
                       (int) llama_hetero_kv_transfer_mode::QNN_RPCMEM);
        t.assert_equal("qnn-prefill/opencl-decode should keep KV on qnn-npu-host storage",
                       contract.storage_backend,
                       std::string("qnn-npu-host"));
        t.assert_true("qnn-prefill/opencl-decode should only promote when zero-copy is available",
                      contract.zero_copy);
    });

    t.test("dynamic qnn prefill and opencl decode do not promote shared kv when opencl cannot alias qnn host", [](testing & t) {
        const auto contract = llama_dynamic_phase_shared_qnn_kv_contract(
                "qnn-npu",
                "opencl",
                /* qnn_host_buffer_available = */ true,
                /* opencl_can_alias_qnn_host = */ false);

        t.assert_true("without an OpenCL alias for qnn host buffers the direct shared path must stay disabled",
                      !contract.stage_boundary_active());
    });

    t.test("single-token qnn to opencl decode prefers state migration when generic kv is enabled", [](testing & t) {
        t.assert_true(
                "qnn->opencl decode should prefer state migration over replay when generic KV is available",
                llama_context_should_attempt_qnn_phase_kv_migration("qnn-npu", "opencl", 1, true));
    });

    t.test("single-token qnn to cpu decode also prefers state migration when generic kv is enabled", [](testing & t) {
        t.assert_true(
                "qnn->cpu decode should prefer state migration over replay when generic KV is available",
                llama_context_should_attempt_qnn_phase_kv_migration("qnn-npu", "cpu", 1, true));
    });

    t.test("single-token cpu to qnn decode prefers direct generic kv import when generic kv is enabled", [](testing & t) {
        t.assert_true(
                "cpu->qnn decode should prepare direct generic KV import instead of prefix replay when generic KV is available",
                llama_context_should_attempt_qnn_phase_kv_migration("cpu", "qnn-npu", 1, true));
    });

    t.test("single-token opencl to qnn decode prefers direct generic kv import when generic kv is enabled", [](testing & t) {
        t.assert_true(
                "opencl->qnn decode should prepare direct generic KV import instead of prefix replay when generic KV is available",
                llama_context_should_attempt_qnn_phase_kv_migration("opencl", "qnn-npu", 1, true));
    });

    t.test("opencl to qnn direct generic kv import synchronizes opencl producer first", [](testing & t) {
        t.assert_true(
                "OpenCL-produced generic KV must be synchronized before QNN imports it into private cache",
                llama_context_should_sync_opencl_before_qnn_direct_import("opencl", "qnn-npu", 1, true));
        t.assert_true(
                "CPU-produced generic KV is already host-readable and does not need OpenCL synchronization",
                !llama_context_should_sync_opencl_before_qnn_direct_import("cpu", "qnn-npu", 1, true));
        t.assert_true(
                "OpenCL sync must stay disabled when generic KV import is disabled",
                !llama_context_should_sync_opencl_before_qnn_direct_import("opencl", "qnn-npu", 1, false));
        t.assert_true(
                "prefill-sized batches should not trigger QNN direct import sync",
                !llama_context_should_sync_opencl_before_qnn_direct_import("opencl", "qnn-npu", 8, true));
    });

    t.test("opencl to qnn direct generic kv import syncs only the active prefix", [](testing & t) {
        t.assert_true(
                "OpenCL->QNN direct import should sync only the active generic KV prefix that QNN will import",
                llama_context_opencl_sync_scope_for_qnn_direct_import("opencl", "qnn-npu", 1, true) ==
                    llama_opencl_external_host_sync_scope::ACTIVE_KV_PREFIX);
        t.assert_true(
                "CPU->QNN direct import does not need an OpenCL producer sync scope",
                llama_context_opencl_sync_scope_for_qnn_direct_import("cpu", "qnn-npu", 1, true) ==
                    llama_opencl_external_host_sync_scope::FULL_BUFFER);
        t.assert_true(
                "OpenCL->QNN full-buffer scope remains the compatibility default without generic KV import",
                llama_context_opencl_sync_scope_for_qnn_direct_import("opencl", "qnn-npu", 1, false) ==
                    llama_opencl_external_host_sync_scope::FULL_BUFFER);
        t.assert_true(
                "prefill-sized OpenCL->QNN batches should keep the full-buffer compatibility default",
                llama_context_opencl_sync_scope_for_qnn_direct_import("opencl", "qnn-npu", 8, true) ==
                    llama_opencl_external_host_sync_scope::FULL_BUFFER);
    });

    t.test("qnn state migration is disabled without generic kv materialization", [](testing & t) {
        t.assert_true(
                "qnn->opencl decode must keep replay fallback when generic KV writeback is disabled",
                !llama_context_should_attempt_qnn_phase_kv_migration("qnn-npu", "opencl", 1, false));
        t.assert_true(
                "cpu->qnn decode must keep replay fallback when generic KV import is disabled",
                !llama_context_should_attempt_qnn_phase_kv_migration("cpu", "qnn-npu", 1, false));
    });

    t.test("qnn state migration only applies to decode-sized batches", [](testing & t) {
        t.assert_true(
                "prefill-sized qnn batches should not trigger phase migration directly",
                !llama_context_should_attempt_qnn_phase_kv_migration("qnn-npu", "opencl", 14, true));
    });

    t.test("kv cells physical prefix predicate rejects non-append-only seq0 states", [](testing & t) {
        llama_kv_cells cells;
        cells.resize(8);

        uint32_t n_tokens = 99;
        t.assert_true("an empty cache is a valid zero-token physical prefix",
                      cells.seq_is_physical_prefix(0, &n_tokens));
        t.assert_equal("empty physical prefix should report zero tokens", n_tokens, (uint32_t) 0);

        cells.pos_set(0, 0);
        cells.seq_add(0, 0);
        cells.pos_set(1, 1);
        cells.seq_add(1, 0);

        t.assert_true("seq0 rows [0, 2) with matching positions are a physical prefix",
                      cells.seq_is_physical_prefix(0, &n_tokens));
        t.assert_equal("physical prefix should report its token count", n_tokens, (uint32_t) 2);

        cells.pos_set(3, 3);
        cells.seq_add(3, 0);
        t.assert_true("a hole in the physical prefix must be rejected",
                      !cells.seq_is_physical_prefix(0, &n_tokens));

        cells.reset();
        cells.pos_set(1, 0);
        cells.seq_add(1, 0);
        t.assert_true("position zero stored outside physical row zero must be rejected",
                      !cells.seq_is_physical_prefix(0, &n_tokens));

        cells.reset();
        cells.pos_set(0, 1);
        cells.seq_add(0, 0);
        t.assert_true("a non-zero logical start must be rejected",
                      !cells.seq_is_physical_prefix(0, &n_tokens));

        cells.reset();
        cells.pos_set(0, 0);
        cells.seq_add(0, 0);
        cells.pos_set(1, 1);
        cells.seq_add(1, 0);
        cells.pos_add(1, 1);
        t.assert_true("pending shifted KV state must be rejected",
                      !cells.seq_is_physical_prefix(0, &n_tokens));

        cells.reset();
        cells.pos_set(0, 0);
        cells.seq_add(0, 0);
        cells.seq_add(0, 1);
        t.assert_true("cells shared with another sequence must be rejected",
                      !cells.seq_is_physical_prefix(0, &n_tokens));

        cells.reset();
        cells.pos_set(0, 0);
        cells.seq_add(0, 0);
        cells.pos_set(1, 1);
        cells.seq_add(1, 1);
        t.assert_true("extra used cells from another sequence must be rejected",
                      !cells.seq_is_physical_prefix(0, &n_tokens));
    });

    t.test("qnn host kv buffer is cpu accessible", [](testing & t) {
        t.assert_true(
                "CPU KV buffer should be CPU accessible",
                llama_context_kv_buft_is_cpu_accessible(ggml_backend_cpu_buffer_type()));

        ggml_backend_dev_t qnn_dev = ggml_backend_dev_by_name("qnn-npu");
        t.assert_true("qnn-npu device should be registered for qnn host KV tests", qnn_dev != nullptr);

        ggml_backend_buffer_type_t qnn_host_buft =
            qnn_dev != nullptr ? ggml_backend_dev_host_buffer_type(qnn_dev) : nullptr;
        t.assert_true("qnn-npu host buffer type should be available", qnn_host_buft != nullptr);

        if (qnn_host_buft != nullptr) {
            t.assert_true(
                    "qnn-npu-host KV buffer should be CPU accessible because the CPU backend supports host bufts",
                    llama_context_kv_buft_is_cpu_accessible(qnn_host_buft));
        }
    });

    t.test("qnn to cpu can reuse flushed live generic kv without state rebuild", [](testing & t) {
        t.assert_true(
                "qnn->cpu can skip state rebuild when QNN generic KV writeback is ready and live KV is CPU-accessible",
                llama_context_should_use_qnn_written_generic_kv_for_cpu(
                        "qnn-npu",
                        "cpu",
                        1,
                        /* generic_kv_enabled = */ true,
                        /* qnn_writeback_ready = */ true,
                        /* live_kv_cpu_accessible = */ true,
                        /* qnn_writeback_flushed = */ false));

        t.assert_true(
                "qnn->cpu must not skip state rebuild before QNN generic KV writeback is ready",
                !llama_context_should_use_qnn_written_generic_kv_for_cpu(
                        "qnn-npu",
                        "cpu",
                        1,
                        /* generic_kv_enabled = */ true,
                        /* qnn_writeback_ready = */ false,
                        /* live_kv_cpu_accessible = */ true,
                        /* qnn_writeback_flushed = */ false));

        t.assert_true(
                "qnn->cpu must not skip state rebuild when no live or flushed generic KV is available",
                !llama_context_should_use_qnn_written_generic_kv_for_cpu(
                        "qnn-npu",
                        "cpu",
                        1,
                        /* generic_kv_enabled = */ true,
                        /* qnn_writeback_ready = */ true,
                        /* live_kv_cpu_accessible = */ false,
                        /* qnn_writeback_flushed = */ false));

        t.assert_true(
                "qnn->cpu must not skip state rebuild after QNN generic KV flush when live KV is not CPU-accessible",
                !llama_context_should_use_qnn_written_generic_kv_for_cpu(
                        "qnn-npu",
                        "cpu",
                        1,
                        /* generic_kv_enabled = */ true,
                        /* qnn_writeback_ready = */ true,
                        /* live_kv_cpu_accessible = */ false,
                        /* qnn_writeback_flushed = */ true));

        t.assert_true(
                "qnn->opencl keeps the existing shared-qnn/opencl path",
                !llama_context_should_use_qnn_written_generic_kv_for_cpu(
                        "qnn-npu",
                        "opencl",
                        1,
                        /* generic_kv_enabled = */ true,
                        /* qnn_writeback_ready = */ true,
                        /* live_kv_cpu_accessible = */ true,
                        /* qnn_writeback_flushed = */ true));
    });

    t.test("qnn to opencl can try qnn-written generic kv handoff when writeback is ready", [](testing & t) {
        t.assert_true(
                "qnn->opencl decode can try active-prefix sync from QNN-written generic KV",
                llama_context_should_try_qnn_written_generic_kv_for_opencl(
                        "qnn-npu",
                        "opencl",
                        1,
                        /* generic_kv_enabled = */ true,
                        /* qnn_writeback_ready = */ true));

        t.assert_true(
                "qnn->opencl must not try generic KV handoff before QNN writeback is ready",
                !llama_context_should_try_qnn_written_generic_kv_for_opencl(
                        "qnn-npu",
                        "opencl",
                        1,
                        /* generic_kv_enabled = */ true,
                        /* qnn_writeback_ready = */ false));

        t.assert_true(
                "prefill-sized qnn->opencl batches must keep the existing migration path",
                !llama_context_should_try_qnn_written_generic_kv_for_opencl(
                        "qnn-npu",
                        "opencl",
                        8,
                        /* generic_kv_enabled = */ true,
                        /* qnn_writeback_ready = */ true));

        t.assert_true(
                "qnn->cpu is handled by the CPU-specific live generic KV predicate",
                !llama_context_should_try_qnn_written_generic_kv_for_opencl(
                        "qnn-npu",
                        "cpu",
                        1,
                        /* generic_kv_enabled = */ true,
                        /* qnn_writeback_ready = */ true));
    });

    t.test("non-qnn producers still use their existing migration logic", [](testing & t) {
        t.assert_true(
                "opencl->cpu should not be routed through qnn state migration",
                !llama_context_should_attempt_qnn_phase_kv_migration("opencl", "cpu", 1, true));
    });

    t.test("single-token qnn to opencl decode uses shared kv directly when the context was pre-allocated on qnn rpcmem", [](testing & t) {
        const auto allocated = llama_dynamic_phase_shared_qnn_kv_contract(
                "qnn-npu",
                "opencl",
                /* qnn_host_buffer_available = */ true,
                /* opencl_can_alias_qnn_host = */ true);

        t.assert_true(
                "a qnn-rpcmem allocated contract should bypass state rebuild for qnn->opencl decode switches",
                llama_context_should_use_qnn_shared_phase_kv("qnn-npu", "opencl", 1, true, allocated));
    });

    t.test("qnn to opencl shared kv handoff syncs only the active prefix", [](testing & t) {
        const auto allocated = llama_dynamic_phase_shared_qnn_kv_contract(
                "qnn-npu",
                "opencl",
                /* qnn_host_buffer_available = */ true,
                /* opencl_can_alias_qnn_host = */ true);
        const auto legacy = llama_dynamic_phase_migration_kv_contract("cpu", "opencl", "legacy-unit-test");

        t.assert_true(
                "qnn->opencl direct shared KV handoff should sync only the active prefix",
                llama_context_opencl_sync_scope_for_qnn_shared_phase_kv("qnn-npu", "opencl", 1, true, allocated) ==
                    llama_opencl_external_host_sync_scope::ACTIVE_KV_PREFIX);
        t.assert_true(
                "legacy allocated KV contracts must keep the full-buffer compatibility default",
                llama_context_opencl_sync_scope_for_qnn_shared_phase_kv("qnn-npu", "opencl", 1, true, legacy) ==
                    llama_opencl_external_host_sync_scope::FULL_BUFFER);
        t.assert_true(
                "prefill-sized qnn->opencl batches must keep the full-buffer compatibility default",
                llama_context_opencl_sync_scope_for_qnn_shared_phase_kv("qnn-npu", "opencl", 8, true, allocated) ==
                    llama_opencl_external_host_sync_scope::FULL_BUFFER);
        t.assert_true(
                "qnn->cpu must not use the OpenCL shared-KV sync scope helper",
                llama_context_opencl_sync_scope_for_qnn_shared_phase_kv("qnn-npu", "cpu", 1, true, allocated) ==
                    llama_opencl_external_host_sync_scope::FULL_BUFFER);
    });

    t.test("single-token qnn to opencl decode does not use the shared kv fast path when the allocated contract is legacy", [](testing & t) {
        const auto allocated = llama_dynamic_phase_migration_kv_contract("cpu", "opencl", "legacy-unit-test");

        t.assert_true(
                "legacy allocated contracts should keep using the explicit qnn state migration path",
                !llama_context_should_use_qnn_shared_phase_kv("qnn-npu", "opencl", 1, true, allocated));
    });

    t.test("single-token qnn to opencl decode can try direct host-ptr visibility when the experiment is enabled", [](testing & t) {
        const auto allocated = llama_dynamic_phase_shared_qnn_kv_contract(
                "qnn-npu",
                "opencl",
                /* qnn_host_buffer_available = */ true,
                /* opencl_can_alias_qnn_host = */ true);

        t.assert_true(
                "the direct qnn rpcmem visibility experiment should only activate on the shared qnn->opencl fast path",
                llama_context_should_try_qnn_opencl_direct_host_ptr_visibility(
                        "qnn-npu",
                        "opencl",
                        1,
                        true,
                        allocated,
                        /* experimental_enabled = */ true));
    });

    t.test("direct host-ptr visibility stays disabled when the experiment is off", [](testing & t) {
        const auto allocated = llama_dynamic_phase_shared_qnn_kv_contract(
                "qnn-npu",
                "opencl",
                /* qnn_host_buffer_available = */ true,
                /* opencl_can_alias_qnn_host = */ true);

        t.assert_true(
                "without the env gate the qnn rpcmem direct-visibility path must remain disabled",
                !llama_context_should_try_qnn_opencl_direct_host_ptr_visibility(
                        "qnn-npu",
                        "opencl",
                        1,
                        true,
                        allocated,
                        /* experimental_enabled = */ false));
    });

    t.test("dynamic decode reserve can choose tg-only scope when the experiment is enabled", [](testing & t) {
        t.assert_true(
                "single-token dynamic decode should be allowed to reserve only the token-generation graph when the experiment is enabled",
                llama_context_should_use_dynamic_decode_tg_only_sched_reserve(
                        /* dynamic_route_enabled = */ true,
                        /* n_tokens = */ 1,
                        /* experimental_enabled = */ true));
    });

    t.test("tg-only reserve stays disabled for prefill or when the experiment is off", [](testing & t) {
        t.assert_true(
                "prefill-sized batches must keep the existing full reserve path",
                !llama_context_should_use_dynamic_decode_tg_only_sched_reserve(
                        /* dynamic_route_enabled = */ true,
                        /* n_tokens = */ 128,
                        /* experimental_enabled = */ true));

        t.assert_true(
                "without the env gate decode must keep the existing full reserve path",
                !llama_context_should_use_dynamic_decode_tg_only_sched_reserve(
                        /* dynamic_route_enabled = */ true,
                        /* n_tokens = */ 1,
                        /* experimental_enabled = */ false));

        t.assert_true(
                "static contexts must not silently opt into tg-only reserve",
                !llama_context_should_use_dynamic_decode_tg_only_sched_reserve(
                        /* dynamic_route_enabled = */ false,
                        /* n_tokens = */ 1,
                        /* experimental_enabled = */ true));
    });

    t.test("dynamic qnn prefill and opencl decode do not prewarm direct host-ptr aliases before qnn writes kv", [](testing & t) {
        const auto allocated = llama_dynamic_phase_shared_qnn_kv_contract(
                "qnn-npu",
                "opencl",
                /* qnn_host_buffer_available = */ true,
                /* opencl_can_alias_qnn_host = */ true);

        t.assert_true(
                "qnn->opencl alias prewarm happens before qnn prefill writes KV, so it must stay disabled unless a separate alias-only prewarm path exists",
                !llama_context_should_prewarm_dynamic_qnn_opencl_kv_aliases(
                        "qnn-npu",
                        "opencl",
                        /* generic_kv_enabled = */ true,
                        allocated,
                        /* experimental_enabled = */ true));
    });

    t.test("qnn opencl alias prewarm stays disabled when the dynamic route or experiment preconditions are missing", [](testing & t) {
        const auto allocated = llama_dynamic_phase_shared_qnn_kv_contract(
                "qnn-npu",
                "opencl",
                /* qnn_host_buffer_available = */ true,
                /* opencl_can_alias_qnn_host = */ true);

        t.assert_true(
                "prefill routes that do not start on qnn-npu should not prewarm the qnn rpcmem alias",
                !llama_context_should_prewarm_dynamic_qnn_opencl_kv_aliases(
                        "opencl",
                        "opencl",
                        /* generic_kv_enabled = */ true,
                        allocated,
                        /* experimental_enabled = */ true));

        t.assert_true(
                "decode routes that do not land on opencl should not prewarm the qnn rpcmem alias",
                !llama_context_should_prewarm_dynamic_qnn_opencl_kv_aliases(
                        "qnn-npu",
                        "cpu",
                        /* generic_kv_enabled = */ true,
                        allocated,
                        /* experimental_enabled = */ true));

        t.assert_true(
                "without generic qnn KV materialization the eager alias should remain disabled",
                !llama_context_should_prewarm_dynamic_qnn_opencl_kv_aliases(
                        "qnn-npu",
                        "opencl",
                        /* generic_kv_enabled = */ false,
                        allocated,
                        /* experimental_enabled = */ true));

        t.assert_true(
                "without the env gate the eager alias creation experiment must remain disabled",
                !llama_context_should_prewarm_dynamic_qnn_opencl_kv_aliases(
                        "qnn-npu",
                        "opencl",
                        /* generic_kv_enabled = */ true,
                allocated,
                /* experimental_enabled = */ false));
    });

    t.test("qnn decode graph preload requires explicit env and qnn dynamic route", [](testing & t) {
        t.assert_true(
                "explicit preload and a qnn dynamic route should preload decode graphs",
                llama_context_should_preload_dynamic_qnn_decode_graphs(
                        /* dynamic_route_enabled = */ true,
                        /* dynamic_route_uses_qnn = */ true,
                        /* preload_enabled = */ true));

        t.assert_true(
                "preload must stay disabled without the explicit env gate",
                !llama_context_should_preload_dynamic_qnn_decode_graphs(
                        /* dynamic_route_enabled = */ true,
                        /* dynamic_route_uses_qnn = */ true,
                        /* preload_enabled = */ false));

        t.assert_true(
                "preload must stay disabled for non-qnn dynamic routes",
                !llama_context_should_preload_dynamic_qnn_decode_graphs(
                        /* dynamic_route_enabled = */ true,
                        /* dynamic_route_uses_qnn = */ false,
                        /* preload_enabled = */ true));

        t.assert_true(
                "static contexts must not preload qnn decode graphs",
                !llama_context_should_preload_dynamic_qnn_decode_graphs(
                        /* dynamic_route_enabled = */ false,
                        /* dynamic_route_uses_qnn = */ true,
                        /* preload_enabled = */ true));
    });

    t.test("qnn graph preload token sizes include decode and prefill batches", [](testing & t) {
        const auto both = llama_context_dynamic_qnn_preload_token_sizes(
                /* dynamic_route_enabled = */ true,
                /* dynamic_prefill_uses_qnn = */ true,
                /* dynamic_decode_uses_qnn = */ true,
                /* preload_enabled = */ true,
                /* prefill_tokens = */ 128);
        t.assert_equal("prefill+decode qnn should preload two token sizes", both.size(), (size_t) 2);
        t.assert_equal("decode graph token size", both[0], (size_t) 1);
        t.assert_equal("prefill graph token size", both[1], (size_t) 128);

        const auto prefill_only = llama_context_dynamic_qnn_preload_token_sizes(
                /* dynamic_route_enabled = */ true,
                /* dynamic_prefill_uses_qnn = */ true,
                /* dynamic_decode_uses_qnn = */ false,
                /* preload_enabled = */ true,
                /* prefill_tokens = */ 128);
        t.assert_equal("prefill-only qnn should preload one token size", prefill_only.size(), (size_t) 1);
        t.assert_equal("prefill graph token size", prefill_only[0], (size_t) 128);

        const auto decode_only = llama_context_dynamic_qnn_preload_token_sizes(
                /* dynamic_route_enabled = */ true,
                /* dynamic_prefill_uses_qnn = */ false,
                /* dynamic_decode_uses_qnn = */ true,
                /* preload_enabled = */ true,
                /* prefill_tokens = */ 128);
        t.assert_equal("decode-only qnn should preload one token size", decode_only.size(), (size_t) 1);
        t.assert_equal("decode graph token size", decode_only[0], (size_t) 1);

        const auto disabled = llama_context_dynamic_qnn_preload_token_sizes(
                /* dynamic_route_enabled = */ true,
                /* dynamic_prefill_uses_qnn = */ true,
                /* dynamic_decode_uses_qnn = */ true,
                /* preload_enabled = */ false,
                /* prefill_tokens = */ 128);
        t.assert_equal("disabled env should not preload qnn graphs", disabled.size(), (size_t) 0);
    });

    t.test("single-token cpu to opencl decode can try UMA KV handoff", [](testing & t) {
        t.assert_true(
                "CPU -> OpenCL decode switch can try shared host KV handoff",
                llama_context_should_try_cpu_opencl_uma_kv_handoff(
                        "cpu",
                        "opencl",
                        1,
                        /* disabled = */ false,
                        /* allow_opencl_to_cpu = */ false));

        t.assert_true(
                "OpenCL -> CPU keeps the state migration path unless the reverse experiment is enabled",
                !llama_context_should_try_cpu_opencl_uma_kv_handoff(
                        "opencl",
                        "cpu",
                        1,
                        /* disabled = */ false,
                        /* allow_opencl_to_cpu = */ false));

        t.assert_true(
                "OpenCL -> CPU can try shared host KV handoff when the reverse experiment is enabled",
                llama_context_should_try_cpu_opencl_uma_kv_handoff(
                        "opencl",
                        "cpu",
                        1,
                        /* disabled = */ false,
                        /* allow_opencl_to_cpu = */ true));

        t.assert_true(
                "QNN -> OpenCL must keep the existing NPU-to-GPU KV handoff path",
                !llama_context_should_try_cpu_opencl_uma_kv_handoff(
                        "qnn-npu",
                        "opencl",
                        1,
                        /* disabled = */ false,
                        /* allow_opencl_to_cpu = */ true));

        t.assert_true(
                "prefill-sized batches must not try CPU/OpenCL UMA handoff",
                !llama_context_should_try_cpu_opencl_uma_kv_handoff(
                        "cpu",
                        "opencl",
                        16,
                        /* disabled = */ false,
                        /* allow_opencl_to_cpu = */ true));

        t.assert_true(
                "the disable env gate must force the state rebuild path",
                !llama_context_should_try_cpu_opencl_uma_kv_handoff(
                        "cpu",
                        "opencl",
                        1,
                        /* disabled = */ true,
                        /* allow_opencl_to_cpu = */ true));

        t.assert_true(
                "the disable env gate must also block reverse OpenCL -> CPU UMA handoff",
                !llama_context_should_try_cpu_opencl_uma_kv_handoff(
                        "opencl",
                        "cpu",
                        1,
                        /* disabled = */ true,
                        /* allow_opencl_to_cpu = */ true));
    });

    t.test("kv placement can use the first decode schedule route as initial decode consumer", [](testing & t) {
        const auto from_schedule =
            llama_kv_cache_dynamic_decode_route_for_initial_placement(nullptr, "1:cpu;33:opencl;65:qnn-npu");

        t.assert_equal(
                "schedule starting at token 1 should provide the initial decode consumer",
                std::string("cpu"),
                llama_hetero_phase_backend_for_route(from_schedule));

        const auto explicit_decode =
            llama_kv_cache_dynamic_decode_route_for_initial_placement("opencl", "1:cpu;33:qnn-npu");

        t.assert_equal(
                "explicit decode route should keep priority over schedule-derived placement",
                std::string("opencl"),
                llama_hetero_phase_backend_for_route(explicit_decode));

        const auto delayed_schedule =
            llama_kv_cache_dynamic_decode_route_for_initial_placement(nullptr, "33:opencl;65:qnn-npu");

        t.assert_true(
                "a schedule that does not start at token 1 should not invent an initial decode consumer",
                !delayed_schedule.has_any_route());
    });

    t.test("context contract promotion can use the first decode schedule route", [](testing & t) {
        auto make_candidate = [](const char * label, const char * route) {
            llama_dynamic_route_candidate candidate;
            candidate.label = label != nullptr ? label : "";
            candidate.plan = llama_hetero_build_execution_plan(route, nullptr);
            candidate.configured = route != nullptr && route[0] != '\0';
            return candidate;
        };

        llama_dynamic_route_runtime_config scheduled;
        scheduled.prefill = make_candidate("prefill", "qnn-npu");
        scheduled.decode_schedule.push_back({
                1,
                make_candidate("decode-schedule@1", "opencl"),
                {},
        });

        const llama_dynamic_route_candidate * from_schedule =
            llama_context_initial_dynamic_decode_candidate(scheduled);
        t.assert_true(
                "schedule starting at token 1 should supply the context initial decode candidate",
                from_schedule != nullptr);
        t.assert_equal(
                "opencl-first schedule should be visible to qnn/opencl contract promotion",
                std::string("opencl"),
                llama_hetero_phase_backend_for_route(from_schedule->plan.route));

        const auto shared = llama_dynamic_phase_shared_qnn_kv_contract(
                llama_hetero_phase_backend_for_route(scheduled.prefill.plan.route),
                llama_hetero_phase_backend_for_route(from_schedule->plan.route),
                /* qnn_host_buffer_available = */ true,
                /* opencl_can_alias_qnn_host = */ true);
        t.assert_true(
                "schedule-derived qnn-prefill/opencl-decode should be eligible for shared qnn kv promotion",
                shared.stage_boundary_active());

        scheduled.decode = make_candidate("decode", "cpu");
        const llama_dynamic_route_candidate * explicit_decode =
            llama_context_initial_dynamic_decode_candidate(scheduled);
        t.assert_true(
                "explicit decode route should stay visible",
                explicit_decode != nullptr);
        t.assert_equal(
                "explicit decode route keeps priority over schedule-derived promotion",
                std::string("cpu"),
                llama_hetero_phase_backend_for_route(explicit_decode->plan.route));

        llama_dynamic_route_runtime_config delayed;
        delayed.decode_schedule.push_back({
                33,
                make_candidate("decode-schedule@33", "opencl"),
                {},
        });
        t.assert_true(
                "a delayed schedule must not invent a context initial decode candidate",
                llama_context_initial_dynamic_decode_candidate(delayed) == nullptr);
    });

    return t.summary();
}
