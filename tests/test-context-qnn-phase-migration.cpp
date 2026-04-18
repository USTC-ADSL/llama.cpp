#include "../src/llama-hetero-route.h"
#include "testing.h"

#include <string>

llama_hetero_kv_contract llama_dynamic_phase_migration_kv_contract(
        const std::string & producer_backend,
        const std::string & consumer_backend,
        const char * reason);

bool llama_context_should_attempt_qnn_phase_kv_migration(
        const std::string & current_attn_backend,
        const std::string & target_attn_backend,
        uint32_t            n_tokens,
        bool                generic_kv_enabled);

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

int main() {
    testing t;

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

    t.test("qnn state migration is disabled without generic kv materialization", [](testing & t) {
        t.assert_true(
                "qnn->opencl decode must keep replay fallback when generic KV writeback is disabled",
                !llama_context_should_attempt_qnn_phase_kv_migration("qnn-npu", "opencl", 1, false));
    });

    t.test("qnn state migration only applies to decode-sized batches", [](testing & t) {
        t.assert_true(
                "prefill-sized qnn batches should not trigger phase migration directly",
                !llama_context_should_attempt_qnn_phase_kv_migration("qnn-npu", "opencl", 14, true));
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

    t.test("single-token qnn to opencl decode does not use the shared kv fast path when the allocated contract is legacy", [](testing & t) {
        const auto allocated = llama_dynamic_phase_migration_kv_contract("cpu", "opencl", "legacy-unit-test");

        t.assert_true(
                "legacy allocated contracts should keep using the explicit qnn state migration path",
                !llama_context_should_use_qnn_shared_phase_kv("qnn-npu", "opencl", 1, true, allocated));
    });

    return t.summary();
}
