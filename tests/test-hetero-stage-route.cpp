#include "../src/llama-hetero-route.h"
#include "testing.h"

#include <string>

int main() {
    testing t;

    t.test("AF simulation route keeps attention on OpenCL and FFN on QNN", [](testing & t) {
        const auto route = llama_hetero_parse_route_spec(
                "attn_proj=opencl,attn_core=opencl,attn_out=opencl,ffn=qnn-npu,output=opencl");

        t.assert_equal("attention projection should run on OpenCL",
                       std::string("opencl"),
                       route.backend_for(llama_hetero_route_stage::ATTN_PROJ));
        t.assert_equal("attention core should run on OpenCL",
                       std::string("opencl"),
                       route.backend_for(llama_hetero_route_stage::ATTN_CORE));
        t.assert_equal("attention output should stay on OpenCL",
                       std::string("opencl"),
                       route.backend_for(llama_hetero_route_stage::ATTN_OUT));
        t.assert_equal("FFN should run on QNN",
                       std::string("qnn-npu"),
                       route.backend_for(llama_hetero_route_stage::FFN));
        t.assert_equal("output tail should stay on OpenCL",
                       std::string("opencl"),
                       route.backend_for(llama_hetero_route_stage::OUTPUT));
    });

    t.test("AF simulation route has no QNN projection to OpenCL attention KV boundary", [](testing & t) {
        const auto plan = llama_hetero_build_execution_plan(
                "attn_proj=opencl,attn_core=opencl,attn_out=opencl,ffn=qnn-npu,output=opencl",
                nullptr);

        t.assert_true("all attention stages on OpenCL should not activate the QNN/OpenCL KV boundary",
                      !plan.attn_kv.stage_boundary_active());
    });

    t.test("mixed stage route is preserved for qnn-opencl-qnn simulation", [](testing & t) {
        const auto route = llama_hetero_parse_route_spec(
                "attn_proj=qnn-npu,attn_core=opencl,attn_out=opencl,ffn=qnn-npu,output=opencl");

        t.assert_equal("attention projection should stay on QNN",
                       std::string("qnn-npu"),
                       route.backend_for(llama_hetero_route_stage::ATTN_PROJ));
        t.assert_equal("attention core should stay on OpenCL",
                       std::string("opencl"),
                       route.backend_for(llama_hetero_route_stage::ATTN_CORE));
        t.assert_equal("attention output should stay on OpenCL",
                       std::string("opencl"),
                       route.backend_for(llama_hetero_route_stage::ATTN_OUT));
        t.assert_equal("FFN should stay on QNN",
                       std::string("qnn-npu"),
                       route.backend_for(llama_hetero_route_stage::FFN));
        t.assert_equal("output tail should stay on OpenCL",
                       std::string("opencl"),
                       route.backend_for(llama_hetero_route_stage::OUTPUT));
    });

    t.test("single backend shorthand remains phase-wide", [](testing & t) {
        const auto route = llama_hetero_parse_route_spec("qnn-npu");

        t.assert_equal("attention projection should inherit shorthand backend",
                       std::string("qnn-npu"),
                       route.backend_for(llama_hetero_route_stage::ATTN_PROJ));
        t.assert_equal("attention core should inherit shorthand backend",
                       std::string("qnn-npu"),
                       route.backend_for(llama_hetero_route_stage::ATTN_CORE));
        t.assert_equal("attention output should inherit shorthand backend",
                       std::string("qnn-npu"),
                       route.backend_for(llama_hetero_route_stage::ATTN_OUT));
        t.assert_equal("FFN should inherit shorthand backend",
                       std::string("qnn-npu"),
                       route.backend_for(llama_hetero_route_stage::FFN));
        t.assert_equal("output should inherit shorthand backend",
                       std::string("qnn-npu"),
                       route.backend_for(llama_hetero_route_stage::OUTPUT));
    });

    t.test("stage route reports qnn and opencl usage", [](testing & t) {
        const auto route = llama_hetero_parse_route_spec(
                "attn_proj=qnn-npu,attn_core=opencl,attn_out=opencl,ffn=qnn-npu,output=opencl");

        t.assert_true("mixed route should report OpenCL usage",
                      llama_hetero_route_uses_backend(route, llama_hetero_is_opencl_backend));
        t.assert_true("mixed route should report QNN usage",
                      llama_hetero_route_uses_backend(route, llama_hetero_is_qnn_backend));
        t.assert_true("mixed route should not report CPU usage",
                      !llama_hetero_route_uses_backend(route, llama_hetero_is_cpu_backend));
    });

    t.test("fine grained layer limit keeps only first requested layers active", [](testing & t) {
        t.assert_true("zero means no layer limit",
                      llama_hetero_fg_layer_allowed(100, 0));
        t.assert_true("negative layer ids are non-layer tensors and should remain allowed",
                      llama_hetero_fg_layer_allowed(-1, 2));
        t.assert_true("layer below limit should be routed",
                      llama_hetero_fg_layer_allowed(1, 2));
        t.assert_true("layer equal to limit should stop fine-grained routing",
                      !llama_hetero_fg_layer_allowed(2, 2));
        t.assert_true("layer above limit should stop fine-grained routing",
                      !llama_hetero_fg_layer_allowed(3, 2));
    });

    t.test("attention projection to opencl core requests qnn rpcmem kv contract", [](testing & t) {
        const auto plan = llama_hetero_build_execution_plan(
                "attn_proj=qnn-npu,attn_core=opencl,attn_out=opencl,ffn=qnn-npu,output=opencl",
                nullptr);

        t.assert_true("QNN projection to OpenCL attention should be a stage boundary",
                      plan.attn_kv.stage_boundary_active());
        t.assert_equal("QNN/OpenCL attention KV boundary should use QNN RPCMEM transfer",
                       (int) llama_hetero_kv_transfer_mode::QNN_RPCMEM,
                       (int) plan.attn_kv.transfer);
        t.assert_equal("QNN/OpenCL attention KV boundary should use qnn-npu-host storage",
                       std::string("qnn-npu-host"),
                       plan.attn_kv.storage_backend);
    });

    return t.summary();
}
