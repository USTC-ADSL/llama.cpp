#include "../src/llama-hetero-route.h"
#include "ggml.h"
#include "testing.h"

#include <unordered_set>

bool llama_model_loader_should_enable_opencl_cpu_extra_cpu_copy(
        const llama_hetero_route_spec & dynamic_prefill_route,
        const llama_hetero_route_spec & dynamic_decode_route,
        bool enable_extra_cpu_copy);

const ggml_tensor * llama_model_resolve_weight_for_cpu_copy(
        const ggml_tensor * original,
        const ggml_tensor * cpu_copy,
        llama_hetero_route_stage stage,
        const llama_hetero_route_spec & route);

bool llama_model_should_publish_tensor_by_name(
        const ggml_tensor * tensor,
        const std::unordered_set<const ggml_tensor *> & private_tensors);

int main() {
    testing t;

    t.test("opencl prefill to cpu decode can request an extra cpu-friendly copy", [](testing & t) {
        t.assert_true(
                "env-enabled OpenCL -> CPU switching should request an extra CPU-friendly copy",
                llama_model_loader_should_enable_opencl_cpu_extra_cpu_copy(
                        llama_hetero_parse_route_spec("opencl"),
                        llama_hetero_parse_route_spec("cpu"),
                        true));
    });

    t.test("cpu prefill to opencl decode can request an extra cpu-friendly copy", [](testing & t) {
        t.assert_true(
                "env-enabled CPU -> OpenCL switching should still keep a CPU-friendly copy for later CPU phases",
                llama_model_loader_should_enable_opencl_cpu_extra_cpu_copy(
                        llama_hetero_parse_route_spec("cpu"),
                        llama_hetero_parse_route_spec("opencl"),
                        true));
    });

    t.test("feature stays disabled without the env gate", [](testing & t) {
        t.assert_true(
                "without the env gate the extra CPU-friendly copy must remain disabled",
                !llama_model_loader_should_enable_opencl_cpu_extra_cpu_copy(
                        llama_hetero_parse_route_spec("opencl"),
                        llama_hetero_parse_route_spec("cpu"),
                        false));
    });

    t.test("non cpu opencl routes do not request an extra cpu-friendly copy", [](testing & t) {
        t.assert_true(
                "QNN/CPU switching should not be treated as an OpenCL/CPU extra-copy request",
                !llama_model_loader_should_enable_opencl_cpu_extra_cpu_copy(
                        llama_hetero_parse_route_spec("qnn-npu"),
                        llama_hetero_parse_route_spec("cpu"),
                        true));
    });

    t.test("cpu stage resolves to the cpu-friendly copy", [](testing & t) {
        ggml_tensor original = {};
        ggml_tensor cpu_copy = {};

        t.assert_true(
                "CPU FFN stages should consume the CPU-friendly duplicate when it exists",
                llama_model_resolve_weight_for_cpu_copy(
                        &original,
                        &cpu_copy,
                        llama_hetero_route_stage::FFN,
                        llama_hetero_parse_route_spec("cpu")) == &cpu_copy);
    });

    t.test("opencl stage keeps the original weight copy", [](testing & t) {
        ggml_tensor original = {};
        ggml_tensor cpu_copy = {};

        t.assert_true(
                "OpenCL FFN stages should keep the original OpenCL-friendly weight copy",
                llama_model_resolve_weight_for_cpu_copy(
                        &original,
                        &cpu_copy,
                        llama_hetero_route_stage::FFN,
                        llama_hetero_parse_route_spec("opencl")) == &original);
    });

    t.test("qnn stage keeps the original weight copy", [](testing & t) {
        ggml_tensor original = {};
        ggml_tensor cpu_copy = {};

        t.assert_true(
                "QNN FFN stages should keep the original QNN/host-resident weight copy",
                llama_model_resolve_weight_for_cpu_copy(
                        &original,
                        &cpu_copy,
                        llama_hetero_route_stage::FFN,
                        llama_hetero_parse_route_spec("qnn-npu")) == &original);
    });

    t.test("manually-constructed per-stage routes only switch the stages that actually run on cpu", [](testing & t) {
        ggml_tensor original = {};
        ggml_tensor cpu_copy = {};
        llama_hetero_route_spec route = {};
        route.attn = "opencl";
        route.ffn = "opencl";
        route.output = "cpu";

        t.assert_true(
                "output stage should use the CPU-friendly copy when only output is routed to CPU",
                llama_model_resolve_weight_for_cpu_copy(
                        &original,
                        &cpu_copy,
                        llama_hetero_route_stage::OUTPUT,
                        route) == &cpu_copy);
        t.assert_true(
                "FFN stage should keep the original weight when FFN remains on OpenCL",
                llama_model_resolve_weight_for_cpu_copy(
                        &original,
                        &cpu_copy,
                        llama_hetero_route_stage::FFN,
                        route) == &original);
    });

    t.test("missing cpu-friendly copies fall back to the original weight", [](testing & t) {
        ggml_tensor original = {};

        t.assert_true(
                "if no CPU-friendly duplicate exists the original weight must still be used",
                llama_model_resolve_weight_for_cpu_copy(
                        &original,
                        nullptr,
                        llama_hetero_route_stage::FFN,
                        llama_hetero_parse_route_spec("cpu")) == &original);
    });

    t.test("extra cpu-friendly copies stay private to route resolution", [](testing & t) {
        ggml_tensor original = {};
        ggml_tensor cpu_copy = {};
        ggml_set_name(&original, "blk.0.ffn_up.weight");
        ggml_set_name(&cpu_copy, "blk.0.ffn_up.weight");

        const std::unordered_set<const ggml_tensor *> private_tensors = { &cpu_copy };

        t.assert_true(
                "the original OpenCL-friendly weight remains published for canonical name lookup",
                llama_model_should_publish_tensor_by_name(&original, private_tensors));
        t.assert_true(
                "the same-name CPU-friendly duplicate must not become the canonical public tensor",
                !llama_model_should_publish_tensor_by_name(&cpu_copy, private_tensors));
    });

    return t.summary();
}
