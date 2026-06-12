#include "../src/llama-hetero-route.h"
#include "testing.h"

int main() {
    testing t;

    t.test("opencl phase route leaves output tail to the scheduler", [](testing & t) {
        const auto route = llama_hetero_parse_route_spec("opencl");

        t.assert_true(
                "OpenCL transformer stages should not force the final logits tail onto a hetero backend",
                llama_hetero_phase_output_tail_backend_for_route(route).empty());
        t.assert_equal(
                "phase backend remains OpenCL",
                std::string("opencl"),
                llama_hetero_phase_backend_for_route(route));
    });

    t.test("qnn phase route keeps output tail on qnn", [](testing & t) {
        t.assert_equal(
                "QNN AoT routes still own their output tail",
                std::string("qnn-npu"),
                llama_hetero_phase_output_tail_backend_for_route(
                        llama_hetero_parse_route_spec("qnn-npu")));
    });

    t.test("cpu phase route keeps output tail on cpu", [](testing & t) {
        t.assert_equal(
                "CPU routes should keep the output tail on CPU",
                std::string("cpu"),
                llama_hetero_phase_output_tail_backend_for_route(
                        llama_hetero_parse_route_spec("cpu")));
    });

    return t.summary();
}
