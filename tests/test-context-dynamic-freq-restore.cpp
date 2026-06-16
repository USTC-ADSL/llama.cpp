#include "../src/llama-hetero-route.h"
#include "testing.h"

bool llama_context_route_uses_opencl_backend(const llama_hetero_route_spec & route);
bool llama_context_route_uses_cpu_backend(const llama_hetero_route_spec & route);

bool llama_context_should_restore_gpu_freq(
        const llama_hetero_route_spec & target_route,
        uint32_t                        n_tokens,
        bool                            gpu_freq_pinned);

bool llama_context_should_restore_cpu_freq(
        const llama_hetero_route_spec & target_route,
        uint32_t                        n_tokens,
        bool                            cpu_freq_pinned);

namespace {

llama_hetero_route_spec route_for(const char * route) {
    return llama_hetero_build_execution_plan(route, nullptr).route;
}

llama_hetero_route_spec split_route(
        const char * attn,
        const char * ffn,
        const char * output) {
    llama_hetero_route_spec route;
    route.attn = llama_hetero_canonical_backend(attn);
    route.ffn = llama_hetero_canonical_backend(ffn);
    route.output = llama_hetero_canonical_backend(output);
    return route;
}

} // namespace

int main() {
    testing t;

    t.test("route backend usage treats an empty route as CPU default", [](testing & t) {
        llama_hetero_route_spec default_route;
        const llama_hetero_route_spec cpu_route = route_for("cpu");
        const llama_hetero_route_spec opencl_route = route_for("opencl");
        const llama_hetero_route_spec qnn_route = route_for("qnn-npu");

        t.assert_true("empty route should use CPU default", llama_context_route_uses_cpu_backend(default_route));
        t.assert_true("CPU route should use CPU", llama_context_route_uses_cpu_backend(cpu_route));
        t.assert_true("OpenCL route should not use CPU", !llama_context_route_uses_cpu_backend(opencl_route));
        t.assert_true("QNN route should not use CPU", !llama_context_route_uses_cpu_backend(qnn_route));

        t.assert_true("empty route should not use OpenCL", !llama_context_route_uses_opencl_backend(default_route));
        t.assert_true("OpenCL route should use OpenCL", llama_context_route_uses_opencl_backend(opencl_route));
        t.assert_true("CPU route should not use OpenCL", !llama_context_route_uses_opencl_backend(cpu_route));
        t.assert_true("QNN route should not use OpenCL", !llama_context_route_uses_opencl_backend(qnn_route));
    });

    t.test("split routes keep a backend pinned while any stage still uses it", [](testing & t) {
        const llama_hetero_route_spec cpu_tail = split_route("qnn-npu", "qnn-npu", "cpu");
        const llama_hetero_route_spec opencl_tail = split_route("qnn-npu", "qnn-npu", "opencl");

        t.assert_true("CPU tail should count as CPU usage", llama_context_route_uses_cpu_backend(cpu_tail));
        t.assert_true("OpenCL tail should count as OpenCL usage", llama_context_route_uses_opencl_backend(opencl_tail));
        t.assert_true(
                "CPU frequency should not restore while a target stage still uses CPU",
                !llama_context_should_restore_cpu_freq(cpu_tail, 1, true));
        t.assert_true(
                "GPU frequency should not restore while a target stage still uses OpenCL",
                !llama_context_should_restore_gpu_freq(opencl_tail, 1, true));
    });

    t.test("leaving OpenCL after a pinned GPU frequency requests restore", [](testing & t) {
        const llama_hetero_route_spec opencl_route = route_for("opencl");
        const llama_hetero_route_spec cpu_route = route_for("cpu");
        const llama_hetero_route_spec qnn_route = route_for("qnn-npu");

        t.assert_true(
                "switching to CPU should restore a pinned GPU frequency",
                llama_context_should_restore_gpu_freq(cpu_route, 1, true));
        t.assert_true(
                "switching to QNN should restore a pinned GPU frequency",
                llama_context_should_restore_gpu_freq(qnn_route, 1, true));
        t.assert_true(
                "staying on OpenCL should not restore GPU frequency",
                !llama_context_should_restore_gpu_freq(opencl_route, 1, true));
        t.assert_true(
                "prefill-sized batches should not restore GPU frequency",
                !llama_context_should_restore_gpu_freq(cpu_route, 512, true));
        t.assert_true(
                "un-pinned GPU frequency should not restore",
                !llama_context_should_restore_gpu_freq(cpu_route, 1, false));
    });

    t.test("leaving CPU after a pinned CPU frequency requests restore", [](testing & t) {
        llama_hetero_route_spec default_route;
        const llama_hetero_route_spec opencl_route = route_for("opencl");
        const llama_hetero_route_spec cpu_route = route_for("cpu");
        const llama_hetero_route_spec qnn_route = route_for("qnn-npu");

        t.assert_true(
                "switching to OpenCL should restore a pinned CPU frequency",
                llama_context_should_restore_cpu_freq(opencl_route, 1, true));
        t.assert_true(
                "switching to QNN should restore a pinned CPU frequency",
                llama_context_should_restore_cpu_freq(qnn_route, 1, true));
        t.assert_true(
                "staying on CPU should not restore CPU frequency",
                !llama_context_should_restore_cpu_freq(cpu_route, 1, true));
        t.assert_true(
                "default route should keep CPU constraints active",
                !llama_context_should_restore_cpu_freq(default_route, 1, true));
        t.assert_true(
                "prefill-sized batches should not restore CPU frequency",
                !llama_context_should_restore_cpu_freq(opencl_route, 512, true));
        t.assert_true(
                "un-pinned CPU frequency should not restore",
                !llama_context_should_restore_cpu_freq(opencl_route, 1, false));
    });

    return t.summary();
}
