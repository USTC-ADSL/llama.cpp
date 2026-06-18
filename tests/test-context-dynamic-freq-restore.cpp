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

bool llama_context_should_continue_dynamic_route_after_state_apply(
        bool state_only_transition,
        bool route_switch_pending,
        bool state_applied);

bool llama_context_should_restore_freq_constraints_on_destroy(
        bool gpu_freq_pinned,
        bool cpu_freq_pinned,
        bool cpu_policy_freq_pinned);

uint64_t llama_context_tracked_freq_after_state_apply(
        uint64_t current_freq,
        uint64_t target_freq,
        bool     state_attempted,
        bool     state_applied);

bool llama_context_freq_restore_readback_matches(
        uint64_t target_min,
        uint64_t target_max,
        uint64_t readback_min,
        uint64_t readback_max);

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

    t.test("pre-route state failure does not block an already selected route switch", [](testing & t) {
        t.assert_true(
                "route switch should continue when a best-effort state apply fails",
                llama_context_should_continue_dynamic_route_after_state_apply(
                        /* state_only_transition = */ false,
                        /* route_switch_pending  = */ true,
                        /* state_applied         = */ false));
        t.assert_true(
                "state-only transitions should stop after handling their state",
                !llama_context_should_continue_dynamic_route_after_state_apply(
                        /* state_only_transition = */ true,
                        /* route_switch_pending  = */ false,
                        /* state_applied         = */ true));
        t.assert_true(
                "state-only failures should stop after the failed state attempt",
                !llama_context_should_continue_dynamic_route_after_state_apply(
                        /* state_only_transition = */ true,
                        /* route_switch_pending  = */ false,
                        /* state_applied         = */ false));
        t.assert_true(
                "successful pre-route state apply should continue to the route switch",
                llama_context_should_continue_dynamic_route_after_state_apply(
                        /* state_only_transition = */ false,
                        /* route_switch_pending  = */ true,
                        /* state_applied         = */ true));
    });

    t.test("context destruction restores any saved frequency constraints", [](testing & t) {
        t.assert_true(
                "destroy should restore saved GPU constraints",
                llama_context_should_restore_freq_constraints_on_destroy(
                        /* gpu_freq_pinned        = */ true,
                        /* cpu_freq_pinned        = */ false,
                        /* cpu_policy_freq_pinned = */ false));
        t.assert_true(
                "destroy should restore saved generic CPU constraints",
                llama_context_should_restore_freq_constraints_on_destroy(
                        /* gpu_freq_pinned        = */ false,
                        /* cpu_freq_pinned        = */ true,
                        /* cpu_policy_freq_pinned = */ false));
        t.assert_true(
                "destroy should restore saved CPU policy constraints",
                llama_context_should_restore_freq_constraints_on_destroy(
                        /* gpu_freq_pinned        = */ false,
                        /* cpu_freq_pinned        = */ false,
                        /* cpu_policy_freq_pinned = */ true));
        t.assert_true(
                "destroy should skip restore when no constraints were saved",
                !llama_context_should_restore_freq_constraints_on_destroy(
                        /* gpu_freq_pinned        = */ false,
                        /* cpu_freq_pinned        = */ false,
                        /* cpu_policy_freq_pinned = */ false));
    });

    t.test("failed frequency attempts still mark the requested target as attempted", [](testing & t) {
        t.assert_true(
                "failed attempted frequency switch should suppress same-target retries",
                llama_context_tracked_freq_after_state_apply(
                        /* current_freq    = */ 660000000,
                        /* target_freq     = */ 967000000,
                        /* state_attempted = */ true,
                        /* state_applied   = */ false) == 967000000);
        t.assert_true(
                "successful frequency switch should track the target",
                llama_context_tracked_freq_after_state_apply(
                        /* current_freq    = */ 660000000,
                        /* target_freq     = */ 967000000,
                        /* state_attempted = */ true,
                        /* state_applied   = */ true) == 967000000);
        t.assert_true(
                "unattempted frequency switch should keep the previous tracked value",
                llama_context_tracked_freq_after_state_apply(
                        /* current_freq    = */ 660000000,
                        /* target_freq     = */ 967000000,
                        /* state_attempted = */ false,
                        /* state_applied   = */ false) == 660000000);
    });

    t.test("frequency restore accepts platform-capped max when min is released", [](testing & t) {
        t.assert_true(
                "exact readback should match restored constraints",
                llama_context_freq_restore_readback_matches(
                        /* target_min   = */ 556800,
                        /* target_max   = */ 3532800,
                        /* readback_min = */ 556800,
                        /* readback_max = */ 3532800));
        t.assert_true(
                "platform-capped max should still count as restored",
                llama_context_freq_restore_readback_matches(
                        /* target_min   = */ 556800,
                        /* target_max   = */ 3532800,
                        /* readback_min = */ 556800,
                        /* readback_max = */ 2400000));
        t.assert_true(
                "unreleased min should not count as restored",
                !llama_context_freq_restore_readback_matches(
                        /* target_min   = */ 556800,
                        /* target_max   = */ 3532800,
                        /* readback_min = */ 2400000,
                        /* readback_max = */ 3532800));
        t.assert_true(
                "max above the saved default should not count as restored",
                !llama_context_freq_restore_readback_matches(
                        /* target_min   = */ 556800,
                        /* target_max   = */ 3532800,
                        /* readback_min = */ 556800,
                        /* readback_max = */ 4320000));
    });

    return t.summary();
}
