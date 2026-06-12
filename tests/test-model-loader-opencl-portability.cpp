#include "../src/llama-hetero-route.h"
#include "testing.h"

bool llama_model_loader_requires_opencl_weight_portability(
        bool hetero_phase_route_active,
        int hetero_phase_backend_kind,
        const llama_hetero_route_spec & dynamic_prefill_route,
        const llama_hetero_route_spec & dynamic_decode_route,
        const llama_hetero_route_spec & dynamic_fallback_route);

bool llama_model_loader_should_preserve_opencl_host_buft_for_mmap(
        bool hetero_phase_route_active,
        bool hetero_portable_cpu_weights_for_opencl_dynamic_stage,
        bool enable_opencl_host_weights,
        const char * buft_dev_name,
        bool buft_is_dev_host);

bool llama_model_loader_requires_cpu_weight_residency(
        const llama_hetero_route_spec & dynamic_prefill_route,
        const llama_hetero_route_spec & dynamic_decode_route,
        const llama_hetero_route_spec & dynamic_fallback_route);

bool llama_model_loader_decode_schedule_requires_cpu_weight_residency(
        const char * decode_schedule);

int main() {
    testing t;

    t.test("dynamic opencl decode requires portable weights when model route is non-opencl", [](testing & t) {
        const auto dynamic_prefill_route = llama_hetero_parse_route_spec("qnn-npu");
        const auto dynamic_decode_route = llama_hetero_parse_route_spec("opencl");
        const auto dynamic_fallback_route = llama_hetero_parse_route_spec("cpu");

        t.assert_true(
                "non-opencl model route should still prepare portable weights for opencl decode",
                llama_model_loader_requires_opencl_weight_portability(
                        /* hetero_phase_route_active = */ true,
                        /* hetero_phase_backend_kind = */ 3,
                        dynamic_prefill_route,
                        dynamic_decode_route,
                        dynamic_fallback_route));
    });

    t.test("dynamic opencl prefill still requires portable weights when model route is non-opencl", [](testing & t) {
        const auto dynamic_prefill_route = llama_hetero_parse_route_spec("opencl");
        const auto dynamic_decode_route = llama_hetero_parse_route_spec("cpu");
        const auto dynamic_fallback_route = llama_hetero_parse_route_spec("cpu");

        t.assert_true(
                "existing opencl prefill portability behavior should remain enabled",
                llama_model_loader_requires_opencl_weight_portability(
                        /* hetero_phase_route_active = */ true,
                        /* hetero_phase_backend_kind = */ 3,
                        dynamic_prefill_route,
                        dynamic_decode_route,
                        dynamic_fallback_route));
    });

    t.test("dynamic opencl decode still requires portable weights when model phase route is unset", [](testing & t) {
        const auto dynamic_prefill_route = llama_hetero_parse_route_spec("qnn-npu");
        const auto dynamic_decode_route = llama_hetero_parse_route_spec("opencl");
        const auto dynamic_fallback_route = llama_hetero_parse_route_spec("cpu");

        t.assert_true(
                "default model-load routing should still prepare portable weights for opencl decode",
                llama_model_loader_requires_opencl_weight_portability(
                        /* hetero_phase_route_active = */ false,
                        /* hetero_phase_backend_kind = */ 0,
                        dynamic_prefill_route,
                        dynamic_decode_route,
                        dynamic_fallback_route));
    });

    t.test("opencl model route does not need extra portability", [](testing & t) {
        const auto dynamic_prefill_route = llama_hetero_parse_route_spec("qnn-npu");
        const auto dynamic_decode_route = llama_hetero_parse_route_spec("opencl");
        const auto dynamic_fallback_route = llama_hetero_parse_route_spec("cpu");

        t.assert_true(
                "when model route is already opencl there should be no extra portability override",
                !llama_model_loader_requires_opencl_weight_portability(
                        /* hetero_phase_route_active = */ true,
                        /* hetero_phase_backend_kind = */ 2,
                        dynamic_prefill_route,
                        dynamic_decode_route,
                        dynamic_fallback_route));
    });

    t.test("routes without opencl do not request portable weights", [](testing & t) {
        const auto dynamic_prefill_route = llama_hetero_parse_route_spec("qnn-npu");
        const auto dynamic_decode_route = llama_hetero_parse_route_spec("cpu");
        const auto dynamic_fallback_route = llama_hetero_parse_route_spec("cpu");

        t.assert_true(
                "no dynamic opencl stage should mean no portability override",
                !llama_model_loader_requires_opencl_weight_portability(
                        /* hetero_phase_route_active = */ true,
                        /* hetero_phase_backend_kind = */ 3,
                        dynamic_prefill_route,
                        dynamic_decode_route,
                        dynamic_fallback_route));
    });

    t.test("dynamic qnn to cpu decode requires cpu weight residency", [](testing & t) {
        const auto dynamic_prefill_route = llama_hetero_parse_route_spec("qnn-npu");
        const auto dynamic_decode_route = llama_hetero_parse_route_spec("cpu");
        const auto dynamic_fallback_route = llama_hetero_parse_route_spec("qnn-npu");

        t.assert_true(
                "QNN -> CPU switching should prepare CPU_REPACK-friendly CPU weight residency",
                llama_model_loader_requires_cpu_weight_residency(
                        dynamic_prefill_route,
                        dynamic_decode_route,
                        dynamic_fallback_route));
    });

    t.test("dynamic opencl to cpu decode requires cpu weight residency", [](testing & t) {
        const auto dynamic_prefill_route = llama_hetero_parse_route_spec("opencl");
        const auto dynamic_decode_route = llama_hetero_parse_route_spec("cpu");
        const auto dynamic_fallback_route = llama_hetero_parse_route_spec("opencl");

        t.assert_true(
                "OpenCL -> CPU switching should prepare CPU_REPACK-friendly CPU weight residency",
                llama_model_loader_requires_cpu_weight_residency(
                        dynamic_prefill_route,
                        dynamic_decode_route,
                        dynamic_fallback_route));
    });

    t.test("dynamic cpu fallback requires cpu weight residency", [](testing & t) {
        const auto dynamic_prefill_route = llama_hetero_parse_route_spec("qnn-npu");
        const auto dynamic_decode_route = llama_hetero_parse_route_spec("opencl");
        const auto dynamic_fallback_route = llama_hetero_parse_route_spec("cpu");

        t.assert_true(
                "CPU fallback should still have CPU_REPACK-friendly weight residency ready",
                llama_model_loader_requires_cpu_weight_residency(
                        dynamic_prefill_route,
                        dynamic_decode_route,
                        dynamic_fallback_route));
    });

    t.test("routes without cpu do not request cpu weight residency", [](testing & t) {
        const auto dynamic_prefill_route = llama_hetero_parse_route_spec("qnn-npu");
        const auto dynamic_decode_route = llama_hetero_parse_route_spec("opencl");
        const auto dynamic_fallback_route = llama_hetero_parse_route_spec("qnn-npu");

        t.assert_true(
                "routes that never target CPU should not allocate CPU_REPACK duplicates",
                !llama_model_loader_requires_cpu_weight_residency(
                        dynamic_prefill_route,
                        dynamic_decode_route,
                        dynamic_fallback_route));
    });

    t.test("decode schedule with cpu entries requires cpu weight residency", [](testing & t) {
        t.assert_true(
                "scheduled CPU decode slices should prepare CPU_REPACK-friendly weight residency",
                llama_model_loader_decode_schedule_requires_cpu_weight_residency("1:qnn-npu;65:cpu"));
    });

    t.test("decode schedule without cpu entries does not request cpu weight residency", [](testing & t) {
        t.assert_true(
                "non-CPU decode schedules should not allocate CPU_REPACK duplicates",
                !llama_model_loader_decode_schedule_requires_cpu_weight_residency("1:qnn-npu;65:opencl"));
    });

    t.test("dynamic opencl portability does not preserve OpenCL host buft by default", [](testing & t) {
        t.assert_true(
                "dynamic opencl routes should allow OpenCL_Host to downgrade unless the experimental host-weight path is enabled",
                !llama_model_loader_should_preserve_opencl_host_buft_for_mmap(
                        /* hetero_phase_route_active = */ false,
                        /* hetero_portable_cpu_weights_for_opencl_dynamic_stage = */ true,
                        /* enable_opencl_host_weights = */ false,
                        /* buft_dev_name = */ "GPUOpenCL",
                        /* buft_is_dev_host = */ true));
    });

    t.test("experimental host-weight flag can preserve OpenCL host buft", [](testing & t) {
        t.assert_true(
                "the explicit experimental flag should preserve OpenCL_Host for manual CPU/OpenCL host-weight experiments",
                llama_model_loader_should_preserve_opencl_host_buft_for_mmap(
                        /* hetero_phase_route_active = */ false,
                        /* hetero_portable_cpu_weights_for_opencl_dynamic_stage = */ true,
                        /* enable_opencl_host_weights = */ true,
                        /* buft_dev_name = */ "GPUOpenCL",
                        /* buft_is_dev_host = */ true));
    });

    t.test("non-opencl portability does not preserve host buft under mmap", [](testing & t) {
        t.assert_true(
                "routes without dynamic opencl portability should still allow mmap host-buft downgrade",
                !llama_model_loader_should_preserve_opencl_host_buft_for_mmap(
                        /* hetero_phase_route_active = */ false,
                        /* hetero_portable_cpu_weights_for_opencl_dynamic_stage = */ false,
                        /* enable_opencl_host_weights = */ false,
                        /* buft_dev_name = */ "GPUOpenCL",
                        /* buft_is_dev_host = */ true));
    });

    return t.summary();
}
