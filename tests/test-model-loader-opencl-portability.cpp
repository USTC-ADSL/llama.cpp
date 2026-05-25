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
        const char * buft_dev_name,
        bool buft_is_dev_host);

bool llama_model_loader_should_route_fg_limited_stage_to_opencl(
        bool hetero_phase_route_active,
        int fg_max_layers,
        int tensor_layer,
        bool is_stage_tensor,
        const llama_hetero_route_spec & route);

bool llama_model_loader_should_prefer_opencl_device_buft_for_stage_residency(
        bool explicit_opencl_stage,
        bool shared_host_portability_required);

bool llama_model_should_use_opencl_only_gguf_weight_devices(
        const llama_hetero_route_spec & route);

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

    t.test("mixed dynamic routes report opencl stage usage for portability", [](testing & t) {
        const auto dynamic_prefill_route = llama_hetero_parse_route_spec("qnn-npu");
        const auto dynamic_decode_route = llama_hetero_parse_route_spec(
                "attn_proj=qnn-npu,attn_core=opencl,attn_out=opencl,ffn=qnn-npu,output=opencl");
        const auto dynamic_fallback_route = llama_hetero_parse_route_spec("cpu");

        t.assert_true(
                "OpenCL attention stages inside a mixed route should request portable weights",
                llama_model_loader_requires_opencl_weight_portability(
                        /* hetero_phase_route_active = */ true,
                        /* hetero_phase_backend_kind = */ 3,
                        dynamic_prefill_route,
                        dynamic_decode_route,
                        dynamic_fallback_route));
    });

    t.test("dynamic opencl portability preserves OpenCL host buft under mmap", [](testing & t) {
        t.assert_true(
                "dynamic opencl routes should keep OpenCL_Host instead of downgrading to CPU_Mapped under mmap",
                llama_model_loader_should_preserve_opencl_host_buft_for_mmap(
                        /* hetero_phase_route_active = */ false,
                        /* hetero_portable_cpu_weights_for_opencl_dynamic_stage = */ true,
                        /* buft_dev_name = */ "GPUOpenCL",
                        /* buft_is_dev_host = */ true));
    });

    t.test("non-opencl portability does not preserve host buft under mmap", [](testing & t) {
        t.assert_true(
                "routes without dynamic opencl portability should still allow mmap host-buft downgrade",
                !llama_model_loader_should_preserve_opencl_host_buft_for_mmap(
                        /* hetero_phase_route_active = */ false,
                        /* hetero_portable_cpu_weights_for_opencl_dynamic_stage = */ false,
                        /* buft_dev_name = */ "GPUOpenCL",
                        /* buft_is_dev_host = */ true));
    });

    t.test("fg layer limit routes later stage weights to opencl", [](testing & t) {
        const auto route = llama_hetero_parse_route_spec(
                "attn_proj=qnn-npu,attn_core=opencl,attn_out=cpu,ffn=qnn-npu,output=cpu");

        t.assert_true(
                "layers outside FG limit execute on OpenCL and should not keep QNN-host stage weights",
                llama_model_loader_should_route_fg_limited_stage_to_opencl(
                        /* hetero_phase_route_active = */ true,
                        /* fg_max_layers = */ 2,
                        /* tensor_layer = */ 2,
                        /* is_stage_tensor = */ true,
                        route));
        t.assert_true(
                "layers inside FG limit should keep their requested QNN/CPU stage placement",
                !llama_model_loader_should_route_fg_limited_stage_to_opencl(
                        /* hetero_phase_route_active = */ true,
                        /* fg_max_layers = */ 2,
                        /* tensor_layer = */ 1,
                        /* is_stage_tensor = */ true,
                        route));
        t.assert_true(
                "non-stage tensors should not be remapped by the FG layer limit",
                !llama_model_loader_should_route_fg_limited_stage_to_opencl(
                        /* hetero_phase_route_active = */ true,
                        /* fg_max_layers = */ 2,
                        /* tensor_layer = */ 2,
                        /* is_stage_tensor = */ false,
                        route));
    });

    t.test("explicit OpenCL stage residency prefers device-local weights", [](testing & t) {
        t.assert_true(
                "OpenCL stage weights should try GPUOpenCL device buffers before OpenCL_Host",
                llama_model_loader_should_prefer_opencl_device_buft_for_stage_residency(
                        /* explicit_opencl_stage = */ true,
                        /* shared_host_portability_required = */ false));
        t.assert_true(
                "shared-host portability keeps OpenCL_Host first for CPU/OpenCL dynamic switching",
                !llama_model_loader_should_prefer_opencl_device_buft_for_stage_residency(
                        /* explicit_opencl_stage = */ true,
                        /* shared_host_portability_required = */ true));
    });

    t.test("all-opencl hetero route excludes qnn from GGUF weight device split", [](testing & t) {
        const auto all_opencl = llama_hetero_parse_route_spec(
                "attn_proj=opencl,attn_core=opencl,attn_out=opencl,ffn=opencl,output=opencl");
        const auto homogeneous_opencl = llama_hetero_parse_route_spec("opencl");
        const auto mixed_qnn_opencl = llama_hetero_parse_route_spec(
                "attn_proj=qnn-npu,attn_core=opencl,attn_out=cpu,ffn=qnn-npu,output=cpu");

        t.assert_true(
                "all explicit OpenCL stages should load GGUF weights only through OpenCL devices",
                llama_model_should_use_opencl_only_gguf_weight_devices(all_opencl));
        t.assert_true(
                "homogeneous OpenCL shorthand should load GGUF weights only through OpenCL devices",
                llama_model_should_use_opencl_only_gguf_weight_devices(homogeneous_opencl));
        t.assert_true(
                "mixed QNN/OpenCL routes still need their requested GGUF placement policy",
                !llama_model_should_use_opencl_only_gguf_weight_devices(mixed_qnn_opencl));
        t.assert_true(
                "unset routes should preserve default device split behavior",
                !llama_model_should_use_opencl_only_gguf_weight_devices({}));
    });

    return t.summary();
}
