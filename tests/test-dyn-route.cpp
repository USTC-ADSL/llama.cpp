#include "../src/llama-dyn-route.h"
#include "testing.h"

#include <string>

namespace {

llama_dynamic_route_request decode_request(
        uint64_t decode_token_index,
        const llama_hetero_execution_plan & current_plan,
        const llama_hetero_execution_plan & base_plan) {
    llama_dynamic_route_request request;
    request.n_tokens = 1;
    request.decode_token_index = decode_token_index;
    request.opencl_backend_available = true;
    request.qnn_backend_available = true;
    request.current_plan = &current_plan;
    request.base_plan = &base_plan;
    return request;
}

std::string route_string(const llama_dynamic_route_decision & decision) {
    return llama_hetero_phase_backend_for_route(decision.plan.route);
}

} // namespace

int main() {
    testing t;

    t.test("public dynamic route config carries decode switch-after boundary", [](testing & t) {
        llama_dynamic_route_config public_config = llama_dynamic_route_default_config();
        public_config.mode = "phase";
        public_config.decode_switch_after = 64;
        public_config.decode_gpu_freq_hz = 967000000;
        public_config.gpu_min_freq_path = "/sys/test_gpu/min_freq";
        public_config.gpu_max_freq_path = "/sys/test_gpu/max_freq";
        public_config.gpu_cur_freq_path = "/sys/test_gpu/cur_freq";
        public_config.decode_cpu_affinity_mask = "CF";
        public_config.decode_cpu_threads = 6;

        llama_dynamic_route_runtime_config config;
        std::string error;
        const bool ok = llama_dynamic_route_build_runtime_config(public_config, config, &error);

        t.assert_true("runtime config should build", ok);
        t.assert_equal("switch boundary", uint64_t(64), config.decode_switch_after);
        t.assert_equal("GPU freq target", uint64_t(967000000), config.decode_gpu_freq_hz);
        t.assert_equal("GPU min path", std::string("/sys/test_gpu/min_freq"), config.gpu_min_freq_path);
        t.assert_equal("GPU max path", std::string("/sys/test_gpu/max_freq"), config.gpu_max_freq_path);
        t.assert_equal("GPU cur path", std::string("/sys/test_gpu/cur_freq"), config.gpu_cur_freq_path);
        t.assert_equal("CPU affinity target", std::string("CF"), config.decode_cpu_affinity_mask);
        t.assert_equal("CPU thread target", int32_t(6), config.decode_cpu_threads);
    });

    t.test("decode switch-after keeps base route until completed-token boundary", [](testing & t) {
        llama_dynamic_route_runtime_config config;
        config.mode = llama_dynamic_route_mode::PHASE_HEURISTIC;
        config.decode_switch_after = 64;
        config.decode.label = "decode";
        config.decode.plan = llama_hetero_build_execution_plan("opencl", nullptr);
        config.decode.configured = true;

        const llama_hetero_execution_plan base_plan =
            llama_hetero_build_execution_plan("qnn-npu", nullptr);

        llama_dynamic_route_decision before = llama_dynamic_route_decide(
                config,
                decode_request(64, base_plan, base_plan));
        t.assert_true(
                "64 completed decode tokens should not apply target route until the next decode call",
                !before.should_apply);
        t.assert_equal("pre-switch request stays on base route", std::string("qnn-npu"), route_string(before));
        t.assert_equal("pre-switch reason", std::string("decode-switch-wait"), before.reason);

        llama_dynamic_route_decision after = llama_dynamic_route_decide(
                config,
                decode_request(65, base_plan, base_plan));
        t.assert_true(
                "65th decode call should switch after 64 completed tokens",
                after.should_apply);
        t.assert_equal("target route", std::string("opencl"), route_string(after));
        t.assert_equal("switch reason", std::string("decode-switch-after"), after.reason);
    });

    return t.summary();
}
