#include "../src/llama-dyn-route.h"
#include "testing.h"

#include <cstdlib>
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

llama_dynamic_route_request prefill_request(
        uint32_t n_tokens,
        const llama_hetero_execution_plan & current_plan,
        const llama_hetero_execution_plan & base_plan) {
    llama_dynamic_route_request request;
    request.n_tokens = n_tokens;
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

    t.test("env config can request GPU freq sync before apply", [](testing & t) {
        setenv("GGML_HETERO_DYNAMIC_MODE", "phase", 1);
        setenv("GGML_HETERO_DECODE_GPU_FREQ_SYNC_BEFORE_APPLY", "1", 1);

        const llama_dynamic_route_runtime_config config = llama_dynamic_route_config_from_env();

        t.assert_true("GPU freq pre-apply sync enabled", config.decode_gpu_freq_sync_before_apply);

        unsetenv("GGML_HETERO_DECODE_GPU_FREQ_SYNC_BEFORE_APPLY");
        unsetenv("GGML_HETERO_DYNAMIC_MODE");
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

    t.test("env config can parse decode route schedule", [](testing & t) {
        setenv("GGML_HETERO_DYNAMIC_MODE", "phase", 1);
        setenv("GGML_HETERO_DYNAMIC_DECODE_SCHEDULE", "1:cpu;33:qnn-npu;65:cpu", 1);

        const llama_dynamic_route_runtime_config config = llama_dynamic_route_config_from_env();

        t.assert_equal("schedule entries", size_t(3), config.decode_schedule.size());
        t.assert_equal("first start", uint64_t(1), config.decode_schedule[0].start_token);
        t.assert_equal("second route", std::string("qnn-npu"), llama_hetero_phase_backend_for_route(config.decode_schedule[1].route.plan.route));
        t.assert_equal("third start", uint64_t(65), config.decode_schedule[2].start_token);

        unsetenv("GGML_HETERO_DYNAMIC_DECODE_SCHEDULE");
        unsetenv("GGML_HETERO_DYNAMIC_MODE");
    });

    t.test("env config can parse prefill qnn workpoint", [](testing & t) {
        setenv("GGML_HETERO_DYNAMIC_MODE", "phase", 1);
        setenv("GGML_HETERO_DYNAMIC_PREFILL_ROUTE", "qnn-npu", 1);
        setenv("GGML_HETERO_DYNAMIC_PREFILL_QNN_WORKPOINT", "low_balanced", 1);

        const llama_dynamic_route_runtime_config config = llama_dynamic_route_config_from_env();

        t.assert_true("prefill QNN workpoint configured", config.prefill_backend_state.has_qnn_workpoint);
        t.assert_equal("prefill QNN workpoint", std::string("low_balanced"), config.prefill_backend_state.qnn_workpoint);

        const llama_hetero_execution_plan cpu_plan = llama_hetero_build_execution_plan("cpu", nullptr);
        const llama_dynamic_route_decision prefill = llama_dynamic_route_decide(
                config,
                prefill_request(128, cpu_plan, cpu_plan));

        t.assert_true("prefill should switch to QNN", prefill.should_apply);
        t.assert_true("prefill decision carries QNN workpoint", prefill.backend_state.has_qnn_workpoint);
        t.assert_equal("prefill decision QNN workpoint", std::string("low_balanced"), prefill.backend_state.qnn_workpoint);

        unsetenv("GGML_HETERO_DYNAMIC_PREFILL_QNN_WORKPOINT");
        unsetenv("GGML_HETERO_DYNAMIC_PREFILL_ROUTE");
        unsetenv("GGML_HETERO_DYNAMIC_MODE");
    });

    t.test("decode route schedule parser is reusable outside env config", [](testing & t) {
        const auto schedule = llama_dynamic_route_parse_decode_schedule("1:cpu;33:opencl;65:qnn-npu");

        t.assert_equal("schedule entries", size_t(3), schedule.size());
        t.assert_equal("first start", uint64_t(1), schedule[0].start_token);
        t.assert_equal("first route", std::string("cpu"), llama_hetero_phase_backend_for_route(schedule[0].route.plan.route));
        t.assert_equal("second route", std::string("opencl"), llama_hetero_phase_backend_for_route(schedule[1].route.plan.route));
        t.assert_equal("third route", std::string("qnn-npu"), llama_hetero_phase_backend_for_route(schedule[2].route.plan.route));
    });

    t.test("decode route schedule parser supports per-entry backend state", [](testing & t) {
        const auto schedule = llama_dynamic_route_parse_decode_schedule(
                "1:cpu{threads=6,affinity=FC,cpu_freq_khz=4320000};"
                "33:opencl{gpu_freq_hz=660000000};"
                "65:qnn-npu{qnn_workpoint=burst}");

        t.assert_equal("schedule entries", size_t(3), schedule.size());
        t.assert_equal("CPU route", std::string("cpu"), llama_hetero_phase_backend_for_route(schedule[0].route.plan.route));
        t.assert_true("CPU threads configured", schedule[0].backend_state.has_cpu_threads);
        t.assert_equal("CPU threads", int32_t(6), schedule[0].backend_state.cpu_threads);
        t.assert_true("CPU affinity configured", schedule[0].backend_state.has_cpu_affinity_mask);
        t.assert_equal("CPU affinity", std::string("FC"), schedule[0].backend_state.cpu_affinity_mask);
        t.assert_true("CPU frequency configured", schedule[0].backend_state.has_cpu_freq_khz);
        t.assert_equal("CPU frequency", uint64_t(4320000), schedule[0].backend_state.cpu_freq_khz);

        t.assert_equal("OpenCL route", std::string("opencl"), llama_hetero_phase_backend_for_route(schedule[1].route.plan.route));
        t.assert_true("GPU frequency configured", schedule[1].backend_state.has_gpu_freq_hz);
        t.assert_equal("GPU frequency", uint64_t(660000000), schedule[1].backend_state.gpu_freq_hz);

        t.assert_equal("QNN route", std::string("qnn-npu"), llama_hetero_phase_backend_for_route(schedule[2].route.plan.route));
        t.assert_true("QNN workpoint configured", schedule[2].backend_state.has_qnn_workpoint);
        t.assert_equal("QNN workpoint", std::string("burst"), schedule[2].backend_state.qnn_workpoint);
    });

    t.test("decode route schedule parser supports per-policy CPU frequencies", [](testing & t) {
        const auto schedule = llama_dynamic_route_parse_decode_schedule(
                "1:cpu{threads=6,affinity=FC,cpu_policy0_freq_khz=3532800,cpu_policy6_freq_khz=4320000}");

        t.assert_equal("schedule entries", size_t(1), schedule.size());
        t.assert_equal("CPU route", std::string("cpu"), llama_hetero_phase_backend_for_route(schedule[0].route.plan.route));
        t.assert_equal("CPU policy frequency count", size_t(2), schedule[0].backend_state.cpu_policy_freqs.size());
        t.assert_equal("CPU policy0 id", uint32_t(0), schedule[0].backend_state.cpu_policy_freqs[0].policy);
        t.assert_equal("CPU policy0 frequency", uint64_t(3532800), schedule[0].backend_state.cpu_policy_freqs[0].freq_khz);
        t.assert_equal("CPU policy6 id", uint32_t(6), schedule[0].backend_state.cpu_policy_freqs[1].policy);
        t.assert_equal("CPU policy6 frequency", uint64_t(4320000), schedule[0].backend_state.cpu_policy_freqs[1].freq_khz);
    });

    t.test("decode route schedule state is carried by selected decision", [](testing & t) {
        llama_dynamic_route_runtime_config config;
        config.mode = llama_dynamic_route_mode::PHASE_HEURISTIC;
        config.decode_schedule = llama_dynamic_route_parse_decode_schedule(
                "1:cpu{threads=6,affinity=FC,cpu_freq_khz=4320000};"
                "33:opencl{gpu_freq_hz=660000000};"
                "65:qnn-npu{workpoint=burst}");

        const llama_hetero_execution_plan cpu_plan = llama_hetero_build_execution_plan("cpu", nullptr);
        const llama_hetero_execution_plan opencl_plan = llama_hetero_build_execution_plan("opencl", nullptr);

        llama_dynamic_route_decision token_33 = llama_dynamic_route_decide(
                config,
                decode_request(33, cpu_plan, cpu_plan));
        t.assert_true("token 33 switches to OpenCL", token_33.should_apply);
        t.assert_true("token 33 carries GPU frequency", token_33.backend_state.has_gpu_freq_hz);
        t.assert_equal("token 33 GPU frequency", uint64_t(660000000), token_33.backend_state.gpu_freq_hz);

        llama_dynamic_route_decision token_64 = llama_dynamic_route_decide(
                config,
                decode_request(64, opencl_plan, cpu_plan));
        t.assert_true("token 64 is a route noop", !token_64.should_apply);
        t.assert_equal("token 64 reason", std::string("already-active"), token_64.reason);
        t.assert_true("token 64 still carries selected GPU frequency", token_64.backend_state.has_gpu_freq_hz);
        t.assert_equal("token 64 GPU frequency", uint64_t(660000000), token_64.backend_state.gpu_freq_hz);

        llama_dynamic_route_decision token_65 = llama_dynamic_route_decide(
                config,
                decode_request(65, opencl_plan, cpu_plan));
        t.assert_true("token 65 switches to QNN", token_65.should_apply);
        t.assert_true("token 65 carries QNN workpoint", token_65.backend_state.has_qnn_workpoint);
        t.assert_equal("token 65 QNN workpoint", std::string("burst"), token_65.backend_state.qnn_workpoint);
    });

    t.test("decode route schedule decision carries per-policy CPU frequencies", [](testing & t) {
        llama_dynamic_route_runtime_config config;
        config.mode = llama_dynamic_route_mode::PHASE_HEURISTIC;
        config.decode_schedule = llama_dynamic_route_parse_decode_schedule(
                "1:opencl;"
                "33:cpu{threads=6,affinity=FC,cpu_policy0_freq_khz=3532800,cpu_policy6_freq_khz=4320000}");

        const llama_hetero_execution_plan opencl_plan = llama_hetero_build_execution_plan("opencl", nullptr);
        const llama_hetero_execution_plan base_plan = llama_hetero_build_execution_plan("opencl", nullptr);

        llama_dynamic_route_decision token_33 = llama_dynamic_route_decide(
                config,
                decode_request(33, opencl_plan, base_plan));
        t.assert_true("token 33 switches to CPU", token_33.should_apply);
        t.assert_equal("token 33 CPU policy frequency count", size_t(2), token_33.backend_state.cpu_policy_freqs.size());
        t.assert_equal("token 33 CPU policy0 id", uint32_t(0), token_33.backend_state.cpu_policy_freqs[0].policy);
        t.assert_equal("token 33 CPU policy0 frequency", uint64_t(3532800), token_33.backend_state.cpu_policy_freqs[0].freq_khz);
        t.assert_equal("token 33 CPU policy6 id", uint32_t(6), token_33.backend_state.cpu_policy_freqs[1].policy);
        t.assert_equal("token 33 CPU policy6 frequency", uint64_t(4320000), token_33.backend_state.cpu_policy_freqs[1].freq_khz);
    });

    t.test("decode route schedule can switch cpu to qnn and back to cpu", [](testing & t) {
        setenv("GGML_HETERO_DYNAMIC_MODE", "phase", 1);
        setenv("GGML_HETERO_DYNAMIC_DECODE_SCHEDULE", "1:cpu;33:qnn-npu;65:cpu", 1);

        const llama_dynamic_route_runtime_config config = llama_dynamic_route_config_from_env();
        const llama_hetero_execution_plan cpu_plan = llama_hetero_build_execution_plan("cpu", nullptr);
        const llama_hetero_execution_plan qnn_plan = llama_hetero_build_execution_plan("qnn-npu", nullptr);

        llama_dynamic_route_decision token_32 = llama_dynamic_route_decide(
                config,
                decode_request(32, cpu_plan, cpu_plan));
        t.assert_true("token 32 stays on CPU", !token_32.should_apply);
        t.assert_equal("token 32 route", std::string("cpu"), route_string(token_32));

        llama_dynamic_route_decision token_33 = llama_dynamic_route_decide(
                config,
                decode_request(33, cpu_plan, cpu_plan));
        t.assert_true("token 33 switches to QNN", token_33.should_apply);
        t.assert_equal("token 33 route", std::string("qnn-npu"), route_string(token_33));
        t.assert_equal("token 33 switch boundary", uint64_t(32), token_33.decode_schedule_switch_after);

        llama_dynamic_route_decision token_64 = llama_dynamic_route_decide(
                config,
                decode_request(64, qnn_plan, cpu_plan));
        t.assert_true("token 64 stays on QNN", !token_64.should_apply);
        t.assert_equal("token 64 route", std::string("qnn-npu"), route_string(token_64));

        llama_dynamic_route_decision token_65 = llama_dynamic_route_decide(
                config,
                decode_request(65, qnn_plan, cpu_plan));
        t.assert_true("token 65 switches back to CPU", token_65.should_apply);
        t.assert_equal("token 65 route", std::string("cpu"), route_string(token_65));
        t.assert_equal("token 65 switch boundary", uint64_t(64), token_65.decode_schedule_switch_after);

        unsetenv("GGML_HETERO_DYNAMIC_DECODE_SCHEDULE");
        unsetenv("GGML_HETERO_DYNAMIC_MODE");
    });

    t.test("decode route schedule supports nonuniform token intervals", [](testing & t) {
        setenv("GGML_HETERO_DYNAMIC_MODE", "phase", 1);
        setenv("GGML_HETERO_DYNAMIC_DECODE_SCHEDULE", "1:cpu;5:opencl;14:qnn-npu;31:cpu", 1);

        const llama_dynamic_route_runtime_config config = llama_dynamic_route_config_from_env();
        const llama_hetero_execution_plan cpu_plan = llama_hetero_build_execution_plan("cpu", nullptr);
        const llama_hetero_execution_plan opencl_plan = llama_hetero_build_execution_plan("opencl", nullptr);
        const llama_hetero_execution_plan qnn_plan = llama_hetero_build_execution_plan("qnn-npu", nullptr);

        llama_dynamic_route_decision token_4 = llama_dynamic_route_decide(
                config,
                decode_request(4, cpu_plan, cpu_plan));
        t.assert_true("token 4 stays on CPU", !token_4.should_apply);
        t.assert_true("token 4 uses schedule", token_4.decode_schedule_active);
        t.assert_equal("token 4 schedule start", uint64_t(1), token_4.decode_schedule_start_token);
        t.assert_equal("token 4 route", std::string("cpu"), route_string(token_4));

        llama_dynamic_route_decision token_5 = llama_dynamic_route_decide(
                config,
                decode_request(5, cpu_plan, cpu_plan));
        t.assert_true("token 5 switches to OpenCL", token_5.should_apply);
        t.assert_equal("token 5 route", std::string("opencl"), route_string(token_5));
        t.assert_equal("token 5 switch boundary", uint64_t(4), token_5.decode_schedule_switch_after);

        llama_dynamic_route_decision token_13 = llama_dynamic_route_decide(
                config,
                decode_request(13, opencl_plan, cpu_plan));
        t.assert_true("token 13 stays on OpenCL", !token_13.should_apply);
        t.assert_equal("token 13 schedule start", uint64_t(5), token_13.decode_schedule_start_token);
        t.assert_equal("token 13 route", std::string("opencl"), route_string(token_13));

        llama_dynamic_route_decision token_14 = llama_dynamic_route_decide(
                config,
                decode_request(14, opencl_plan, cpu_plan));
        t.assert_true("token 14 switches to QNN", token_14.should_apply);
        t.assert_equal("token 14 route", std::string("qnn-npu"), route_string(token_14));
        t.assert_equal("token 14 switch boundary", uint64_t(13), token_14.decode_schedule_switch_after);

        llama_dynamic_route_decision token_30 = llama_dynamic_route_decide(
                config,
                decode_request(30, qnn_plan, cpu_plan));
        t.assert_true("token 30 stays on QNN", !token_30.should_apply);
        t.assert_equal("token 30 schedule start", uint64_t(14), token_30.decode_schedule_start_token);
        t.assert_equal("token 30 route", std::string("qnn-npu"), route_string(token_30));

        llama_dynamic_route_decision token_31 = llama_dynamic_route_decide(
                config,
                decode_request(31, qnn_plan, cpu_plan));
        t.assert_true("token 31 switches back to CPU", token_31.should_apply);
        t.assert_equal("token 31 route", std::string("cpu"), route_string(token_31));
        t.assert_equal("token 31 switch boundary", uint64_t(30), token_31.decode_schedule_switch_after);

        unsetenv("GGML_HETERO_DYNAMIC_DECODE_SCHEDULE");
        unsetenv("GGML_HETERO_DYNAMIC_MODE");
    });

    return t.summary();
}
