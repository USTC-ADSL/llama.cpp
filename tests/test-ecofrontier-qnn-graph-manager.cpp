#include "../src/llama-ecofrontier-qnn-graph-manager.h"
#include "testing.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {

ecofrontier::qnn_graph_descriptor make_graph(
        const std::string & id,
        ecofrontier::qnn_graph_phase phase,
        uint64_t usable_kv_slots,
        double energy_mj,
        uint64_t exec_us,
        uint64_t load_us = 1000,
        uint64_t warmup_us = 100,
        uint64_t memory_bytes = 4096,
        uint64_t safety_margin = 8,
        std::vector<std::string> workpoints = { "burst" }) {
    ecofrontier::qnn_graph_descriptor graph;
    graph.graph_id = id;
    graph.path = "/tmp/" + id + ".bin";
    graph.phase = phase;
    graph.chunk_size = phase == ecofrontier::qnn_graph_phase::PREFILL ? 128 : 1;
    graph.usable_kv_slots = usable_kv_slots;
    graph.safety_margin = safety_margin;
    graph.supported_workpoints = std::move(workpoints);
    graph.profiled_load_us = load_us;
    graph.profiled_warmup_us = warmup_us;
    graph.profiled_exec_us = exec_us;
    graph.profiled_energy_mj = energy_mj;
    graph.memory_bytes = memory_bytes;
    graph.supported = true;
    return graph;
}

ecofrontier::qnn_graph_select_request make_request(
        ecofrontier::qnn_graph_phase phase,
        const std::string & workpoint = "burst",
        uint64_t current_context_len = 100,
        uint64_t predicted_output_hi = 20) {
    ecofrontier::qnn_graph_select_request request;
    request.phase = phase;
    request.npu_workpoint = workpoint;
    request.current_context_len = current_context_len;
    request.predicted_output_hi = predicted_output_hi;
    return request;
}

ecofrontier::qnn_graph_choice choose_one(
        const ecofrontier::qnn_graph_descriptor & graph,
        uint64_t current_context_len,
        uint64_t predicted_output_hi) {
    ecofrontier::qnn_graph_catalog catalog;
    catalog.add_graph(graph);
    return catalog.choose_graph(make_request(graph.phase, "burst", current_context_len, predicted_output_hi));
}

std::string test_source_dir() {
    const char * env_source_dir = std::getenv("ECOFRONTIER_TEST_SOURCE_DIR");
    if (env_source_dir != nullptr && env_source_dir[0] != '\0') {
        return env_source_dir;
    }
    return ECOFRONTIER_TEST_SOURCE_DIR;
}

bool require_graph_paths() {
    const char * require = std::getenv("ECOFRONTIER_TEST_REQUIRE_GRAPH_PATHS");
    return require != nullptr && std::string(require) == "1";
}

bool file_exists(const std::string & path) {
    std::ifstream in(path);
    return in.good();
}

} // namespace

int main() {
    testing t;

    t.test("loads valid qnn_graphs.json", [](testing & t) {
        const char * path = "qnn_graphs.json";
        std::ofstream out(path);
        out << R"json({
  "graphs": [
    {
      "graph_id": "qwen25_3b_decode_ctx2048",
      "path": "models/Qwen2.5/Qwen2.5-3B-AoT/qnn/config.json",
      "phase": "decode",
      "chunk_size": 1,
      "usable_kv_slots": 1920,
      "safety_margin": 32,
      "supported_workpoints": ["burst", "low_balanced"],
      "profiled_load_us": 1000,
      "profiled_warmup_us": 200,
      "profiled_exec_us": 300,
      "profiled_energy_mj": 1.5,
      "memory_bytes": 4096,
      "supported": true
    }
  ]
})json";
        out.close();

        const auto parsed = ecofrontier::qnn_graph_manifest_load_file(path);
        std::remove(path);

        t.assert_true("valid qnn_graphs.json should parse", parsed.ok);
        t.assert_equal("one graph should be loaded", (size_t) 1, parsed.graphs.size());
        t.assert_equal("graph id should round-trip", std::string("qwen25_3b_decode_ctx2048"), parsed.graphs[0].graph_id);
        t.assert_equal("usable kv slots must come from manifest", (uint64_t) 1920, parsed.graphs[0].usable_kv_slots);
    });

    t.test("loads checked-in Qwen2.5 3B qnn_graphs.json", [](testing & t) {
        const std::string path = test_source_dir() + "/configs/ecofrontier/qnn_graphs.json";
        const auto parsed = ecofrontier::qnn_graph_manifest_load_file(path);

        t.assert_true("checked-in 3B graph manifest should parse", parsed.ok);
        t.assert_equal("checked-in manifest should expose prefill/decode context graphs", (size_t) 8, parsed.graphs.size());
        t.assert_equal("checked-in manifest should not reject graph entries", (size_t) 0, parsed.rejected_entries.size());

        if (require_graph_paths()) {
            for (const auto & graph : parsed.graphs) {
                t.assert_true("manifest graph path should exist on device: " + graph.path, file_exists(graph.path));
            }
        }

        ecofrontier::qnn_graph_catalog catalog;
        for (const auto & graph : parsed.graphs) {
            catalog.add_graph(graph);
        }

        const auto choice = catalog.choose_graph(
                make_request(ecofrontier::qnn_graph_phase::DECODE, "burst", 1000, 64));

        t.assert_true("3B decode request should select a capacity-safe graph", !choice.fallback);
        t.assert_equal("capacity guard should skip ctx1024 and select ctx2048",
                       std::string("qwen25_3b_decode_ctx2048_burst"),
                       choice.chosen_graph);
        t.assert_equal("checked-in manifest capacity must be explicit", (uint64_t) 1920, choice.usable_kv_slots);
        t.assert_true("missing manifest energy should remain incomplete", !choice.energy_complete);
    });

    t.test("rejects graph missing usable_kv_slots", [](testing & t) {
        const auto parsed = ecofrontier::qnn_graph_manifest_parse(R"json({
  "graphs": [
    {
      "graph_id": "missing_capacity",
      "path": "missing.bin",
      "phase": "decode",
      "chunk_size": 1,
      "safety_margin": 8,
      "supported_workpoints": ["burst"],
      "profiled_load_us": 1,
      "profiled_warmup_us": 1,
      "profiled_exec_us": 1,
      "profiled_energy_mj": 1.0,
      "memory_bytes": 1,
      "supported": true
    }
  ]
})json");

        t.assert_true("manifest syntax is valid", parsed.ok);
        t.assert_equal("graph should not enter the catalog", (size_t) 0, parsed.graphs.size());
        t.assert_equal("missing capacity should be reported", std::string("MissingUsableKvSlots"), parsed.rejected_entries[0].reason);
    });

    t.test("does not infer usable_kv_slots from qnn_aot_context_size", [](testing & t) {
        const auto parsed = ecofrontier::qnn_graph_manifest_parse(R"json({
  "graphs": [
    {
      "graph_id": "context_size_is_not_capacity",
      "path": "ctx4096.bin",
      "phase": "decode",
      "chunk_size": 1,
      "qnn_aot_context_size": 4096,
      "safety_margin": 8,
      "supported_workpoints": ["burst"],
      "profiled_load_us": 1,
      "profiled_warmup_us": 1,
      "profiled_exec_us": 1,
      "profiled_energy_mj": 1.0,
      "memory_bytes": 1,
      "supported": true
    }
  ]
})json");

        t.assert_true("manifest syntax is valid", parsed.ok);
        t.assert_equal("context size alone must not create a usable graph", (size_t) 0, parsed.graphs.size());
        t.assert_equal("missing usable_kv_slots is the rejection reason", std::string("MissingUsableKvSlots"), parsed.rejected_entries[0].reason);
    });

    t.test("computes required_kv correctly", [](testing & t) {
        t.assert_equal("required_kv = current_context_len + predicted_output_hi + safety_margin",
                       (uint64_t) 132,
                       ecofrontier::qnn_graph_required_kv(96, 32, 4));
    });

    t.test("accepts graph when required_kv <= usable_kv_slots", [](testing & t) {
        const auto graph = make_graph("fits_at_boundary", ecofrontier::qnn_graph_phase::DECODE, 128, 1.0, 10);
        const auto choice = choose_one(graph, 100, 20);

        t.assert_true("boundary capacity should be feasible", !choice.fallback);
        t.assert_equal("chosen graph should be the safe graph", std::string("fits_at_boundary"), choice.chosen_graph);
        t.assert_equal("required_kv should include the graph safety margin", (uint64_t) 128, choice.required_kv);
    });

    t.test("rejects graph when required_kv > usable_kv_slots", [](testing & t) {
        const auto graph = make_graph("too_small", ecofrontier::qnn_graph_phase::DECODE, 127, 1.0, 10);
        const auto choice = choose_one(graph, 100, 20);

        t.assert_true("capacity overflow should fall back", choice.fallback);
        t.assert_equal("fallback reason should identify capacity", std::string("NoGraphWithSufficientCapacity"), choice.fallback_reason);
    });

    t.test("filters by phase", [](testing & t) {
        ecofrontier::qnn_graph_catalog catalog;
        catalog.add_graph(make_graph("prefill_graph", ecofrontier::qnn_graph_phase::PREFILL, 128, 0.1, 1));
        catalog.add_graph(make_graph("decode_graph", ecofrontier::qnn_graph_phase::DECODE, 128, 9.0, 1000));

        const auto choice = catalog.choose_graph(make_request(ecofrontier::qnn_graph_phase::DECODE));
        t.assert_true("decode request should not select a prefill graph", !choice.fallback);
        t.assert_equal("decode graph should be selected", std::string("decode_graph"), choice.chosen_graph);
    });

    t.test("filters by workpoint", [](testing & t) {
        ecofrontier::qnn_graph_catalog catalog;
        catalog.add_graph(make_graph("burst_only", ecofrontier::qnn_graph_phase::DECODE, 128, 0.1, 1, 1000, 100, 4096, 8, { "burst" }));
        catalog.add_graph(make_graph("low_balanced_only", ecofrontier::qnn_graph_phase::DECODE, 128, 9.0, 1000, 1000, 100, 4096, 8, { "low_balanced" }));

        const auto choice = catalog.choose_graph(make_request(ecofrontier::qnn_graph_phase::DECODE, "low_balanced"));
        t.assert_true("workpoint-compatible graph should be selected", !choice.fallback);
        t.assert_equal("incompatible workpoint graph should be filtered out", std::string("low_balanced_only"), choice.chosen_graph);
    });

    t.test("empty supported_workpoints matches generic requested workpoint", [](testing & t) {
        ecofrontier::qnn_graph_catalog catalog;
        catalog.add_graph(make_graph("generic_graph", ecofrontier::qnn_graph_phase::DECODE, 128, 0.1, 1, 1000, 100, 4096, 8, {}));

        const auto choice = catalog.choose_graph(make_request(ecofrontier::qnn_graph_phase::DECODE, "burst"));
        t.assert_true("empty supported_workpoints should be generic, not incompatible", !choice.fallback);
        t.assert_equal("generic graph should be selected for requested workpoint", std::string("generic_graph"), choice.chosen_graph);
    });

    t.test("does not select Failed graph", [](testing & t) {
        ecofrontier::qnn_graph_catalog catalog;
        catalog.add_graph(make_graph("failed_low_cost", ecofrontier::qnn_graph_phase::DECODE, 128, 0.1, 1));
        catalog.add_graph(make_graph("healthy_higher_cost", ecofrontier::qnn_graph_phase::DECODE, 128, 9.0, 1000));
        catalog.set_residency("failed_low_cost", ecofrontier::qnn_graph_residency::FAILED);

        const auto choice = catalog.choose_graph(make_request(ecofrontier::qnn_graph_phase::DECODE));
        t.assert_true("a non-failed graph remains feasible", !choice.fallback);
        t.assert_equal("failed graph must be excluded", std::string("healthy_higher_cost"), choice.chosen_graph);
    });

    t.test("selects lower-cost feasible graph", [](testing & t) {
        ecofrontier::qnn_graph_catalog catalog;
        catalog.add_graph(make_graph("smaller_higher_energy", ecofrontier::qnn_graph_phase::DECODE, 128, 8.0, 10));
        catalog.add_graph(make_graph("larger_lower_energy", ecofrontier::qnn_graph_phase::DECODE, 512, 2.0, 1000));

        const auto choice = catalog.choose_graph(make_request(ecofrontier::qnn_graph_phase::DECODE));
        t.assert_true("a safe graph should be selected", !choice.fallback);
        t.assert_equal("energy cost should beat smaller capacity", std::string("larger_lower_energy"), choice.chosen_graph);
    });

    t.test("does not treat missing energy as zero", [](testing & t) {
        ecofrontier::qnn_graph_catalog catalog;
        auto missing_energy = make_graph("missing_energy_high_exec", ecofrontier::qnn_graph_phase::DECODE, 128, 0.0, 1000);
        missing_energy.profiled_energy_mj.reset();
        catalog.add_graph(missing_energy);
        catalog.add_graph(make_graph("known_energy_low_exec", ecofrontier::qnn_graph_phase::DECODE, 128, 5.0, 100));

        const auto choice = catalog.choose_graph(make_request(ecofrontier::qnn_graph_phase::DECODE));
        t.assert_true("a safe graph should be selected", !choice.fallback);
        t.assert_equal("missing energy should fall through to exec cost, not compare as zero",
                       std::string("known_energy_low_exec"),
                       choice.chosen_graph);
    });

    t.test("sets energy_complete=false when energy is missing", [](testing & t) {
        ecofrontier::qnn_graph_catalog catalog;
        auto missing_energy = make_graph("missing_energy_low_exec", ecofrontier::qnn_graph_phase::DECODE, 128, 0.0, 100);
        missing_energy.profiled_energy_mj.reset();
        catalog.add_graph(missing_energy);
        catalog.add_graph(make_graph("known_energy_high_exec", ecofrontier::qnn_graph_phase::DECODE, 128, 5.0, 500));

        const auto choice = catalog.choose_graph(make_request(ecofrontier::qnn_graph_phase::DECODE));
        t.assert_true("missing-energy graph should be selectable by exec cost", !choice.fallback);
        t.assert_equal("expected missing-energy graph", std::string("missing_energy_low_exec"), choice.chosen_graph);
        t.assert_true("energy_complete should be false for missing profiled_energy_mj", !choice.energy_complete);
        t.assert_true("missing energy terms should name the missing field", !choice.missing_energy_terms.empty());
    });

    t.test("uses smaller capacity only as tie-breaker", [](testing & t) {
        {
            ecofrontier::qnn_graph_catalog catalog;
            catalog.add_graph(make_graph("smaller_higher_energy", ecofrontier::qnn_graph_phase::DECODE, 128, 9.0, 100));
            catalog.add_graph(make_graph("larger_lower_energy", ecofrontier::qnn_graph_phase::DECODE, 512, 1.0, 100));

            const auto choice = catalog.choose_graph(make_request(ecofrontier::qnn_graph_phase::DECODE));
            t.assert_equal("smaller capacity must not beat lower energy",
                           std::string("larger_lower_energy"),
                           choice.chosen_graph);
        }

        {
            ecofrontier::qnn_graph_catalog catalog;
            catalog.add_graph(make_graph("larger_equal_cost", ecofrontier::qnn_graph_phase::DECODE, 512, 1.0, 100, 10, 10, 10));
            catalog.add_graph(make_graph("smaller_equal_cost", ecofrontier::qnn_graph_phase::DECODE, 128, 1.0, 100, 10, 10, 10));

            const auto choice = catalog.choose_graph(make_request(ecofrontier::qnn_graph_phase::DECODE));
            t.assert_equal("smaller capacity should break otherwise equal costs",
                           std::string("smaller_equal_cost"),
                           choice.chosen_graph);
        }
    });

    t.test("ResidentWarm has lower exposed load than NotLoaded", [](testing & t) {
        const auto graph = make_graph("resident_warm", ecofrontier::qnn_graph_phase::DECODE, 128, 1.0, 100, 1234);

        t.assert_true("resident warm graph should expose less load than a cold load",
                      ecofrontier::qnn_graph_exposed_load_us(graph, ecofrontier::qnn_graph_residency::RESIDENT_WARM) <
                      ecofrontier::qnn_graph_exposed_load_us(graph, ecofrontier::qnn_graph_residency::NOT_LOADED));
    });

    t.test("ResidentCold exposes profiled warmup cost", [](testing & t) {
        const auto graph = make_graph("resident_cold", ecofrontier::qnn_graph_phase::DECODE, 128, 1.0, 100, 1234, 345);

        t.assert_equal("ResidentCold should expose profiled_warmup_us",
                       (uint64_t) 345,
                       ecofrontier::qnn_graph_exposed_load_us(graph, ecofrontier::qnn_graph_residency::RESIDENT_COLD));
        t.assert_equal("ResidentWarm should expose no load",
                       (uint64_t) 0,
                       ecofrontier::qnn_graph_exposed_load_us(graph, ecofrontier::qnn_graph_residency::RESIDENT_WARM));
        t.assert_equal("NotLoaded should expose profiled_load_us",
                       (uint64_t) 1234,
                       ecofrontier::qnn_graph_exposed_load_us(graph, ecofrontier::qnn_graph_residency::NOT_LOADED));
    });

    t.test("no feasible graph returns fallback", [](testing & t) {
        ecofrontier::qnn_graph_catalog catalog;
        catalog.add_graph(make_graph("too_small_a", ecofrontier::qnn_graph_phase::DECODE, 64, 1.0, 100));
        catalog.add_graph(make_graph("too_small_b", ecofrontier::qnn_graph_phase::DECODE, 65, 2.0, 200));

        const auto choice = catalog.choose_graph(make_request(ecofrontier::qnn_graph_phase::DECODE));
        t.assert_true("no capacity-safe graph should fall back", choice.fallback);
        t.assert_equal("fallback reason should identify capacity", std::string("NoGraphWithSufficientCapacity"), choice.fallback_reason);
    });

    t.test("trace contains required_kv, usable_kv_slots, residency, and fallback_reason", [](testing & t) {
        ecofrontier::qnn_graph_catalog catalog;
        catalog.add_graph(make_graph("trace_graph", ecofrontier::qnn_graph_phase::DECODE, 128, 1.0, 100, 777));

        const auto choice = catalog.choose_graph(make_request(ecofrontier::qnn_graph_phase::DECODE));
        const auto jsonl = ecofrontier::qnn_graph_choice_to_jsonl(choice);

        t.assert_true("choice should carry emitted JSONL trace", !choice.trace_jsonl.empty());
        t.assert_equal("carried JSONL trace should match serialized choice", jsonl, choice.trace_jsonl);
        t.assert_true("trace should include event name", jsonl.find("\"event\":\"ecofrontier_graph_choice\"") != std::string::npos);
        t.assert_true("trace should include required_kv", jsonl.find("\"required_kv\":128") != std::string::npos);
        t.assert_true("trace should include usable_kv_slots", jsonl.find("\"usable_kv_slots\":128") != std::string::npos);
        t.assert_true("trace should include residency", jsonl.find("\"residency\":\"NotLoaded\"") != std::string::npos);
        t.assert_true("trace should include fallback_reason", jsonl.find("\"fallback_reason\":\"\"") != std::string::npos);

        const auto fallback = catalog.choose_graph(make_request(ecofrontier::qnn_graph_phase::DECODE, "burst", 500, 20));
        const auto fallback_jsonl = ecofrontier::qnn_graph_choice_to_jsonl(fallback);
        t.assert_true("fallback choice should carry emitted JSONL trace", !fallback.trace_jsonl.empty());
        t.assert_equal("fallback carried trace should match serialized choice", fallback_jsonl, fallback.trace_jsonl);
        t.assert_true("fallback trace should include computed required_kv",
                      fallback_jsonl.find("\"required_kv\":528") != std::string::npos);
        t.assert_true("fallback trace should include rejected graph usable capacity",
                      fallback_jsonl.find("\"usable_kv_slots\":128") != std::string::npos);
        t.assert_true("fallback trace should include rejected graph residency",
                      fallback_jsonl.find("\"residency\":\"NotLoaded\"") != std::string::npos);
        t.assert_true("fallback trace should include precise reason",
                      fallback_jsonl.find("\"fallback_reason\":\"NoGraphWithSufficientCapacity\"") != std::string::npos);
    });

    t.test("synchronous load and unsupported preload report explicit status", [](testing & t) {
        ecofrontier::qnn_graph_catalog catalog;
        catalog.add_graph(make_graph("loadable_graph", ecofrontier::qnn_graph_phase::DECODE, 128, 1.0, 100, 555));

        const auto preload = catalog.preload_async("loadable_graph");
        t.assert_equal("v1 async preload should be unsupported",
                       (int) ecofrontier::qnn_graph_preload_status::UNSUPPORTED,
                       (int) preload.status);
        t.assert_equal("unsupported preload must not hide load time",
                       (int) ecofrontier::qnn_graph_residency::NOT_LOADED,
                       (int) catalog.residency_of("loadable_graph"));

        const auto loaded = catalog.load_sync("loadable_graph");
        t.assert_equal("synchronous load should report loaded status",
                       (int) ecofrontier::qnn_graph_load_status::LOADED,
                       (int) loaded.status);
        t.assert_equal("load should expose profiled load time",
                       (uint64_t) 555,
                       loaded.exposed_load_us);
        t.assert_equal("loaded graph should become ResidentCold",
                       (int) ecofrontier::qnn_graph_residency::RESIDENT_COLD,
                       (int) catalog.residency_of("loadable_graph"));
    });

    return t.summary();
}
