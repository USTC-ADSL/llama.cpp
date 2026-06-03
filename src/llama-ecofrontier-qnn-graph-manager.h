#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace ecofrontier {

enum class qnn_graph_phase {
    PREFILL,
    DECODE,
    UNKNOWN,
};

enum class qnn_graph_residency {
    NOT_LOADED,
    LOADING,
    RESIDENT_COLD,
    RESIDENT_WARM,
    FAILED,
};

enum class qnn_graph_load_status {
    LOADED,
    ALREADY_RESIDENT,
    NOT_FOUND,
    FAILED,
};

enum class qnn_graph_preload_status {
    UNSUPPORTED,
    NOT_FOUND,
};

struct qnn_graph_descriptor {
    std::string graph_id;
    std::string path;
    qnn_graph_phase phase = qnn_graph_phase::UNKNOWN;
    uint64_t chunk_size = 0;
    uint64_t usable_kv_slots = 0;
    uint64_t safety_margin = 0;
    std::vector<std::string> supported_workpoints;
    uint64_t profiled_load_us = 0;
    uint64_t profiled_warmup_us = 0;
    uint64_t profiled_exec_us = 0;
    std::optional<double> profiled_energy_mj;
    uint64_t memory_bytes = 0;
    bool supported = false;
};

struct qnn_graph_rejected_entry {
    std::string graph_id;
    std::string reason;
};

struct qnn_graph_manifest_parse_result {
    bool ok = false;
    std::string error;
    std::vector<qnn_graph_descriptor> graphs;
    std::vector<qnn_graph_rejected_entry> rejected_entries;
};

struct qnn_graph_select_request {
    qnn_graph_phase phase = qnn_graph_phase::UNKNOWN;
    std::string npu_workpoint;
    uint64_t current_context_len = 0;
    uint64_t predicted_output_hi = 0;
};

struct qnn_graph_choice {
    std::string event = "ecofrontier_graph_choice";
    qnn_graph_phase phase = qnn_graph_phase::UNKNOWN;
    std::string npu_workpoint;
    uint64_t current_context_len = 0;
    uint64_t predicted_output_hi = 0;
    uint64_t safety_margin = 0;
    uint64_t required_kv = 0;
    std::string chosen_graph;
    uint64_t usable_kv_slots = 0;
    uint64_t chunk_size = 0;
    qnn_graph_residency residency = qnn_graph_residency::NOT_LOADED;
    uint64_t exposed_load_us = 0;
    bool energy_complete = true;
    std::vector<std::string> missing_energy_terms;
    bool fallback = true;
    std::string fallback_reason;
    std::string trace_jsonl;
};

struct qnn_graph_load_result {
    qnn_graph_load_status status = qnn_graph_load_status::NOT_FOUND;
    uint64_t exposed_load_us = 0;
    std::string reason;
};

struct qnn_graph_preload_result {
    qnn_graph_preload_status status = qnn_graph_preload_status::UNSUPPORTED;
    std::string reason;
};

uint64_t qnn_graph_required_kv(
        uint64_t current_context_len,
        uint64_t predicted_output_hi,
        uint64_t safety_margin);

uint64_t qnn_graph_exposed_load_us(
        const qnn_graph_descriptor & graph,
        qnn_graph_residency residency);

const char * qnn_graph_phase_name(qnn_graph_phase phase);
const char * qnn_graph_residency_name(qnn_graph_residency residency);

qnn_graph_manifest_parse_result qnn_graph_manifest_parse(const std::string & json_text);
qnn_graph_manifest_parse_result qnn_graph_manifest_load_file(const std::string & path);

std::string qnn_graph_choice_to_jsonl(const qnn_graph_choice & choice);

class qnn_graph_catalog {
  public:
    bool add_graph(const qnn_graph_descriptor & graph);
    size_t size() const;

    qnn_graph_residency residency_of(const std::string & graph_id) const;
    bool set_residency(const std::string & graph_id, qnn_graph_residency residency);

    qnn_graph_choice choose_graph(const qnn_graph_select_request & request) const;

    qnn_graph_load_result load_sync(const std::string & graph_id);
    qnn_graph_preload_result preload_async(const std::string & graph_id);

  private:
    const qnn_graph_descriptor * find_graph(const std::string & graph_id) const;
    qnn_graph_residency residency_of_existing(const std::string & graph_id) const;

    std::vector<qnn_graph_descriptor> graphs;
    std::unordered_map<std::string, size_t> index_by_graph_id;
    std::unordered_map<std::string, qnn_graph_residency> residency_by_graph_id;
};

} // namespace ecofrontier
