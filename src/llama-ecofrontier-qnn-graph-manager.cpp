#include "llama-ecofrontier-qnn-graph-manager.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <sstream>
#include <utility>

#include <nlohmann/json.hpp>

namespace ecofrontier {

namespace {

using json = nlohmann::ordered_json;

std::string to_lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        if (ch >= 'A' && ch <= 'Z') {
            return static_cast<char>(ch - 'A' + 'a');
        }
        return static_cast<char>(ch);
    });
    return value;
}

qnn_graph_phase parse_phase(const std::string & value) {
    const std::string normalized = to_lower_ascii(value);
    if (normalized == "prefill") {
        return qnn_graph_phase::PREFILL;
    }
    if (normalized == "decode") {
        return qnn_graph_phase::DECODE;
    }
    return qnn_graph_phase::UNKNOWN;
}

bool has_required_field(
        const json & entry,
        const char * field,
        std::string & reason) {
    if (entry.contains(field)) {
        return true;
    }

    reason = std::string("Missing") + field;
    return false;
}

bool read_string_field(
        const json & entry,
        const char * field,
        std::string & out,
        std::string & reason) {
    if (!has_required_field(entry, field, reason)) {
        return false;
    }
    if (!entry[field].is_string()) {
        reason = std::string("Invalid") + field;
        return false;
    }
    out = entry[field].get<std::string>();
    return true;
}

bool read_bool_field(
        const json & entry,
        const char * field,
        bool & out,
        std::string & reason) {
    if (!has_required_field(entry, field, reason)) {
        return false;
    }
    if (!entry[field].is_boolean()) {
        reason = std::string("Invalid") + field;
        return false;
    }
    out = entry[field].get<bool>();
    return true;
}

bool read_uint64_field(
        const json & entry,
        const char * field,
        uint64_t & out,
        std::string & reason) {
    if (!has_required_field(entry, field, reason)) {
        return false;
    }
    if (!entry[field].is_number_unsigned()) {
        reason = std::string("Invalid") + field;
        return false;
    }
    out = entry[field].get<uint64_t>();
    return true;
}

bool read_optional_double_field(
        const json & entry,
        const char * field,
        std::optional<double> & out,
        std::string & reason) {
    out.reset();
    if (!entry.contains(field) || entry[field].is_null()) {
        return true;
    }
    if (!entry[field].is_number()) {
        reason = std::string("Invalid") + field;
        return false;
    }
    out = entry[field].get<double>();
    return true;
}

bool read_workpoints(
        const json & entry,
        std::vector<std::string> & out,
        std::string & reason) {
    if (!has_required_field(entry, "supported_workpoints", reason)) {
        return false;
    }
    if (!entry["supported_workpoints"].is_array()) {
        reason = "Invalidsupported_workpoints";
        return false;
    }

    out.clear();
    for (const auto & item : entry["supported_workpoints"]) {
        if (!item.is_string()) {
            reason = "Invalidsupported_workpoints";
            return false;
        }
        out.push_back(item.get<std::string>());
    }
    return true;
}

std::string graph_id_for_rejection(const json & entry, size_t index) {
    if (entry.is_object() && entry.contains("graph_id") && entry["graph_id"].is_string()) {
        return entry["graph_id"].get<std::string>();
    }
    return std::string("<entry-") + std::to_string(index) + ">";
}

bool parse_graph_entry(
        const json & entry,
        qnn_graph_descriptor & graph,
        std::string & reason) {
    if (!entry.is_object()) {
        reason = "InvalidGraphEntry";
        return false;
    }

    std::string phase;
    if (!read_string_field(entry, "graph_id", graph.graph_id, reason) ||
        !read_string_field(entry, "path", graph.path, reason) ||
        !read_string_field(entry, "phase", phase, reason)) {
        return false;
    }

    graph.phase = parse_phase(phase);
    if (graph.phase == qnn_graph_phase::UNKNOWN) {
        reason = "Invalidphase";
        return false;
    }

    if (!read_uint64_field(entry, "chunk_size", graph.chunk_size, reason)) {
        return false;
    }

    if (!entry.contains("usable_kv_slots")) {
        reason = "MissingUsableKvSlots";
        return false;
    }
    if (!entry["usable_kv_slots"].is_number_unsigned()) {
        reason = "Invalidusable_kv_slots";
        return false;
    }
    graph.usable_kv_slots = entry["usable_kv_slots"].get<uint64_t>();

    if (!read_uint64_field(entry, "safety_margin", graph.safety_margin, reason) ||
        !read_workpoints(entry, graph.supported_workpoints, reason) ||
        !read_uint64_field(entry, "profiled_load_us", graph.profiled_load_us, reason) ||
        !read_uint64_field(entry, "profiled_warmup_us", graph.profiled_warmup_us, reason) ||
        !read_uint64_field(entry, "profiled_exec_us", graph.profiled_exec_us, reason) ||
        !read_optional_double_field(entry, "profiled_energy_mj", graph.profiled_energy_mj, reason) ||
        !read_uint64_field(entry, "memory_bytes", graph.memory_bytes, reason) ||
        !read_bool_field(entry, "supported", graph.supported, reason)) {
        return false;
    }

    return true;
}

bool workpoint_matches(
        const qnn_graph_descriptor & graph,
        const std::string & workpoint) {
    if (workpoint.empty() || graph.supported_workpoints.empty()) {
        return true;
    }

    return std::find(
            graph.supported_workpoints.begin(),
            graph.supported_workpoints.end(),
            workpoint) != graph.supported_workpoints.end();
}

bool compare_optional_energy(
        const std::optional<double> & left,
        const std::optional<double> & right,
        bool & decided) {
    decided = false;
    if (left.has_value() && right.has_value() && *left != *right) {
        decided = true;
        return *left < *right;
    }
    return false;
}

bool graph_cost_less(
        const qnn_graph_descriptor & left,
        qnn_graph_residency left_residency,
        const qnn_graph_descriptor & right,
        qnn_graph_residency right_residency) {
    bool decided = false;
    const bool left_lower_energy = compare_optional_energy(left.profiled_energy_mj, right.profiled_energy_mj, decided);
    if (decided) {
        return left_lower_energy;
    }

    if (left.profiled_exec_us != right.profiled_exec_us) {
        return left.profiled_exec_us < right.profiled_exec_us;
    }

    const uint64_t left_load = qnn_graph_exposed_load_us(left, left_residency);
    const uint64_t right_load = qnn_graph_exposed_load_us(right, right_residency);
    if (left_load != right_load) {
        return left_load < right_load;
    }

    if (left.profiled_warmup_us != right.profiled_warmup_us) {
        return left.profiled_warmup_us < right.profiled_warmup_us;
    }

    if (left.memory_bytes != right.memory_bytes) {
        return left.memory_bytes < right.memory_bytes;
    }

    if (left.usable_kv_slots != right.usable_kv_slots) {
        return left.usable_kv_slots < right.usable_kv_slots;
    }

    return left.graph_id < right.graph_id;
}

std::string fallback_reason_for_filters(
        bool saw_phase,
        bool saw_supported,
        bool saw_workpoint,
        bool saw_not_failed,
        bool saw_capacity_candidate) {
    if (!saw_phase) {
        return "NoGraphForPhase";
    }
    if (!saw_supported) {
        return "NoSupportedGraph";
    }
    if (!saw_workpoint) {
        return "NoGraphForWorkpoint";
    }
    if (!saw_not_failed) {
        return "NoGraphNotFailed";
    }
    if (!saw_capacity_candidate) {
        return "NoGraphWithSufficientCapacity";
    }
    return "NoFeasibleGraph";
}

json missing_terms_to_json(const std::vector<std::string> & terms) {
    json out = json::array();
    for (const std::string & term : terms) {
        out.push_back(term);
    }
    return out;
}

} // namespace

uint64_t qnn_graph_required_kv(
        uint64_t current_context_len,
        uint64_t predicted_output_hi,
        uint64_t safety_margin) {
    const uint64_t max = std::numeric_limits<uint64_t>::max();

    if (current_context_len > max - predicted_output_hi) {
        return max;
    }
    const uint64_t partial = current_context_len + predicted_output_hi;
    if (partial > max - safety_margin) {
        return max;
    }
    return partial + safety_margin;
}

uint64_t qnn_graph_exposed_load_us(
        const qnn_graph_descriptor & graph,
        qnn_graph_residency residency) {
    switch (residency) {
        case qnn_graph_residency::NOT_LOADED:
        case qnn_graph_residency::FAILED:
            return graph.profiled_load_us;
        case qnn_graph_residency::LOADING:
            return graph.profiled_load_us;
        case qnn_graph_residency::RESIDENT_COLD:
            return graph.profiled_warmup_us;
        case qnn_graph_residency::RESIDENT_WARM:
            return 0;
    }

    return graph.profiled_load_us;
}

const char * qnn_graph_phase_name(qnn_graph_phase phase) {
    switch (phase) {
        case qnn_graph_phase::PREFILL:
            return "prefill";
        case qnn_graph_phase::DECODE:
            return "decode";
        case qnn_graph_phase::UNKNOWN:
            return "unknown";
    }

    return "unknown";
}

const char * qnn_graph_residency_name(qnn_graph_residency residency) {
    switch (residency) {
        case qnn_graph_residency::NOT_LOADED:
            return "NotLoaded";
        case qnn_graph_residency::LOADING:
            return "Loading";
        case qnn_graph_residency::RESIDENT_COLD:
            return "ResidentCold";
        case qnn_graph_residency::RESIDENT_WARM:
            return "ResidentWarm";
        case qnn_graph_residency::FAILED:
            return "Failed";
    }

    return "Unknown";
}

qnn_graph_manifest_parse_result qnn_graph_manifest_parse(const std::string & json_text) {
    qnn_graph_manifest_parse_result result;

    json root;
    try {
        root = json::parse(json_text);
    } catch (const json::exception & e) {
        result.ok = false;
        result.error = e.what();
        return result;
    }

    const json * graph_entries = nullptr;
    if (root.is_object() && root.contains("graphs") && root["graphs"].is_array()) {
        graph_entries = &root["graphs"];
    } else if (root.is_array()) {
        graph_entries = &root;
    } else {
        result.ok = false;
        result.error = "Expected qnn_graphs.json root object with graphs array";
        return result;
    }

    result.ok = true;

    for (size_t i = 0; i < graph_entries->size(); ++i) {
        const json & entry = (*graph_entries)[i];
        qnn_graph_descriptor graph;
        std::string reason;
        if (parse_graph_entry(entry, graph, reason)) {
            result.graphs.push_back(std::move(graph));
        } else {
            result.rejected_entries.push_back({ graph_id_for_rejection(entry, i), reason });
        }
    }

    return result;
}

qnn_graph_manifest_parse_result qnn_graph_manifest_load_file(const std::string & path) {
    std::ifstream in(path);
    if (!in) {
        qnn_graph_manifest_parse_result result;
        result.ok = false;
        result.error = "Failed to open manifest: " + path;
        return result;
    }

    std::ostringstream text;
    text << in.rdbuf();
    return qnn_graph_manifest_parse(text.str());
}

std::string qnn_graph_choice_to_jsonl(const qnn_graph_choice & choice) {
    json event = {
        { "event", choice.event },
        { "phase", qnn_graph_phase_name(choice.phase) },
        { "npu_workpoint", choice.npu_workpoint },
        { "current_context_len", choice.current_context_len },
        { "predicted_output_hi", choice.predicted_output_hi },
        { "safety_margin", choice.safety_margin },
        { "required_kv", choice.required_kv },
        { "chosen_graph", choice.chosen_graph },
        { "usable_kv_slots", choice.usable_kv_slots },
        { "chunk_size", choice.chunk_size },
        { "residency", qnn_graph_residency_name(choice.residency) },
        { "exposed_load_us", choice.exposed_load_us },
        { "energy_complete", choice.energy_complete },
        { "missing_energy_terms", missing_terms_to_json(choice.missing_energy_terms) },
        { "fallback", choice.fallback },
        { "fallback_reason", choice.fallback_reason },
    };

    return event.dump(-1, ' ', false, json::error_handler_t::replace) + "\n";
}

bool qnn_graph_catalog::add_graph(const qnn_graph_descriptor & graph) {
    if (graph.graph_id.empty() || index_by_graph_id.find(graph.graph_id) != index_by_graph_id.end()) {
        return false;
    }

    index_by_graph_id[graph.graph_id] = graphs.size();
    graphs.push_back(graph);
    residency_by_graph_id[graph.graph_id] = qnn_graph_residency::NOT_LOADED;
    return true;
}

size_t qnn_graph_catalog::size() const {
    return graphs.size();
}

qnn_graph_residency qnn_graph_catalog::residency_of(const std::string & graph_id) const {
    if (index_by_graph_id.find(graph_id) == index_by_graph_id.end()) {
        return qnn_graph_residency::FAILED;
    }
    return residency_of_existing(graph_id);
}

bool qnn_graph_catalog::set_residency(const std::string & graph_id, qnn_graph_residency residency) {
    if (index_by_graph_id.find(graph_id) == index_by_graph_id.end()) {
        return false;
    }
    residency_by_graph_id[graph_id] = residency;
    return true;
}

qnn_graph_choice qnn_graph_catalog::choose_graph(const qnn_graph_select_request & request) const {
    qnn_graph_choice choice;
    choice.phase = request.phase;
    choice.npu_workpoint = request.npu_workpoint;
    choice.current_context_len = request.current_context_len;
    choice.predicted_output_hi = request.predicted_output_hi;

    const qnn_graph_descriptor * best = nullptr;
    qnn_graph_residency best_residency = qnn_graph_residency::NOT_LOADED;

    bool saw_phase = false;
    bool saw_supported = false;
    bool saw_workpoint = false;
    bool saw_not_failed = false;
    bool saw_capacity_candidate = false;
    uint64_t smallest_required_kv_for_rejected_capacity = 0;
    uint64_t safety_margin_for_rejected_capacity = 0;
    uint64_t usable_kv_slots_for_rejected_capacity = 0;
    uint64_t chunk_size_for_rejected_capacity = 0;
    qnn_graph_residency residency_for_rejected_capacity = qnn_graph_residency::NOT_LOADED;
    uint64_t exposed_load_us_for_rejected_capacity = 0;

    for (const qnn_graph_descriptor & graph : graphs) {
        if (graph.phase != request.phase) {
            continue;
        }
        saw_phase = true;

        if (!graph.supported) {
            continue;
        }
        saw_supported = true;

        if (!workpoint_matches(graph, request.npu_workpoint)) {
            continue;
        }
        saw_workpoint = true;

        const qnn_graph_residency residency = residency_of_existing(graph.graph_id);
        if (residency == qnn_graph_residency::FAILED) {
            continue;
        }
        saw_not_failed = true;

        const uint64_t required_kv = qnn_graph_required_kv(
                request.current_context_len,
                request.predicted_output_hi,
                graph.safety_margin);
        if (required_kv > graph.usable_kv_slots) {
            if (smallest_required_kv_for_rejected_capacity == 0 || required_kv < smallest_required_kv_for_rejected_capacity) {
                smallest_required_kv_for_rejected_capacity = required_kv;
                safety_margin_for_rejected_capacity = graph.safety_margin;
                usable_kv_slots_for_rejected_capacity = graph.usable_kv_slots;
                chunk_size_for_rejected_capacity = graph.chunk_size;
                residency_for_rejected_capacity = residency;
                exposed_load_us_for_rejected_capacity = qnn_graph_exposed_load_us(graph, residency);
            }
            continue;
        }
        saw_capacity_candidate = true;

        if (best == nullptr || graph_cost_less(graph, residency, *best, best_residency)) {
            best = &graph;
            best_residency = residency;
        }
    }

    if (best == nullptr) {
        choice.fallback = true;
        choice.fallback_reason = fallback_reason_for_filters(
                saw_phase,
                saw_supported,
                saw_workpoint,
                saw_not_failed,
                saw_capacity_candidate);
        choice.required_kv = smallest_required_kv_for_rejected_capacity;
        choice.safety_margin = safety_margin_for_rejected_capacity;
        choice.usable_kv_slots = usable_kv_slots_for_rejected_capacity;
        choice.chunk_size = chunk_size_for_rejected_capacity;
        choice.residency = residency_for_rejected_capacity;
        choice.exposed_load_us = exposed_load_us_for_rejected_capacity;
        choice.trace_jsonl = qnn_graph_choice_to_jsonl(choice);
        return choice;
    }

    choice.fallback = false;
    choice.fallback_reason.clear();
    choice.chosen_graph = best->graph_id;
    choice.safety_margin = best->safety_margin;
    choice.required_kv = qnn_graph_required_kv(
            request.current_context_len,
            request.predicted_output_hi,
            best->safety_margin);
    choice.usable_kv_slots = best->usable_kv_slots;
    choice.chunk_size = best->chunk_size;
    choice.residency = best_residency;
    choice.exposed_load_us = qnn_graph_exposed_load_us(*best, best_residency);
    choice.energy_complete = best->profiled_energy_mj.has_value();
    if (!choice.energy_complete) {
        choice.missing_energy_terms.push_back("profiled_energy_mj");
    }

    choice.trace_jsonl = qnn_graph_choice_to_jsonl(choice);
    return choice;
}

qnn_graph_load_result qnn_graph_catalog::load_sync(const std::string & graph_id) {
    qnn_graph_load_result result;
    const qnn_graph_descriptor * graph = find_graph(graph_id);
    if (graph == nullptr) {
        result.status = qnn_graph_load_status::NOT_FOUND;
        result.reason = "GraphNotFound";
        return result;
    }

    const qnn_graph_residency residency = residency_of_existing(graph_id);
    if (residency == qnn_graph_residency::FAILED) {
        result.status = qnn_graph_load_status::FAILED;
        result.reason = "GraphFailed";
        return result;
    }

    result.exposed_load_us = qnn_graph_exposed_load_us(*graph, residency);
    if (residency == qnn_graph_residency::RESIDENT_COLD || residency == qnn_graph_residency::RESIDENT_WARM) {
        result.status = qnn_graph_load_status::ALREADY_RESIDENT;
        return result;
    }

    residency_by_graph_id[graph_id] = qnn_graph_residency::RESIDENT_COLD;
    result.status = qnn_graph_load_status::LOADED;
    return result;
}

qnn_graph_preload_result qnn_graph_catalog::preload_async(const std::string & graph_id) {
    qnn_graph_preload_result result;
    if (find_graph(graph_id) == nullptr) {
        result.status = qnn_graph_preload_status::NOT_FOUND;
        result.reason = "GraphNotFound";
        return result;
    }

    result.status = qnn_graph_preload_status::UNSUPPORTED;
    result.reason = "AsyncPreloadUnsupported";
    return result;
}

const qnn_graph_descriptor * qnn_graph_catalog::find_graph(const std::string & graph_id) const {
    const auto it = index_by_graph_id.find(graph_id);
    if (it == index_by_graph_id.end()) {
        return nullptr;
    }
    return &graphs[it->second];
}

qnn_graph_residency qnn_graph_catalog::residency_of_existing(const std::string & graph_id) const {
    const auto it = residency_by_graph_id.find(graph_id);
    if (it == residency_by_graph_id.end()) {
        return qnn_graph_residency::NOT_LOADED;
    }
    return it->second;
}

} // namespace ecofrontier
