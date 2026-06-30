#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef SYSTEM_BENEFIT_PLANNER_NO_MAIN
namespace system_benefit_planner {
#else
namespace {
#endif

struct profile_record {
    std::string phase;
    std::string backend;
    std::string state_name;
    std::string state_group;
    int bucket_lo = 0;
    int bucket_hi = 0;
    int bucket_tokens = 0;
    double throughput_tps = 0.0;
    double power_mw = 0.0;

    double mean_tbt_ms() const {
        return 1000.0 / throughput_tps;
    }

    double latency_ms_for_tokens(int tokens) const {
        return static_cast<double>(tokens) / throughput_tps * 1000.0;
    }

    double energy_mj_for_tokens(int tokens) const {
        return power_mw * latency_ms_for_tokens(tokens) / 1000.0;
    }
};

struct segment_candidate {
    int segment_id = 0;
    int context_lo = 0;
    int context_hi = 0;
    int profile_query_lo = 0;
    int profile_query_hi = 0;
    int tokens = 0;
    int state_id = -1;
    const profile_record * record = nullptr;
    std::string match_kind;

    double latency_ms() const {
        return record->latency_ms_for_tokens(tokens);
    }

    double energy_mj() const {
        return record->energy_mj_for_tokens(tokens);
    }
};

struct transition_cost {
    double latency_ms = 0.0;
    double energy_mj = 0.0;
    std::string source = "same_state";
};

struct dp_node {
    bool valid = false;
    double total_energy_mj = 0.0;
    double total_latency_ms = 0.0;
    int slo_violations = 0;
    double total_slo_miss_ms = 0.0;
    int prev_state_index = -1;
    segment_candidate candidate;
    transition_cost transition;
    double step_latency_ms = 0.0;
    double step_energy_mj = 0.0;
    double step_slo_deadline_ms = 0.0;
    bool step_slo_ok = true;
    double step_slo_miss_ms = 0.0;
};

struct prefill_plan {
    std::string state_name = "npu_burst";
    std::string backend = "NPU";
    std::string state_group = "burst";
    std::string route_spec = "qnn-npu{workpoint=burst}";
    std::string qnn_workpoint = "burst";
    uint64_t qnn_context_size = 0;
    uint64_t qnn_required_kv_slots = 0;
    bool from_profile = false;
    bool slo_ok = false;
    double latency_ms = std::numeric_limits<double>::quiet_NaN();
    double energy_mj = std::numeric_limits<double>::quiet_NaN();
};

struct decode_plan {
    std::vector<dp_node> nodes;
    std::string schedule;
    int slo_satisfied_steps = 0;
    int slo_total_steps = 0;
    double latency_ms = 0.0;
    double energy_mj = 0.0;
};

struct options {
    std::string prefill_profile;
    std::string decode_profile;
    double ttft_slo_ms = 0.0;
    double tbt_slo_ms = 0.0;
    int input_len = -1;
    int output_len = -1;
    int bucket_size = 32;
    std::string context_match = "exact";
    std::string output_format = "env";
    std::string initial_decode_state;
};

struct decode_segment_range {
    int segment_id = 0;
    int context_lo = 0;
    int context_hi = 0;
    int profile_query_lo = 0;
    int profile_query_hi = 0;
    int tokens = 0;
};

struct qnn_graph_capacity {
    uint64_t context_size = 0;
    uint64_t usable_kv_slots = 0;
};

struct indexed_profile_record {
    const profile_record * record = nullptr;
    int row_index = 0;
    double mid = 0.0;
};

struct planner_state_meta {
    int id = -1;
    int backend_id = -1;
    std::string backend;
    std::string state_name;
    std::string state_group;
    std::set<std::string> aliases;
    uint64_t qnn_explicit_context_size = 0;
    std::string npu_workpoint;
    std::string static_route_spec;
};

struct decode_profile_index {
    std::vector<planner_state_meta> states;
    std::vector<std::string> backend_names;
    std::vector<std::vector<int>> states_by_backend;
    std::vector<std::vector<indexed_profile_record>> records_by_state;
    std::vector<std::vector<const indexed_profile_record *>> records_by_state_mid;
    std::vector<std::unordered_map<std::string, const indexed_profile_record *>> exact_bucket_by_state;
    std::unordered_map<std::string, int> state_id_by_backend_alias;
    std::vector<std::vector<transition_cost>> backend_transition_costs;
};

double default_backend_transition_latency_ms(const std::string & from_backend, const std::string & to_backend);
bool better_dp_node(const dp_node & lhs, const dp_node & rhs);

static constexpr qnn_graph_capacity QNN_GRAPH_CAPACITIES[] = {
    {2048, 1920},
    {4096, 3968},
    {6144, 6016},
};

std::string trim(const std::string & s) {
    size_t first = 0;
    while (first < s.size() && std::isspace(static_cast<unsigned char>(s[first]))) {
        ++first;
    }
    size_t last = s.size();
    while (last > first && std::isspace(static_cast<unsigned char>(s[last - 1]))) {
        --last;
    }
    return s.substr(first, last - first);
}

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::string to_upper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return s;
}

std::vector<std::string> split_csv_line(const std::string & line) {
    std::vector<std::string> out;
    std::string cell;
    bool quoted = false;
    for (size_t i = 0; i < line.size(); ++i) {
        const char ch = line[i];
        if (ch == '"') {
            if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
                cell.push_back('"');
                ++i;
            } else {
                quoted = !quoted;
            }
        } else if (ch == ',' && !quoted) {
            out.push_back(trim(cell));
            cell.clear();
        } else {
            cell.push_back(ch);
        }
    }
    out.push_back(trim(cell));
    return out;
}

double parse_double(const std::string & value, double def = std::numeric_limits<double>::quiet_NaN()) {
    const std::string text = trim(value);
    if (text.empty()) {
        return def;
    }
    char * end = nullptr;
    const double parsed = std::strtod(text.c_str(), &end);
    if (end == text.c_str() || !std::isfinite(parsed)) {
        return def;
    }
    return parsed;
}

int parse_int(const std::string & value, int def = 0) {
    const double parsed = parse_double(value, std::numeric_limits<double>::quiet_NaN());
    if (!std::isfinite(parsed)) {
        return def;
    }
    return static_cast<int>(std::llround(parsed));
}

std::string normalize_backend(const std::string & value) {
    const std::string upper = to_upper(trim(value));
    if (upper == "QNN_NPU" || upper == "QNN-NPU" || upper == "HTP" || upper.find("NPU") != std::string::npos) {
        return "NPU";
    }
    if (upper.rfind("GPU", 0) == 0 || upper == "OPENCL") {
        return "GPU";
    }
    if (upper.rfind("CPU", 0) == 0) {
        return "CPU";
    }
    return upper;
}

std::vector<int> extract_numbers(const std::string & text) {
    std::vector<int> numbers;
    static const std::regex number_re("(\\d+)");
    for (std::sregex_iterator it(text.begin(), text.end(), number_re), end; it != end; ++it) {
        numbers.push_back(std::stoi((*it)[1].str()));
    }
    return numbers;
}

std::string remove_prefix(const std::string & text, const std::string & prefix) {
    if (text.rfind(prefix, 0) == 0) {
        return text.substr(prefix.size());
    }
    return text;
}

std::set<std::string> state_aliases(const std::string & backend_in, const std::string & state_name) {
    const std::string backend = normalize_backend(backend_in);
    const std::string clean = trim(state_name);
    std::set<std::string> aliases{clean};
    if (backend == "GPU") {
        const std::string suffix = remove_prefix(clean, "gpu_");
        aliases.insert(suffix);
        aliases.insert("gpu_" + suffix);
    } else if (backend == "NPU") {
        const std::string suffix = remove_prefix(clean, "npu_");
        aliases.insert(suffix);
        aliases.insert("npu_" + suffix);
    } else if (backend == "CPU") {
        const std::string suffix = remove_prefix(clean, "cpu_");
        aliases.insert(suffix);
        aliases.insert("cpu_" + suffix);
        static const std::regex simple_re("([A-Za-z0-9]+)_(\\d+)_(\\d+)");
        static const std::regex verbose_re("([A-Za-z0-9]+)_big(\\d+)_little(\\d+)");
        std::smatch match;
        if (std::regex_match(suffix, match, simple_re)) {
            aliases.insert(match[1].str() + "_big" + match[2].str() + "_little" + match[3].str());
            aliases.insert("cpu_" + match[1].str() + "_big" + match[2].str() + "_little" + match[3].str());
        }
        if (std::regex_match(suffix, match, verbose_re)) {
            aliases.insert(match[1].str() + "_" + match[2].str() + "_" + match[3].str());
            aliases.insert("cpu_" + match[1].str() + "_" + match[2].str() + "_" + match[3].str());
        }
    }
    return aliases;
}

bool state_matches(const std::string & backend, const std::string & profile_state, const std::string & requested_state) {
    const auto lhs = state_aliases(backend, profile_state);
    const auto rhs = state_aliases(backend, requested_state);
    for (const auto & item : lhs) {
        if (rhs.count(item)) {
            return true;
        }
    }
    return false;
}

std::vector<profile_record> load_profile_csv(const std::string & path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open profile: " + path);
    }

    std::string line;
    if (!std::getline(in, line)) {
        throw std::runtime_error("empty profile: " + path);
    }
    const std::vector<std::string> header = split_csv_line(line);
    std::unordered_map<std::string, size_t> col;
    for (size_t i = 0; i < header.size(); ++i) {
        col[header[i]] = i;
    }

    const std::vector<std::string> required = {
        "phase", "backend", "state_name", "state_group", "bucket_lo", "bucket_hi",
        "bucket_tokens", "throughput_tps", "power_mw",
    };
    for (const auto & name : required) {
        if (!col.count(name)) {
            throw std::runtime_error("profile " + path + " misses required column: " + name);
        }
    }

    auto field = [&](const std::vector<std::string> & row, const std::string & name) -> std::string {
        const auto it = col.find(name);
        if (it == col.end() || it->second >= row.size()) {
            return "";
        }
        return row[it->second];
    };

    std::vector<profile_record> records;
    while (std::getline(in, line)) {
        if (trim(line).empty()) {
            continue;
        }
        const std::vector<std::string> row = split_csv_line(line);
        profile_record record;
        record.phase = to_lower(field(row, "phase"));
        record.backend = normalize_backend(field(row, "backend"));
        record.state_name = field(row, "state_name");
        record.state_group = field(row, "state_group");
        record.bucket_lo = parse_int(field(row, "bucket_lo"));
        record.bucket_hi = parse_int(field(row, "bucket_hi"));
        record.bucket_tokens = parse_int(field(row, "bucket_tokens"));
        record.throughput_tps = parse_double(field(row, "throughput_tps"), 0.0);
        record.power_mw = parse_double(field(row, "power_mw"), -1.0);
        if (record.phase.empty() || record.backend.empty() || record.state_name.empty()) {
            continue;
        }
        if (record.bucket_lo <= 0 || record.bucket_hi <= 0 || record.throughput_tps <= 0.0 || record.power_mw < 0.0) {
            continue;
        }
        records.push_back(record);
    }
    return records;
}

double bucket_mid(int lo, int hi) {
    return (static_cast<double>(lo) + static_cast<double>(hi)) / 2.0;
}

std::pair<double, std::string> bucket_match(const profile_record & record, int query_lo, int query_hi, const std::string & mode) {
    const double no_match = std::numeric_limits<double>::infinity();
    if (record.bucket_lo == record.bucket_hi) {
        const int point = record.bucket_hi;
        if (point == query_hi) {
            return {0.0, "exact"};
        }
        if (mode == "floor" && point <= query_hi) {
            return {static_cast<double>(std::abs(point - query_hi)), "floor"};
        }
        if (mode == "ceil" && point >= query_hi) {
            return {static_cast<double>(std::abs(point - query_hi)), "ceil"};
        }
        if (mode == "nearest") {
            return {static_cast<double>(std::abs(point - query_hi)), "nearest"};
        }
        return {no_match, ""};
    }

    if (record.bucket_lo == query_lo && record.bucket_hi == query_hi) {
        return {0.0, "exact"};
    }
    if (record.bucket_lo <= query_lo && record.bucket_hi >= query_hi) {
        return {0.0, mode == "exact" ? "covering_partial_bucket" : "covering_bucket"};
    }
    const double distance = std::abs(bucket_mid(record.bucket_lo, record.bucket_hi) - bucket_mid(query_lo, query_hi));
    if (mode == "floor" && record.bucket_hi <= query_hi) {
        return {distance, "floor"};
    }
    if (mode == "ceil" && record.bucket_lo >= query_lo) {
        return {distance, "ceil"};
    }
    if (mode == "nearest") {
        return {distance, "nearest"};
    }
    return {no_match, ""};
}

std::vector<decode_segment_range> aligned_decode_segments(int input_len, int output_len, int bucket_size) {
    std::vector<decode_segment_range> out;
    const int total_context_hi = input_len + output_len;
    int context_lo = input_len + 1;
    int profile_hi = ((input_len + bucket_size - 1) / bucket_size + 1) * bucket_size;
    int segment_id = 0;
    while (context_lo <= total_context_hi) {
        const int profile_lo = profile_hi - bucket_size + 1;
        const int context_hi = std::min(total_context_hi, profile_hi);
        out.push_back({
            segment_id,
            context_lo,
            context_hi,
            profile_lo,
            profile_hi,
            context_hi - context_lo + 1,
        });
        context_lo = context_hi + 1;
        profile_hi += bucket_size;
        ++segment_id;
    }
    return out;
}

std::pair<double, std::string> prefill_match(const profile_record & record, int input_len, const std::string & mode) {
    if (record.bucket_lo <= input_len && record.bucket_hi >= input_len) {
        return {0.0, record.bucket_lo == input_len && record.bucket_hi == input_len ? "exact" : "covering_length"};
    }
    const int length = record.bucket_hi;
    if (length == input_len) {
        return {0.0, "exact"};
    }
    if (mode == "floor" && length <= input_len) {
        return {static_cast<double>(std::abs(length - input_len)), "floor"};
    }
    if (mode == "ceil" && length >= input_len) {
        return {static_cast<double>(std::abs(length - input_len)), "ceil"};
    }
    if (mode == "nearest") {
        return {static_cast<double>(std::abs(length - input_len)), "nearest"};
    }
    return {std::numeric_limits<double>::infinity(), ""};
}

std::string cpu_route_spec_for_state(const std::string & state_name, const std::string & state_group) {
    const std::string suffix = remove_prefix(state_name, "cpu_");
    const std::string text = suffix + " " + state_group;
    const std::vector<int> numbers = extract_numbers(text);
    std::vector<int> freqs;
    for (int n : numbers) {
        if (n >= 100000) {
            freqs.push_back(n);
        }
    }
    std::string group = state_group.empty() ? suffix.substr(0, suffix.find('_')) : state_group;
    group = to_upper(group);
    if (group.find("B2S4") != std::string::npos && freqs.size() >= 2) {
        return "cpu{threads=6,affinity=FC,cpu_policy0_freq_khz=" + std::to_string(freqs[1]) +
               ",cpu_policy6_freq_khz=" + std::to_string(freqs[0]) + "}";
    }
    if (group.find("B2S2") != std::string::npos && freqs.size() >= 2) {
        return "cpu{threads=4,affinity=CC,cpu_policy0_freq_khz=" + std::to_string(freqs[1]) +
               ",cpu_policy6_freq_khz=" + std::to_string(freqs[0]) + "}";
    }
    if (group.find("S6") != std::string::npos && !freqs.empty()) {
        return "cpu{threads=6,affinity=3F,cpu_policy0_freq_khz=" + std::to_string(freqs.back()) + "}";
    }
    if (group.find("B2") != std::string::npos && !freqs.empty()) {
        return "cpu{threads=2,affinity=C0,cpu_policy6_freq_khz=" + std::to_string(freqs[0]) + "}";
    }
    if (group.find("B1") != std::string::npos && !freqs.empty()) {
        return "cpu{threads=1,affinity=40,cpu_policy6_freq_khz=" + std::to_string(freqs[0]) + "}";
    }
    return "cpu";
}

std::string npu_workpoint_for_state(const std::string & state_name, const std::string & state_group) {
    const std::string text = state_name + " " + state_group;
    const std::string lowered = to_lower(text);
    const std::vector<std::string> candidates = {
        "low_balanced", "high_performance", "high_power_saver", "low_power_saver",
        "extreme_power_saver", "power_saver", "balanced", "burst", "native", "low",
    };
    for (const auto & candidate : candidates) {
        if (lowered.find(candidate) != std::string::npos) {
            return candidate;
        }
    }

    std::string workpoint = remove_prefix(state_name, "npu_");
    if (workpoint == state_name) {
        workpoint = remove_prefix(state_name, "qnn_npu_");
    }
    return workpoint.empty() ? "burst" : workpoint;
}

const qnn_graph_capacity * qnn_graph_capacity_for_context_size(uint64_t context_size) {
    for (const auto & capacity : QNN_GRAPH_CAPACITIES) {
        if (capacity.context_size == context_size) {
            return &capacity;
        }
    }
    return nullptr;
}

const qnn_graph_capacity * qnn_graph_capacity_for_required(uint64_t required_context) {
    if (required_context == 0) {
        return nullptr;
    }
    for (const auto & capacity : QNN_GRAPH_CAPACITIES) {
        if (required_context <= capacity.usable_kv_slots) {
            return &capacity;
        }
    }
    return nullptr;
}

uint64_t qnn_context_size_for_required(uint64_t required_context) {
    const qnn_graph_capacity * capacity = qnn_graph_capacity_for_required(required_context);
    if (capacity != nullptr) {
        return capacity->context_size;
    }
    return QNN_GRAPH_CAPACITIES[sizeof(QNN_GRAPH_CAPACITIES) / sizeof(QNN_GRAPH_CAPACITIES[0]) - 1].context_size;
}

uint64_t qnn_route_required_kv_slots_for_required(uint64_t required_context) {
    const qnn_graph_capacity * capacity = qnn_graph_capacity_for_required(required_context);
    return capacity != nullptr ? capacity->usable_kv_slots : required_context;
}

uint64_t qnn_context_size_from_state(const std::string & state_name, const std::string & state_group) {
    const std::string text = to_lower(state_name + " " + state_group);
    static const std::regex cap_re("(?:cap|ctx|context)(\\d+)");
    std::smatch match;
    if (std::regex_search(text, match, cap_re)) {
        const uint64_t context_size = static_cast<uint64_t>(std::strtoull(match[1].str().c_str(), nullptr, 10));
        if (qnn_graph_capacity_for_context_size(context_size) != nullptr) {
            return context_size;
        }
    }
    return 0;
}

std::string npu_route_spec_for_workpoint(
        const std::string & workpoint,
        uint64_t            required_context,
        uint64_t            explicit_context_size = 0) {
    const qnn_graph_capacity * explicit_capacity =
        explicit_context_size != 0 ? qnn_graph_capacity_for_context_size(explicit_context_size) : nullptr;
    const qnn_graph_capacity * capacity =
        explicit_capacity != nullptr ? explicit_capacity : qnn_graph_capacity_for_required(required_context);
    if (required_context == 0 && capacity == nullptr) {
        return "qnn-npu{workpoint=" + workpoint + "}";
    }
    const uint64_t context_size = capacity != nullptr ? capacity->context_size : qnn_context_size_for_required(required_context);
    const uint64_t route_required_kv_slots =
        capacity != nullptr ? capacity->usable_kv_slots : qnn_route_required_kv_slots_for_required(required_context);
    return "qnn-npu{workpoint=" + workpoint +
           ",qnn_context_size=" + std::to_string(context_size) +
           ",qnn_required_kv_slots=" + std::to_string(route_required_kv_slots) + "}";
}

std::string route_spec_for_state(
        const std::string & backend_in,
        const std::string & state_name,
        const std::string & state_group,
        uint64_t required_context = 0) {
    const std::string backend = normalize_backend(backend_in);
    const std::string text = state_name + " " + state_group;
    if (backend == "GPU") {
        const std::vector<int> numbers = extract_numbers(text);
        if (numbers.empty()) {
            return "opencl";
        }
        const int freq_mhz = numbers.back();
        return "opencl{gpu_freq_hz=" + std::to_string(static_cast<int64_t>(freq_mhz) * 1000000LL) + "}";
    }
    if (backend == "NPU") {
        return npu_route_spec_for_workpoint(
                npu_workpoint_for_state(state_name, state_group),
                required_context,
                qnn_context_size_from_state(state_name, state_group));
    }
    if (backend == "CPU") {
        return cpu_route_spec_for_state(state_name, state_group);
    }
    return state_name;
}

std::string route_spec_for_state(const planner_state_meta & state, uint64_t required_context = 0) {
    if (state.backend == "NPU") {
        return npu_route_spec_for_workpoint(
                state.npu_workpoint,
                required_context,
                state.qnn_explicit_context_size);
    }
    if (!state.static_route_spec.empty()) {
        return state.static_route_spec;
    }
    return route_spec_for_state(state.backend, state.state_name, state.state_group, required_context);
}

std::string state_alias_key(const std::string & backend, const std::string & alias) {
    return backend + "\t" + alias;
}

std::string bucket_key(int lo, int hi) {
    return std::to_string(lo) + ":" + std::to_string(hi);
}

int find_state_id(const decode_profile_index & index, const std::string & backend, const std::string & state) {
    const auto aliases = state_aliases(backend, state);
    for (const auto & alias : aliases) {
        const auto it = index.state_id_by_backend_alias.find(state_alias_key(normalize_backend(backend), alias));
        if (it != index.state_id_by_backend_alias.end()) {
            return it->second;
        }
    }
    return -1;
}

bool indexed_qnn_record_matches_query_capacity(
        const planner_state_meta & state,
        const profile_record &     record,
        int                        profile_query_hi) {
    if (state.backend != "NPU") {
        return true;
    }

    const qnn_graph_capacity * query_capacity =
        qnn_graph_capacity_for_required(static_cast<uint64_t>(profile_query_hi));
    if (query_capacity == nullptr) {
        return false;
    }

    if (state.qnn_explicit_context_size != 0) {
        const qnn_graph_capacity * state_capacity =
            qnn_graph_capacity_for_context_size(state.qnn_explicit_context_size);
        return state_capacity != nullptr &&
               static_cast<uint64_t>(profile_query_hi) <= state_capacity->usable_kv_slots;
    }

    const qnn_graph_capacity * record_capacity =
        qnn_graph_capacity_for_required(static_cast<uint64_t>(record.bucket_hi));
    return record_capacity != nullptr &&
           record_capacity->context_size == query_capacity->context_size;
}

decode_profile_index build_decode_profile_index(const std::vector<profile_record> & records) {
    decode_profile_index index;
    std::map<std::string, const profile_record *> first_by_state;
    std::map<std::string, int> backend_id_by_name;

    for (const auto & record : records) {
        if (record.phase != "decode") {
            continue;
        }
        first_by_state.emplace(record.state_name, &record);
        if (!backend_id_by_name.count(record.backend)) {
            const int backend_id = static_cast<int>(backend_id_by_name.size());
            backend_id_by_name[record.backend] = backend_id;
            index.backend_names.push_back(record.backend);
        }
    }

    index.states_by_backend.resize(index.backend_names.size());
    index.states.reserve(first_by_state.size());
    for (const auto & item : first_by_state) {
        const profile_record & record = *item.second;
        planner_state_meta state;
        state.id = static_cast<int>(index.states.size());
        state.backend = record.backend;
        state.backend_id = backend_id_by_name[record.backend];
        state.state_name = record.state_name;
        state.state_group = record.state_group;
        state.aliases = state_aliases(record.backend, record.state_name);
        state.qnn_explicit_context_size = qnn_context_size_from_state(record.state_name, record.state_group);
        state.npu_workpoint = npu_workpoint_for_state(record.state_name, record.state_group);
        state.static_route_spec = route_spec_for_state(record.backend, record.state_name, record.state_group, 0);
        for (const auto & alias : state.aliases) {
            index.state_id_by_backend_alias[state_alias_key(state.backend, alias)] = state.id;
        }
        index.states_by_backend[state.backend_id].push_back(state.id);
        index.states.push_back(std::move(state));
    }

    index.records_by_state.resize(index.states.size());
    int row_index = 0;
    for (const auto & record : records) {
        if (record.phase != "decode") {
            ++row_index;
            continue;
        }
        const int state_id = find_state_id(index, record.backend, record.state_name);
        if (state_id >= 0) {
            index.records_by_state[state_id].push_back({
                &record,
                row_index,
                bucket_mid(record.bucket_lo, record.bucket_hi),
            });
        }
        ++row_index;
    }

    index.records_by_state_mid.resize(index.states.size());
    index.exact_bucket_by_state.resize(index.states.size());
    for (size_t state_id = 0; state_id < index.records_by_state.size(); ++state_id) {
        auto & records_for_state = index.records_by_state[state_id];
        for (const auto & indexed_record : records_for_state) {
            const profile_record & record = *indexed_record.record;
            const std::string key = bucket_key(record.bucket_lo, record.bucket_hi);
            auto & exact = index.exact_bucket_by_state[state_id];
            if (!exact.count(key)) {
                exact[key] = &indexed_record;
            }
            index.records_by_state_mid[state_id].push_back(&indexed_record);
        }
        std::sort(
                index.records_by_state_mid[state_id].begin(),
                index.records_by_state_mid[state_id].end(),
                [](const indexed_profile_record * lhs, const indexed_profile_record * rhs) {
                    if (std::abs(lhs->mid - rhs->mid) > 1e-9) {
                        return lhs->mid < rhs->mid;
                    }
                    return lhs->row_index < rhs->row_index;
                });
    }

    const size_t backend_count = index.backend_names.size();
    index.backend_transition_costs.assign(
            backend_count,
            std::vector<transition_cost>(backend_count));
    for (size_t from = 0; from < backend_count; ++from) {
        for (size_t to = 0; to < backend_count; ++to) {
            const double latency_ms =
                default_backend_transition_latency_ms(index.backend_names[from], index.backend_names[to]);
            index.backend_transition_costs[from][to] = {
                latency_ms,
                0.0,
                latency_ms <= 0.0
                    ? (from == to ? "same_backend" : "default")
                    : "backend_default",
            };
        }
    }

    return index;
}

double default_backend_transition_latency_ms(const std::string & from_backend, const std::string & to_backend) {
    if (from_backend.empty() || to_backend.empty() || from_backend == to_backend) {
        return 0.0;
    }
    if (from_backend == "NPU" && to_backend == "CPU") return 5.0;
    if (from_backend == "GPU" && to_backend == "CPU") return 15.0;
    if (from_backend == "CPU" && to_backend == "GPU") return 20.0;
    if (from_backend == "NPU" && to_backend == "GPU") return 20.0;
    if (from_backend == "CPU" && to_backend == "NPU") return 80.0;
    if (from_backend == "GPU" && to_backend == "NPU") return 80.0;
    return 0.0;
}

bool better_indexed_match(
        const indexed_profile_record * candidate,
        double                         distance,
        const indexed_profile_record * current,
        double                         current_distance) {
    if (candidate == nullptr) {
        return false;
    }
    if (current == nullptr) {
        return true;
    }
    if (std::abs(distance - current_distance) > 1e-9) {
        return distance < current_distance;
    }
    return candidate->row_index < current->row_index;
}

const indexed_profile_record * best_record_for_state_scan(
        const decode_profile_index & index,
        int                          state_id,
        int                          profile_query_lo,
        int                          profile_query_hi,
        const std::string &          context_match,
        std::string &                match_kind) {
    const planner_state_meta & state = index.states[state_id];
    const indexed_profile_record * best = nullptr;
    double best_distance = std::numeric_limits<double>::infinity();
    std::string best_kind;
    for (const auto & indexed_record : index.records_by_state[state_id]) {
        const profile_record & record = *indexed_record.record;
        if (!indexed_qnn_record_matches_query_capacity(state, record, profile_query_hi)) {
            continue;
        }
        const auto match = bucket_match(record, profile_query_lo, profile_query_hi, context_match);
        if (!std::isfinite(match.first)) {
            continue;
        }
        if (better_indexed_match(&indexed_record, match.first, best, best_distance)) {
            best = &indexed_record;
            best_distance = match.first;
            best_kind = match.second;
        }
    }
    match_kind = best_kind;
    return best;
}

const indexed_profile_record * best_record_for_state_indexed(
        const decode_profile_index & index,
        int                          state_id,
        int                          profile_query_lo,
        int                          profile_query_hi,
        const std::string &          context_match,
        std::string &                match_kind) {
    const planner_state_meta & state = index.states[state_id];
    const std::string exact_key = bucket_key(profile_query_lo, profile_query_hi);
    const auto exact_it = index.exact_bucket_by_state[state_id].find(exact_key);
    if (exact_it != index.exact_bucket_by_state[state_id].end()) {
        const indexed_profile_record * exact = exact_it->second;
        if (indexed_qnn_record_matches_query_capacity(state, *exact->record, profile_query_hi)) {
            match_kind = "exact";
            return exact;
        }
    }

    if (context_match == "exact" || context_match == "floor" || context_match == "ceil") {
        return best_record_for_state_scan(
                index,
                state_id,
                profile_query_lo,
                profile_query_hi,
                context_match,
                match_kind);
    }

    const double query_mid = bucket_mid(profile_query_lo, profile_query_hi);
    const auto & by_mid = index.records_by_state_mid[state_id];
    const auto lower = std::lower_bound(
            by_mid.begin(),
            by_mid.end(),
            query_mid,
            [](const indexed_profile_record * lhs, double rhs) {
                return lhs->mid < rhs;
            });

    const indexed_profile_record * best = nullptr;
    double best_distance = std::numeric_limits<double>::infinity();
    std::string best_kind;

    auto consider = [&](const indexed_profile_record * indexed_record) {
        const profile_record & record = *indexed_record->record;
        if (!indexed_qnn_record_matches_query_capacity(state, record, profile_query_hi)) {
            return;
        }
        const auto match = bucket_match(record, profile_query_lo, profile_query_hi, context_match);
        if (!std::isfinite(match.first)) {
            return;
        }
        if (better_indexed_match(indexed_record, match.first, best, best_distance)) {
            best = indexed_record;
            best_distance = match.first;
            best_kind = match.second;
        }
    };

    for (auto it = lower; it != by_mid.end(); ++it) {
        const double possible_distance = std::abs((*it)->mid - query_mid);
        if (best != nullptr && possible_distance > best_distance + 1e-9) {
            break;
        }
        consider(*it);
    }
    for (auto it = lower; it != by_mid.begin();) {
        --it;
        const double possible_distance = std::abs((*it)->mid - query_mid);
        if (best != nullptr && possible_distance > best_distance + 1e-9) {
            break;
        }
        consider(*it);
    }

    if (best == nullptr) {
        return best_record_for_state_scan(
                index,
                state_id,
                profile_query_lo,
                profile_query_hi,
                context_match,
                match_kind);
    }
    match_kind = best_kind;
    return best;
}

transition_cost lookup_transition(
        const decode_profile_index & index,
        int                          from_state_id,
        const segment_candidate &    to_candidate) {
    if (from_state_id < 0) {
        return {0.0, 0.0, "default"};
    }
    if (from_state_id == to_candidate.state_id) {
        return {0.0, 0.0, "same_state"};
    }
    const planner_state_meta & from_state = index.states[from_state_id];
    const planner_state_meta & to_state = index.states[to_candidate.state_id];
    transition_cost transition = index.backend_transition_costs[from_state.backend_id][to_state.backend_id];
    if (transition.latency_ms <= 0.0) {
        transition.energy_mj = 0.0;
        transition.source = from_state.backend_id == to_state.backend_id ? "same_backend" : "default";
        return transition;
    }
    transition.energy_mj = to_candidate.record->power_mw * transition.latency_ms / 1000.0;
    return transition;
}

std::vector<int> best_layer_indices_by_backend(
        const std::vector<dp_node> &  layer,
        const decode_profile_index &  index) {
    std::vector<int> best(index.backend_names.size(), -1);
    for (size_t i = 0; i < layer.size(); ++i) {
        const dp_node & node = layer[i];
        if (!node.valid || node.candidate.state_id < 0) {
            continue;
        }
        const int backend_id = index.states[node.candidate.state_id].backend_id;
        const int current = best[backend_id];
        if (current < 0 || better_dp_node(node, layer[current])) {
            best[backend_id] = static_cast<int>(i);
        }
    }
    std::vector<int> out;
    out.reserve(best.size());
    for (int idx : best) {
        if (idx >= 0) {
            out.push_back(idx);
        }
    }
    return out;
}

bool better_dp_node(const dp_node & lhs, const dp_node & rhs) {
    if (!rhs.valid) return true;
    if (lhs.slo_violations != rhs.slo_violations) return lhs.slo_violations < rhs.slo_violations;
    if (std::abs(lhs.total_slo_miss_ms - rhs.total_slo_miss_ms) > 1e-9) return lhs.total_slo_miss_ms < rhs.total_slo_miss_ms;
    if (std::abs(lhs.total_energy_mj - rhs.total_energy_mj) > 1e-9) return lhs.total_energy_mj < rhs.total_energy_mj;
    return lhs.total_latency_ms < rhs.total_latency_ms;
}

dp_node make_node(const dp_node * prev, int prev_state_index, const std::string & prev_state, const segment_candidate & candidate, const transition_cost & transition, double tbt_slo_ms) {
    dp_node node;
    node.valid = true;
    node.prev_state_index = prev_state_index;
    node.candidate = candidate;
    node.transition = transition;
    node.step_latency_ms = candidate.latency_ms() + transition.latency_ms;
    node.step_energy_mj = candidate.energy_mj() + transition.energy_mj;
    node.step_slo_deadline_ms = tbt_slo_ms * static_cast<double>(candidate.tokens);
    node.step_slo_miss_ms = std::max(0.0, node.step_latency_ms - node.step_slo_deadline_ms);
    node.step_slo_ok = node.step_slo_miss_ms <= 1e-9;
    node.total_energy_mj = (prev ? prev->total_energy_mj : 0.0) + node.step_energy_mj;
    node.total_latency_ms = (prev ? prev->total_latency_ms : 0.0) + node.step_latency_ms;
    node.slo_violations = (prev ? prev->slo_violations : 0) + (node.step_slo_ok ? 0 : 1);
    node.total_slo_miss_ms = (prev ? prev->total_slo_miss_ms : 0.0) + node.step_slo_miss_ms;
    (void) prev_state;
    return node;
}

std::vector<segment_candidate> candidates_for_segment(
        const decode_profile_index & index,
        int segment_id,
        int context_lo,
        int context_hi,
        int profile_query_lo,
        int profile_query_hi,
        int tokens,
        const std::string & context_match) {
    std::vector<segment_candidate> out;
    out.reserve(index.states.size());
    for (const auto & state : index.states) {
        std::string match_kind;
        const indexed_profile_record * indexed_record = best_record_for_state_indexed(
                index,
                state.id,
                profile_query_lo,
                profile_query_hi,
                context_match,
                match_kind);
        if (indexed_record == nullptr) {
            continue;
        }
        out.push_back({
            segment_id,
            context_lo,
            context_hi,
            profile_query_lo,
            profile_query_hi,
            tokens,
            state.id,
            indexed_record->record,
            match_kind,
        });
    }
    return out;
}

decode_plan plan_decode(const std::vector<profile_record> & records, const options & args) {
    decode_plan plan;
    if (args.output_len <= 0) {
        return plan;
    }

    const decode_profile_index profile_index = build_decode_profile_index(records);
    std::vector<std::vector<segment_candidate>> segments;
    for (const auto & range : aligned_decode_segments(args.input_len, args.output_len, args.bucket_size)) {
        auto candidates = candidates_for_segment(
                profile_index,
                range.segment_id,
                range.context_lo,
                range.context_hi,
                range.profile_query_lo,
                range.profile_query_hi,
                range.tokens,
                args.context_match);
        if (candidates.empty()) {
            std::ostringstream oss;
            oss << "no decode profile candidate for segment " << range.segment_id
                << " context_bucket=" << range.context_lo << "-" << range.context_hi
                << " profile_query_bucket=" << range.profile_query_lo << "-" << range.profile_query_hi;
            throw std::runtime_error(oss.str());
        }
        segments.push_back(std::move(candidates));
    }

    std::vector<std::vector<dp_node>> dp;
    dp.reserve(segments.size());
    const int initial_state_id = args.initial_decode_state.empty()
        ? -1
        : [&]() {
            for (const auto & state : profile_index.states) {
                if (state_matches(state.backend, state.state_name, args.initial_decode_state)) {
                    return state.id;
                }
            }
            return -1;
        }();
    for (size_t segment_index = 0; segment_index < segments.size(); ++segment_index) {
        std::vector<dp_node> layer(segments[segment_index].size());
        const std::vector<int> prev_frontier =
            segment_index == 0
                ? std::vector<int>{}
                : best_layer_indices_by_backend(dp[segment_index - 1], profile_index);
        for (size_t cand_idx = 0; cand_idx < segments[segment_index].size(); ++cand_idx) {
            const auto & candidate = segments[segment_index][cand_idx];
            if (segment_index == 0) {
                const transition_cost transition = args.initial_decode_state.empty()
                    ? transition_cost{0.0, 0.0, "initial"}
                    : lookup_transition(
                            profile_index,
                            initial_state_id,
                            candidate);
                layer[cand_idx] = make_node(nullptr, -1, args.initial_decode_state, candidate, transition, args.tbt_slo_ms);
                continue;
            }

            dp_node best;
            for (int prev_idx : prev_frontier) {
                const auto & prev = dp[segment_index - 1][prev_idx];
                const std::string prev_state = prev.candidate.record->state_name;
                const transition_cost transition = lookup_transition(
                        profile_index,
                        prev.candidate.state_id,
                        candidate);
                dp_node node = make_node(&prev, static_cast<int>(prev_idx), prev_state, candidate, transition, args.tbt_slo_ms);
                if (better_dp_node(node, best)) {
                    best = node;
                }
            }
            layer[cand_idx] = best;
        }
        dp.push_back(std::move(layer));
    }

    int best_idx = -1;
    for (size_t i = 0; i < dp.back().size(); ++i) {
        if (best_idx < 0 || better_dp_node(dp.back()[i], dp.back()[best_idx])) {
            best_idx = static_cast<int>(i);
        }
    }
    if (best_idx < 0) {
        throw std::runtime_error("DP failed to produce a decode plan");
    }

    std::vector<dp_node> chosen;
    for (int layer = static_cast<int>(dp.size()) - 1, idx = best_idx; layer >= 0; --layer) {
        const dp_node node = dp[layer][idx];
        chosen.push_back(node);
        idx = node.prev_state_index;
    }
    std::reverse(chosen.begin(), chosen.end());
    plan.nodes = chosen;

    std::string previous_spec;
    int generated_before_segment = 0;
    for (const auto & node : plan.nodes) {
        const auto & state = profile_index.states[node.candidate.state_id];
        const uint64_t required_context =
            state.backend == "NPU" ? (uint64_t) node.candidate.profile_query_hi : 0;
        const std::string spec = route_spec_for_state(state, required_context);
        if (spec != previous_spec) {
            if (!plan.schedule.empty()) {
                plan.schedule += ";";
            }
            const int start_token = generated_before_segment + 1;
            plan.schedule += std::to_string(start_token) + ":" + spec;
            previous_spec = spec;
        }
        plan.latency_ms += node.step_latency_ms;
        plan.energy_mj += node.step_energy_mj;
        plan.slo_satisfied_steps += node.step_slo_ok ? 1 : 0;
        plan.slo_total_steps += 1;
        generated_before_segment += node.candidate.tokens;
    }
    return plan;
}

prefill_plan plan_prefill(const std::vector<profile_record> & records, const options & args) {
    prefill_plan fallback;
    if (args.input_len == 0) {
        fallback.slo_ok = true;
        fallback.latency_ms = 0.0;
        fallback.energy_mj = 0.0;
        return fallback;
    }
    fallback.qnn_required_kv_slots = (uint64_t) args.input_len;
    const qnn_graph_capacity * fallback_capacity = qnn_graph_capacity_for_required(fallback.qnn_required_kv_slots);
    if (fallback_capacity == nullptr) {
        throw std::runtime_error("no NPU prefill graph capacity for input_len=" + std::to_string(args.input_len));
    }
    fallback.qnn_context_size = fallback_capacity->context_size;
    fallback.qnn_required_kv_slots = fallback_capacity->usable_kv_slots;
    fallback.route_spec = npu_route_spec_for_workpoint(fallback.qnn_workpoint, (uint64_t) args.input_len);
    if (args.prefill_profile.empty()) {
        return fallback;
    }

    const profile_record * best = nullptr;
    double best_miss_ms = std::numeric_limits<double>::infinity();
    double best_energy_mj = std::numeric_limits<double>::infinity();
    bool best_slo_ok = false;

    for (const auto & record : records) {
        if (record.phase != "prefill" || record.backend != "NPU") {
            continue;
        }
        const auto match = prefill_match(record, args.input_len, args.context_match);
        if (!std::isfinite(match.first)) {
            continue;
        }
        const double latency_ms = record.latency_ms_for_tokens(args.input_len);
        const double energy_mj = record.energy_mj_for_tokens(args.input_len);
        const bool slo_ok = latency_ms <= args.ttft_slo_ms + 1e-9;
        const double miss_ms = std::max(0.0, latency_ms - args.ttft_slo_ms);

        bool better = false;
        if (!best) {
            better = true;
        } else if (slo_ok != best_slo_ok) {
            better = slo_ok;
        } else if (slo_ok) {
            better = energy_mj < best_energy_mj;
        } else if (std::abs(miss_ms - best_miss_ms) > 1e-9) {
            better = miss_ms < best_miss_ms;
        } else {
            better = energy_mj < best_energy_mj;
        }

        if (better) {
            best = &record;
            best_miss_ms = miss_ms;
            best_energy_mj = energy_mj;
            best_slo_ok = slo_ok;
        }
    }

    if (!best) {
        throw std::runtime_error("no NPU prefill profile candidate for input_len=" + std::to_string(args.input_len));
    }

    prefill_plan out;
    out.state_name = best->state_name;
    out.backend = best->backend;
    out.state_group = best->state_group;
    out.qnn_workpoint = npu_workpoint_for_state(best->state_name, best->state_group);
    const qnn_graph_capacity * prefill_capacity = qnn_graph_capacity_for_required((uint64_t) args.input_len);
    if (prefill_capacity == nullptr) {
        throw std::runtime_error("no NPU prefill graph capacity for input_len=" + std::to_string(args.input_len));
    }
    out.qnn_required_kv_slots = prefill_capacity->usable_kv_slots;
    out.qnn_context_size = prefill_capacity->context_size;
    out.route_spec = route_spec_for_state(
            best->backend,
            best->state_name,
            best->state_group,
            (uint64_t) args.input_len);
    out.from_profile = true;
    out.slo_ok = best_slo_ok;
    out.latency_ms = best->latency_ms_for_tokens(args.input_len);
    out.energy_mj = best->energy_mj_for_tokens(args.input_len);
    return out;
}

std::string shell_quote(const std::string & value) {
    std::string out = "'";
    for (char ch : value) {
        if (ch == '\'') {
            out += "'\\''";
        } else {
            out.push_back(ch);
        }
    }
    out += "'";
    return out;
}

std::string route_spec_without_state_suffix(const std::string & route_spec) {
    std::string spec = trim(route_spec);
    if (spec.empty() || spec.back() != '}') {
        return spec;
    }
    const size_t open = spec.rfind('{');
    if (open == std::string::npos) {
        return spec;
    }
    return trim(spec.substr(0, open));
}

void print_export(const std::string & name, const std::string & value) {
    std::cout << "export " << name << "=" << shell_quote(value) << "\n";
}

void print_unset(const std::string & name) {
    std::cout << "unset " << name << "\n";
}

void print_env_output(const prefill_plan & prefill, const decode_plan & decode) {
    const std::string prefill_schedule = "1:" + prefill.route_spec;
    const std::string prefill_route = route_spec_without_state_suffix(prefill.route_spec);

    print_export("GGML_HETERO_DYNAMIC_MODE", "phase");
    print_export("GGML_HETERO_DYNAMIC_PREFILL_ROUTE", prefill_route);
    print_export("GGML_HETERO_DYNAMIC_PREFILL_SCHEDULE", prefill_schedule);
    if (normalize_backend(prefill.backend) == "NPU") {
        const std::string workpoint = npu_workpoint_for_state(prefill.state_name, prefill.state_group);
        print_export("GGML_HETERO_DYNAMIC_PREFILL_QNN_WORKPOINT", workpoint);
        print_export("GGML_QNN_HTP_WORKPOINT", workpoint);
    } else {
        print_unset("GGML_HETERO_DYNAMIC_PREFILL_QNN_WORKPOINT");
        print_unset("GGML_QNN_HTP_WORKPOINT");
    }

    print_unset("GGML_HETERO_DYNAMIC_DECODE_ROUTE");
    print_unset("GGML_HETERO_DYNAMIC_DECODE_GPU_FREQ_HZ");
    print_unset("GGML_HETERO_DECODE_GPU_FREQ_HZ");
    print_unset("GGML_HETERO_DYNAMIC_DECODE_CPU_FREQ_KHZ");
    print_unset("GGML_HETERO_DECODE_CPU_FREQ_KHZ");
    print_unset("GGML_HETERO_DYNAMIC_DECODE_CPU_AFFINITY_MASK");
    print_unset("GGML_HETERO_DECODE_CPU_AFFINITY_MASK");
    print_unset("GGML_HETERO_DYNAMIC_DECODE_CPU_THREADS");
    print_unset("GGML_HETERO_DECODE_CPU_THREADS");
    print_unset("GGML_HETERO_DYNAMIC_DECODE_QNN_WORKPOINT");
    print_export("GGML_HETERO_DYNAMIC_DECODE_SCHEDULE", decode.schedule);
    print_export("GGML_HETERO_DECODE_ROUTE_SCHEDULE", decode.schedule);

    if (decode.schedule.find("qnn-npu") != std::string::npos) {
        print_export("GGML_HETERO_DYNAMIC_PRELOAD_QNN_DECODE", "1");
    } else {
        print_unset("GGML_HETERO_DYNAMIC_PRELOAD_QNN_DECODE");
    }
}

double elapsed_ms_since(std::chrono::steady_clock::time_point start) {
    const auto elapsed = std::chrono::steady_clock::now() - start;
    return std::chrono::duration<double, std::milli>(elapsed).count();
}

void print_summary(
        std::ostream &       os,
        const prefill_plan & prefill,
        const decode_plan &  decode,
        double               planner_elapsed_ms = -1.0) {
    os << std::fixed << std::setprecision(3);
    if (planner_elapsed_ms >= 0.0) {
        os << "planner_elapsed_ms=" << planner_elapsed_ms << "\n";
    }
    os << "prefill_state=" << prefill.state_name
              << " prefill_schedule=1:" << prefill.route_spec
              << " source=" << (prefill.from_profile ? "profile" : "default")
              << " ttft_slo_ok=" << (prefill.slo_ok ? "true" : "unknown");
    if (std::isfinite(prefill.latency_ms)) {
        os << " latency_ms=" << prefill.latency_ms << " energy_mj=" << prefill.energy_mj;
    }
    os << "\n";
    os << "decode_schedule=" << decode.schedule << "\n";
    os << "decode_latency_ms=" << decode.latency_ms
              << " decode_energy_mj=" << decode.energy_mj
              << " slo_satisfaction_rate="
              << (decode.slo_total_steps ? static_cast<double>(decode.slo_satisfied_steps) / decode.slo_total_steps : 1.0)
              << " (" << decode.slo_satisfied_steps << "/" << decode.slo_total_steps << ")\n";
}

[[noreturn]] void usage(const char * argv0, int code) {
    std::ostream & os = code == 0 ? std::cout : std::cerr;
    os << "Usage: " << argv0 << " --decode-profile PATH --input-len N --output-len N "
       << "--ttft-slo-ms MS --tbt-slo-ms MS [options]\n\n"
       << "Options:\n"
       << "  --prefill-profile PATH     Optional CSV profile for prefill. If omitted, use NPU burst.\n"
       << "  --bucket-size N            Decode bucket size. Default: 32.\n"
       << "  --context-match MODE       exact|floor|ceil|nearest. Default: exact.\n"
       << "  --initial-decode-state S   Optional state used for first decode transition cost.\n"
       << "  --output-format FORMAT     env|summary. Default: env.\n"
       << "  -h, --help                 Show this help.\n";
    std::exit(code);
}

options parse_args(int argc, char ** argv) {
    options args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto need_value = [&](const std::string & name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + name);
            }
            return argv[++i];
        };
        if (key == "-h" || key == "--help") {
            usage(argv[0], 0);
        } else if (key == "--prefill-profile") {
            args.prefill_profile = need_value(key);
        } else if (key == "--decode-profile") {
            args.decode_profile = need_value(key);
        } else if (key == "--ttft-slo-ms") {
            args.ttft_slo_ms = parse_double(need_value(key), -1.0);
        } else if (key == "--tbt-slo-ms") {
            args.tbt_slo_ms = parse_double(need_value(key), -1.0);
        } else if (key == "--input-len") {
            args.input_len = parse_int(need_value(key), -1);
        } else if (key == "--output-len") {
            args.output_len = parse_int(need_value(key), -1);
        } else if (key == "--bucket-size") {
            args.bucket_size = parse_int(need_value(key), 32);
        } else if (key == "--context-match") {
            args.context_match = need_value(key);
        } else if (key == "--initial-decode-state") {
            args.initial_decode_state = need_value(key);
        } else if (key == "--output-format") {
            args.output_format = need_value(key);
        } else {
            throw std::runtime_error("unknown argument: " + key);
        }
    }

    if (args.decode_profile.empty()) throw std::runtime_error("--decode-profile is required");
    if (args.input_len < 0) throw std::runtime_error("--input-len must be >= 0");
    if (args.output_len < 0) throw std::runtime_error("--output-len must be >= 0");
    if (args.ttft_slo_ms < 0.0) throw std::runtime_error("--ttft-slo-ms must be >= 0");
    if (args.tbt_slo_ms <= 0.0) throw std::runtime_error("--tbt-slo-ms must be > 0");
    if (args.bucket_size <= 0) throw std::runtime_error("--bucket-size must be > 0");
    if (args.context_match != "exact" && args.context_match != "floor" && args.context_match != "ceil" && args.context_match != "nearest") {
        throw std::runtime_error("--context-match must be exact, floor, ceil, or nearest");
    }
    if (args.output_format != "env" && args.output_format != "summary") {
        throw std::runtime_error("--output-format must be env or summary");
    }
    return args;
}

#ifdef SYSTEM_BENEFIT_PLANNER_NO_MAIN
} // namespace system_benefit_planner
#else
} // namespace
#endif

#ifndef SYSTEM_BENEFIT_PLANNER_NO_MAIN
int main(int argc, char ** argv) {
    try {
        const options args = parse_args(argc, argv);
        std::vector<profile_record> prefill_records;
        if (!args.prefill_profile.empty()) {
            prefill_records = load_profile_csv(args.prefill_profile);
        }
        std::vector<profile_record> decode_records = load_profile_csv(args.decode_profile);

        const auto planner_start = std::chrono::steady_clock::now();
        const prefill_plan prefill = plan_prefill(prefill_records, args);
        const decode_plan decode = plan_decode(decode_records, args);
        const double planner_elapsed_ms = elapsed_ms_since(planner_start);

        if (args.output_format == "summary") {
            print_summary(std::cout, prefill, decode, planner_elapsed_ms);
        } else {
            print_env_output(prefill, decode);
            print_summary(std::cerr, prefill, decode, planner_elapsed_ms);
        }
        return 0;
    } catch (const std::exception & err) {
        std::cerr << "error: " << err.what() << "\n";
        return 1;
    }
}
#endif
