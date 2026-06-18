#include <algorithm>
#include <cctype>
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

namespace {

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
    int tokens = 0;
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

std::string route_spec_for_state(const std::string & backend_in, const std::string & state_name, const std::string & state_group) {
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
        const std::string lowered = to_lower(text);
        std::string workpoint = remove_prefix(state_name, "npu_");
        const std::vector<std::string> candidates = {
            "low_balanced", "balanced", "burst", "low_power_saver", "power_saver", "low", "native",
        };
        for (const auto & candidate : candidates) {
            if (lowered.find(candidate) != std::string::npos) {
                workpoint = candidate;
                break;
            }
        }
        return "qnn-npu{workpoint=" + workpoint + "}";
    }
    if (backend == "CPU") {
        return cpu_route_spec_for_state(state_name, state_group);
    }
    return state_name;
}

double default_backend_transition_latency_ms(const std::string & from_backend, const std::string & to_backend) {
    if (from_backend.empty() || to_backend.empty() || from_backend == to_backend) {
        return 0.0;
    }
    if (from_backend == "NPU" && to_backend == "CPU") return 5.0;
    if (from_backend == "GPU" && to_backend == "CPU") return 15.0;
    if (from_backend == "CPU" && to_backend == "GPU") return 50.0;
    if (from_backend == "NPU" && to_backend == "GPU") return 50.0;
    if (from_backend == "CPU" && to_backend == "NPU") return 50.0;
    if (from_backend == "GPU" && to_backend == "NPU") return 50.0;
    return 0.0;
}

const profile_record * find_record_for_state(const std::vector<profile_record> & records, const std::string & state) {
    for (const auto & record : records) {
        if (record.phase == "decode" && state_matches(record.backend, record.state_name, state)) {
            return &record;
        }
    }
    return nullptr;
}

transition_cost lookup_transition(const std::vector<profile_record> & records, const std::string & from_state, const profile_record & to_record) {
    if (from_state.empty() || state_matches(to_record.backend, to_record.state_name, from_state)) {
        return {0.0, 0.0, "same_state"};
    }
    const profile_record * from_record = find_record_for_state(records, from_state);
    const std::string from_backend = from_record ? from_record->backend : "";
    const std::string to_backend = to_record.backend;
    const double latency_ms = default_backend_transition_latency_ms(from_backend, to_backend);
    if (latency_ms <= 0.0) {
        return {0.0, 0.0, from_backend == to_backend ? "same_backend" : "default"};
    }
    return {latency_ms, to_record.power_mw * latency_ms / 1000.0, "backend_default"};
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
        const std::vector<profile_record> & records,
        int segment_id,
        int context_lo,
        int context_hi,
        int tokens,
        const std::string & context_match) {
    struct best_match {
        double distance = std::numeric_limits<double>::infinity();
        const profile_record * record = nullptr;
        std::string kind;
    };

    std::map<std::string, best_match> by_state;
    for (const auto & record : records) {
        if (record.phase != "decode") {
            continue;
        }
        const auto match = bucket_match(record, context_lo, context_hi, context_match);
        if (!std::isfinite(match.first)) {
            continue;
        }
        auto & best = by_state[record.state_name];
        if (match.first < best.distance) {
            best.distance = match.first;
            best.record = &record;
            best.kind = match.second;
        }
    }

    std::vector<segment_candidate> out;
    for (const auto & item : by_state) {
        if (!item.second.record) {
            continue;
        }
        out.push_back({segment_id, context_lo, context_hi, tokens, item.second.record, item.second.kind});
    }
    return out;
}

decode_plan plan_decode(const std::vector<profile_record> & records, const options & args) {
    decode_plan plan;
    if (args.output_len <= 0) {
        return plan;
    }

    std::vector<std::vector<segment_candidate>> segments;
    for (int generated = 0, segment_id = 0; generated < args.output_len; generated += args.bucket_size, ++segment_id) {
        const int tokens = std::min(args.bucket_size, args.output_len - generated);
        const int context_lo = args.input_len + generated + 1;
        const int context_hi = args.input_len + generated + tokens;
        auto candidates = candidates_for_segment(records, segment_id, context_lo, context_hi, tokens, args.context_match);
        if (candidates.empty()) {
            std::ostringstream oss;
            oss << "no decode profile candidate for segment " << segment_id << " context_bucket=" << context_lo << "-" << context_hi;
            throw std::runtime_error(oss.str());
        }
        segments.push_back(std::move(candidates));
    }

    std::vector<std::vector<dp_node>> dp;
    dp.reserve(segments.size());
    for (size_t index = 0; index < segments.size(); ++index) {
        std::vector<dp_node> layer(segments[index].size());
        for (size_t cand_idx = 0; cand_idx < segments[index].size(); ++cand_idx) {
            const auto & candidate = segments[index][cand_idx];
            if (index == 0) {
                const transition_cost transition = args.initial_decode_state.empty()
                    ? transition_cost{0.0, 0.0, "initial"}
                    : lookup_transition(records, args.initial_decode_state, *candidate.record);
                layer[cand_idx] = make_node(nullptr, -1, args.initial_decode_state, candidate, transition, args.tbt_slo_ms);
                continue;
            }

            dp_node best;
            for (size_t prev_idx = 0; prev_idx < dp[index - 1].size(); ++prev_idx) {
                const auto & prev = dp[index - 1][prev_idx];
                const std::string prev_state = prev.candidate.record->state_name;
                const transition_cost transition = lookup_transition(records, prev_state, *candidate.record);
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
    for (size_t i = 0; i < plan.nodes.size(); ++i) {
        const auto & node = plan.nodes[i];
        const auto & record = *node.candidate.record;
        const std::string spec = route_spec_for_state(record.backend, record.state_name, record.state_group);
        if (spec != previous_spec) {
            if (!plan.schedule.empty()) {
                plan.schedule += ";";
            }
            const int start_token = static_cast<int>(i) * args.bucket_size + 1;
            plan.schedule += std::to_string(start_token) + ":" + spec;
            previous_spec = spec;
        }
        plan.latency_ms += node.step_latency_ms;
        plan.energy_mj += node.step_energy_mj;
        plan.slo_satisfied_steps += node.step_slo_ok ? 1 : 0;
        plan.slo_total_steps += 1;
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
    out.route_spec = route_spec_for_state(best->backend, best->state_name, best->state_group);
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

void print_env_output(const prefill_plan & prefill, const decode_plan & decode) {
    const std::string prefill_schedule = "1:" + prefill.route_spec;
    std::cout << "GGML_HETERO_DYNAMIC_PREFILL_SCHEDULE=" << shell_quote(prefill_schedule) << "\n";
    std::cout << "GGML_HETERO_DYNAMIC_PREFILL_ROUTE=" << shell_quote(prefill.route_spec) << "\n";
    std::cout << "GGML_HETERO_DYNAMIC_DECODE_SCHEDULE=" << shell_quote(decode.schedule) << "\n";
    std::cout << "export GGML_HETERO_DYNAMIC_PREFILL_ROUTE=" << shell_quote(prefill.route_spec) << "\n";
    std::cout << "export GGML_HETERO_DYNAMIC_DECODE_SCHEDULE=" << shell_quote(decode.schedule) << "\n";
}

void print_summary(std::ostream & os, const prefill_plan & prefill, const decode_plan & decode) {
    os << std::fixed << std::setprecision(3);
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

} // namespace

int main(int argc, char ** argv) {
    try {
        const options args = parse_args(argc, argv);
        std::vector<profile_record> prefill_records;
        if (!args.prefill_profile.empty()) {
            prefill_records = load_profile_csv(args.prefill_profile);
        }
        std::vector<profile_record> decode_records = load_profile_csv(args.decode_profile);

        const prefill_plan prefill = plan_prefill(prefill_records, args);
        const decode_plan decode = plan_decode(decode_records, args);

        if (args.output_format == "summary") {
            print_summary(std::cout, prefill, decode);
        } else {
            print_env_output(prefill, decode);
            print_summary(std::cerr, prefill, decode);
        }
        return 0;
    } catch (const std::exception & err) {
        std::cerr << "error: " << err.what() << "\n";
        return 1;
    }
}
