#include "aot.hpp"

#include "logger.hpp"

#include <HTP/QnnHtpContext.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace {

struct mapped_file {
    int         fd   = -1;
    size_t      size = 0;
    const void * data = nullptr;

    explicit mapped_file(const std::string & path) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            return;
        }

        struct stat st = {};
        if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
            ::close(fd);
            fd = -1;
            return;
        }

        size = static_cast<size_t>(st.st_size);
        data = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (data == MAP_FAILED) {
            data = nullptr;
            ::close(fd);
            fd = -1;
            size = 0;
        }
    }

    ~mapped_file() {
        if (data) {
            ::munmap(const_cast<void *>(data), size);
        }
        if (fd >= 0) {
            ::close(fd);
        }
    }

    bool is_valid() const {
        return fd >= 0 && data != nullptr && size > 0;
    }
};

struct tensor_io_sets {
    std::vector<ggml_tensor *> inputs;
    std::vector<ggml_tensor *> outputs;
};

bool is_graph_skip_op(const ggml_tensor * tensor) {
    return tensor->op == GGML_OP_NONE || tensor->op == GGML_OP_VIEW || tensor->op == GGML_OP_PERMUTE;
}

tensor_io_sets get_io_tensors_from_graph(const ggml_cgraph * cgraph) {
    struct connectivity_info {
        size_t in_degree    = 0;
        size_t out_degree   = 0;
        size_t insert_index = 0;
    };

    std::unordered_map<ggml_tensor *, connectivity_info> connectivity_map;

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        auto * dst = cgraph->nodes[i];
        if (ggml_is_empty(dst) || is_graph_skip_op(dst)) {
            continue;
        }

        if (connectivity_map.count(dst) == 0) {
            connectivity_map[dst] = { 1, 0, connectivity_map.size() };
        } else {
            ++connectivity_map[dst].in_degree;
        }

        for (size_t j = 0; j < GGML_MAX_DIMS && dst->src[j]; ++j) {
            auto * src = dst->src[j];
            if (connectivity_map.count(src) == 0) {
                connectivity_map[src] = { 0, 1, connectivity_map.size() };
            } else {
                ++connectivity_map[src].out_degree;
            }
        }
    }

    tensor_io_sets result;
    for (const auto & kv : connectivity_map) {
        if (kv.second.in_degree == 0) {
            result.inputs.push_back(kv.first);
        }
        if (kv.second.out_degree == 0) {
            result.outputs.push_back(kv.first);
        }
    }

    std::sort(result.inputs.begin(), result.inputs.end(), [&connectivity_map](ggml_tensor * lhs, ggml_tensor * rhs) {
        return connectivity_map[lhs].insert_index < connectivity_map[rhs].insert_index;
    });
    std::sort(result.outputs.begin(), result.outputs.end(), [&connectivity_map](ggml_tensor * lhs, ggml_tensor * rhs) {
        return connectivity_map[lhs].insert_index < connectivity_map[rhs].insert_index;
    });

    return result;
}

size_t tensor_nbytes(const Qnn_Tensor_t & tensor) {
    size_t n_elements = 1;
    for (size_t i = 0; i < QNN_TENSOR_GET_RANK(tensor); ++i) {
        n_elements *= QNN_TENSOR_GET_DIMENSIONS(tensor)[i];
    }
    return n_elements * qnn::qnn_datatype_size(QNN_TENSOR_GET_DATA_TYPE(tensor));
}

void copy_ggml_rows_to_contiguous(const ggml_tensor * src, size_t row_offset, size_t n_rows, size_t row_elems, float * dst) {
    for (size_t i = 0; i < n_rows; ++i) {
        const auto * src_ptr = reinterpret_cast<const float *>(reinterpret_cast<const char *>(src->data) +
                                                               (row_offset + i) * src->nb[1]);
        std::memcpy(dst + i * row_elems, src_ptr, row_elems * sizeof(float));
    }
}

void copy_contiguous_rows_to_ggml(ggml_tensor * dst, size_t row_offset, size_t n_rows, size_t row_elems, const float * src) {
    for (size_t i = 0; i < n_rows; ++i) {
        auto * dst_ptr = reinterpret_cast<float *>(reinterpret_cast<char *>(dst->data) + (row_offset + i) * dst->nb[1]);
        std::memcpy(dst_ptr, src + i * row_elems, row_elems * sizeof(float));
    }
}

}  // namespace

namespace qnn {

bool qnn_aot_config::load(const std::string & config_path) {
    std::ifstream file(config_path);
    if (!file.is_open()) {
        QNN_LOG_WARN("[aot] failed to open config: %s\n", config_path.c_str());
        return false;
    }

    nlohmann::json json;
    try {
        file >> json;
    } catch (const std::exception & e) {
        QNN_LOG_WARN("[aot] failed to parse config %s: %s\n", config_path.c_str(), e.what());
        return false;
    }

    try {
        const auto & model_json = json.at("model_parameters");
        model_json.at("n_layers").get_to(model.n_layers);
        model_json.at("vocab_size").get_to(model.vocab_size);
        model_json.at("embed_dim").get_to(model.embed_dim);
        model_json.at("ffn_hidden_dim").get_to(model.ffn_hidden_dim);
        model_json.at("head_dim").get_to(model.head_dim);
        model_json.at("n_kv_heads").get_to(model.n_kv_heads);
        model_json.at("rope_theta").get_to(model.rope_theta);
        model_json.at("rms_norm_eps").get_to(model.rms_norm_eps);
        model_json.at("attention_mask_value").get_to(model.attention_mask_value);
        model_json.at("tie_embedding").get_to(model.tie_embedding);

        if (json.contains("qnn_parameters")) {
            const auto & qnn_json = json.at("qnn_parameters");
            if (qnn_json.contains("n_hvx_threads")) {
                qnn_json.at("n_hvx_threads").get_to(n_hvx_threads);
            }
        }

        if (json.contains("graphs")) {
            for (const auto & graph_json : json.at("graphs")) {
                qnn_aot_graph_config graph;
                graph_json.at("type").get_to(graph.type);
                graph_json.at("graph_name").get_to(graph.graph_name);
                graph_json.at("model_path").get_to(graph.model_path);
                graph_json.at("x_name").get_to(graph.x_name);
                graph_json.at("out_name").get_to(graph.out_name);
                graph_json.at("batch_size").get_to(graph.batch_size);
                graph_json.at("cache_size").get_to(graph.cache_size);
                graph_json.at("context_size").get_to(graph.context_size);
                graph_json.at("start_layer_id").get_to(graph.start_layer_id);
                graph_json.at("end_layer_id").get_to(graph.end_layer_id);
                graph_json.at("kv_size").get_to(graph.kv_size);
                if (graph_json.contains("kv_path_format")) {
                    graph_json.at("kv_path_format").get_to(graph.kv_path_format);
                }
                if (graph.type == "transformers") {
                    transformer_graphs.push_back(std::move(graph));
                }
            }
        }
    } catch (const std::exception & e) {
        QNN_LOG_WARN("[aot] invalid config schema in %s: %s\n", config_path.c_str(), e.what());
        return false;
    }

    return !transformer_graphs.empty();
}

qnn_aot_context::qnn_aot_context(qnn_instance_ptr instance, const std::string & binary_path) :
    instance(std::move(instance)),
    binary_path(binary_path) {
    auto qnn_interface = this->instance->get_qnn_interface();
    auto qnn_system    = this->instance->get_qnn_system_interface();
    if (!qnn_interface || !qnn_system) {
        QNN_LOG_WARN("[aot] qnn interface is not initialized\n");
        return;
    }

    mapped_file binary(binary_path);
    if (!binary.is_valid()) {
        QNN_LOG_WARN("[aot] failed to mmap binary: %s, errno=%d\n", binary_path.c_str(), errno);
        return;
    }

    std::vector<const QnnContext_Config_t *> context_configs;
    QnnHtpContext_CustomConfig_t htp_io_estimation_config = {
        .option          = QNN_HTP_CONTEXT_CONFIG_OPTION_IO_MEM_ESTIMATION,
        .ioMemEstimation = true,
    };
    QnnContext_Config_t io_estimation_config = {
        .option       = QNN_CONTEXT_CONFIG_OPTION_CUSTOM,
        .customConfig = &htp_io_estimation_config,
    };
    context_configs.push_back(&io_estimation_config);
    context_configs.push_back(nullptr);

    auto error = qnn_interface->qnn_context_create_from_binary(
        this->instance->get_qnn_backend_handle(),
        this->instance->get_qnn_device_handle(),
        context_configs.data(),
        binary.data,
        binary.size,
        &context_handle,
        nullptr);
    if (error != QNN_SUCCESS || context_handle == nullptr) {
        QNN_LOG_WARN("[aot] contextCreateFromBinary failed for %s: %s\n", binary_path.c_str(), get_qnn_error_string(error));
        context_handle = nullptr;
        return;
    }

    error = qnn_system->qnn_system_context_create(&system_context);
    if (error != QNN_SUCCESS || system_context == nullptr) {
        QNN_LOG_WARN("[aot] systemContextCreate failed for %s: %s\n", binary_path.c_str(), get_qnn_error_string(error));
        return;
    }

    QNN_LOG_INFO("[aot] loaded context binary %s\n", binary_path.c_str());
}

qnn_aot_context::~qnn_aot_context() {
    auto qnn_interface = instance ? instance->get_qnn_interface() : qnn_interface_ptr();
    auto qnn_system    = instance ? instance->get_qnn_system_interface() : std::shared_ptr<qnn_system_interface>();

    if (system_context && qnn_system) {
        auto error = qnn_system->qnn_system_context_free(system_context);
        if (error != QNN_SUCCESS) {
            QNN_LOG_WARN("[aot] failed to free system context: %s\n", get_qnn_error_string(error));
        }
        system_context = nullptr;
    }

    if (context_handle && qnn_interface) {
        auto error = qnn_interface->qnn_context_free(context_handle, nullptr);
        if (error != QNN_SUCCESS) {
            QNN_LOG_WARN("[aot] failed to free aot context: %s\n", get_qnn_error_string(error));
        }
        context_handle = nullptr;
    }
}

qnn_aot_graph::qnn_aot_graph(qnn_instance_ptr instance, std::shared_ptr<qnn_aot_context> context, qnn_aot_graph_config config) :
    _instance(std::move(instance)),
    _qnn_interface(_instance->get_qnn_interface()),
    _context(std::move(context)),
    _config(std::move(config)) {
    if (!_qnn_interface || !_context || !_context->is_valid()) {
        QNN_LOG_WARN("[aot] invalid context for graph %s\n", _config.graph_name.c_str());
        return;
    }

    _valid = retrieve_graph_metadata() && allocate_tensor_buffers() && set_hvx_threads(_config.n_hvx_threads);
}

bool qnn_aot_graph::retrieve_graph_metadata() {
    auto error = _qnn_interface->qnn_graph_retrieve(_context->context_handle, _config.graph_name.c_str(), &_graph_handle);
    if (error != QNN_SUCCESS || _graph_handle == nullptr) {
        QNN_LOG_WARN("[aot] graphRetrieve failed for %s: %s\n", _config.graph_name.c_str(), get_qnn_error_string(error));
        return false;
    }

    auto process_graph_info = [this](auto & graph_info) {
        _inputs.reserve(graph_info.numGraphInputs);
        for (size_t i = 0; i < graph_info.numGraphInputs; ++i) {
            const auto & tensor = graph_info.graphInputs[i];
            _input_index.emplace(QNN_TENSOR_GET_NAME(tensor), _inputs.size());
            _inputs.push_back(tensor);
        }

        _outputs.reserve(graph_info.numGraphOutputs);
        for (size_t i = 0; i < graph_info.numGraphOutputs; ++i) {
            const auto & tensor = graph_info.graphOutputs[i];
            _output_index.emplace(QNN_TENSOR_GET_NAME(tensor), _outputs.size());
            _outputs.push_back(tensor);
        }
    };

    auto qnn_system = _instance->get_qnn_system_interface();
    if (!qnn_system) {
        QNN_LOG_WARN("[aot] qnn system interface is not initialized\n");
        return false;
    }

    mapped_file binary(_context->binary_path);
    if (!binary.is_valid()) {
        QNN_LOG_WARN("[aot] failed to mmap binary metadata: %s, errno=%d\n", _context->binary_path.c_str(), errno);
        return false;
    }

    const QnnSystemContext_BinaryInfo_t * binary_info = nullptr;
    Qnn_ContextBinarySize_t               binary_info_size = 0;
    error = qnn_system->qnn_system_context_get_binary_info(_context->system_context, const_cast<void *>(binary.data), binary.size,
                                                           &binary_info, &binary_info_size);
    if (error != QNN_SUCCESS || binary_info == nullptr) {
        QNN_LOG_WARN("[aot] systemContextGetBinaryInfo failed for %s: %s\n", _context->binary_path.c_str(),
                     get_qnn_error_string(error));
        return false;
    }

    bool found = false;
    switch (binary_info->version) {
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1: {
            const auto & info = binary_info->contextBinaryInfoV1;
            for (size_t i = 0; i < info.numGraphs; ++i) {
                const auto & current_graph = info.graphs[i];
                if (current_graph.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1 &&
                    std::strcmp(current_graph.graphInfoV1.graphName, _config.graph_name.c_str()) == 0) {
                    process_graph_info(current_graph.graphInfoV1);
                    found = true;
                    break;
                }
            }
        } break;
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2: {
            const auto & info = binary_info->contextBinaryInfoV2;
            for (size_t i = 0; i < info.numGraphs; ++i) {
                const auto & current_graph = info.graphs[i];
                if (current_graph.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2 &&
                    std::strcmp(current_graph.graphInfoV2.graphName, _config.graph_name.c_str()) == 0) {
                    process_graph_info(current_graph.graphInfoV2);
                    found = true;
                    break;
                }
            }
        } break;
#if (QNN_API_VERSION_MINOR >= 21)
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3: {
            const auto & info = binary_info->contextBinaryInfoV3;
            for (size_t i = 0; i < info.numGraphs; ++i) {
                const auto & current_graph = info.graphs[i];
                if (current_graph.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3 &&
                    std::strcmp(current_graph.graphInfoV3.graphName, _config.graph_name.c_str()) == 0) {
                    process_graph_info(current_graph.graphInfoV3);
                    found = true;
                    break;
                }
            }
        } break;
#endif
        default:
            QNN_LOG_WARN("[aot] unsupported binary info version: %d\n", (int) binary_info->version);
            break;
    }
    if (!found) {
        QNN_LOG_WARN("[aot] graph info not found for %s\n", _config.graph_name.c_str());
    }
    return found;
}

bool qnn_aot_graph::allocate_tensor_buffers() {
    auto bind_tensor = [this](Qnn_Tensor_t & tensor) -> bool {
        const char * name = QNN_TENSOR_GET_NAME(tensor);
        if (!name || name[0] == '\0') {
            QNN_LOG_WARN("[aot] anonymous tensor in graph %s\n", _config.graph_name.c_str());
            return false;
        }

        const size_t size = tensor_nbytes(tensor);
        auto buffer = std::make_shared<qnn_rpc_buffer>(_instance, size, QNN_TENSOR_GET_RANK(tensor),
                                                       QNN_TENSOR_GET_DIMENSIONS(tensor), QNN_TENSOR_GET_DATA_TYPE(tensor),
                                                       _context->context_handle);
        if (!buffer->is_valid()) {
            QNN_LOG_WARN("[aot] failed to allocate rpc buffer for %s (%zu bytes)\n", name, size);
            return false;
        }

        _buffers.emplace(name, buffer);
        QNN_TENSOR_SET_MEM_TYPE(tensor, QNN_TENSORMEMTYPE_MEMHANDLE);
        QNN_TENSOR_SET_MEM_HANDLE(tensor, buffer->get_mem_handle());
        std::memset(buffer->get_buffer(), 0, buffer->get_size());
        return true;
    };

    for (auto & tensor : _inputs) {
        if (!bind_tensor(tensor)) {
            return false;
        }
    }
    for (auto & tensor : _outputs) {
        if (!bind_tensor(tensor)) {
            return false;
        }
    }
    return true;
}

bool qnn_aot_graph::set_hvx_threads(size_t n_threads) {
    if (_config.batch_size == 0) {
        return true;
    }

    QnnHtpGraph_CustomConfig_t htp_hvx_thread_config = {
        .option        = QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS,
        .numHvxThreads = static_cast<uint32_t>(n_threads),
    };
    QnnGraph_Config_t hvx_thread_config = {
        .option       = QNN_GRAPH_CONFIG_OPTION_CUSTOM,
        .customConfig = &htp_hvx_thread_config,
    };
    const QnnGraph_Config_t * graph_configs[] = { &hvx_thread_config, nullptr };

    auto error = _qnn_interface->qnn_graph_set_config(_graph_handle, graph_configs);
    if (error != QNN_SUCCESS) {
        QNN_LOG_WARN("[aot] graphSetConfig failed for %s: %s\n", _config.graph_name.c_str(), get_qnn_error_string(error));
        return false;
    }
    return true;
}

bool qnn_aot_graph::execute() {
    auto error = _qnn_interface->qnn_graph_execute(_graph_handle, _inputs.data(), _inputs.size(), _outputs.data(), _outputs.size(),
                                                   nullptr, nullptr);
    if (error != QNN_SUCCESS) {
        std::fprintf(stderr, "[aot] graphExecute failed for %s: err=%d (%s)\n",
                     _config.graph_name.c_str(), (int) error, get_qnn_error_string(error));
        return false;
    }
    return true;
}

void * qnn_aot_graph::buffer_data(const std::string & name) {
    auto it = _buffers.find(name);
    return it != _buffers.end() ? it->second->get_buffer() : nullptr;
}

const void * qnn_aot_graph::buffer_data(const std::string & name) const {
    auto it = _buffers.find(name);
    return it != _buffers.end() ? it->second->get_buffer() : nullptr;
}

size_t qnn_aot_graph::buffer_size(const std::string & name) const {
    auto it = _buffers.find(name);
    return it != _buffers.end() ? it->second->get_size() : 0;
}

bool qnn_aot_graph::has_buffer(const std::string & name) const {
    return _buffers.count(name) != 0;
}

qnn_aot_runtime::qnn_aot_runtime(qnn_instance_ptr instance, backend_index_type device) :
    _instance(std::move(instance)),
    _device(device) {}

bool qnn_aot_runtime::initialize(const std::string & config_path, const std::string & model_dir) {
    _config_path = config_path;
    _model_dir   = model_dir;
    _profile     = std::getenv("GGML_QNN_AOT_PROFILE") != nullptr;

    if (!_config.load(config_path)) {
        return false;
    }

    if (_config.transformer_graphs.empty()) {
        QNN_LOG_WARN("[aot] no transformer graphs defined in %s\n", config_path.c_str());
        return false;
    }

    std::vector<qnn_aot_graph_config> graph_configs = _config.transformer_graphs;
    std::sort(graph_configs.begin(), graph_configs.end(), [](const qnn_aot_graph_config & lhs, const qnn_aot_graph_config & rhs) {
        if (lhs.batch_size != rhs.batch_size) {
            return lhs.batch_size < rhs.batch_size;
        }
        return lhs.graph_name < rhs.graph_name;
    });

    for (auto graph_config : graph_configs) {
        graph_config.n_hvx_threads = _config.n_hvx_threads;
        if (_transformer_graphs.count(graph_config.batch_size) != 0) {
            QNN_LOG_WARN("[aot] duplicate transformer batch size %zu, skip graph %s\n", graph_config.batch_size,
                         graph_config.graph_name.c_str());
            continue;
        }

        const auto model_path = (std::filesystem::path(_model_dir) / graph_config.model_path).string();
        auto context_it = _contexts.find(model_path);
        if (context_it == _contexts.end()) {
            auto context = std::make_shared<qnn_aot_context>(_instance, model_path);
            if (!context->is_valid()) {
                return false;
            }
            context_it = _contexts.emplace(model_path, std::move(context)).first;
        }

        auto graph = std::make_unique<qnn_aot_graph>(_instance, context_it->second, graph_config);
        if (!graph->is_valid()) {
            return false;
        }

        QNN_LOG_INFO("[aot] initialized transformer graph %s (batch=%zu) from %s\n", graph_config.graph_name.c_str(),
                     graph_config.batch_size, model_path.c_str());
        _transformer_graphs.emplace(graph_config.batch_size, std::move(graph));
    }

    if (_transformer_graphs.empty()) {
        QNN_LOG_WARN("[aot] failed to initialize any transformer graph from %s\n", config_path.c_str());
        return false;
    }

    compute_rope_embeds();
    reset_state();
    _enabled = true;
    return true;
}

bool qnn_aot_runtime::has_prefix(const char * name, const char * prefix) {
    if (!name || !prefix) {
        return false;
    }
    return std::strncmp(name, prefix, std::strlen(prefix)) == 0;
}

bool qnn_aot_runtime::is_transformer_stage_name(const char * name) {
    return has_prefix(name, "norm-") || has_prefix(name, "attn_norm-") || has_prefix(name, "attn_out-") || has_prefix(name, "Qcur-") ||
           has_prefix(name, "Kcur-") || has_prefix(name, "Vcur-") || has_prefix(name, "cache_k_l") || has_prefix(name, "cache_v_l") ||
           has_prefix(name, "kq-") || has_prefix(name, "kq_soft_max-") || has_prefix(name, "kqv-") || has_prefix(name, "kqv_out-") ||
           has_prefix(name, "ffn_inp-") || has_prefix(name, "ffn_norm-") || has_prefix(name, "ffn_gate-") ||
           has_prefix(name, "ffn_up-") || has_prefix(name, "ffn_swiglu-") || has_prefix(name, "ffn_out-") ||
           has_prefix(name, "l_out-");
}

bool qnn_aot_runtime::is_cpu_stage_name(const char * name) {
    if (!name) {
        return false;
    }
    return std::strcmp(name, "inp_tokens") == 0 || std::strcmp(name, "embd") == 0 ||
           std::strcmp(name, "result_norm") == 0 || std::strcmp(name, "result_output") == 0;
}

bool qnn_aot_runtime::is_embedding_lookup(const ggml_tensor * op) {
    if (op->op != GGML_OP_GET_ROWS || op->src[1] == nullptr) {
        return false;
    }
    const char * src1_name = ggml_get_name(op->src[1]);
    return src1_name && std::strcmp(src1_name, "inp_tokens") == 0;
}

bool qnn_aot_runtime::prefers_cpu_op(const ggml_tensor * op) const {
    if (op == nullptr) {
        return false;
    }

    const char * name = ggml_get_name(op);
    return is_cpu_stage_name(name) || has_prefix(name, "cache_k_l") || has_prefix(name, "cache_v_l") || is_embedding_lookup(op);
}

bool qnn_aot_runtime::supports_op(const ggml_tensor * op) const {
    if (!_enabled || op == nullptr || op->op == GGML_OP_NONE) {
        return false;
    }

    const char * name = ggml_get_name(op);
    if (prefers_cpu_op(op)) {
        return false;
    }

    if (is_transformer_stage_name(name)) {
        return true;
    }

    bool has_transformer_src = false;
    for (size_t i = 0; i < GGML_MAX_SRC && op->src[i]; ++i) {
        const char * src_name = ggml_get_name(op->src[i]);
        if (is_cpu_stage_name(src_name)) {
            return false;
        }
        has_transformer_src = has_transformer_src || is_transformer_stage_name(src_name);
    }

    return has_transformer_src;
}

qnn_aot_runtime::aot_match_result qnn_aot_runtime::match_transformer_graph(ggml_cgraph * cgraph) const {
    aot_match_result result;
    if (!_enabled || cgraph == nullptr || cgraph->n_nodes == 0) {
        return result;
    }

    bool seen_transformer = false;
    bool seen_result_norm = false;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const char * name = ggml_get_name(cgraph->nodes[i]);
        seen_transformer = seen_transformer || is_transformer_stage_name(name);
        seen_result_norm = seen_result_norm || (name && std::strcmp(name, "result_norm") == 0);
    }

    if (!seen_transformer || seen_result_norm) {
        if (_profile) {
            const char * first_name = cgraph->n_nodes > 0 ? ggml_get_name(cgraph->nodes[0]) : nullptr;
            const char * last_name  = cgraph->n_nodes > 0 ? ggml_get_name(cgraph->nodes[cgraph->n_nodes - 1]) : nullptr;
            std::fprintf(stderr,
                         "[aot] skip cgraph: seen_transformer=%d seen_result_norm=%d n_nodes=%d first=%s last=%s\n",
                         (int) seen_transformer, (int) seen_result_norm, cgraph->n_nodes,
                         first_name ? first_name : "<null>", last_name ? last_name : "<null>");
        }
        return result;
    }

    const auto io = get_io_tensors_from_graph(cgraph);
    for (auto * input : io.inputs) {
        const char * name = ggml_get_name(input);
        if (name && std::strcmp(name, "embd") == 0) {
            result.embd = input;
            break;
        }
    }

    if (!io.outputs.empty()) {
        for (auto * output : io.outputs) {
            const char * name = ggml_get_name(output);
            if (is_transformer_stage_name(name) && has_prefix(name, "l_out-")) {
                result.out = output;
            }
        }
    }

    if (!result.embd || !result.out) {
        if (_profile) {
            const char * first_name = cgraph->n_nodes > 0 ? ggml_get_name(cgraph->nodes[0]) : nullptr;
            const char * last_name  = cgraph->n_nodes > 0 ? ggml_get_name(cgraph->nodes[cgraph->n_nodes - 1]) : nullptr;
            std::fprintf(stderr,
                         "[aot] skip cgraph: missing_io embd=%s out=%s n_nodes=%d first=%s last=%s\n",
                         result.embd ? ggml_get_name(result.embd) : "<null>",
                         result.out ? ggml_get_name(result.out) : "<null>",
                         cgraph->n_nodes,
                         first_name ? first_name : "<null>",
                         last_name ? last_name : "<null>");
        }
        return result;
    }

    result.n_tokens       = static_cast<size_t>(result.embd->ne[1]);
    result.inferred_pos   = infer_start_pos(io.inputs, result.n_tokens);
    result.is_transformer = result.n_tokens > 0;
    if (_profile) {
        std::fprintf(stderr, "[aot] matched cgraph: n_nodes=%d embd=%s out=%s n_tokens=%zu inferred_pos=%zu\n",
                     cgraph->n_nodes,
                     ggml_get_name(result.embd) ? ggml_get_name(result.embd) : "<null>",
                     ggml_get_name(result.out) ? ggml_get_name(result.out) : "<null>",
                     result.n_tokens, result.inferred_pos);
    }
    return result;
}

qnn_aot_graph * qnn_aot_runtime::select_transformer_graph(size_t n_tokens) const {
    if (_transformer_graphs.empty()) {
        return nullptr;
    }

    size_t target_tokens = std::max<size_t>(1, n_tokens);
    auto it = _transformer_graphs.lower_bound(target_tokens);

    if (it != _transformer_graphs.end()) {
        return it->second.get();
    }

    return _transformer_graphs.rbegin()->second.get();
}

void qnn_aot_runtime::compute_rope_embeds() {
    const auto head_dim = _config.model.head_dim;
    std::vector<float> inv_freqs(head_dim / 2);
    for (size_t i = 0; i < head_dim / 2; ++i) {
        inv_freqs[i] = 1.0f / std::pow(_config.model.rope_theta, 2.0f * i / head_dim);
    }

    size_t max_positions = 0;
    for (const auto & graph : _config.transformer_graphs) {
        max_positions = std::max(max_positions, graph.context_size);
    }
    if (max_positions == 0) {
        return;
    }

    _rope_embeds.resize(max_positions);
    for (size_t pos = 0; pos < max_positions; ++pos) {
        auto & rope = _rope_embeds[pos];
        rope.cos_values.resize(head_dim / 2);
        rope.sin_values.resize(head_dim / 2);
        for (size_t i = 0; i < head_dim / 2; ++i) {
            const float freq = static_cast<float>(pos) * inv_freqs[i];
            rope.cos_values[i] = std::cos(freq);
            rope.sin_values[i] = std::sin(freq);
        }
    }
}

void qnn_aot_runtime::fill_rope_embeds(qnn_aot_graph & graph, size_t start_pos, size_t n_tokens) {
    auto * cos_buffer = static_cast<float *>(graph.buffer_data("rope_embed_cos"));
    auto * sin_buffer = static_cast<float *>(graph.buffer_data("rope_embed_sin"));
    if (!cos_buffer || !sin_buffer) {
        return;
    }

    const size_t row_elems = _config.model.head_dim / 2;
    std::memset(cos_buffer, 0, graph.buffer_size("rope_embed_cos"));
    std::memset(sin_buffer, 0, graph.buffer_size("rope_embed_sin"));

    for (size_t i = 0; i < n_tokens; ++i) {
        const size_t pos = std::min(start_pos + i, _rope_embeds.size() - 1);
        std::memcpy(cos_buffer + i * row_elems, _rope_embeds[pos].cos_values.data(), row_elems * sizeof(float));
        std::memcpy(sin_buffer + i * row_elems, _rope_embeds[pos].sin_values.data(), row_elems * sizeof(float));
    }
}

void qnn_aot_runtime::fill_attention_bias(qnn_aot_graph & graph, size_t n_tokens) {
    auto * attn_bias = static_cast<__fp16 *>(graph.buffer_data("attn_bias"));
    if (!attn_bias) {
        return;
    }

    const auto & graph_config = graph.config();
    const size_t graph_batch  = graph.batch_size();
    const size_t context_size = graph_config.context_size;
    const size_t cache_size   = graph_config.cache_size;
    const __fp16 mask_value   = (__fp16) _config.model.attention_mask_value;

    std::fill(attn_bias, attn_bias + graph_batch * context_size, mask_value);
    for (size_t i = 0; i < n_tokens; ++i) {
        auto * row = attn_bias + i * context_size;
        for (size_t j = 0; j < std::min(_kv_position, cache_size); ++j) {
            row[j] = (__fp16) 0.0f;
        }
        for (size_t j = 0; j < n_tokens; ++j) {
            row[cache_size + j] = j <= i ? (__fp16) 0.0f : mask_value;
        }
    }
}

void qnn_aot_runtime::save_kv(qnn_aot_graph & graph, size_t n_tokens) {
    const auto & graph_config = graph.config();
    const size_t head_dim     = _config.model.head_dim;

    for (size_t layer = graph_config.start_layer_id; layer < graph_config.end_layer_id; ++layer) {
        for (size_t head = 0; head < _config.model.n_kv_heads; ++head) {
            const auto key_out_name     = std::string("layer_") + std::to_string(layer) + "_key_" + std::to_string(head);
            const auto value_out_name   = std::string("layer_") + std::to_string(layer) + "_value_" + std::to_string(head);
            const auto key_cache_name   = std::string("layer_") + std::to_string(layer) + "_key_t_cache_" + std::to_string(head);
            const auto value_cache_name = std::string("layer_") + std::to_string(layer) + "_value_cache_" + std::to_string(head);

            auto * key_out     = static_cast<char *>(graph.buffer_data(key_out_name));
            auto * value_out   = static_cast<char *>(graph.buffer_data(value_out_name));
            auto * key_cache   = static_cast<char *>(graph.buffer_data(key_cache_name));
            auto * value_cache = static_cast<char *>(graph.buffer_data(value_cache_name));
            if (!key_out || !value_out || !key_cache || !value_cache) {
                continue;
            }

            const size_t key_element_size = graph.buffer_size(key_out_name) / (graph.batch_size() * head_dim);
            const size_t value_element_size = graph.buffer_size(value_out_name) / (graph.batch_size() * head_dim);

            for (size_t token = 0; token < n_tokens; ++token) {
                for (size_t dim = 0; dim < head_dim; ++dim) {
                    std::memcpy(key_cache + dim * graph_config.cache_size * key_element_size + (_kv_position + token) * key_element_size,
                                key_out + (token * head_dim + dim) * key_element_size,
                                key_element_size);
                }
                std::memcpy(value_cache + (_kv_position + token) * head_dim * value_element_size,
                            value_out + token * head_dim * value_element_size,
                            head_dim * value_element_size);
            }
        }
    }
}

void qnn_aot_runtime::zero_transformer_state() {
    for (auto & kv : _transformer_graphs) {
        auto & graph = *kv.second;
        for (const char * tensor_name : { "x", "attn_bias", "rope_embed_cos", "rope_embed_sin", "out" }) {
            if (graph.has_buffer(tensor_name)) {
                std::memset(graph.buffer_data(tensor_name), 0, graph.buffer_size(tensor_name));
            }
        }

        const auto & graph_config = graph.config();
        for (size_t layer = graph_config.start_layer_id; layer < graph_config.end_layer_id; ++layer) {
            for (size_t head = 0; head < _config.model.n_kv_heads; ++head) {
                const auto key_cache_name   = std::string("layer_") + std::to_string(layer) + "_key_t_cache_" + std::to_string(head);
                const auto value_cache_name = std::string("layer_") + std::to_string(layer) + "_value_cache_" + std::to_string(head);
                if (graph.has_buffer(key_cache_name)) {
                    std::memset(graph.buffer_data(key_cache_name), 0, graph.buffer_size(key_cache_name));
                }
                if (graph.has_buffer(value_cache_name)) {
                    std::memset(graph.buffer_data(value_cache_name), 0, graph.buffer_size(value_cache_name));
                }
            }
        }
    }
}

size_t qnn_aot_runtime::infer_start_pos(const std::vector<ggml_tensor *> & inputs, size_t n_tokens) const {
    size_t inferred = _kv_position;
    for (auto * tensor : inputs) {
        if (!tensor || tensor->type != GGML_TYPE_I32 || ggml_n_dims(tensor) != 1 || tensor->data == nullptr) {
            continue;
        }
        if (static_cast<size_t>(tensor->ne[0]) < n_tokens || n_tokens == 0) {
            continue;
        }

        const auto * values = static_cast<const int32_t *>(tensor->data);
        bool contiguous = true;
        for (size_t i = 1; i < n_tokens; ++i) {
            if (values[i] != values[0] + static_cast<int32_t>(i)) {
                contiguous = false;
                break;
            }
        }
        if (contiguous && values[0] >= 0) {
            inferred = std::max(inferred, static_cast<size_t>(values[0]));
        }
    }
    return inferred;
}

bool qnn_aot_runtime::execute_transformer(ggml_cgraph * cgraph, const aot_match_result & match) {
    GGML_UNUSED(cgraph);
    std::fprintf(stderr, "[aot] execute_transformer start: n_tokens=%zu inferred_pos=%zu kv_position=%zu\n",
                 match.n_tokens, match.inferred_pos, _kv_position);
    auto * graph = select_transformer_graph(match.n_tokens);
    std::fprintf(stderr, "[aot] selected graph: %s batch_size=%zu\n",
                 graph ? graph->config().graph_name.c_str() : "<null>",
                 graph ? graph->batch_size() : 0);
    if (!graph || !match.embd || !match.out) {
        std::fprintf(stderr, "[aot] execute_transformer fail: graph=%p embd=%p out=%p\n",
                     (void *) graph, (void *) match.embd, (void *) match.out);
        return false;
    }

    if ((match.inferred_pos == 0 && _kv_position != 0) ||
        (_kv_position + match.n_tokens > graph->config().cache_size && match.inferred_pos == 0)) {
        std::fprintf(stderr, "[aot] reset transformer state: kv_position=%zu inferred_pos=%zu cache_size=%zu\n",
                     _kv_position, match.inferred_pos, graph->config().cache_size);
        reset_state();
    }

    if (match.embd->type != GGML_TYPE_F32 || match.out->type != GGML_TYPE_F32) {
        QNN_LOG_WARN("[aot] transformer IO expects F32 tensors, got %s -> %s\n", ggml_type_name(match.embd->type),
                     ggml_type_name(match.out->type));
        std::fprintf(stderr, "[aot] execute_transformer fail: io type mismatch %s -> %s\n",
                     ggml_type_name(match.embd->type), ggml_type_name(match.out->type));
        return false;
    }

    const size_t embed_dim = _config.model.embed_dim;
    if (static_cast<size_t>(match.embd->ne[0]) != embed_dim || static_cast<size_t>(match.out->ne[0]) != embed_dim) {
        QNN_LOG_WARN("[aot] transformer IO dim mismatch, expected %zu, got %lld -> %lld\n", embed_dim,
                     (long long) match.embd->ne[0], (long long) match.out->ne[0]);
        std::fprintf(stderr, "[aot] execute_transformer fail: io dim mismatch expected=%zu got=%lld -> %lld\n",
                     embed_dim, (long long) match.embd->ne[0], (long long) match.out->ne[0]);
        return false;
    }

    auto * x_buffer   = static_cast<float *>(graph->buffer_data(graph->config().x_name));
    auto * out_buffer = static_cast<float *>(graph->buffer_data(graph->config().out_name));
    if (!x_buffer || !out_buffer) {
        QNN_LOG_WARN("[aot] missing transformer x/out buffers\n");
        std::fprintf(stderr, "[aot] execute_transformer fail: missing x/out buffers x=%p out=%p\n",
                     (void *) x_buffer, (void *) out_buffer);
        return false;
    }

    size_t offset = 0;
    while (offset < match.n_tokens) {
        const size_t step = std::min(match.n_tokens - offset, graph->batch_size());

        std::memset(x_buffer, 0, graph->buffer_size(graph->config().x_name));
        std::memset(out_buffer, 0, graph->buffer_size(graph->config().out_name));
        copy_ggml_rows_to_contiguous(match.embd, offset, step, embed_dim, x_buffer);
        fill_rope_embeds(*graph, _kv_position, step);
        fill_attention_bias(*graph, step);

        if (!graph->execute()) {
            std::fprintf(stderr, "[aot] execute_transformer fail: graphExecute returned false at offset=%zu step=%zu\n",
                         offset, step);
            return false;
        }

        copy_contiguous_rows_to_ggml(match.out, offset, step, embed_dim, out_buffer);
        save_kv(*graph, step);
        _kv_position += step;
        offset += step;
    }

    std::fprintf(stderr, "[aot] execute_transformer done: n_tokens=%zu new_kv_position=%zu\n",
                 match.n_tokens, _kv_position);
    return true;
}

bool qnn_aot_runtime::maybe_execute(ggml_cgraph * cgraph) {
    const auto match = match_transformer_graph(cgraph);
    if (!match.is_transformer) {
        return false;
    }
    return execute_transformer(cgraph, match);
}

void qnn_aot_runtime::reset_state() {
    _kv_position = 0;
    zero_transformer_state();
}

}  // namespace qnn
