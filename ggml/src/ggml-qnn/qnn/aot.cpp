#include "aot.hpp"

#include "logger.hpp"

#include <HTP/QnnHtpContext.h>
#include <HTP/QnnHtpCommon.h>
#include <HTP/QnnHtpGraph.h>

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
#include <unordered_set>
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

        for (size_t j = 0; j < GGML_MAX_SRC && dst->src[j]; ++j) {
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

void deep_copy_tensor(Qnn_Tensor_t & dst, const Qnn_Tensor_t & src) {
    dst = QNN_TENSOR_INIT;
    dst.version = src.version;

    const char * tensor_name = QNN_TENSOR_GET_NAME(src);
    QNN_TENSOR_SET_NAME(dst, tensor_name ? ::strdup(tensor_name) : nullptr);
    QNN_TENSOR_SET_ID(dst, QNN_TENSOR_GET_ID(src));
    QNN_TENSOR_SET_TYPE(dst, QNN_TENSOR_GET_TYPE(src));
    QNN_TENSOR_SET_DATA_FORMAT(dst, QNN_TENSOR_GET_DATA_FORMAT(src));
    QNN_TENSOR_SET_DATA_TYPE(dst, QNN_TENSOR_GET_DATA_TYPE(src));

    Qnn_QuantizeParams_t qparams = QNN_QUANTIZE_PARAMS_INIT;
    qparams.encodingDefinition   = QNN_TENSOR_GET_QUANT_PARAMS(src).encodingDefinition;
    qparams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;

    if (QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
        qparams.quantizationEncoding = QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding;
        qparams.scaleOffsetEncoding  = QNN_TENSOR_GET_QUANT_PARAMS(src).scaleOffsetEncoding;
    } else if (QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
        qparams.quantizationEncoding         = QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding;
        qparams.axisScaleOffsetEncoding.axis = QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.axis;
        qparams.axisScaleOffsetEncoding.numScaleOffsets =
            QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.numScaleOffsets;
        if (qparams.axisScaleOffsetEncoding.numScaleOffsets > 0) {
            auto * scale_offsets = static_cast<Qnn_ScaleOffset_t *>(std::malloc(
                qparams.axisScaleOffsetEncoding.numScaleOffsets * sizeof(Qnn_ScaleOffset_t)));
            if (scale_offsets != nullptr) {
                std::memcpy(scale_offsets,
                            QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.scaleOffset,
                            qparams.axisScaleOffsetEncoding.numScaleOffsets * sizeof(Qnn_ScaleOffset_t));
            }
            qparams.axisScaleOffsetEncoding.scaleOffset = scale_offsets;
        }
    }

    QNN_TENSOR_SET_QUANT_PARAMS(dst, qparams);
    QNN_TENSOR_SET_RANK(dst, QNN_TENSOR_GET_RANK(src));
    QNN_TENSOR_SET_DIMENSIONS(dst, nullptr);
    if (QNN_TENSOR_GET_RANK(src) > 0) {
        auto * dims = static_cast<uint32_t *>(std::malloc(QNN_TENSOR_GET_RANK(src) * sizeof(uint32_t)));
        if (dims != nullptr) {
            std::memcpy(dims, QNN_TENSOR_GET_DIMENSIONS(src), QNN_TENSOR_GET_RANK(src) * sizeof(uint32_t));
        }
        QNN_TENSOR_SET_DIMENSIONS(dst, dims);
    }
}

void free_deep_copied_tensor(Qnn_Tensor_t & tensor) {
    auto qparams = QNN_TENSOR_GET_QUANT_PARAMS(tensor);
    if (qparams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
        std::free(qparams.axisScaleOffsetEncoding.scaleOffset);
    }

    std::free(QNN_TENSOR_GET_DIMENSIONS(tensor));
    std::free(const_cast<char *>(QNN_TENSOR_GET_NAME(tensor)));
    tensor = QNN_TENSOR_INIT;
}

size_t binary_info_num_graphs(const QnnSystemContext_BinaryInfo_t * binary_info) {
    if (binary_info == nullptr) {
        return 0;
    }

    switch (binary_info->version) {
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1:
            return binary_info->contextBinaryInfoV1.numGraphs;
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2:
            return binary_info->contextBinaryInfoV2.numGraphs;
#if (QNN_API_VERSION_MINOR >= 21)
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3:
            return binary_info->contextBinaryInfoV3.numGraphs;
#endif
        default:
            return 0;
    }
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

void copy_contiguous_to_strided(void * dst, size_t dst_stride, const void * src, size_t n_elements, size_t element_size) {
    auto *       dst_bytes = static_cast<char *>(dst);
    const auto * src_bytes = static_cast<const char *>(src);

    if (element_size == sizeof(uint16_t)) {
        const auto * src_u16 = reinterpret_cast<const uint16_t *>(src_bytes);
        for (size_t i = 0; i < n_elements; ++i) {
            *reinterpret_cast<uint16_t *>(dst_bytes) = src_u16[i];
            dst_bytes += dst_stride;
        }
        return;
    }

    if (element_size == sizeof(uint32_t)) {
        const auto * src_u32 = reinterpret_cast<const uint32_t *>(src_bytes);
        for (size_t i = 0; i < n_elements; ++i) {
            *reinterpret_cast<uint32_t *>(dst_bytes) = src_u32[i];
            dst_bytes += dst_stride;
        }
        return;
    }

    for (size_t i = 0; i < n_elements; ++i) {
        std::memcpy(dst_bytes, src_bytes + i * element_size, element_size);
        dst_bytes += dst_stride;
    }
}

void replace_all(std::string & text, const char * needle, const std::string & replacement) {
    if (needle == nullptr || needle[0] == '\0') {
        return;
    }

    const std::string pattern = needle;
    size_t pos = 0;
    while ((pos = text.find(pattern, pos)) != std::string::npos) {
        text.replace(pos, pattern.size(), replacement);
        pos += replacement.size();
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

        if (json.contains("embeddings")) {
            for (const auto & graph_json : json.at("embeddings")) {
                qnn_aot_graph_config graph;
                graph.type = "lm_head";
                graph_json.at("graph_name").get_to(graph.graph_name);
                graph_json.at("model_path").get_to(graph.model_path);
                graph_json.at("x_name").get_to(graph.x_name);
                graph_json.at("out_name").get_to(graph.out_name);
                graph_json.at("batch_size").get_to(graph.batch_size);
                lm_head_graphs.push_back(std::move(graph));
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

    error = qnn_system->qnn_system_context_get_binary_info(system_context, const_cast<void *>(binary.data), binary.size,
                                                           &binary_info, &binary_info_size);
    if (error != QNN_SUCCESS || binary_info == nullptr) {
        QNN_LOG_WARN("[aot] systemContextGetBinaryInfo failed for %s: %s\n", binary_path.c_str(), get_qnn_error_string(error));
        binary_info      = nullptr;
        binary_info_size = 0;
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
    binary_info      = nullptr;
    binary_info_size = 0;

    if (context_handle && qnn_interface) {
        auto error = qnn_interface->qnn_context_free(context_handle, nullptr);
        if (error != QNN_SUCCESS) {
            QNN_LOG_WARN("[aot] failed to free aot context: %s\n", get_qnn_error_string(error));
        }
        context_handle = nullptr;
    }
}

qnn_aot_graph::qnn_aot_graph(qnn_instance_ptr instance,
                             std::shared_ptr<qnn_aot_context> context,
                             qnn_aot_graph_config config,
                             const qnn_aot_graph * sibling) :
    _instance(std::move(instance)),
    _qnn_interface(_instance->get_qnn_interface()),
    _context(std::move(context)),
    _config(std::move(config)),
    _sibling(sibling) {
    if (!_qnn_interface || !_context || !_context->is_valid()) {
        QNN_LOG_WARN("[aot] invalid context for graph %s\n", _config.graph_name.c_str());
        return;
    }

    _valid = retrieve_graph_metadata() && allocate_tensor_buffers() && set_hvx_threads(_config.n_hvx_threads);
}

qnn_aot_graph::~qnn_aot_graph() {
    for (auto & tensor : _inputs) {
        free_deep_copied_tensor(tensor);
    }
    for (auto & tensor : _outputs) {
        free_deep_copied_tensor(tensor);
    }
}

bool qnn_aot_graph::retrieve_graph_metadata() {
    const auto * binary_info = _context ? _context->binary_info : nullptr;
    if (binary_info == nullptr) {
        QNN_LOG_WARN("[aot] missing binary info for graph %s\n", _config.graph_name.c_str());
        return false;
    }

    auto process_graph_info = [this](auto & graph_info) -> bool {
        _inputs.reserve(graph_info.numGraphInputs);
        for (size_t i = 0; i < graph_info.numGraphInputs; ++i) {
            const auto & tensor = graph_info.graphInputs[i];
            const char * tensor_name = QNN_TENSOR_GET_NAME(tensor);
            if (tensor_name == nullptr || tensor_name[0] == '\0') {
                QNN_LOG_WARN("[aot] anonymous input tensor in graph %s\n", _config.graph_name.c_str());
                return false;
            }
            _input_index.emplace(tensor_name, _inputs.size());
            _inputs.push_back(QNN_TENSOR_INIT);
            deep_copy_tensor(_inputs.back(), tensor);
        }

        _outputs.reserve(graph_info.numGraphOutputs);
        for (size_t i = 0; i < graph_info.numGraphOutputs; ++i) {
            const auto & tensor = graph_info.graphOutputs[i];
            const char * tensor_name = QNN_TENSOR_GET_NAME(tensor);
            if (tensor_name == nullptr || tensor_name[0] == '\0') {
                QNN_LOG_WARN("[aot] anonymous output tensor in graph %s\n", _config.graph_name.c_str());
                return false;
            }
            _output_index.emplace(tensor_name, _outputs.size());
            _outputs.push_back(QNN_TENSOR_INIT);
            deep_copy_tensor(_outputs.back(), tensor);
        }
        auto error = _qnn_interface->qnn_graph_retrieve(_context->context_handle, _config.graph_name.c_str(), &_graph_handle);
        if (error != QNN_SUCCESS || _graph_handle == nullptr) {
            QNN_LOG_WARN("[aot] graphRetrieve failed for %s: %s\n", _config.graph_name.c_str(), get_qnn_error_string(error));
            return false;
        }
        return true;
    };

    bool found = false;
    switch (binary_info->version) {
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1: {
            const auto & info = binary_info->contextBinaryInfoV1;
            for (size_t i = 0; i < info.numGraphs; ++i) {
                const auto & current_graph = info.graphs[i];
                if (current_graph.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1 &&
                    std::strcmp(current_graph.graphInfoV1.graphName, _config.graph_name.c_str()) == 0) {
                    found = process_graph_info(current_graph.graphInfoV1);
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
                    found = process_graph_info(current_graph.graphInfoV2);
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
                    found = process_graph_info(current_graph.graphInfoV3);
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
    size_t total_bytes = 0;
    size_t unique_tensors = 0;

    const bool use_htp_shared_buffers =
        _instance && _instance->get_qnn_interface() &&
        _instance->get_qnn_interface()->get_backend_id() == QNN_BACKEND_ID_HTP;

    auto find_sibling_buffer = [this](const char * name, size_t required_size) -> qnn_buffer_ptr {
        if (_sibling == nullptr || name == nullptr || name[0] == '\0') {
            return nullptr;
        }

        auto sibling_it = _sibling->_buffers.find(name);
        if (sibling_it == _sibling->_buffers.end() || !sibling_it->second) {
            return nullptr;
        }

        if (sibling_it->second->get_size() < required_size) {
            QNN_LOG_WARN("[aot] sibling buffer too small for %s in graph %s: need=%zu have=%zu\n",
                         name,
                         _config.graph_name.c_str(),
                         required_size,
                         sibling_it->second->get_size());
            return nullptr;
        }

        return sibling_it->second;
    };

    auto accumulate_tensor = [&total_bytes, &unique_tensors](const Qnn_Tensor_t & tensor,
                                                             std::unordered_set<std::string> & seen_names,
                                                             const auto & find_sibling_buffer_fn) {
        const char * name = QNN_TENSOR_GET_NAME(tensor);
        if (name == nullptr || name[0] == '\0') {
            return;
        }

        if (!seen_names.insert(name).second) {
            return;
        }

        if (find_sibling_buffer_fn(name, tensor_nbytes(tensor)) != nullptr) {
            return;
        }

        total_bytes = static_cast<size_t>(qnn::align_to(64, static_cast<intptr_t>(total_bytes)));
        total_bytes += tensor_nbytes(tensor);
        unique_tensors += 1;
    };

    if (use_htp_shared_buffers) {
        std::unordered_set<std::string> seen_names;
        seen_names.reserve(_inputs.size() + _outputs.size());
        for (const auto & tensor : _inputs) {
            accumulate_tensor(tensor, seen_names, find_sibling_buffer);
        }
        for (const auto & tensor : _outputs) {
            accumulate_tensor(tensor, seen_names, find_sibling_buffer);
        }

        if (total_bytes > 0) {
            _shared_allocator = std::make_shared<qnn_shared_buffer_allocator>(_instance, total_bytes);
            if (!_shared_allocator || !_shared_allocator->is_valid()) {
                QNN_LOG_WARN("[aot] failed to allocate shared buffer arena for %s, bytes=%zu unique=%zu\n",
                             _config.graph_name.c_str(), total_bytes, unique_tensors);
                return false;
            }
        }
    }

    auto bind_tensor = [this, &total_bytes, &find_sibling_buffer, use_htp_shared_buffers](Qnn_Tensor_t & tensor) -> bool {
        const char * name = QNN_TENSOR_GET_NAME(tensor);
        if (!name || name[0] == '\0') {
            QNN_LOG_WARN("[aot] anonymous tensor in graph %s\n", _config.graph_name.c_str());
            return false;
        }

        // Check if a buffer for this name already exists (e.g. a tensor shared between
        // inputs and outputs). Reuse the existing buffer instead of allocating a new one
        // to avoid a dangling Qnn_MemHandle_t caused by the silently-failing emplace.
        auto existing = _buffers.find(name);
        if (existing != _buffers.end()) {
            QNN_TENSOR_SET_MEM_TYPE(tensor, QNN_TENSORMEMTYPE_MEMHANDLE);
            QNN_TENSOR_SET_MEM_HANDLE(tensor, existing->second->get_mem_handle());
            return true;
        }

        const size_t size = tensor_nbytes(tensor);
        qnn_buffer_ptr buffer = find_sibling_buffer(name, size);
        if (buffer) {
            _buffers.emplace(name, buffer);
            QNN_TENSOR_SET_MEM_TYPE(tensor, QNN_TENSORMEMTYPE_MEMHANDLE);
            QNN_TENSOR_SET_MEM_HANDLE(tensor, buffer->get_mem_handle());
            return true;
        }

        if (!_shared_allocator && use_htp_shared_buffers) {
            QNN_LOG_WARN("[aot] missing shared allocator for unique tensor %s in graph %s\n",
                         name, _config.graph_name.c_str());
            return false;
        }

        if (_shared_allocator) {
            buffer = std::make_shared<qnn_htp_shared_buffer>(_instance,
                                                             _shared_allocator,
                                                             size,
                                                             QNN_TENSOR_GET_DATA_TYPE(tensor),
                                                             _context ? _context->context_handle : nullptr);
        } else {
            buffer = std::make_shared<qnn_rpc_buffer>(_instance,
                                                      size,
                                                      QNN_TENSOR_GET_RANK(tensor),
                                                      QNN_TENSOR_GET_DIMENSIONS(tensor),
                                                      QNN_TENSOR_GET_DATA_TYPE(tensor),
                                                      _context ? _context->context_handle : nullptr);
        }
        if (!buffer || !buffer->is_valid()) {
            QNN_LOG_WARN("[aot] failed to allocate %s buffer for %s (%zu bytes)\n",
                         _shared_allocator ? "shared" : "rpc",
                         name,
                         size);
            return false;
        }

        _buffers.emplace(name, buffer);
        QNN_TENSOR_SET_MEM_TYPE(tensor, QNN_TENSORMEMTYPE_MEMHANDLE);
        QNN_TENSOR_SET_MEM_HANDLE(tensor, buffer->get_mem_handle());
        std::memset(buffer->get_buffer(), 0, buffer->get_size());
        if (!_shared_allocator) {
            total_bytes += size;
        }
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
        std::fprintf(stderr, "[aot] graphExecute failed for %s: handle=%p err=%d (%s)\n",
                     _config.graph_name.c_str(), (void *) _graph_handle, (int) error, get_qnn_error_string(error));
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
    auto in_it = _input_index.find(name);
    if (in_it != _input_index.end() && in_it->second < _inputs.size()) {
        return tensor_nbytes(_inputs[in_it->second]);
    }

    auto out_it = _output_index.find(name);
    if (out_it != _output_index.end() && out_it->second < _outputs.size()) {
        return tensor_nbytes(_outputs[out_it->second]);
    }

    auto it = _buffers.find(name);
    return it != _buffers.end() ? it->second->get_size() : 0;
}

bool qnn_aot_graph::has_buffer(const std::string & name) const {
    return _buffers.count(name) != 0;
}

Qnn_DataType_t qnn_aot_graph::tensor_data_type(const std::string & name) const {
    auto in_it = _input_index.find(name);
    if (in_it != _input_index.end() && in_it->second < _inputs.size()) {
        return QNN_TENSOR_GET_DATA_TYPE(_inputs[in_it->second]);
    }
    auto out_it = _output_index.find(name);
    if (out_it != _output_index.end() && out_it->second < _outputs.size()) {
        return QNN_TENSOR_GET_DATA_TYPE(_outputs[out_it->second]);
    }
    return QNN_DATATYPE_UNDEFINED;
}

qnn_aot_runtime::qnn_aot_runtime(qnn_instance_ptr instance, backend_index_type device) :
    _instance(std::move(instance)),
    _device(device) {}

qnn_aot_runtime::~qnn_aot_runtime() {
    // Destroy graphs before contexts so buffer/memhandle deregistration still sees a live QNN context.
    _transformer_graphs.clear();
    _lm_head_graphs.clear();
    _contexts.clear();
}

bool qnn_aot_runtime::initialize(const std::string & config_path, const std::string & model_dir) {
    _config_path = config_path;
    _model_dir   = model_dir;

    if (!_config.load(config_path)) {
        return false;
    }

    if (_config.transformer_graphs.empty()) {
        QNN_LOG_WARN("[aot] no transformer graphs defined in %s\n", config_path.c_str());
        return false;
    }

    auto load_graph_family = [this](std::vector<qnn_aot_graph_config> graph_configs,
                                    std::map<size_t, std::unique_ptr<qnn_aot_graph>> & graphs,
                                    const char * family_name) -> bool {
        std::sort(graph_configs.begin(), graph_configs.end(), [](const qnn_aot_graph_config & lhs, const qnn_aot_graph_config & rhs) {
            if (lhs.batch_size != rhs.batch_size) {
                return lhs.batch_size > rhs.batch_size;
            }
            return lhs.graph_name < rhs.graph_name;
        });

        for (auto graph_config : graph_configs) {
            graph_config.n_hvx_threads = _config.n_hvx_threads;
            if (graphs.count(graph_config.batch_size) != 0) {
                QNN_LOG_WARN("[aot] duplicate %s batch size %zu, skip graph %s\n",
                             family_name, graph_config.batch_size, graph_config.graph_name.c_str());
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

            const qnn_aot_graph * sibling = nullptr;
            for (auto existing_graph = graphs.rbegin(); existing_graph != graphs.rend(); ++existing_graph) {
                if (existing_graph->second &&
                    existing_graph->second->config().model_path == graph_config.model_path) {
                    sibling = existing_graph->second.get();
                    break;
                }
            }

            auto graph = std::make_unique<qnn_aot_graph>(_instance, context_it->second, graph_config, sibling);
            if (!graph->is_valid()) {
                return false;
            }

            QNN_LOG_INFO("[aot] initialized %s graph %s (batch=%zu) from %s\n",
                         family_name, graph_config.graph_name.c_str(), graph_config.batch_size, model_path.c_str());
            graphs.emplace(graph_config.batch_size, std::move(graph));
        }

        return true;
    };

    if (!load_graph_family(_config.transformer_graphs, _transformer_graphs, "transformer")) {
        return false;
    }

    if (!_config.lm_head_graphs.empty() && !load_graph_family(_config.lm_head_graphs, _lm_head_graphs, "lm_head")) {
        return false;
    }

    if (_transformer_graphs.empty()) {
        QNN_LOG_WARN("[aot] failed to initialize any transformer graph from %s\n", config_path.c_str());
        return false;
    }

    _seed_kv_size = 0;
    for (const auto & graph_config : _config.transformer_graphs) {
        if (graph_config.kv_size == 0) {
            continue;
        }

        if (_seed_kv_size == 0) {
            _seed_kv_size = graph_config.kv_size;
            continue;
        }

        if (_seed_kv_size != graph_config.kv_size) {
            QNN_LOG_WARN("[aot] inconsistent seed KV sizes in %s: %zu vs %zu\n",
                         config_path.c_str(), _seed_kv_size, graph_config.kv_size);
            return false;
        }
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
    if (name == nullptr) {
        return false;
    }

    return has_prefix(name, "norm-") || has_prefix(name, "attn_norm-") || has_prefix(name, "attn_out-") || has_prefix(name, "Qcur-") ||
           has_prefix(name, "Qcur_normed-") || has_prefix(name, "Kcur-") || has_prefix(name, "Kcur_normed-") ||
           has_prefix(name, "Vcur-") || has_prefix(name, "cache_k_l") || has_prefix(name, "cache_v_l") ||
           has_prefix(name, "kq-") || has_prefix(name, "kq_soft_max-") || has_prefix(name, "kqv-") || has_prefix(name, "kqv_out-") ||
           has_prefix(name, "ffn_inp-") || has_prefix(name, "ffn_norm-") || has_prefix(name, "ffn_gate-") ||
           has_prefix(name, "ffn_up-") || has_prefix(name, "ffn_swiglu-") || has_prefix(name, "ffn_out-") ||
           has_prefix(name, "l_out-") || std::strcmp(name, "self_kq_mask_cnv") == 0 ||
           std::strcmp(name, "self_kq_mask_swa_cnv") == 0;
}

bool qnn_aot_runtime::is_cpu_stage_name(const char * name) {
    if (!name) {
        return false;
    }
    return std::strcmp(name, "inp_tokens") == 0 || std::strcmp(name, "embd") == 0;
}

bool qnn_aot_runtime::is_lm_head_stage_name(const char * name) {
    if (!name) {
        return false;
    }
    return std::strcmp(name, "norm") == 0 ||
           std::strcmp(name, "result_norm") == 0 ||
           std::strcmp(name, "result_output") == 0;
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
    if (is_embedding_lookup(op) || is_cpu_stage_name(name)) {
        return true;
    }

    if (is_lm_head_stage_name(name)) {
        return _lm_head_graphs.empty();
    }

    return false;
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

    if (!_lm_head_graphs.empty() && is_lm_head_stage_name(name)) {
        return true;
    }

    bool has_aot_src = false;
    for (size_t i = 0; i < GGML_MAX_SRC && op->src[i]; ++i) {
        const char * src_name = ggml_get_name(op->src[i]);
        if (is_cpu_stage_name(src_name)) {
            return false;
        }
        has_aot_src = has_aot_src || is_transformer_stage_name(src_name) || is_lm_head_stage_name(src_name);
    }

    return has_aot_src;
}

qnn_aot_runtime::aot_match_result qnn_aot_runtime::match_transformer_graph(ggml_cgraph * cgraph) const {
    aot_match_result result;
    if (!_enabled || cgraph == nullptr || cgraph->n_nodes == 0) {
        return result;
    }

    bool seen_transformer = false;
    bool seen_result_norm = false;
    ggml_tensor * norm_node = nullptr;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        auto *       node = cgraph->nodes[i];
        const char * name = ggml_get_name(node);
        seen_transformer = seen_transformer || is_transformer_stage_name(name);
        seen_result_norm = seen_result_norm || (name && std::strcmp(name, "result_norm") == 0);
        if (name && std::strcmp(name, "norm") == 0) {
            norm_node = node;
        }
    }

    if (!seen_transformer) {
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

    if (seen_result_norm && norm_node != nullptr) {
        for (size_t i = 0; i < GGML_MAX_SRC && norm_node->src[i]; ++i) {
            auto * src = norm_node->src[i];
            if (src != nullptr && (src->flags & GGML_TENSOR_FLAG_PARAM) == 0) {
                result.out = src;
                break;
            }
        }
    }

    if (result.out == nullptr && !io.outputs.empty()) {
        for (auto * output : io.outputs) {
            const char * name = ggml_get_name(output);
            if (is_transformer_stage_name(name) && has_prefix(name, "l_out-")) {
                result.out = output;
            }
        }
    }

    if (result.out == nullptr) {
        for (int i = cgraph->n_nodes - 1; i >= 0; --i) {
            auto * node = cgraph->nodes[i];
            if (is_transformer_output_candidate(node)) {
                result.out = node;
                break;
            }
        }
    }

    if (!result.embd || !result.out) {
        return result;
    }

    result.n_tokens       = static_cast<size_t>(result.embd->ne[1]);
    result.inferred_pos   = infer_start_pos(io.inputs, result.n_tokens);
    result.is_transformer = result.n_tokens > 0;
    return result;
}

qnn_aot_runtime::aot_match_result qnn_aot_runtime::match_lm_head_graph(ggml_cgraph * cgraph) const {
    aot_match_result result;
    if (!_enabled || _lm_head_graphs.empty() || cgraph == nullptr || cgraph->n_nodes == 0) {
        return result;
    }

    bool seen_result_output = false;
    ggml_tensor * norm_node          = nullptr;
    ggml_tensor * result_norm_node   = nullptr;
    ggml_tensor * result_output_node = nullptr;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        auto *       node = cgraph->nodes[i];
        const char * name = ggml_get_name(node);
        seen_result_output = seen_result_output || (name && std::strcmp(name, "result_output") == 0);
        if (name && std::strcmp(name, "norm") == 0) {
            norm_node = node;
        }
        if (name && std::strcmp(name, "result_norm") == 0) {
            result_norm_node = node;
        }
        if (name && std::strcmp(name, "result_output") == 0) {
            result_output_node = node;
        }
    }

    if (!seen_result_output) {
        return result;
    }

    auto is_activation_input = [this](const ggml_tensor * tensor) {
        if (tensor == nullptr || (tensor->flags & GGML_TENSOR_FLAG_PARAM) != 0) {
            return false;
        }

        const char * name = ggml_get_name(tensor);
        if (name && (std::strcmp(name, "norm") == 0 || has_prefix(name, "l_out-") ||
                     std::strcmp(name, "result_embd") == 0 || std::strcmp(name, "embd") == 0)) {
            return true;
        }

        return tensor->type == GGML_TYPE_F32 && static_cast<size_t>(tensor->ne[0]) == _config.model.embed_dim &&
               tensor->ne[1] > 0;
    };

    const auto io = get_io_tensors_from_graph(cgraph);
    if (norm_node != nullptr) {
        for (size_t i = 0; i < GGML_MAX_SRC && norm_node->src[i]; ++i) {
            auto * src = norm_node->src[i];
            if (is_activation_input(src)) {
                result.embd = src;
                break;
            }
        }
    }

    if (result.embd == nullptr && result_norm_node != nullptr) {
        for (size_t i = 0; i < GGML_MAX_SRC && result_norm_node->src[i]; ++i) {
            auto * src = result_norm_node->src[i];
            if (is_activation_input(src)) {
                result.embd = src;
                break;
            }
        }
    }

    if (result.embd == nullptr) {
        for (auto * input : io.inputs) {
            if (is_activation_input(input)) {
                result.embd = input;
                break;
            }
        }
    }

    if (result_output_node != nullptr) {
        result.out = result_output_node;
    }

    for (auto * output : io.outputs) {
        const char * name = ggml_get_name(output);
        if (name && std::strcmp(name, "result_output") == 0) {
            result.out = output;
            break;
        }
    }

    if (!result.embd || !result.out) {
        return result;
    }

    result.n_tokens   = static_cast<size_t>(result.embd->ne[1]);
    result.is_lm_head = result.n_tokens > 0;
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

qnn_aot_graph * qnn_aot_runtime::select_lm_head_graph(size_t n_tokens) const {
    if (_lm_head_graphs.empty()) {
        return nullptr;
    }

    size_t target_tokens = std::max<size_t>(1, n_tokens);
    auto it = _lm_head_graphs.lower_bound(target_tokens);

    if (it != _lm_head_graphs.end()) {
        return it->second.get();
    }

    return _lm_head_graphs.rbegin()->second.get();
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

    // Bug 4 fix: guard against underflow when _rope_embeds is empty (size()-1 wraps to SIZE_MAX).
    if (_rope_embeds.empty()) {
        QNN_LOG_WARN("[aot] fill_rope_embeds: _rope_embeds is empty, skipping\n");
        return;
    }

    const size_t row_elems  = _config.model.head_dim / 2;
    const size_t max_pos    = _rope_embeds.size() - 1;
    if (n_tokens < graph.batch_size()) {
        std::memset(cos_buffer, 0, graph.buffer_size("rope_embed_cos"));
        std::memset(sin_buffer, 0, graph.buffer_size("rope_embed_sin"));
    }

    for (size_t i = 0; i < n_tokens; ++i) {
        const size_t pos = std::min(start_pos + i, max_pos);
        std::memcpy(cos_buffer + i * row_elems, _rope_embeds[pos].cos_values.data(), row_elems * sizeof(float));
        std::memcpy(sin_buffer + i * row_elems, _rope_embeds[pos].sin_values.data(), row_elems * sizeof(float));
    }
}

void qnn_aot_runtime::fill_attention_bias(qnn_aot_graph & graph, size_t n_tokens) {
    void * attn_bias_raw = graph.buffer_data("attn_bias");
    if (!attn_bias_raw) {
        return;
    }

    const auto & graph_config  = graph.config();
    const size_t graph_batch   = graph.batch_size();
    const size_t context_size  = graph_config.context_size;
    const size_t cache_size    = graph_config.cache_size;
    const float  mask_valuef   = _config.model.attention_mask_value;

    // Bug 6 fix: check the actual QNN tensor data type rather than blindly casting to __fp16.
    const Qnn_DataType_t dtype = graph.tensor_data_type("attn_bias");

    auto fill_bias = [&](auto * bias, auto mask_val, auto zero_val) {
        std::fill(bias, bias + graph_batch * context_size, mask_val);
        for (size_t i = 0; i < n_tokens; ++i) {
            auto * row = bias + i * context_size;
            for (size_t j = 0; j < std::min(_kv_position, cache_size); ++j) {
                row[j] = zero_val;
            }
            for (size_t j = 0; j < n_tokens; ++j) {
                row[cache_size + j] = (j <= i) ? zero_val : mask_val;
            }
        }
    };

    if (dtype == QNN_DATATYPE_FLOAT_32) {
        fill_bias(static_cast<float *>(attn_bias_raw), mask_valuef, 0.0f);
    } else {
        if (dtype != QNN_DATATYPE_FLOAT_16 && dtype != QNN_DATATYPE_UNDEFINED) {
            QNN_LOG_WARN("[aot] fill_attention_bias: unexpected dtype %d for attn_bias, assuming fp16\n", (int) dtype);
        }
        fill_bias(static_cast<__fp16 *>(attn_bias_raw), (__fp16) mask_valuef, (__fp16) 0.0f);
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

            const size_t key_element_size   = graph.buffer_size(key_out_name) / (graph.batch_size() * head_dim);
            const size_t value_element_size = graph.buffer_size(value_out_name) / (graph.batch_size() * head_dim);
            const size_t key_cache_stride   = graph_config.cache_size * key_element_size;
            const size_t value_row_bytes    = head_dim * value_element_size;

            for (size_t token = 0; token < n_tokens; ++token) {
                // Bug 5 fix: guard against writing past the end of the KV cache.
                const size_t cache_slot = _kv_position + token;
                if (cache_slot >= graph_config.cache_size) {
                    QNN_LOG_WARN("[aot] save_kv: cache slot %zu >= cache_size %zu, dropping token %zu\n",
                                 cache_slot, graph_config.cache_size, token);
                    break;
                }
                copy_contiguous_to_strided(key_cache + cache_slot * key_element_size,
                                           key_cache_stride,
                                           key_out + token * head_dim * key_element_size,
                                           head_dim,
                                           key_element_size);
                std::memcpy(value_cache + cache_slot * value_row_bytes,
                            value_out + token * value_row_bytes,
                            value_row_bytes);
            }
        }
    }
}

std::string qnn_aot_runtime::format_kv_path(const qnn_aot_graph_config & config, size_t layer_id, const char * kv_type, size_t head_id) const {
    std::string relative = config.kv_path_format;
    replace_all(relative, "{layer_id}", std::to_string(layer_id));
    replace_all(relative, "{kv_type}", kv_type ? kv_type : "");
    replace_all(relative, "{head_id}", std::to_string(head_id));
    return (std::filesystem::path(_model_dir) / relative).string();
}

bool qnn_aot_runtime::load_seed_kv() {
    _seed_kv_loaded = false;

    if (_seed_kv_size == 0) {
        return true;
    }

    auto convert_and_store = [](char * dst,
                                size_t elem_size,
                                Qnn_DataType_t dtype,
                                const float * src,
                                size_t n_values,
                                size_t dst_stride) -> bool {
        switch (dtype) {
            case QNN_DATATYPE_FLOAT_16:
            case QNN_DATATYPE_UNDEFINED: {
                auto * dst_bytes = dst;
                for (size_t i = 0; i < n_values; ++i) {
                    *reinterpret_cast<ggml_fp16_t *>(dst_bytes) = ggml_fp32_to_fp16(src[i]);
                    dst_bytes += dst_stride;
                }
                return true;
            }
            case QNN_DATATYPE_FLOAT_32: {
                auto * dst_bytes = dst;
                for (size_t i = 0; i < n_values; ++i) {
                    *reinterpret_cast<float *>(dst_bytes) = src[i];
                    dst_bytes += dst_stride;
                }
                return true;
            }
            default:
                return false;
        }
    };

    for (const auto & graph_it : _transformer_graphs) {
        auto & graph = *graph_it.second;
        const auto & graph_config = graph.config();
        if (graph_config.kv_size == 0) {
            continue;
        }

        for (size_t layer = graph_config.start_layer_id; layer < graph_config.end_layer_id; ++layer) {
            for (size_t head = 0; head < _config.model.n_kv_heads; ++head) {
                const auto key_cache_name   = std::string("layer_") + std::to_string(layer) + "_key_t_cache_" + std::to_string(head);
                const auto value_cache_name = std::string("layer_") + std::to_string(layer) + "_value_cache_" + std::to_string(head);

                auto * key_cache   = static_cast<char *>(graph.buffer_data(key_cache_name));
                auto * value_cache = static_cast<char *>(graph.buffer_data(value_cache_name));
                if (!key_cache || !value_cache) {
                    QNN_LOG_WARN("[aot] missing seed KV cache buffers for layer=%zu head=%zu\n", layer, head);
                    return false;
                }

                const size_t head_dim = _config.model.head_dim;
                const size_t key_elem_size = graph.buffer_size(key_cache_name) / (graph_config.cache_size * head_dim);
                const size_t value_elem_size = graph.buffer_size(value_cache_name) / (graph_config.cache_size * head_dim);
                const auto key_dtype = graph.tensor_data_type(key_cache_name);
                const auto value_dtype = graph.tensor_data_type(value_cache_name);

                const auto key_path = format_kv_path(graph_config, layer, "key", head);
                const auto value_path = format_kv_path(graph_config, layer, "value", head);
                mapped_file key_file(key_path);
                mapped_file value_file(value_path);

                const size_t expected_bytes = graph_config.kv_size * head_dim * sizeof(float);
                if (!key_file.is_valid() || key_file.size < expected_bytes || !value_file.is_valid() || value_file.size < expected_bytes) {
                    if (!_seed_kv_missing_warned) {
                        QNN_LOG_WARN("[aot] seed KV files missing or truncated for config %s (need kv_size=%zu). "
                                     "Expected files like %s and %s. Runtime will continue from an empty cache, "
                                     "which does not match PowerServe artifacts.\n",
                                     _config_path.c_str(),
                                     _seed_kv_size,
                                     key_path.c_str(),
                                     value_path.c_str());
                        _seed_kv_missing_warned = true;
                    }
                    return false;
                }

                const auto * key_src = static_cast<const float *>(key_file.data);
                const auto * value_src = static_cast<const float *>(value_file.data);

                const size_t key_dst_stride = graph_config.cache_size * key_elem_size;
                const size_t value_row_bytes = head_dim * value_elem_size;
                for (size_t token = 0; token < graph_config.kv_size; ++token) {
                    if (!convert_and_store(key_cache + token * key_elem_size,
                                           key_elem_size,
                                           key_dtype,
                                           key_src + token * head_dim,
                                           head_dim,
                                           key_dst_stride)) {
                        QNN_LOG_WARN("[aot] unsupported key cache dtype %d for layer=%zu head=%zu\n",
                                     (int) key_dtype, layer, head);
                        return false;
                    }

                    if (!convert_and_store(value_cache + token * value_row_bytes,
                                           value_elem_size,
                                           value_dtype,
                                           value_src + token * head_dim,
                                           head_dim,
                                           value_elem_size)) {
                        QNN_LOG_WARN("[aot] unsupported value cache dtype %d for layer=%zu head=%zu\n",
                                     (int) value_dtype, layer, head);
                        return false;
                    }
                }
            }
        }
    }

    _seed_kv_loaded = true;
    return true;
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

bool qnn_aot_runtime::is_transformer_output_candidate(const ggml_tensor * tensor) const {
    if (tensor == nullptr || (tensor->flags & GGML_TENSOR_FLAG_PARAM) != 0) {
        return false;
    }

    if (tensor->type != GGML_TYPE_F32 || ggml_n_dims(tensor) < 1) {
        return false;
    }

    if (static_cast<size_t>(tensor->ne[0]) != _config.model.embed_dim) {
        return false;
    }

    if (ggml_n_dims(tensor) == 1) {
        return true;
    }

    return tensor->ne[1] > 0;
}

bool qnn_aot_runtime::execute_transformer(ggml_cgraph * cgraph, const aot_match_result & match) {
    GGML_UNUSED(cgraph);
    auto * graph = select_transformer_graph(match.n_tokens);
    if (!graph || !match.embd || !match.out) {
        return false;
    }

    if ((match.inferred_pos == 0 && _kv_position != 0) ||
        (_kv_position + match.n_tokens > graph->config().cache_size && match.inferred_pos == 0)) {
        reset_state();
    }

    if (match.embd->type != GGML_TYPE_F32 || match.out->type != GGML_TYPE_F32) {
        QNN_LOG_WARN("[aot] transformer IO expects F32 tensors, got %s -> %s\n", ggml_type_name(match.embd->type),
                     ggml_type_name(match.out->type));
        return false;
    }

    const size_t embed_dim = _config.model.embed_dim;
    if (static_cast<size_t>(match.embd->ne[0]) != embed_dim || static_cast<size_t>(match.out->ne[0]) != embed_dim) {
        QNN_LOG_WARN("[aot] transformer IO dim mismatch, expected %zu, got %lld -> %lld\n", embed_dim,
                     (long long) match.embd->ne[0], (long long) match.out->ne[0]);
        return false;
    }

    auto * x_buffer   = static_cast<float *>(graph->buffer_data(graph->config().x_name));
    auto * out_buffer = static_cast<float *>(graph->buffer_data(graph->config().out_name));
    if (!x_buffer || !out_buffer) {
        QNN_LOG_WARN("[aot] missing transformer x/out buffers\n");
        return false;
    }

    size_t offset = 0;
    while (offset < match.n_tokens) {
        const size_t step = std::min(match.n_tokens - offset, graph->batch_size());

        if (step < graph->batch_size()) {
            std::memset(x_buffer, 0, graph->buffer_size(graph->config().x_name));
        }

        copy_ggml_rows_to_contiguous(match.embd, offset, step, embed_dim, x_buffer);

        fill_rope_embeds(*graph, _kv_position, step);

        fill_attention_bias(*graph, step);

        if (!graph->execute()) {
            return false;
        }

        copy_contiguous_rows_to_ggml(match.out, offset, step, embed_dim, out_buffer);

        save_kv(*graph, step);

        _kv_position += step;
        offset += step;
    }

    return true;
}

bool qnn_aot_runtime::execute_lm_head(ggml_cgraph * cgraph, const aot_match_result & match) {
    GGML_UNUSED(cgraph);
    auto * graph = select_lm_head_graph(match.n_tokens);
    if (!graph || !match.embd || !match.out) {
        return false;
    }

    if (match.embd->type != GGML_TYPE_F32 || match.out->type != GGML_TYPE_F32) {
        QNN_LOG_WARN("[aot] lm_head IO expects F32 tensors, got %s -> %s\n", ggml_type_name(match.embd->type),
                     ggml_type_name(match.out->type));
        return false;
    }

    const size_t embed_dim  = _config.model.embed_dim;
    const size_t vocab_size = _config.model.vocab_size;
    if (static_cast<size_t>(match.embd->ne[0]) != embed_dim || static_cast<size_t>(match.out->ne[0]) != vocab_size) {
        QNN_LOG_WARN("[aot] lm_head IO dim mismatch, expected %zu -> %zu, got %lld -> %lld\n",
                     embed_dim, vocab_size,
                     (long long) match.embd->ne[0], (long long) match.out->ne[0]);
        return false;
    }

    auto * x_buffer   = static_cast<float *>(graph->buffer_data(graph->config().x_name));
    auto * out_buffer = static_cast<float *>(graph->buffer_data(graph->config().out_name));
    if (!x_buffer || !out_buffer) {
        QNN_LOG_WARN("[aot] missing lm_head x/out buffers\n");
        return false;
    }

    size_t offset = 0;
    while (offset < match.n_tokens) {
        const size_t step = std::min(match.n_tokens - offset, graph->batch_size());

        if (step < graph->batch_size()) {
            std::memset(x_buffer, 0, graph->buffer_size(graph->config().x_name));
        }

        copy_ggml_rows_to_contiguous(match.embd, offset, step, embed_dim, x_buffer);

        if (!graph->execute()) {
            return false;
        }

        copy_contiguous_rows_to_ggml(match.out, offset, step, vocab_size, out_buffer);
        offset += step;
    }

    return true;
}

bool qnn_aot_runtime::maybe_execute(ggml_cgraph * cgraph) {
    struct combined_match_result {
        aot_match_result transformer;
        aot_match_result lm_head;
        bool is_combined = false;
    };

    auto match_combined_graph = [this](ggml_cgraph * graph) -> combined_match_result {
        combined_match_result combined;
        if (!_enabled || _lm_head_graphs.empty() || graph == nullptr || graph->n_nodes == 0) {
            return combined;
        }

        bool          seen_transformer   = false;
        bool          seen_result_norm   = false;
        bool          seen_result_output = false;
        ggml_tensor * norm_node          = nullptr;
        ggml_tensor * result_norm_node   = nullptr;
        ggml_tensor * result_output_node = nullptr;

        for (int i = 0; i < graph->n_nodes; ++i) {
            auto *       node = graph->nodes[i];
            const char * name = ggml_get_name(node);
            seen_transformer   = seen_transformer || is_transformer_stage_name(name);
            seen_result_output = seen_result_output || (name && std::strcmp(name, "result_output") == 0);
            seen_result_norm   = seen_result_norm || (name && std::strcmp(name, "result_norm") == 0);

            if (name && std::strcmp(name, "norm") == 0) {
                norm_node = node;
            }
            if (name && std::strcmp(name, "result_norm") == 0) {
                result_norm_node = node;
            }
            if (name && std::strcmp(name, "result_output") == 0) {
                result_output_node = node;
            }
        }

        if (!seen_transformer || !seen_result_output) {
            return combined;
        }

        const auto io = get_io_tensors_from_graph(graph);
        for (auto * input : io.inputs) {
            const char * name = ggml_get_name(input);
            if (name && std::strcmp(name, "embd") == 0) {
                combined.transformer.embd = input;
                break;
            }
        }

        if (seen_result_norm && norm_node != nullptr) {
            for (size_t i = 0; i < GGML_MAX_SRC && norm_node->src[i]; ++i) {
                auto * src = norm_node->src[i];
                if (src != nullptr && (src->flags & GGML_TENSOR_FLAG_PARAM) == 0) {
                    combined.transformer.out = src;
                    break;
                }
            }
        }

        if (combined.transformer.out == nullptr && !io.outputs.empty()) {
            for (auto * output : io.outputs) {
                const char * name = ggml_get_name(output);
                if (is_transformer_stage_name(name) && has_prefix(name, "l_out-")) {
                    combined.transformer.out = output;
                }
            }
        }

        auto is_activation_input = [this](const ggml_tensor * tensor) {
            if (tensor == nullptr || (tensor->flags & GGML_TENSOR_FLAG_PARAM) != 0) {
                return false;
            }

            const char * name = ggml_get_name(tensor);
            if (name && (std::strcmp(name, "norm") == 0 || has_prefix(name, "l_out-") ||
                         std::strcmp(name, "result_embd") == 0 || std::strcmp(name, "embd") == 0)) {
                return true;
            }

            return tensor->type == GGML_TYPE_F32 && static_cast<size_t>(tensor->ne[0]) == _config.model.embed_dim &&
                   tensor->ne[1] > 0;
        };

        if (norm_node != nullptr) {
            for (size_t i = 0; i < GGML_MAX_SRC && norm_node->src[i]; ++i) {
                auto * src = norm_node->src[i];
                if (is_activation_input(src)) {
                    combined.lm_head.embd = src;
                    break;
                }
            }
        }

        if (combined.lm_head.embd == nullptr && result_norm_node != nullptr) {
            for (size_t i = 0; i < GGML_MAX_SRC && result_norm_node->src[i]; ++i) {
                auto * src = result_norm_node->src[i];
                if (is_activation_input(src)) {
                    combined.lm_head.embd = src;
                    break;
                }
            }
        }

        if (combined.lm_head.embd == nullptr) {
            for (auto * input : io.inputs) {
                if (is_activation_input(input)) {
                    combined.lm_head.embd = input;
                    break;
                }
            }
        }

        if (result_output_node != nullptr) {
            combined.lm_head.out = result_output_node;
        }

        for (auto * output : io.outputs) {
            const char * name = ggml_get_name(output);
            if (name && std::strcmp(name, "result_output") == 0) {
                combined.lm_head.out = output;
                break;
            }
        }

        if (!combined.transformer.embd || !combined.transformer.out || !combined.lm_head.embd || !combined.lm_head.out) {
            return combined;
        }

        combined.transformer.n_tokens       = static_cast<size_t>(combined.transformer.embd->ne[1]);
        combined.transformer.inferred_pos   = infer_start_pos(io.inputs, combined.transformer.n_tokens);
        combined.transformer.is_transformer = combined.transformer.n_tokens > 0;

        combined.lm_head.n_tokens   = static_cast<size_t>(combined.lm_head.embd->ne[1]);
        combined.lm_head.is_lm_head = combined.lm_head.n_tokens > 0;

        combined.is_combined = combined.transformer.is_transformer && combined.lm_head.is_lm_head;
        return combined;
    };

    const auto combined_match = match_combined_graph(cgraph);

    if (combined_match.is_combined) {
        const auto & transformer_match = combined_match.transformer;
        const auto & lm_head_match     = combined_match.lm_head;
        if (!execute_transformer(cgraph, transformer_match)) {
            return false;
        }
        return execute_lm_head(cgraph, lm_head_match);
    }

    const auto transformer_match = match_transformer_graph(cgraph);
    const auto lm_head_match     = match_lm_head_graph(cgraph);

    if (transformer_match.is_transformer) {
        return execute_transformer(cgraph, transformer_match);
    }

    if (lm_head_match.is_lm_head) {
        return execute_lm_head(cgraph, lm_head_match);
    }

    return false;
}

void qnn_aot_runtime::reset_state() {
    zero_transformer_state();
    _kv_position = 0;
    if (_seed_kv_size > 0 && load_seed_kv()) {
        _kv_position = _seed_kv_size;
    }
}

}  // namespace qnn
