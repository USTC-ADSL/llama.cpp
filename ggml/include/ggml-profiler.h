#pragma once

//
// GGML Stage Profiler
//
// This header provides utilities for classifying tensor operations into
// computational stages (FFN, Attention, etc.) for low-overhead profiling.
//
// The stage classifier analyzes tensor names to determine which computational
// stage they belong to, enabling aggregated performance analysis without
// the overhead of callback-based profiling.
//

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

// Include string.h for strstr - must be before extern "C" for C compatibility
#ifdef __cplusplus
#include <cstring>
#else
#include <string.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

//
// Stage type enumeration
//
// Defines the computational stages in a typical transformer model.
// These stages are used to aggregate profiling data for analysis.
//
enum ggml_stage_type {
    GGML_STAGE_FFN,           // Feed-forward network (up_proj, down_proj, gate_proj, mlp)
    GGML_STAGE_ATTN_PROJ,     // Attention projections (q_proj, k_proj, v_proj, o_proj)
    GGML_STAGE_ATTN_CORE,     // Attention core (RoPE, KV cache, score, softmax, flash_attn)
    GGML_STAGE_OUTPUT_PROJ,   // Output projection (lm_head, output)
    GGML_STAGE_NORM,          // Normalization layers (layernorm, rmsnorm)
    GGML_STAGE_EMBEDDING,     // Embedding layers (token_embd, position_embd)
    GGML_STAGE_OTHER,         // Other operations
    GGML_STAGE_COUNT          // Number of stages (for array sizing)
};

//
// ggml_classify_stage - Classify a tensor into a computational stage
//
// Parameters:
//   tensor_name - The name of the tensor to classify (can be NULL)
//
// Returns:
//   The stage type that the tensor belongs to.
//   Returns GGML_STAGE_OTHER if the name is NULL or doesn't match any pattern.
//
// The classification is based on common naming patterns in transformer models:
//   - FFN: contains "ffn", "mlp", "up_proj", "down_proj", "gate", "w1", "w2", "w3"
//   - ATTN_PROJ: contains "attn" with "q", "k", "v", "o" or "*_proj"
//   - ATTN_CORE: contains "rope", "kv", "score", "softmax", "flash"
//   - OUTPUT_PROJ: contains "output", "lm_head"
//   - NORM: contains "norm", "_ln"
//   - EMBEDDING: contains "embed", "token_embd", "wte", "wpe"
//
static inline enum ggml_stage_type ggml_classify_stage(const char * tensor_name) {
    if (tensor_name == NULL || tensor_name[0] == '\0') {
        return GGML_STAGE_OTHER;
    }

    // Helper function to check if string contains substring
    #define CONTAINS(haystack, needle) (strstr(haystack, needle) != NULL)
    
    // Helper to check if string starts with prefix
    #define STARTS_WITH(str, prefix) (strncmp(str, prefix, strlen(prefix)) == 0)

    // FFN patterns - check first as they're most common
    // Matches: ffn_*, mlp*, up_proj, down_proj, gate_proj, feed_forward, .w1/.w2/.w3
    if (CONTAINS(tensor_name, "ffn") ||
        CONTAINS(tensor_name, "mlp") ||
        CONTAINS(tensor_name, "up_proj") ||
        CONTAINS(tensor_name, "down_proj") ||
        CONTAINS(tensor_name, "gate_proj") ||
        CONTAINS(tensor_name, "gate.") ||
        CONTAINS(tensor_name, ".w1") ||
        CONTAINS(tensor_name, ".w2") ||
        CONTAINS(tensor_name, ".w3") ||
        CONTAINS(tensor_name, "feed_forward")) {
        return GGML_STAGE_FFN;
    }

    // Attention projection patterns
    // Matches: Qcur*, Kcur*, Vcur* (intermediate Q/K/V tensors from llama.cpp)
    // Also matches: attn_q, attn_k, attn_v, attn_output, *_proj patterns
    if (STARTS_WITH(tensor_name, "Qcur") ||
        STARTS_WITH(tensor_name, "Kcur") ||
        STARTS_WITH(tensor_name, "Vcur")) {
        return GGML_STAGE_ATTN_PROJ;
    }
    
    // Attention projection - weight tensor patterns
    if ((CONTAINS(tensor_name, "attn") || CONTAINS(tensor_name, "attention") || CONTAINS(tensor_name, "self_attn")) &&
        (CONTAINS(tensor_name, "_q") || CONTAINS(tensor_name, ".q") ||
         CONTAINS(tensor_name, "_k") || CONTAINS(tensor_name, ".k") ||
         CONTAINS(tensor_name, "_v") || CONTAINS(tensor_name, ".v") ||
         CONTAINS(tensor_name, "_o") || CONTAINS(tensor_name, ".o") ||
         CONTAINS(tensor_name, "q_proj") || CONTAINS(tensor_name, "k_proj") ||
         CONTAINS(tensor_name, "v_proj") || CONTAINS(tensor_name, "o_proj") ||
         CONTAINS(tensor_name, "qkv") || CONTAINS(tensor_name, "c_attn") ||
         CONTAINS(tensor_name, "c_proj") || CONTAINS(tensor_name, "out_proj") ||
         CONTAINS(tensor_name, "attn_output"))) {
        return GGML_STAGE_ATTN_PROJ;
    }

    // Attention core patterns (KV cache, attention computation, RoPE)
    // Matches: cache_k_l*, cache_v_l*, kv_cache, kq-*, kqv*, v_cont*, KQ*, softmax, rope
    if (CONTAINS(tensor_name, "cache_k") ||
        CONTAINS(tensor_name, "cache_v") ||
        CONTAINS(tensor_name, "kv_cache") ||
        STARTS_WITH(tensor_name, "kq-") ||
        STARTS_WITH(tensor_name, "kq_") ||
        STARTS_WITH(tensor_name, "kqv") ||
        STARTS_WITH(tensor_name, "v_cont") ||
        CONTAINS(tensor_name, "KQ") ||
        CONTAINS(tensor_name, "score") ||
        CONTAINS(tensor_name, "softmax") ||
        CONTAINS(tensor_name, "flash_attn") ||
        CONTAINS(tensor_name, "flash-attn") ||
        CONTAINS(tensor_name, "rope") ||
        CONTAINS(tensor_name, "rotary")) {
        return GGML_STAGE_ATTN_CORE;
    }

    // Output projection patterns
    if (CONTAINS(tensor_name, "lm_head") ||
        CONTAINS(tensor_name, "result_output") ||
        CONTAINS(tensor_name, "output.weight") ||
        CONTAINS(tensor_name, "output_proj") ||
        CONTAINS(tensor_name, "classifier") ||
        CONTAINS(tensor_name, "lm_proj")) {
        return GGML_STAGE_OUTPUT_PROJ;
    }

    // Normalization patterns
    // Note: Check after FFN to avoid matching ffn_norm as just NORM
    if (CONTAINS(tensor_name, "norm") ||
        CONTAINS(tensor_name, "_ln") ||
        CONTAINS(tensor_name, "layernorm") ||
        CONTAINS(tensor_name, "layer_norm") ||
        CONTAINS(tensor_name, "rms_norm") ||
        CONTAINS(tensor_name, "rmsnorm")) {
        return GGML_STAGE_NORM;
    }

    // Embedding patterns
    // Matches: inp_embd, token_embd, embed*, wte, wpe
    if (CONTAINS(tensor_name, "embd") ||
        CONTAINS(tensor_name, "embed") ||
        CONTAINS(tensor_name, "wte") ||
        CONTAINS(tensor_name, "wpe") ||
        CONTAINS(tensor_name, "inp_tokens")) {
        return GGML_STAGE_EMBEDDING;
    }

    #undef CONTAINS
    #undef STARTS_WITH

    return GGML_STAGE_OTHER;
}

//
// ggml_stage_name - Get the human-readable name of a stage
//
// Parameters:
//   stage - The stage type
//
// Returns:
//   A string representation of the stage name.
//   Returns "OTHER" for invalid stage values.
//
static inline const char * ggml_stage_name(enum ggml_stage_type stage) {
    static const char * const names[] = {
        "FFN",
        "ATTN_PROJ",
        "ATTN_CORE",
        "OUTPUT_PROJ",
        "NORM",
        "EMBEDDING",
        "OTHER"
    };

    if (stage >= 0 && stage < GGML_STAGE_COUNT) {
        return names[stage];
    }
    return names[GGML_STAGE_OTHER];
}

//
// ggml_extract_layer_id - Extract the layer number from a tensor name
//
// Parameters:
//   tensor_name - The name of the tensor (can be NULL)
//
// Returns:
//   The layer number if found, or -1 if not found or invalid.
//
// Common patterns supported:
//   - "blk.N.*" -> N (llama.cpp style)
//   - "layers.N.*" -> N (HuggingFace style)
//   - "h.N.*" -> N (GPT-2 style)
//   - "layer_N" or "layer.N" -> N
//   - "block_N" or "block.N" -> N
//   - "*-N" (suffix style, e.g. "Qcur-0")
//   - "*_lN*" (cache style, e.g. "cache_k_l0")
//
static inline int ggml_extract_layer_id(const char * tensor_name) {
    if (tensor_name == NULL || tensor_name[0] == '\0') {
        return -1;
    }

    const char * p = tensor_name;

    // Look for common layer patterns
    // Pattern: "blk.N." (llama.cpp style)
    const char * blk = strstr(p, "blk.");
    if (blk != NULL) {
        blk += 4;  // Skip "blk."
        int layer = 0;
        while (*blk >= '0' && *blk <= '9') {
            layer = layer * 10 + (*blk - '0');
            blk++;
        }
        if (*blk == '.' || *blk == '_') {
            return layer;
        }
    }

    // Pattern: "layers.N." (HuggingFace style)
    const char * layers = strstr(p, "layers.");
    if (layers != NULL) {
        layers += 7;  // Skip "layers."
        int layer = 0;
        while (*layers >= '0' && *layers <= '9') {
            layer = layer * 10 + (*layers - '0');
            layers++;
        }
        if (*layers == '.' || *layers == '_' || *layers == '\0') {
            return layer;
        }
    }

    // Pattern: "h.N." (GPT-2 style)
    const char * h = strstr(p, "h.");
    if (h != NULL) {
        h += 2;  // Skip "h."
        int layer = 0;
        while (*h >= '0' && *h <= '9') {
            layer = layer * 10 + (*h - '0');
            h++;
        }
        if (*h == '.' || *h == '_') {
            return layer;
        }
    }

    // Pattern: "layer_N" or "layer.N"
    const char * layer_prefix = strstr(p, "layer");
    if (layer_prefix != NULL) {
        layer_prefix += 5;  // Skip "layer"
        if (*layer_prefix == '_' || *layer_prefix == '.') {
            layer_prefix++;
            int layer = 0;
            bool found_digit = false;
            while (*layer_prefix >= '0' && *layer_prefix <= '9') {
                layer = layer * 10 + (*layer_prefix - '0');
                layer_prefix++;
                found_digit = true;
            }
            if (found_digit) {
                return layer;
            }
        }
    }

    // Pattern: "block_N" or "block.N"
    const char * block_prefix = strstr(p, "block");
    if (block_prefix != NULL) {
        block_prefix += 5;  // Skip "block"
        if (*block_prefix == '_' || *block_prefix == '.') {
            block_prefix++;
            int layer = 0;
            bool found_digit = false;
            while (*block_prefix >= '0' && *block_prefix <= '9') {
                layer = layer * 10 + (*block_prefix - '0');
                block_prefix++;
                found_digit = true;
            }
            if (found_digit) {
                return layer;
            }
        }
    }

    // Pattern: suffix "-N" (e.g. "Qcur-0", "ffn_inp-31")
    const char * dash = strrchr(p, '-');
    if (dash != NULL && *(dash + 1) != '\0') {
        const char * q = dash + 1;
        int layer = 0;
        bool found_digit = false;
        while (*q >= '0' && *q <= '9') {
            layer = layer * 10 + (*q - '0');
            q++;
            found_digit = true;
        }
        if (found_digit && *q == '\0') {
            return layer;
        }
    }

    // Pattern: "_lN" (e.g. "cache_k_l0", "cache_v_l12 (view)")
    const char * lpat = p;
    while ((lpat = strstr(lpat, "_l")) != NULL) {
        lpat += 2; // skip "_l"
        int layer = 0;
        bool found_digit = false;
        while (*lpat >= '0' && *lpat <= '9') {
            layer = layer * 10 + (*lpat - '0');
            lpat++;
            found_digit = true;
        }
        if (found_digit && (*lpat == '\0' || *lpat == ' ' || *lpat == '.' || *lpat == '_' || *lpat == '(' || *lpat == '[')) {
            return layer;
        }
    }

    return -1;
}

//
// Stage statistics structure for aggregating profiling data
//
struct ggml_stage_stats {
    uint64_t total_time_ns;    // Total time in nanoseconds
    uint64_t count;            // Number of operations
    uint64_t min_time_ns;      // Minimum operation time
    uint64_t max_time_ns;      // Maximum operation time
};

//
// ggml_stage_stats_init - Initialize stage statistics
//
static inline void ggml_stage_stats_init(struct ggml_stage_stats * stats) {
    stats->total_time_ns = 0;
    stats->count = 0;
    stats->min_time_ns = UINT64_MAX;
    stats->max_time_ns = 0;
}

//
// ggml_stage_stats_update - Update stage statistics with a new measurement
//
static inline void ggml_stage_stats_update(struct ggml_stage_stats * stats, uint64_t time_ns) {
    stats->total_time_ns += time_ns;
    stats->count++;
    if (time_ns < stats->min_time_ns) {
        stats->min_time_ns = time_ns;
    }
    if (time_ns > stats->max_time_ns) {
        stats->max_time_ns = time_ns;
    }
}

//
// ggml_stage_stats_avg_ns - Get average time in nanoseconds
//
static inline uint64_t ggml_stage_stats_avg_ns(const struct ggml_stage_stats * stats) {
    if (stats->count == 0) {
        return 0;
    }
    return stats->total_time_ns / stats->count;
}

#ifdef __cplusplus
}
#endif
