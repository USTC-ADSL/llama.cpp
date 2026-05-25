#pragma once

namespace qnn {

enum class qnn_aot_stage_fragment_plan {
    unsupported,
    full_attention_then_ffn,
    split_projection_core_ffn,
};

inline bool qnn_aot_should_prefer_cpu_for_ffn_only_boundary(
        bool has_transformer_graphs,
        bool has_attention_graphs,
        bool has_attn_proj_graphs,
        bool has_attn_core_graphs,
        bool has_ffn_graphs) {
    return !has_transformer_graphs &&
           !has_attention_graphs &&
           !has_attn_proj_graphs &&
           !has_attn_core_graphs &&
           has_ffn_graphs;
}

inline qnn_aot_stage_fragment_plan qnn_aot_select_stage_fragment_plan(
        bool has_attention_graphs,
        bool has_attn_proj_graphs,
        bool has_attn_core_graphs,
        bool has_ffn_graphs) {
    if (!has_ffn_graphs) {
        return qnn_aot_stage_fragment_plan::unsupported;
    }

    if (has_attention_graphs && has_attn_proj_graphs) {
        return qnn_aot_stage_fragment_plan::full_attention_then_ffn;
    }

    if (has_attn_proj_graphs && has_attn_core_graphs) {
        return qnn_aot_stage_fragment_plan::split_projection_core_ffn;
    }

    return qnn_aot_stage_fragment_plan::unsupported;
}

inline bool qnn_aot_can_decompose_stage_fragments(
        bool has_transformer_graphs,
        bool has_attention_graphs,
        bool has_attn_proj_graphs,
        bool has_attn_core_graphs,
        bool has_ffn_graphs) {
    return !has_transformer_graphs &&
           qnn_aot_select_stage_fragment_plan(
                   has_attention_graphs,
                   has_attn_proj_graphs,
                   has_attn_core_graphs,
                   has_ffn_graphs) != qnn_aot_stage_fragment_plan::unsupported;
}

inline bool qnn_aot_can_execute_adjacent_stage_sequence(
        bool has_transformer_graphs,
        bool has_attention_graphs,
        bool has_attn_proj_graphs,
        bool /*has_attn_core_graphs*/,
        bool has_ffn_graphs) {
    return !has_transformer_graphs &&
           has_ffn_graphs &&
           (has_attention_graphs || has_attn_proj_graphs);
}

}  // namespace qnn
