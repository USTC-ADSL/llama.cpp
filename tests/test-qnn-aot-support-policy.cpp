#include "../ggml/src/ggml-qnn/qnn/aot-support-policy.hpp"

#include <iostream>

int main() {
    if (!qnn::qnn_aot_should_prefer_cpu_for_ffn_only_boundary(
                false, false, false, false, true)) {
        std::cerr << "ffn-only artifact should prefer CPU for FFN boundary glue\n";
        return 1;
    }

    if (qnn::qnn_aot_should_prefer_cpu_for_ffn_only_boundary(
                false, true, true, false, true)) {
        std::cerr << "stage attention+projection+ffn artifact was misclassified as ffn-only\n";
        return 1;
    }

    if (qnn::qnn_aot_should_prefer_cpu_for_ffn_only_boundary(
                true, false, false, false, true)) {
        std::cerr << "transformer artifact should not use the ffn-only CPU boundary policy\n";
        return 1;
    }

    if (!qnn::qnn_aot_can_decompose_stage_fragments(
                false, true, true, false, true)) {
        std::cerr << "attn_proj+attention+ffn artifact should be decomposable into stage fragments\n";
        return 1;
    }
    if (qnn::qnn_aot_select_stage_fragment_plan(
                true, true, false, true) != qnn::qnn_aot_stage_fragment_plan::full_attention_then_ffn) {
        std::cerr << "attn_proj+attention+ffn artifact should use the full attention graph, not split projection/core\n";
        return 1;
    }

    if (!qnn::qnn_aot_can_decompose_stage_fragments(
                false, false, true, true, true)) {
        std::cerr << "attn_proj+attn_core+ffn artifact should be decomposable into stage fragments\n";
        return 1;
    }
    if (qnn::qnn_aot_select_stage_fragment_plan(
                false, true, true, true) != qnn::qnn_aot_stage_fragment_plan::split_projection_core_ffn) {
        std::cerr << "attn_proj+attn_core+ffn artifact should use projection/core/ffn fragments\n";
        return 1;
    }

    if (qnn::qnn_aot_can_decompose_stage_fragments(
                true, false, true, true, true)) {
        std::cerr << "full transformer artifact should not use stage-fragment decomposition\n";
        return 1;
    }

    if (qnn::qnn_aot_can_decompose_stage_fragments(
                false, true, false, false, true)) {
        std::cerr << "artifact without projection graph should not use stage-fragment decomposition\n";
        return 1;
    }

    if (!qnn::qnn_aot_can_execute_adjacent_stage_sequence(
                false, false, true, false, true)) {
        std::cerr << "attn_proj+ffn artifact should allow adjacent mixed QNN stage sequence\n";
        return 1;
    }

    if (qnn::qnn_aot_can_execute_adjacent_stage_sequence(
                true, false, true, false, true)) {
        std::cerr << "full transformer artifact should not use adjacent mixed stage sequence fallback\n";
        return 1;
    }

    if (qnn::qnn_aot_can_execute_adjacent_stage_sequence(
                false, false, false, false, true)) {
        std::cerr << "ffn-only artifact should not use adjacent mixed stage sequence fallback\n";
        return 1;
    }

    return 0;
}
