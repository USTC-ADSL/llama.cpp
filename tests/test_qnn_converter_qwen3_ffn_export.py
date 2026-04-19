import sys
import tempfile
import unittest
from pathlib import Path

import onnx
import onnx.numpy_helper
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
QNN_CONVERTER_DIR = REPO_ROOT / "ref" / "PowerServe" / "tools" / "qnn_converter"
sys.path.insert(0, str(QNN_CONVERTER_DIR))

from export_to_onnx import LlamaModelChunk  # noqa: E402
from llama_model import LlamaRMSNorm, RMSNORM_POW_INPUT_PRESCALE  # noqa: E402
from model_params import Qwen3_1_7B_Params  # noqa: E402


class Qwen3ExportStructureTest(unittest.TestCase):
    @staticmethod
    def _get_constant_scalar(node: onnx.NodeProto) -> float:
        for attr in node.attribute:
            if attr.name == "value":
                return float(onnx.numpy_helper.to_array(attr.t).reshape(()))
        raise AssertionError(f"Expected Constant node, got {node.op_type}")

    def test_qwen3_1p7b_uses_full_fp16_attention_structure(self):
        params = Qwen3_1_7B_Params()
        chunk = LlamaModelChunk(
            start_layer_id=0,
            end_layer_id=1,
            embed_dim=params.embed_dim,
            n_heads=params.n_heads,
            n_kv_heads=params.n_kv_heads,
            context_size=16,
            batch_size=1,
            ffn_hidden_dim=params.ffn_hidden_dim,
            rms_norm_eps=params.rms_norm_eps,
            has_qkv_bias=params.has_qkv_bias,
            use_qk_norm=params.use_qk_norm,
            use_drelu=params.use_drelu,
            cache_size=0,
            n_fp16_heads=params.n_fp16_heads,
            n_fp16_neurons=params.n_fp16_neurons,
            stat_folder=REPO_ROOT / "tmp" / "missing-qnn-stats",
        )

        attn = chunk.layers[0].attn
        self.assertEqual(params.fp16_attention_layers, list(range(params.n_layers)))
        self.assertEqual(len(attn.fp16_head_ids), params.n_heads)
        self.assertEqual(len(attn.int4_head_ids), 0)
        self.assertTrue(hasattr(attn, "fp16_o_proj"))
        self.assertFalse(hasattr(attn, "int4_o_proj"))

    def test_qwen3_1p7b_uses_full_fp16_ffn_structure(self):
        params = Qwen3_1_7B_Params()
        chunk = LlamaModelChunk(
            start_layer_id=0,
            end_layer_id=1,
            embed_dim=params.embed_dim,
            n_heads=params.n_heads,
            n_kv_heads=params.n_kv_heads,
            context_size=16,
            batch_size=1,
            ffn_hidden_dim=params.ffn_hidden_dim,
            rms_norm_eps=params.rms_norm_eps,
            has_qkv_bias=params.has_qkv_bias,
            use_qk_norm=params.use_qk_norm,
            use_drelu=params.use_drelu,
            cache_size=0,
            n_fp16_heads=params.n_fp16_heads,
            n_fp16_neurons=params.n_fp16_neurons,
            stat_folder=REPO_ROOT / "tmp" / "missing-qnn-stats",
        )

        ffn = chunk.layers[0].ffn
        self.assertEqual(len(ffn.fp16_neuron_ids), params.ffn_hidden_dim)
        self.assertEqual(len(ffn.int4_neuron_ids), 0)
        self.assertTrue(hasattr(ffn, "fp16_chunk"))
        self.assertFalse(hasattr(ffn, "int4_chunk"))

    def test_qwen3_1p7b_uses_bounded_unique_rmsnorm_scales(self):
        params = Qwen3_1_7B_Params()
        chunk = LlamaModelChunk(
            start_layer_id=0,
            end_layer_id=1,
            embed_dim=params.embed_dim,
            n_heads=params.n_heads,
            n_kv_heads=params.n_kv_heads,
            context_size=16,
            batch_size=1,
            ffn_hidden_dim=params.ffn_hidden_dim,
            rms_norm_eps=params.rms_norm_eps,
            has_qkv_bias=params.has_qkv_bias,
            use_qk_norm=params.use_qk_norm,
            use_drelu=params.use_drelu,
            cache_size=0,
            n_fp16_heads=params.n_fp16_heads,
            n_fp16_neurons=params.n_fp16_neurons,
            stat_folder=REPO_ROOT / "tmp" / "missing-qnn-stats",
        )

        attn = chunk.layers[0].attn
        ffn = chunk.layers[0].ffn
        scales = [
            attn.norm.sum_weight_scale,
            *[norm.sum_weight_scale for norm in attn.q_norms],
            *[norm.sum_weight_scale for norm in attn.k_norms],
            ffn.norm.sum_weight_scale,
        ]

        self.assertEqual(len(scales), len(set(scales)))
        self.assertLess(max(scales), 2.0)
        self.assertGreater(min(scales), 1.0)

    def test_rmsnorm_export_prescales_pow_input_and_compensates_reduction_scale(self):
        module = LlamaRMSNorm(embed_dim=8, eps=1e-5, device=torch.device("cpu"), sum_weight_scale=1.125).eval()
        x = torch.randn(2, 8, dtype=torch.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "rmsnorm.onnx"
            torch.onnx.export(
                module,
                (x,),
                str(onnx_path),
                input_names=["x"],
                output_names=["y"],
                opset_version=17,
            )
            graph = onnx.load(onnx_path, load_external_data=False).graph

        producers = {output: node for node in graph.node for output in node.output}
        pow_node = next(node for node in graph.node if node.op_type == "Pow")
        self.assertNotEqual(pow_node.input[0], "x")

        prescale_mul = producers[pow_node.input[0]]
        self.assertEqual(prescale_mul.op_type, "Mul")
        self.assertIn("x", prescale_mul.input)
        prescale_const_name = next(name for name in prescale_mul.input if name != "x")
        prescale_const = producers[prescale_const_name]
        self.assertAlmostEqual(
            self._get_constant_scalar(prescale_const),
            RMSNORM_POW_INPUT_PRESCALE,
            places=7,
        )

        matmul_node = next(node for node in graph.node if node.op_type == "MatMul")
        reduction_mul = next(
            node for node in graph.node if node.op_type == "Mul" and matmul_node.output[0] in node.input
        )
        reduction_scale_name = next(name for name in reduction_mul.input if name != matmul_node.output[0])
        reduction_scale_const = producers[reduction_scale_name]
        expected_scale = 1.0 / (
            module.embed_dim * module.sum_weight_scale * (RMSNORM_POW_INPUT_PRESCALE ** 2)
        )
        self.assertAlmostEqual(self._get_constant_scalar(reduction_scale_const), expected_scale, delta=2e-6)


if __name__ == "__main__":
    unittest.main()
