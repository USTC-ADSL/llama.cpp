from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Literal, NamedTuple, Optional, Tuple, Union


parser = argparse.ArgumentParser()
parser.add_argument("--n-threads", type=int, default=1)
parser.add_argument("--model-folder", type=Path, required=True)
parser.add_argument("--model-name", required=True)
parser.add_argument("--graph-name", required=True)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--system-prompt-file", type=Path)
parser.add_argument("--prompt-file", type=Path, required=True)
parser.add_argument("--n-model-chunks", type=int, default=1)
parser.add_argument("--max-n-tokens", type=int, required=True)
parser.add_argument("--output-folder", type=Path, required=True)
parser.add_argument("--layers", type=int, nargs="*")
args = parser.parse_args()

import onnx  # noqa: E402
import torch  # noqa: E402
from onnx import shape_inference  # noqa: E402
from torch import nn  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PS_QNN_CONVERTER = REPO_ROOT / "ref" / "PowerServe" / "tools" / "qnn_converter"
if str(PS_QNN_CONVERTER) not in sys.path:
    sys.path.insert(0, str(PS_QNN_CONVERTER))

import export_to_onnx as ps_export  # noqa: E402
from graph_params import graph_map, GraphParams  # noqa: E402
from model_params import model_map, ModelParams  # noqa: E402

if args.model_name not in model_map:
    parser.error(f"--model-name must be one of: {', '.join(sorted(model_map.keys()))}")
if args.graph_name not in graph_map:
    parser.error(f"--graph-name must be one of: {', '.join(sorted(graph_map.keys()))}")

torch.manual_seed(42)
torch.set_num_threads(args.n_threads)
device = torch.device(args.device)
ps_export.device = device


def export_json(obj, path: Path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


class Sample(NamedTuple):
    inputs: Tuple[torch.Tensor, ...]
    outputs: Tuple[torch.Tensor, ...]


class ExportableAttentionStage(nn.Module):
    def __init__(self, layer_id: int, n_kv_heads: int, attention_module: nn.Module):
        super().__init__()
        self.layer_id = layer_id
        self.n_kv_heads = n_kv_heads
        self.attention = attention_module
        self.saved_samples: List[Sample] = []

    @property
    def kv_cache_names(self) -> List[str]:
        return [
            *[f"layer_{self.layer_id}_key_t_cache_{head}" for head in range(self.n_kv_heads)],
            *[f"layer_{self.layer_id}_value_cache_{head}" for head in range(self.n_kv_heads)],
        ]

    @property
    def kv_names(self) -> List[str]:
        return [
            *[f"layer_{self.layer_id}_key_{head}" for head in range(self.n_kv_heads)],
            *[f"layer_{self.layer_id}_value_{head}" for head in range(self.n_kv_heads)],
        ]

    @property
    def input_names(self) -> List[str]:
        return ["x", "attn_bias", "rope_embed_cos", "rope_embed_sin", *self.kv_cache_names]

    @property
    def output_names(self) -> List[str]:
        return ["out", *self.kv_names]

    @property
    def dtype_preserved_io_names(self) -> List[str]:
        return ["x", "out", "rope_embed_cos", "rope_embed_sin"]

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor,
        rope_embed_cos: torch.Tensor,
        rope_embed_sin: torch.Tensor,
        *caches: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        if len(caches) != 2 * self.n_kv_heads:
            raise RuntimeError(
                f"layer {self.layer_id}: expected {2 * self.n_kv_heads} cache tensors, got {len(caches)}"
            )

        key_t_caches = caches[: self.n_kv_heads]
        value_caches = caches[self.n_kv_heads :]
        out, keys, values = self.attention(x, key_t_caches, value_caches, attn_bias, (rope_embed_cos, rope_embed_sin))
        return (out, *keys, *values)


class AttentionExporter:
    def __init__(self, graph_name: str, model_chunk: ExportableAttentionStage, output_folder: Path):
        self.graph_name = graph_name
        self.model_chunk = model_chunk
        self.output_folder = output_folder

    @torch.no_grad()
    def export_onnx_model(self):
        onnx_model_folder = self.output_folder / "onnx_model"
        onnx_model_folder.mkdir(parents=True, exist_ok=True)

        onnx_model_path = onnx_model_folder / f"{self.graph_name}.onnx"
        torch.onnx.export(
            model=self.model_chunk,
            args=self.model_chunk.saved_samples[0].inputs,
            f=str(onnx_model_path),
            input_names=self.model_chunk.input_names,
            output_names=self.model_chunk.output_names,
        )

        onnx_model = onnx.load(onnx_model_path, load_external_data=False)
        self.onnx_model = shape_inference.infer_shapes(onnx_model)

    def export_io_spec(self):
        def dump_info_list(io_type: Literal["in", "out"], names: List[str], tensors: Tuple[torch.Tensor, ...]) -> List[dict]:
            return [
                {
                    "name": name,
                    "type": io_type,
                    "dtype": "float32",
                    "preserve_dtype": name in self.model_chunk.dtype_preserved_io_names,
                    "shape": list(tensor.shape),
                }
                for name, tensor in zip(names, tensors, strict=True)
            ]

        io_spec = [
            *dump_info_list("in", self.model_chunk.input_names, self.model_chunk.saved_samples[0].inputs),
            *dump_info_list("out", self.model_chunk.output_names, self.model_chunk.saved_samples[0].outputs),
        ]
        export_json(io_spec, self.output_folder / f"{self.graph_name}.io.json")

    def export_quantization_config(self):
        class Encoding(NamedTuple):
            category: Literal["activation", "param"]
            bitwidth: int
            dtype: Literal["float", "int"]

        encoding_map: Dict[str, Encoding] = {}

        def update_encoding(name: str, encoding: Encoding):
            if name not in encoding_map or encoding.bitwidth > encoding_map[name].bitwidth:
                encoding_map[name] = encoding

        def encode_activation(node: Union[str, onnx.ValueInfoProto], bitwidth: int):
            if not isinstance(node, str):
                node = node.name
            update_encoding(node, Encoding("activation", bitwidth, "float"))

        def encode_output(node: onnx.NodeProto, bitwidth: int):
            for name in node.output:
                update_encoding(name, Encoding("activation", bitwidth, "float"))

        def encode_param(node: Union[str, onnx.NodeProto], bitwidth: int, dtype: Literal["float", "int"]):
            if not isinstance(node, str):
                node = node.name
            update_encoding(node, Encoding("param", bitwidth, dtype))

        def match(target: Union[str, onnx.NodeProto, onnx.TensorProto, onnx.ValueInfoProto], pattern: str):
            if not isinstance(target, str):
                target = target.name
            return re.fullmatch(pattern, target) is not None

        graph = self.onnx_model.graph

        encode_activation("x", 32)
        encode_activation("attn_bias", 16)
        encode_activation("rope_embed_cos", 16)
        encode_activation("rope_embed_sin", 16)

        for node in graph.input:
            if match(node, "layer_[0-9]+_(key_t_cache|value_cache)_[0-9]+"):
                encode_activation(node, 16)
        for node in graph.output:
            if match(node, "layer_[0-9]+_(key|value)_[0-9]+"):
                encode_activation(node, 16)

        for node in graph.node:
            if match(node, "(.*/)?core.*"):
                encode_output(node, 16)

        for node in graph.initializer:
            if match(node, "(.*/)?(norm|q_norms\\.[0-9]+|k_norms\\.[0-9]+)\\.(weight|sum_weights)"):
                encode_param(node, 32, "float")
        for node in graph.node:
            if match(node, "(.*/)?(norm|q_norms\\.[0-9]+|k_norms\\.[0-9]+).*"):
                encode_output(node, 32)
            elif match(node, "(.*/)?Add(_[0-9]+|)"):
                encode_output(node, 32)

        for node in graph.initializer:
            if "fp16_" in node.name:
                encode_param(node, 16, "float")
        for node in graph.node:
            if "fp16_" in node.name:
                encode_output(node, 16)

        config = {
            "version": "0.6.1",
            "quantizer_args": {
                "activation_bitwidth": 16,
                "param_bitwidth": 4,
                "dtype": "int",
                "per_channel_quantization": True,
                "quant_scheme": "post_training_tf",
            },
            "activation_encodings": {},
            "param_encodings": {},
        }

        for name, encoding in sorted(encoding_map.items(), key=lambda item: item[0]):
            config[f"{encoding.category}_encodings"][name] = [{
                "bitwidth": encoding.bitwidth,
                "dtype": encoding.dtype,
                "is_symmetric": str(encoding.category == "param"),
            }]

        export_json(config, self.output_folder / f"{self.graph_name}.encodings")

    def export_sample_inputs(self):
        input_list = []
        for i, samples in enumerate(self.model_chunk.saved_samples):
            data_folder = (self.output_folder / "data" / str(i)).resolve()
            data_folder.mkdir(parents=True, exist_ok=True)

            tensor_paths = []
            for name, tensor in zip(self.model_chunk.input_names, samples.inputs, strict=True):
                output_path = data_folder / f"{name}.raw"
                tensor.cpu().numpy().tofile(output_path)
                tensor_paths.append(f"{name}:={output_path}")

            input_list.append(" ".join(tensor_paths))

        with open(self.output_folder / "input_list.txt", "w") as f:
            f.write("\n".join(input_list))

    def export(self):
        self.export_onnx_model()
        self.export_io_spec()
        self.export_quantization_config()
        self.export_sample_inputs()


def clone_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu() if tensor.device.type != "cpu" else tensor.detach().clone()


def flatten_attention_inputs(inputs: Tuple[object, ...]) -> Tuple[torch.Tensor, ...]:
    if len(inputs) != 5:
        raise RuntimeError(f"expected 5 attention inputs, got {len(inputs)}")

    x, key_t_caches, value_caches, attn_bias, rope_embeds = inputs
    if not isinstance(key_t_caches, tuple) or not isinstance(value_caches, tuple) or not isinstance(rope_embeds, tuple):
        raise RuntimeError("unexpected attention hook input structure")

    return (
        clone_cpu_tensor(x),
        clone_cpu_tensor(attn_bias),
        clone_cpu_tensor(rope_embeds[0]),
        clone_cpu_tensor(rope_embeds[1]),
        *(clone_cpu_tensor(t) for t in key_t_caches),
        *(clone_cpu_tensor(t) for t in value_caches),
    )


def flatten_attention_outputs(outputs: Tuple[object, ...]) -> Tuple[torch.Tensor, ...]:
    if len(outputs) != 3:
        raise RuntimeError(f"expected 3 attention outputs, got {len(outputs)}")

    out, keys, values = outputs
    if not isinstance(keys, tuple) or not isinstance(values, tuple):
        raise RuntimeError("unexpected attention hook output structure")

    return (
        clone_cpu_tensor(out),
        *(clone_cpu_tensor(t) for t in keys),
        *(clone_cpu_tensor(t) for t in values),
    )


def snapshot_seed_kv(
    model: ps_export.LlamaModel,
    selected_layers: List[int],
    kv_size: int,
) -> Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]]:
    snapshot: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
    if kv_size == 0:
        return snapshot

    selected = set(selected_layers)
    for model_chunk in model.model_chunks:
        for layer_id in range(model_chunk.start_layer_id, model_chunk.end_layer_id):
            if layer_id not in selected:
                continue

            rel_layer_id = layer_id - model_chunk.start_layer_id
            for head in range(model_chunk.n_kv_heads):
                key_t = clone_cpu_tensor(model_chunk.key_t_cache_data[rel_layer_id, head][:, :kv_size]).transpose(0, 1).contiguous()
                value = clone_cpu_tensor(model_chunk.value_cache_data[rel_layer_id, head][:kv_size, :]).contiguous()
                snapshot[(layer_id, head)] = (key_t, value)

    return snapshot


print("Creating model...")

model_params: ModelParams = model_map[args.model_name]()
graph_params: GraphParams = graph_map[args.graph_name]()

assert model_params.n_layers % args.n_model_chunks == 0
n_layers_per_model_chunk = model_params.n_layers // args.n_model_chunks

model_chunks = [
    ps_export.LlamaModelChunk(
        start_layer_id=i,
        end_layer_id=(i + n_layers_per_model_chunk),
        embed_dim=model_params.embed_dim,
        n_heads=model_params.n_heads,
        n_kv_heads=model_params.n_kv_heads,
        context_size=graph_params.context_size,
        batch_size=graph_params.batch_size,
        ffn_hidden_dim=model_params.ffn_hidden_dim,
        rms_norm_eps=model_params.rms_norm_eps,
        has_qkv_bias=model_params.has_qkv_bias,
        use_qk_norm=model_params.use_qk_norm,
        use_drelu=model_params.use_drelu,
        cache_size=graph_params.cache_size,
        n_fp16_heads=model_params.n_fp16_heads,
        n_fp16_neurons=model_params.n_fp16_neurons,
        stat_folder=args.model_folder / "stat",
    )
    for i in range(0, model_params.n_layers, n_layers_per_model_chunk)
]

model = ps_export.LlamaModel(
    model_folder=args.model_folder,
    model_params=model_params,
    graph_params=graph_params,
    model_chunks=model_chunks,
)

print("Loading model weights...")
model.load_weights()

selected_layers = sorted(set(args.layers if args.layers else range(model_params.n_layers)))
attention_modules: Dict[int, ExportableAttentionStage] = {}
for model_chunk in model.model_chunks:
    for layer in model_chunk.layers:
        if layer.layer_id in selected_layers:
            attention_modules[layer.layer_id] = ExportableAttentionStage(
                layer_id=layer.layer_id,
                n_kv_heads=model_params.n_kv_heads,
                attention_module=layer.attn,
            )

if sorted(attention_modules.keys()) != selected_layers:
    missing = sorted(set(selected_layers).difference(attention_modules.keys()))
    raise RuntimeError(f"Missing attention modules for layers: {missing}")

seed_kv_snapshot: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
kv_size = 0
if args.system_prompt_file is not None:
    with open(args.system_prompt_file, "r") as f:
        system_prompt = f.read()
    model.eval_system_prompt(system_prompt)
    kv_size = model.system_prompt_length
    seed_kv_snapshot = snapshot_seed_kv(model, selected_layers, kv_size)

hook_handles = []
for model_chunk in model.model_chunks:
    for layer in model_chunk.layers:
        if layer.layer_id not in attention_modules:
            continue

        def save_sample(module, inputs, output, layer_id=layer.layer_id):
            if not inputs:
                return
            if inputs[0].shape[0] != graph_params.batch_size:
                return
            if not isinstance(output, tuple):
                output = (output,)
            x = flatten_attention_inputs(inputs)
            y = flatten_attention_outputs(output)
            attention_modules[layer_id].saved_samples.append(Sample(x, y))

        hook_handles.append(layer.attn.register_forward_hook(save_sample))

with open(args.prompt_file, "r") as f:
    prompt = f.read()

model.eval_prompt(
    prompt=prompt,
    batch_size=graph_params.batch_size,
    save_samples=False,
    max_n_tokens=args.max_n_tokens,
)
print(f"Collected attention samples for {len(attention_modules)} layers")

for handle in hook_handles:
    handle.remove()

args.output_folder.mkdir(parents=True, exist_ok=True)

if kv_size > 0:
    kv_folder = args.output_folder / "kv"
    kv_folder.mkdir(parents=True, exist_ok=True)
    for layer_id in selected_layers:
        for head in range(model_params.n_kv_heads):
            key_tensor, value_tensor = seed_kv_snapshot[(layer_id, head)]
            key_tensor.numpy().tofile(kv_folder / f"layer_{layer_id}_key_{head}.raw")
            value_tensor.numpy().tofile(kv_folder / f"layer_{layer_id}_value_{head}.raw")

config = {
    "model_parameters": {
        "n_layers": model_params.n_layers,
        "vocab_size": model_params.vocab_size,
        "embed_dim": model_params.embed_dim,
        "ffn_hidden_dim": model_params.ffn_hidden_dim,
        "head_dim": model_params.head_dim,
        "n_kv_heads": model_params.n_kv_heads,
        "rope_theta": model_params.rope_theta,
        "rms_norm_eps": model_params.rms_norm_eps,
        "attention_mask_value": model_params.attention_mask_value,
        "tie_embedding": model_params.tie_embedding,
    },
    "qnn_parameters": {"n_hvx_threads": 4},
    "graphs": [],
}

for layer_id in selected_layers:
    graph_name = f"attention_layer_{layer_id}_{args.graph_name}"
    output_folder = args.output_folder / f"attention_layer_{layer_id}"
    output_folder.mkdir(parents=True, exist_ok=True)

    if not attention_modules[layer_id].saved_samples:
        raise RuntimeError(f"No attention calibration samples captured for layer {layer_id}")

    exporter = AttentionExporter(
        graph_name=graph_name,
        model_chunk=attention_modules[layer_id],
        output_folder=output_folder,
    )
    exporter.export()

    graph_config = {
        "type": "attention",
        "graph_name": graph_name,
        "model_path": f"attention_layer_{layer_id}/attention_layer_{layer_id}.bin",
        "x_name": "x",
        "out_name": "out",
        "batch_size": graph_params.batch_size,
        "cache_size": graph_params.cache_size,
        "context_size": graph_params.context_size,
        "start_layer_id": layer_id,
        "end_layer_id": layer_id + 1,
        "kv_size": kv_size,
    }
    if kv_size > 0:
        graph_config["kv_path_format"] = "kv/layer_{layer_id}_{kv_type}_{head_id}.raw"

    config["graphs"].append(graph_config)

export_json(config, args.output_folder / f"config_{args.graph_name}.json")
