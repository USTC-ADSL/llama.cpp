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
from graph_params import GraphParams, graph_map  # noqa: E402
from model_params import ModelParams, model_map  # noqa: E402

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


class ExportableFFNStage(nn.Module):
    def __init__(self, layer_id: int, ffn_module: nn.Module):
        super().__init__()
        self.layer_id = layer_id
        self.ffn = ffn_module
        self.saved_samples: List[Sample] = []

    @property
    def input_names(self) -> List[str]:
        return ["x"]

    @property
    def output_names(self) -> List[str]:
        return ["out"]

    @property
    def dtype_preserved_io_names(self) -> List[str]:
        return ["x", "out"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class FFNExporter:
    def __init__(self, graph_name: str, model_chunk: ExportableFFNStage, output_folder: Path):
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

        for node in graph.initializer:
            if match(node, "(ffn\\.)?norm\\.(weight|sum_weights)"):
                encode_param(node, 32, "float")
        for node in graph.node:
            if match(node, "(.*/)?(ffn/)?norm.*"):
                encode_output(node, 32)
            elif match(node, ".*/Add(_[0-9]+|)") or match(node, "Add(_[0-9]+|)"):
                encode_output(node, 32)
            elif "fp16_" in node.name:
                encode_output(node, 16)
        for node in graph.initializer:
            if "fp16_" in node.name:
                encode_param(node, 16, "float")

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
ffn_modules: Dict[int, ExportableFFNStage] = {}
for model_chunk in model.model_chunks:
    for layer in model_chunk.layers:
        if layer.layer_id in selected_layers:
            ffn_modules[layer.layer_id] = ExportableFFNStage(layer.layer_id, layer.ffn)

if sorted(ffn_modules.keys()) != selected_layers:
    missing = sorted(set(selected_layers).difference(ffn_modules.keys()))
    raise RuntimeError(f"Missing FFN modules for layers: {missing}")

hook_handles = []
for model_chunk in model.model_chunks:
    for layer in model_chunk.layers:
        if layer.layer_id not in ffn_modules:
            continue

        def save_sample(module, inputs, output, layer_id=layer.layer_id):
            if not inputs:
                return
            x = inputs[0]
            if not isinstance(x, torch.Tensor) or x.shape[0] != graph_params.batch_size:
                return
            sample_inputs = tuple(clone_cpu_tensor(t) for t in inputs)
            if not isinstance(output, tuple):
                output = (output,)
            sample_outputs = tuple(clone_cpu_tensor(t) for t in output)
            ffn_modules[layer_id].saved_samples.append(Sample(sample_inputs, sample_outputs))

        hook_handles.append(layer.ffn.register_forward_hook(save_sample))

if args.system_prompt_file is not None:
    with open(args.system_prompt_file, "r") as f:
        system_prompt = f.read()
    model.eval_system_prompt(system_prompt)

with open(args.prompt_file, "r") as f:
    prompt = f.read()

model.eval_prompt(
    prompt=prompt,
    batch_size=graph_params.batch_size,
    save_samples=False,
    max_n_tokens=args.max_n_tokens,
)
print(f"Collected FFN samples for {len(ffn_modules)} layers")

for handle in hook_handles:
    handle.remove()

args.output_folder.mkdir(parents=True, exist_ok=True)

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
    "embeddings": [],
}

for layer_id in selected_layers:
    graph_name = f"ffn_layer_{layer_id}_{args.graph_name}"
    output_folder = args.output_folder / f"ffn_layer_{layer_id}"
    output_folder.mkdir(parents=True, exist_ok=True)

    if not ffn_modules[layer_id].saved_samples:
        raise RuntimeError(f"No FFN calibration samples captured for layer {layer_id}")

    exporter = FFNExporter(
        graph_name=graph_name,
        model_chunk=ffn_modules[layer_id],
        output_folder=output_folder,
    )
    exporter.export()

    config["graphs"].append({
        "type": "ffn",
        "graph_name": graph_name,
        "model_path": f"ffn_layer_{layer_id}/ffn_layer_{layer_id}.bin",
        "x_name": "x",
        "out_name": "out",
        "batch_size": graph_params.batch_size,
        "cache_size": graph_params.cache_size,
        "context_size": graph_params.context_size,
        "start_layer_id": layer_id,
        "end_layer_id": layer_id + 1,
        "kv_size": 0,
    })

export_json(config, args.output_folder / f"config_{args.graph_name}.json")
