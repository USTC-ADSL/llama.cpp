# QNN AoT Stage Export

`export_attention_to_onnx.py` exports per-layer QNN Attention AoT artifacts that match the runtime's `type=attention` loader in `ggml/src/ggml-qnn/qnn/aot.cpp`.

It reuses the reference PowerServe converter code under `ref/PowerServe/tools/qnn_converter`, but emits a config layout that the current `llama.cpp` QNN AoT runtime can consume directly.

Example:

```bash
python tools/qnn-aot-export/export_attention_to_onnx.py \
  --model-folder /path/to/hf-model \
  --model-name llama3_2_1b \
  --graph-name batch_4 \
  --prompt-file ref/PowerServe/tools/qnn_converter/prompt/lab_intro_llama.md \
  --output-folder /tmp/qnn-attn-export \
  --max-n-tokens 128 \
  --layers 0 1 2 3
```

`export_attn_proj_to_onnx.py` exports a narrower `type=attn_proj` graph that stops at attention projection and emits flattened `Q/K/V` outputs. This is the intended contract for `ATTN_PROJ -> ATTN_CORE` mixed routing when QNN only produces KV-ready projections and CPU/OpenCL still own `cpy_k/cpy_v`, `kq`, and `kqv`.

Example:

```bash
python tools/qnn-aot-export/export_attn_proj_to_onnx.py \
  --model-folder /path/to/hf-model \
  --model-name qwen2.5_0.5b \
  --graph-name batch_128 \
  --prompt-file ref/PowerServe/tools/qnn_converter/prompt/lab_intro_qwen.md \
  --output-folder /tmp/qnn-attn-proj-export \
  --max-n-tokens 128 \
  --layers 0 1 2 3
```

If `--system-prompt-file` is provided, the script also snapshots seed KV files into `OUTPUT_DIR/kv/` and sets `kv_size` in the generated config.

`export_attn_core_to_onnx.py` exports the complementary `type=attn_core` graph for the 3-way transformer split. Its contract is:

- input `x`: residual input that will be added back after the attention output projection
- input `qcur / kcur / vcur`: current-token attention projections from the previous `attn_proj` stage
- input `cache_k / cache_v`: the shared KV cache layout that CPU/OpenCL already use
- input `attn_bias`: the fixed-size attention bias buffer the runtime materializes from `self_kq_mask`
- output `out`: `ffn_inp`, i.e. the post-attention residual that should cross the `attn_core -> ffn` boundary

Current attn-core export uses `float16` for the shared external inputs `attn_bias`, `cache_k`, and `cache_v` by default so the split graph matches the main `W4A16` route more closely.
If the current QNN converter/calibration path still rejects those external inputs as `F16` on a given model/toolchain, pass `--shared-input-dtype float32` to fall back to the legacy workaround.
For zero-copy `attn_proj <-> attn_core` KV sharing across CPU / OpenCL / QNN AoT, the runtime KV layout must match the exported graph input dtype; the runtime now validates this against the actual AoT graph metadata instead of assuming `F32`.

Example:

```bash
python tools/qnn-aot-export/export_attn_core_to_onnx.py \
  --model-folder /path/to/hf-model \
  --model-name qwen2.5_0.5b \
  --graph-name batch_1 \
  --shared-input-dtype float16 \
  --prompt-file ref/PowerServe/tools/qnn_converter/prompt/lab_intro_qwen.md \
  --output-folder /tmp/qnn-attn-core-export \
  --max-n-tokens 128 \
  --layers 0 1 2 3
```

`export_ffn_to_onnx.py` exports per-layer `type=ffn` AoT assets that match the residual-stage matcher in the current QNN runtime. This is the path needed when prefill and decode should use different FFN batch buckets such as `batch_128` and `batch_1`.

Example:

```bash
python tools/qnn-aot-export/export_ffn_to_onnx.py \
  --model-folder /path/to/hf-model \
  --model-name qwen2.5_0.5b \
  --graph-name batch_1 \
  --prompt-file ref/PowerServe/tools/qnn_converter/prompt/lab_intro_qwen.md \
  --output-folder /tmp/qnn-ffn-export \
  --max-n-tokens 128
```

When the runtime should choose between multiple batch buckets from a single `GGML_QNN_AOT_CONFIG`, merge the per-batch configs with `merge_aot_configs.py` and point `GGML_QNN_AOT_MODEL_DIR` at the common root that contains all referenced artifact folders.

Example:

```bash
python tools/qnn-aot-export/merge_aot_configs.py \
  --entry qnn_ffn_batch1=/models/qwen2_0.5b/qnn_ffn_batch1/config_batch_1.json \
          qnn_ffn_batch128=/models/qwen2_0.5b/qnn_ffn_batch128/config_batch_128.json \
  --output /models/qwen2_0.5b/qnn_ffn_combined.json
```

Validated mixed-route status:

- `type=ffn` can now use separate `batch_128` and `batch_1` exports from one merged config, so prefill and decode do not need to share a single FFN AoT bucket.
- `type=attn_proj` can also use separate `batch_128` and `batch_1` exports from one merged config, matching the intended `ATTN_PROJ -> ATTN_CORE` stage split.
- The hetero compute path can now place CPU, OpenCL, and QNN AoT compute tensors on the same `qnn-npu-host` shared-host allocation when the adjacent stage boundary requires it.
- With `attn_proj=qnn-npu -> attn_core=cpu`, the hetero KV contract allocates `layout=stage-shared transfer=qnn-rpcmem storage=qnn-npu-host`, and the KV cache lives on `qnn-npu-host`.
- With `attn_proj=qnn-npu -> attn_core=opencl`, the same stage-shared KV contract now also resolves to `storage=qnn-npu-host`, and OpenCL consumes that host allocation through its external host-buffer alias path.
- In other words, when QNN only produces attention projections and CPU/OpenCL own `cpy_k` / `cpy_v` / `kq` / `kqv`, the KV cache layout can stay shared across CPU, OpenCL, and the QNN-produced stage boundary without an extra KV migration step at the `attn_proj -> attn_core` split.

Current limitation:

- `type=attention` still follows the PowerServe full-attention contract and keeps its own QNN-side KV state.
- `type=attn_core` is intentionally scoped to the 3-way split where `attn_out` is folded into `attn_core`; it is not the old 4-way `attn_core + attn_out` route.
- The current `attn_core` runtime path is scoped to the single-stream shared-KV layout that decode/prefill currently use in this tree; it is not yet a generic replacement for every historical attention layout variant.
- The current QNN `attn_core` path is zero-copy only when the runtime KV cache dtype matches the exported `attn_core` graph inputs for `cache_k/cache_v`. The runtime checks this at graph load time.
