## Task

Organize the currently used Qwen2 and Qwen3 QNN AoT artifacts under the repository `models/` tree so the working set is no longer scattered across device paths and ad-hoc local folders.

## Result

Created these top-level directories:

- `models/Qwen2-AoT`
- `models/Qwen3-AoT`

Final Qwen2 layout:

- `models/Qwen2-AoT/qwen2-qnn-attn-proj/batch-1`
- `models/Qwen2-AoT/qwen2-qnn-attn-proj/batch-128`
- `models/Qwen2-AoT/qwen2-qnn-attn-core/batch-1`
- `models/Qwen2-AoT/qwen2-qnn-attn-core/batch-128`
- `models/Qwen2-AoT/qwen2-qnn-ffn/batch-1`
- `models/Qwen2-AoT/qwen2-qnn-ffn/batch-128`
- `models/Qwen2-AoT/qwen2-qnn-full/qnn`
- `models/Qwen2-AoT/qwen2-qnn-full/*.json`

Final Qwen3 layout:

- `models/Qwen3-AoT/qwen3-qnn-full/qnn`

## Source Mapping

Qwen2 split-stage runtime assets:

- Source device: `db6c02cf`
- Source root: `/data/local/tmp/acom-attn-proj-verify/models/qwen2_0.5b`
- Copied only the runtime-minimal stage assets:
  - each batch config JSON
  - each per-layer `.bin`
- Intentionally excluded export/debug byproducts such as:
  - `onnx_model/`
  - `data/`
  - `build_*.log`
  - `*.io.json`
  - `*.encodings`
  - `x86_64-linux-clang/`

Qwen2 full-graph runtime:

- Main runtime body copied from local reference:
  - `/mnt/sda1/pzw/HeteroCompute/llama.cpp/ref/PowerServe/models/qwen2_0.5b/qnn`
- Device-only config variants pulled from `db6c02cf`:
  - `config-only1.json`
  - `config-only128.json`
  - `config-no-embeddings.json`
  - `config-noseed.json`
- Route JSONs were staged into `models/Qwen2-AoT/qwen2-qnn-full/`

Qwen3 full-graph runtime:

- Batch-1 runtime body copied locally from:
  - `/mnt/sda1/pzw/HeteroCompute/qwen3_1.7b_runtime_pkg_20260319a/models/Qwen3/qnn`
- Batch-128 MM variant copied locally from:
  - `/mnt/sda1/pzw/HeteroCompute/qwen3_1.7b_batch128_only_qnn_mmfix`
- Device-original batch/config variants pulled from `db6c02cf`:
  - `config.batch1.bak.json`
  - `config_hvx8.json`
  - `config_hvx16.json`
  - `lm_head_b128.bin`
- The following files were normalized into the final `qnn/` folder:
  - `qwen3_1.7b_0.bin`
  - `qwen3_1.7b_b128mm_0.bin`
  - `lm_head.bin`
  - `lm_head_b128.bin`
  - `config.json`
  - `config.batch1.bak.json`
  - `config_hvx8.json`
  - `config_hvx16.json`

## Validation

Qwen2 split bins present:

- `attn_proj`: `48`
- `attn_core`: `48`
- `ffn`: `48`

Approximate directory sizes after consolidation:

- `models/Qwen2-AoT`: `764M`
- `models/Qwen3-AoT`: `2.3G`

## Notes

- Qwen3 split subgraph AoT directories were not found as a current working set, so only the full-graph QNN runtime was consolidated for Qwen3.
- `models/` is ignored by git via `.gitignore` rule `/models/*`, so this reorganization is intentionally a workspace artifact layout change rather than a tracked source-tree change.
