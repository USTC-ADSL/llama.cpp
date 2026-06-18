# Transition Profile TODO

No fake transition rows should be written. Add rows to `profiles/transition_profile.csv` only after a real measurement.

Expected CSV schema:

```csv
from_state,to_state,from_backend,to_backend,from_graph_capacity,to_graph_capacity,from_workpoint,to_workpoint,latency_ms,energy_mj,cold_or_warm,affects_boundary_tbt,run_id,stable,notes
```

Command template variables:

```text
{from_state} {to_state} {from_backend} {to_backend} {from_graph_capacity} {to_graph_capacity} {from_workpoint} {to_workpoint} {run_id} {output_json} {log_path}
```

Manual command template to adapt in `configs/offline_profile.yaml`:

```yaml
transition_command_template: >-
  YOUR_TRANSITION_MEASUREMENT_TOOL
  --from-state {from_state}
  --to-state {to_state}
  --from-backend {from_backend}
  --to-backend {to_backend}
  --from-graph-capacity {from_graph_capacity}
  --to-graph-capacity {to_graph_capacity}
  --from-workpoint {from_workpoint}
  --to-workpoint {to_workpoint}
  --run-id {run_id}
  --output-json {output_json}
  > {log_path} 2>&1
```

After adapting the template, preview or run with:

```bash
scripts/run_transition_profile.sh --dry-run --resume
scripts/run_transition_profile.sh --resume
```

Config: `configs/offline_profile.yaml`

Minimum transition set:

- `npu_low_balanced_cap2048` -> `npu_burst_cap2048`: NPU workpoint switch
- `npu_burst_cap2048` -> `npu_low_balanced_cap2048`: NPU workpoint switch
- `npu_balanced_cap2048` -> `npu_burst_cap2048`: NPU workpoint switch
- `npu_burst_cap2048` -> `npu_balanced_cap2048`: NPU workpoint switch
- `npu_burst_cap2048` -> `npu_burst_cap4096`: QNN graph load/switch
- `npu_burst_cap4096` -> `npu_burst_cap6144`: QNN graph load/switch
- `cold` -> `npu_burst_cap2048`: cold load cap2048
- `cold` -> `npu_burst_cap4096`: cold load cap4096
- `cold` -> `npu_burst_cap6144`: cold load cap6144
- `warm` -> `npu_burst_cap2048`: warm load cap2048
- `warm` -> `npu_burst_cap4096`: warm load cap4096
- `warm` -> `npu_burst_cap6144`: warm load cap6144
- `npu_burst_cap2048` -> `cpu_B2_2649`: Backend switch NPU -> CPU
- `cpu_B2_2649` -> `npu_burst_cap2048`: Backend switch CPU -> NPU
- `npu_burst_cap2048` -> `gpu_734`: Backend switch NPU -> GPU
- `gpu_734` -> `npu_burst_cap2048`: Backend switch GPU -> NPU
- `cpu_B2_2649` -> `gpu_734`: Backend switch CPU -> GPU
- `gpu_734` -> `cpu_B2_2649`: Backend switch GPU -> CPU
- `cpu_B1_2649` -> `cpu_B2_2649`: CPU state switch B1 -> B2
- `cpu_B2_1804` -> `cpu_B2_2649`: CPU state switch B2 lowfreq -> highfreq
- `gpu_305` -> `gpu_734`: GPU frequency switch low -> high
- `gpu_734` -> `gpu_305`: GPU frequency switch high -> low

Expected output JSON from a future transition command:

```json
{"latency_ms": 1.23, "energy_mj": 4.56, "stable": true, "notes": "measured on device"}
```

Reasons this remains TODO by default:

- CPU affinity/frequency control is device-specific.
- NPU workpoint and QNN graph switching APIs are platform-specific.
- GPU frequency control may require privileged sysfs or vendor tooling.
- Power/energy sampling source is not standardized in this repository.
