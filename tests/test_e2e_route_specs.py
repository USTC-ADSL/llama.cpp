from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script_module(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CPU_FIXED_ROUTES = {
    "CPU_B1_3513600": "cpu{threads=1,affinity=40,cpu_policy6_freq_khz=3513600}",
    "CPU_B2_1958400": "cpu{threads=2,affinity=C0,cpu_policy6_freq_khz=1958400}",
    "CPU_B2S2_4320000_3532800": (
        "cpu{threads=4,affinity=CC,cpu_policy0_freq_khz=3532800,cpu_policy6_freq_khz=4320000}"
    ),
    "CPU_B2S4_4320000_3532800": (
        "cpu{threads=6,affinity=FC,cpu_policy0_freq_khz=3532800,cpu_policy6_freq_khz=4320000}"
    ),
    "CPU_S6_1804800": "cpu{threads=6,affinity=3F,cpu_policy0_freq_khz=1804800}",
}


def test_run_e2e_experiment_fixed_cpu_states_expand_to_full_routes() -> None:
    module = load_script_module("run_e2e_experiment_for_test", "scripts/run_e2e_experiment.py")

    for state, expected in CPU_FIXED_ROUTES.items():
        assert module.route_spec_for_fixed_state(state) == expected


def test_run_e2e_system_benefit_fixed_cpu_states_expand_to_full_routes() -> None:
    module = load_script_module("run_e2e_system_benefit_for_test", "scripts/run_e2e_system_benefit.py")

    for state, expected in CPU_FIXED_ROUTES.items():
        assert module.route_spec_for_state(state) == expected


def test_run_e2e_experiment_selected_distribution_keeps_active_route() -> None:
    module = load_script_module("run_e2e_experiment_for_trace_test", "scripts/run_e2e_experiment.py")
    log_text = "\n".join(
        [
            "llama-e2e-bench: SAMPLE_BEGIN rep=1 sample=1/1",
            (
                "synchronize: timing phase=decode n_tokens=1 total_wall_us=1000 "
                "label=decode-schedule@1 reason=decode-schedule "
                "target=attn=qnn-npu,ffn=qnn-npu,output=qnn-npu"
            ),
            (
                "synchronize: timing phase=decode n_tokens=1 total_wall_us=2000 "
                "label=<none> reason=already-active target=<default>"
            ),
            "sample,rep=1,index=1,prompt_tokens=1,gen_tokens=2,elapsed_ms=3.000,tok_s=666.667",
        ]
    )

    samples, summary = module.parse_log_metrics(log_text, 10.0)

    assert len(samples) == 1
    assert summary["selected_state_distribution"] == {
        "attn=qnn-npu,ffn=qnn-npu,output=qnn-npu": {
            "tokens": 2,
            "fraction": 1.0,
        }
    }
