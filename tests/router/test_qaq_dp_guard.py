import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

qaq_linear_module = importlib.import_module("any_precision.modules.QAQDPLLM_Linear")
QAQDPLLM_Linear = qaq_linear_module.QAQDPLLM_Linear
from any_precision.modules.QAQDPLLMForCausalLM import QAQDPLLMForCausalLM


def install_fake_kernels(monkeypatch):
    def fake_matmul_kbit(x, qweight, lut, bit):
        out_features = qweight.shape[1]
        return torch.full((*x.shape[:-1], out_features), float(bit), dtype=x.dtype, device=x.device)

    def fake_dequant_kbit(qweight, lut, bit):
        out_features = qweight.shape[1]
        return torch.full((out_features, 32), float(bit), dtype=lut.dtype, device=lut.device)

    monkeypatch.setattr(qaq_linear_module, "matmul_kbit", fake_matmul_kbit)
    monkeypatch.setattr(qaq_linear_module, "dequant_kbit", fake_dequant_kbit)


def make_threshold_linear(router=None, router_mode="dp_threshold_only", confidence_threshold=None):
    return QAQDPLLM_Linear(
        in_features=32,
        out_features=1,
        supported_bits=[3, 4, 5, 6],
        router=router,
        route_id=0,
        route_name="0.q_proj",
        bias=False,
        precisions=[3, 4, 5, 6],
        dtype=torch.float32,
        device=torch.device("cpu"),
        maxmem=6,
        router_mode=router_mode,
        confidence_threshold=confidence_threshold,
        fallback_bits=1,
        prefill_by_router=True,
        est_linear=True,
        est_params=(torch.tensor(1.0), torch.tensor(0.0)),
        est_T=torch.tensor(1.0),
        b_l=3,
        b_h=6,
    )


def threshold_inputs():
    x = torch.zeros(2, 32)
    x[0, 0] = 0.5
    x[1, 0] = 2.0
    return x


def test_dp_threshold_only_uses_low_and_high_threshold_branches(monkeypatch):
    install_fake_kernels(monkeypatch)
    linear = make_threshold_linear(router_mode="dp_threshold_only")

    y = linear(threshold_inputs())

    assert y.squeeze(-1).tolist() == [3.0, 6.0]
    assert linear.comp_count[3] == 1
    assert linear.comp_count[6] == 1
    assert linear.dp_threshold_token_count == 2
    assert linear.dp_threshold_high_count == 1
    assert linear.fallback_count == 0
    assert linear.dp_guard_count == 0


class StaticRouter(nn.Module):
    bits = [3, 4, 6]
    use_estimated_error = False

    def forward(self, x, layer_ids, estimated_error=None):
        return torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 10.0],
            ],
            dtype=x.dtype,
            device=x.device,
        )[: x.shape[0]]


def test_mlp_multibit_dp_guard_raises_router_bit_and_counts_separately(monkeypatch):
    install_fake_kernels(monkeypatch)
    linear = make_threshold_linear(
        router=StaticRouter(),
        router_mode="mlp_multibit_dp_guard",
        confidence_threshold=0.9,
    )

    y = linear(threshold_inputs().flip(0))

    assert y.squeeze(-1).tolist() == [6.0, 6.0]
    assert linear.comp_count[6] == 2
    assert linear.fallback_count == 1
    assert linear.dp_guard_count == 1
    assert linear.routed_token_count == 2
    assert linear.dp_threshold_token_count == 2
    assert linear.dp_threshold_high_count == 1


def test_mlp_multibit_dp_guard_phase_timers_report_and_clear(monkeypatch):
    install_fake_kernels(monkeypatch)
    linear = make_threshold_linear(
        router=StaticRouter(),
        router_mode="mlp_multibit_dp_guard",
        confidence_threshold=0.9,
    )
    linear.set_phase_timing_enabled(True)

    linear(threshold_inputs().flip(0))

    timing = linear.get_phase_timing_stats()
    for phase in ["router", "estimator", "grouping", "dequant_matmul", "total"]:
        assert timing[phase]["count"] > 0
        assert timing[phase]["wall_time_s"] >= 0
    assert timing["total"]["count"] == 1

    model = QAQDPLLMForCausalLM.__new__(QAQDPLLMForCausalLM)
    model.ap_linears = [linear]
    stats = QAQDPLLMForCausalLM.get_router_stats(model)

    assert "phase_timing" in stats
    assert stats["phase_timing"]["router"]["count"] == timing["router"]["count"]
    assert stats["per_layer"]["0.q_proj"]["phase_timing"]["total"]["count"] == 1

    linear.clear_stats()

    cleared = linear.get_phase_timing_stats()
    assert all(phase_stats["count"] == 0 for phase_stats in cleared.values())


def test_from_quantized_loads_t_d_from_estimator_results(monkeypatch, tmp_path):
    expected_T_d = {(0, "q_proj"): (3, 6, torch.tensor(1.0))}
    expected_linear_reg_d = {(0, "q_proj"): (torch.tensor(1.0), torch.tensor(0.0))}
    expected_jl_d = {}
    expected_max_mem_dict = {(0, "q_proj"): 6}

    torch.save(expected_T_d, tmp_path / "T_d.pt")
    torch.save(expected_linear_reg_d, tmp_path / "linear_reg_d.pt")
    torch.save(expected_jl_d, tmp_path / "jl_d.pt")
    torch.save(expected_max_mem_dict, tmp_path / "max_mem_dict.pt")

    captured = {}
    monkeypatch.setattr(QAQDPLLMForCausalLM, "_load_config", staticmethod(lambda *args, **kwargs: object()))

    def fake_init(self, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(QAQDPLLMForCausalLM, "__init__", fake_init)

    QAQDPLLMForCausalLM.from_quantized(
        "unused-model",
        estimator_results=str(tmp_path),
        router_mode="dp_threshold_only",
    )

    assert captured["T_d"].keys() == expected_T_d.keys()
    assert captured["T_d"][(0, "q_proj")][0:2] == (3, 6)
    torch.testing.assert_close(captured["T_d"][(0, "q_proj")][2], torch.tensor(1.0))
    assert captured["max_mem_dict"] == expected_max_mem_dict
    assert captured["linear_reg_d"].keys() == expected_linear_reg_d.keys()


def test_router_stats_reports_fallback_and_dp_guard_counts_separately():
    model = QAQDPLLMForCausalLM.__new__(QAQDPLLMForCausalLM)
    model.ap_linears = [
        SimpleNamespace(
            route_name="0.q_proj",
            comp_count={3: 1, 6: 1},
            fallback_count=1,
            dp_guard_count=1,
            dp_threshold_token_count=2,
            dp_threshold_high_count=1,
            routed_token_count=2,
            in_features=2,
            out_features=3,
        )
    ]

    stats = QAQDPLLMForCausalLM.get_router_stats(model)

    assert stats["total_fallbacks"] == 1
    assert stats["total_dp_guard_triggers"] == 1
    assert stats["fallback_fraction"] == 0.5
    assert stats["dp_guard_trigger_fraction"] == 0.5
    assert stats["dp_threshold_high_fraction"] == 0.5
    assert stats["per_layer"]["0.q_proj"]["fallback_count"] == 1
    assert stats["per_layer"]["0.q_proj"]["dp_guard_trigger_count"] == 1


def test_estimator_tensors_are_not_meta_buffers_under_empty_init(monkeypatch):
    from accelerate.big_modeling import init_empty_weights

    install_fake_kernels(monkeypatch)
    with init_empty_weights():
        linear = QAQDPLLM_Linear(
            in_features=32,
            out_features=1,
            supported_bits=[3, 4, 5, 6],
            router=None,
            route_id=0,
            route_name="0.q_proj",
            bias=False,
            precisions=[3, 4, 5, 6],
            dtype=torch.float32,
            device=torch.device("cpu"),
            router_mode="dp_threshold_only",
            est_linear=False,
            est_params=torch.ones(2, 32),
            est_T=torch.tensor(1.0),
            b_l=3,
            b_h=6,
        )

    assert "jl" not in linear.state_dict()
    assert "est_T" not in linear.state_dict()
    assert linear.jl.device.type == "cpu"
    assert linear.est_T.device.type == "cpu"
