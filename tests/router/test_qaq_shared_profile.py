import importlib
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

profile_module = importlib.import_module("any_precision.modules.QAQProfile")
qaq_linear_module = importlib.import_module("any_precision.modules.QAQDPLLM_Linear")
from any_precision.modules.QAQDPLLMForCausalLM import QAQDPLLMForCausalLM


def install_fake_kernels(monkeypatch):
    def fake_matmul_kbit(x, qweight, lut, bit):
        return torch.full((*x.shape[:-1], qweight.shape[1]), float(bit), dtype=x.dtype, device=x.device)

    def fake_dequant_kbit(qweight, lut, bit):
        return torch.full((qweight.shape[1], 32), float(bit), dtype=lut.dtype, device=lut.device)

    monkeypatch.setattr(qaq_linear_module, "matmul_kbit", fake_matmul_kbit)
    monkeypatch.setattr(qaq_linear_module, "dequant_kbit", fake_dequant_kbit)


def make_linear(route_name="0.q_proj", router=None, router_mode="shared_profile", maxmem=6, precisions=None):
    if precisions is None:
        precisions = [3, 4, 5, 6]
    return qaq_linear_module.QAQDPLLM_Linear(
        in_features=32,
        out_features=1,
        supported_bits=[3, 4, 5, 6],
        router=router,
        route_id=0,
        route_name=route_name,
        bias=False,
        precisions=precisions,
        dtype=torch.float32,
        device=torch.device("cpu"),
        maxmem=maxmem,
        router_mode=router_mode,
        prefill_by_router=True,
        est_linear=True,
        est_params=(torch.tensor(1.0), torch.tensor(0.0)),
        est_T=torch.tensor(1.0),
        b_l=3,
        b_h=6,
    )


def route_map():
    return [
        {"route_id": 0, "layer": 0, "parent": "self_attn", "name": "q_proj", "route_name": "0.q_proj"},
        {"route_id": 1, "layer": 4, "parent": "self_attn", "name": "q_proj", "route_name": "4.q_proj"},
    ]


def valid_bits():
    return {"0.q_proj": [3, 4, 5, 6], "4.q_proj": [3, 5]}


def test_profile_helpers_validate_compose_project_and_account():
    build = profile_module.build_max_shared_profile(
        [("a", (4.0, 6.0)), ("b", (5.0, 3.0))],
        layer_group_size=4,
        route_map=route_map(),
        route_valid_bits=valid_bits(),
    )

    assert build["shared_group_profile"] == [5.0, 6.0]
    assert build["shared_route_profile"] == {"0.q_proj": 5, "4.q_proj": 5}
    assert build["capped_routes"]["4.q_proj"]["max_valid_bit"] == 5
    assert build["request_projected_route_profiles"]["a"] == {"0.q_proj": 4, "4.q_proj": 5}

    accounting = profile_module.account_profile_execution(
        build,
        {"0.q_proj": 5, "4.q_proj": 5},
        route_map(),
        valid_bits(),
    )
    assert accounting["decision_count"] == 4
    assert accounting["profile_under_precision_count"] == 0
    assert accounting["profile_exact_precision_count"] == 2
    assert accounting["profile_over_precision_count"] == 2
    assert accounting["signed_bit_gap_sum"] == 3
    assert accounting["absolute_bit_gap_sum"] == 3

    assert profile_module.project_demand_to_valid_bit(4.0, [3, 4, 5, 6]) == (4, False)
    assert profile_module.project_demand_to_valid_bit(4.1, [3, 4, 5, 6]) == (5, False)
    assert profile_module.project_demand_to_valid_bit(5.7, [3, 4, 5]) == (5, True)
    assert profile_module.project_demand_to_valid_bit(2.0, [3, 4, 5, 6]) == (3, False)
    with pytest.raises(ValueError, match="duplicate route"):
        profile_module.validate_route_profile(
            [("0.q_proj", 4), ("0.q_proj", 5), ("4.q_proj", 5)], route_map(), valid_bits()
        )

def test_profile_helpers_reject_invalid_profiles_and_route_maps():
    with pytest.raises(ValueError, match="finite"):
        profile_module.validate_group_profile([4.0, float("nan")], expected_dimension=2, layer_group_size=4)
    with pytest.raises(ValueError, match="dimension"):
        profile_module.validate_group_profile([4.0], expected_dimension=2, layer_group_size=4)
    with pytest.raises(ValueError, match="duplicate"):
        profile_module.validate_route_map(route_map() + [route_map()[0]], layer_group_size=4, profile_dimension=2)
    with pytest.raises(ValueError, match="missing"):
        profile_module.validate_route_profile(
            {"0.q_proj": 4}, route_map(), valid_bits(),
        )
    with pytest.raises(ValueError, match="unknown"):
        profile_module.validate_route_profile(
            {"0.q_proj": 4, "4.q_proj": 5, "unknown": 3}, route_map(), valid_bits(),
        )
    with pytest.raises(ValueError, match="valid bits"):
        profile_module.validate_route_profile(
            {"0.q_proj": 4, "4.q_proj": 4}, route_map(), valid_bits(),
        )
    with pytest.raises(ValueError, match="unknown routes"):
        profile_module.project_group_profile_to_routes(
            [4.0, 5.0], route_map(), {**valid_bits(), "unknown": [3]}, layer_group_size=4
        )


def test_max_profile_is_monotonic_and_singleton_is_not_special_cased():
    singleton = profile_module.build_max_shared_profile(
        [("only", (4.1, 5.0))], 4, route_map(), valid_bits()
    )
    pair = profile_module.build_max_shared_profile(
        [("only", (4.1, 5.0)), ("other", (3.0, 4.0))], 4, route_map(), valid_bits()
    )
    assert singleton["shared_group_profile"] == [4.1, 5.0]
    for route, bit in singleton["request_projected_route_profiles"]["only"].items():
        assert singleton["shared_route_profile"][route] >= bit
    for request_id, target in pair["request_projected_route_profiles"].items():
        for route, bit in target.items():
            assert pair["shared_route_profile"][route] >= bit, request_id


class ExplodingRouter(nn.Module):
    bits = [3, 4, 5, 6]
    use_estimated_error = False

    def forward(self, *args, **kwargs):
        raise AssertionError("shared-profile execution must not call the router")


def test_linear_shared_profile_uses_one_bit_for_every_row_and_bypasses_router(monkeypatch):
    install_fake_kernels(monkeypatch)
    linear = make_linear(router=ExplodingRouter())
    linear._choose_router_bits = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("router path must not run")
    )
    linear._choose_dp_threshold_bits = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("DP path must not run")
    )
    observed = []
    linear.set_decision_observer(lambda module, x, bits: observed.append(bits.clone()))
    x = torch.zeros(3, 32)
    x[:, 0] = 1.0

    linear.set_shared_precision(5)
    y = linear(x)

    assert y.squeeze(-1).tolist() == [5.0, 5.0, 5.0]
    assert linear.comp_count[5] == 3
    assert linear.shared_profile_token_count == 3
    assert linear.routed_token_count == 0
    assert linear.fallback_count == 0
    assert linear.dp_guard_count == 0
    assert [bits.tolist() for bits in observed] == [[5, 5, 5]]


def test_linear_shared_mode_requires_a_supplied_bit(monkeypatch):
    install_fake_kernels(monkeypatch)
    with pytest.raises(RuntimeError, match="shared precision"):
        make_linear()(torch.zeros(2, 32))


def make_fake_model(linears, mode="mlp_multibit"):
    model = QAQDPLLMForCausalLM.__new__(QAQDPLLMForCausalLM)
    model.ap_linears = linears
    model.route_map = route_map()
    model.router_mode = mode
    model._shared_profile_active = False
    model._shared_route_profile = None
    return model


def test_model_shared_profile_restores_state_after_success_and_exception(monkeypatch):
    install_fake_kernels(monkeypatch)
    first = make_linear("0.q_proj", router_mode="mlp_multibit")
    second = make_linear("4.q_proj", router_mode="mlp_multibit", maxmem=5, precisions=[3, 5])
    model = make_fake_model([first, second])
    profile = {"0.q_proj": 4, "4.q_proj": 5}

    with model.shared_profile(profile):
        assert model.router_mode == "shared_profile"
        assert first.shared_precision == 4
        assert model.batch_policy == "shared_profile"
        assert second.shared_precision == 5
    assert model.router_mode == "mlp_multibit"
    assert first.shared_precision is None and second.shared_precision is None
    assert model.batch_policy == "group"
    assert first.batch_policy == "group" and second.batch_policy == "group"
    assert first.router_mode == "mlp_multibit" and second.router_mode == "mlp_multibit"

    with pytest.raises(RuntimeError, match="boom"):
        with model.shared_profile(profile):
            raise RuntimeError("boom")

    assert model.router_mode == "mlp_multibit"
    assert model.batch_policy == "group"
    assert first.batch_policy == "group" and second.batch_policy == "group"
    assert first.shared_precision is None and second.shared_precision is None

def test_model_stats_report_actual_shared_execution_without_routed_rows(monkeypatch):
    install_fake_kernels(monkeypatch)
    first = make_linear("0.q_proj", router_mode="mlp_multibit")
    second = make_linear("4.q_proj", router_mode="mlp_multibit", maxmem=5, precisions=[3, 5])
    model = make_fake_model([first, second])
    with model.shared_profile({"0.q_proj": 4, "4.q_proj": 5}):
        first(torch.zeros(2, 32))
        second(torch.zeros(2, 32))

    stats = model.get_router_stats()
    assert stats["shared_profile_execution"] is True
    assert stats["total_shared_profile_tokens"] == 4
    assert stats["total_fallbacks"] == 0
    assert stats["total_dp_guard_triggers"] == 0
    assert {bit: count for bit, count in stats["per_layer"]["0.q_proj"]["bit_counts"].items() if count} == {"4": 2}
    assert {bit: count for bit, count in stats["per_layer"]["4.q_proj"]["bit_counts"].items() if count} == {"5": 2}
    assert stats["per_layer"]["0.q_proj"]["routed_token_count"] == 0


def test_model_shared_profile_rejects_fixed_modes_and_invalid_route_bits():
    first = make_linear("0.q_proj", router_mode="fixed_high")
    second = make_linear("4.q_proj", router_mode="fixed_high", maxmem=5, precisions=[3, 5])
    model = make_fake_model([first, second], mode="fixed_high")
    with pytest.raises(RuntimeError, match="incompatible"):
        with model.shared_profile({"0.q_proj": 4, "4.q_proj": 5}):
            pass

    model.router_mode = "mlp_multibit"
    with pytest.raises(ValueError, match="complete route profile"):
        with model.shared_profile(None):
            pass

    with pytest.raises(ValueError, match="valid bits"):
        with model.shared_profile({"0.q_proj": 4, "4.q_proj": 4}):
            pass
