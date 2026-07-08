import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from any_precision.modules.QAQRouter import (
    QAQRouter,
    load_qaq_router_checkpoint,
    save_qaq_router_checkpoint,
)
import scripts.train_qaq_router as train_qaq_router


def test_qaq_router_forward_shape():
    router = QAQRouter(
        hidden_size=4,
        input_feature_dim=4,
        num_layers=3,
        bits=[3, 4, 6],
        router_hidden_dim=8,
        router_layers=2,
    )

    x = torch.randn(2, 5, 4)
    logits = router(x, layer_ids=1)

    assert logits.shape == (10, 3)


def test_qaq_router_handles_different_input_feature_dim_values():
    padded_router = QAQRouter(
        hidden_size=8,
        input_feature_dim=6,
        num_layers=2,
        bits=[3, 6],
        router_hidden_dim=8,
        router_layers=1,
    )
    smaller_x = torch.randn(4, 3)

    assert padded_router(smaller_x, layer_ids=0).shape == (4, 2)

    exact_router = QAQRouter(
        hidden_size=8,
        input_feature_dim=3,
        num_layers=2,
        bits=[3, 6],
        router_hidden_dim=8,
        router_layers=1,
    )
    assert exact_router(smaller_x, layer_ids=torch.tensor([0, 1, 0, 1])).shape == (4, 2)

    with pytest.raises(ValueError, match="expected input feature dim"):
        exact_router(torch.randn(4, 4), layer_ids=0)


def test_estimated_error_feature_is_required_when_configured():
    router = QAQRouter(
        hidden_size=4,
        input_feature_dim=4,
        num_layers=1,
        bits=[3, 6],
        use_estimated_error=True,
    )
    x = torch.randn(3, 4)

    with pytest.raises(ValueError, match="estimated_error is required"):
        router(x, layer_ids=0)

    logits = router(x, layer_ids=0, estimated_error=torch.tensor([0.1, 0.2, 0.3]))
    assert logits.shape == (3, 2)


def test_checkpoint_save_load_roundtrip(tmp_path):
    torch.manual_seed(0)
    router = QAQRouter(
        hidden_size=4,
        input_feature_dim=4,
        num_layers=2,
        bits=[3, 4, 6],
        router_hidden_dim=8,
        router_layers=2,
    ).eval()
    route_map = [
        {"route_id": 0, "layer": 0, "parent": "self_attn", "name": "q_proj"},
        {"route_id": 1, "layer": 0, "parent": "self_attn", "name": "k_proj"},
    ]
    x = torch.randn(2, 4)
    expected_logits = router(x, layer_ids=torch.tensor([0, 1]))

    checkpoint_path = tmp_path / "router.pt"
    save_qaq_router_checkpoint(
        str(checkpoint_path),
        router,
        training_config={"seed": 0},
        label_mode="multibit",
        error_threshold=0.01,
        target_bits=4.5,
        route_map=route_map,
        stats={"unit_test": True},
    )

    loaded_router, metadata = load_qaq_router_checkpoint(str(checkpoint_path))
    actual_logits = loaded_router(x, layer_ids=torch.tensor([0, 1]))

    assert metadata["format"] == "qaq_router_v1"
    assert metadata["candidate_bits"] == [3, 4, 6]
    assert metadata["route_map"] == route_map
    assert metadata["label_mode"] == "multibit"
    assert metadata["error_threshold"] == 0.01
    assert metadata["target_bits"] == 4.5
    torch.testing.assert_close(actual_logits, expected_logits)


def test_duplicate_bits_are_rejected():
    with pytest.raises(ValueError, match="bits must be unique"):
        QAQRouter(hidden_size=4, num_layers=1, bits=[3, 3])


def test_binary_mode_requires_exactly_two_bits(monkeypatch, tmp_path):
    monkeypatch.setattr(train_qaq_router, "dequant_kbit", object())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_qaq_router.py",
            "--model_path",
            "unused-original",
            "--ap_model_path",
            "unused-anyprecision",
            "--save_path",
            str(tmp_path / "router.pt"),
            "--label_mode",
            "binary",
            "--bits",
            "3",
            "4",
            "6",
            "--device",
            "cpu",
        ],
    )

    with pytest.raises(ValueError, match="binary requires exactly two"):
        train_qaq_router.main()


class FakeLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("qweight", torch.empty(1))
        self.register_buffer("lut3", torch.tensor([[0.95, 0.0, 0.0]]))
        self.register_buffer("lut4", torch.tensor([[1.0, 0.9, 0.0]]))
        self.register_buffer("lut6", torch.tensor([[1.0, 1.0, 1.0]]))


def test_label_generation_chooses_smallest_safe_bit(monkeypatch):
    def fake_dequant_kbit(qweight, lut, bit):
        return lut

    monkeypatch.setattr(train_qaq_router, "dequant_kbit", fake_dequant_kbit)
    x_cpu = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    labels = train_qaq_router.make_labels_for_linear(
        x_cpu=x_cpu,
        linear=FakeLinear(),
        bits=[3, 4, 6],
        reference_bit=6,
        error_threshold=0.2,
        label_mode="multibit",
        batch_size=2,
        device=torch.device("cpu"),
    )

    assert labels.tolist() == [0, 1, 2]
