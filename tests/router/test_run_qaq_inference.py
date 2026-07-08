from pathlib import Path
from types import SimpleNamespace
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_qaq_inference import generate_and_report


class Encoded(dict):
    def __getattr__(self, name):
        return self[name]

    def to(self, device):
        return self


class FakeTokenizer:
    def __call__(self, prompts, return_tensors, padding):
        assert return_tensors == "pt"
        assert padding is True
        return Encoded({
            "input_ids": torch.ones((len(prompts), 3), dtype=torch.long),
            "attention_mask": torch.ones((len(prompts), 3), dtype=torch.long),
        })

    def batch_decode(self, generated, skip_special_tokens):
        assert skip_special_tokens is True
        return ["decoded"] * generated.shape[0]


class FakeModel:
    def __init__(self):
        self.counter = 0
        self.events = []

    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self

    def clear_router_stats(self):
        self.events.append("clear")
        self.counter = 0

    def get_router_stats(self):
        self.events.append(f"stats:{self.counter}")
        return {"counter": self.counter}

    def generate(self, **kwargs):
        assert kwargs["router_mode"] == "mlp_multibit"
        self.events.append("generate")
        self.counter += 1
        batch = kwargs["input_ids"].shape[0]
        return torch.ones((batch, 5), dtype=torch.long)

    def __call__(self, **kwargs):
        assert kwargs["router_mode"] == "mlp_multibit"
        self.events.append("forward")
        self.counter += 10
        batch, seq_len = kwargs["input_ids"].shape
        return SimpleNamespace(logits=torch.zeros((batch, seq_len, 4)))


def test_generate_and_report_separates_generation_and_sanity_stats():
    model = FakeModel()

    report = generate_and_report(
        model,
        FakeTokenizer(),
        prompts=["prompt"],
        device="cpu",
        max_new_tokens=2,
        router_mode="mlp_multibit",
    )

    assert report["generation_latency_s"] >= 0
    assert report["generation_tokens_per_s"] >= 0
    assert report["generation_router_stats"] == {"counter": 1}
    assert report["finite_logits"] is True
    assert report["sanity_check_router_stats"] == {"counter": 10}
    assert report["outputs"] == ["decoded"]
    assert model.events == [
        "clear",
        "generate",
        "stats:1",
        "clear",
        "forward",
        "stats:10",
    ]
