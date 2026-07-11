from pathlib import Path
from types import SimpleNamespace
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_qaq_decode_traces import (
    TRACE_SCHEMA_VERSION,
    DecodeRouteObserver,
    collect_autoregressive_request,
)


class Encoded(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    eos_token_id = 99

    def __call__(self, prompts, return_tensors, padding):
        assert return_tensors == "pt"
        assert padding is True
        return Encoded(
            {
                "input_ids": torch.tensor([[10, 11, 12]], dtype=torch.long),
                "attention_mask": torch.ones((1, 3), dtype=torch.long),
            }
        )

    def batch_decode(self, token_ids, skip_special_tokens):
        assert skip_special_tokens is True
        return ["prompt generated"]


class FakeModel:
    def __init__(self):
        self.observer = None
        self.mode = None
        self.events = []
        self.total_tokens = 0
        self.fallbacks = 0
        self.guards = 0
        self.bit_counts = {"0.q_proj": {"3": 0, "6": 0}}

    def eval(self):
        return self

    def set_decision_observer(self, observer=None):
        self.observer = observer

    def clear_router_stats(self):
        self.events.append("clear")
        self.total_tokens = 0
        self.fallbacks = 0
        self.guards = 0
        self.bit_counts = {"0.q_proj": {"3": 0, "6": 0}}

    def get_router_stats(self):
        self.events.append(f"stats:{self.total_tokens}")
        return {
            "total_tokens": self.total_tokens,
            "total_fallbacks": self.fallbacks,
            "total_dp_guard_triggers": self.guards,
            "per_layer": {
                "0.q_proj": {
                    "bit_counts": dict(self.bit_counts["0.q_proj"]),
                    "fallback_count": self.fallbacks,
                    "dp_guard_trigger_count": self.guards,
                }
            },
        }

    def __call__(self, **kwargs):
        self.mode = kwargs["router_mode"]
        is_prefill = kwargs.get("past_key_values") is None
        selected_bit = 6 if self.mode == "fixed_high" else 3
        if self.mode == "mlp_multibit_dp_guard" and not is_prefill:
            selected_bit = 6
            self.guards += 1
        if self.mode == "mlp_multibit" and not is_prefill:
            self.fallbacks += 1

        rows = kwargs["input_ids"].shape[1]
        self.total_tokens += rows
        self.bit_counts["0.q_proj"][str(selected_bit)] += rows
        if self.observer is not None:
            linear = SimpleNamespace(route_name="0.q_proj")
            self.observer(linear, kwargs["input_ids"].float().unsqueeze(-1), torch.full((rows,), selected_bit))

        vocab_size = 100
        logits = torch.zeros((1, rows, vocab_size))
        logits[..., 13] = 1.0
        return SimpleNamespace(logits=logits, past_key_values=("cache", self.total_tokens))


def test_observer_keeps_decode_profiles_out_of_prefill():
    observer = DecodeRouteObserver()

    observer(torch.tensor(0), torch.ones((3, 1)), torch.tensor([6, 6, 6]))
    assert observer.token_profiles == []

    observer.begin_token(1)
    observer(SimpleNamespace(route_name="0.q_proj"), torch.ones((1, 1)), torch.tensor([3]))
    observer.finish_token(13, 0.25, {
        "fallback_count": 1,
        "dp_guard_trigger_count": 0,
        "fallback_events_by_layer": {"0.q_proj": 1},
        "dp_guard_events_by_layer": {},
    })

    assert observer.token_profiles[0]["selected_bits_by_layer"] == {"0.q_proj": [3]}
    assert observer.token_profiles[0]["fallback_count"] == 1


def test_collect_request_clears_prefill_before_decode_and_records_profiles():
    model = FakeModel()
    result = collect_autoregressive_request(
        model=model,
        tokenizer=FakeTokenizer(),
        prompt="hello",
        mode="mlp_multibit_dp_guard",
        device="cpu",
        max_new_tokens=4,
        seed=7,
    )

    assert result["trace_schema_version"] == TRACE_SCHEMA_VERSION
    assert result["output_token_count"] == 4
    assert result["decode_token_count"] == 3
    assert result["prefill_router_stats"]["total_tokens"] == 3
    assert result["decode_router_stats"]["total_tokens"] == 3
    assert result["decode_router_stats"]["total_dp_guard_triggers"] == 3
    assert len(result["per_token_route_profiles"]) == 3
    assert all(
        profile["selected_bits_by_layer"] == {"0.q_proj": [6]}
        for profile in result["per_token_route_profiles"]
    )
    assert len(result["generated_token_hashes"]) == 4
    assert result["generated_token_ids_sha256"]
    assert result["ttft_s"] == result["prefill_time_s"]
    assert result["tpot_s"] >= 0
    assert result["quality_evaluation"] == "separate_teacher_forced_artifact_required"
    assert model.events[0] == "clear"
    assert "clear" in model.events[1:]


def test_fixed_modes_share_prompt_and_produce_distinct_profiles():
    tokenizer = FakeTokenizer()
    results = {
        mode: collect_autoregressive_request(
            model=FakeModel(),
            tokenizer=tokenizer,
            prompt="identical prompt",
            mode=mode,
            device="cpu",
            max_new_tokens=3,
            seed=0,
        )
        for mode in ("fixed_low", "fixed_high")
    }

    assert results["fixed_low"]["prompt_length_tokens"] == results["fixed_high"]["prompt_length_tokens"]
    assert results["fixed_low"]["generated_token_ids_sha256"] == results["fixed_high"]["generated_token_ids_sha256"]
    assert results["fixed_low"]["decode_selected_bit_profile"]["overall"] == {"3": 2}
    assert results["fixed_high"]["decode_selected_bit_profile"]["overall"] == {"6": 2}
