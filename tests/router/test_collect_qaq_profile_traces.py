import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_qaq_profile_traces import (
    TRACE_SCHEMA_VERSION,
    UNVALIDATED,
    RequestSpec,
    build_trace_record,
    load_requests,
    run_request,
)


class Encoded(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "</s>"

    def __call__(self, prompts, return_tensors, padding):
        assert return_tensors == "pt"
        assert padding is True
        return Encoded({
            "input_ids": torch.ones((len(prompts), 3), dtype=torch.long),
            "attention_mask": torch.ones((len(prompts), 3), dtype=torch.long),
        })

    def batch_decode(self, generated, skip_special_tokens):
        assert skip_special_tokens is True
        return [f"decoded_{idx}" for idx in range(generated.shape[0])]


class FakeModel:
    def __init__(self):
        self.events = []
        self.counter = 0

    def clear_router_stats(self):
        self.events.append("clear")
        self.counter = 0

    def get_router_stats(self):
        self.events.append(f"stats:{self.counter}")
        return {
            "average_selected_bit": 4.5,
            "effective_bits": 4.25,
            "fallback_fraction": 0.1,
            "dp_guard_trigger_fraction": 0.05,
            "total_fallbacks": self.counter,
            "total_dp_guard_triggers": 1,
            "per_layer": {
                "0.q_proj": {
                    "bit_counts": {"3": 2, "6": 1},
                    "fallback_count": self.counter,
                    "dp_guard_trigger_count": 1,
                }
            },
        }

    def generate(self, **kwargs):
        assert kwargs["router_mode"] == "mlp_multibit"
        self.events.append("generate")
        self.counter = 2
        batch = kwargs["input_ids"].shape[0]
        return torch.ones((batch, 5), dtype=torch.long)

    def __call__(self, **kwargs):
        assert kwargs["router_mode"] == "mlp_multibit"
        self.events.append("forward")
        self.counter = 10
        batch, seq_len = kwargs["input_ids"].shape
        return SimpleNamespace(logits=torch.zeros((batch, seq_len, 4)))


def make_args(**overrides):
    values = {
        "prompt": None,
        "prompt_file": None,
        "max_requests": None,
        "arrival_interval_s": 0.5,
        "workload_type": "chat",
        "qos_deadline_ms": None,
        "reference_mode": "fixed_high",
        "router_mode": "qaq",
        "max_new_tokens": 2,
        "device": "cpu",
        "check_finite_logits": False,
        "include_text": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_load_requests_accepts_prompts_and_jsonl(tmp_path):
    prompt_file = tmp_path / "prompts.jsonl"
    prompt_file.write_text(
        json.dumps({
            "request_id": "json_req",
            "prompt": "from json",
            "arrival_time_s": 7.0,
            "workload_type": "math",
            "qos_deadline_ms": 1000,
            "target_output_length_tokens": 8,
        })
        + "\n"
    )
    args = make_args(prompt=["from arg"], prompt_file=str(prompt_file))

    requests = load_requests(args)

    assert [request.request_id for request in requests] == ["req_000000", "json_req"]
    assert requests[0].arrival_time_s == 0.0
    assert requests[0].workload_type == "chat"
    assert requests[0].qos_deadline_ms == UNVALIDATED
    assert requests[1].arrival_time_s == 7.0
    assert requests[1].workload_type == "math"
    assert requests[1].qos_deadline_ms == 1000.0
    assert requests[1].target_output_length_tokens == 8


def test_run_request_keeps_generation_stats_separate_from_finite_logit_check():
    args = make_args(router_mode="qaq", check_finite_logits=True)
    model = FakeModel()

    result = run_request(
        model=model,
        tokenizer=FakeTokenizer(),
        request=RequestSpec(
            request_id="req",
            prompt="hello",
            arrival_time_s=0.0,
            workload_type="chat",
            qos_deadline_ms=UNVALIDATED,
            target_output_length_tokens=UNVALIDATED,
            reference_mode="fixed_high",
        ),
        args=args,
    )

    assert result["observed_output_length_tokens"] == 2
    assert result["generation_router_stats"]["total_fallbacks"] == 2
    assert result["finite_logits"] is True
    assert model.events == [
        "clear",
        "generate",
        "stats:2",
        "clear",
        "forward",
        "clear",
    ]


def test_build_trace_record_emits_required_schema_fields():
    args = make_args(include_text=True)
    request = RequestSpec(
        request_id="req_1",
        prompt="prompt text",
        arrival_time_s=1.5,
        workload_type="code",
        qos_deadline_ms=20.0,
        target_output_length_tokens=4,
        reference_mode="fixed_high",
    )
    result = {
        "gpu_execution_ms": 12.0,
        "generated_text": "answer",
        "generated_text_hash": "hash",
        "observed_output_length_tokens": 2,
        "finite_logits": UNVALIDATED,
        "generation_router_stats": {
            "average_selected_bit": 4.5,
            "effective_bits": 4.25,
            "fallback_fraction": 0.1,
            "dp_guard_trigger_fraction": 0.05,
            "total_fallbacks": 2,
            "total_dp_guard_triggers": 1,
            "per_layer": {"0.q_proj": {"bit_counts": {"3": 2, "6": 1}}},
        },
    }
    metadata = {
        "created_at": "2026-07-09T00:00:00+00:00",
        "git_commit": "abc",
        "ap_model_path": "/model",
        "router_checkpoint": "/router.pt",
        "estimator_results": "/estimator",
        "candidate_bits": [3, 4, 5, 6],
        "router_mode": "mlp_multibit",
        "last_prompt_length_tokens": 3,
    }

    record = build_trace_record(request, result, args, metadata)

    assert record["trace_schema_version"] == TRACE_SCHEMA_VERSION
    assert record["request_id"] == "req_1"
    assert record["prompt_length_tokens"] == 3
    assert record["observed_output_length_tokens"] == 2
    assert record["average_selected_bit"] == 4.5
    assert record["effective_bits"] == 4.25
    assert record["per_layer_bit_counts"] == {"0.q_proj": {"3": 2, "6": 1}}
    assert record["fallback_count"] == 2
    assert record["dp_guard_trigger_count"] == 1
    assert record["queue_delay_ms"] == 0.0
    assert record["gpu_execution_ms"] == 12.0
    assert record["end_to_end_latency_ms"] == 12.0
    assert record["deadline_missed"] is False
    assert record["predicted_scalar_bit_budget"] == UNVALIDATED
    assert record["transfer_bytes_per_token"] == UNVALIDATED
    assert record["quality_metric_value"] == UNVALIDATED
    assert record["prompt_text"] == "prompt text"
    assert record["generated_text"] == "answer"
