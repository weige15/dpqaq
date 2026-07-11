import sys
from pathlib import Path

import pytest
import torch
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_qaq_online_scheduler_replay import (
    ReplayRequest,
    choose_online_batch,
    compatible,
    deterministic_arrivals,
    execute_batch,
    predictor_map,
    profile_distance,
    summarize_scenario,
)


def request(request_id, prompt=128, continuation=32, profile=(4.0, 4.0), confidence=0.9, cutoff=0.5, arrival=0.0):
    return ReplayRequest(
        request_id=request_id,
        dataset="sample",
        document_id=f"doc-{request_id}",
        prompt_length=prompt,
        continuation_length=continuation,
        prompt_ids=torch.arange(prompt),
        arrival_ms=arrival,
        predicted_profile=profile,
        classification_confidence=confidence,
        uncertainty_cutoff=cutoff,
    )


def test_primary_policy_compatibility_enforces_registered_buckets_and_fallback_lane():
    first = request("a")
    mixed_prompt = request("b", prompt=512)
    mixed_continuation = request("c", continuation=128)
    distant = request("d", profile=(5.0, 5.0))
    uncertain = request("e", confidence=0.1)

    assert compatible(first, mixed_prompt, "ordinary_fcfs")
    assert not compatible(first, mixed_prompt, "length_fcfs")
    assert not compatible(first, mixed_continuation, "ordinary_fcfs")
    assert profile_distance(first, distant) == 1.0
    assert not compatible(first, distant, "predicted_block_fallback_lane")
    assert not compatible(first, uncertain, "predicted_block_fallback_lane")
    assert compatible(uncertain, request("f", confidence=0.2), "predicted_block_fallback_lane")


def test_online_batch_waits_until_full_or_registered_fifty_ms():
    full = [request(str(i), arrival=float(i * 10)) for i in range(4)]
    batch, start, overhead = choose_online_batch(full, current_time_ms=0.0, policy="ordinary_fcfs")
    assert [item.request_id for item in batch] == ["0", "1", "2", "3"]
    assert start == 30.0
    assert overhead >= 0.0

    partial = [request("a", arrival=5.0), request("b", arrival=20.0)]
    batch, start, _ = choose_online_batch(partial, current_time_ms=0.0, policy="ordinary_fcfs")
    assert len(batch) == 2
    assert start == 55.0


def test_arrivals_are_reproducible_and_preserve_all_requests():
    requests = [
        request(f"a{i}", prompt=128, continuation=32) for i in range(4)
    ] + [
        request(f"b{i}", prompt=512, continuation=128) for i in range(4)
    ]
    first = deterministic_arrivals(requests, "sample", seed=101, arrival_rate=10.0)
    second = deterministic_arrivals(requests, "sample", seed=101, arrival_rate=10.0)

    assert [(item.request_id, item.arrival_ms) for item in first] == [
        (item.request_id, item.arrival_ms) for item in second
    ]
    assert {item.request_id for item in first} == {item.request_id for item in requests}
    assert first[0].arrival_ms == 0.0
    assert all(a.arrival_ms <= b.arrival_ms for a, b in zip(first, first[1:]))


def test_predictor_seeds_pair_by_index_with_registered_scheduler_seeds():
    analysis = {
        "h2_predecode_predictability": {
            "test_by_dataset": {
                "sample": {
                    str(seed): {
                        "minimum_safe_precision_classifier": {
                            "uncertainty_cutoff_for_90pct_calibration_coverage": seed / 100,
                        },
                        "predictions": [{"request_id": f"from-{seed}"}],
                    }
                    for seed in (17, 29, 43)
                }
            }
        }
    }

    for scheduling_seed, predictor_seed in zip((101, 202, 303), (17, 29, 43), strict=True):
        predictions, cutoff = predictor_map(analysis, "sample", scheduling_seed)
        assert list(predictions) == [f"from-{predictor_seed}"]
        assert cutoff == predictor_seed / 100

    with pytest.raises(ValueError, match="unregistered scheduling seed"):
        predictor_map(analysis, "sample", 17)


def test_scenario_summary_aggregates_lane_and_route_histograms():
    requests = [{
        "end_to_end_latency_ms": 20.0,
        "queue_delay_ms": 5.0,
        "ttft_ms": 8.0,
        "tpot_ms": 2.0,
        "generated_tokens": 3,
        "deadline_missed": False,
    }]
    batches = [{
        "batch_size": 1,
        "lane_id": "ordinary",
        "prompt_token_slots": 4,
        "prompt_nonpadding_tokens": 3,
        "scheduler_cpu_overhead_ms": 0.1,
        "cuda_memory": {"max_allocated_bytes": 10, "max_reserved_bytes": 20},
        "router_stats": {
            "effective_bits": 5.0,
            "average_selected_bit": 5.0,
            "total_fallbacks": 0,
            "total_dp_guard_triggers": 1,
            "per_layer": {"0.q_proj": {"bit_counts": {"5": 3}}},
        },
    }]

    summary = summarize_scenario(requests, batches, 0.0, 20.0)

    assert summary["lane_occupancy"] == {"ordinary": 1}
    assert summary["per_layer_bit_histogram"] == {"0.q_proj": {"5": 3}}
    assert summary["total_dp_guard_triggers"] == 1


class FakeCachedModel:
    def __init__(self):
        self.calls = []

    def clear_router_stats(self):
        pass

    def get_router_stats(self):
        return {"average_selected_bit": 5.0, "effective_bits": 5.0}

    def __call__(self, **kwargs):
        self.calls.append({
            "input_shape": tuple(kwargs["input_ids"].shape),
            "position_ids": kwargs["position_ids"].clone(),
            "attention_mask": kwargs["attention_mask"].clone(),
        })
        batch = kwargs["input_ids"].shape[0]
        logits = torch.zeros((batch, kwargs["input_ids"].shape[1], 8))
        logits[..., 3] = 1.0
        return SimpleNamespace(logits=logits, past_key_values=("cache", len(self.calls)))


def test_manual_cached_decode_uses_left_padding_aware_position_ids():
    model = FakeCachedModel()
    short = request("short", prompt=2, continuation=3)
    short.prompt_ids = torch.tensor([10, 11])
    long = request("long", prompt=4, continuation=3)
    long.prompt_ids = torch.tensor([20, 21, 22, 23])

    result = execute_batch(model, [short, long], "mlp_multibit_dp_guard", 0, torch.device("cpu"))

    assert model.calls[0]["position_ids"].tolist() == [[0, 0, 0, 1], [0, 1, 2, 3]]
    assert model.calls[1]["position_ids"].tolist() == [[2], [4]]
    assert model.calls[2]["position_ids"].tolist() == [[3], [5]]
    assert result["generated_token_slots"] == 6
    assert set(result["generated_token_sha256"]) == {"short", "long"}
