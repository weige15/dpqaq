import sys
from argparse import Namespace
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmark_qaq_profile_batching import (
    Request,
    aggregate_router_stats,
    build_batch_plan,
    choose_batch,
    is_compatible,
    make_arrival_trace,
    profile_padding,
    request_stream_hash,
)


def request(request_id, arrival=0.0, scalar=5.0, predicted=(5.0, 5.0), observed=(5.0, 5.0), confidence=0.9, cutoff=0.5):
    return Request(
        request_id=request_id,
        dataset="sample",
        document_id=f"doc-{request_id}",
        prompt_length=4,
        continuation_length=3,
        prompt_ids=torch.arange(4),
        arrival_ms=arrival,
        predicted_scalar=scalar,
        predicted_profile=predicted,
        observed_profile=observed,
        classification_confidence=confidence,
        uncertainty_cutoff=cutoff,
    )


def test_arrival_trace_is_reproducible_and_hashes_identically():
    requests = [request("a"), request("b"), request("c", scalar=5.5)]
    first = make_arrival_trace(requests, seed=101, arrival_rate=20.0)
    second = make_arrival_trace(requests, seed=101, arrival_rate=20.0)

    assert [(item.request_id, item.arrival_ms) for item in first] == [
        (item.request_id, item.arrival_ms) for item in second
    ]
    assert request_stream_hash(first) == request_stream_hash(second)


def test_policy_compatibility_uses_real_profile_and_uncertainty_signals():
    first = request("a", predicted=(5.0, 5.0), observed=(5.0, 5.0))
    scalar_far = request("b", scalar=5.5)
    profile_far = request("c", predicted=(5.8, 5.8), observed=(5.8, 5.8))
    uncertain = request("d", confidence=0.1)

    assert not is_compatible(first, scalar_far, "scalar_predicted", 0.25, 0.25)
    assert not is_compatible(first, profile_far, "predicted_profile", 0.25, 0.25)
    assert not is_compatible(first, uncertain, "uncertainty_fallback", 0.25, 0.25)
    assert is_compatible(uncertain, request("e", confidence=0.2), "uncertainty_fallback", 0.25, 0.25)


def test_choose_batch_respects_wait_window_and_reports_predictor_overhead():
    requests = [request("a", arrival=0.0), request("b", arrival=10.0), request("c", arrival=20.0)]
    batch, start, scheduler_overhead, predictor_overhead = choose_batch(
        requests,
        current_time_ms=0.0,
        policy="scalar_predicted",
        max_batch_size=4,
        max_wait_ms=50.0,
        scalar_bucket_size=0.25,
        profile_threshold=0.25,
    )

    assert [item.request_id for item in batch] == ["a", "b", "c"]
    assert start == 50.0
    assert scheduler_overhead >= 0.0 and predictor_overhead >= 0.0


def test_profile_padding_is_max_shared_profile_padding():
    batch = [
        request("a", predicted=(4.0, 5.0)),
        request("b", predicted=(5.0, 5.0)),
    ]
    padding = profile_padding(batch, "predicted_profile")

    assert padding["mean_bits"] == 0.25
    assert padding["fraction"] == 1.0 / 19.0
    assert padding["max_span_bits"] == 0.5


def test_batch_plan_covers_each_request_once():
    requests = [request("a"), request("b", scalar=5.5), request("c", scalar=5.5)]
    args = Namespace(
        max_batch_size=2,
        max_wait_ms=50.0,
        scalar_bucket_size=0.25,
        profile_distance=0.25,
    )

    plan = build_batch_plan(requests, args, "scalar_predicted")
    flattened = [item.request_id for batch in plan for item in batch]

    assert sorted(flattened) == ["a", "b", "c"]
    assert len(flattened) == len(set(flattened))


def test_aggregate_router_stats_weights_effective_bits_and_merges_histograms():
    result = aggregate_router_stats([
        {
            "router_stats": {
                "total_tokens": 10,
                "average_selected_bit": 5.0,
                "effective_bits": 4.0,
                "total_fallbacks": 1,
                "total_dp_guard_triggers": 2,
                "per_layer": {"0.q_proj": {"bit_counts": {"4": 10}}},
            }
        },
        {
            "router_stats": {
                "total_tokens": 20,
                "average_selected_bit": 6.0,
                "effective_bits": 6.0,
                "total_fallbacks": 0,
                "total_dp_guard_triggers": 1,
                "per_layer": {"0.q_proj": {"bit_counts": {"6": 20}}},
            }
        },
    ])

    assert result["effective_bits"] == 16.0 / 3.0
    assert result["fallback_rate"] == 1.0 / 30.0
    assert result["dp_guard_rate"] == 0.1
    assert result["per_layer_bit_histogram"]["0.q_proj"] == {"4": 10, "6": 20}
