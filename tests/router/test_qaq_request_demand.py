import json
from collections import Counter
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_qaq_request_demand import (
    exact_profile_oracle,
    feature_matrix,
    predictor_analysis,
    scheduler_oracle_analysis,
)
from scripts.qaq_request_demand_protocol import (
    aggregate_records,
    atomic_write_jsonl,
    document_partition,
    ensure_shard_metadata,
    enumerate_document_candidates,
    validate_record,
)
from scripts.build_qaq_request_demand_dataset import (
    build_request_windows,
    fixed_mode_specs,
    minimum_safe_precision,
    profile_from_stats,
    prompt_token_features,
)


class FakeTokenizer:
    def decode(self, tokens, skip_special_tokens=False):
        assert skip_special_tokens is False
        return "Hello 123!\n"


def test_build_request_windows_splits_non_overlapping_prompt_and_continuation():
    requests = build_request_windows(
        torch.arange(30),
        prompt_length=4,
        continuation_length=2,
        start=1,
        count=2,
    )

    assert requests[0][0].tolist() == [6, 7, 8, 9]
    assert requests[0][1].tolist() == [10, 11]
    assert requests[1][2].tolist() == [12, 13, 14, 15, 16, 17]


def test_prompt_token_features_are_predecode_and_numeric():
    features = prompt_token_features(torch.tensor([1, 1, 2, 3]), FakeTokenizer())

    assert features["prompt_length_tokens"] == 4
    assert features["unique_token_fraction"] == 0.75
    assert features["digit_fraction"] > 0
    assert features["line_count"] == 2
    assert all(isinstance(value, float) for value in features.values())


def test_fixed_mode_specs_cover_low_intermediate_and_high():
    assert fixed_mode_specs([3, 4, 5, 6]) == [
        ("fixed_low", "fixed_low", None),
        ("fixed_4", "fixed_precision", 4),
        ("fixed_5", "fixed_precision", 5),
        ("fixed_high", "fixed_high", None),
    ]


def test_minimum_safe_precision_chooses_smallest_quality_safe_bit():
    fixed = {
        3: {"mean_nll": 1.20, "runtime_stats": {"effective_bits": 3.0}},
        4: {"mean_nll": 1.08, "runtime_stats": {"effective_bits": 4.0}},
        5: {"mean_nll": 1.01, "runtime_stats": {"effective_bits": 5.0}},
        6: {"mean_nll": 1.00, "runtime_stats": {"effective_bits": 6.0}},
    }

    demand = minimum_safe_precision(fixed, reference_nll=1.0, safe_nll_delta=0.02)

    assert demand["requested_bit"] == 5
    assert demand["actual_effective_bits"] == 5.0
    assert demand["fixed_nll_deltas"]["4"] == pytest.approx(0.08)


def test_profile_from_stats_emits_route_and_group_vectors():
    stats = {
        "average_selected_bit": 4.0,
        "effective_bits": 4.2,
        "total_fallbacks": 2,
        "fallback_fraction": 0.1,
        "total_dp_guard_triggers": 1,
        "dp_guard_trigger_fraction": 0.05,
        "per_layer": {
            "0.q_proj": {"bit_counts": {"3": 3, "5": 1}},
            "1.q_proj": {"bit_counts": {"4": 1, "6": 1}},
            "2.q_proj": {"bit_counts": {"5": 2}},
            "3.q_proj": {"bit_counts": {"6": 2}},
        },
    }

    profile = profile_from_stats(stats, layer_group_size=2)

    assert profile["route_expected_bits"]["0.q_proj"] == 3.5
    assert profile["route_majority_bits"]["1.q_proj"] == 6
    assert profile["group_expected_bits"] == [4.25, 5.5]


def synthetic_record(index, profile, safe_bit):
    feature = float(index % 2)
    return {
        "schema_version": "qaq_request_demand_v1",
        "request_id": f"r{index}",
        "prompt_features": {
            "feature_a": feature,
            "feature_b": float(index),
        },
        "minimum_safe_precision": {"requested_bit": safe_bit},
        "observed_qaq_profiles": {
            "mlp_multibit_dp_guard": {
                "group_expected_bits": profile,
                "effective_bits": float(np.mean(profile)),
            }
        },
    }


def test_exact_profile_oracle_beats_mixed_fcfs_pairing():
    profiles = np.asarray([
        [3.0, 3.0],
        [6.0, 6.0],
        [3.0, 3.0],
        [6.0, 6.0],
    ])
    records = [
        synthetic_record(index, profile.tolist(), 3 if profile[0] == 3 else 6)
        for index, profile in enumerate(profiles)
    ]

    result = scheduler_oracle_analysis(
        records,
        profiles,
        profile_mode="mlp_multibit_dp_guard",
        batch_size=2,
        time_limit_s=10.0,
    )

    assert result["solver_status"] == 0
    assert result["oracle_precision_work"] < result["fcfs_precision_work"]
    assert result["oracle_advantage_vs_fcfs_fraction"] > 0


def test_predictor_analysis_uses_only_prompt_features_and_cross_validation():
    records = []
    for index in range(20):
        high = index % 2
        profile = [3.0 + 3.0 * high, 3.5 + 2.5 * high]
        records.append(synthetic_record(index, profile, 3 if not high else 6))
    profiles = np.asarray([
        record["observed_qaq_profiles"]["mlp_multibit_dp_guard"]["group_expected_bits"]
        for record in records
    ])

    result = predictor_analysis(
        records,
        profiles,
        profile_mode="mlp_multibit_dp_guard",
        cv_folds=5,
        seed=0,
        trees=30,
    )

    assert result["status"] == "REQUEST_LEVEL_CROSS_VALIDATION"
    assert result["feature_names"] == ["feature_a", "feature_b"]
    assert len(result["group_profile_regressor"]["predictions"]) == 20
    assert result["minimum_safe_precision_classifier"]["status"] == "CROSS_VALIDATED"
    assert "continuation" not in " ".join(result["feature_names"])


def protocol_request(request_id="r0", continuation_length=2):
    return {
        "request_id": request_id,
        "request_index": 0,
        "dataset": "wikitext2",
        "source_index": 10,
        "document_id": "d" * 64,
        "partition": "test",
        "start_token": 0,
        "end_token": 6,
        "prompt_length_tokens": 4,
        "continuation_length_tokens": continuation_length,
        "selection_sha256": "s" * 64,
        "prompt_token_sha256": "p" * 64,
        "continuation_token_sha256": "c" * 64,
        "request_token_sha256": "r" * 64,
    }


def protocol_record(request, mean_nll=1.0):
    modes = [
        "fixed_low",
        "fixed_4",
        "fixed_5",
        "fixed_high",
        "dp_threshold_only",
        "mlp_multibit",
        "mlp_multibit_dp_guard",
    ]
    quality = {
        mode: {
            "mean_nll": mean_nll,
            "nll_delta_vs_fixed_high": 0.0,
            "perplexity": float(np.exp(mean_nll)),
            "finite_logits": True,
            "target_token_count": request["continuation_length_tokens"],
            "average_selected_bit": 5.0,
            "effective_bits": 5.0,
            "fallback_count": 0,
            "fallback_fraction": 0.0,
            "dp_guard_trigger_count": 0,
            "dp_guard_trigger_fraction": 0.0,
        }
        for mode in modes
    }
    return {
        "schema_version": "qaq_request_demand_v2",
        "manifest_sha256": "m" * 64,
        "request_id": request["request_id"],
        "prompt_token_sha256": request["prompt_token_sha256"],
        "continuation_token_sha256": request["continuation_token_sha256"],
        "request_token_sha256": request["request_token_sha256"],
        "prompt_length_tokens": request["prompt_length_tokens"],
        "continuation_length_tokens": request["continuation_length_tokens"],
        "source": {"partition": request["partition"]},
        "quality_by_mode": quality,
        "minimum_safe_precision": {"requested_bit": 5},
        "contains_raw_text": False,
    }


def test_document_partition_is_stable_and_document_level():
    doc_id = "a" * 64
    assert document_partition(doc_id) == document_partition(doc_id)
    assert document_partition(doc_id) in {"development", "calibration", "test"}


def test_candidate_enumeration_enforces_gap_and_document_caps():
    candidates = enumerate_document_candidates(
        dataset_name="wikitext2",
        source_index=1,
        doc_id="b" * 64,
        partition="development",
        token_ids=list(range(20000)),
    )

    assert len(candidates) == 16
    per_cell = Counter(
        (item["prompt_length_tokens"], item["continuation_length_tokens"])
        for item in candidates
    )
    assert max(per_cell.values()) <= 4
    intervals = sorted((item["start_token"], item["end_token"]) for item in candidates)
    assert all(current[0] - previous[1] >= 128 for previous, current in zip(intervals, intervals[1:]))


def test_validate_record_requires_continuation_only_metrics_and_rejects_raw_text():
    request = protocol_request()
    record = protocol_record(request)
    expected_modes = list(record["quality_by_mode"])

    validate_record(record, request, "m" * 64, expected_modes)

    record["quality_by_mode"]["fixed_low"]["target_token_count"] = 3
    with pytest.raises(ValueError, match="continuation-only"):
        validate_record(record, request, "m" * 64, expected_modes)

    record = protocol_record(request)
    record["prompt_text"] = "copyrighted payload"
    with pytest.raises(ValueError, match="Raw text"):
        validate_record(record, request, "m" * 64, expected_modes)


def test_validated_shard_roundtrip_is_idempotent(tmp_path):
    request = protocol_request()
    record = protocol_record(request)
    shard = tmp_path / "shard-00000.jsonl"
    atomic_write_jsonl(shard, [record])
    expected_modes = list(record["quality_by_mode"])

    first = ensure_shard_metadata(shard, [request], "m" * 64, expected_modes)
    second = ensure_shard_metadata(shard, [request], "m" * 64, expected_modes)

    assert first == second
    assert first["validation_status"] == "VALIDATED"
    assert first["record_count"] == 1
    assert shard.with_suffix(".meta.json").exists()


def test_aggregate_records_weights_nll_by_continuation_tokens():
    short_request = protocol_request("short", continuation_length=2)
    long_request = protocol_request("long", continuation_length=4)
    short_record = protocol_record(short_request, mean_nll=1.0)
    long_record = protocol_record(long_request, mean_nll=2.0)

    summary = aggregate_records([short_record, long_record])

    assert summary["request_count"] == 2
    assert summary["continuation_token_count"] == 6
    assert summary["quality_by_mode"]["fixed_high"][
        "continuation_token_weighted_mean_nll"
    ] == pytest.approx(5.0 / 3.0)
