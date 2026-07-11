import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_qaq_route_safety_supplement import (
    MODES,
    aggregate_records,
    validate_output_dir,
    validate_record,
)


def counts(under, over, exact):
    total = under + over + exact
    return {
        "decision_count": total,
        "under_precision_count": under,
        "over_precision_count": over,
        "exact_precision_count": exact,
        "signed_bit_gap_sum": over - under,
        "absolute_bit_gap_sum": over + under,
        "under_precision_rate": under / total,
        "over_precision_rate": over / total,
        "exact_precision_rate": exact / total,
        "mean_signed_bit_gap": (over - under) / total,
        "mean_absolute_bit_gap": (over + under) / total,
    }


def request():
    return {
        "request_id": "r0",
        "prompt_token_sha256": "p" * 64,
        "continuation_token_sha256": "c" * 64,
        "request_token_sha256": "r" * 64,
    }


def record():
    mode_values = {}
    for mode in MODES:
        mode_values[mode] = {
            "continuation_mean_nll": 1.0,
            "finite_logits": True,
            "precision_metrics": counts(1, 2, 7),
            "per_layer_precision_metrics": {"0.q_proj": counts(1, 2, 7)},
        }
    return {
        "schema_version": "qaq_route_safety_supplement_record_v1",
        "supplement_id": "s" * 64,
        "request_id": "r0",
        "prompt_token_sha256": "p" * 64,
        "continuation_token_sha256": "c" * 64,
        "request_token_sha256": "r" * 64,
        "continuation_length_tokens": 32,
        "modes": mode_values,
    }


def test_record_validation_requires_real_decisions_and_rejects_raw_payload():
    value = record()
    validate_record(value, request(), "s" * 64)

    value["modes"][MODES[0]]["precision_metrics"]["decision_count"] = 0
    with pytest.raises(ValueError, match="No real route decisions"):
        validate_record(value, request(), "s" * 64)

    value = record()
    value["token_ids"] = [1, 2]
    with pytest.raises(ValueError, match="Raw text/token"):
        validate_record(value, request(), "s" * 64)


def test_aggregate_records_sums_real_precision_counts():
    result = aggregate_records([record(), record()])
    metrics = result["modes"]["mlp_multibit_dp_guard"]["precision_metrics"]

    assert result["request_count"] == 2
    assert metrics["decision_count"] == 20
    assert metrics["under_precision_count"] == 2
    assert metrics["under_precision_rate"] == pytest.approx(0.1)


def test_supplement_output_must_be_outside_frozen_collection(tmp_path):
    collection = tmp_path / "collection"
    collection.mkdir()
    with pytest.raises(ValueError, match="outside"):
        validate_output_dir(collection, collection)
    with pytest.raises(ValueError, match="outside"):
        validate_output_dir(collection / "child", collection)
    validate_output_dir(tmp_path / "supplement", collection)
