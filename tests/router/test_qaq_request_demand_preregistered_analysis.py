import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_qaq_request_demand_preregistered import (
    canonical_json_bytes,
    collection_file_manifest,
    demand_summary,
    quality_summary,
    verify_freeze,
)


def test_recursive_freeze_rejects_collection_mutation(tmp_path):
    collection = tmp_path / "collection"
    collection.mkdir()
    (collection / "shard.jsonl").write_text('{"value": 1}\n')
    files, tree_hash = collection_file_manifest(collection)
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({
        "validation_status": "REAL_GPU_REQUEST_DEMAND_COMPLETE",
        "files": files,
        "collection_tree_sha256": tree_hash,
    }))

    assert verify_freeze(collection, freeze)["collection_tree_sha256"] == tree_hash
    (collection / "shard.jsonl").write_text('{"value": 2}\n')
    with pytest.raises(RuntimeError, match="differs from the recursively frozen"):
        verify_freeze(collection, freeze)


def synthetic_record(index, safe_bit, guarded_delta=0.01, unguarded_delta=0.03):
    quality = {}
    for mode, delta, effective in (
        ("fixed_low", 0.20, 3.0),
        ("fixed_4", 0.08, 4.0),
        ("fixed_5", 0.01, 5.0),
        ("fixed_high", 0.0, 6.0),
        ("dp_threshold_only", 0.015, 4.5),
        ("mlp_multibit", unguarded_delta, 5.1),
        ("mlp_multibit_dp_guard", guarded_delta, 5.2),
    ):
        quality[mode] = {
            "nll_delta_vs_fixed_high": delta,
            "effective_bits": effective,
            "finite_logits": True,
        }
    return {
        "source": {"dataset": "sample", "document_id": f"doc-{index}"},
        "minimum_safe_precision": {"requested_bit": safe_bit},
        "quality_by_mode": quality,
    }


def test_h1_document_bootstrap_detects_heterogeneous_demand():
    records = [synthetic_record(index, 3 if index % 2 else 6) for index in range(40)]
    result = demand_summary(records, replicates=200, seed=1729)

    assert result["minimum_safe_precision_counts"] == {"3": 20, "6": 20}
    assert result["document_cluster_bootstrap_std_ci95"][0] > 0
    assert result["h1_pass"] is True


def test_quality_summary_uses_registered_request_failure_and_reports_missing_route_gate():
    records = [synthetic_record(index, 5) for index in range(20)]
    result = quality_summary(records, replicates=100, seed=1729)

    assert result["by_mode"]["mlp_multibit_dp_guard"]["quality_gate_pass"] is True
    assert result["by_mode"]["mlp_multibit"]["quality_gate_pass"] is False
    assert result["guard_efficacy_request_quality_pass"] is True
    assert result["guard_noncollapse_pass"] is True
    assert result["route_under_precision_gate"]["status"] == "UNAVAILABLE_FROM_COLLECTION"
    assert result["h3_status"] == "NOT_EVALUABLE_ROUTE_SAFETY_ENDPOINT_MISSING"

    failing = quality_summary(
        [synthetic_record(index, 5, guarded_delta=0.03) for index in range(20)],
        replicates=100,
        seed=1729,
    )
    assert failing["h3_status"] == "FAIL_GUARDED_QUALITY_GATE_ROUTE_SAFETY_UNAVAILABLE"
