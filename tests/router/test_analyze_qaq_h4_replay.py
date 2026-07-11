import sys

import pytest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_qaq_h4_replay import (
    analyze_comparisons,
    paired_document_bootstrap,
    paired_point,
)


def scenario(dataset, load, seed, repeat, policy, rate, p95, misses, service):
    return {
        "dataset": dataset,
        "load_fraction": load,
        "scheduling_seed": seed,
        "repeat_index": repeat,
        "policy": policy,
        "summary": {
            "requests_per_s": rate,
            "end_to_end_latency_ms": {"p95": p95},
            "deadline_miss_fraction": misses,
        },
        "requests": [
            {
                "request_id": f"r{i}",
                "document_id": f"d{i // 2}",
                "gpu_service_share_ms": service,
            }
            for i in range(8)
        ],
    }


def test_paired_point_reports_registered_direction():
    primary = scenario("d", 0.8, 101, 0, "predicted_block_fallback_lane", 110, 102, 0.02, 8)
    baseline = scenario("d", 0.8, 101, 0, "ordinary_fcfs", 100, 100, 0.01, 10)
    result = paired_point(primary, baseline)
    assert result["throughput_improvement_fraction"] == pytest.approx(0.1)
    assert result["p95_latency_increase_fraction"] == pytest.approx(0.02)
    assert result["deadline_miss_increase"] == pytest.approx(0.01)


def test_document_bootstrap_is_paired_and_reproducible():
    primary = scenario("d", 0.8, 101, 0, "predicted_block_fallback_lane", 110, 100, 0, 8)
    baseline = scenario("d", 0.8, 101, 0, "ordinary_fcfs", 100, 100, 0, 10)
    first = paired_document_bootstrap([(primary, baseline)], 100, 1729)
    second = paired_document_bootstrap([(primary, baseline)], 100, 1729)
    assert (first == second).all()
    assert (first > 0).all()


def test_comparison_analysis_assigns_holm_fields_for_primary_loads():
    scenarios = []
    for dataset in ("c4_new", "wikitext2"):
        for load in (0.5, 0.8, 0.95):
            for seed in (101, 202, 303):
                for repeat in range(3):
                    scenarios.extend([
                        scenario(dataset, load, seed, repeat, "ordinary_fcfs", 100, 100, 0.01, 10),
                        scenario(dataset, load, seed, repeat, "length_fcfs", 102, 99, 0.01, 9.8),
                        scenario(dataset, load, seed, repeat, "predicted_block_fallback_lane", 110, 101, 0.01, 8),
                    ])
    result = analyze_comparisons(scenarios, replicates=100, seed=1729)
    comparison = result["c4_new"]["0.8"]["ordinary_fcfs"]
    assert comparison["paired_run_count"] == 9
    assert comparison["median_throughput_improvement_fraction"] > 0.05
    assert "holm_corrected_lower_bound" in comparison
    assert comparison["direction_positive_seed_count"] == 3
