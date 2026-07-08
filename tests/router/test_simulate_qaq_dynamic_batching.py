import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.simulate_qaq_dynamic_batching import (
    UNVALIDATED,
    majority_route_bits,
    simulate,
)


def write_trace(path: Path, records):
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def record(request_id, arrival_time_s, average_bit, gpu_ms, route_bits):
    return {
        "request_id": request_id,
        "arrival_time_s": arrival_time_s,
        "workload_type": "unit",
        "gpu_execution_ms": gpu_ms,
        "average_selected_bit": average_bit,
        "effective_bits": average_bit,
        "fallback_count": 0,
        "dp_guard_trigger_count": 0,
        "per_layer_bit_counts": {
            route: {str(bit): 1}
            for route, bit in route_bits.items()
        },
    }


def args_for(trace_path, output_path, **overrides):
    values = {
        "trace_jsonl": str(trace_path),
        "output_json": str(output_path),
        "policies": [
            "ordinary_dynamic_batching",
            "scalar_budget_batching",
            "block_profile_batching",
            "max_profile_sharing",
            "quantile_profile_sharing",
        ],
        "max_batch_size": 2,
        "max_wait_ms": 100.0,
        "compatibility_threshold": 0.25,
        "scalar_bucket_size": 0.25,
        "quantile": 0.5,
        "batch_overhead_ms": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_majority_route_bits_tie_breaks_to_higher_bit():
    assert majority_route_bits({"0.q_proj": {"3": 2, "6": 2}}) == {"0.q_proj": 6}


def test_simulator_batches_by_arrival_window_and_reports_queue_delay(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    output_path = tmp_path / "sim.json"
    write_trace(
        trace_path,
        [
            record("a", 0.00, 4.0, 100.0, {"0.q_proj": 4}),
            record("b", 0.05, 4.0, 120.0, {"0.q_proj": 4}),
            record("c", 0.30, 4.0, 80.0, {"0.q_proj": 4}),
        ],
    )

    result = simulate(args_for(trace_path, output_path))
    ordinary = result["policies"]["ordinary_dynamic_batching"]

    assert ordinary["batch_count"] == 2
    assert ordinary["mean_batch_size"] == 1.5
    assert ordinary["requests"][0]["queue_delay_ms"] == 50.0
    assert ordinary["requests"][1]["queue_delay_ms"] == 0.0
    assert ordinary["latency_ms"]["p95"] is not None


def test_scalar_policy_splits_incompatible_bit_buckets(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    output_path = tmp_path / "sim.json"
    write_trace(
        trace_path,
        [
            record("low", 0.00, 3.1, 100.0, {"0.q_proj": 3}),
            record("high", 0.01, 5.9, 100.0, {"0.q_proj": 6}),
        ],
    )

    result = simulate(args_for(trace_path, output_path, scalar_bucket_size=0.5))
    scalar = result["policies"]["scalar_budget_batching"]

    assert scalar["batch_count"] == 2
    assert scalar["mean_batch_size"] == 1.0
    assert sorted(scalar["lane_occupancy"].values()) == [1, 1]


def test_shared_profile_policies_report_under_and_over_precision(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    output_path = tmp_path / "sim.json"
    write_trace(
        trace_path,
        [
            record("low", 0.00, 3.0, 100.0, {"0.q_proj": 3, "0.k_proj": 3}),
            record("high", 0.01, 6.0, 100.0, {"0.q_proj": 6, "0.k_proj": 6}),
        ],
    )

    result = simulate(args_for(trace_path, output_path, quantile=0.0))
    max_policy = result["policies"]["max_profile_sharing"]
    quantile_policy = result["policies"]["quantile_profile_sharing"]

    assert max_policy["under_precision_rate"] == 0.0
    assert max_policy["over_precision_rate"] == 0.5
    assert quantile_policy["under_precision_rate"] == 0.5
    assert quantile_policy["over_precision_rate"] == 0.0


def test_non_shared_policies_keep_under_over_precision_unvalidated(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    output_path = tmp_path / "sim.json"
    write_trace(trace_path, [record("a", 0.00, 4.0, 100.0, {"0.q_proj": 4})])

    result = simulate(args_for(trace_path, output_path))

    assert result["policies"]["ordinary_dynamic_batching"]["under_precision_rate"] == UNVALIDATED
    assert result["assumptions"]["status"] == "SIMULATED_ONLY"
