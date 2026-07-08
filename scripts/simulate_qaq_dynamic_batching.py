import argparse
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


UNVALIDATED = "UNVALIDATED"
SIM_SCHEMA_VERSION = "qaq_dynamic_batching_simulation_v1"
DEFAULT_POLICIES = [
    "ordinary_dynamic_batching",
    "scalar_budget_batching",
    "block_profile_batching",
    "max_profile_sharing",
    "quantile_profile_sharing",
]


@dataclass(frozen=True)
class TraceRequest:
    request_id: str
    arrival_ms: float
    workload_type: str
    gpu_execution_ms: float
    average_selected_bit: float
    effective_bits: float
    fallback_count: int
    dp_guard_trigger_count: int
    route_bits: dict[str, int]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Trace-driven simulator for QAQ precision-aware batching. "
            "Outputs simulated scheduling metrics only; it does not prove GPU speedup."
        )
    )
    parser.add_argument("--trace_jsonl", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--policies", nargs="+", default=DEFAULT_POLICIES, choices=DEFAULT_POLICIES)
    parser.add_argument("--max_batch_size", type=int, default=4)
    parser.add_argument("--max_wait_ms", type=float, default=100.0)
    parser.add_argument("--compatibility_threshold", type=float, default=0.25)
    parser.add_argument("--scalar_bucket_size", type=float, default=0.25)
    parser.add_argument("--quantile", type=float, default=0.75)
    parser.add_argument("--batch_overhead_ms", type=float, default=0.0)
    return parser.parse_args()


def load_trace(path: str | Path) -> list[TraceRequest]:
    requests = []
    with Path(path).open() as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            requests.append(trace_request_from_record(raw, line_no))
    if not requests:
        raise ValueError(f"No trace records found in {path}")
    return sorted(requests, key=lambda request: (request.arrival_ms, request.request_id))


def trace_request_from_record(raw: dict[str, Any], line_no: int) -> TraceRequest:
    route_bits = majority_route_bits(raw.get("per_layer_bit_counts", {}))
    if not route_bits:
        raise ValueError(f"Trace record on line {line_no} has no per_layer_bit_counts")
    return TraceRequest(
        request_id=str(raw["request_id"]),
        arrival_ms=1000.0 * float(raw.get("arrival_time_s", 0.0)),
        workload_type=str(raw.get("workload_type", UNVALIDATED)),
        gpu_execution_ms=float(raw["gpu_execution_ms"]),
        average_selected_bit=float(raw["average_selected_bit"]),
        effective_bits=float(raw["effective_bits"]),
        fallback_count=int(raw.get("fallback_count", 0)),
        dp_guard_trigger_count=int(raw.get("dp_guard_trigger_count", 0)),
        route_bits=route_bits,
    )


def majority_route_bits(per_layer_bit_counts: dict[str, dict[str, int]]) -> dict[str, int]:
    route_bits = {}
    for route_name, counts in per_layer_bit_counts.items():
        if not counts:
            continue
        bit, _ = max(
            ((int(bit), int(count)) for bit, count in counts.items()),
            key=lambda item: (item[1], item[0]),
        )
        route_bits[str(route_name)] = int(bit)
    return route_bits


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"values": [], "mean": None, "min": None, "max": None, "p50": None, "p95": None, "p99": None}
    return {
        "values": values,
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


def scalar_bucket(request: TraceRequest, bucket_size: float) -> int:
    return int(math.floor(request.average_selected_bit / bucket_size))


def block_distance(a: TraceRequest, b: TraceRequest) -> float:
    routes = sorted(set(a.route_bits) | set(b.route_bits))
    if not routes:
        return 1.0
    observed_bits = [*a.route_bits.values(), *b.route_bits.values()]
    bit_span = max(observed_bits) - min(observed_bits)
    normalizer = max(bit_span, 1)
    total = 0.0
    for route in routes:
        total += abs(a.route_bits.get(route, max(observed_bits)) - b.route_bits.get(route, max(observed_bits)))
    return total / (len(routes) * normalizer)


def route_vector_hash(route_bits: dict[str, int]) -> str:
    payload = json.dumps(sorted(route_bits.items()), separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def choose_batch(
    pending: list[TraceRequest],
    first: TraceRequest,
    policy: str,
    max_batch_size: int,
    scalar_bucket_size: float,
    compatibility_threshold: float,
) -> list[TraceRequest]:
    if policy in {"ordinary_dynamic_batching", "max_profile_sharing", "quantile_profile_sharing"}:
        return pending[:max_batch_size]

    if policy == "scalar_budget_batching":
        first_bucket = scalar_bucket(first, scalar_bucket_size)
        compatible = [
            request
            for request in pending
            if scalar_bucket(request, scalar_bucket_size) == first_bucket
        ]
        return compatible[:max_batch_size]

    if policy == "block_profile_batching":
        compatible = [
            request
            for request in pending
            if block_distance(first, request) <= compatibility_threshold
        ]
        return compatible[:max_batch_size]

    raise ValueError(f"Unsupported policy: {policy}")


def compose_shared_profile(batch: list[TraceRequest], policy: str, quantile: float) -> dict[str, int] | None:
    if policy == "max_profile_sharing":
        routes = sorted({route for request in batch for route in request.route_bits})
        return {route: max(request.route_bits.get(route, 0) for request in batch) for route in routes}

    if policy == "quantile_profile_sharing":
        routes = sorted({route for request in batch for route in request.route_bits})
        shared = {}
        for route in routes:
            values = sorted(request.route_bits.get(route, max(request.route_bits.values())) for request in batch)
            idx = min(len(values) - 1, max(0, math.ceil((len(values) - 1) * quantile)))
            shared[route] = int(values[idx])
        return shared

    return None


def precision_accounting(batch: list[TraceRequest], shared_profile: dict[str, int] | None) -> dict[str, Any]:
    if shared_profile is None:
        return {
            "under_precision_rate": UNVALIDATED,
            "over_precision_rate": UNVALIDATED,
            "shared_average_bit": UNVALIDATED,
        }

    under = 0
    over = 0
    total = 0
    shared_bits = []
    for request in batch:
        for route, requested_bit in request.route_bits.items():
            if route not in shared_profile:
                continue
            shared_bit = shared_profile[route]
            shared_bits.append(shared_bit)
            under += int(shared_bit < requested_bit)
            over += int(shared_bit > requested_bit)
            total += 1

    return {
        "under_precision_rate": under / total if total else 0.0,
        "over_precision_rate": over / total if total else 0.0,
        "shared_average_bit": statistics.fmean(shared_bits) if shared_bits else 0.0,
    }


def estimate_batch_service_ms(batch: list[TraceRequest], accounting: dict[str, Any], batch_overhead_ms: float) -> float:
    base_ms = max(request.gpu_execution_ms for request in batch)
    if accounting["shared_average_bit"] == UNVALIDATED:
        bit_multiplier = 1.0
    else:
        mean_observed_bit = statistics.fmean(request.average_selected_bit for request in batch)
        bit_multiplier = float(accounting["shared_average_bit"]) / mean_observed_bit if mean_observed_bit > 0 else 1.0
    return base_ms * bit_multiplier + batch_overhead_ms * max(0, len(batch) - 1)


def simulate_policy(
    requests: list[TraceRequest],
    policy: str,
    max_batch_size: int,
    max_wait_ms: float,
    scalar_bucket_size: float,
    compatibility_threshold: float,
    quantile: float,
    batch_overhead_ms: float,
) -> dict[str, Any]:
    unscheduled = list(requests)
    current_time_ms = 0.0
    batch_results = []
    request_results = []
    batch_index = 0

    while unscheduled:
        first = unscheduled[0]
        ready_window_end = first.arrival_ms + max_wait_ms
        pending = [request for request in unscheduled if request.arrival_ms <= ready_window_end]
        batch = choose_batch(
            pending=pending,
            first=first,
            policy=policy,
            max_batch_size=max_batch_size,
            scalar_bucket_size=scalar_bucket_size,
            compatibility_threshold=compatibility_threshold,
        )
        if not batch:
            batch = [first]

        batch_ids = {request.request_id for request in batch}
        latest_arrival_ms = max(request.arrival_ms for request in batch)
        if len(batch) >= max_batch_size:
            schedule_start_ms = max(current_time_ms, latest_arrival_ms)
        else:
            schedule_start_ms = max(current_time_ms, ready_window_end)

        shared_profile = compose_shared_profile(batch, policy, quantile)
        accounting = precision_accounting(batch, shared_profile)
        service_ms = estimate_batch_service_ms(batch, accounting, batch_overhead_ms)
        finish_ms = schedule_start_ms + service_ms
        batch_id = f"{policy}_batch_{batch_index:05d}"
        lane_id = lane_for_batch(batch, policy, scalar_bucket_size)

        batch_results.append({
            "batch_id": batch_id,
            "lane_id": lane_id,
            "request_ids": [request.request_id for request in batch],
            "batch_size": len(batch),
            "schedule_start_ms": schedule_start_ms,
            "simulated_gpu_execution_ms": service_ms,
            "finish_ms": finish_ms,
            "shared_profile_policy": shared_policy_name(policy),
            **accounting,
        })

        for request in batch:
            request_results.append({
                "request_id": request.request_id,
                "policy": policy,
                "batch_id": batch_id,
                "lane_id": lane_id,
                "queue_delay_ms": schedule_start_ms - request.arrival_ms,
                "simulated_gpu_execution_ms": service_ms,
                "end_to_end_latency_ms": finish_ms - request.arrival_ms,
                "average_selected_bit": request.average_selected_bit,
                "effective_bits": request.effective_bits,
                "fallback_count": request.fallback_count,
                "dp_guard_trigger_count": request.dp_guard_trigger_count,
                "under_precision_rate": accounting["under_precision_rate"],
                "over_precision_rate": accounting["over_precision_rate"],
            })

        unscheduled = [request for request in unscheduled if request.request_id not in batch_ids]
        current_time_ms = finish_ms
        batch_index += 1

    return summarize_policy(policy, batch_results, request_results, requests)


def lane_for_batch(batch: list[TraceRequest], policy: str, scalar_bucket_size: float) -> str:
    first = batch[0]
    if policy == "scalar_budget_batching":
        return f"scalar_bucket_{scalar_bucket(first, scalar_bucket_size)}"
    if policy == "block_profile_batching":
        return f"block_{route_vector_hash(first.route_bits)}"
    if policy in {"max_profile_sharing", "quantile_profile_sharing"}:
        return shared_policy_name(policy)
    return "ordinary"


def shared_policy_name(policy: str) -> str:
    if policy == "max_profile_sharing":
        return "max"
    if policy == "quantile_profile_sharing":
        return "quantile"
    return "none"


def summarize_policy(
    policy: str,
    batch_results: list[dict[str, Any]],
    request_results: list[dict[str, Any]],
    requests: list[TraceRequest],
) -> dict[str, Any]:
    latencies = [float(result["end_to_end_latency_ms"]) for result in request_results]
    queue_delays = [float(result["queue_delay_ms"]) for result in request_results]
    service_times = [float(result["simulated_gpu_execution_ms"]) for result in batch_results]
    makespan_ms = max((batch["finish_ms"] for batch in batch_results), default=0.0) - min(
        (request.arrival_ms for request in requests),
        default=0.0,
    )
    lane_occupancy = {}
    for batch in batch_results:
        lane_occupancy[batch["lane_id"]] = lane_occupancy.get(batch["lane_id"], 0) + int(batch["batch_size"])

    numeric_under = [
        float(result["under_precision_rate"])
        for result in request_results
        if result["under_precision_rate"] != UNVALIDATED
    ]
    numeric_over = [
        float(result["over_precision_rate"])
        for result in request_results
        if result["over_precision_rate"] != UNVALIDATED
    ]

    return {
        "policy": policy,
        "request_count": len(request_results),
        "batch_count": len(batch_results),
        "mean_batch_size": statistics.fmean(batch["batch_size"] for batch in batch_results) if batch_results else 0.0,
        "latency_ms": summarize(latencies),
        "queue_delay_ms": summarize(queue_delays),
        "batch_service_ms": summarize(service_times),
        "requests_per_s": 1000.0 * len(request_results) / makespan_ms if makespan_ms > 0 else 0.0,
        "lane_occupancy": lane_occupancy,
        "mean_effective_bits": statistics.fmean(request.effective_bits for request in requests),
        "total_fallbacks": sum(request.fallback_count for request in requests),
        "total_dp_guard_triggers": sum(request.dp_guard_trigger_count for request in requests),
        "under_precision_rate": statistics.fmean(numeric_under) if numeric_under else UNVALIDATED,
        "over_precision_rate": statistics.fmean(numeric_over) if numeric_over else UNVALIDATED,
        "batches": batch_results,
        "requests": request_results,
    }


def simulate(args) -> dict[str, Any]:
    requests = load_trace(args.trace_jsonl)
    policies = {}
    for policy in args.policies:
        policies[policy] = simulate_policy(
            requests=requests,
            policy=policy,
            max_batch_size=args.max_batch_size,
            max_wait_ms=args.max_wait_ms,
            scalar_bucket_size=args.scalar_bucket_size,
            compatibility_threshold=args.compatibility_threshold,
            quantile=args.quantile,
            batch_overhead_ms=args.batch_overhead_ms,
        )

    return {
        "simulation_schema_version": SIM_SCHEMA_VERSION,
        "trace_jsonl": str(args.trace_jsonl),
        "request_count": len(requests),
        "assumptions": {
            "status": "SIMULATED_ONLY",
            "service_time_model": (
                "batch_service_ms = max(single_request_gpu_execution_ms) * "
                "(shared_average_bit / mean_observed_average_selected_bit) + batch_overhead_ms"
            ),
            "profile_source": "observed QAQ per-layer majority bits from trace, not a predictor",
            "quality_status": UNVALIDATED,
            "transfer_bytes_status": UNVALIDATED,
            "kernel_switch_status": UNVALIDATED,
        },
        "config": {
            "policies": args.policies,
            "max_batch_size": args.max_batch_size,
            "max_wait_ms": args.max_wait_ms,
            "compatibility_threshold": args.compatibility_threshold,
            "scalar_bucket_size": args.scalar_bucket_size,
            "quantile": args.quantile,
            "batch_overhead_ms": args.batch_overhead_ms,
        },
        "policies": policies,
    }


def main():
    args = parse_args()
    result = simulate(args)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {output_path}")
    for policy, summary in result["policies"].items():
        print(
            f"{policy}: batches={summary['batch_count']} "
            f"mean_batch={summary['mean_batch_size']:.2f} "
            f"p95_latency_ms={summary['latency_ms']['p95']:.2f} "
            f"requests_per_s={summary['requests_per_s']:.3f}"
        )


if __name__ == "__main__":
    main()
