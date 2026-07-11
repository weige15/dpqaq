"""Paired document-bootstrap analysis for the primary preregistered H4 GPU replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from scripts.qaq_request_demand_protocol import atomic_write_json, file_sha256, read_jsonl
from scripts.run_qaq_online_scheduler_replay import POLICIES, SCENARIO_SCHEMA

SCHEMA_VERSION = "qaq_h4_preregistered_analysis_v1"
PRIMARY_POLICY = "predicted_block_fallback_lane"
BASELINES = ("ordinary_fcfs", "length_fcfs")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze complete primary H4 online replay.")
    parser.add_argument("--replay_dir", required=True)
    parser.add_argument("--route_safety_dir", required=True)
    parser.add_argument("--request_analysis_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--bootstrap_replicates", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=1729)
    return parser.parse_args()


def load_scenarios(replay_dir: Path) -> list[dict[str, Any]]:
    run = json.loads((replay_dir / "run-manifest.json").read_text())
    summary = json.loads((replay_dir / "summary.json").read_text())
    if summary.get("validation_status") != "REAL_GPU_ONLINE_REPLAY_COMPLETE":
        raise RuntimeError("Online replay is not complete")
    scenarios = []
    for path in sorted((replay_dir / "scenarios").glob("*.json")):
        item = json.loads(path.read_text())
        if item.get("scenario_schema_version") != SCENARIO_SCHEMA or item.get("replay_id") != run["replay_id"]:
            raise RuntimeError(f"Scenario mismatch: {path}")
        scenarios.append(item)
    if len(scenarios) != summary["expected_scenario_count"]:
        raise RuntimeError("Scenario count mismatch")
    return scenarios


def scenario_key(item: dict[str, Any]) -> tuple[str, float, int, int]:
    return item["dataset"], float(item["load_fraction"]), int(item["scheduling_seed"]), int(item["repeat_index"])


def throughput_proxy(requests: list[dict[str, Any]]) -> float:
    service = sum(float(item["gpu_service_share_ms"]) for item in requests)
    return 1000.0 * len(requests) / service if service else 0.0


def paired_point(primary: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    p = primary["summary"]
    b = baseline["summary"]
    return {
        "throughput_improvement_fraction": p["requests_per_s"] / b["requests_per_s"] - 1.0,
        "p95_latency_increase_fraction": p["end_to_end_latency_ms"]["p95"] / b["end_to_end_latency_ms"]["p95"] - 1.0,
        "deadline_miss_increase": p["deadline_miss_fraction"] - b["deadline_miss_fraction"],
    }


def paired_document_bootstrap(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]], replicates: int, seed: int
) -> np.ndarray:
    documents = sorted({item["document_id"] for primary, _ in pairs for item in primary["requests"]})
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    by_pair = []
    for primary, baseline in pairs:
        p_docs = defaultdict(list)
        b_docs = defaultdict(list)
        for item in primary["requests"]:
            p_docs[item["document_id"]].append(item)
        for item in baseline["requests"]:
            b_docs[item["document_id"]].append(item)
        by_pair.append((p_docs, b_docs))
    for index in range(replicates):
        sampled = [documents[int(i)] for i in rng.integers(0, len(documents), size=len(documents))]
        differences = []
        for p_docs, b_docs in by_pair:
            p_requests = [item for document in sampled for item in p_docs[document]]
            b_requests = [item for document in sampled for item in b_docs[document]]
            p_rate = throughput_proxy(p_requests)
            b_rate = throughput_proxy(b_requests)
            differences.append(p_rate / b_rate - 1.0 if b_rate else 0.0)
        values[index] = float(np.median(differences))
    return values


def route_safety_bootstrap(route_dir: Path, dataset: str, replicates: int, seed: int) -> dict[str, Any]:
    records = []
    for path in sorted((route_dir / "datasets" / dataset / "shards").glob("shard-*.jsonl")):
        records.extend(read_jsonl(path))
    if len(records) != 96:
        raise RuntimeError(f"Route supplement incomplete for {dataset}")
    by_document = defaultdict(list)
    for record in records:
        by_document[record["document_id"]].append(record)
    documents = list(by_document)
    rng = np.random.default_rng(seed)
    rates = np.empty(replicates, dtype=np.float64)
    mode = "mlp_multibit_dp_guard"
    for index in range(replicates):
        sampled = [documents[int(i)] for i in rng.integers(0, len(documents), size=len(documents))]
        under = 0
        total = 0
        for document in sampled:
            for record in by_document[document]:
                metrics = record["modes"][mode]["precision_metrics"]
                under += int(metrics["under_precision_count"])
                total += int(metrics["decision_count"])
        rates[index] = under / total
    point_under = sum(record["modes"][mode]["precision_metrics"]["under_precision_count"] for record in records)
    point_total = sum(record["modes"][mode]["precision_metrics"]["decision_count"] for record in records)
    return {
        "request_count": len(records),
        "document_count": len(documents),
        "under_precision_rate": point_under / point_total,
        "one_sided_95_upper": float(np.quantile(rates, 0.95)),
        "gate_pass": bool(point_under / point_total <= 0.01 and np.quantile(rates, 0.95) <= 0.02),
    }


def analyze_comparisons(scenarios: list[dict[str, Any]], replicates: int, seed: int) -> dict[str, Any]:
    indexed = {(scenario_key(item), item["policy"]): item for item in scenarios}
    datasets = sorted({key[0][0] for key in indexed})
    loads = sorted({key[0][1] for key in indexed})
    results = {}
    family = []
    for dataset_index, dataset in enumerate(datasets):
        results[dataset] = {}
        for load_index, load in enumerate(loads):
            results[dataset][str(load)] = {}
            keys = sorted({key for key, policy in indexed if key[0] == dataset and key[1] == load})
            for baseline_index, baseline in enumerate(BASELINES):
                pairs = [(indexed[(key, PRIMARY_POLICY)], indexed[(key, baseline)]) for key in keys]
                points = [paired_point(primary, base) for primary, base in pairs]
                bootstrap = paired_document_bootstrap(
                    pairs, replicates, seed + dataset_index * 100 + load_index * 10 + baseline_index
                )
                by_seed = defaultdict(list)
                for key, point in zip(keys, points):
                    by_seed[key[2]].append(point["throughput_improvement_fraction"])
                direction_seeds = sum(float(np.median(values)) > 0.0 for values in by_seed.values())
                result = {
                    "paired_run_count": len(points),
                    "median_throughput_improvement_fraction": float(np.median([p["throughput_improvement_fraction"] for p in points])),
                    "median_p95_latency_increase_fraction": float(np.median([p["p95_latency_increase_fraction"] for p in points])),
                    "median_deadline_miss_increase": float(np.median([p["deadline_miss_increase"] for p in points])),
                    "throughput_bootstrap_unadjusted_ci95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
                    "one_sided_p_value": float((1 + np.count_nonzero(bootstrap <= 0.0)) / (replicates + 1)),
                    "direction_positive_seed_count": direction_seeds,
                    "bootstrap_values_sha256": hashlib.sha256(bootstrap.tobytes()).hexdigest(),
                    "_bootstrap": bootstrap,
                }
                results[dataset][str(load)][baseline] = result
                if load in (0.8, 0.95):
                    family.append((dataset, load, baseline, result))

    ordered = sorted(family, key=lambda item: item[3]["one_sided_p_value"])
    continuing = True
    family_size = len(ordered)
    for rank, (dataset, load, baseline, result) in enumerate(ordered):
        alpha = 0.05 / (family_size - rank)
        rejected = continuing and result["one_sided_p_value"] <= alpha
        continuing = continuing and rejected
        bootstrap = result.pop("_bootstrap")
        result["holm_rank"] = rank + 1
        result["holm_alpha"] = alpha
        result["holm_corrected_lower_bound"] = float(np.quantile(bootstrap, alpha))
        result["holm_reject_nonpositive_throughput"] = bool(rejected)
    for dataset in results.values():
        for load, baselines in dataset.items():
            for result in baselines.values():
                result.pop("_bootstrap", None)
    return results


def main():
    args = parse_args()
    replay_dir = Path(args.replay_dir)
    route_dir = Path(args.route_safety_dir)
    request_analysis = json.loads(Path(args.request_analysis_json).read_text())
    scenarios = load_scenarios(replay_dir)
    comparisons = analyze_comparisons(scenarios, args.bootstrap_replicates, args.bootstrap_seed)
    route = {
        dataset: route_safety_bootstrap(route_dir, dataset, args.bootstrap_replicates, args.bootstrap_seed + index)
        for index, dataset in enumerate(sorted(comparisons))
    }
    quality_pass = all(
        request_analysis["h3_guarded_mlp"]["by_dataset"][dataset]["by_mode"]["mlp_multibit_dp_guard"]["quality_gate_pass"]
        for dataset in comparisons
    )
    primary_gates = {}
    for dataset, loads in comparisons.items():
        primary_gates[dataset] = {}
        for load in (0.8, 0.95):
            ordinary = loads[str(load)]["ordinary_fcfs"]
            length = loads[str(load)]["length_fcfs"]
            operational = {
                baseline: bool(
                    result["median_throughput_improvement_fraction"] >= 0.05
                    and result["holm_corrected_lower_bound"] > 0.0
                    and result["holm_reject_nonpositive_throughput"]
                    and result["median_p95_latency_increase_fraction"] <= 0.05
                    and result["median_deadline_miss_increase"] <= 0.01
                    and result["direction_positive_seed_count"] >= 2
                )
                for baseline, result in (("ordinary_fcfs", ordinary), ("length_fcfs", length))
            }
            primary_gates[dataset][str(load)] = operational
    performance_pass_ordinary = all(
        values["ordinary_fcfs"] for dataset in primary_gates.values() for values in dataset.values()
    )
    output = {
        "analysis_schema_version": SCHEMA_VERSION,
        "analysis_status": "REAL_GPU_ONLINE_REPLAY_ANALYSIS_COMPLETE",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "replay_summary_sha256": file_sha256(replay_dir / "summary.json"),
        "route_summary_sha256": file_sha256(route_dir / "combined-summary.json"),
        "request_analysis_sha256": file_sha256(args.request_analysis_json),
        "config": vars(args),
        "comparisons": comparisons,
        "route_safety": route,
        "guarded_quality_gate_pass_both_datasets": quality_pass,
        "primary_operational_gates": primary_gates,
        "h4_pass": bool(
            performance_pass_ordinary
            and quality_pass
            and all(value["gate_pass"] for value in route.values())
        ),
        "h4_failure_reasons": [
            reason for condition, reason in (
                (performance_pass_ordinary, "primary performance criteria failed"),
                (quality_pass, "guarded continuation-quality gate failed"),
                (all(value["gate_pass"] for value in route.values()), "guarded route-safety gate failed"),
            ) if not condition
        ],
        "contains_raw_text": False,
    }
    atomic_write_json(args.output_json, output)
    print(json.dumps({
        "analysis_status": output["analysis_status"],
        "h4_pass": output["h4_pass"],
        "h4_failure_reasons": output["h4_failure_reasons"],
    }, indent=2))


if __name__ == "__main__":
    main()
