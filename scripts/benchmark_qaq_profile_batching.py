"""CUDA benchmark for request-level precision-aware dynamic batching.

The benchmark replays one frozen request stream through every policy. It uses
the existing manual cached prefill/decode executor so TTFT and TPOT are
measured separately, and uses the existing QAQ precision auditor in a second
replay for route-level quality violations. Quality auditing is intentionally
not part of timed runs because it dequantizes reference weights for every
observed decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore", message=r".*using max selected bit.*", category=RuntimeWarning)
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from transformers import AutoTokenizer

from any_precision.modules.QAQProfile import (
    account_profile_execution,
    aggregate_profile_accounting,
    build_max_shared_profile,
    route_profile_from_stats,
)
from any_precision import QAQDPLLMForCausalLM, load_qaq_router_checkpoint
from scripts.evaluate_qaq_heldout import QAQPrecisionAuditor
from scripts.qaq_request_demand_protocol import build_dataset_manifest
from scripts.run_qaq_online_scheduler_replay import execute_batch


POLICIES = (
    "fcfs",
    "scalar_predicted",
    "oracle_profile",
    "predicted_profile",
    "uncertainty_fallback",
    "fixed_high",
    "max_profile_sharing",
)
PROFILE_MODE = "mlp_multibit_dp_guard"
SCHEMA_VERSION = "qaq_profile_batching_benchmark_v2_shared_profile"
KERNEL_MAX_BATCH_SIZE = 8


@dataclass(frozen=True)
class Request:
    request_id: str
    dataset: str
    document_id: str
    prompt_length: int
    continuation_length: int
    prompt_ids: torch.Tensor
    arrival_ms: float = 0.0
    predicted_scalar: float = 0.0
    predicted_profile: tuple[float, ...] = ()
    observed_profile: tuple[float, ...] = ()
    layer_group_size: int = 0
    classification_confidence: float = 1.0
    uncertainty_cutoff: float = 0.0
    oracle_safe_bit: int | None = None

    @property
    def cell(self) -> tuple[int, int]:
        return self.prompt_length, self.continuation_length

    @property
    def uncertain(self) -> bool:
        return self.classification_confidence < self.uncertainty_cutoff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark FCFS, scalar/profile-aware, uncertainty-aware, and fixed-high QAQ batching on one CUDA request trace."
    )
    parser.add_argument("--collection_dir", required=True, help="Frozen request-demand collection directory.")
    parser.add_argument("--analysis_json", required=True, help="Held-out prompt-only predictor analysis JSON.")
    parser.add_argument("--ap_model_path", required=True)
    parser.add_argument("--router_checkpoint", required=True)
    parser.add_argument("--estimator_results", required=True)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--datasets", nargs="+", default=["wikitext2", "c4_new"])
    parser.add_argument("--request_limit", type=int, default=8, help="Maximum test requests per dataset; 0 uses the full frozen test split.")
    parser.add_argument("--min_uncertain_requests", type=int, default=0, help="Require this many calibrated-uncertain held-out requests per dataset; selected first when limiting the stream.")
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--arrival_rate", type=float, default=20.0, help="Synthetic replay arrivals per second; the trace is fixed for all policies.")
    parser.add_argument("--arrival_seed", type=int, default=101)
    parser.add_argument("--predictor_seed", type=int, default=17, choices=[17, 29, 43])
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=POLICIES,
        default=list(POLICIES),
        help="Policies to run; use one policy per process when parallelizing independent CUDA runs.",
    )
    parser.add_argument("--max_batch_size", type=int, default=4)
    parser.add_argument("--max_wait_ms", type=float, default=50.0)
    parser.add_argument("--scalar_bucket_size", type=float, default=0.25)
    parser.add_argument("--profile_distance", type=float, default=0.25)
    parser.add_argument("--warmup_batches", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--confidence_threshold", type=float, default=None)
    parser.add_argument("--fallback_bits", type=int, default=1)
    parser.add_argument("--oracle_batch_size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch_dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--skip_quality_audit", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no_trust_remote_code", action="store_false", dest="trust_remote_code")
    return parser.parse_args()


def percentile(values: Iterable[float], q: float) -> float | None:
    values = list(values)
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def summary(values: Iterable[float]) -> dict[str, float | None]:
    values = [float(value) for value in values]
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def git_commit() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def git_dirty() -> bool:
    result = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    return bool(result.stdout.strip())


def environment(device: torch.device) -> dict[str, Any]:
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": smi.stdout.strip() if smi.returncode == 0 else None,
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        result.update({
            "gpu_name": torch.cuda.get_device_name(device),
            "cuda_capability": list(torch.cuda.get_device_capability(device)),
        })
    return result


def read_jsonl_records(collection_dir: Path, dataset: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    shard_dir = collection_dir / "datasets" / dataset / "shards"
    for path in sorted(shard_dir.glob("*.jsonl")):
        with path.open() as source:
            for line in source:
                record = json.loads(line)
                if record.get("partition", record.get("source", {}).get("partition")) == "test":
                    request_id = str(record["request_id"])
                    if request_id in records:
                        raise ValueError(f"Duplicate request_id in collection: {request_id}")
                    records[request_id] = record
    if not records:
        raise ValueError(f"No test records found for dataset {dataset} under {shard_dir}")
    return records


def load_prediction_map(analysis_path: Path, dataset: str, predictor_seed: int) -> tuple[dict[str, dict[str, Any]], float]:
    analysis = json.loads(analysis_path.read_text())
    try:
        result = analysis["h2_predecode_predictability"]["test_by_dataset"][dataset][str(predictor_seed)]
        predictions = {str(item["request_id"]): item for item in result["predictions"]}
        cutoff = float(result["minimum_safe_precision_classifier"]["uncertainty_cutoff_for_90pct_calibration_coverage"])
    except (KeyError, TypeError) as error:
        raise ValueError(f"Analysis has no held-out predictions for dataset={dataset}, seed={predictor_seed}") from error
    if not predictions:
        raise ValueError(f"Empty prediction map for {dataset}, seed={predictor_seed}")
    return predictions, cutoff


def make_requests(
    collection_dir: Path,
    analysis_path: Path,
    tokenizer,
    datasets: list[str],
    request_limit: int,
    max_new_tokens: int,
    predictor_seed: int,
    min_uncertain_requests: int,
) -> list[Request]:
    requests: list[Request] = []
    for dataset in datasets:
        manifest, tensors = build_dataset_manifest(dataset, tokenizer)
        collection_records = read_jsonl_records(collection_dir, dataset)
        predictions, cutoff = load_prediction_map(analysis_path, dataset, predictor_seed)
        selected = [item for item in manifest["requests"] if item["partition"] == "test"]
        selected.sort(key=lambda item: str(item["request_id"]))
        uncertain = [
            item for item in selected
            if float(predictions[str(item["request_id"])]["classification_confidence"]) < cutoff
        ]
        if len(uncertain) < min_uncertain_requests:
            raise ValueError(
                f"Dataset {dataset} has only {len(uncertain)} calibrated-uncertain held-out requests; "
                f"need {min_uncertain_requests}"
            )
        if request_limit:
            if request_limit < min_uncertain_requests:
                raise ValueError("--request_limit must be at least --min_uncertain_requests")
            mandatory = sorted(
                uncertain,
                key=lambda item: (
                    float(predictions[str(item["request_id"])]["classification_confidence"]),
                    str(item["request_id"]),
                ),
            )[:min_uncertain_requests]
            mandatory_ids = {str(item["request_id"]) for item in mandatory}
            selected = mandatory + [
                item for item in selected if str(item["request_id"]) not in mandatory_ids
            ][: request_limit - len(mandatory)]
        if not selected:
            raise ValueError(f"No test requests selected for {dataset}")
        for item in selected:
            request_id = str(item["request_id"])
            record = collection_records.get(request_id)
            prediction = predictions.get(request_id)
            if record is None or prediction is None:
                raise ValueError(f"Missing frozen observed/predicted profile for {request_id}")
            observed = tuple(float(value) for value in record["observed_qaq_profiles"][PROFILE_MODE]["group_expected_bits"])
            predicted = tuple(float(value) for value in prediction["predicted_group_profile"])
            profile_metadata = record["observed_qaq_profiles"][PROFILE_MODE]
            layer_group_size = int(profile_metadata["layer_group_size"])
            profile_dimension = len(profile_metadata["group_expected_bits"])
            if layer_group_size < 1 or len(predicted) != profile_dimension:
                raise ValueError(f"invalid frozen profile metadata for {request_id}")
            if len(predicted) != len(observed):
                raise ValueError(f"profile length mismatch for {request_id}")
            if len(observed) != len(predicted) or not observed:
                raise ValueError(f"Profile length mismatch for {request_id}")
            prompt_length = int(item["prompt_length_tokens"])
            continuation_length = min(int(item["continuation_length_tokens"]), max_new_tokens)
            if continuation_length < 1:
                raise ValueError("--max_new_tokens must be positive")
            requests.append(Request(
                request_id=request_id,
                dataset=dataset,
                document_id=str(item["document_id"]),
                prompt_length=prompt_length,
                continuation_length=continuation_length,
                prompt_ids=tensors[request_id][:prompt_length].clone(),
                layer_group_size=layer_group_size,
                predicted_scalar=float(prediction["predicted_effective_bits"]),
                predicted_profile=predicted,
                observed_profile=observed,
                classification_confidence=float(prediction["classification_confidence"]),
                uncertainty_cutoff=cutoff,
                oracle_safe_bit=int(record["minimum_safe_precision"]["requested_bit"]),
            ))
    if len({request.request_id for request in requests}) != len(requests):
        raise ValueError("Request IDs must be unique across datasets")
    metadata = {(request.layer_group_size, len(request.predicted_profile)) for request in requests}
    if len(metadata) != 1:
        raise ValueError(
            f"all scheduled requests must share frozen profile metadata, got {sorted(metadata)}"
        )
    return requests


def make_arrival_trace(requests: list[Request], seed: int, arrival_rate: float) -> list[Request]:
    if arrival_rate <= 0:
        raise ValueError("--arrival_rate must be positive")
    rng = np.random.default_rng(seed)
    by_cell: dict[tuple[int, int], list[Request]] = defaultdict(list)
    for request in sorted(requests, key=lambda item: item.request_id):
        by_cell[request.cell].append(request)
    per_cell: dict[tuple[int, int], list[Request]] = {}
    for cell, values in sorted(by_cell.items()):
        order = rng.permutation(len(values))
        per_cell[cell] = [values[int(index)] for index in order]
    labels = [cell for cell, values in per_cell.items() for _ in values]
    cell_order = rng.permutation(len(labels))
    positions = {cell: 0 for cell in per_cell}
    ordered: list[Request] = []
    for index in cell_order:
        cell = labels[int(index)]
        request = per_cell[cell][positions[cell]]
        positions[cell] += 1
        ordered.append(request)
    gaps = rng.exponential(1000.0 / arrival_rate, size=len(ordered))
    arrivals = np.cumsum(gaps) - gaps[0]
    return [replace(request, arrival_ms=float(arrivals[index])) for index, request in enumerate(ordered)]


def profile_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right) or not left:
        return math.inf
    return statistics.fmean(abs(a - b) for a, b in zip(left, right))


def scalar_bucket(value: float, bucket_size: float) -> int:
    if bucket_size <= 0:
        raise ValueError("bucket size must be positive")
    return math.floor(float(value) / bucket_size)


def is_compatible(first: Request, candidate: Request, policy: str, scalar_bucket_size: float, profile_threshold: float) -> bool:
    if first.continuation_length != candidate.continuation_length:
        return False
    if policy in {"fcfs", "fixed_high", "max_profile_sharing"}:
        return True
    if policy == "scalar_predicted":
        return scalar_bucket(first.predicted_scalar, scalar_bucket_size) == scalar_bucket(candidate.predicted_scalar, scalar_bucket_size)
    if policy == "oracle_profile":
        return profile_distance(first.observed_profile, candidate.observed_profile) <= profile_threshold
    if policy == "predicted_profile":
        return profile_distance(first.predicted_profile, candidate.predicted_profile) <= profile_threshold
    if policy == "uncertainty_fallback":
        if first.uncertain != candidate.uncertain:
            return False
        return first.uncertain or profile_distance(first.predicted_profile, candidate.predicted_profile) <= profile_threshold
    raise ValueError(f"Unknown policy: {policy}")


def choose_batch(
    unscheduled: list[Request],
    current_time_ms: float,
    policy: str,
    max_batch_size: int,
    max_wait_ms: float,
    scalar_bucket_size: float,
    profile_threshold: float,
) -> tuple[list[Request], float, float, float]:
    if not unscheduled:
        raise ValueError("Cannot choose a batch from an empty queue")
    if max_batch_size < 1 or max_wait_ms < 0:
        raise ValueError("Invalid batch size or wait window")
    ordered = sorted(unscheduled, key=lambda item: (item.arrival_ms, item.request_id))
    first = ordered[0]
    decision_start = time.perf_counter()
    predictor_start = time.perf_counter() if policy not in {"fcfs", "fixed_high"} else None
    window_end = first.arrival_ms + max_wait_ms
    candidates = [
        request for request in ordered
        if request.arrival_ms <= window_end
        and is_compatible(first, request, policy, scalar_bucket_size, profile_threshold)
    ]
    batch = candidates[:max_batch_size]
    predictor_overhead_ms = 0.0
    if predictor_start is not None:
        predictor_overhead_ms = 1000.0 * (time.perf_counter() - predictor_start)
    if len(batch) == max_batch_size:
        schedule_start_ms = max(current_time_ms, max(request.arrival_ms for request in batch))
    else:
        schedule_start_ms = max(current_time_ms, window_end)
    overhead_ms = 1000.0 * (time.perf_counter() - decision_start)
    return batch, schedule_start_ms, overhead_ms, predictor_overhead_ms


def build_batch_plan(requests: list[Request], args: argparse.Namespace, policy: str) -> list[list[Request]]:
    remaining = list(requests)
    plan: list[list[Request]] = []
    while remaining:
        batch, _, _, _ = choose_batch(
            remaining,
            0.0,
            policy,
            args.max_batch_size,
            args.max_wait_ms,
            args.scalar_bucket_size,
            args.profile_distance,
        )
        selected = {request.request_id for request in batch}
        plan.append(batch)
        remaining = [request for request in remaining if request.request_id not in selected]
    return plan


def set_execution_policy(model, policy: str, batch_size: int | None = None) -> str:
    if policy == "max_profile_sharing":
        execution_batch_policy = "shared_profile"
        batch_policy = "group"
    else:
        execution_batch_policy = None
        batch_policy = "group" if policy in {"fcfs", "fixed_high"} or batch_size == 1 else "max"
    for linear in model.ap_linears:
        linear.batch_policy = batch_policy

    model.batch_policy = batch_policy


    return execution_batch_policy or batch_policy
def policy_mode(policy: str, batch: list[Request]) -> str:
    if policy == "max_profile_sharing":
        return "shared_profile"
    if policy == "fixed_high":
        return "fixed_high"
    if policy == "uncertainty_fallback" and batch[0].uncertain:
        return "fixed_high"
    return PROFILE_MODE


def profile_padding(batch: list[Request], policy: str) -> dict[str, float]:
    if policy in {"fcfs", "fixed_high", "max_profile_sharing"}:
        return {"mean_bits": 0.0, "fraction": 0.0, "max_span_bits": 0.0}
    if policy == "scalar_predicted":
        values = np.asarray([[request.predicted_scalar] for request in batch], dtype=np.float64)
    elif policy == "oracle_profile":
        values = np.asarray([request.observed_profile for request in batch], dtype=np.float64)
    elif policy in {"predicted_profile", "uncertainty_fallback"}:
        if policy == "uncertainty_fallback" and batch[0].uncertain:
            return {"mean_bits": 0.0, "fraction": 0.0, "max_span_bits": 0.0}
        values = np.asarray([request.predicted_profile for request in batch], dtype=np.float64)
    else:
        raise ValueError(f"Unknown policy: {policy}")
    shared = values.max(axis=0)
    padding = np.maximum(shared[None, :] - values, 0.0)
    denominator = max(float(values.sum()), 1e-12)
    return {
        "mean_bits": float(padding.mean()),
        "fraction": float(padding.sum() / denominator),
        "max_span_bits": float((values.max(axis=0) - values.min(axis=0)).mean()),
    }


def build_shared_batch_profile(model, batch: list[Request]) -> dict[str, Any]:
    group_sizes = {int(request.layer_group_size) for request in batch}
    if len(group_sizes) != 1 or not next(iter(group_sizes)):
        raise ValueError("shared-profile batch requests must have one positive layer_group_size")
    return build_max_shared_profile(
        [(request.request_id, request.predicted_profile) for request in batch],
        next(iter(group_sizes)),
        model.route_map,
        model.shared_route_valid_bits(),
    )


def request_stream_hash(requests: list[Request]) -> str:
    payload = [
        {
            "request_id": request.request_id,
            "arrival_ms": request.arrival_ms,
            "prompt_length": request.prompt_length,
            "continuation_length": request.continuation_length,
        }
        for request in requests
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def merge_bit_histogram(target: dict[str, dict[str, int]], stats: dict[str, Any]) -> None:
    for route, route_stats in stats.get("per_layer", {}).items():
        output = target.setdefault(route, defaultdict(int))
        for bit, count in route_stats.get("bit_counts", {}).items():
            output[str(bit)] += int(count)


def aggregate_router_stats(batch_results: list[dict[str, Any]]) -> dict[str, Any]:
    total_tokens = 0
    selected_bits = 0
    weighted_effective = 0.0
    shared_profile_tokens = 0
    fallback = 0
    guard = 0
    histogram: dict[str, dict[str, int]] = {}
    for batch in batch_results:
        stats = batch["router_stats"]
        tokens = int(stats.get("total_tokens", 0))
        total_tokens += tokens
        selected_bits += int(round(float(stats.get("average_selected_bit", 0.0)) * tokens))
        weighted_effective += float(stats.get("effective_bits", 0.0)) * tokens
        fallback += int(stats.get("total_fallbacks", 0))
        guard += int(stats.get("total_dp_guard_triggers", 0))
        shared_profile_tokens += int(stats.get("total_shared_profile_tokens", 0))
        merge_bit_histogram(histogram, stats)
    return {
        "total_tokens": total_tokens,
        "average_selected_bit": selected_bits / total_tokens if total_tokens else 0.0,
        "effective_bits": weighted_effective / total_tokens if total_tokens else 0.0,
        "total_fallbacks": fallback,
        "fallback_rate": fallback / total_tokens if total_tokens else 0.0,
        "total_dp_guard_triggers": guard,
        "dp_guard_rate": guard / total_tokens if total_tokens else 0.0,
        "per_layer_bit_histogram": {route: dict(sorted(counts.items())) for route, counts in sorted(histogram.items())},
        "shared_profile_execution": bool(shared_profile_tokens),
        "total_shared_profile_tokens": shared_profile_tokens,
        "shared_profile_row_fraction": shared_profile_tokens / total_tokens if total_tokens else 0.0,
    }


def summarize_repeat(request_results: list[dict[str, Any]], batch_results: list[dict[str, Any]], start_ms: float, finish_ms: float, args: argparse.Namespace) -> dict[str, Any]:
    elapsed_s = max((finish_ms - start_ms) / 1000.0, 1e-12)
    generated = sum(int(item["generated_tokens"]) for item in request_results)
    prompt_slots = sum(int(item["prompt_token_slots"]) for item in batch_results)
    prompt_nonpadding = sum(int(item["prompt_nonpadding_tokens"]) for item in batch_results)
    batch_sizes = [int(item["batch_size"]) for item in batch_results]
    padding_values = [float(item["profile_padding"]["mean_bits"]) for item in batch_results]
    padding_fractions = [float(item["profile_padding"]["fraction"]) for item in batch_results]
    return {
        "request_count": len(request_results),
        "batch_count": len(batch_results),
        "mean_batch_size": statistics.fmean(batch_sizes),
        "batch_occupancy": statistics.fmean(size / args.max_batch_size for size in batch_sizes),
        "prompt_token_occupancy": prompt_nonpadding / prompt_slots if prompt_slots else 0.0,
        "requests_per_s": len(request_results) / elapsed_s,
        "generated_tokens_per_s": generated / elapsed_s,
        "token_slots_per_s": (prompt_slots + generated) / elapsed_s,
        "latency_ms": summary(item["end_to_end_latency_ms"] for item in request_results),
        "ttft_ms": summary(item["ttft_ms"] for item in request_results),
        "tpot_ms": summary(item["tpot_ms"] for item in request_results),
        "queue_delay_ms": summary(item["queue_delay_ms"] for item in request_results),
        "profile_padding_bits": statistics.fmean(padding_values) if padding_values else 0.0,
        "profile_padding_fraction": statistics.fmean(padding_fractions) if padding_fractions else 0.0,
        "predictor_overhead_ms": sum(float(item["predictor_overhead_ms"]) for item in batch_results),
        "scheduler_overhead_ms": sum(float(item["scheduler_overhead_ms"]) for item in batch_results),
        "uncertainty_fallback_requests": sum(bool(item["uncertain_fallback_lane"]) for item in request_results),
        "finish_ms": finish_ms,
    }


def run_policy(model, requests: list[Request], tokenizer, args: argparse.Namespace, policy: str) -> dict[str, Any]:
    templates = build_batch_plan(requests, args, policy)
    batch_policy = set_execution_policy(model, policy, args.max_batch_size)
    for warmup_index, batch in enumerate(templates[: args.warmup_batches]):
        set_execution_policy(model, policy, len(batch))
        shared_profile = build_shared_batch_profile(model, batch) if policy == "max_profile_sharing" else None
        execute_batch(
            model,
            batch,
            policy_mode(policy, batch),
            tokenizer.pad_token_id,
            torch.device(args.device),
            shared_profile=shared_profile["shared_route_profile"] if shared_profile else None,
        )
        print(f"warmup policy={policy} batch={warmup_index + 1}/{min(args.warmup_batches, len(templates))}", flush=True)

    repeats: list[dict[str, Any]] = []
    for repeat_index in range(args.repeat):
        remaining = list(requests)
        current_time_ms = 0.0
        batch_results: list[dict[str, Any]] = []
        request_results: list[dict[str, Any]] = []
        while remaining:
            batch, schedule_start_ms, scheduler_overhead_ms, predictor_overhead_ms = choose_batch(
                remaining,
                current_time_ms,
                policy,
                args.max_batch_size,
                args.max_wait_ms,
                args.scalar_bucket_size,
                args.profile_distance,
            )
            mode = policy_mode(policy, batch)
            batch_policy_for_execution = set_execution_policy(model, policy, len(batch))
            shared_profile = build_shared_batch_profile(model, batch) if policy == "max_profile_sharing" else None
            gpu_start_ms = schedule_start_ms + scheduler_overhead_ms
            execution = execute_batch(
                model,
                batch,
                mode,
                tokenizer.pad_token_id,
                torch.device(args.device),
                shared_profile=shared_profile["shared_route_profile"] if shared_profile else None,
            )
            if shared_profile is not None:
                executed = route_profile_from_stats(execution["router_stats"])
                shared_profile["executed_route_profile"] = executed
                shared_profile["profile_accounting"] = account_profile_execution(
                    shared_profile, executed, model.route_map, model.shared_route_valid_bits()
                )
            finish_ms = gpu_start_ms + float(execution["service_ms"])
            padding = profile_padding(batch, policy)
            batch_id = f"{policy}-repeat{repeat_index}-batch{len(batch_results):04d}"
            batch_result = {
                "batch_id": batch_id,
                "request_ids": [request.request_id for request in batch],
                "shared_profile": shared_profile,
                "batch_size": len(batch),
                "policy": policy,
                "mode": mode,
                "batch_policy": batch_policy_for_execution,
                "schedule_start_ms": schedule_start_ms,
                "gpu_start_ms": gpu_start_ms,
                "finish_ms": finish_ms,
                "scheduler_overhead_ms": scheduler_overhead_ms,
                "predictor_overhead_ms": predictor_overhead_ms,
                "profile_padding": padding,
                **execution,
            }
            batch_results.append(batch_result)
            for request in batch:
                queue_delay_ms = gpu_start_ms - request.arrival_ms
                request_results.append({
                    "request_id": request.request_id,
                    "batch_id": batch_id,
                    "arrival_ms": request.arrival_ms,
                    "queue_delay_ms": queue_delay_ms,
                    "end_to_end_latency_ms": finish_ms - request.arrival_ms,
                    "ttft_ms": queue_delay_ms + float(execution["ttft_ms"]),
                    "tpot_ms": float(execution["tpot_ms"]),
                    "generated_tokens": request.continuation_length,
                    "uncertain_fallback_lane": bool(policy == "uncertainty_fallback" and request.uncertain),
                })
            selected = {request.request_id for request in batch}
            remaining = [request for request in remaining if request.request_id not in selected]
            current_time_ms = finish_ms
            print(
                f"{policy} repeat={repeat_index} batch={len(batch_results)}/{len(templates)} "
                f"size={len(batch)} service={execution['service_ms']:.2f}ms mode={mode}",
                flush=True,
            )
        repeats.append({
            "repeat_index": repeat_index,
            "summary": summarize_repeat(request_results, batch_results, min(request.arrival_ms for request in requests), current_time_ms, args),
            "batches": batch_results,
            "requests": request_results,
        })

    all_batches = [batch for repeat in repeats for batch in repeat["batches"]]
    all_requests = [request for repeat in repeats for request in repeat["requests"]]
    repeat_summaries = [repeat["summary"] for repeat in repeats]
    return {
        "policy": policy,
        "execution_batch_policy": batch_policy,
        "batch_plan": [[request.request_id for request in batch] for batch in templates],
        "shared_profile_batch_count": sum(bool(item.get("shared_profile")) for item in all_batches),
        "profile_accounting": aggregate_profile_accounting(
            [item["shared_profile"]["profile_accounting"] for item in all_batches if item.get("shared_profile")]
        ) if any(item.get("shared_profile") for item in all_batches) else None,
        "warmup_batches": min(args.warmup_batches, len(templates)),
        "repeats": repeats,
        "summary": {
            "repeat_count": args.repeat,
            "request_count": len(requests),
            "batch_count_per_repeat": len(templates),
            "requests_per_s": statistics.fmean(item["requests_per_s"] for item in repeat_summaries),
            "generated_tokens_per_s": statistics.fmean(item["generated_tokens_per_s"] for item in repeat_summaries),
            "token_slots_per_s": statistics.fmean(item["token_slots_per_s"] for item in repeat_summaries),
            "latency_ms": summary(item["end_to_end_latency_ms"] for item in all_requests),
            "ttft_ms": summary(item["ttft_ms"] for item in all_requests),
            "tpot_ms": summary(item["tpot_ms"] for item in all_requests),
            "queue_delay_ms": summary(item["queue_delay_ms"] for item in all_requests),
            "batch_occupancy": statistics.fmean(item["batch_occupancy"] for item in repeat_summaries),
            "prompt_token_occupancy": statistics.fmean(item["prompt_token_occupancy"] for item in repeat_summaries),
            "profile_padding_bits": statistics.fmean(item["profile_padding_bits"] for item in repeat_summaries),
            "profile_padding_fraction": statistics.fmean(item["profile_padding_fraction"] for item in repeat_summaries),
            "predictor_overhead_ms": sum(item["predictor_overhead_ms"] for item in repeat_summaries) / args.repeat,
            "scheduler_overhead_ms": sum(item["scheduler_overhead_ms"] for item in repeat_summaries) / args.repeat,
            "uncertainty_fallback_rate": sum(item["uncertainty_fallback_requests"] for item in repeat_summaries) / max(len(requests) * args.repeat, 1),
            "router_stats": aggregate_router_stats(all_batches),
        },
    }


def merge_quality_counts(target: dict[str, int], report: dict[str, Any]) -> None:
    keys = (
        "decision_count",
        "under_precision_count",
        "over_precision_count",
        "exact_precision_count",
        "signed_bit_gap_sum",
        "absolute_bit_gap_sum",
    )
    for key in keys:
        target[key] = target.get(key, 0) + int(report.get(key, 0))


def quality_audit(model, policy: str, batch_plan: list[list[Request]], tokenizer, args: argparse.Namespace, error_threshold: float) -> dict[str, Any]:
    batch_policy = set_execution_policy(model, policy, args.max_batch_size)
    auditor = QAQPrecisionAuditor(error_threshold, args.oracle_batch_size)
    model.set_decision_observer(auditor)
    aggregate: dict[str, int] = {}
    for batch_index, batch in enumerate(batch_plan):
        mode = policy_mode(policy, batch)
        set_execution_policy(model, policy, len(batch))
        shared_profile = build_shared_batch_profile(model, batch) if policy == "max_profile_sharing" else None
        auditor.start_mode(policy)
        auditor.start_example(batch_index)
        execute_batch(
            model, batch, mode, tokenizer.pad_token_id, torch.device(args.device),
            shared_profile=shared_profile["shared_route_profile"] if shared_profile else None,
        )
        merge_quality_counts(aggregate, auditor.report()["summary"])
        print(f"quality-audit policy={policy} batch={batch_index + 1}/{len(batch_plan)}", flush=True)
    model.set_decision_observer(None)
    total = aggregate.get("decision_count", 0)
    return {
        "execution_batch_policy": batch_policy,
        "quality_violation_count": aggregate.get("under_precision_count", 0),
        "quality_decision_count": total,
        "quality_violation_rate": aggregate.get("under_precision_count", 0) / total if total else None,
        "over_precision_rate": aggregate.get("over_precision_count", 0) / total if total else None,
        "exact_precision_rate": aggregate.get("exact_precision_count", 0) / total if total else None,
        "mean_signed_bit_gap": aggregate.get("signed_bit_gap_sum", 0) / total if total else None,
        "mean_absolute_bit_gap": aggregate.get("absolute_bit_gap_sum", 0) / total if total else None,
        "precision_counts": aggregate,
        "definition": "quality_violation_rate is the fraction of real low-bit/reference-bit route decisions where selected bit was below the auditor-required safe bit.",
    }


def validate_args(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA device")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES explicitly before running the CUDA benchmark")
    if args.request_limit < 0 or args.min_uncertain_requests < 0 or args.max_batch_size < 1 or args.repeat < 1 or args.warmup_batches < 0:
        raise ValueError("request limit, uncertainty minimum, batch size, repeat, and warmups must be valid")
    if args.max_batch_size > KERNEL_MAX_BATCH_SIZE:
        raise ValueError(f"--max_batch_size must be <= {KERNEL_MAX_BATCH_SIZE} for the installed Any-Precision CUDA kernel")
    for filename in ("linear_reg_d.pt", "jl_d.pt", "T_d.pt", "max_mem_dict.pt"):
        if not (Path(args.estimator_results) / filename).is_file():
            raise FileNotFoundError(f"Missing estimator artifact: {Path(args.estimator_results) / filename}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device = torch.device(args.device)
    tokenizer_path = args.tokenizer_path or args.ap_model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=args.local_files_only, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    requests = make_requests(
        Path(args.collection_dir),
        Path(args.analysis_json),
        tokenizer,
        args.datasets,
        args.request_limit,
        args.max_new_tokens,
        args.predictor_seed,
        args.min_uncertain_requests,
    )
    requests = make_arrival_trace(requests, args.arrival_seed, args.arrival_rate)
    stream_hash = request_stream_hash(requests)
    print(f"request stream: {len(requests)} requests hash={stream_hash}", flush=True)

    router, checkpoint = load_qaq_router_checkpoint(args.router_checkpoint)
    error_threshold = float(checkpoint["error_threshold"])
    model = QAQDPLLMForCausalLM.from_quantized(
        args.ap_model_path,
        router=router,
        router_metadata=checkpoint,
        estimator_results=args.estimator_results,
        precisions=[3, 4, 5, 6],
        torch_dtype={"float16": torch.float16, "bfloat16": torch.bfloat16}[args.torch_dtype],
        router_mode=PROFILE_MODE,
        confidence_threshold=args.confidence_threshold,
        fallback_bits=args.fallback_bits,
        prefill_by_router=True,
        batch_policy="group",
        trust_remote_code=args.trust_remote_code,
    ).eval().to(device)

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "REAL_CUDA_BENCHMARK",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment(device),
        "request_stream": {
            "request_count": len(requests),
            "request_ids": [request.request_id for request in requests],
            "arrival_ms": [request.arrival_ms for request in requests],
            "stream_sha256": stream_hash,
            "datasets": args.datasets,
            "max_new_tokens": args.max_new_tokens,
            "arrival_rate": args.arrival_rate,
            "arrival_seed": args.arrival_seed,
            "predictor_seed": args.predictor_seed,
            "uncertain_request_count": sum(request.uncertain for request in requests),
            "uncertain_request_ids": [request.request_id for request in requests if request.uncertain],
        },
        "config": {
            "collection_dir": args.collection_dir,
            "analysis_json": args.analysis_json,
            "ap_model_path": args.ap_model_path,
            "router_checkpoint": args.router_checkpoint,
            "estimator_results": args.estimator_results,
            "bits": [3, 4, 5, 6],
            "max_batch_size": args.max_batch_size,
            "max_wait_ms": args.max_wait_ms,
            "scalar_bucket_size": args.scalar_bucket_size,
            "profile_distance": args.profile_distance,
            "warmup_batches": args.warmup_batches,
            "repeat": args.repeat,
            "min_uncertain_requests": args.min_uncertain_requests,
            "confidence_threshold": args.confidence_threshold,
            "fallback_bits": args.fallback_bits,
            "quality_audit": not args.skip_quality_audit,
            "quality_error_threshold": error_threshold,
        },
        "shared_profile_contract": {
            "policy": "max_profile_sharing",
            "profile_source": "predicted_group_profile",
            "composition": "component-wise maximum over the actual scheduled batch",
            "projection_rule": "route-valid conservative ceiling",
            "singleton_behavior": "singleton predicted profile is still applied",
            "quantile_sharing": "pending",
        },
        "policies": {},
        "limitations": [
            "Arrival times are a deterministic replay trace generated from the selected frozen test requests; all policies consume the identical trace.",
            "max_profile_sharing applies one projected route-level profile throughout prefill and decode; its continuous group profile and executed route profile are recorded per batch.",
            "Scheduler-profile under/exact/over accounting compares executed route bits with each request's projected predicted target; it is separate from QAQPrecisionAuditor route safety.",
            "Quantile profile sharing is pending and is not included in this schema.",
            "Quality violations are measured by the real output-error auditor in a separate replay and are not included in timed latency or throughput.",
            "Predictor overhead is the measured CPU cost of consuming held-out predictor outputs and applying bucketing/fallback decisions; predictor model fitting is not timed.",
        ],
    }
    for policy in args.policies:
        result["policies"][policy] = run_policy(model, requests, tokenizer, args, policy)
        if not args.skip_quality_audit:
            request_by_id = {request.request_id: request for request in requests}
            plan = [
                [request_by_id[request_id] for request_id in batch_ids]
                for batch_ids in result["policies"][policy]["batch_plan"]
            ]
            result["policies"][policy]["quality_audit"] = quality_audit(
                model, policy, plan, tokenizer, args, error_threshold
            )
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n")

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
