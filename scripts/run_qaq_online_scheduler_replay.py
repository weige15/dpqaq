"""Resumable preregistered online-queue GPU replay for the primary H4 policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import socket
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import accelerate
import datasets
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer

from any_precision import QAQDPLLMForCausalLM
from scripts.analyze_qaq_request_demand_preregistered import verify_freeze
from scripts.qaq_request_demand_protocol import (
    atomic_write_json,
    build_dataset_manifest,
    file_manifest,
    file_sha256,
    object_sha256,
    tokenizer_file_manifest,
    token_ids_sha256,
)

RUN_SCHEMA = "qaq_online_scheduler_replay_run_v1"
SCENARIO_SCHEMA = "qaq_online_scheduler_replay_scenario_v1"
SUMMARY_SCHEMA = "qaq_online_scheduler_replay_summary_v1"
POLICIES = ("ordinary_fcfs", "length_fcfs", "predicted_block_fallback_lane")
LOAD_FRACTIONS = (0.50, 0.80, 0.95)
SCHEDULING_SEEDS = (101, 202, 303)
PREDICTOR_SEEDS = (17, 29, 43)
PREDICTOR_SEED_BY_SCHEDULING_SEED = dict(zip(SCHEDULING_SEEDS, PREDICTOR_SEEDS, strict=True))
MAX_BATCH_SIZE = 4
MAX_WAIT_MS = 50.0
PROFILE_DISTANCE = 0.25
FORBIDDEN_ARTIFACT_KEYS = {"text", "prompt_text", "generated_text", "input_ids", "prompt_ids", "token_ids", "tokens"}


@dataclass
class ReplayRequest:
    request_id: str
    dataset: str
    document_id: str
    prompt_length: int
    continuation_length: int
    prompt_ids: torch.Tensor
    arrival_ms: float = 0.0
    predicted_profile: tuple[float, ...] = ()
    classification_confidence: float = 1.0
    uncertainty_cutoff: float = 0.0

    @property
    def uncertain(self) -> bool:
        return self.classification_confidence < self.uncertainty_cutoff

    @property
    def cell(self) -> tuple[int, int]:
        return self.prompt_length, self.continuation_length


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real online-queue replay for primary H4 policies.")
    parser.add_argument("--collection_dir", required=True)
    parser.add_argument("--freeze_manifest", required=True)
    parser.add_argument("--analysis_json", required=True)
    parser.add_argument("--route_safety_dir", required=True)
    parser.add_argument("--ap_model_path", required=True)
    parser.add_argument("--router_checkpoint", required=True)
    parser.add_argument("--estimator_results", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bits", type=int, nargs="+", default=[3, 4, 5, 6])
    parser.add_argument("--confidence_threshold", type=float, default=0.6)
    parser.add_argument("--fallback_bits", type=int, default=1)
    parser.add_argument("--warmup_batches", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--bootstrap_replicates", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=1729)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch_dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--max_scenarios", type=int, default=None, help="Diagnostic only; omit for registered replay.")
    parser.add_argument("--diagnostic_skip_warmups", action="store_true")
    parser.add_argument("--diagnostic_request_limit", type=int, default=None)
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q)) if values else 0.0


def source_hashes() -> dict[str, str]:
    paths = (
        "scripts/run_qaq_online_scheduler_replay.py",
        "scripts/analyze_qaq_request_demand_preregistered.py",
        "scripts/collect_qaq_route_safety_supplement.py",
        "scripts/qaq_request_demand_protocol.py",
        "doc/qaq-request-demand-preregistered-protocol.md",
        "any_precision/modules/QAQDPLLM_Linear.py",
        "any_precision/modules/QAQDPLLMForCausalLM.py",
    )
    return {path: file_sha256(REPO_ROOT / path) for path in paths}


def environment_metadata(device: torch.device | None) -> dict[str, Any]:
    driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True)
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "accelerate": accelerate.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_driver": driver.stdout.splitlines()[0].strip() if driver.returncode == 0 else None,
        "command": [sys.executable, *sys.argv],
    }
    if device is not None and device.type == "cuda":
        result["cuda_device"] = torch.cuda.get_device_name(device)
        result["cuda_capability"] = list(torch.cuda.get_device_capability(device))
    return result


def validate_external_output(output_dir: Path, collection_dir: Path, route_dir: Path) -> None:
    resolved = output_dir.resolve()
    for immutable in (collection_dir.resolve(), route_dir.resolve()):
        if resolved == immutable or immutable in resolved.parents:
            raise ValueError("Replay output must be outside immutable input artifacts")


def profile_distance(left: ReplayRequest, right: ReplayRequest) -> float:
    if len(left.predicted_profile) != len(right.predicted_profile) or not left.predicted_profile:
        return math.inf
    return statistics.fmean(abs(a - b) for a, b in zip(left.predicted_profile, right.predicted_profile))


def compatible(first: ReplayRequest, candidate: ReplayRequest, policy: str) -> bool:
    if first.continuation_length != candidate.continuation_length:
        return False
    if policy == "ordinary_fcfs":
        return True
    if first.cell != candidate.cell:
        return False
    if policy == "length_fcfs":
        return True
    if policy == "predicted_block_fallback_lane":
        if first.uncertain != candidate.uncertain:
            return False
        return first.uncertain or profile_distance(first, candidate) <= PROFILE_DISTANCE
    raise ValueError(f"Unknown policy: {policy}")


def choose_online_batch(
    unscheduled: list[ReplayRequest], current_time_ms: float, policy: str
) -> tuple[list[ReplayRequest], float, float]:
    ordered = sorted(unscheduled, key=lambda request: (request.arrival_ms, request.request_id))
    first = ordered[0]
    decision_start = time.perf_counter()
    window_end = first.arrival_ms + MAX_WAIT_MS
    horizon = max(current_time_ms, window_end)
    candidates = [request for request in ordered if request.arrival_ms <= horizon and compatible(first, request, policy)]
    batch = candidates[:MAX_BATCH_SIZE]
    if len(batch) == MAX_BATCH_SIZE:
        schedule_start = max(current_time_ms, max(request.arrival_ms for request in batch))
    else:
        schedule_start = max(current_time_ms, window_end)
    overhead_ms = 1000.0 * (time.perf_counter() - decision_start)
    return batch, schedule_start, overhead_ms


def deterministic_arrivals(
    requests: list[ReplayRequest], dataset: str, seed: int, arrival_rate: float
) -> list[ReplayRequest]:
    dataset_salt = int(hashlib.sha256(dataset.encode("ascii")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed + dataset_salt)
    by_cell: dict[tuple[int, int], list[ReplayRequest]] = {}
    for cell in sorted({request.cell for request in requests}):
        values = [request for request in requests if request.cell == cell]
        order = rng.permutation(len(values))
        by_cell[cell] = [values[int(index)] for index in order]
    cell_labels = [cell for cell, values in by_cell.items() for _ in values]
    cell_order = rng.permutation(len(cell_labels))
    positions = {cell: 0 for cell in by_cell}
    ordered = []
    for index in cell_order:
        cell = cell_labels[int(index)]
        request = by_cell[cell][positions[cell]]
        positions[cell] += 1
        ordered.append(request)
    interarrival_ms = rng.exponential(1000.0 / arrival_rate, size=len(ordered))
    arrivals = np.cumsum(interarrival_ms) - interarrival_ms[0]
    return [
        ReplayRequest(**{**request.__dict__, "arrival_ms": float(arrivals[index])})
        for index, request in enumerate(ordered)
    ]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def left_pad_batch(requests: list[ReplayRequest], pad_token_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(request.prompt_ids.numel() for request in requests)
    input_ids = torch.full((len(requests), width), pad_token_id, dtype=torch.long)
    attention = torch.zeros((len(requests), width), dtype=torch.long)
    for row, request in enumerate(requests):
        length = request.prompt_ids.numel()
        input_ids[row, -length:] = request.prompt_ids
        attention[row, -length:] = 1
    return input_ids.to(device), attention.to(device)


@torch.no_grad()
def execute_batch(
    model,
    requests: list[ReplayRequest],
    mode: str,
    pad_token_id: int,
    device: torch.device,
) -> dict[str, Any]:
    continuation_length = requests[0].continuation_length
    if any(request.continuation_length != continuation_length for request in requests):
        raise ValueError("A real batch cannot mix continuation lengths")
    input_ids, attention_mask = left_pad_batch(requests, pad_token_id, device)
    model.clear_router_stats()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    prefill_start = time.perf_counter()
    position_ids = attention_mask.cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
        router_mode=mode,
    )
    next_tokens = outputs.logits[:, -1, :].argmax(dim=-1)
    generated = [next_tokens]
    past_key_values = outputs.past_key_values
    synchronize(device)
    ttft_ms = 1000.0 * (time.perf_counter() - prefill_start)

    synchronize(device)
    decode_start = time.perf_counter()
    for _ in range(1, continuation_length):
        attention_mask = torch.cat(
            [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=device)],
            dim=1,
        )
        position_ids = attention_mask.sum(dim=-1, keepdim=True) - 1
        outputs = model(
            input_ids=next_tokens.unsqueeze(1),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            router_mode=mode,
        )
        past_key_values = outputs.past_key_values
        next_tokens = outputs.logits[:, -1, :].argmax(dim=-1)
        generated.append(next_tokens)
    synchronize(device)
    decode_ms = 1000.0 * (time.perf_counter() - decode_start)
    generated_ids = torch.stack(generated, dim=1).cpu()
    service_ms = ttft_ms + decode_ms
    stats = model.get_router_stats()
    return {
        "batch_size": len(requests),
        "mode": mode,
        "prompt_token_slots": int(input_ids.numel()),
        "prompt_nonpadding_tokens": int(attention_mask[:, :input_ids.shape[1]].sum().item()),
        "generated_token_slots": int(generated_ids.numel()),
        "ttft_ms": ttft_ms,
        "decode_ms": decode_ms,
        "service_ms": service_ms,
        "tpot_ms": decode_ms / max(continuation_length - 1, 1),
        "generated_token_sha256": {
            request.request_id: token_ids_sha256(generated_ids[index]) for index, request in enumerate(requests)
        },
        "router_stats": stats,
        "cuda_memory": {
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
        },
    }


def summarize_scenario(request_results: list[dict[str, Any]], batch_results: list[dict[str, Any]], start_ms: float, finish_ms: float) -> dict[str, Any]:
    end_to_end = [item["end_to_end_latency_ms"] for item in request_results]
    queue = [item["queue_delay_ms"] for item in request_results]
    ttft = [item["ttft_ms"] for item in request_results]
    tpot = [item["tpot_ms"] for item in request_results]
    generated = sum(item["generated_tokens"] for item in request_results)
    prompt_slots = sum(batch["prompt_token_slots"] for batch in batch_results)
    prompt_nonpadding = sum(batch["prompt_nonpadding_tokens"] for batch in batch_results)
    elapsed_s = max((finish_ms - start_ms) / 1000.0, 1e-12)
    batch_sizes = [batch["batch_size"] for batch in batch_results]
    lane_occupancy = defaultdict(int)
    per_layer_histogram: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    effective_bits = []
    average_bits = []
    fallbacks = 0
    guard_triggers = 0
    profiles = []
    for batch in batch_results:
        lane_occupancy[batch["lane_id"]] += batch["batch_size"]
        stats = batch["router_stats"]
        effective_bits.append(float(stats["effective_bits"]))
        average_bits.append(float(stats["average_selected_bit"]))
        fallbacks += int(stats["total_fallbacks"])
        guard_triggers += int(stats["total_dp_guard_triggers"])
        profile = []
        for route, route_stats in sorted(stats["per_layer"].items()):
            counts = route_stats["bit_counts"]
            for bit, count in counts.items():
                per_layer_histogram[route][bit] += int(count)
            profile.append(max(((int(count), int(bit)) for bit, count in counts.items()), default=(0, 0))[1])
        profiles.append(tuple(profile))
    profile_switches = sum(left != right for left, right in zip(profiles, profiles[1:]))
    return {
        "request_count": len(request_results),
        "batch_count": len(batch_results),
        "mean_batch_size": len(request_results) / len(batch_results),
        "batch_size_distribution": {str(size): batch_sizes.count(size) for size in sorted(set(batch_sizes))},
        "batch_occupancy_fraction": statistics.fmean(size / MAX_BATCH_SIZE for size in batch_sizes),
        "fragmentation_fraction": 1.0 - statistics.fmean(size / MAX_BATCH_SIZE for size in batch_sizes),
        "lane_count": len(lane_occupancy),
        "lane_occupancy": dict(sorted(lane_occupancy.items())),
        "requests_per_s": len(request_results) / elapsed_s,
        "generated_tokens_per_s": generated / elapsed_s,
        "token_slots_per_s": (prompt_slots + generated) / elapsed_s,
        "nonpadding_tokens_per_s": (prompt_nonpadding + generated) / elapsed_s,
        "mean_effective_bits": statistics.fmean(effective_bits),
        "mean_average_selected_bit": statistics.fmean(average_bits),
        "total_fallbacks": fallbacks,
        "total_dp_guard_triggers": guard_triggers,
        "profile_switch_count": profile_switches,
        "per_layer_bit_histogram": {route: dict(sorted(counts.items())) for route, counts in sorted(per_layer_histogram.items())},
        "queue_delay_ms": {"p50": percentile(queue, 0.50), "p95": percentile(queue, 0.95), "p99": percentile(queue, 0.99)},
        "end_to_end_latency_ms": {"p50": percentile(end_to_end, 0.50), "p95": percentile(end_to_end, 0.95), "p99": percentile(end_to_end, 0.99)},
        "ttft_ms": {"p50": percentile(ttft, 0.50), "p95": percentile(ttft, 0.95), "p99": percentile(ttft, 0.99)},
        "tpot_ms": {"p50": percentile(tpot, 0.50), "p95": percentile(tpot, 0.95), "p99": percentile(tpot, 0.99)},
        "deadline_miss_fraction": statistics.fmean(item["deadline_missed"] for item in request_results),
        "scheduler_cpu_overhead_ms": sum(batch["scheduler_cpu_overhead_ms"] for batch in batch_results),
        "generated_tokens": generated,
        "peak_allocated_bytes": max(batch["cuda_memory"]["max_allocated_bytes"] for batch in batch_results),
        "peak_reserved_bytes": max(batch["cuda_memory"]["max_reserved_bytes"] for batch in batch_results),
        "oom_count": 0,
    }


def run_online_scenario(
    model,
    requests: list[ReplayRequest],
    policy: str,
    deadlines_ms: dict[str, float],
    pad_token_id: int,
    device: torch.device,
    scenario_id: str,
) -> dict[str, Any]:
    unscheduled = list(requests)
    current_time_ms = 0.0
    batch_results = []
    request_results = []
    while unscheduled:
        batch, schedule_start_ms, overhead_ms = choose_online_batch(unscheduled, current_time_ms, policy)
        mode = "fixed_high" if policy == "predicted_block_fallback_lane" and batch[0].uncertain else "mlp_multibit_dp_guard"
        if policy == "predicted_block_fallback_lane":
            if batch[0].uncertain:
                lane_id = "fallback_fixed6"
            else:
                payload = json.dumps(batch[0].predicted_profile, separators=(",", ":")).encode("ascii")
                lane_id = "predicted_block_" + hashlib.sha256(payload).hexdigest()[:12]
        else:
            lane_id = policy
        gpu_start_ms = schedule_start_ms + overhead_ms
        execution = execute_batch(model, batch, mode, pad_token_id, device)
        finish_ms = gpu_start_ms + execution["service_ms"]
        batch_id = f"{scenario_id}-batch-{len(batch_results):05d}"
        batch_results.append({
            "batch_id": batch_id,
            "request_ids": [request.request_id for request in batch],
            "schedule_start_ms": schedule_start_ms,
            "gpu_start_ms": gpu_start_ms,
            "finish_ms": finish_ms,
            "scheduler_cpu_overhead_ms": overhead_ms,
            "lane_id": lane_id,
            **execution,
        })
        for request in batch:
            deadline = deadlines_ms[f"{request.prompt_length}p:{request.continuation_length}c"]
            queue_delay = gpu_start_ms - request.arrival_ms
            end_to_end = finish_ms - request.arrival_ms
            request_results.append({
                "request_id": request.request_id,
                "document_id": request.document_id,
                "arrival_ms": request.arrival_ms,
                "batch_id": batch_id,
                "mode": mode,
                "queue_delay_ms": queue_delay,
                "gpu_service_share_ms": execution["service_ms"] / len(batch),
                "end_to_end_latency_ms": end_to_end,
                "ttft_ms": queue_delay + execution["ttft_ms"],
                "tpot_ms": execution["tpot_ms"],
                "deadline_ms": deadline,
                "deadline_missed": bool(end_to_end > deadline),
                "generated_tokens": request.continuation_length,
                "generated_token_sha256": execution["generated_token_sha256"][request.request_id],
                "uncertain_fallback_lane": request.uncertain,
            })
        batch_ids = {request.request_id for request in batch}
        unscheduled = [request for request in unscheduled if request.request_id not in batch_ids]
        current_time_ms = finish_ms
    start_ms = min(request.arrival_ms for request in requests)
    summary = summarize_scenario(request_results, batch_results, start_ms, current_time_ms)
    return {
        "scenario_schema_version": SCENARIO_SCHEMA,
        "scenario_id": scenario_id,
        "policy": policy,
        "summary": summary,
        "batches": batch_results,
        "requests": sorted(request_results, key=lambda item: item["request_id"]),
        "contains_raw_text": False,
    }


def predictor_map(
    analysis: dict[str, Any], dataset: str, scheduling_seed: int
) -> tuple[dict[str, dict[str, Any]], float]:
    try:
        predictor_seed = PREDICTOR_SEED_BY_SCHEDULING_SEED[scheduling_seed]
    except KeyError as error:
        raise ValueError(f"unregistered scheduling seed: {scheduling_seed}") from error
    result = analysis["h2_predecode_predictability"]["test_by_dataset"][dataset][str(predictor_seed)]
    cutoff = result["minimum_safe_precision_classifier"]["uncertainty_cutoff_for_90pct_calibration_coverage"]
    return {item["request_id"]: item for item in result["predictions"]}, float(cutoff)


def make_requests(
    manifest: dict[str, Any],
    tensors: dict[str, torch.Tensor],
    partition: str,
    predictions: dict[str, dict[str, Any]] | None = None,
    cutoff: float = 0.0,
) -> list[ReplayRequest]:
    result = []
    for item in manifest["requests"]:
        if item["partition"] != partition:
            continue
        prediction = predictions.get(item["request_id"]) if predictions else None
        result.append(ReplayRequest(
            request_id=item["request_id"],
            dataset=item["dataset"],
            document_id=item["document_id"],
            prompt_length=item["prompt_length_tokens"],
            continuation_length=item["continuation_length_tokens"],
            prompt_ids=tensors[item["request_id"]][:item["prompt_length_tokens"]].clone(),
            predicted_profile=tuple(prediction["predicted_group_profile"]) if prediction else (),
            classification_confidence=float(prediction["classification_confidence"]) if prediction else 1.0,
            uncertainty_cutoff=cutoff,
        ))
    return result


def calibrate_dataset(model, requests: list[ReplayRequest], pad_token_id: int, device: torch.device, warmups: int, repeats: int) -> dict[str, Any]:
    by_cell = {cell: [request for request in requests if request.cell == cell] for cell in sorted({request.cell for request in requests})}
    cells = {}
    total_service_ms = 0.0
    total_requests = 0
    for cell, values in by_cell.items():
        batches = [values[start:start + MAX_BATCH_SIZE] for start in range(0, len(values), MAX_BATCH_SIZE)]
        for _ in range(warmups):
            execute_batch(model, batches[0], "mlp_multibit_dp_guard", pad_token_id, device)
        measurements = []
        for repeat in range(repeats):
            for batch in batches:
                execution = execute_batch(model, batch, "mlp_multibit_dp_guard", pad_token_id, device)
                measurements.append({"repeat": repeat, "request_ids": [r.request_id for r in batch], **execution})
                total_service_ms += execution["service_ms"]
                total_requests += len(batch)
        service_per_request = [measurement["service_ms"] for measurement in measurements for _ in measurement["request_ids"]]
        key = f"{cell[0]}p:{cell[1]}c"
        cells[key] = {
            "measurements": measurements,
            "p95_service_ms": percentile(service_per_request, 0.95),
            "deadline_ms": 2.0 * percentile(service_per_request, 0.95),
        }
    return {
        "request_count_per_repeat": len(requests),
        "warmup_batches_per_cell": warmups,
        "measured_repeats": repeats,
        "saturated_request_rate": total_requests / (total_service_ms / 1000.0),
        "cells": cells,
    }


def run_registered_warmups(
    model,
    manifests: dict[str, dict[str, Any]],
    tensors: dict[str, dict[str, torch.Tensor]],
    analysis: dict[str, Any],
    output_dir: Path,
    replay_id: str,
    pad_token_id: int,
    device: torch.device,
    warmup_batches: int,
) -> None:
    for dataset, manifest in manifests.items():
        predictions, cutoff = predictor_map(analysis, dataset, 101)
        requests = make_requests(manifest, tensors[dataset], "test", predictions, cutoff)
        for policy in POLICIES:
            for cell in sorted({request.cell for request in requests}):
                values = sorted(
                    [request for request in requests if request.cell == cell],
                    key=lambda request: request.request_id,
                )
                lane_groups = [("ordinary", values)]
                if policy == "predicted_block_fallback_lane":
                    lane_groups = [
                        (lane, [request for request in values if request.uncertain == uncertain])
                        for lane, uncertain in (("predicted", False), ("fallback", True))
                    ]
                    lane_groups = [(lane, group) for lane, group in lane_groups if group]
                for lane, lane_values in lane_groups:
                    marker = output_dir / "warmup" / dataset / f"{policy}-{lane}-{cell[0]}p-{cell[1]}c.json"
                    if marker.exists():
                        stored = json.loads(marker.read_text())
                        if stored.get("replay_id") != replay_id or stored.get("warmup_batches") != warmup_batches:
                            raise RuntimeError(f"Warmup marker mismatch: {marker}")
                        continue
                    first = lane_values[0]
                    batch = [request for request in lane_values if compatible(first, request, policy)][:MAX_BATCH_SIZE]
                    mode = "fixed_high" if policy == "predicted_block_fallback_lane" and first.uncertain else "mlp_multibit_dp_guard"
                    for _ in range(warmup_batches):
                        execute_batch(model, batch, mode, pad_token_id, device)
                    atomic_write_json(marker, {
                        "validation_status": "REAL_GPU_WARMUP_COMPLETE",
                        "replay_id": replay_id,
                        "dataset": dataset,
                        "policy": policy,
                        "lane": lane,
                        "length_cell": list(cell),
                        "mode": mode,
                        "batch_size": len(batch),
                        "warmup_batches": warmup_batches,
                        "contains_raw_text": False,
                    })
                    print(f"warmup {dataset} {policy} {lane} {cell}: complete", flush=True)


def build_run_manifest(args, freeze, frozen_run, analysis, route_summary, device) -> dict[str, Any]:
    inputs = {
        "ap_model": file_manifest(args.ap_model_path),
        "router_checkpoint": file_manifest(args.router_checkpoint),
        "estimator_results": file_manifest(args.estimator_results),
        "tokenizer": tokenizer_file_manifest(args.tokenizer_path),
    }
    stable = {
        "run_schema": RUN_SCHEMA,
        "parent_collection_id": frozen_run["collection_id"],
        "parent_collection_tree_sha256": freeze["collection_tree_sha256"],
        "analysis_sha256": file_sha256(args.analysis_json),
        "route_safety_supplement_id": route_summary["supplement_id"],
        "input_hashes": {name: value.get("tree_sha256", value.get("files", [{}])[0].get("sha256")) for name, value in inputs.items()},
        "source_files_sha256": source_hashes(),
        "policies": list(POLICIES),
        "load_fractions": list(LOAD_FRACTIONS),
        "scheduling_seeds": list(SCHEDULING_SEEDS),
        "max_batch_size": MAX_BATCH_SIZE,
        "max_wait_ms": MAX_WAIT_MS,
        "profile_distance": PROFILE_DISTANCE,
        "warmup_batches": args.warmup_batches,
        "repeat": args.repeat,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "candidate_bits": args.bits,
        "confidence_threshold": args.confidence_threshold,
        "fallback_bits": args.fallback_bits,
        "torch_dtype": args.torch_dtype,
        "max_scenarios": args.max_scenarios,
        "diagnostic_skip_warmups": args.diagnostic_skip_warmups,
        "diagnostic_request_limit": args.diagnostic_request_limit,
        "determinism": {
            "global_seed": 0,
            "torch_deterministic_algorithms": "warn_only",
            "tf32": False,
            "cublas_workspace_config": ":4096:8",
            "greedy_decode": True,
        },
    }
    return {
        "run_schema_version": RUN_SCHEMA,
        "validation_status": "PREFLIGHT_VALIDATED",
        "replay_id": object_sha256(stable),
        "config": stable,
        "input_artifacts": inputs,
        "environment": environment_metadata(device),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip(),
        "git_worktree_dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()),
        "contains_raw_text": False,
    }


def ensure_run_manifest(path: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("replay_id") != candidate["replay_id"]:
            raise RuntimeError("Existing replay has a different frozen configuration")
        return existing
    atomic_write_json(path, candidate)
    return candidate


def walk_keys(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_keys(child)


def validate_scenario(path: Path, replay_id: str) -> dict[str, Any]:
    result = json.loads(path.read_text())
    if result.get("replay_id") != replay_id or result.get("scenario_schema_version") != SCENARIO_SCHEMA:
        raise RuntimeError(f"Scenario identity mismatch: {path}")
    if result.get("contains_raw_text") is not False or not result.get("requests"):
        raise RuntimeError(f"Invalid scenario artifact: {path}")
    if any(key in FORBIDDEN_ARTIFACT_KEYS for key in walk_keys(result)):
        raise RuntimeError(f"Raw text/token payload found in scenario: {path}")
    return result


def refresh_summary(output_dir: Path, run_manifest: dict[str, Any], expected_count: int, route_summary: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    scenario_paths = sorted((output_dir / "scenarios").glob("*.json")) if (output_dir / "scenarios").exists() else []
    scenarios = [validate_scenario(path, run_manifest["replay_id"]) for path in scenario_paths]
    complete = len(scenarios) == expected_count and run_manifest["config"]["max_scenarios"] is None
    summary = {
        "summary_schema_version": SUMMARY_SCHEMA,
        "validation_status": "REAL_GPU_ONLINE_REPLAY_COMPLETE" if complete else "REAL_GPU_ONLINE_REPLAY_IN_PROGRESS",
        "replay_id": run_manifest["replay_id"],
        "expected_scenario_count": expected_count,
        "validated_scenario_count": len(scenarios),
        "h4_status": (
            "FAIL_PREREGISTERED_GUARDED_QUALITY_GATE" if analysis["h3_guarded_mlp"]["overall_status"].startswith("FAIL_")
            else "PENDING_PERFORMANCE_ANALYSIS" if not complete
            else "PERFORMANCE_ANALYSIS_REQUIRED"
        ),
        "route_safety_status": route_summary["validation_status"],
        "scenario_summaries": [
            {"scenario_id": item["scenario_id"], "policy": item["policy"], "summary": item["summary"]}
            for item in scenarios
        ],
        "contains_raw_text": False,
    }
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    collection_dir = Path(args.collection_dir).resolve()
    route_dir = Path(args.route_safety_dir).resolve()
    output_dir = Path(args.output_dir)
    validate_external_output(output_dir, collection_dir, route_dir)
    freeze = verify_freeze(collection_dir, Path(args.freeze_manifest).resolve())
    frozen_run = json.loads((collection_dir / "run-manifest.json").read_text())
    analysis = json.loads(Path(args.analysis_json).read_text())
    route_summary = json.loads((route_dir / "combined-summary.json").read_text())
    if route_summary.get("validation_status") != "REAL_GPU_ROUTE_SAFETY_COMPLETE":
        raise RuntimeError("Route-safety supplement must be complete before H4 replay")
    if analysis.get("collection_tree_sha256") != freeze["collection_tree_sha256"]:
        raise RuntimeError("Predictor analysis does not match frozen collection")
    device = None if args.validate_only else torch.device(args.device)
    if not args.validate_only:
        if device.type != "cuda" or not torch.cuda.is_available() or not os.environ.get("CUDA_VISIBLE_DEVICES"):
            raise RuntimeError("Real replay requires explicit CUDA_VISIBLE_DEVICES")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=args.local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    manifests = {}
    tensors = {}
    for dataset in frozen_run["config"]["datasets"]:
        manifest, values = build_dataset_manifest(dataset, tokenizer)
        frozen_manifest = json.loads((collection_dir / "manifests" / f"{dataset}.json").read_text())
        if manifest != frozen_manifest:
            raise RuntimeError(f"Rebuilt manifest differs: {dataset}")
        manifests[dataset] = manifest
        tensors[dataset] = values

    candidate = build_run_manifest(args, freeze, frozen_run, analysis, route_summary, device)
    run_manifest = ensure_run_manifest(output_dir / "run-manifest.json", candidate)
    expected_count = len(manifests) * len(LOAD_FRACTIONS) * len(SCHEDULING_SEEDS) * args.repeat * len(POLICIES)
    if args.validate_only:
        print(json.dumps(refresh_summary(output_dir, run_manifest, expected_count, route_summary, analysis), indent=2))
        return

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    model = QAQDPLLMForCausalLM.from_quantized(
        args.ap_model_path,
        router_checkpoint=args.router_checkpoint,
        estimator_results=args.estimator_results,
        precisions=args.bits,
        torch_dtype={"float16": torch.float16, "bfloat16": torch.bfloat16}[args.torch_dtype],
        router_mode="mlp_multibit_dp_guard",
        confidence_threshold=args.confidence_threshold,
        fallback_bits=args.fallback_bits,
        prefill_by_router=True,
        batch_policy="group",
        trust_remote_code=True,
    ).eval().to(device)

    calibrations = {}
    for dataset, manifest in manifests.items():
        path = output_dir / "calibration" / f"{dataset}.json"
        if path.exists():
            calibrations[dataset] = json.loads(path.read_text())
        else:
            requests = make_requests(manifest, tensors[dataset], "calibration")
            if args.diagnostic_request_limit is not None:
                requests = requests[:args.diagnostic_request_limit]
            result = calibrate_dataset(model, requests, tokenizer.pad_token_id, device, args.warmup_batches, args.repeat)
            result.update({"dataset": dataset, "replay_id": run_manifest["replay_id"], "contains_raw_text": False})
            atomic_write_json(path, result)
            calibrations[dataset] = result

    if not args.diagnostic_skip_warmups:
        run_registered_warmups(
            model, manifests, tensors, analysis, output_dir, run_manifest["replay_id"],
            tokenizer.pad_token_id, device, args.warmup_batches,
        )

    scenario_dir = output_dir / "scenarios"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    scenario_counter = 0
    stop = False
    for dataset_index, dataset in enumerate(sorted(manifests)):
        calibration = calibrations[dataset]
        deadlines = {cell: values["deadline_ms"] for cell, values in calibration["cells"].items()}
        for load_index, load in enumerate(LOAD_FRACTIONS):
            arrival_rate = load * calibration["saturated_request_rate"]
            for seed_index, seed in enumerate(SCHEDULING_SEEDS):
                predictions, cutoff = predictor_map(analysis, dataset, seed)
                base = make_requests(manifests[dataset], tensors[dataset], "test", predictions, cutoff)
                if args.diagnostic_request_limit is not None:
                    base = base[:args.diagnostic_request_limit]
                arrivals = deterministic_arrivals(base, dataset, seed, arrival_rate)
                for repeat_index in range(args.repeat):
                    rotation = (dataset_index + load_index + seed_index + repeat_index) % len(POLICIES)
                    policy_order = list(POLICIES[rotation:]) + list(POLICIES[:rotation])
                    for policy in policy_order:
                        scenario_id = f"{dataset}-load{int(load*100):02d}-seed{seed}-repeat{repeat_index}-{policy}"
                        path = scenario_dir / f"{scenario_id}.json"
                        if path.exists():
                            validate_scenario(path, run_manifest["replay_id"])
                            print(f"{scenario_id}: validated existing", flush=True)
                        else:
                            result = run_online_scenario(model, arrivals, policy, deadlines, tokenizer.pad_token_id, device, scenario_id)
                            result.update({
                                "replay_id": run_manifest["replay_id"],
                                "dataset": dataset,
                                "load_fraction": load,
                                "arrival_rate_requests_per_s": arrival_rate,
                                "scheduling_seed": seed,
                                "repeat_index": repeat_index,
                            })
                            temporary = path.with_suffix(".json.tmp")
                            temporary.write_text(json.dumps(result, sort_keys=True, ensure_ascii=True) + "\n", encoding="ascii")
                            validate_scenario(temporary, run_manifest["replay_id"])
                            os.replace(temporary, path)
                            print(f"{scenario_id}: committed", flush=True)
                        scenario_counter += 1
                        refresh_summary(output_dir, run_manifest, expected_count, route_summary, analysis)
                        if args.max_scenarios is not None and scenario_counter >= args.max_scenarios:
                            stop = True
                            break
                    if stop:
                        break
                if stop:
                    break
            if stop:
                break
        if stop:
            break
    verify_freeze(collection_dir, Path(args.freeze_manifest).resolve())
    print(json.dumps(refresh_summary(output_dir, run_manifest, expected_count, route_summary, analysis), indent=2))


if __name__ == "__main__":
    main()
