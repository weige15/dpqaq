"""Profile QAQ router phases on one real native CUDA request batch."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from any_precision import QAQDPLLMForCausalLM, load_qaq_router_checkpoint
from scripts.benchmark_qaq_profile_batching import (
    PROFILE_MODE,
    build_batch_plan,
    make_arrival_trace,
    make_requests,
    policy_mode,
    request_stream_hash,
    set_execution_policy,
)
from scripts.run_qaq_online_scheduler_replay import execute_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection_dir", required=True)
    parser.add_argument("--analysis_json", required=True)
    parser.add_argument("--ap_model_path", required=True)
    parser.add_argument("--router_checkpoint", required=True)
    parser.add_argument("--estimator_results", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--datasets", nargs="+", default=["wikitext2", "c4_new"])
    parser.add_argument("--request_limit", type=int, default=0)
    parser.add_argument("--min_uncertain_requests", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--arrival_rate", type=float, default=50.0)
    parser.add_argument("--arrival_seed", type=int, default=101)
    parser.add_argument("--predictor_seed", type=int, default=17)
    parser.add_argument(
        "--profile_policy",
        choices=("fcfs", "predicted_profile", "uncertainty_fallback"),
        default="predicted_profile",
    )
    parser.add_argument("--max_batch_size", type=int, default=8)
    parser.add_argument("--max_wait_ms", type=float, default=50.0)
    parser.add_argument("--scalar_bucket_size", type=float, default=0.25)
    parser.add_argument("--profile_distance", type=float, default=0.25)
    parser.add_argument("--confidence_threshold", type=float, default=0.6)
    parser.add_argument("--fallback_bits", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


def profiler_phase_rows(profiler: torch.profiler.profile) -> list[dict[str, object]]:
    rows = []
    for event in profiler.key_averages():
        if not event.key.startswith("qaq."):
            continue
        rows.append({
            "name": event.key,
            "count": int(event.count),
            "cpu_total_ms": float(event.cpu_time_total) / 1000.0,
            "cpu_self_ms": float(event.self_cpu_time_total) / 1000.0,
            "cuda_total_ms": float(event.device_time_total) / 1000.0,
            "cuda_self_ms": float(event.self_device_time_total) / 1000.0,
        })
    return sorted(rows, key=lambda row: row["cuda_total_ms"], reverse=True)


def main() -> None:
    args = parse_args()
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES explicitly")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This profiler requires CUDA")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        local_files_only=args.local_files_only,
    )
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

    class BatchArgs:
        max_batch_size = args.max_batch_size
        max_wait_ms = args.max_wait_ms
        scalar_bucket_size = args.scalar_bucket_size
        profile_distance = args.profile_distance

    plan = build_batch_plan(requests, BatchArgs, args.profile_policy)
    max_continuation = max(request.continuation_length for request in requests)
    candidates = [
        batch for batch in plan
        if batch[0].continuation_length == max_continuation
    ]
    if not candidates:
        raise RuntimeError("No native 128-token batch found")
    batch = max(candidates, key=len)

    router, checkpoint = load_qaq_router_checkpoint(args.router_checkpoint)
    model = QAQDPLLMForCausalLM.from_quantized(
        args.ap_model_path,
        router=router,
        router_metadata=checkpoint,
        estimator_results=args.estimator_results,
        precisions=[3, 4, 5, 6],
        torch_dtype=torch.float16,
        router_mode=PROFILE_MODE,
        confidence_threshold=args.confidence_threshold,
        fallback_bits=args.fallback_bits,
        prefill_by_router=True,
        batch_policy="max",
    ).eval().to(device)
    set_execution_policy(model, args.profile_policy, len(batch))
    mode = policy_mode(args.profile_policy, batch)
    model.set_phase_timing_enabled(True)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(output_dir / "trace")),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        execute_batch(model, batch, mode, tokenizer.pad_token_id, device)
        profiler.step()
        execute_batch(model, batch, mode, tokenizer.pad_token_id, device)
        profiler.step()

    stats = model.get_router_stats()
    phase_rows = profiler_phase_rows(profiler)
    result = {
        "schema_version": "qaq_phase_profile_v1",
        "status": "REAL_CUDA_TORCH_PROFILER",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
        "config": vars(args),
        "request_stream": {
            "request_count": len(requests),
            "stream_sha256": request_stream_hash(requests),
            "uncertain_request_count": sum(request.uncertain for request in requests),
        },
        "profiled_batch": {
            "request_ids": [request.request_id for request in batch],
            "batch_size": len(batch),
            "prompt_lengths": [request.prompt_length for request in batch],
            "continuation_length": max_continuation,
            "policy": args.profile_policy,
            "mode": mode,
        },
        "qa_phase_ranges": phase_rows,
        "module_phase_timing": stats.get("phase_timing", {}),
        "router_stats": stats,
        "trace_directory": str(output_dir / "trace"),
    }
    (output_dir / "profile.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "profile": str(output_dir / "profile.json"),
        "trace": str(output_dir / "trace"),
        "phases": phase_rows,
    }, indent=2))


if __name__ == "__main__":
    main()
