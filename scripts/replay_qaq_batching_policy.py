import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoTokenizer

from any_precision import QAQDPLLMForCausalLM


REPLAY_SCHEMA_VERSION = "qaq_dynamic_batching_gpu_replay_v1"
UNVALIDATED = "UNVALIDATED"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay simulator-selected QAQ batches through real batched generation. "
            "This validates real grouped-batch execution only; it does not implement "
            "max/quantile shared-profile overrides."
        )
    )
    parser.add_argument("--ap_model_path", default=os.environ.get("AP_MODEL_PATH"), help="Any-Precision model path.")
    parser.add_argument("--router_checkpoint", required=True)
    parser.add_argument("--estimator_results", required=True)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--prompt_file", required=True)
    parser.add_argument("--simulation_json", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--bits", type=int, nargs="+", default=[3, 4, 5, 6])
    parser.add_argument("--router_mode", default="mlp_multibit_dp_guard")
    parser.add_argument("--batch_policy", default="group", choices=["group", "max"])
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--warmup_batches", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--confidence_threshold", type=float, default=None)
    parser.add_argument("--fallback_bits", type=int, default=1)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no_trust_remote_code", action="store_false", dest="trust_remote_code")
    return parser.parse_args()


def require_path(value: str | None, name: str) -> str:
    if not value:
        raise ValueError(f"{name} is required")
    return value


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def git_dirty_status() -> list[str]:
    try:
        return subprocess.check_output(["git", "status", "--short"], text=True).splitlines()
    except Exception:
        return []


def load_prompt_map(path: str | Path) -> dict[str, str]:
    prompts: dict[str, str] = {}
    with Path(path).open() as f:
        for index, line in enumerate(f):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("{"):
                raw = json.loads(stripped)
            else:
                raw = {"request_id": f"req_{index:06d}", "prompt": stripped}
            request_id = str(raw.get("request_id", f"req_{index:06d}"))
            prompt = raw.get("prompt", raw.get("text"))
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(f"Prompt record {index} is missing prompt/text")
            if request_id in prompts:
                raise ValueError(f"Duplicate request_id in prompt file: {request_id}")
            prompts[request_id] = prompt
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def load_policy_batches(simulation_path: str | Path, policy: str, max_batches: int | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    simulation = json.loads(Path(simulation_path).read_text())
    policies = simulation.get("policies", {})
    if policy not in policies:
        available = ", ".join(sorted(policies))
        raise ValueError(f"Policy {policy!r} not found in simulation. Available: {available}")
    batches = list(policies[policy].get("batches", []))
    if max_batches is not None:
        batches = batches[:max_batches]
    if not batches:
        raise ValueError(f"Policy {policy!r} has no batches to replay")
    return simulation, batches


def validate_prompts_for_batches(prompts: dict[str, str], batches: list[dict[str, Any]]) -> None:
    missing = []
    for batch in batches:
        for request_id in batch.get("request_ids", []):
            if request_id not in prompts:
                missing.append(request_id)
    if missing:
        preview = ", ".join(missing[:8])
        raise ValueError(f"Prompt file is missing {len(missing)} request IDs used by simulation: {preview}")


def move_encoding_to_device(encoded, device: str):
    if hasattr(encoded, "to"):
        return encoded.to(device)
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in encoded.items()}


def synchronize_if_cuda(device: str) -> None:
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def clear_model_stats(model) -> None:
    if hasattr(model, "clear_router_stats"):
        model.clear_router_stats()
    elif hasattr(model, "clear_comp_count"):
        model.clear_comp_count()


def collect_model_stats(model) -> dict[str, Any]:
    if not hasattr(model, "get_router_stats"):
        return {}
    return model.get_router_stats()


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"values": [], "mean": None, "min": None, "max": None, "p50": None, "p95": None, "p99": None}
    ordered = sorted(values)

    def pct(q: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        pos = (len(ordered) - 1) * q
        lo = int(pos)
        hi = min(len(ordered) - 1, lo + 1)
        if lo == hi:
            return ordered[lo]
        frac = pos - lo
        return ordered[lo] * (1.0 - frac) + ordered[hi] * frac

    return {
        "values": values,
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "p50": pct(0.50),
        "p95": pct(0.95),
        "p99": pct(0.99),
    }


def cuda_memory_snapshot(device: str) -> dict[str, int]:
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        return {}
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def aggregate_replay_batches(batch_results: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(batch["gpu_execution_ms"]) for batch in batch_results]
    token_slots = [float(batch["generated_token_slots"]) for batch in batch_results]
    total_tokens = sum(token_slots)
    total_latency_s = sum(latencies) / 1000.0
    avg_bits = [
        float(batch["router_stats"]["average_selected_bit"])
        for batch in batch_results
        if batch.get("router_stats", {}).get("average_selected_bit") is not None
    ]
    effective_bits = [
        float(batch["router_stats"]["effective_bits"])
        for batch in batch_results
        if batch.get("router_stats", {}).get("effective_bits") is not None
    ]
    fallbacks = sum(int(batch.get("router_stats", {}).get("total_fallbacks", 0)) for batch in batch_results)
    dp_guard = sum(int(batch.get("router_stats", {}).get("total_dp_guard_triggers", 0)) for batch in batch_results)
    return {
        "batch_count": len(batch_results),
        "request_count": sum(int(batch["batch_size"]) for batch in batch_results),
        "gpu_execution_ms": summarize(latencies),
        "generated_token_slots": summarize(token_slots),
        "tokens_per_s": total_tokens / total_latency_s if total_latency_s > 0 else 0.0,
        "average_selected_bit": summarize(avg_bits) if avg_bits else {},
        "effective_bits": summarize(effective_bits) if effective_bits else {},
        "total_fallbacks": int(fallbacks),
        "total_dp_guard_triggers": int(dp_guard),
    }


@torch.no_grad()
def replay_one_batch(
    model,
    tokenizer,
    prompts: dict[str, str],
    batch: dict[str, Any],
    args: argparse.Namespace,
    repeat_index: int | str,
) -> dict[str, Any]:
    request_ids = [str(request_id) for request_id in batch["request_ids"]]
    prompt_texts = [prompts[request_id] for request_id in request_ids]
    encoded = move_encoding_to_device(tokenizer(prompt_texts, return_tensors="pt", padding=True), args.device)
    clear_model_stats(model)
    kwargs = {
        **encoded,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "router_mode": args.router_mode,
    }

    synchronize_if_cuda(args.device)
    start = time.perf_counter()
    generated = model.generate(**kwargs)
    synchronize_if_cuda(args.device)
    gpu_execution_ms = 1000.0 * (time.perf_counter() - start)
    generated_token_slots = int(generated.numel() - encoded["input_ids"].numel())
    router_stats = collect_model_stats(model)

    return {
        "repeat_index": repeat_index,
        "batch_id": batch["batch_id"],
        "lane_id": batch.get("lane_id", UNVALIDATED),
        "request_ids": request_ids,
        "batch_size": len(request_ids),
        "gpu_execution_ms": gpu_execution_ms,
        "generated_token_slots": generated_token_slots,
        "tokens_per_s": generated_token_slots / (gpu_execution_ms / 1000.0) if gpu_execution_ms > 0 else 0.0,
        "router_stats": router_stats,
    }


def load_model(args: argparse.Namespace):
    model = QAQDPLLMForCausalLM.from_quantized(
        args.ap_model_path,
        router_checkpoint=args.router_checkpoint,
        estimator_results=args.estimator_results,
        precisions=args.bits,
        router_mode=args.router_mode,
        confidence_threshold=args.confidence_threshold,
        fallback_bits=args.fallback_bits,
        batch_policy=args.batch_policy,
        trust_remote_code=args.trust_remote_code,
    )
    return model.eval().to(args.device)


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    args.ap_model_path = require_path(args.ap_model_path, "--ap_model_path")
    if torch.device(args.device).type == "cuda" and "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES explicitly before running CUDA replay.")

    prompts = load_prompt_map(args.prompt_file)
    simulation, batches = load_policy_batches(args.simulation_json, args.policy, max_batches=args.max_batches)
    validate_prompts_for_batches(prompts, batches)

    tokenizer_path = args.tokenizer_path or args.ap_model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = load_model(args)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    warmup_results = []
    for warmup_index, batch in enumerate(batches[: args.warmup_batches]):
        warmup_results.append(replay_one_batch(model, tokenizer, prompts, batch, args, repeat_index=f"warmup_{warmup_index}"))

    repeats = []
    for repeat_index in range(args.repeat):
        print(f"Replay {args.policy} repeat {repeat_index}...", flush=True)
        batch_results = []
        for batch_index, batch in enumerate(batches):
            result = replay_one_batch(model, tokenizer, prompts, batch, args, repeat_index=repeat_index)
            batch_results.append(result)
            print(
                f"{args.policy} repeat={repeat_index} batch={batch_index + 1}/{len(batches)} "
                f"size={result['batch_size']} latency={result['gpu_execution_ms']:.2f}ms",
                flush=True,
            )
        repeats.append({
            "repeat_index": repeat_index,
            "summary": aggregate_replay_batches(batch_results),
            "batches": batch_results,
        })

    aggregate = aggregate_replay_batches([
        batch
        for repeat in repeats
        for batch in repeat["batches"]
    ])

    return {
        "replay_schema_version": REPLAY_SCHEMA_VERSION,
        "status": "REAL_GPU_BATCHED_REPLAY_NO_SHARED_PROFILE_OVERRIDES",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "git_dirty_status": git_dirty_status(),
        "config": {
            "ap_model_path": args.ap_model_path,
            "router_checkpoint": args.router_checkpoint,
            "estimator_results": args.estimator_results,
            "prompt_file": args.prompt_file,
            "simulation_json": args.simulation_json,
            "policy": args.policy,
            "bits": args.bits,
            "router_mode": args.router_mode,
            "batch_policy": args.batch_policy,
            "max_new_tokens": args.max_new_tokens,
            "max_batches": args.max_batches,
            "warmup_batches": args.warmup_batches,
            "repeat": args.repeat,
            "device": args.device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "simulation_policy_summary": {
            key: simulation["policies"][args.policy].get(key)
            for key in [
                "request_count",
                "batch_count",
                "mean_batch_size",
                "requests_per_s",
                "lane_occupancy",
                "under_precision_rate",
                "over_precision_rate",
            ]
        },
        "limitations": [
            "This replay uses real batched model.generate calls and CUDA synchronization.",
            "It replays simulator batch membership, but it does not reproduce simulator queue delay.",
            "It does not implement max_profile_sharing or quantile_profile_sharing shared precision overrides.",
            "Generated token count is reported as returned token slots for padded batches.",
        ],
        "warmup": warmup_results,
        "repeats": repeats,
        "aggregate": aggregate,
        "cuda_memory": cuda_memory_snapshot(args.device),
    }


def main() -> None:
    args = parse_args()
    result = run_replay(args)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    aggregate = result["aggregate"]
    print(
        f"Wrote {output_path}; policy={args.policy} "
        f"batches={aggregate['batch_count']} requests={aggregate['request_count']} "
        f"mean_batch_gpu_ms={aggregate['gpu_execution_ms']['mean']:.2f} "
        f"tokens_per_s={aggregate['tokens_per_s']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
