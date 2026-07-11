"""Collect held-out request-level QAQ precision demand and prefill features."""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import accelerate
import datasets
import torch
import transformers
from tqdm import tqdm
from transformers import AutoTokenizer

from any_precision import QAQDPLLMForCausalLM, load_qaq_router_checkpoint
from scripts.evaluate_qaq_heldout import (
    build_token_windows,
    load_heldout_text,
    safe_perplexity,
    token_sha256,
)


SCHEMA_VERSION = "qaq_request_demand_v1"
DEFAULT_QAQ_MODES = ("dp_threshold_only", "mlp_multibit", "mlp_multibit_dp_guard")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a held-out prompt/continuation dataset with fixed-bit quality demand, "
            "observed QAQ profiles, and pre-decode prompt features."
        )
    )
    parser.add_argument("--ap_model_path", required=True)
    parser.add_argument("--router_checkpoint", required=True)
    parser.add_argument("--estimator_results", required=True)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--dataset", choices=["wikitext2", "c4_new"], default="wikitext2")
    parser.add_argument("--dataset_start", type=int, default=0)
    parser.add_argument("--num_requests", type=int, default=32)
    parser.add_argument("--prompt_length", type=int, default=128)
    parser.add_argument("--continuation_length", type=int, default=64)
    parser.add_argument("--bits", type=int, nargs="+", default=None)
    parser.add_argument("--qaq_modes", nargs="+", choices=list(DEFAULT_QAQ_MODES), default=list(DEFAULT_QAQ_MODES))
    parser.add_argument("--safe_nll_delta", type=float, default=0.02)
    parser.add_argument("--profile_layer_group_size", type=int, default=4)
    parser.add_argument("--confidence_threshold", type=float, default=0.6)
    parser.add_argument("--fallback_bits", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch_dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no_trust_remote_code", action="store_false", dest="trust_remote_code")
    return parser.parse_args()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "UNAVAILABLE"


def source_provenance() -> dict[str, Any]:
    paths = (
        Path("scripts/build_qaq_request_demand_dataset.py"),
        Path("any_precision/modules/QAQDPLLM_Linear.py"),
        Path("any_precision/modules/QAQDPLLMForCausalLM.py"),
    )
    hashes = {
        str(path): hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest()
        for path in paths
    }
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True,
        capture_output=True, check=False,
    )
    return {"git_worktree_dirty": bool(status.stdout.strip()), "source_files_sha256": hashes}


def build_request_windows(
    input_ids: torch.Tensor,
    prompt_length: int,
    continuation_length: int,
    start: int,
    count: int,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if prompt_length < 2 or continuation_length < 1:
        raise ValueError("prompt_length must be >= 2 and continuation_length must be positive")
    windows = build_token_windows(
        input_ids,
        context_length=prompt_length + continuation_length,
        start=start,
        count=count,
    )
    return [
        (window[:prompt_length], window[prompt_length:], window)
        for window in windows
    ]


def token_entropy(token_ids: torch.Tensor) -> float:
    counts = Counter(int(token) for token in token_ids.reshape(-1).tolist())
    total = sum(counts.values())
    return -sum((count / total) * math.log(count / total) for count in counts.values())


def prompt_token_features(prompt_ids: torch.Tensor, tokenizer) -> dict[str, float]:
    flat = prompt_ids.reshape(-1).to(dtype=torch.float64)
    text = tokenizer.decode(prompt_ids.tolist(), skip_special_tokens=False)
    char_count = max(len(text), 1)
    return {
        "prompt_length_tokens": float(flat.numel()),
        "unique_token_fraction": float(torch.unique(flat).numel() / flat.numel()),
        "token_id_mean": float(flat.mean().item()),
        "token_id_std": float(flat.std(unbiased=False).item()),
        "token_id_min": float(flat.min().item()),
        "token_id_max": float(flat.max().item()),
        "token_entropy": float(token_entropy(prompt_ids)),
        "character_count": float(len(text)),
        "whitespace_fraction": sum(char.isspace() for char in text) / char_count,
        "digit_fraction": sum(char.isdigit() for char in text) / char_count,
        "alpha_fraction": sum(char.isalpha() for char in text) / char_count,
        "punctuation_fraction": sum(not char.isalnum() and not char.isspace() for char in text) / char_count,
        "line_count": float(text.count("\n") + 1),
    }


@torch.no_grad()
def prompt_model_features(model, prompt_ids: torch.Tensor, device: torch.device) -> dict[str, float]:
    encoded = prompt_ids.unsqueeze(0).to(device)
    model.clear_router_stats()
    outputs = model(
        input_ids=encoded,
        labels=encoded,
        use_cache=False,
        output_hidden_states=True,
        router_mode="fixed_high",
    )
    logits = outputs.logits[:, -1, :].float()
    if not torch.isfinite(logits).all() or not torch.isfinite(outputs.loss):
        raise RuntimeError("Non-finite fixed-high prompt feature output")
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    top_probs = torch.topk(probs, k=2, dim=-1).values
    hidden = outputs.hidden_states[-1].float()
    norms = hidden.norm(dim=-1)
    model.clear_router_stats()
    return {
        "fixed_high_prompt_nll": float(outputs.loss.float().item()),
        "last_token_entropy": float((-(probs * log_probs).sum(dim=-1)).item()),
        "last_token_top1_probability": float(top_probs[0, 0].item()),
        "last_token_probability_margin": float((top_probs[0, 0] - top_probs[0, 1]).item()),
        "final_hidden_norm_mean": float(norms.mean().item()),
        "final_hidden_norm_std": float(norms.std(unbiased=False).item()),
        "final_hidden_norm_last": float(norms[0, -1].item()),
        "final_hidden_abs_mean": float(hidden.abs().mean().item()),
    }


def expected_bit(bit_counts: dict[str, int]) -> float:
    total = sum(int(count) for count in bit_counts.values())
    return (
        sum(int(bit) * int(count) for bit, count in bit_counts.items()) / total
        if total else 0.0
    )


def majority_bit(bit_counts: dict[str, int]) -> int:
    if not bit_counts:
        raise ValueError("Cannot compute majority bit from empty counts")
    return max(
        ((int(bit), int(count)) for bit, count in bit_counts.items()),
        key=lambda item: (item[1], item[0]),
    )[0]


def profile_from_stats(stats: dict[str, Any], layer_group_size: int) -> dict[str, Any]:
    route_expected_bits = {}
    route_majority_bits = {}
    route_bit_counts = {}
    grouped = defaultdict(list)
    for route, route_stats in sorted(stats["per_layer"].items()):
        counts = {str(bit): int(count) for bit, count in route_stats["bit_counts"].items()}
        route_bit_counts[route] = counts
        route_expected_bits[route] = expected_bit(counts)
        route_majority_bits[route] = majority_bit(counts)
        layer = int(route.split(".", 1)[0])
        grouped[layer // layer_group_size].append(route_expected_bits[route])
    group_expected_bits = [
        sum(grouped[group]) / len(grouped[group])
        for group in sorted(grouped)
    ]
    return {
        "average_selected_bit": float(stats["average_selected_bit"]),
        "effective_bits": float(stats["effective_bits"]),
        "fallback_count": int(stats["total_fallbacks"]),
        "fallback_fraction": float(stats["fallback_fraction"]),
        "dp_guard_trigger_count": int(stats["total_dp_guard_triggers"]),
        "dp_guard_trigger_fraction": float(stats["dp_guard_trigger_fraction"]),
        "route_expected_bits": route_expected_bits,
        "route_majority_bits": route_majority_bits,
        "route_bit_counts": route_bit_counts,
        "layer_group_size": int(layer_group_size),
        "group_expected_bits": group_expected_bits,
    }


@torch.no_grad()
def evaluate_quality_mode(
    model,
    full_ids: torch.Tensor,
    prompt_length: int,
    device: torch.device,
    router_mode: str,
    precision: int | None = None,
) -> dict[str, Any]:
    input_ids = full_ids.unsqueeze(0).to(device)
    labels = input_ids.clone()
    labels[:, :prompt_length] = -100
    model.clear_router_stats()
    kwargs = {
        "input_ids": input_ids,
        "labels": labels,
        "use_cache": False,
        "router_mode": router_mode,
    }
    if precision is not None:
        kwargs["precision"] = precision
    outputs = model(**kwargs)
    mean_nll = float(outputs.loss.float().item())
    finite_logits = bool(torch.isfinite(outputs.logits).all().item())
    if not math.isfinite(mean_nll) or not finite_logits:
        raise RuntimeError(f"Non-finite quality output for {router_mode}, precision={precision}")
    return {
        "mean_nll": mean_nll,
        "perplexity": safe_perplexity(mean_nll),
        "finite_logits": finite_logits,
        "target_token_count": int(full_ids.numel() - prompt_length),
        "runtime_stats": model.get_router_stats(),
    }


def fixed_mode_specs(bits: list[int]) -> list[tuple[str, str, int | None]]:
    specs = [("fixed_low", "fixed_low", None)]
    specs.extend((f"fixed_{bit}", "fixed_precision", bit) for bit in bits[1:-1])
    specs.append(("fixed_high", "fixed_high", None))
    return specs


def minimum_safe_precision(
    fixed_results: dict[int, dict[str, Any]],
    reference_nll: float,
    safe_nll_delta: float,
) -> dict[str, Any]:
    deltas = {
        int(bit): float(result["mean_nll"] - reference_nll)
        for bit, result in fixed_results.items()
    }
    safe_bits = [bit for bit in sorted(deltas) if deltas[bit] <= safe_nll_delta]
    if not safe_bits:
        raise RuntimeError("No fixed precision met the safe threshold; fixed_high should always qualify")
    selected = min(safe_bits)
    return {
        "safe_nll_delta_threshold": float(safe_nll_delta),
        "requested_bit": int(selected),
        "actual_effective_bits": float(fixed_results[selected]["runtime_stats"]["effective_bits"]),
        "fixed_nll_deltas": {str(bit): delta for bit, delta in deltas.items()},
    }


def compact_quality_result(result: dict[str, Any], reference_nll: float) -> dict[str, Any]:
    stats = result["runtime_stats"]
    return {
        "mean_nll": float(result["mean_nll"]),
        "nll_delta_vs_fixed_high": float(result["mean_nll"] - reference_nll),
        "perplexity": float(result["perplexity"]),
        "finite_logits": bool(result["finite_logits"]),
        "target_token_count": int(result["target_token_count"]),
        "average_selected_bit": float(stats["average_selected_bit"]),
        "effective_bits": float(stats["effective_bits"]),
        "fallback_count": int(stats["total_fallbacks"]),
        "fallback_fraction": float(stats["fallback_fraction"]),
        "dp_guard_trigger_count": int(stats["total_dp_guard_triggers"]),
        "dp_guard_trigger_fraction": float(stats["dp_guard_trigger_fraction"]),
    }


def collect_request(
    model,
    tokenizer,
    request_index: int,
    prompt_ids: torch.Tensor,
    continuation_ids: torch.Tensor,
    full_ids: torch.Tensor,
    bits: list[int],
    qaq_modes: list[str],
    safe_nll_delta: float,
    layer_group_size: int,
    device: torch.device,
) -> dict[str, Any]:
    features = prompt_token_features(prompt_ids, tokenizer)
    features.update(prompt_model_features(model, prompt_ids, device))

    raw_results = {}
    fixed_by_bit = {}
    for name, mode, precision in fixed_mode_specs(bits):
        result = evaluate_quality_mode(model, full_ids, prompt_ids.numel(), device, mode, precision)
        raw_results[name] = result
        requested_bit = bits[0] if name == "fixed_low" else bits[-1] if name == "fixed_high" else int(precision)
        fixed_by_bit[requested_bit] = result

    for mode in qaq_modes:
        raw_results[mode] = evaluate_quality_mode(
            model, full_ids, prompt_ids.numel(), device, mode
        )

    reference_nll = raw_results["fixed_high"]["mean_nll"]
    quality = {
        name: compact_quality_result(result, reference_nll)
        for name, result in raw_results.items()
    }
    profiles = {
        mode: profile_from_stats(raw_results[mode]["runtime_stats"], layer_group_size)
        for mode in qaq_modes
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "request_id": f"request_{request_index:06d}",
        "request_index": int(request_index),
        "prompt_token_sha256": token_sha256(prompt_ids),
        "continuation_token_sha256": token_sha256(continuation_ids),
        "request_token_sha256": token_sha256(full_ids),
        "prompt_length_tokens": int(prompt_ids.numel()),
        "continuation_length_tokens": int(continuation_ids.numel()),
        "prompt_features": features,
        "quality_by_mode": quality,
        "minimum_safe_precision": minimum_safe_precision(
            fixed_by_bit, reference_nll, safe_nll_delta
        ),
        "observed_qaq_profiles": profiles,
    }


def summarize_records(records: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    demand_counts = Counter(
        int(record["minimum_safe_precision"]["requested_bit"]) for record in records
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "validation_status": "REAL_GPU_REQUEST_DEMAND",
        "metadata": metadata,
        "request_count": len(records),
        "minimum_safe_precision_counts": {
            str(bit): int(count) for bit, count in sorted(demand_counts.items())
        },
        "mean_quality_by_mode": {
            mode: {
                "mean_nll": sum(record["quality_by_mode"][mode]["mean_nll"] for record in records) / len(records),
                "mean_nll_delta_vs_fixed_high": sum(
                    record["quality_by_mode"][mode]["nll_delta_vs_fixed_high"] for record in records
                ) / len(records),
                "mean_effective_bits": sum(
                    record["quality_by_mode"][mode]["effective_bits"] for record in records
                ) / len(records),
            }
            for mode in records[0]["quality_by_mode"]
        },
    }


def validate_args(args, checkpoint: dict[str, Any]) -> list[int]:
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Request-demand collection requires a real CUDA device")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES explicitly")
    if args.num_requests < 1 or args.profile_layer_group_size < 1:
        raise ValueError("num_requests and profile_layer_group_size must be positive")
    if args.safe_nll_delta < 0:
        raise ValueError("safe_nll_delta must be non-negative")
    if checkpoint.get("label_mode") != "multibit":
        raise ValueError("Request-demand collection requires a multibit router checkpoint")
    bits = sorted(int(bit) for bit in checkpoint["candidate_bits"])
    if args.bits is not None and sorted(args.bits) != bits:
        raise ValueError(f"--bits {sorted(args.bits)} do not match checkpoint bits {bits}")
    return bits


def main():
    args = parse_args()
    router, checkpoint = load_qaq_router_checkpoint(args.router_checkpoint)
    bits = validate_args(args, checkpoint)
    device = torch.device(args.device)

    tokenizer_path = args.tokenizer_path or args.ap_model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    text, dataset_metadata = load_heldout_text(args.dataset)
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False, verbose=False).input_ids
    requests = build_request_windows(
        encoded,
        args.prompt_length,
        args.continuation_length,
        args.dataset_start,
        args.num_requests,
    )

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.torch_dtype]
    model = QAQDPLLMForCausalLM.from_quantized(
        args.ap_model_path,
        router=router,
        router_metadata=checkpoint,
        estimator_results=args.estimator_results,
        precisions=bits,
        torch_dtype=dtype,
        router_mode="mlp_multibit_dp_guard",
        confidence_threshold=args.confidence_threshold,
        fallback_bits=args.fallback_bits,
        prefill_by_router=True,
        trust_remote_code=args.trust_remote_code,
    ).eval().to(device)

    records = []
    for index, (prompt_ids, continuation_ids, full_ids) in enumerate(
        tqdm(requests, desc="request demand", unit="request")
    ):
        records.append(
            collect_request(
                model=model,
                tokenizer=tokenizer,
                request_index=args.dataset_start + index,
                prompt_ids=prompt_ids,
                continuation_ids=continuation_ids,
                full_ids=full_ids,
                bits=bits,
                qaq_modes=list(args.qaq_modes),
                safe_nll_delta=args.safe_nll_delta,
                layer_group_size=args.profile_layer_group_size,
                device=device,
            )
        )

    subset_hash = hashlib.sha256(
        "".join(record["request_token_sha256"] for record in records).encode("ascii")
    ).hexdigest()
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        **source_provenance(),
        "hostname": os.uname().nodename,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "accelerate": accelerate.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "command": [sys.executable, *sys.argv],
        "ap_model_path": str(Path(args.ap_model_path).resolve()),
        "router_checkpoint": str(Path(args.router_checkpoint).resolve()),
        "estimator_results": str(Path(args.estimator_results).resolve()),
        "tokenizer_path": str(Path(tokenizer_path).resolve()),
        "candidate_bits": bits,
        "qaq_modes": list(args.qaq_modes),
        "safe_nll_delta": args.safe_nll_delta,
        "confidence_threshold": args.confidence_threshold,
        "fallback_bits": args.fallback_bits,
        "prefill_by_router": True,
        "dataset": {
            "name": args.dataset,
            **dataset_metadata,
            "window_policy": "non_overlapping_prompt_plus_continuation",
            "window_start": args.dataset_start,
            "num_requests": args.num_requests,
            "prompt_length": args.prompt_length,
            "continuation_length": args.continuation_length,
            "subset_token_sha256": subset_hash,
            "held_out_from_router_training": True,
            "router_training_dataset": checkpoint.get("training_config", {}).get("dataset"),
            "router_training_split": "train",
        },
    }

    output_path = Path(args.output_jsonl)
    summary_path = Path(args.summary_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    summary_tmp = summary_path.with_suffix(summary_path.suffix + ".tmp")
    with output_tmp.open("w") as out:
        for record in records:
            out.write(json.dumps(record) + "\n")
    with summary_tmp.open("w") as out:
        json.dump(summarize_records(records, metadata), out, indent=2)
    os.replace(output_tmp, output_path)
    os.replace(summary_tmp, summary_path)
    print(json.dumps({
        "validation_status": "REAL_GPU_REQUEST_DEMAND",
        "dataset_jsonl": str(output_path),
        "summary_json": str(summary_path),
        "request_count": len(records),
        "subset_token_sha256": subset_hash,
        "minimum_safe_precision_counts": summarize_records(records, metadata)["minimum_safe_precision_counts"],
    }, indent=2))


if __name__ == "__main__":
    main()
