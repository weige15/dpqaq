import argparse
import gc
import json
import math
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

from any_precision import DPLLMForCausalLM, QAQDPLLMForCausalLM


QAQ_RUNTIME_MODES = ["fixed_low", "fixed_high", "qaq", "dp_threshold_only", "mlp_multibit_dp_guard"]
ALL_MODES = QAQ_RUNTIME_MODES + ["dp_threshold"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Repeated CUDA-synchronized generation benchmark for QAQ and DP-LLM modes."
    )
    parser.add_argument("--ap_model_path", default=os.environ.get("AP_MODEL_PATH"), help="Any-Precision model path.")
    parser.add_argument("--router_checkpoint", default=os.environ.get("ROUTER_CHECKPOINT"), help="QAQ router checkpoint.")
    parser.add_argument("--estimator_results", default=os.environ.get("ESTIMATOR_RESULTS"), help="Estimator result directory.")
    parser.add_argument("--tokenizer_path", default=None, help="Tokenizer path. Defaults to --ap_model_path.")
    parser.add_argument("--prompt", action="append", default=None, help="Prompt. Can be repeated for batch benchmarking.")
    parser.add_argument("--bits", type=int, nargs="+", default=[3, 4, 5, 6])
    parser.add_argument("--modes", nargs="+", default=ALL_MODES, choices=ALL_MODES)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--confidence_threshold", type=float, default=None)
    parser.add_argument("--fallback_bits", type=int, default=1)
    parser.add_argument(
        "--no_phase_timers",
        action="store_false",
        dest="phase_timers",
        help="Disable QAQDPLLM_Linear phase timers for mlp_multibit_dp_guard.",
    )
    parser.set_defaults(phase_timers=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no_trust_remote_code", action="store_false", dest="trust_remote_code")
    return parser.parse_args()


def require_path(value: str | None, name: str):
    if value is None:
        raise ValueError(f"{name} is required")
    return value


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def git_dirty_status():
    try:
        status = subprocess.check_output(["git", "status", "--short"], text=True).splitlines()
        return status
    except Exception:
        return []


def load_estimator_results(estimator_results):
    estimator_results = require_path(estimator_results, "--estimator_results")
    paths = {
        "max_mem_dict": os.path.join(estimator_results, "max_mem_dict.pt"),
        "linear_reg_d": os.path.join(estimator_results, "linear_reg_d.pt"),
        "jl_d": os.path.join(estimator_results, "jl_d.pt"),
        "T_d": os.path.join(estimator_results, "T_d.pt"),
    }
    missing = [path for path in paths.values() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing estimator files: {missing}")
    return {key: torch.load(path, map_location="cpu", weights_only=False) for key, path in paths.items()}


def clear_model_stats(model):
    if hasattr(model, "clear_router_stats"):
        model.clear_router_stats()
    elif hasattr(model, "clear_comp_count"):
        model.clear_comp_count()


def collect_model_stats(model):
    if hasattr(model, "get_router_stats"):
        return model.get_router_stats()
    if hasattr(model, "ap_linears"):
        per_layer = {}
        total_tokens = 0
        total_selected_bits = 0
        for idx, linear in enumerate(model.ap_linears):
            bit_counts = {str(bit): int(count) for bit, count in linear.comp_count.items()}
            per_layer[f"linear_{idx}"] = {"bit_counts": bit_counts}
            total_tokens += sum(linear.comp_count.values())
            total_selected_bits += sum(bit * count for bit, count in linear.comp_count.items())
        return {
            "average_selected_bit": total_selected_bits / total_tokens if total_tokens > 0 else 0,
            "effective_bits": model.get_effective_bits() if hasattr(model, "get_effective_bits") else None,
            "total_tokens": int(total_tokens),
            "per_layer": per_layer,
        }
    return {"effective_bits": model.get_effective_bits() if hasattr(model, "get_effective_bits") else None}


def synchronize_if_cuda(device):
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def cuda_memory_snapshot(device):
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        return {}
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * (q / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def summarize_values(values):
    if not values:
        return {}
    return {
        "values": values,
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def merge_bit_counts(target, source):
    for bit, count in source.items():
        target[bit] = target.get(bit, 0) + int(count)


def merge_phase_timing(target, source):
    for phase, stats in source.items():
        total = target.setdefault(phase, {"wall_time_s": 0.0, "cuda_time_s": 0.0, "count": 0})
        total["wall_time_s"] += float(stats.get("wall_time_s", 0.0))
        total["cuda_time_s"] += float(stats.get("cuda_time_s", 0.0))
        total["count"] += int(stats.get("count", 0))


def finalize_phase_timing(totals):
    return {
        phase: {
            "wall_time_s": float(values["wall_time_s"]),
            "cuda_time_s": float(values["cuda_time_s"]),
            "count": int(values["count"]),
            "mean_wall_ms": (
                1000.0 * values["wall_time_s"] / values["count"]
                if values["count"] > 0 else 0.0
            ),
            "mean_cuda_ms": (
                1000.0 * values["cuda_time_s"] / values["count"]
                if values["count"] > 0 else 0.0
            ),
        }
        for phase, values in totals.items()
    }


def aggregate_router_stats(stats_list: list[dict[str, Any]]):
    if not stats_list:
        return {}

    effective_bits = [s.get("effective_bits") for s in stats_list if s.get("effective_bits") is not None]
    aggregate: dict[str, Any] = {}
    if effective_bits:
        aggregate["effective_bits"] = summarize_values([float(v) for v in effective_bits])

    per_layer: dict[str, Any] = {}
    total_tokens = 0
    total_selected_bits = 0
    total_fallbacks = 0
    total_dp_guard_triggers = 0
    total_dp_threshold_tokens = 0
    total_dp_threshold_high = 0
    phase_timing_totals = {}

    for stats in stats_list:
        for route_name, layer_stats in stats.get("per_layer", {}).items():
            route = per_layer.setdefault(route_name, {"bit_counts": {}})
            merge_bit_counts(route["bit_counts"], layer_stats.get("bit_counts", {}))
            for key in [
                "fallback_count",
                "dp_guard_trigger_count",
                "dp_threshold_token_count",
                "dp_threshold_high_count",
                "routed_token_count",
            ]:
                if key in layer_stats:
                    route[key] = route.get(key, 0) + int(layer_stats[key])
            if "phase_timing" in layer_stats:
                merge_phase_timing(route.setdefault("phase_timing", {}), layer_stats["phase_timing"])

        if "phase_timing" in stats:
            merge_phase_timing(phase_timing_totals, stats["phase_timing"])

        total_tokens += int(stats.get("total_tokens", 0))
        total_fallbacks += int(stats.get("total_fallbacks", 0))
        total_dp_guard_triggers += int(stats.get("total_dp_guard_triggers", 0))
        total_dp_threshold_tokens += int(stats.get("total_dp_threshold_tokens", 0))
        total_dp_threshold_high += int(stats.get("total_dp_threshold_high", 0))

    for layer_stats in per_layer.values():
        for bit, count in layer_stats["bit_counts"].items():
            total_selected_bits += int(bit) * int(count)
        if "phase_timing" in layer_stats:
            layer_stats["phase_timing"] = finalize_phase_timing(layer_stats["phase_timing"])

    if phase_timing_totals:
        aggregate["phase_timing"] = finalize_phase_timing(phase_timing_totals)

    if total_tokens > 0:
        aggregate.update({
            "average_selected_bit": total_selected_bits / total_tokens,
            "fallback_fraction": total_fallbacks / total_tokens,
            "dp_guard_trigger_fraction": total_dp_guard_triggers / total_tokens,
        })
    if total_dp_threshold_tokens > 0:
        aggregate["dp_threshold_high_fraction"] = total_dp_threshold_high / total_dp_threshold_tokens

    aggregate.update({
        "total_tokens": int(total_tokens),
        "total_fallbacks": int(total_fallbacks),
        "total_dp_guard_triggers": int(total_dp_guard_triggers),
        "total_dp_threshold_tokens": int(total_dp_threshold_tokens),
        "total_dp_threshold_high": int(total_dp_threshold_high),
        "per_layer": per_layer,
    })
    return aggregate


def router_mode_for(mode):
    return "mlp_multibit" if mode == "qaq" else mode


def prepare_inputs(tokenizer, prompts, device):
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    return {key: value.to(device) for key, value in encoded.items()}


def generated_token_count(generated, encoded):
    return int(generated.numel() - encoded["input_ids"].numel())


def set_phase_timers(model, enabled):
    if hasattr(model, "set_phase_timing_enabled"):
        model.set_phase_timing_enabled(enabled)


def format_phase_timing(phase_timing):
    if not phase_timing:
        return ""
    parts = []
    for phase in ["router", "estimator", "grouping", "dequant_matmul", "total"]:
        stats = phase_timing.get(phase)
        if not stats or stats.get("count", 0) == 0:
            continue
        cuda_ms = 1000.0 * stats.get("cuda_time_s", 0.0)
        wall_ms = 1000.0 * stats.get("wall_time_s", 0.0)
        if cuda_ms > 0:
            parts.append(f"{phase}: cuda={cuda_ms:.2f}ms wall={wall_ms:.2f}ms")
        else:
            parts.append(f"{phase}: wall={wall_ms:.2f}ms")
    return "; ".join(parts)


@torch.no_grad()
def run_one_generation(model, encoded, max_new_tokens, device, router_mode=None):
    clear_model_stats(model)
    kwargs = {
        **encoded,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    if router_mode is not None:
        kwargs["router_mode"] = router_mode

    synchronize_if_cuda(device)
    start = time.perf_counter()
    generated = model.generate(**kwargs)
    synchronize_if_cuda(device)
    latency_s = time.perf_counter() - start

    new_tokens = generated_token_count(generated, encoded)
    return {
        "latency_s": latency_s,
        "tokens_per_s": new_tokens / latency_s if latency_s > 0 else 0,
        "generated_tokens": new_tokens,
        "router_stats": collect_model_stats(model),
        "generated_ids": generated,
    }


@torch.no_grad()
def check_finite_logits(model, encoded, router_mode=None):
    clear_model_stats(model)
    kwargs = dict(encoded)
    if router_mode is not None:
        kwargs["router_mode"] = router_mode
    logits = model(**kwargs).logits
    return bool(torch.isfinite(logits).all().item())


def benchmark_mode(model, tokenizer, encoded, prompts, args, mode, router_mode=None):
    print(f"Benchmarking {mode}...", flush=True)
    phase_timers_enabled = bool(args.phase_timers and mode == "mlp_multibit_dp_guard")
    set_phase_timers(model, phase_timers_enabled)
    if torch.cuda.is_available() and torch.device(args.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    warmups = []
    for _ in range(args.warmup):
        result = run_one_generation(model, encoded, args.max_new_tokens, args.device, router_mode=router_mode)
        warmups.append({
            "latency_s": result["latency_s"],
            "tokens_per_s": result["tokens_per_s"],
            "generated_tokens": result["generated_tokens"],
        })

    repeats = []
    last_generated = None
    for repeat_i in range(args.repeat):
        result = run_one_generation(model, encoded, args.max_new_tokens, args.device, router_mode=router_mode)
        last_generated = result.pop("generated_ids")
        result["repeat_index"] = repeat_i
        repeats.append(result)

    finite_logits = check_finite_logits(model, encoded, router_mode=router_mode)
    outputs = tokenizer.batch_decode(last_generated, skip_special_tokens=True) if last_generated is not None else []

    latencies = [r["latency_s"] for r in repeats]
    tokens_per_s = [r["tokens_per_s"] for r in repeats]
    generated_tokens = [r["generated_tokens"] for r in repeats]
    router_stats = [r["router_stats"] for r in repeats]

    summary = {
        "warmup": warmups,
        "repeat_count": args.repeat,
        "latency_s": summarize_values(latencies),
        "tokens_per_s": summarize_values(tokens_per_s),
        "generated_tokens": summarize_values(generated_tokens),
        "finite_logits": finite_logits,
        "phase_timers_enabled": phase_timers_enabled,
        "aggregate_router_stats": aggregate_router_stats(router_stats),
        "repeat_results": repeats,
        "outputs": outputs,
        "cuda_memory": cuda_memory_snapshot(args.device),
    }
    print(
        f"{mode}: p50={summary['latency_s'].get('p50'):.4f}s "
        f"p95={summary['latency_s'].get('p95'):.4f}s "
        f"mean_toks={summary['tokens_per_s'].get('mean'):.2f}",
        flush=True,
    )
    phase_line = format_phase_timing(summary["aggregate_router_stats"].get("phase_timing", {}))
    if phase_line:
        print(f"{mode} phase timing totals: {phase_line}", flush=True)
    set_phase_timers(model, False)
    return summary


def load_qaq_model(args, initial_mode):
    return QAQDPLLMForCausalLM.from_quantized(
        args.ap_model_path,
        router_checkpoint=args.router_checkpoint,
        estimator_results=args.estimator_results,
        precisions=args.bits,
        router_mode=initial_mode,
        confidence_threshold=args.confidence_threshold,
        fallback_bits=args.fallback_bits,
        trust_remote_code=args.trust_remote_code,
    )


def load_dp_model(args):
    estimator = load_estimator_results(args.estimator_results)
    return DPLLMForCausalLM.from_quantized(
        args.ap_model_path,
        precisions=args.bits,
        prefill_by_decode=False,
        trust_remote_code=args.trust_remote_code,
        **estimator,
    )


def model_to_device(model, device):
    model = model.eval()
    return model.to(device)


def main():
    args = parse_args()
    args.ap_model_path = require_path(args.ap_model_path, "--ap_model_path")
    if any(mode in {"qaq", "mlp_multibit_dp_guard"} for mode in args.modes):
        args.router_checkpoint = require_path(args.router_checkpoint, "--router_checkpoint")
    if any(mode in {"dp_threshold", "dp_threshold_only", "mlp_multibit_dp_guard"} for mode in args.modes):
        args.estimator_results = require_path(args.estimator_results, "--estimator_results")

    if torch.device(args.device).type == "cuda" and "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES explicitly before running CUDA benchmarks.")

    prompts = args.prompt or ["Explain mixed-precision inference in one sentence."]
    tokenizer_path = args.tokenizer_path or args.ap_model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded = prepare_inputs(tokenizer, prompts, args.device)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "benchmark_config": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
            "git_dirty_status": git_dirty_status(),
            "ap_model_path": args.ap_model_path,
            "router_checkpoint": args.router_checkpoint,
            "estimator_results": args.estimator_results,
            "bits": args.bits,
            "modes": args.modes,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "prompt_count": len(prompts),
            "prompt_token_count": int(encoded["input_ids"].numel()),
            "max_new_tokens": args.max_new_tokens,
            "device": args.device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "confidence_threshold": args.confidence_threshold,
            "fallback_bits": args.fallback_bits,
            "phase_timers": args.phase_timers,
        },
        "modes": {},
    }

    requested_qaq_modes = [mode for mode in QAQ_RUNTIME_MODES if mode in args.modes]
    if requested_qaq_modes:
        initial_mode = "mlp_multibit" if args.router_checkpoint is not None else router_mode_for(requested_qaq_modes[0])
        print("Loading QAQ model...", flush=True)
        qaq_model = model_to_device(load_qaq_model(args, initial_mode), args.device)
        for mode in requested_qaq_modes:
            result["modes"][mode] = benchmark_mode(
                qaq_model,
                tokenizer,
                encoded,
                prompts,
                args,
                mode,
                router_mode=router_mode_for(mode),
            )
            with output_path.open("w") as f:
                json.dump(result, f, indent=2)
        del qaq_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if "dp_threshold" in args.modes:
        print("Loading DP-LLM threshold model...", flush=True)
        dp_model = model_to_device(load_dp_model(args), args.device)
        result["modes"]["dp_threshold"] = benchmark_mode(
            dp_model,
            tokenizer,
            encoded,
            prompts,
            args,
            "dp_threshold",
            router_mode=None,
        )
        del dp_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with output_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
