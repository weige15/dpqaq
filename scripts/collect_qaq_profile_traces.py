import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoTokenizer

from any_precision import QAQDPLLMForCausalLM


UNVALIDATED = "UNVALIDATED"
TRACE_SCHEMA_VERSION = "qaq_dynamic_batching_trace_v1"
QAQ_MLP_MODES = {"qaq", "mlp_binary", "mlp_multibit", "mlp_multibit_dp_guard"}
QAQ_DP_MODES = {"dp_threshold_only", "mlp_multibit_dp_guard"}


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    prompt: str
    arrival_time_s: float
    workload_type: str
    qos_deadline_ms: float | str
    target_output_length_tokens: int | str
    reference_mode: str


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect per-request QAQ precision-profile traces from real generation runs."
    )
    parser.add_argument("--ap_model_path", required=True, help="Any-Precision quantized model path.")
    parser.add_argument("--router_checkpoint", default=None, help="QAQ router checkpoint.")
    parser.add_argument("--estimator_results", default=None, help="Directory with DP-LLM estimator artifacts.")
    parser.add_argument("--tokenizer_path", default=None, help="Tokenizer path. Defaults to --ap_model_path.")
    parser.add_argument("--prompt", action="append", default=None, help="Prompt text. Can be repeated.")
    parser.add_argument(
        "--prompt_file",
        default=None,
        help=(
            "Text file with one prompt per line, or JSONL with prompt/text plus optional "
            "request_id, arrival_time_s, workload_type, qos_deadline_ms, "
            "target_output_length_tokens, and reference_mode."
        ),
    )
    parser.add_argument("--max_requests", type=int, default=None)
    parser.add_argument("--arrival_interval_s", type=float, default=0.0)
    parser.add_argument("--workload_type", default="UNVALIDATED")
    parser.add_argument("--qos_deadline_ms", type=float, default=None)
    parser.add_argument("--reference_mode", default="fixed_high")
    parser.add_argument("--bits", type=int, nargs="+", default=[3, 4, 5, 6])
    parser.add_argument(
        "--router_mode",
        default="mlp_multibit",
        choices=["qaq", "fixed_low", "fixed_high", "mlp_binary", "mlp_multibit", "dp_threshold_only", "mlp_multibit_dp_guard"],
    )
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--confidence_threshold", type=float, default=None)
    parser.add_argument("--fallback_bits", type=int, default=1)
    parser.add_argument("--prefill_by_router", action="store_true")
    parser.add_argument("--batch_policy", default="group", choices=["group", "max"])
    parser.add_argument("--check_finite_logits", action="store_true")
    parser.add_argument("--include_text", action="store_true", help="Include raw prompt and generated text in trace records.")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no_trust_remote_code", action="store_false", dest="trust_remote_code")
    return parser.parse_args()


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def normalize_router_mode(mode: str) -> str:
    return "mlp_multibit" if mode == "qaq" else mode


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def unvalidated_if_none(value):
    return UNVALIDATED if value is None else value


def coerce_optional_float(value):
    if value is None or value == UNVALIDATED:
        return UNVALIDATED
    return float(value)


def load_requests(args) -> list[RequestSpec]:
    requests: list[RequestSpec] = []

    if args.prompt:
        for prompt in args.prompt:
            requests.append(
                make_request_spec(
                    raw={"prompt": prompt},
                    index=len(requests),
                    default_workload_type=args.workload_type,
                    default_qos_deadline_ms=args.qos_deadline_ms,
                    default_reference_mode=args.reference_mode,
                    arrival_interval_s=args.arrival_interval_s,
                )
            )

    if args.prompt_file:
        path = Path(args.prompt_file)
        with path.open() as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("{"):
                    raw = json.loads(stripped)
                else:
                    raw = {"prompt": stripped}
                requests.append(
                    make_request_spec(
                        raw=raw,
                        index=len(requests),
                        default_workload_type=args.workload_type,
                        default_qos_deadline_ms=args.qos_deadline_ms,
                        default_reference_mode=args.reference_mode,
                        arrival_interval_s=args.arrival_interval_s,
                    )
                )

    if args.max_requests is not None:
        requests = requests[: args.max_requests]

    if not requests:
        raise ValueError("Provide at least one --prompt or --prompt_file entry.")
    return requests


def make_request_spec(
    raw: dict[str, Any],
    index: int,
    default_workload_type: str,
    default_qos_deadline_ms: float | None,
    default_reference_mode: str,
    arrival_interval_s: float,
) -> RequestSpec:
    prompt = raw.get("prompt", raw.get("text"))
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"Prompt record {index} is missing non-empty prompt/text.")

    return RequestSpec(
        request_id=str(raw.get("request_id", f"req_{index:06d}")),
        prompt=prompt,
        arrival_time_s=float(raw.get("arrival_time_s", index * arrival_interval_s)),
        workload_type=str(raw.get("workload_type", default_workload_type)),
        qos_deadline_ms=coerce_optional_float(raw.get("qos_deadline_ms", default_qos_deadline_ms)),
        target_output_length_tokens=raw.get("target_output_length_tokens", UNVALIDATED),
        reference_mode=str(raw.get("reference_mode", default_reference_mode)),
    )


def move_encoding_to_device(encoded, device):
    if hasattr(encoded, "to"):
        return encoded.to(device)
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in encoded.items()}


def clear_model_stats(model):
    if hasattr(model, "clear_router_stats"):
        model.clear_router_stats()
    elif hasattr(model, "clear_comp_count"):
        model.clear_comp_count()


def collect_model_stats(model):
    if hasattr(model, "get_router_stats"):
        return model.get_router_stats()
    raise RuntimeError("QAQ trace collection requires model.get_router_stats().")


def synchronize_if_cuda(device):
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def generated_token_count(generated, encoded) -> int:
    return int(generated.numel() - encoded["input_ids"].numel())


@torch.no_grad()
def run_request(model, tokenizer, request: RequestSpec, args) -> dict[str, Any]:
    clear_model_stats(model)
    encoded = move_encoding_to_device(
        tokenizer([request.prompt], return_tensors="pt", padding=True),
        args.device,
    )
    router_mode = normalize_router_mode(args.router_mode)
    gen_kwargs = {
        **encoded,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "router_mode": router_mode,
    }

    synchronize_if_cuda(args.device)
    start = time.perf_counter()
    generated = model.generate(**gen_kwargs)
    synchronize_if_cuda(args.device)
    gpu_execution_ms = 1000.0 * (time.perf_counter() - start)

    generation_stats = collect_model_stats(model)
    generated_text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    observed_output_length_tokens = generated_token_count(generated, encoded)

    finite_logits: bool | str = UNVALIDATED
    if args.check_finite_logits:
        clear_model_stats(model)
        fwd_kwargs = {**encoded, "router_mode": router_mode}
        logits = model(**fwd_kwargs).logits
        finite_logits = bool(torch.isfinite(logits).all().item())
        clear_model_stats(model)

    return {
        "gpu_execution_ms": gpu_execution_ms,
        "generated_text": generated_text,
        "generated_text_hash": text_hash(generated_text),
        "observed_output_length_tokens": observed_output_length_tokens,
        "generation_router_stats": generation_stats,
        "finite_logits": finite_logits,
    }


def per_layer_bit_counts(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        route_name: dict(layer_stats.get("bit_counts", {}))
        for route_name, layer_stats in stats.get("per_layer", {}).items()
    }


def deadline_missed(qos_deadline_ms, end_to_end_latency_ms):
    if qos_deadline_ms == UNVALIDATED:
        return UNVALIDATED
    return bool(end_to_end_latency_ms > float(qos_deadline_ms))


def build_trace_record(
    request: RequestSpec,
    result: dict[str, Any],
    args,
    run_metadata: dict[str, Any],
) -> dict[str, Any]:
    stats = result["generation_router_stats"]
    queue_delay_ms = 0.0
    gpu_execution_ms = float(result["gpu_execution_ms"])
    end_to_end_latency_ms = queue_delay_ms + gpu_execution_ms

    record = {
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "collector_created_at": run_metadata["created_at"],
        "git_commit": run_metadata["git_commit"],
        "ap_model_path": run_metadata["ap_model_path"],
        "router_checkpoint": run_metadata["router_checkpoint"],
        "estimator_results": run_metadata["estimator_results"],
        "candidate_bits": run_metadata["candidate_bits"],
        "router_mode": run_metadata["router_mode"],
        "request_id": request.request_id,
        "arrival_time_s": request.arrival_time_s,
        "workload_type": request.workload_type,
        "prompt_text_hash": text_hash(request.prompt),
        "prompt_length_tokens": int(run_metadata["last_prompt_length_tokens"]),
        "target_output_length_tokens": request.target_output_length_tokens,
        "observed_output_length_tokens": int(result["observed_output_length_tokens"]),
        "qos_deadline_ms": request.qos_deadline_ms,
        "reference_mode": request.reference_mode,
        "predicted_scalar_bit_budget": UNVALIDATED,
        "predicted_block_precision_vector": UNVALIDATED,
        "profile_id": UNVALIDATED,
        "profile_distance": UNVALIDATED,
        "uncertainty_score": UNVALIDATED,
        "fallback_probability": UNVALIDATED,
        "average_selected_bit": float(stats.get("average_selected_bit", 0.0)),
        "effective_bits": float(stats.get("effective_bits", 0.0)),
        "per_layer_bit_counts": per_layer_bit_counts(stats),
        "fallback_count": int(stats.get("total_fallbacks", 0)),
        "fallback_fraction": float(stats.get("fallback_fraction", 0.0)),
        "dp_guard_trigger_count": int(stats.get("total_dp_guard_triggers", 0)),
        "dp_guard_trigger_fraction": float(stats.get("dp_guard_trigger_fraction", 0.0)),
        "under_precision_label": UNVALIDATED,
        "over_precision_label": UNVALIDATED,
        "batch_id": f"single_{request.request_id}",
        "lane_id": "single_request",
        "batch_policy": "single_request_trace",
        "shared_profile_policy": "none",
        "compatibility_threshold": UNVALIDATED,
        "queue_delay_ms": queue_delay_ms,
        "gpu_execution_ms": gpu_execution_ms,
        "end_to_end_latency_ms": end_to_end_latency_ms,
        "ttft_ms": UNVALIDATED,
        "tpot_ms": UNVALIDATED,
        "deadline_missed": deadline_missed(request.qos_deadline_ms, end_to_end_latency_ms),
        "kernel_launches_per_token": UNVALIDATED,
        "profile_switches_per_token": UNVALIDATED,
        "transfer_bytes_per_token": UNVALIDATED,
        "hbm_bytes_per_token": UNVALIDATED,
        "prefetch_hit_fraction": UNVALIDATED,
        "cuda_graph_reuse_fraction": UNVALIDATED,
        "quality_metric_name": UNVALIDATED,
        "quality_metric_value": UNVALIDATED,
        "reference_quality_metric_value": UNVALIDATED,
        "quality_delta_vs_reference": UNVALIDATED,
        "finite_logits": result["finite_logits"],
        "generated_text_hash": result["generated_text_hash"],
    }

    if args.include_text:
        record["prompt_text"] = request.prompt
        record["generated_text"] = result["generated_text"]
    return record


def summarize_records(records: list[dict[str, Any]], run_metadata: dict[str, Any]) -> dict[str, Any]:
    latencies = [float(record["end_to_end_latency_ms"]) for record in records]
    generated_tokens = [int(record["observed_output_length_tokens"]) for record in records]
    return {
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "run_metadata": run_metadata,
        "request_count": len(records),
        "total_generated_tokens": sum(generated_tokens),
        "latency_ms": {
            "values": latencies,
            "mean": sum(latencies) / len(latencies) if latencies else 0.0,
            "min": min(latencies) if latencies else 0.0,
            "max": max(latencies) if latencies else 0.0,
        },
        "note": "Single-request QAQ trace collection. No dynamic batching performance claim is validated by this artifact.",
    }


def make_run_metadata(args) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "ap_model_path": args.ap_model_path,
        "router_checkpoint": args.router_checkpoint,
        "estimator_results": args.estimator_results,
        "candidate_bits": args.bits,
        "router_mode": normalize_router_mode(args.router_mode),
        "max_new_tokens": args.max_new_tokens,
        "device": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "confidence_threshold": args.confidence_threshold,
        "fallback_bits": args.fallback_bits,
        "prefill_by_router": args.prefill_by_router,
        "batch_policy": args.batch_policy,
        "check_finite_logits": args.check_finite_logits,
        "last_prompt_length_tokens": UNVALIDATED,
    }


def validate_args(args):
    if normalize_router_mode(args.router_mode) in QAQ_MLP_MODES and args.router_checkpoint is None:
        raise ValueError(f"{args.router_mode} requires --router_checkpoint.")
    if normalize_router_mode(args.router_mode) in QAQ_DP_MODES and args.estimator_results is None:
        raise ValueError(f"{args.router_mode} requires --estimator_results.")
    if torch.device(args.device).type == "cuda" and "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES explicitly before running CUDA trace collection.")


def main():
    args = parse_args()
    validate_args(args)
    requests = load_requests(args)

    tokenizer_path = args.tokenizer_path or args.ap_model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = QAQDPLLMForCausalLM.from_quantized(
        args.ap_model_path,
        router_checkpoint=args.router_checkpoint,
        estimator_results=args.estimator_results,
        precisions=args.bits,
        router_mode=normalize_router_mode(args.router_mode),
        confidence_threshold=args.confidence_threshold,
        fallback_bits=args.fallback_bits,
        prefill_by_router=args.prefill_by_router,
        batch_policy=args.batch_policy,
        trust_remote_code=args.trust_remote_code,
    ).eval().to(args.device)

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_metadata = make_run_metadata(args)
    records = []

    with output_path.open("w") as out:
        for request in requests:
            prompt_encoded = tokenizer([request.prompt], return_tensors="pt", padding=True)
            run_metadata["last_prompt_length_tokens"] = int(prompt_encoded["input_ids"].numel())
            result = run_request(model, tokenizer, request, args)
            record = build_trace_record(request, result, args, run_metadata)
            out.write(json.dumps(record) + "\n")
            out.flush()
            records.append(record)
            print(
                f"{request.request_id}: latency={record['end_to_end_latency_ms']:.2f}ms "
                f"avg_bit={record['average_selected_bit']:.3f} "
                f"fallbacks={record['fallback_count']}",
                flush=True,
            )

    if args.summary_json is not None:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w") as f:
            json.dump(summarize_records(records, run_metadata), f, indent=2)

    print(f"Wrote {len(records)} trace records to {output_path}", flush=True)


if __name__ == "__main__":
    main()
