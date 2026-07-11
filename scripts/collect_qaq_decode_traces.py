"""Collect prefill-separated autoregressive QAQ decode traces.

This collector deliberately does not run a teacher-forced quality evaluation.  It
records routing and timing evidence only; quality must come from a separate
artifact such as ``scripts/evaluate_qaq_heldout.py``.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoTokenizer

from any_precision import QAQDPLLMForCausalLM
from scripts.collect_qaq_profile_traces import (
    RequestSpec,
    load_requests,
    text_hash,
)


TRACE_SCHEMA_VERSION = "qaq_autoregressive_decode_trace_v1"
UNVALIDATED = "UNVALIDATED"
QUALITY_ARTIFACT_NOTE = "separate_teacher_forced_artifact_required"
DEFAULT_MODES = [
    "fixed_low",
    "fixed_high",
    "dp_threshold_only",
    "mlp_multibit",
    "mlp_multibit_dp_guard",
]
QAQ_MLP_MODES = {"mlp_binary", "mlp_multibit", "mlp_multibit_dp_guard"}
QAQ_DP_MODES = {"dp_threshold_only", "mlp_multibit_dp_guard"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Collect prefill-separated, greedy autoregressive QAQ decode traces "
            "for identical prompts across routing modes."
        )
    )
    parser.add_argument("--ap_model_path", required=True, help="Any-Precision quantized model path.")
    parser.add_argument("--router_checkpoint", default=None, help="QAQ router checkpoint.")
    parser.add_argument("--estimator_results", default=None, help="Directory with DP-LLM estimator artifacts.")
    parser.add_argument("--tokenizer_path", default=None, help="Tokenizer path. Defaults to --ap_model_path.")
    parser.add_argument("--prompt", action="append", default=None, help="Prompt text. Can be repeated.")
    parser.add_argument(
        "--prompt_file",
        default=None,
        help="Text file with one prompt per line, or JSONL containing prompt/text fields.",
    )
    parser.add_argument("--max_requests", type=int, default=None)
    parser.add_argument("--arrival_interval_s", type=float, default=0.0)
    parser.add_argument("--workload_type", default="UNVALIDATED")
    parser.add_argument("--qos_deadline_ms", type=float, default=None)
    parser.add_argument("--reference_mode", default="fixed_high")
    parser.add_argument("--bits", type=int, nargs="+", default=[3, 4, 5, 6])
    parser.add_argument(
        "--modes",
        nargs="+",
        default=DEFAULT_MODES,
        choices=["qaq", "fixed_low", "fixed_high", "mlp_binary", "mlp_multibit", "dp_threshold_only", "mlp_multibit_dp_guard"],
        help="Run every listed mode on every identical prompt.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--confidence_threshold", type=float, default=None)
    parser.add_argument("--fallback_bits", type=int, default=1)
    parser.add_argument("--prefill_by_router", action="store_true", help="Route prompt tokens during prefill too.")
    parser.add_argument("--batch_policy", default="group", choices=["group", "max"])
    parser.add_argument("--include_text", action="store_true")
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


def move_encoding_to_device(encoded, device):
    if hasattr(encoded, "to"):
        return encoded.to(device)
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in encoded.items()}


def clear_model_stats(model):
    if hasattr(model, "clear_router_stats"):
        model.clear_router_stats()
    elif hasattr(model, "clear_comp_count"):
        model.clear_comp_count()
    else:
        raise RuntimeError("Decode trace collection requires model.clear_router_stats().")


def collect_model_stats(model):
    if not hasattr(model, "get_router_stats"):
        raise RuntimeError("Decode trace collection requires model.get_router_stats().")
    return model.get_router_stats()


def synchronize_if_cuda(device):
    device = torch.device(device)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize(device)


def set_deterministic_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def token_hash(token_id: int) -> str:
    return hashlib.sha256(int(token_id).to_bytes(8, byteorder="little", signed=False)).hexdigest()


def token_sequence_hash(token_ids) -> str:
    values = [int(value) for value in token_ids]
    payload = b"".join(value.to_bytes(8, byteorder="little", signed=False) for value in values)
    return hashlib.sha256(payload).hexdigest()


def _counter_delta(after: dict[str, Any], before: dict[str, Any], key: str) -> int:
    return int(after.get(key, 0)) - int(before.get(key, 0))


def _per_layer_event_delta(after: dict[str, Any], before: dict[str, Any], key: str) -> dict[str, int]:
    after_layers = after.get("per_layer", {})
    before_layers = before.get("per_layer", {})
    result = {}
    for route_name in sorted(set(after_layers) | set(before_layers)):
        delta = int(after_layers.get(route_name, {}).get(key, 0)) - int(
            before_layers.get(route_name, {}).get(key, 0)
        )
        if delta:
            result[route_name] = delta
    return result


class DecodeRouteObserver:
    """Capture selected bits for one actual cached decode forward."""

    def __init__(self):
        self.current_token_index = None
        self.current_profile = {}
        self.token_profiles = []

    def reset(self):
        self.current_token_index = None
        self.current_profile = {}
        self.token_profiles = []

    def begin_token(self, token_index: int):
        if self.current_token_index is not None:
            raise RuntimeError("A decode route profile was not finalized before the next token.")
        self.current_token_index = int(token_index)
        self.current_profile = {}

    def __call__(self, linear, flat_x, chosen_bits):
        if self.current_token_index is None:
            # Prefill decisions are intentionally ignored.  They are represented
            # by prefill_router_stats and must not leak into decode profiles.
            return
        bits = [int(bit) for bit in chosen_bits.detach().reshape(-1).cpu().tolist()]
        self.current_profile[str(linear.route_name)] = bits

    def finish_token(self, token_id: int, elapsed_s: float, event_delta: dict[str, Any]):
        if self.current_token_index is None:
            raise RuntimeError("finish_token called without begin_token.")
        self.token_profiles.append(
            {
                "generated_token_index": self.current_token_index,
                "generated_token_id": int(token_id),
                "generated_token_hash": token_hash(token_id),
                "decode_time_s": float(elapsed_s),
                "selected_bits_by_layer": self.current_profile,
                "fallback_count": int(event_delta["fallback_count"]),
                "dp_guard_trigger_count": int(event_delta["dp_guard_trigger_count"]),
                "fallback_events_by_layer": event_delta["fallback_events_by_layer"],
                "dp_guard_events_by_layer": event_delta["dp_guard_events_by_layer"],
            }
        )
        self.current_token_index = None
        self.current_profile = {}


def _event_delta(after: dict[str, Any], before: dict[str, Any]) -> dict[str, Any]:
    return {
        "fallback_count": _counter_delta(after, before, "total_fallbacks"),
        "dp_guard_trigger_count": _counter_delta(after, before, "total_dp_guard_triggers"),
        "fallback_events_by_layer": _per_layer_event_delta(after, before, "fallback_count"),
        "dp_guard_events_by_layer": _per_layer_event_delta(after, before, "dp_guard_trigger_count"),
    }


def _position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    position_ids = attention_mask.cumsum(dim=-1) - 1
    return position_ids.masked_fill(attention_mask == 0, 0)


def _append_attention_token(attention_mask: torch.Tensor) -> torch.Tensor:
    new_column = torch.ones(
        (attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device
    )
    return torch.cat([attention_mask, new_column], dim=1)


def _generated_token(outputs) -> torch.Tensor:
    if not hasattr(outputs, "logits"):
        raise RuntimeError("Causal model output does not contain logits.")
    return outputs.logits[:, -1, :].argmax(dim=-1)


def _eos_ids(tokenizer) -> set[int]:
    value = getattr(tokenizer, "eos_token_id", None)
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {int(item) for item in value}
    return {int(value)}


def _profile_histogram(token_profiles: list[dict[str, Any]]) -> dict[str, Any]:
    overall = Counter()
    per_layer = defaultdict(Counter)
    for token_profile in token_profiles:
        for route_name, bits in token_profile["selected_bits_by_layer"].items():
            for bit in bits:
                overall[str(bit)] += 1
                per_layer[route_name][str(bit)] += 1
    return {
        "overall": dict(sorted(overall.items(), key=lambda item: int(item[0]))),
        "per_layer": {
            route_name: dict(sorted(counts.items(), key=lambda item: int(item[0])))
            for route_name, counts in sorted(per_layer.items())
        },
    }


@torch.no_grad()
def collect_autoregressive_request(model, tokenizer, prompt: str, mode: str, device, max_new_tokens: int, seed: int = 0):
    """Run one prompt in one mode and return prefill/decode-separated evidence."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    mode = normalize_router_mode(mode)
    set_deterministic_seed(seed)
    encoded = move_encoding_to_device(tokenizer([prompt], return_tensors="pt", padding=True), device)
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    position_ids = _position_ids(attention_mask)

    observer = DecodeRouteObserver()
    if not hasattr(model, "set_decision_observer"):
        raise RuntimeError("Decode trace collection requires model.set_decision_observer().")
    model.set_decision_observer(observer)
    clear_model_stats(model)
    observer.reset()

    try:
        synchronize_if_cuda(device)
        prefill_start = time.perf_counter()
        prefill_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            router_mode=mode,
        )
        first_token = _generated_token(prefill_outputs)
        synchronize_if_cuda(device)
        prefill_time_s = time.perf_counter() - prefill_start
        prefill_stats = collect_model_stats(model)

        # This is the mandatory phase boundary: all prompt-token routing is
        # retained as prefill evidence, then removed before decode accounting.
        clear_model_stats(model)
        observer.reset()

        generated_ids = [int(first_token[0].item())] if max_new_tokens > 0 else []
        attention_mask = attention_mask
        past_key_values = getattr(prefill_outputs, "past_key_values", None)
        decode_time_s = 0.0
        eos_ids = _eos_ids(tokenizer)
        if generated_ids and generated_ids[-1] in eos_ids:
            generated_ids = generated_ids[:1]

        for generated_index in range(1, max_new_tokens):
            if generated_ids and generated_ids[-1] in eos_ids:
                break
            if past_key_values is None:
                raise RuntimeError("Causal model did not return past_key_values for cached decode.")

            before_stats = collect_model_stats(model)
            observer.begin_token(generated_index)
            attention_mask = _append_attention_token(attention_mask)
            next_input = torch.tensor(
                [[generated_ids[-1]]], dtype=input_ids.dtype, device=input_ids.device
            )
            next_position_ids = attention_mask.sum(dim=-1, keepdim=True) - 1

            synchronize_if_cuda(device)
            decode_start = time.perf_counter()
            decode_outputs = model(
                input_ids=next_input,
                attention_mask=attention_mask,
                position_ids=next_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                router_mode=mode,
            )
            next_token = _generated_token(decode_outputs)
            synchronize_if_cuda(device)
            elapsed_s = time.perf_counter() - decode_start
            decode_time_s += elapsed_s
            after_stats = collect_model_stats(model)
            observer.finish_token(
                int(next_token[0].item()), elapsed_s, _event_delta(after_stats, before_stats)
            )
            past_key_values = getattr(decode_outputs, "past_key_values", None)
            generated_ids.append(int(next_token[0].item()))

        decode_stats = collect_model_stats(model)
    finally:
        model.set_decision_observer(None)

    generated_tensor = torch.tensor(generated_ids, dtype=torch.long)
    generated_text = tokenizer.batch_decode(
        torch.cat([input_ids.detach().cpu(), generated_tensor.unsqueeze(0)], dim=1),
        skip_special_tokens=True,
    )[0]
    token_profiles = observer.token_profiles
    output_token_count = len(generated_ids)
    decode_token_count = len(token_profiles)

    return {
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "router_mode": mode,
        "deterministic_decoding": True,
        "seed": int(seed),
        "prompt_length_tokens": int(input_ids.shape[-1]),
        "output_token_count": output_token_count,
        "decode_token_count": decode_token_count,
        "generated_token_ids_sha256": token_sequence_hash(generated_ids),
        "generated_token_hashes": [token_hash(token_id) for token_id in generated_ids],
        "generated_text_hash": text_hash(generated_text),
        "prefill_time_s": float(prefill_time_s),
        "ttft_s": float(prefill_time_s),
        "decode_time_s": float(decode_time_s),
        "tpot_s": float(decode_time_s / decode_token_count) if decode_token_count else 0.0,
        "cuda_synchronized_timing": torch.device(device).type == "cuda",
        "ttft_definition": "synchronized prefill through first output-token selection",
        "tpot_definition": "synchronized decode-forward time divided by decode forwards after prefill",
        "prefill_router_stats": prefill_stats,
        "decode_router_stats": decode_stats,
        "decode_selected_bit_profile": _profile_histogram(token_profiles),
        "per_token_route_profiles": token_profiles,
        "quality_evaluation": QUALITY_ARTIFACT_NOTE,
        "generated_text": generated_text,
    }


def make_run_metadata(args) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "ap_model_path": args.ap_model_path,
        "router_checkpoint": args.router_checkpoint,
        "estimator_results": args.estimator_results,
        "candidate_bits": args.bits,
        "modes": [normalize_router_mode(mode) for mode in args.modes],
        "max_new_tokens": args.max_new_tokens,
        "device": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "seed": args.seed,
        "deterministic_decoding": True,
        "identical_prompts_across_modes": True,
        "prefill_stats_cleared_before_decode": True,
        "decode_timing_cuda_synchronized": torch.device(args.device).type == "cuda",
        "quality_evaluation": QUALITY_ARTIFACT_NOTE,
    }


def validate_args(args):
    modes = [normalize_router_mode(mode) for mode in args.modes]
    if len(set(modes)) != len(modes):
        raise ValueError("Each mode may be listed only once.")
    if any(mode in QAQ_MLP_MODES for mode in modes) and args.router_checkpoint is None:
        raise ValueError("MLP QAQ modes require --router_checkpoint.")
    if any(mode in QAQ_DP_MODES for mode in modes) and args.estimator_results is None:
        raise ValueError("DP threshold modes require --estimator_results.")
    device = torch.device(args.device)
    if device.type != "cuda" or device.index not in (None, 0):
        raise ValueError("Decode trace collection must run on CUDA device 0 (use --device cuda:0).")
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible_devices or visible_devices.split(",")[0].strip() != "0":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES with physical Device 0 first, for example CUDA_VISIBLE_DEVICES=0.")
    if args.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1 so TTFT and an output token are recorded.")


def _initial_router_mode(modes: list[str]) -> str:
    for preferred in ("mlp_multibit_dp_guard", "dp_threshold_only", "mlp_multibit", "fixed_high"):
        if preferred in modes:
            return preferred
    return modes[0]


def build_trace_record(request: RequestSpec, mode_result: dict[str, Any], args, run_metadata: dict[str, Any]):
    record = {
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "collector_created_at": run_metadata["created_at"],
        "git_commit": run_metadata["git_commit"],
        "ap_model_path": run_metadata["ap_model_path"],
        "router_checkpoint": run_metadata["router_checkpoint"],
        "estimator_results": run_metadata["estimator_results"],
        "candidate_bits": run_metadata["candidate_bits"],
        "router_mode": mode_result["router_mode"],
        "request_id": request.request_id,
        "prompt_text_hash": text_hash(request.prompt),
        "prompt_length_tokens": mode_result["prompt_length_tokens"],
        "output_token_count": mode_result["output_token_count"],
        "decode_token_count": mode_result["decode_token_count"],
        "generated_token_ids_sha256": mode_result["generated_token_ids_sha256"],
        "generated_token_hashes": mode_result["generated_token_hashes"],
        "generated_text_hash": mode_result["generated_text_hash"],
        "prefill_time_s": mode_result["prefill_time_s"],
        "ttft_s": mode_result["ttft_s"],
        "decode_time_s": mode_result["decode_time_s"],
        "tpot_s": mode_result["tpot_s"],
        "cuda_synchronized_timing": mode_result["cuda_synchronized_timing"],
        "ttft_definition": mode_result["ttft_definition"],
        "tpot_definition": mode_result["tpot_definition"],
        "prefill_router_stats": mode_result["prefill_router_stats"],
        "decode_router_stats": mode_result["decode_router_stats"],
        "decode_selected_bit_profile": mode_result["decode_selected_bit_profile"],
        "per_token_route_profiles": mode_result["per_token_route_profiles"],
        "fallback_count": int(mode_result["decode_router_stats"].get("total_fallbacks", 0)),
        "dp_guard_trigger_count": int(mode_result["decode_router_stats"].get("total_dp_guard_triggers", 0)),
        "deterministic_decoding": mode_result["deterministic_decoding"],
        "seed": mode_result["seed"],
        "quality_evaluation": mode_result["quality_evaluation"],
    }
    if args.include_text:
        record["prompt_text"] = request.prompt
        record["generated_text"] = mode_result["generated_text"]
    return record


def summarize_records(records: list[dict[str, Any]], run_metadata: dict[str, Any]) -> dict[str, Any]:
    by_mode = defaultdict(list)
    for record in records:
        by_mode[record["router_mode"]].append(record)
    summary = {
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "run_metadata": run_metadata,
        "record_count": len(records),
        "prompt_count": len({record["request_id"] for record in records}),
        "by_mode": {},
        "quality_evaluation": QUALITY_ARTIFACT_NOTE,
    }
    for mode, mode_records in sorted(by_mode.items()):
        decode_times = [float(record["decode_time_s"]) for record in mode_records]
        ttft = [float(record["ttft_s"]) for record in mode_records]
        summary["by_mode"][mode] = {
            "request_count": len(mode_records),
            "output_token_count": sum(record["output_token_count"] for record in mode_records),
            "decode_token_count": sum(record["decode_token_count"] for record in mode_records),
            "mean_ttft_s": sum(ttft) / len(ttft) if ttft else 0.0,
            "mean_decode_time_s": sum(decode_times) / len(decode_times) if decode_times else 0.0,
        }
    return summary


def main():
    args = parse_args()
    validate_args(args)
    requests = load_requests(args)
    modes = [normalize_router_mode(mode) for mode in args.modes]

    tokenizer_path = args.tokenizer_path or args.ap_model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = QAQDPLLMForCausalLM.from_quantized(
        args.ap_model_path,
        router_checkpoint=args.router_checkpoint,
        estimator_results=args.estimator_results,
        precisions=args.bits,
        router_mode=_initial_router_mode(modes),
        confidence_threshold=args.confidence_threshold,
        fallback_bits=args.fallback_bits,
        prefill_by_router=args.prefill_by_router,
        batch_policy=args.batch_policy,
        trust_remote_code=args.trust_remote_code,
    ).eval().to(args.device)

    run_metadata = make_run_metadata(args)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with output_path.open("w") as out:
        for request in requests:
            for mode in modes:
                result = collect_autoregressive_request(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=request.prompt,
                    mode=mode,
                    device=args.device,
                    max_new_tokens=args.max_new_tokens,
                    seed=args.seed,
                )
                record = build_trace_record(request, result, args, run_metadata)
                out.write(json.dumps(record) + "\n")
                out.flush()
                records.append(record)
                print(
                    f"{request.request_id} mode={mode}: ttft={record['ttft_s'] * 1000:.2f}ms "
                    f"decode={record['decode_time_s'] * 1000:.2f}ms "
                    f"tokens={record['output_token_count']}",
                    flush=True,
                )

    if args.summary_json is not None:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w") as f:
            json.dump(summarize_records(records, run_metadata), f, indent=2)

    print(f"Wrote {len(records)} mode/request trace records to {output_path}", flush=True)


if __name__ == "__main__":
    main()
