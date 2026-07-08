import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import time

import torch
from transformers import AutoTokenizer

from any_precision import DPLLMForCausalLM, QAQDPLLMForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description="Run generation sanity checks for DP-LLM and QAQ routing.")
    parser.add_argument("--ap_model_path", required=True, help="Any-Precision quantized model path.")
    parser.add_argument("--router_checkpoint", default=None, help="QAQ router checkpoint from train_qaq_router.py.")
    parser.add_argument("--tokenizer_path", default=None, help="Tokenizer path. Defaults to --ap_model_path.")
    parser.add_argument("--estimator_results", default=None, help="Directory with DP-LLM max_mem/linear_reg/jl/T files.")
    parser.add_argument("--prompt", action="append", default=None, help="Prompt. Can be repeated.")
    parser.add_argument("--bits", type=int, nargs="+", default=[3, 4, 5, 6])
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["qaq", "fixed_low", "fixed_high", "dp_threshold"],
        choices=[
            "qaq",
            "fixed_low",
            "fixed_high",
            "dp_threshold",
            "dp_threshold_only",
            "mlp_multibit_dp_guard",
        ],
    )
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--confidence_threshold", type=float, default=None)
    parser.add_argument("--fallback_bits", type=int, default=1)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no_trust_remote_code", action="store_false", dest="trust_remote_code")
    return parser.parse_args()


def load_estimator_results(estimator_results):
    if estimator_results is None:
        raise ValueError("DP threshold modes require --estimator_results")
    return {
        "max_mem_dict": torch.load(os.path.join(estimator_results, "max_mem_dict.pt"), map_location="cpu", weights_only=False),
        "linear_reg_d": torch.load(os.path.join(estimator_results, "linear_reg_d.pt"), map_location="cpu", weights_only=False),
        "jl_d": torch.load(os.path.join(estimator_results, "jl_d.pt"), map_location="cpu", weights_only=False),
        "T_d": torch.load(os.path.join(estimator_results, "T_d.pt"), map_location="cpu", weights_only=False),
    }


def clear_model_stats(model):
    if hasattr(model, "clear_router_stats"):
        model.clear_router_stats()
    elif hasattr(model, "clear_comp_count"):
        model.clear_comp_count()


def collect_model_stats(model):
    if hasattr(model, "get_router_stats"):
        return model.get_router_stats()
    return {"effective_bits": model.get_effective_bits() if hasattr(model, "get_effective_bits") else None}


def synchronize_if_cuda(device):
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def generate_and_report(model, tokenizer, prompts, device, max_new_tokens, router_mode=None):
    model = model.eval().to(device)
    clear_model_stats(model)

    encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    gen_kwargs = {
        **encoded,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    fwd_kwargs = dict(encoded)
    if router_mode is not None:
        gen_kwargs["router_mode"] = router_mode
        fwd_kwargs["router_mode"] = router_mode

    synchronize_if_cuda(device)
    start = time.perf_counter()
    generated = model.generate(**gen_kwargs)
    synchronize_if_cuda(device)
    generation_latency_s = time.perf_counter() - start
    generation_router_stats = collect_model_stats(model)

    clear_model_stats(model)
    logits = model(**fwd_kwargs).logits
    finite_logits = bool(torch.isfinite(logits).all().item())
    sanity_check_router_stats = collect_model_stats(model)
    texts = tokenizer.batch_decode(generated, skip_special_tokens=True)

    return {
        "generation_latency_s": generation_latency_s,
        "generation_tokens_per_s": (
            (generated.numel() - encoded.input_ids.numel()) / generation_latency_s
            if generation_latency_s > 0 else 0
        ),
        "generation_router_stats": generation_router_stats,
        "finite_logits": finite_logits,
        "sanity_check_router_stats": sanity_check_router_stats,
        "outputs": texts,
    }


def main():
    args = parse_args()
    tokenizer_path = args.tokenizer_path or args.ap_model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = args.prompt or ["Explain mixed-precision inference in one sentence."]
    results = {}

    qaq_runtime_modes = ["qaq", "fixed_low", "fixed_high", "dp_threshold_only", "mlp_multibit_dp_guard"]
    requested_qaq_modes = [mode for mode in qaq_runtime_modes if mode in args.modes]
    if requested_qaq_modes:
        router_required_modes = {"qaq", "mlp_multibit_dp_guard"}
        dp_required_modes = {"dp_threshold_only", "mlp_multibit_dp_guard"}
        if any(mode in router_required_modes for mode in requested_qaq_modes) and args.router_checkpoint is None:
            raise ValueError("QAQ MLP modes require --router_checkpoint")
        if any(mode in dp_required_modes for mode in requested_qaq_modes) and args.estimator_results is None:
            raise ValueError("QAQ DP threshold modes require --estimator_results")

        initial_router_mode = "mlp_multibit" if args.router_checkpoint is not None else requested_qaq_modes[0]
        if initial_router_mode == "qaq":
            initial_router_mode = "mlp_multibit"

        qaq_model = QAQDPLLMForCausalLM.from_quantized(
            args.ap_model_path,
            router_checkpoint=args.router_checkpoint,
            estimator_results=args.estimator_results,
            precisions=args.bits,
            router_mode=initial_router_mode,
            confidence_threshold=args.confidence_threshold,
            fallback_bits=args.fallback_bits,
            trust_remote_code=args.trust_remote_code,
        )
        for mode in qaq_runtime_modes:
            if mode not in args.modes:
                continue
            router_mode = "mlp_multibit" if mode == "qaq" else mode
            results[mode] = generate_and_report(
                qaq_model,
                tokenizer,
                prompts,
                args.device,
                args.max_new_tokens,
                router_mode=router_mode,
            )
        del qaq_model
        torch.cuda.empty_cache()

    if "dp_threshold" in args.modes:
        estimator = load_estimator_results(args.estimator_results)
        dp_model = DPLLMForCausalLM.from_quantized(
            args.ap_model_path,
            precisions=args.bits,
            prefill_by_decode=False,
            trust_remote_code=args.trust_remote_code,
            **estimator,
        )
        results["dp_threshold"] = generate_and_report(
            dp_model,
            tokenizer,
            prompts,
            args.device,
            args.max_new_tokens,
        )

    print(json.dumps(results, indent=2))
    if args.output_json is not None:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
