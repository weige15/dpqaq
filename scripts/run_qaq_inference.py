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
    parser.add_argument("--router_checkpoint", required=True, help="QAQ router checkpoint from train_qaq_router.py.")
    parser.add_argument("--tokenizer_path", default=None, help="Tokenizer path. Defaults to --ap_model_path.")
    parser.add_argument("--estimator_results", default=None, help="Directory with DP-LLM max_mem/linear_reg/jl/T files.")
    parser.add_argument("--prompt", action="append", default=None, help="Prompt. Can be repeated.")
    parser.add_argument("--bits", type=int, nargs="+", default=[3, 4, 5, 6])
    parser.add_argument("--modes", nargs="+", default=["qaq", "fixed_low", "fixed_high", "dp_threshold"],
                        choices=["qaq", "fixed_low", "fixed_high", "dp_threshold"])
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
        raise ValueError("dp_threshold mode requires --estimator_results")
    return {
        "max_mem_dict": torch.load(os.path.join(estimator_results, "max_mem_dict.pt"), map_location="cpu", weights_only=False),
        "linear_reg_d": torch.load(os.path.join(estimator_results, "linear_reg_d.pt"), map_location="cpu", weights_only=False),
        "jl_d": torch.load(os.path.join(estimator_results, "jl_d.pt"), map_location="cpu", weights_only=False),
        "T_d": torch.load(os.path.join(estimator_results, "T_d.pt"), map_location="cpu", weights_only=False),
    }


@torch.no_grad()
def generate_and_report(model, tokenizer, prompts, device, max_new_tokens, router_mode=None):
    model = model.eval().to(device)
    if hasattr(model, "clear_router_stats"):
        model.clear_router_stats()
    elif hasattr(model, "clear_comp_count"):
        model.clear_comp_count()

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

    start = time.perf_counter()
    generated = model.generate(**gen_kwargs)
    latency_s = time.perf_counter() - start

    logits = model(**fwd_kwargs).logits
    finite_logits = bool(torch.isfinite(logits).all().item())
    texts = tokenizer.batch_decode(generated, skip_special_tokens=True)

    if hasattr(model, "get_router_stats"):
        stats = model.get_router_stats()
    else:
        stats = {"effective_bits": model.get_effective_bits() if hasattr(model, "get_effective_bits") else None}

    stats.update({
        "latency_s": latency_s,
        "tokens_per_s": (generated.numel() - encoded.input_ids.numel()) / latency_s if latency_s > 0 else 0,
        "finite_logits": finite_logits,
        "outputs": texts,
    })
    return stats


def main():
    args = parse_args()
    tokenizer_path = args.tokenizer_path or args.ap_model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = args.prompt or ["Explain mixed-precision inference in one sentence."]
    results = {}

    if any(mode in args.modes for mode in ["qaq", "fixed_low", "fixed_high"]):
        qaq_model = QAQDPLLMForCausalLM.from_quantized(
            args.ap_model_path,
            router_checkpoint=args.router_checkpoint,
            estimator_results=args.estimator_results,
            precisions=args.bits,
            router_mode="mlp_multibit",
            confidence_threshold=args.confidence_threshold,
            fallback_bits=args.fallback_bits,
            trust_remote_code=args.trust_remote_code,
        )
        for mode in ["qaq", "fixed_low", "fixed_high"]:
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
