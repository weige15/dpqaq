"""Held-out quality and routing evaluation for QAQ Any-Precision models.

This command is intentionally CUDA-only: a successfully written artifact has
run the real quantized kernels on a GPU. CPU tests exercise the accounting
helpers without pretending to validate model quality.
"""

import argparse
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
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
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from any_precision import QAQDPLLMForCausalLM, load_qaq_router_checkpoint
from any_precision.modules.QAQDPLLM_Linear import dequant_kbit


SCHEMA_VERSION = "qaq_heldout_eval_v1"
MODES = (
    "fixed_low",
    "fixed_high",
    "dp_threshold_only",
    "mlp_multibit",
    "mlp_multibit_dp_guard",
)
EXECUTION_ORDER = (
    "fixed_high",
    "fixed_low",
    "dp_threshold_only",
    "mlp_multibit",
    "mlp_multibit_dp_guard",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all five QAQ modes on identical held-out token windows and write "
            "a real-GPU JSON artifact."
        )
    )
    parser.add_argument("--ap_model_path", required=True, help="Any-Precision quantized model path.")
    parser.add_argument("--router_checkpoint", required=True, help="Multibit QAQ router checkpoint.")
    parser.add_argument("--estimator_results", required=True, help="Directory containing DP estimator and T_d artifacts.")
    parser.add_argument("--tokenizer_path", default=None, help="Tokenizer path; defaults to --ap_model_path.")
    parser.add_argument("--dataset", choices=["wikitext2", "c4_new"], default="wikitext2")
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--dataset_start", type=int, default=0, help="First non-overlapping token window.")
    parser.add_argument("--num_examples", type=int, default=16)
    parser.add_argument("--bits", type=int, nargs="+", default=None, help="Must match checkpoint candidate bits.")
    parser.add_argument("--confidence_threshold", type=float, default=None)
    parser.add_argument("--fallback_bits", type=int, default=1)
    parser.add_argument("--oracle_batch_size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch_dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no_trust_remote_code", action="store_false", dest="trust_remote_code")
    return parser.parse_args()


def safe_perplexity(nll: float) -> float:
    return math.exp(nll) if nll < math.log(sys.float_info.max) else math.inf


def build_token_windows(
    input_ids: torch.Tensor,
    context_length: int,
    start: int,
    count: int,
) -> list[torch.Tensor]:
    if context_length < 2:
        raise ValueError("--context_length must be at least 2")
    if start < 0 or count < 1:
        raise ValueError("--dataset_start must be non-negative and --num_examples must be positive")
    tokens = input_ids.reshape(-1)
    available = tokens.numel() // context_length
    if start + count > available:
        raise ValueError(
            f"Requested windows [{start}, {start + count}), but only {available} full "
            f"windows of length {context_length} are available."
        )
    return [
        tokens[index * context_length:(index + 1) * context_length].clone()
        for index in range(start, start + count)
    ]


def token_sha256(token_ids: torch.Tensor) -> str:
    payload = json.dumps(token_ids.reshape(-1).tolist(), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def smallest_safe_bits(
    x: torch.Tensor,
    weights: dict[int, torch.Tensor],
    reference_bit: int,
    error_threshold: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return the smallest bit whose real linear output is safe versus reference."""
    bits = sorted(weights)
    if reference_bit not in weights or reference_bit != max(bits):
        raise ValueError("reference_bit must be the largest available candidate bit")
    w_ref = weights[reference_bit]
    y_ref = torch.matmul(x.to(dtype=w_ref.dtype), w_ref.T).float()
    required = torch.full((x.shape[0],), reference_bit, dtype=torch.long, device=x.device)
    unassigned = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
    for bit in bits:
        if bit == reference_bit or not unassigned.any():
            break
        w_bit = weights[bit]
        y_bit = torch.matmul(x.to(dtype=w_bit.dtype), w_bit.T).float()
        rel_error = (y_ref - y_bit).norm(dim=-1) / (y_ref.norm(dim=-1) + eps)
        safe = unassigned & (rel_error <= error_threshold)
        required[safe] = bit
        unassigned[safe] = False
    return required


def precision_counts(selected: torch.Tensor, required: torch.Tensor) -> dict[str, int | float]:
    selected = selected.reshape(-1).to(dtype=torch.long)
    required = required.reshape(-1).to(device=selected.device, dtype=torch.long)
    if selected.shape != required.shape:
        raise ValueError("selected and required precision tensors must have the same shape")
    total = selected.numel()
    under = int((selected < required).count_nonzero().item())
    over = int((selected > required).count_nonzero().item())
    exact = total - under - over
    signed_gap = int((selected - required).sum().item())
    absolute_gap = int((selected - required).abs().sum().item())
    return {
        "decision_count": total,
        "under_precision_count": under,
        "over_precision_count": over,
        "exact_precision_count": exact,
        "signed_bit_gap_sum": signed_gap,
        "absolute_bit_gap_sum": absolute_gap,
    }


def merge_precision_counts(target: dict[str, int], update: dict[str, int | float]) -> None:
    for key in (
        "decision_count",
        "under_precision_count",
        "over_precision_count",
        "exact_precision_count",
        "signed_bit_gap_sum",
        "absolute_bit_gap_sum",
    ):
        target[key] = int(target.get(key, 0)) + int(update[key])


def summarize_precision_counts(counts: dict[str, int]) -> dict[str, int | float]:
    total = int(counts.get("decision_count", 0))
    summary = {key: int(value) for key, value in counts.items()}
    summary.update({
        "under_precision_rate": summary.get("under_precision_count", 0) / total if total else 0.0,
        "over_precision_rate": summary.get("over_precision_count", 0) / total if total else 0.0,
        "exact_precision_rate": summary.get("exact_precision_count", 0) / total if total else 0.0,
        "mean_signed_bit_gap": summary.get("signed_bit_gap_sum", 0) / total if total else 0.0,
        "mean_absolute_bit_gap": summary.get("absolute_bit_gap_sum", 0) / total if total else 0.0,
    })
    return summary


class QAQPrecisionAuditor:
    """Observe actual runtime decisions and compare them with real output-error labels."""

    def __init__(self, error_threshold: float, oracle_batch_size: int):
        self.error_threshold = float(error_threshold)
        self.oracle_batch_size = int(oracle_batch_size)
        if self.oracle_batch_size < 1:
            raise ValueError("--oracle_batch_size must be positive")
        self.mode = None
        self.example_index = None
        self.mode_counts: dict[str, int] = {}
        self.example_counts: dict[int, dict[str, int]] = {}
        self.route_counts: dict[str, dict[str, int]] = {}

    def start_mode(self, mode: str) -> None:
        self.mode = mode
        self.example_index = None
        self.mode_counts = {}
        self.example_counts = {}
        self.route_counts = {}

    def start_example(self, example_index: int) -> None:
        if self.mode is None:
            raise RuntimeError("start_mode must be called before start_example")
        self.example_index = int(example_index)
        self.example_counts[self.example_index] = {}

    @torch.no_grad()
    def __call__(self, linear, flat_x: torch.Tensor, selected_bits: torch.Tensor) -> None:
        if self.example_index is None:
            raise RuntimeError("Precision decision observed outside an active example")
        valid_bits = sorted(linear._valid_bits())
        reference_bit = max(valid_bits)
        w_ref = dequant_kbit(linear.qweight, linear._buffers[f"lut{reference_bit}"], reference_bit)
        required = torch.full(
            (flat_x.shape[0],), reference_bit, dtype=torch.long, device=flat_x.device
        )
        unassigned = torch.ones(flat_x.shape[0], dtype=torch.bool, device=flat_x.device)

        for bit in valid_bits:
            if bit == reference_bit or not unassigned.any():
                break
            w_bit = dequant_kbit(linear.qweight, linear._buffers[f"lut{bit}"], bit)
            for start in range(0, flat_x.shape[0], self.oracle_batch_size):
                stop = min(start + self.oracle_batch_size, flat_x.shape[0])
                x = flat_x[start:stop]
                y_ref = torch.matmul(x.to(dtype=w_ref.dtype), w_ref.T).float()
                y_bit = torch.matmul(x.to(dtype=w_bit.dtype), w_bit.T).float()
                rel_error = (y_ref - y_bit).norm(dim=-1) / (y_ref.norm(dim=-1) + 1e-8)
                safe = unassigned[start:stop] & (rel_error <= self.error_threshold)
                required_slice = required[start:stop]
                unassigned_slice = unassigned[start:stop]
                required_slice[safe] = bit
                unassigned_slice[safe] = False
            del w_bit

        counts = precision_counts(selected_bits, required)
        merge_precision_counts(self.mode_counts, counts)
        merge_precision_counts(self.example_counts[self.example_index], counts)
        route_counts = self.route_counts.setdefault(linear.route_name, {})
        merge_precision_counts(route_counts, counts)
        del w_ref, required

    def report(self) -> dict[str, Any]:
        return {
            "summary": summarize_precision_counts(self.mode_counts),
            "per_example": {
                str(index): summarize_precision_counts(counts)
                for index, counts in sorted(self.example_counts.items())
            },
            "per_layer": {
                route: summarize_precision_counts(counts)
                for route, counts in sorted(self.route_counts.items())
            },
        }


def load_heldout_text(dataset_name: str) -> tuple[str, dict[str, Any]]:
    if dataset_name == "wikitext2":
        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        return "\n\n".join(dataset["text"]), {
            "hf_dataset": "Salesforce/wikitext",
            "config": "wikitext-2-raw-v1",
            "split": "test",
            "text_join": "double_newline",
            "dataset_fingerprint": dataset._fingerprint,
        }
    dataset = load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )
    return " ".join(dataset[:1100]["text"]), {
        "hf_dataset": "allenai/c4",
        "config": None,
        "split": "validation",
        "data_file": "en/c4-validation.00000-of-00008.json.gz",
        "source_rows": "first_1100",
        "text_join": "space",
        "dataset_fingerprint": dataset._fingerprint,
    }


def attach_nll_deltas(mode_results: dict[str, dict[str, Any]]) -> None:
    baseline_examples = mode_results["fixed_high"]["examples"]
    baseline = {item["example_index"]: item for item in baseline_examples}
    for mode in MODES:
        for item in mode_results[mode]["examples"]:
            high = baseline[item["example_index"]]
            item["nll_delta_vs_fixed_high"] = item["mean_nll"] - high["mean_nll"]
            item["perplexity_delta_vs_fixed_high"] = item["perplexity"] - high["perplexity"]
        mode_results[mode]["mean_nll_delta_vs_fixed_high"] = (
            mode_results[mode]["mean_nll"] - mode_results["fixed_high"]["mean_nll"]
        )
        mode_results[mode]["perplexity_delta_vs_fixed_high"] = (
            mode_results[mode]["perplexity"] - mode_results["fixed_high"]["perplexity"]
        )


def validate_fixed_high_precision(metrics: dict[str, int | float]) -> None:
    if int(metrics["under_precision_count"]) > 0:
        raise RuntimeError(
            "fixed_high executed below a per-route reference precision; artifact not written"
        )


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "UNAVAILABLE"


def source_provenance() -> dict[str, Any]:
    source_paths = (
        Path("scripts/evaluate_qaq_heldout.py"),
        Path("any_precision/modules/QAQDPLLM_Linear.py"),
        Path("any_precision/modules/QAQDPLLMForCausalLM.py"),
    )
    hashes = {
        str(path): hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest()
        for path in source_paths
    }
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return {"git_worktree_dirty": bool(status.stdout.strip()), "source_files_sha256": hashes}


def environment_metadata(device: torch.device) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "accelerate": accelerate.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "command": [sys.executable, *sys.argv],
        **source_provenance(),
    }


@torch.no_grad()
def evaluate_mode(model, windows, mode, device, auditor):
    model.set_router_mode(mode)
    model.clear_router_stats()
    auditor.start_mode(mode)
    examples = []
    weighted_nll = 0.0
    total_target_tokens = 0

    for example_index, window in enumerate(tqdm(windows, desc=mode, unit="example")):
        auditor.start_example(example_index)
        input_ids = window.unsqueeze(0).to(device)
        outputs = model(input_ids=input_ids, labels=input_ids, use_cache=False)
        mean_nll = float(outputs.loss.float().item())
        finite_logits = bool(torch.isfinite(outputs.logits).all().item())
        if not math.isfinite(mean_nll) or not finite_logits:
            raise RuntimeError(
                f"Non-finite model output in mode {mode}, example {example_index}; artifact not written"
            )
        target_tokens = int(input_ids.shape[1] - 1)
        weighted_nll += mean_nll * target_tokens
        total_target_tokens += target_tokens
        examples.append({
            "example_index": example_index,
            "token_sha256": token_sha256(window),
            "target_token_count": target_tokens,
            "mean_nll": mean_nll,
            "perplexity": safe_perplexity(mean_nll),
        })

    mean_nll = weighted_nll / total_target_tokens
    routing_quality = auditor.report()
    for item in examples:
        item["precision_metrics"] = routing_quality["per_example"][str(item["example_index"])]
    return {
        "mean_nll": mean_nll,
        "perplexity": safe_perplexity(mean_nll),
        "target_token_count": total_target_tokens,
        "finite_logits": True,
        "runtime_stats": model.get_router_stats(),
        "precision_metrics": routing_quality["summary"],
        "per_layer_precision_metrics": routing_quality["per_layer"],
        "examples": examples,
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "Held-out QAQ artifacts require a real CUDA run. Use CPU only for tests; no artifact was written."
        )
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError(
            "Set CUDA_VISIBLE_DEVICES explicitly before held-out evaluation; no artifact was written."
        )
    if dequant_kbit is None:
        raise ModuleNotFoundError("Install any_precision_ext from any_precision/modules/kernels first.")

    router, checkpoint = load_qaq_router_checkpoint(args.router_checkpoint)
    if checkpoint.get("label_mode") != "multibit":
        raise ValueError("Held-out mlp_multibit evaluation requires a multibit router checkpoint.")
    error_threshold = checkpoint.get("error_threshold")
    if error_threshold is None:
        raise ValueError("Router checkpoint is missing error_threshold for real precision labels.")
    training_dataset = checkpoint.get("training_config", {}).get("dataset")
    if training_dataset not in {"c4", "wikitext2"}:
        raise ValueError(
            "Router checkpoint must document a supported training dataset to prove split isolation."
        )
    bits = [int(bit) for bit in checkpoint["candidate_bits"]]
    if args.bits is not None and sorted(args.bits) != bits:
        raise ValueError(f"--bits {sorted(args.bits)} do not match checkpoint candidate bits {bits}")

    tokenizer_path = args.tokenizer_path or args.ap_model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    text, dataset_metadata = load_heldout_text(args.dataset)
    input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False, verbose=False).input_ids
    windows = build_token_windows(
        input_ids, args.context_length, args.dataset_start, args.num_examples
    )
    subset_sha256 = hashlib.sha256(
        "".join(token_sha256(window) for window in windows).encode("ascii")
    ).hexdigest()

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

    auditor = QAQPrecisionAuditor(error_threshold, args.oracle_batch_size)
    model.set_decision_observer(auditor)
    mode_results = {
        mode: evaluate_mode(model, windows, mode, device, auditor)
        for mode in EXECUTION_ORDER
    }
    model.set_decision_observer(None)
    attach_nll_deltas(mode_results)

    validate_fixed_high_precision(mode_results["fixed_high"]["precision_metrics"])

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "validation_status": "REAL_GPU_HELDOUT",
        "environment": environment_metadata(device),
        "inputs": {
            "ap_model_path": str(Path(args.ap_model_path).resolve()),
            "router_checkpoint": str(Path(args.router_checkpoint).resolve()),
            "estimator_results": str(Path(args.estimator_results).resolve()),
            "tokenizer_path": str(Path(tokenizer_path).resolve()),
            "candidate_bits": bits,
            "error_threshold": float(error_threshold),
            "confidence_threshold": args.confidence_threshold,
            "fallback_bits": args.fallback_bits,
            "prefill_by_router": True,
        },
        "dataset": {
            "name": args.dataset,
            **dataset_metadata,
            "context_length": args.context_length,
            "window_policy": "non_overlapping_contiguous_tokens",
            "window_start": args.dataset_start,
            "num_examples": args.num_examples,
            "add_special_tokens": False,
            "subset_token_sha256": subset_sha256,
            "held_out_from_router_training": True,
            "router_training_dataset": training_dataset,
            "router_training_split": "train",
        },
        "metric_definitions": {
            "mean_nll": "Mean next-token negative log likelihood; lower is better.",
            "perplexity": "exp(mean_nll).",
            "required_bit": (
                "Smallest available bit with relative linear-output error <= checkpoint error_threshold "
                "versus the highest available reference bit, evaluated on the mode's real activation."
            ),
            "under_precision": "actual selected bit < required_bit",
            "over_precision": "actual selected bit > required_bit",
            "effective_bits": "Parameter-count and execution-count weighted bit width from runtime stats.",
        },
        "modes": {mode: mode_results[mode] for mode in MODES},
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w") as f:
        json.dump(artifact, f, indent=2)
    os.replace(temporary_path, output_path)
    print(json.dumps({
        "artifact": str(output_path),
        "validation_status": artifact["validation_status"],
        "subset_token_sha256": subset_sha256,
        "mode_summary": {
            mode: {
                "perplexity": mode_results[mode]["perplexity"],
                "perplexity_delta_vs_fixed_high": mode_results[mode]["perplexity_delta_vs_fixed_high"],
                "effective_bits": mode_results[mode]["runtime_stats"]["effective_bits"],
                "fallback_fraction": mode_results[mode]["runtime_stats"]["fallback_fraction"],
                "dp_guard_trigger_fraction": mode_results[mode]["runtime_stats"]["dp_guard_trigger_fraction"],
                "under_precision_rate": mode_results[mode]["precision_metrics"]["under_precision_rate"],
                "over_precision_rate": mode_results[mode]["precision_metrics"]["over_precision_rate"],
            }
            for mode in MODES
        },
    }, indent=2))


if __name__ == "__main__":
    main()
