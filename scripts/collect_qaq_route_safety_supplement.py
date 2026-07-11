"""Resumable real-output route-safety supplement for frozen request-demand test requests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import accelerate
import datasets
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer

from any_precision import QAQDPLLMForCausalLM, load_qaq_router_checkpoint
from scripts.analyze_qaq_request_demand_preregistered import verify_freeze
from scripts.evaluate_qaq_heldout import QAQPrecisionAuditor, safe_perplexity
from scripts.qaq_request_demand_protocol import (
    atomic_write_json,
    build_dataset_manifest,
    file_manifest,
    file_sha256,
    object_sha256,
    read_jsonl,
    tokenizer_file_manifest,
)

SCHEMA_VERSION = "qaq_route_safety_supplement_record_v1"
RUN_SCHEMA_VERSION = "qaq_route_safety_supplement_run_v1"
SUMMARY_SCHEMA_VERSION = "qaq_route_safety_supplement_summary_v1"
MODES = ("mlp_multibit", "mlp_multibit_dp_guard")
FORBIDDEN_KEYS = {"text", "prompt_text", "continuation_text", "input_ids", "token_ids", "tokens"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect real required-bit labels on frozen test requests.")
    parser.add_argument("--collection_dir", required=True)
    parser.add_argument("--freeze_manifest", required=True)
    parser.add_argument("--ap_model_path", required=True)
    parser.add_argument("--router_checkpoint", required=True)
    parser.add_argument("--estimator_results", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bits", type=int, nargs="+", default=[3, 4, 5, 6])
    parser.add_argument("--confidence_threshold", type=float, default=0.6)
    parser.add_argument("--fallback_bits", type=int, default=1)
    parser.add_argument("--oracle_batch_size", type=int, default=128)
    parser.add_argument("--shard_size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch_dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--max_shards", type=int, default=None, help="Diagnostic only; omit for full supplement.")
    return parser.parse_args()


def walk_keys(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_keys(child)


def source_hashes() -> dict[str, str]:
    paths = (
        "scripts/collect_qaq_route_safety_supplement.py",
        "scripts/evaluate_qaq_heldout.py",
        "scripts/qaq_request_demand_protocol.py",
        "scripts/analyze_qaq_request_demand_preregistered.py",
        "any_precision/modules/QAQDPLLM_Linear.py",
        "any_precision/modules/QAQDPLLMForCausalLM.py",
        "any_precision/modules/QAQRouter.py",
    )
    return {path: file_sha256(REPO_ROOT / path) for path in paths}


def environment_metadata(device: torch.device | None) -> dict[str, Any]:
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, check=False,
    )
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


def validate_output_dir(output_dir: Path, collection_dir: Path) -> None:
    if output_dir.resolve() == collection_dir.resolve() or collection_dir.resolve() in output_dir.resolve().parents:
        raise ValueError("Supplement output must be outside the frozen collection")


def actual_input_manifests(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import any_precision_ext
        extension_path = Path(any_precision_ext.__file__).resolve()
        extension = {"path": str(extension_path), "size": extension_path.stat().st_size, "sha256": file_sha256(extension_path)}
    except Exception as exc:
        extension = {"status": "UNAVAILABLE", "error": type(exc).__name__}
    return {
        "ap_model": file_manifest(args.ap_model_path),
        "router_checkpoint": file_manifest(args.router_checkpoint),
        "estimator_results": file_manifest(args.estimator_results),
        "tokenizer": tokenizer_file_manifest(args.tokenizer_path),
        "any_precision_extension": extension,
    }


def input_hash(manifest: dict[str, Any]) -> str | None:
    if "tree_sha256" in manifest:
        return manifest["tree_sha256"]
    if "files" in manifest and len(manifest["files"]) == 1:
        return manifest["files"][0]["sha256"]
    return manifest.get("sha256")


def build_run_manifest(
    args: argparse.Namespace,
    freeze: dict[str, Any],
    frozen_run: dict[str, Any],
    manifests: dict[str, dict[str, Any]],
    checkpoint: dict[str, Any],
    device: torch.device | None,
) -> dict[str, Any]:
    inputs = actual_input_manifests(args)
    recorded = frozen_run["input_artifacts"]
    for name in ("ap_model", "router_checkpoint", "estimator_results", "tokenizer", "any_precision_extension"):
        if input_hash(inputs[name]) != input_hash(recorded[name]):
            raise RuntimeError(f"Input artifact differs from frozen collection: {name}")
    sources = source_hashes()
    stable = {
        "supplement_schema": RUN_SCHEMA_VERSION,
        "parent_collection_id": frozen_run["collection_id"],
        "parent_collection_tree_sha256": freeze["collection_tree_sha256"],
        "manifest_hashes": {name: value["manifest_sha256"] for name, value in manifests.items()},
        "input_hashes": {name: input_hash(value) for name, value in inputs.items()},
        "source_files_sha256": sources,
        "candidate_bits": list(args.bits),
        "modes": list(MODES),
        "error_threshold": float(checkpoint["error_threshold"]),
        "confidence_threshold": args.confidence_threshold,
        "fallback_bits": args.fallback_bits,
        "oracle_batch_size": args.oracle_batch_size,
        "shard_size": args.shard_size,
        "torch_dtype": args.torch_dtype,
        "partition": "test",
        "label_scope": "prompt_and_continuation_route_decisions",
        "max_shards": args.max_shards,
    }
    return {
        "run_schema_version": RUN_SCHEMA_VERSION,
        "validation_status": "PREFLIGHT_VALIDATED",
        "supplement_id": object_sha256(stable),
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
        if existing.get("supplement_id") != candidate["supplement_id"]:
            raise RuntimeError("Existing supplement has a different frozen configuration")
        return existing
    atomic_write_json(path, candidate)
    return candidate


def request_test_subset(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    requests = [request for request in manifest["requests"] if request["partition"] == "test"]
    if len(requests) != 96:
        raise RuntimeError("Expected exactly 96 test requests per dataset")
    return requests


def validate_record(record: dict[str, Any], request: dict[str, Any], supplement_id: str) -> None:
    if record.get("schema_version") != SCHEMA_VERSION or record.get("supplement_id") != supplement_id:
        raise ValueError("Supplement record schema or ID mismatch")
    for key in ("request_id", "prompt_token_sha256", "continuation_token_sha256", "request_token_sha256"):
        expected = request[key]
        if record.get(key) != expected:
            raise ValueError(f"Supplement record {key} mismatch")
    if any(key in FORBIDDEN_KEYS for key in walk_keys(record)):
        raise ValueError("Raw text/token payload found in supplement")
    if set(record.get("modes", {})) != set(MODES):
        raise ValueError("Supplement mode mismatch")
    for mode, result in record["modes"].items():
        precision = result["precision_metrics"]
        if precision["decision_count"] <= 0:
            raise ValueError(f"No real route decisions recorded for {mode}")
        if not math.isfinite(float(result["continuation_mean_nll"])) or not result["finite_logits"]:
            raise ValueError(f"Invalid quality output for {mode}")


def validate_shard(path: Path, requests: list[dict[str, Any]], supplement_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = read_jsonl(path)
    if len(records) != len(requests):
        raise ValueError(f"Shard length mismatch: {path}")
    for record, request in zip(records, requests):
        validate_record(record, request, supplement_id)
    meta = {
        "validation_status": "VALIDATED",
        "path": str(path),
        "sha256": file_sha256(path),
        "record_count": len(records),
        "request_ids": [record["request_id"] for record in records],
        "supplement_id": supplement_id,
    }
    return records, meta


def ensure_shard_meta(path: Path, requests: list[dict[str, Any]], supplement_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records, meta = validate_shard(path, requests, supplement_id)
    meta_path = path.with_suffix(".meta.json")
    if meta_path.exists():
        existing = json.loads(meta_path.read_text())
        if any(existing.get(key) != value for key, value in meta.items()):
            raise RuntimeError(f"Shard sidecar mismatch: {meta_path}")
    else:
        atomic_write_json(meta_path, meta)
    return records, meta


def merge_counts(target: dict[str, int], source: dict[str, Any]) -> None:
    for key in ("decision_count", "under_precision_count", "over_precision_count", "exact_precision_count", "signed_bit_gap_sum", "absolute_bit_gap_sum"):
        target[key] += int(source[key])


def summarize_counts(counts: dict[str, int]) -> dict[str, Any]:
    total = counts["decision_count"]
    return {
        **counts,
        "under_precision_rate": counts["under_precision_count"] / total if total else 0.0,
        "over_precision_rate": counts["over_precision_count"] / total if total else 0.0,
        "exact_precision_rate": counts["exact_precision_count"] / total if total else 0.0,
        "mean_signed_bit_gap": counts["signed_bit_gap_sum"] / total if total else 0.0,
        "mean_absolute_bit_gap": counts["absolute_bit_gap_sum"] / total if total else 0.0,
    }


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    modes = {}
    for mode in MODES:
        counts = defaultdict(int)
        route_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for record in records:
            merge_counts(counts, record["modes"][mode]["precision_metrics"])
            for route, metrics in record["modes"][mode]["per_layer_precision_metrics"].items():
                merge_counts(route_counts[route], metrics)
        modes[mode] = {
            "precision_metrics": summarize_counts(counts),
            "per_layer_precision_metrics": {route: summarize_counts(values) for route, values in sorted(route_counts.items())},
            "all_finite_logits": all(record["modes"][mode]["finite_logits"] for record in records),
            "continuation_token_weighted_mean_nll": (
                sum(record["modes"][mode]["continuation_mean_nll"] * record["continuation_length_tokens"] for record in records)
                / sum(record["continuation_length_tokens"] for record in records)
                if records else None
            ),
        }
    return {"request_count": len(records), "modes": modes}


def refresh_summaries(output_dir: Path, manifests: dict[str, dict[str, Any]], run_manifest: dict[str, Any], shard_size: int) -> dict[str, Any]:
    dataset_summaries = {}
    all_records = []
    for dataset, manifest in manifests.items():
        requests = request_test_subset(manifest)
        records = []
        metas = []
        shard_dir = output_dir / "datasets" / dataset / "shards"
        for start in range(0, len(requests), shard_size):
            path = shard_dir / f"shard-{start // shard_size:05d}.jsonl"
            if not path.exists():
                continue
            shard_records, meta = ensure_shard_meta(path, requests[start:start + shard_size], run_manifest["supplement_id"])
            records.extend(shard_records)
            metas.append(meta)
        summary = {
            "summary_schema_version": SUMMARY_SCHEMA_VERSION,
            "validation_status": "REAL_GPU_ROUTE_SAFETY_COMPLETE" if len(records) == len(requests) else "REAL_GPU_ROUTE_SAFETY_IN_PROGRESS",
            "supplement_id": run_manifest["supplement_id"],
            "dataset": dataset,
            "parent_manifest_sha256": manifest["manifest_sha256"],
            "expected_request_count": len(requests),
            "validated_shard_count": len(metas),
            "shards": metas,
            "aggregate": aggregate_records(records),
            "contains_raw_text": False,
        }
        atomic_write_json(output_dir / "datasets" / dataset / "summary.json", summary)
        dataset_summaries[dataset] = summary
        all_records.extend(records)
    complete = all(value["validation_status"] == "REAL_GPU_ROUTE_SAFETY_COMPLETE" for value in dataset_summaries.values())
    combined = {
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "validation_status": "REAL_GPU_ROUTE_SAFETY_COMPLETE" if complete else "REAL_GPU_ROUTE_SAFETY_IN_PROGRESS",
        "supplement_id": run_manifest["supplement_id"],
        "parent_collection_id": run_manifest["config"]["parent_collection_id"],
        "parent_collection_tree_sha256": run_manifest["config"]["parent_collection_tree_sha256"],
        "datasets": {
            name: {"status": value["validation_status"], "request_count": value["aggregate"]["request_count"]}
            for name, value in dataset_summaries.items()
        },
        "aggregate": aggregate_records(all_records),
        "contains_raw_text": False,
    }
    atomic_write_json(output_dir / "combined-summary.json", combined)
    return combined


@torch.no_grad()
def collect_request(
    model,
    auditor: QAQPrecisionAuditor,
    request: dict[str, Any],
    full_ids: torch.Tensor,
    device: torch.device,
    supplement_id: str,
) -> dict[str, Any]:
    prompt_length = request["prompt_length_tokens"]
    encoded = full_ids.unsqueeze(0).to(device)
    labels = encoded.clone()
    labels[:, :prompt_length] = -100
    mode_results = {}
    for mode in MODES:
        model.set_router_mode(mode)
        model.clear_router_stats()
        auditor.start_mode(mode)
        auditor.start_example(0)
        outputs = model(input_ids=encoded, labels=labels, use_cache=False)
        finite = bool(torch.isfinite(outputs.logits).all().item())
        mean_nll = float(outputs.loss.float().item())
        report = auditor.report()
        mode_results[mode] = {
            "continuation_mean_nll": mean_nll,
            "continuation_perplexity": safe_perplexity(mean_nll),
            "finite_logits": finite,
            "precision_metrics": report["summary"],
            "per_layer_precision_metrics": report["per_layer"],
            "runtime_stats": model.get_router_stats(),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "supplement_id": supplement_id,
        "request_id": request["request_id"],
        "dataset": request["dataset"],
        "document_id": request["document_id"],
        "partition": "test",
        "prompt_length_tokens": request["prompt_length_tokens"],
        "continuation_length_tokens": request["continuation_length_tokens"],
        "prompt_token_sha256": request["prompt_token_sha256"],
        "continuation_token_sha256": request["continuation_token_sha256"],
        "request_token_sha256": request["request_token_sha256"],
        "label_scope": "prompt_and_continuation_route_decisions",
        "quality_scope": "continuation_only_teacher_forced",
        "modes": mode_results,
        "contains_raw_text": False,
    }


def main() -> None:
    args = parse_args()
    collection_dir = Path(args.collection_dir).resolve()
    output_dir = Path(args.output_dir)
    validate_output_dir(output_dir, collection_dir)
    if args.shard_size < 1 or args.oracle_batch_size < 1:
        raise ValueError("Shard and oracle batch sizes must be positive")
    if not args.validate_only:
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Real route-safety supplement requires CUDA")
        if not os.environ.get("CUDA_VISIBLE_DEVICES"):
            raise RuntimeError("Set CUDA_VISIBLE_DEVICES explicitly")
    else:
        device = None

    freeze = verify_freeze(collection_dir, Path(args.freeze_manifest).resolve())
    frozen_run = json.loads((collection_dir / "run-manifest.json").read_text())
    router, checkpoint = load_qaq_router_checkpoint(args.router_checkpoint)
    if checkpoint.get("label_mode") != "multibit" or float(checkpoint.get("error_threshold", -1)) != 0.05:
        raise ValueError("Supplement requires the frozen multibit threshold-0.05 router")
    if sorted(args.bits) != sorted(int(bit) for bit in checkpoint["candidate_bits"]):
        raise ValueError("Candidate bits differ from router checkpoint")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=args.local_files_only)
    manifests = {}
    tensors = {}
    for dataset in frozen_run["config"]["datasets"]:
        rebuilt, rebuilt_tensors = build_dataset_manifest(dataset, tokenizer)
        frozen_manifest = json.loads((collection_dir / "manifests" / f"{dataset}.json").read_text())
        if rebuilt != frozen_manifest:
            raise RuntimeError(f"Rebuilt frozen manifest differs: {dataset}")
        manifests[dataset] = rebuilt
        tensors[dataset] = rebuilt_tensors

    candidate_run = build_run_manifest(args, freeze, frozen_run, manifests, checkpoint, device)
    run_manifest = ensure_run_manifest(output_dir / "run-manifest.json", candidate_run)
    if args.validate_only:
        result = refresh_summaries(output_dir, manifests, run_manifest, args.shard_size)
        print(json.dumps(result, indent=2))
        return

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.torch_dtype]
    model = QAQDPLLMForCausalLM.from_quantized(
        args.ap_model_path,
        router=router,
        router_metadata=checkpoint,
        estimator_results=args.estimator_results,
        precisions=sorted(args.bits),
        torch_dtype=dtype,
        router_mode="mlp_multibit_dp_guard",
        confidence_threshold=args.confidence_threshold,
        fallback_bits=args.fallback_bits,
        prefill_by_router=True,
        trust_remote_code=True,
    ).eval().to(device)
    auditor = QAQPrecisionAuditor(float(checkpoint["error_threshold"]), args.oracle_batch_size)
    model.set_decision_observer(auditor)

    committed_shards = 0
    stop = False
    for dataset, manifest in manifests.items():
        requests = request_test_subset(manifest)
        shard_dir = output_dir / "datasets" / dataset / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(requests), args.shard_size):
            shard_requests = requests[start:start + args.shard_size]
            shard_path = shard_dir / f"shard-{start // args.shard_size:05d}.jsonl"
            if shard_path.exists():
                ensure_shard_meta(shard_path, shard_requests, run_manifest["supplement_id"])
                print(f"{dataset} shard {start // args.shard_size:05d}: validated existing", flush=True)
                continue
            records = []
            for request in shard_requests:
                records.append(collect_request(
                    model, auditor, request, tensors[dataset][request["request_id"]], device, run_manifest["supplement_id"]
                ))
                print(f"{dataset} shard {start // args.shard_size:05d} {len(records)}/{len(shard_requests)} {request['request_id']}", flush=True)
            temporary = shard_path.with_suffix(".jsonl.tmp")
            with temporary.open("w", encoding="ascii") as target:
                for record in records:
                    target.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")
                target.flush()
                os.fsync(target.fileno())
            validate_shard(temporary, shard_requests, run_manifest["supplement_id"])
            os.replace(temporary, shard_path)
            ensure_shard_meta(shard_path, shard_requests, run_manifest["supplement_id"])
            refresh_summaries(output_dir, manifests, run_manifest, args.shard_size)
            print(f"{dataset} shard {start // args.shard_size:05d}: committed", flush=True)
            committed_shards += 1
            if args.max_shards is not None and committed_shards >= args.max_shards:
                stop = True
                break
        if stop:
            break
    model.set_decision_observer(None)
    result = refresh_summaries(output_dir, manifests, run_manifest, args.shard_size)
    verify_freeze(collection_dir, Path(args.freeze_manifest).resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
