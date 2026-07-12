"""Collect a source-document-preserving FineWeb-Edu request-demand subset.

This is an extension collector for the existing v2 request-demand schema. It
uses a bounded, deterministic prefix of cached parquet rows, hashes documents
into the same development/calibration/test partitions, and runs the real QAQ
quality/profile callback on CUDA. Raw text is never written to the artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

import datasets
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from any_precision import QAQDPLLMForCausalLM, load_qaq_router_checkpoint
from scripts.build_qaq_request_demand_dataset import collect_request
from scripts.qaq_request_demand_protocol import (
    aggregate_records,
    atomic_write_json,
    atomic_write_jsonl,
    document_id,
    document_partition,
    environment_metadata,
    expected_mode_names,
    file_manifest,
    make_collection_record,
    object_sha256,
    tokenizer_file_manifest,
    validate_record,
)

DATASET_NAME = "fineweb_edu"
DATASET_IDENTITY = "HuggingFaceFW/fineweb-edu|sample-10BT"
DATASET_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
SOURCE_SPLIT = "train"
PROTOCOL_VERSION = "qaq_fineweb_request_demand_v1"
RECORD_SCHEMA_VERSION = "qaq_request_demand_v2"
LENGTH_CELLS = ((128, 32), (128, 128), (512, 32), (512, 128))
PARTITION_QUOTAS = {"development": 32, "calibration": 8, "test": 24}
PARTITION_ORDER = {"development": 0, "calibration": 1, "test": 2}
QAQ_MODES = ("dp_threshold_only", "mlp_multibit", "mlp_multibit_dp_guard")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect real FineWeb-Edu QAQ demand with source-document splits."
    )
    parser.add_argument("--ap_model_path", required=True)
    parser.add_argument("--router_checkpoint", required=True)
    parser.add_argument("--estimator_results", required=True)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--parquet_shards", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--row_limit", type=int, default=50000)
    parser.add_argument("--candidate_multiplier", type=int, default=8)
    parser.add_argument("--bits", type=int, nargs="+", default=None)
    parser.add_argument("--safe_nll_delta", type=float, default=0.02)
    parser.add_argument("--profile_layer_group_size", type=int, default=4)
    parser.add_argument("--confidence_threshold", type=float, default=0.6)
    parser.add_argument("--fallback_bits", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch_dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--manifest_only", action="store_true")
    return parser.parse_args()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "UNAVAILABLE"


def selection_hash(doc_id: str, source_index: int, prompt: int, continuation: int) -> str:
    value = f"{doc_id}|{source_index}|{prompt}|{continuation}".encode("ascii")
    return hashlib.sha256(value).hexdigest()


def canonical_metadata(parquet_paths: list[Path], row_limit: int) -> dict[str, Any]:
    files = []
    for path in parquet_paths:
        files.append({
            "name": path.name,
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    return {
        "name": DATASET_NAME,
        "hf_dataset": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "split": SOURCE_SPLIT,
        "revision": DATASET_REVISION,
        "document_unit": "parquet_row",
        "row_limit": int(row_limit),
        "parquet_files": files,
        "fingerprint": object_sha256(files),
    }


def candidate_bucket(doc_id: str) -> tuple[int, int]:
    return LENGTH_CELLS[int(doc_id[:8], 16) % len(LENGTH_CELLS)]


def collect_candidates(
    tokenizer,
    parquet_paths: list[Path],
    row_limit: int,
    candidate_multiplier: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if row_limit < 1 or candidate_multiplier < 1:
        raise ValueError("row_limit and candidate_multiplier must be positive")
    dataset = load_dataset(
        "parquet",
        data_files=[str(path) for path in parquet_paths],
        split="train",
        streaming=True,
    )
    candidates: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    seen_documents: set[str] = set()
    scanned_rows = 0
    for row in dataset:
        if scanned_rows >= row_limit:
            break
        scanned_rows += 1
        text = str(row.get("text") or "").replace("\r\n", "\n").replace("\r", "\n").strip("\n")
        if not text:
            continue
        doc_id = document_id(DATASET_IDENTITY, SOURCE_SPLIT, text)
        if doc_id in seen_documents:
            continue
        seen_documents.add(doc_id)
        partition = document_partition(doc_id)
        prompt, continuation = candidate_bucket(doc_id)
        required_tokens = prompt + continuation
        if int(row.get("token_count") or 0) < required_tokens:
            continue
        candidate = {
            "dataset": DATASET_NAME,
            "source_index": scanned_rows - 1,
            "document_id": doc_id,
            "partition": partition,
            "prompt_length_tokens": prompt,
            "continuation_length_tokens": continuation,
            "start_token": 0,
            "end_token": required_tokens,
            "selection_sha256": selection_hash(doc_id, scanned_rows - 1, prompt, continuation),
            "source_row_id_sha256": hashlib.sha256(str(row.get("id")).encode("utf-8")).hexdigest(),
            "parquet_name": next(path.name for path in parquet_paths if path.exists()),
            "_text": text,
        }
        bucket = (partition, prompt, continuation)
        candidates[bucket].append(candidate)

    selected: list[dict[str, Any]] = []
    stratum_counts: dict[str, int] = {}
    for partition in PARTITION_ORDER:
        for prompt, continuation in LENGTH_CELLS:
            bucket = (partition, prompt, continuation)
            quota = PARTITION_QUOTAS[partition]
            ordered = sorted(candidates.get(bucket, []), key=lambda item: item["selection_sha256"])
            chosen = []
            for item in ordered[: quota * candidate_multiplier]:
                token_ids = tokenizer(
                    item["_text"], add_special_tokens=False, verbose=False
                )["input_ids"]
                if len(token_ids) < item["end_token"]:
                    continue
                item = dict(item)
                item["token_ids"] = token_ids[: item["end_token"]]
                del item["_text"]
                chosen.append(item)
                if len(chosen) == quota:
                    break
            if len(chosen) != quota:
                raise RuntimeError(
                    f"FineWeb subset cannot fill {bucket}: {len(chosen)} / {quota}; "
                    "increase --row_limit or --candidate_multiplier"
                )
            selected.extend(chosen)
            stratum_counts[f"{partition}:{prompt}p:{continuation}c"] = len(chosen)

    selected.sort(
        key=lambda item: (
            PARTITION_ORDER[item["partition"]],
            item["prompt_length_tokens"],
            item["continuation_length_tokens"],
            item["selection_sha256"],
        )
    )
    for request_index, item in enumerate(selected):
        item["request_index"] = request_index
        item["request_id"] = f"{DATASET_NAME}-{item['partition'][:3]}-{item['selection_sha256'][:16]}"
    metadata = {
        "scanned_row_count": scanned_rows,
        "unique_document_count": len(seen_documents),
        "selected_request_count": len(selected),
        "stratum_counts": stratum_counts,
    }
    return selected, metadata


def build_manifest(
    requests: list[dict[str, Any]],
    dataset_metadata: dict[str, Any],
    tokenizer_metadata: dict[str, Any],
) -> dict[str, Any]:
    public_requests = []
    documents: dict[str, dict[str, Any]] = {}
    for request in requests:
        public = {
            key: request[key]
            for key in (
                "request_id",
                "request_index",
                "dataset",
                "source_index",
                "document_id",
                "partition",
                "start_token",
                "end_token",
                "prompt_length_tokens",
                "continuation_length_tokens",
                "selection_sha256",
                "source_row_id_sha256",
                "parquet_name",
            )
        }
        public_requests.append(public)
        documents.setdefault(
            request["document_id"],
            {
                "document_id": request["document_id"],
                "partition": request["partition"],
                "source_index": request["source_index"],
                "source_row_id_sha256": request["source_row_id_sha256"],
                "parquet_name": request["parquet_name"],
            },
        )
    manifest = {
        "manifest_schema_version": "qaq_fineweb_request_manifest_v1",
        "protocol_version": PROTOCOL_VERSION,
        "dataset": dataset_metadata,
        "tokenizer": tokenizer_metadata,
        "selection_policy": {
            "row_order": "sorted parquet shard order, first row_limit rows",
            "document_partition": "document_partition(sha256(dataset_identity|split|normalized_text))",
            "one_request_per_document": True,
            "candidate_cell": "sha256(document_id) modulo four registered length cells",
            "candidate_order": "selection_sha256 ascending within partition and cell",
            "partition_quotas": PARTITION_QUOTAS,
        },
        "documents": [documents[key] for key in sorted(documents)],
        "requests": public_requests,
        "request_count": len(public_requests),
    }
    manifest["subset_token_sha256"] = hashlib.sha256(
        "".join(request["selection_sha256"] for request in public_requests).encode("ascii")
    ).hexdigest()
    manifest["manifest_sha256"] = object_sha256(manifest)
    return manifest


def main() -> None:
    args = parse_args()
    if args.safe_nll_delta < 0 or args.profile_layer_group_size < 1:
        raise ValueError("safe_nll_delta must be non-negative and group size must be positive")
    parquet_paths = sorted(Path(path).resolve() for path in args.parquet_shards)
    if not parquet_paths or any(not path.is_file() for path in parquet_paths):
        raise FileNotFoundError("Every --parquet_shards path must be an existing file")
    tokenizer_path = args.tokenizer_path or args.ap_model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True, local_files_only=args.local_files_only
    )
    dataset_metadata = canonical_metadata(parquet_paths, args.row_limit)
    tokenizer_metadata = tokenizer_file_manifest(tokenizer_path)
    requests, selection_metadata = collect_candidates(
        tokenizer, parquet_paths, args.row_limit, args.candidate_multiplier
    )
    manifest = build_manifest(requests, dataset_metadata, tokenizer_metadata)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "manifest.json", {**manifest, "selection": selection_metadata})

    if args.manifest_only:
        print(json.dumps({
            "status": "MANIFEST_ONLY",
            "output_dir": str(output_dir),
            "request_count": len(requests),
            "manifest_sha256": manifest["manifest_sha256"],
        }, indent=2))
        return

    router, checkpoint = load_qaq_router_checkpoint(args.router_checkpoint)
    bits = sorted(int(bit) for bit in checkpoint["candidate_bits"])
    if args.bits is not None and sorted(args.bits) != bits:
        raise ValueError(f"--bits {sorted(args.bits)} do not match checkpoint bits {bits}")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("FineWeb collection requires CUDA")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES explicitly")
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
        trust_remote_code=True,
    ).eval().to(device)

    raw_records = []
    for request in requests:
        full_ids = torch.tensor(request.pop("token_ids"), dtype=torch.long)
        prompt_length = request["prompt_length_tokens"]
        raw = collect_request(
            model=model,
            tokenizer=tokenizer,
            request_index=request["request_index"],
            prompt_ids=full_ids[:prompt_length],
            continuation_ids=full_ids[prompt_length:],
            full_ids=full_ids,
            bits=bits,
            qaq_modes=list(QAQ_MODES),
            safe_nll_delta=args.safe_nll_delta,
            layer_group_size=args.profile_layer_group_size,
            device=device,
        )
        dataset_metadata_for_record = {
            **dataset_metadata,
            "fingerprint": dataset_metadata["fingerprint"],
        }
        record = make_collection_record(
            raw,
            request,
            manifest["manifest_sha256"],
            dataset_metadata_for_record,
        )
        record["source"]["source_row_id_sha256"] = request["source_row_id_sha256"]
        record["source"]["parquet_name"] = request["parquet_name"]
        for key in ("prompt_token_sha256", "continuation_token_sha256", "request_token_sha256"):
            request[key] = record[key]
        raw_records.append(record)

    expected_modes = expected_mode_names(bits, list(QAQ_MODES))
    for record, request in zip(raw_records, requests):
        validate_record(record, request, manifest["manifest_sha256"], expected_modes)

    run_metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": git_commit(),
        "command": [sys.executable, *sys.argv],
        "dataset": dataset_metadata,
        "selection": selection_metadata,
        "manifest_sha256": manifest["manifest_sha256"],
        "source_files_sha256": {
            "collector": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "shared_collect_callback": hashlib.sha256(
                (REPO_ROOT / "scripts/build_qaq_request_demand_dataset.py").read_bytes()
            ).hexdigest(),
        },
        "input_artifacts": {
            "ap_model": file_manifest(args.ap_model_path),
            "router_checkpoint": file_manifest(args.router_checkpoint),
            "estimator_results": file_manifest(args.estimator_results),
            "tokenizer": tokenizer_metadata,
        },
        "environment": environment_metadata(device),
    }
    atomic_write_json(output_dir / "run-manifest.json", run_metadata)
    atomic_write_jsonl(output_dir / "records.jsonl", raw_records)
    atomic_write_json(
        output_dir / "summary.json",
        {
            "schema_version": "qaq_fineweb_request_summary_v1",
            "validation_status": "REAL_GPU_REQUEST_DEMAND",
            "manifest_sha256": manifest["manifest_sha256"],
            "aggregate": aggregate_records(raw_records),
            "provenance": run_metadata,
        },
    )
    print(json.dumps({
        "status": "REAL_GPU_REQUEST_DEMAND",
        "output_dir": str(output_dir),
        "request_count": len(raw_records),
        "manifest_sha256": manifest["manifest_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()

