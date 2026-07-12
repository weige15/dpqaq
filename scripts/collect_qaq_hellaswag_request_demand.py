"""Collect a source-document-preserving HellaSwag request-demand subset.

Each HellaSwag source_id is treated as one source document. The document text
contains only the real context fields ctx_a and ctx_b from that source; answer
endings and labels are deliberately excluded. The collector uses the shared
real QAQ quality/profile callback and writes the same v2 request schema as the
existing WikiText, C4, and FineWeb collections.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from any_precision import QAQDPLLMForCausalLM, load_qaq_router_checkpoint
from scripts.build_qaq_request_demand_dataset import collect_request
from scripts.qaq_request_demand_protocol import (
    PROTOCOL_VERSION,
    RECORD_SCHEMA_VERSION,
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

DATASET_NAME = "hellaswag"
DATASET_IDENTITY = "Rowan/hellaswag|train|source_id"
DATASET_REVISION = "218ec52e09a7e7462a5400043bb9a69a41d06b76"
SOURCE_SPLIT = "train"
LENGTH_CELLS = ((32, 16), (32, 32), (64, 16), (64, 32))
PARTITION_QUOTAS = {"development": 32, "calibration": 8, "test": 24}
PARTITION_ORDER = {"development": 0, "calibration": 1, "test": 2}
QAQ_MODES = ("dp_threshold_only", "mlp_multibit", "mlp_multibit_dp_guard")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect real HellaSwag QAQ demand with source-document splits."
    )
    parser.add_argument("--ap_model_path", required=True)
    parser.add_argument("--router_checkpoint", required=True)
    parser.add_argument("--estimator_results", required=True)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--parquet_shards", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--row_limit", type=int, default=39905)
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


def selection_hash(doc_id_value: str, source_index: int, prompt: int, continuation: int) -> str:
    value = f"{doc_id_value}|{source_index}|{prompt}|{continuation}".encode("ascii")
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
        "hf_dataset": "Rowan/hellaswag",
        "config": None,
        "split": SOURCE_SPLIT,
        "revision": DATASET_REVISION,
        "document_unit": "source_id_context_concatenation",
        "input_fields": ["ctx_a", "ctx_b"],
        "excluded_fields": ["endings", "label"],
        "row_limit": int(row_limit),
        "parquet_files": files,
        "fingerprint": object_sha256(files),
    }


def candidate_bucket(doc_id_value: str) -> tuple[int, int]:
    return LENGTH_CELLS[int(doc_id_value[:8], 16) % len(LENGTH_CELLS)]


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
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scanned_rows = 0
    for row in dataset:
        if scanned_rows >= row_limit:
            break
        scanned_rows += 1
        source_id = str(row.get("source_id") or "").strip()
        ctx_a = str(row.get("ctx_a") or "").strip()
        ctx_b = str(row.get("ctx_b") or "").strip()
        if not source_id or not ctx_a:
            continue
        grouped[source_id].append({
            "ind": int(row.get("ind") or scanned_rows - 1),
            "text": f"{ctx_a} {ctx_b}".strip(),
        })

    candidates: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for source_id, rows in sorted(grouped.items()):
        rows.sort(key=lambda item: item["ind"])
        text = "\n".join(item["text"] for item in rows if item["text"])
        if not text:
            continue
        doc_id_value = document_id(DATASET_IDENTITY, SOURCE_SPLIT, source_id)
        prompt, continuation = candidate_bucket(doc_id_value)
        source_index = min(item["ind"] for item in rows)
        row_identity = "|".join(str(item["ind"]) for item in rows)
        candidate = {
            "dataset": DATASET_NAME,
            "source_index": source_index,
            "document_id": doc_id_value,
            "partition": document_partition(doc_id_value),
            "prompt_length_tokens": prompt,
            "continuation_length_tokens": continuation,
            "start_token": 0,
            "end_token": prompt + continuation,
            "selection_sha256": selection_hash(
                doc_id_value, source_index, prompt, continuation
            ),
            "source_id_sha256": hashlib.sha256(source_id.encode("utf-8")).hexdigest(),
            "source_row_id_sha256": hashlib.sha256(row_identity.encode("ascii")).hexdigest(),
            "source_row_count": len(rows),
            "source_id": source_id,
            "parquet_name": parquet_paths[0].name,
            "_text": text,
        }
        candidates[(candidate["partition"], prompt, continuation)].append(candidate)

    selected: list[dict[str, Any]] = []
    stratum_counts: dict[str, int] = {}
    for partition in PARTITION_ORDER:
        for prompt, continuation in LENGTH_CELLS:
            bucket = (partition, prompt, continuation)
            quota = PARTITION_QUOTAS[partition]
            ordered = sorted(
                candidates.get(bucket, []),
                key=lambda item: item["selection_sha256"],
            )
            chosen = []
            for item in ordered:
                token_ids = tokenizer(
                    item["_text"], add_special_tokens=False, verbose=False
                )["input_ids"]
                if len(token_ids) < item["end_token"]:
                    continue
                public = dict(item)
                public["token_ids"] = token_ids[: item["end_token"]]
                del public["_text"]
                del public["source_id"]
                chosen.append(public)
                if len(chosen) == quota:
                    break
            if len(chosen) != quota:
                raise RuntimeError(
                    f"HellaSwag subset cannot fill {bucket}: "
                    f"{len(chosen)} / {quota}; increase --row_limit"
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
        item["request_id"] = (
            f"{DATASET_NAME}-{item['partition'][:3]}-"
            f"{item['selection_sha256'][:16]}"
        )
    metadata = {
        "scanned_row_count": scanned_rows,
        "source_document_count": len(grouped),
        "selected_request_count": len(selected),
        "stratum_counts": stratum_counts,
        "candidate_multiplier": int(candidate_multiplier),
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
                "source_id_sha256",
                "source_row_id_sha256",
                "source_row_count",
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
                "source_id_sha256": request["source_id_sha256"],
                "source_row_id_sha256": request["source_row_id_sha256"],
                "source_row_count": request["source_row_count"],
                "parquet_name": request["parquet_name"],
            },
        )
    manifest = {
        "manifest_schema_version": "qaq_hellaswag_request_manifest_v1",
        "protocol_version": PROTOCOL_VERSION,
        "dataset": dataset_metadata,
        "tokenizer": tokenizer_metadata,
        "selection_policy": {
            "row_order": "parquet row order, bounded by row_limit",
            "document_unit": "one source_id; ctx_a and ctx_b concatenated in ind order",
            "excluded_fields": ["endings", "label"],
            "document_partition": "document_partition(document_id)",
            "one_request_per_document": True,
            "candidate_cell": "sha256(document_id) modulo four short length cells",
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
        tokenizer_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    dataset_metadata = canonical_metadata(parquet_paths, args.row_limit)
    tokenizer_metadata = tokenizer_file_manifest(tokenizer_path)
    requests, selection_metadata = collect_candidates(
        tokenizer,
        parquet_paths,
        args.row_limit,
        args.candidate_multiplier,
    )
    manifest = build_manifest(requests, dataset_metadata, tokenizer_metadata)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        output_dir / "manifest.json",
        {**manifest, "selection": selection_metadata},
    )

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
        raise RuntimeError("HellaSwag collection requires CUDA")
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
        record = make_collection_record(
            raw,
            request,
            manifest["manifest_sha256"],
            dataset_metadata,
        )
        for key in (
            "source_id_sha256",
            "source_row_id_sha256",
            "source_row_count",
            "parquet_name",
        ):
            record["source"][key] = request[key]
        for key in (
            "prompt_token_sha256",
            "continuation_token_sha256",
            "request_token_sha256",
        ):
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
            "schema_version": "qaq_hellaswag_request_summary_v1",
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
