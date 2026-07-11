"""Preregistered, resumable request-demand collection utilities.

This module owns deterministic document manifests and validated shard lifecycle.
The model execution callback remains in build_qaq_request_demand_dataset.py so
legacy and preregistered collection use the same real QAQ path.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import platform
import socket
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import accelerate
import huggingface_hub
import datasets
import numpy as np
import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer

from any_precision import QAQDPLLMForCausalLM


PROTOCOL_VERSION = "qaq_request_demand_preregistered_v1"
RECORD_SCHEMA_VERSION = "qaq_request_demand_v2"
SHARD_SCHEMA_VERSION = "qaq_request_demand_shard_v1"
SUMMARY_SCHEMA_VERSION = "qaq_request_demand_summary_v2"
MANIFEST_SCHEMA_VERSION = "qaq_request_manifest_v1"

WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
C4_REVISION = "607bd4c8450a42878aa9ddc051a65a055450ef87"
C4_DATA_FILE = "en/c4-validation.00000-of-00008.json.gz"
SPLIT_SALT = "dpqaq-request-demand-v1"
PILOT_WIKITEXT_TOKEN_LIMIT = 8192
TOKEN_GAP = 128
MAX_REQUESTS_PER_DOCUMENT = 16
MAX_REQUESTS_PER_CELL_DOCUMENT = 4

LENGTH_CELLS = ((128, 32), (128, 128), (512, 32), (512, 128))
PARTITION_QUOTAS = {"development": 32, "calibration": 8, "test": 24}
PARTITION_ORDER = {"development": 0, "calibration": 1, "test": 2}
DEFAULT_DATASETS = ("wikitext2", "c4_new")
PILOT_FILENAMES = {
    "qaq-request-demand-wikitext2-32x128p64c.jsonl",
    "qaq-request-demand-wikitext2-32x128p64c-summary.json",
}

FORBIDDEN_ARTIFACT_KEYS = {
    "text",
    "prompt_text",
    "continuation_text",
    "generated_text",
    "input_ids",
    "prompt_ids",
    "continuation_ids",
    "token_ids",
    "tokens",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def token_ids_sha256(token_ids: list[int] | torch.Tensor) -> str:
    if isinstance(token_ids, torch.Tensor):
        values = token_ids.reshape(-1).tolist()
    else:
        values = list(token_ids)
    payload = json.dumps(values, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_manifest(path: str | Path) -> dict[str, Any]:
    root = Path(path).resolve()
    if root.is_file():
        return {
            "root": str(root),
            "files": [{"path": root.name, "size": root.stat().st_size, "sha256": file_sha256(root)}],
        }
    files = []
    for item in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        files.append({
            "path": str(item.relative_to(root)),
            "size": item.stat().st_size,
            "sha256": file_sha256(item),
        })
    manifest = {"root": str(root), "files": files}
    manifest["tree_sha256"] = object_sha256(files)
    return manifest


def tokenizer_file_manifest(path: str | Path) -> dict[str, Any]:
    root = Path(path).resolve()
    names = {
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "added_tokens.json", "vocab.json", "merges.txt", "tokenizer.model",
    }
    files = []
    for item in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        if item.name not in names:
            continue
        files.append({
            "path": str(item.relative_to(root)),
            "size": item.stat().st_size,
            "sha256": file_sha256(item),
        })
    if not files:
        raise FileNotFoundError(f"No tokenizer files found under {root}")
    return {"root": str(root), "files": files, "tree_sha256": object_sha256(files)}


def atomic_write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="ascii") as out:
        json.dump(value, out, indent=2, sort_keys=True, ensure_ascii=True)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    os.replace(temporary, target)


def atomic_write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="ascii") as out:
        for record in records:
            out.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")
        out.flush()
        os.fsync(out.fileno())
    os.replace(temporary, target)


def normalize_document_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip("\n")


def document_id(dataset_identity: str, split: str, text: str) -> str:
    payload = f"{dataset_identity}|{split}|{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def document_partition(doc_id: str) -> str:
    digest = hashlib.sha256(f"{SPLIT_SALT}|{doc_id}".encode("ascii")).digest()
    residue = int.from_bytes(digest[:8], "big") % 9
    if residue <= 3:
        return "development"
    if residue == 4:
        return "calibration"
    return "test"


def parse_wikitext_articles(rows: list[str]) -> list[tuple[int, str]]:
    import re

    top_level = re.compile(r"^ = [^=\n]+ = \n?$")
    articles: list[tuple[int, str]] = []
    current: list[str] = []
    start_index = 0
    for row_index, row in enumerate(rows):
        if top_level.fullmatch(row) and any(part.strip() for part in current):
            articles.append((start_index, normalize_document_text("".join(current))))
            current = []
            start_index = row_index
        current.append(row)
    if any(part.strip() for part in current):
        articles.append((start_index, normalize_document_text("".join(current))))
    return [(index, text) for index, text in articles if text]


def load_source_documents(dataset_name: str) -> tuple[list[tuple[int, str]], dict[str, Any], list[str]]:
    if dataset_name == "wikitext2":
        dataset = load_dataset(
            "Salesforce/wikitext",
            "wikitext-2-raw-v1",
            revision=WIKITEXT_REVISION,
            split="test",
        )
        documents = parse_wikitext_articles(list(dataset["text"]))
        metadata = {
            "name": dataset_name,
            "hf_dataset": "Salesforce/wikitext",
            "config": "wikitext-2-raw-v1",
            "split": "test",
            "revision": WIKITEXT_REVISION,
            "fingerprint": dataset._fingerprint,
            "document_unit": "top_level_article",
            "source_row_count": len(dataset),
            "source_document_count": len(documents),
            "cache_files": [
                {
                    "sha256": file_sha256(item["filename"]),
                    "size": Path(item["filename"]).stat().st_size,
                }
                for item in dataset.cache_files
            ],
        }
        return documents, metadata, list(dataset["text"])

    if dataset_name == "c4_new":
        from huggingface_hub import snapshot_download

        snapshot = Path(snapshot_download(
            repo_id="allenai/c4",
            repo_type="dataset",
            revision=C4_REVISION,
            allow_patterns=[C4_DATA_FILE],
            local_files_only=True,
        ))
        source_file = snapshot / C4_DATA_FILE
        dataset = load_dataset(
            "json",
            data_files={"validation": str(source_file)},
            split="validation",
        )
        documents = [
            (row_index, normalize_document_text(text))
            for row_index, text in enumerate(dataset["text"])
            if normalize_document_text(text)
        ]
        metadata = {
            "name": dataset_name,
            "hf_dataset": "allenai/c4",
            "config": None,
            "split": "validation",
            "data_file": C4_DATA_FILE,
            "revision": C4_REVISION,
            "fingerprint": dataset._fingerprint,
            "document_unit": "dataset_row",
            "source_row_count": len(dataset),
            "source_document_count": len(documents),
            "source_file_sha256": file_sha256(source_file),
            "source_file_size": source_file.stat().st_size,
            "cache_files": [
                {
                    "sha256": file_sha256(item["filename"]),
                    "size": Path(item["filename"]).stat().st_size,
                }
                for item in dataset.cache_files
            ],
        }
        return documents, metadata, []

    raise ValueError(f"Unsupported preregistered dataset: {dataset_name}")


def pilot_excluded_wikitext_documents(
    documents: list[tuple[int, str]],
    legacy_rows: list[str],
    tokenizer,
) -> set[str]:
    legacy_prefix = "\n\n".join(legacy_rows)
    prefix_ids = tokenizer(
        legacy_prefix,
        add_special_tokens=False,
        verbose=False,
    )["input_ids"][:PILOT_WIKITEXT_TOKEN_LIMIT]
    if not prefix_ids:
        return set()

    excluded: set[str] = set()
    cumulative = ""
    identity = "Salesforce/wikitext|wikitext-2-raw-v1"
    for _, text in documents:
        cumulative = text if not cumulative else cumulative + "\n\n" + text
        excluded.add(document_id(identity, "test", text))
        token_count = len(tokenizer(cumulative, add_special_tokens=False, verbose=False)["input_ids"])
        if token_count >= PILOT_WIKITEXT_TOKEN_LIMIT:
            break
    return excluded


def _candidate_sort_key(doc_id: str, start: int, prompt_length: int, continuation_length: int) -> str:
    payload = f"{doc_id}|{start}|{prompt_length}|{continuation_length}".encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _retain_candidate(
    heaps: dict[tuple[str, int, int], list[tuple[int, int, dict[str, Any]]]],
    bucket: tuple[str, int, int],
    candidate: dict[str, Any],
    quota: int,
    serial: int,
) -> None:
    heap = heaps[bucket]
    sort_value = int(candidate["selection_sha256"], 16)
    item = (-sort_value, serial, candidate)
    if len(heap) < quota:
        heapq.heappush(heap, item)
    elif sort_value < -heap[0][0]:
        heapq.heapreplace(heap, item)


def enumerate_document_candidates(
    dataset_name: str,
    source_index: int,
    doc_id: str,
    partition: str,
    token_ids: list[int],
) -> list[dict[str, Any]]:
    rotation = int(doc_id[:8], 16) % len(LENGTH_CELLS)
    per_cell = Counter()
    candidates = []
    position = 0

    for request_slot in range(MAX_REQUESTS_PER_DOCUMENT):
        preferred = (rotation + request_slot) % len(LENGTH_CELLS)
        choices = [
            (preferred + offset) % len(LENGTH_CELLS)
            for offset in range(len(LENGTH_CELLS))
        ]
        chosen = None
        for cell_index in choices:
            if per_cell[cell_index] >= MAX_REQUESTS_PER_CELL_DOCUMENT:
                continue
            prompt_length, continuation_length = LENGTH_CELLS[cell_index]
            if position + prompt_length + continuation_length <= len(token_ids):
                chosen = cell_index
                break
        if chosen is None:
            break

        prompt_length, continuation_length = LENGTH_CELLS[chosen]
        length = prompt_length + continuation_length
        full_ids = token_ids[position:position + length]
        selection_hash = _candidate_sort_key(doc_id, position, prompt_length, continuation_length)
        candidates.append({
            "dataset": dataset_name,
            "source_index": int(source_index),
            "document_id": doc_id,
            "partition": partition,
            "start_token": int(position),
            "end_token": int(position + length),
            "prompt_length_tokens": int(prompt_length),
            "continuation_length_tokens": int(continuation_length),
            "selection_sha256": selection_hash,
            "_token_ids": full_ids,
        })
        per_cell[chosen] += 1
        position += length + TOKEN_GAP

    return candidates


def _validate_request_intervals(requests: list[dict[str, Any]]) -> None:
    by_document: dict[str, list[tuple[int, int]]] = defaultdict(list)
    document_partitions: dict[str, set[str]] = defaultdict(set)
    for request in requests:
        doc_id = request["document_id"]
        by_document[doc_id].append((request["start_token"], request["end_token"]))
        document_partitions[doc_id].add(request["partition"])

    if any(len(partitions) != 1 for partitions in document_partitions.values()):
        raise ValueError("A document appears in more than one partition")

    for doc_id, intervals in by_document.items():
        ordered = sorted(intervals)
        for previous, current in zip(ordered, ordered[1:]):
            if current[0] - previous[1] < TOKEN_GAP:
                raise ValueError(f"Overlapping or insufficiently separated requests in document {doc_id}")


def build_dataset_manifest(dataset_name: str, tokenizer) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    documents, dataset_metadata, legacy_rows = load_source_documents(dataset_name)
    identity = dataset_metadata["hf_dataset"]
    if dataset_metadata.get("config"):
        identity += "|" + str(dataset_metadata["config"])
    split = dataset_metadata["split"]

    excluded = (
        pilot_excluded_wikitext_documents(documents, legacy_rows, tokenizer)
        if dataset_name == "wikitext2"
        else set()
    )

    heaps: dict[tuple[str, int, int], list[tuple[int, int, dict[str, Any]]]] = defaultdict(list)
    serial = 0
    partition_document_counts = Counter()
    eligible_document_counts = Counter()

    for source_index, text in documents:
        doc_id = document_id(identity, split, text)
        if doc_id in excluded:
            continue
        partition = document_partition(doc_id)
        partition_document_counts[partition] += 1
        token_ids = tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"]
        candidates = enumerate_document_candidates(
            dataset_name, source_index, doc_id, partition, token_ids
        )
        if candidates:
            eligible_document_counts[partition] += 1
        for candidate in candidates:
            bucket = (
                partition,
                candidate["prompt_length_tokens"],
                candidate["continuation_length_tokens"],
            )
            _retain_candidate(heaps, bucket, candidate, PARTITION_QUOTAS[partition], serial)
            serial += 1

    selected = []
    quota_counts = {}
    for partition, quota in PARTITION_QUOTAS.items():
        for prompt_length, continuation_length in LENGTH_CELLS:
            bucket = (partition, prompt_length, continuation_length)
            candidates = [item[2] for item in heaps[bucket]]
            if len(candidates) != quota:
                raise RuntimeError(
                    f"{dataset_name} cannot fill {bucket}: {len(candidates)} candidates for quota {quota}"
                )
            quota_counts[f"{partition}:{prompt_length}p:{continuation_length}c"] = len(candidates)
            selected.extend(candidates)

    selected.sort(key=lambda item: (
        PARTITION_ORDER[item["partition"]],
        item["prompt_length_tokens"],
        item["continuation_length_tokens"],
        item["selection_sha256"],
    ))
    _validate_request_intervals(selected)

    input_tensors: dict[str, torch.Tensor] = {}
    requests = []
    for request_index, candidate in enumerate(selected):
        token_ids = candidate.pop("_token_ids")
        prompt_length = candidate["prompt_length_tokens"]
        prompt_ids = token_ids[:prompt_length]
        continuation_ids = token_ids[prompt_length:]
        request_id = (
            f"{dataset_name}-{candidate['partition'][:3]}-"
            f"{prompt_length}p{candidate['continuation_length_tokens']}c-"
            f"{candidate['selection_sha256'][:16]}"
        )
        candidate.update({
            "request_id": request_id,
            "request_index": request_index,
            "prompt_token_sha256": token_ids_sha256(prompt_ids),
            "continuation_token_sha256": token_ids_sha256(continuation_ids),
            "request_token_sha256": token_ids_sha256(token_ids),
        })
        input_tensors[request_id] = torch.tensor(token_ids, dtype=torch.long)
        requests.append(candidate)

    subset_hash = hashlib.sha256(
        "".join(request["request_token_sha256"] for request in requests).encode("ascii")
    ).hexdigest()
    manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "dataset": dataset_metadata,
        "selection": {
            "split_salt": SPLIT_SALT,
            "partition_residues": {
                "development": [0, 1, 2, 3],
                "calibration": [4],
                "test": [5, 6, 7, 8],
            },
            "length_cells": [list(cell) for cell in LENGTH_CELLS],
            "partition_quota_per_cell": PARTITION_QUOTAS,
            "token_gap": TOKEN_GAP,
            "max_requests_per_document": MAX_REQUESTS_PER_DOCUMENT,
            "max_requests_per_cell_document": MAX_REQUESTS_PER_CELL_DOCUMENT,
            "pilot_wikitext_token_limit": (
                PILOT_WIKITEXT_TOKEN_LIMIT if dataset_name == "wikitext2" else None
            ),
            "excluded_pilot_document_ids": sorted(excluded),
            "partition_document_counts": dict(partition_document_counts),
            "eligible_document_counts": dict(eligible_document_counts),
            "quota_counts": quota_counts,
        },
        "tokenizer": {
            "name_or_path": str(tokenizer.name_or_path),
            "class": type(tokenizer).__name__,
            "vocab_size": int(tokenizer.vocab_size),
            "add_special_tokens": False,
        },
        "request_count": len(requests),
        "subset_token_sha256": subset_hash,
        "contains_raw_text": False,
        "requests": requests,
    }
    manifest["manifest_sha256"] = object_sha256(manifest)
    return manifest, input_tensors


def ensure_manifest(path: Path, manifest: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("manifest_sha256") != manifest["manifest_sha256"] or existing != manifest:
            raise RuntimeError(f"Existing manifest differs from deterministic rebuild: {path}")
        return
    atomic_write_json(path, manifest)


def _walk_keys(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def validate_record(
    record: dict[str, Any],
    request: dict[str, Any],
    manifest_sha256: str,
    expected_modes: list[str],
) -> None:
    if record.get("schema_version") != RECORD_SCHEMA_VERSION:
        raise ValueError("Unexpected request-demand record schema")
    if record.get("request_id") != request["request_id"]:
        raise ValueError("Shard request order or identity mismatch")
    if record.get("manifest_sha256") != manifest_sha256:
        raise ValueError("Record manifest hash mismatch")
    for key in ("prompt_token_sha256", "continuation_token_sha256", "request_token_sha256"):
        if record.get(key) != request[key]:
            raise ValueError(f"Record {key} mismatch")
    if record.get("prompt_length_tokens") != request["prompt_length_tokens"]:
        raise ValueError("Prompt length mismatch")
    if record.get("continuation_length_tokens") != request["continuation_length_tokens"]:
        raise ValueError("Continuation length mismatch")
    if any(key in FORBIDDEN_ARTIFACT_KEYS for key in _walk_keys(record)):
        raise ValueError("Raw text or token payload key found in artifact")

    quality = record.get("quality_by_mode", {})
    if set(quality) != set(expected_modes):
        raise ValueError(f"Mode mismatch: {sorted(quality)}")
    continuation_length = request["continuation_length_tokens"]
    for mode, metrics in quality.items():
        if metrics.get("target_token_count") != continuation_length:
            raise ValueError(f"{mode} does not report continuation-only target count")
        for key in ("mean_nll", "perplexity", "nll_delta_vs_fixed_high"):
            if not math.isfinite(float(metrics[key])):
                raise ValueError(f"Non-finite {mode}.{key}")
        if not metrics.get("finite_logits"):
            raise ValueError(f"Non-finite logits in {mode}")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open(encoding="ascii") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError(f"Blank line in {path}:{line_number}")
            records.append(json.loads(line))
    return records


def validate_shard(
    shard_path: Path,
    expected_requests: list[dict[str, Any]],
    manifest_sha256: str,
    expected_modes: list[str],
) -> dict[str, Any]:
    records = read_jsonl(shard_path)
    if len(records) != len(expected_requests):
        raise ValueError(
            f"Shard {shard_path} has {len(records)} records; expected {len(expected_requests)}"
        )
    for record, request in zip(records, expected_requests):
        validate_record(record, request, manifest_sha256, expected_modes)
    return {
        "shard_schema_version": SHARD_SCHEMA_VERSION,
        "validation_status": "VALIDATED",
        "path": str(shard_path),
        "record_count": len(records),
        "request_ids": [record["request_id"] for record in records],
        "sha256": file_sha256(shard_path),
        "manifest_sha256": manifest_sha256,
    }


def ensure_shard_metadata(
    shard_path: Path,
    expected_requests: list[dict[str, Any]],
    manifest_sha256: str,
    expected_modes: list[str],
) -> dict[str, Any]:
    metadata = validate_shard(shard_path, expected_requests, manifest_sha256, expected_modes)
    metadata_path = shard_path.with_suffix(".meta.json")
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text())
        comparable = {key: existing.get(key) for key in metadata}
        if comparable != metadata:
            raise RuntimeError(f"Shard metadata mismatch: {metadata_path}")
    else:
        atomic_write_json(metadata_path, metadata)
    return metadata


def make_collection_record(
    raw_record: dict[str, Any],
    request: dict[str, Any],
    manifest_sha256: str,
    dataset_metadata: dict[str, Any],
) -> dict[str, Any]:
    record = dict(raw_record)
    record.update({
        "schema_version": RECORD_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "manifest_sha256": manifest_sha256,
        "request_id": request["request_id"],
        "request_index": request["request_index"],
        "source": {
            "dataset": request["dataset"],
            "dataset_revision": dataset_metadata["revision"],
            "dataset_fingerprint": dataset_metadata["fingerprint"],
            "source_split": dataset_metadata["split"],
            "source_index": request["source_index"],
            "document_id": request["document_id"],
            "partition": request["partition"],
            "start_token": request["start_token"],
            "end_token": request["end_token"],
            "selection_sha256": request["selection_sha256"],
        },
        "quality_scope": "continuation_only_teacher_forced",
        "contains_raw_text": False,
    })
    return record


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "request_count": 0,
            "continuation_token_count": 0,
            "minimum_safe_precision_counts": {},
            "quality_by_mode": {},
        }

    continuation_tokens = sum(record["continuation_length_tokens"] for record in records)
    demand = Counter(record["minimum_safe_precision"]["requested_bit"] for record in records)
    modes = sorted(records[0]["quality_by_mode"])
    quality = {}
    for mode in modes:
        weighted_nll = sum(
            record["quality_by_mode"][mode]["mean_nll"]
            * record["quality_by_mode"][mode]["target_token_count"]
            for record in records
        ) / continuation_tokens
        weighted_delta = sum(
            record["quality_by_mode"][mode]["nll_delta_vs_fixed_high"]
            * record["quality_by_mode"][mode]["target_token_count"]
            for record in records
        ) / continuation_tokens
        quality[mode] = {
            "continuation_token_weighted_mean_nll": weighted_nll,
            "continuation_token_weighted_perplexity": math.exp(weighted_nll),
            "continuation_token_weighted_nll_delta_vs_fixed_high": weighted_delta,
            "mean_effective_bits": sum(
                record["quality_by_mode"][mode]["effective_bits"] for record in records
            ) / len(records),
            "mean_average_selected_bit": sum(
                record["quality_by_mode"][mode]["average_selected_bit"] for record in records
            ) / len(records),
            "total_fallbacks": sum(
                record["quality_by_mode"][mode]["fallback_count"] for record in records
            ),
            "total_dp_guard_triggers": sum(
                record["quality_by_mode"][mode]["dp_guard_trigger_count"] for record in records
            ),
            "all_finite_logits": all(
                record["quality_by_mode"][mode]["finite_logits"] for record in records
            ),
        }

    strata = Counter(
        (
            record["source"]["partition"],
            record["prompt_length_tokens"],
            record["continuation_length_tokens"],
        )
        for record in records
    )
    return {
        "request_count": len(records),
        "continuation_token_count": continuation_tokens,
        "minimum_safe_precision_counts": {
            str(bit): int(count) for bit, count in sorted(demand.items())
        },
        "stratum_counts": {
            f"{partition}:{prompt}p:{continuation}c": count
            for (partition, prompt, continuation), count in sorted(strata.items())
        },
        "quality_scope": "continuation_only_teacher_forced",
        "quality_by_mode": quality,
    }


def environment_metadata(device: torch.device | None) -> dict[str, Any]:
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        check=False,
    )
    metadata = {
        "created_at": utc_now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "accelerate": accelerate.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_driver": driver.stdout.splitlines()[0].strip() if driver.returncode == 0 else None,
        "command": [sys.executable, *sys.argv],
    }
    if device is not None and device.type == "cuda":
        metadata["cuda_device"] = torch.cuda.get_device_name(device)
        metadata["cuda_capability"] = list(torch.cuda.get_device_capability(device))
    return metadata


def build_run_manifest(
    args,
    manifests: dict[str, dict[str, Any]],
    source_provenance: dict[str, Any],
    device: torch.device | None,
) -> dict[str, Any]:
    try:
        import any_precision_ext

        extension_path = Path(any_precision_ext.__file__).resolve()
        extension = {
            "path": str(extension_path),
            "size": extension_path.stat().st_size,
            "sha256": file_sha256(extension_path),
        }
    except Exception as exc:
        extension = {"status": "UNAVAILABLE", "error": type(exc).__name__}

    artifact_manifests = {
        "ap_model": file_manifest(args.ap_model_path),
        "router_checkpoint": file_manifest(args.router_checkpoint),
        "estimator_results": file_manifest(args.estimator_results),
        "tokenizer": tokenizer_file_manifest(args.tokenizer_path or args.ap_model_path),
        "any_precision_extension": extension,
    }
    stable_config = {
        "protocol_version": PROTOCOL_VERSION,
        "datasets": list(args.datasets),
        "candidate_bits": list(args.bits or [3, 4, 5, 6]),
        "qaq_modes": list(args.qaq_modes),
        "safe_nll_delta": args.safe_nll_delta,
        "profile_layer_group_size": args.profile_layer_group_size,
        "confidence_threshold": args.confidence_threshold,
        "fallback_bits": args.fallback_bits,
        "torch_dtype": args.torch_dtype,
        "shard_size": args.shard_size,
        "manifest_hashes": {
            name: manifest["manifest_sha256"] for name, manifest in manifests.items()
        },
        "source_files_sha256": source_provenance["source_files_sha256"],
        "input_tree_hashes": {
            name: value.get("tree_sha256", value.get("files", [{}])[0].get("sha256"))
            if isinstance(value, dict)
            else None
            for name, value in artifact_manifests.items()
        },
    }
    return {
        "run_manifest_schema_version": "qaq_request_demand_run_v1",
        "validation_status": "PREFLIGHT_VALIDATED",
        "collection_id": object_sha256(stable_config),
        "config": stable_config,
        "environment": environment_metadata(device),
        "source_provenance": source_provenance,
        "input_artifacts": artifact_manifests,
        "contains_raw_text": False,
    }


def ensure_run_manifest(path: Path, run_manifest: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("collection_id") != run_manifest["collection_id"]:
            raise RuntimeError(
                "Existing output directory belongs to a different collection configuration"
            )
        return existing
    atomic_write_json(path, run_manifest)
    return run_manifest


def expected_mode_names(bits: list[int], qaq_modes: list[str]) -> list[str]:
    names = ["fixed_low", *[f"fixed_{bit}" for bit in bits[1:-1]], "fixed_high"]
    return [*names, *qaq_modes]


def _shard_slices(requests: list[dict[str, Any]], shard_size: int):
    for start in range(0, len(requests), shard_size):
        yield start // shard_size, requests[start:start + shard_size]


def validate_dataset_shards(
    dataset_dir: Path,
    manifest: dict[str, Any],
    expected_modes: list[str],
    shard_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = []
    shard_metadata = []
    shard_dir = dataset_dir / "shards"
    for shard_index, expected_requests in _shard_slices(manifest["requests"], shard_size):
        shard_path = shard_dir / f"shard-{shard_index:05d}.jsonl"
        if not shard_path.exists():
            continue
        metadata = ensure_shard_metadata(
            shard_path,
            expected_requests,
            manifest["manifest_sha256"],
            expected_modes,
        )
        shard_metadata.append(metadata)
        records.extend(read_jsonl(shard_path))
    return records, shard_metadata


def summary_provenance(run_manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "collection_id": run_manifest["collection_id"],
        "git_commit": run_manifest["source_provenance"].get("git_commit"),
        "git_worktree_dirty": run_manifest["source_provenance"].get("git_worktree_dirty"),
        "source_files_sha256": run_manifest["source_provenance"]["source_files_sha256"],
        "environment": run_manifest["environment"],
        "input_tree_hashes": run_manifest["config"]["input_tree_hashes"],
    }


def write_dataset_summary(
    dataset_dir: Path,
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    shard_metadata: list[dict[str, Any]],
    run_manifest: dict[str, Any],
) -> dict[str, Any]:
    complete = len(records) == manifest["request_count"]
    summary = {
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "validation_status": (
            "REAL_GPU_REQUEST_DEMAND_COMPLETE" if complete else "REAL_GPU_REQUEST_DEMAND_IN_PROGRESS"
        ),
        "protocol_version": PROTOCOL_VERSION,
        "collection_id": run_manifest["collection_id"],
        "dataset": manifest["dataset"],
        "manifest_sha256": manifest["manifest_sha256"],
        "subset_token_sha256": manifest["subset_token_sha256"],
        "expected_request_count": manifest["request_count"],
        "validated_shard_count": len(shard_metadata),
        "shards": shard_metadata,
        "aggregate": aggregate_records(records),
        "provenance": summary_provenance(run_manifest),
        "contains_raw_text": False,
    }
    atomic_write_json(dataset_dir / "summary.json", summary)
    return summary


def write_combined_summary(
    output_dir: Path,
    dataset_summaries: dict[str, dict[str, Any]],
    all_records: list[dict[str, Any]],
    run_manifest: dict[str, Any],
) -> dict[str, Any]:
    complete = (
        set(dataset_summaries) == set(run_manifest["config"]["datasets"])
        and all(
            summary["validation_status"] == "REAL_GPU_REQUEST_DEMAND_COMPLETE"
            for summary in dataset_summaries.values()
        )
    )
    combined = {
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "validation_status": (
            "REAL_GPU_REQUEST_DEMAND_COMPLETE" if complete else "REAL_GPU_REQUEST_DEMAND_IN_PROGRESS"
        ),
        "protocol_version": PROTOCOL_VERSION,
        "collection_id": run_manifest["collection_id"],
        "dataset_summaries": {
            name: {
                "path": str(output_dir / "datasets" / name / "summary.json"),
                "validation_status": summary["validation_status"],
                "manifest_sha256": summary["manifest_sha256"],
                "request_count": summary["aggregate"]["request_count"],
                "continuation_token_count": summary["aggregate"]["continuation_token_count"],
            }
            for name, summary in sorted(dataset_summaries.items())
        },
        "aggregate": aggregate_records(all_records),
        "provenance": summary_provenance(run_manifest),
        "contains_raw_text": False,
    }
    atomic_write_json(output_dir / "combined-summary.json", combined)
    return combined


def refresh_summaries(
    output_dir: Path,
    manifests: dict[str, dict[str, Any]],
    expected_modes: list[str],
    shard_size: int,
    run_manifest: dict[str, Any],
) -> dict[str, Any]:
    dataset_summaries = {}
    all_records = []
    for dataset_name, manifest in manifests.items():
        dataset_dir = output_dir / "datasets" / dataset_name
        records, shard_metadata = validate_dataset_shards(
            dataset_dir, manifest, expected_modes, shard_size
        )
        dataset_summaries[dataset_name] = write_dataset_summary(
            dataset_dir, manifest, records, shard_metadata, run_manifest
        )
        all_records.extend(records)
    return write_combined_summary(output_dir, dataset_summaries, all_records, run_manifest)


def validate_output_directory(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    artifacts_root = (Path.cwd() / "artifacts").resolve()
    if resolved == artifacts_root:
        raise ValueError("--output_dir must be a dedicated subdirectory, not artifacts/")
    if resolved.name in PILOT_FILENAMES:
        raise ValueError("Refusing to use an existing pilot artifact as output")
    for pilot in PILOT_FILENAMES:
        if resolved == (artifacts_root / pilot).resolve():
            raise ValueError("Refusing to overwrite the 32-request pilot")


def run_preregistered_collection(
    args,
    router,
    checkpoint: dict[str, Any],
    collect_request: Callable[..., dict[str, Any]],
    source_provenance: dict[str, Any],
) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    validate_output_directory(output_dir)
    tokenizer_path = args.tokenizer_path or args.ap_model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )

    manifests = {}
    input_tensors = {}
    for dataset_name in args.datasets:
        manifest, tensors = build_dataset_manifest(dataset_name, tokenizer)
        manifest_path = output_dir / "manifests" / f"{dataset_name}.json"
        ensure_manifest(manifest_path, manifest)
        manifests[dataset_name] = manifest
        input_tensors[dataset_name] = tensors

    bits = sorted(int(bit) for bit in checkpoint["candidate_bits"])
    modes = expected_mode_names(bits, list(args.qaq_modes))

    if args.manifest_only:
        result = {
            "validation_status": "MANIFESTS_VALIDATED",
            "output_dir": str(output_dir),
            "manifests": {
                name: {
                    "sha256": manifest["manifest_sha256"],
                    "request_count": manifest["request_count"],
                    "dataset_revision": manifest["dataset"]["revision"],
                    "dataset_fingerprint": manifest["dataset"]["fingerprint"],
                }
                for name, manifest in manifests.items()
            },
        }
        atomic_write_json(output_dir / "manifest-preflight-summary.json", result)
        return result

    device = None if args.validate_only else torch.device(args.device)
    run_manifest = build_run_manifest(args, manifests, source_provenance, device)
    run_manifest = ensure_run_manifest(output_dir / "run-manifest.json", run_manifest)

    if args.validate_only:
        return refresh_summaries(output_dir, manifests, modes, args.shard_size, run_manifest)

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

    for dataset_name in args.datasets:
        manifest = manifests[dataset_name]
        dataset_dir = output_dir / "datasets" / dataset_name
        shard_dir = dataset_dir / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)

        for shard_index, expected_requests in _shard_slices(
            manifest["requests"], args.shard_size
        ):
            shard_path = shard_dir / f"shard-{shard_index:05d}.jsonl"
            if shard_path.exists():
                ensure_shard_metadata(
                    shard_path, expected_requests, manifest["manifest_sha256"], modes
                )
                print(
                    f"{dataset_name} shard {shard_index:05d}: validated existing shard",
                    flush=True,
                )
                continue

            records = []
            temporary = shard_path.with_suffix(shard_path.suffix + ".tmp")
            if temporary.exists():
                temporary.unlink()
            for request in expected_requests:
                full_ids = input_tensors[dataset_name][request["request_id"]]
                prompt_length = request["prompt_length_tokens"]
                raw = collect_request(
                    model=model,
                    tokenizer=tokenizer,
                    request_index=request["request_index"],
                    prompt_ids=full_ids[:prompt_length],
                    continuation_ids=full_ids[prompt_length:],
                    full_ids=full_ids,
                    bits=bits,
                    qaq_modes=list(args.qaq_modes),
                    safe_nll_delta=args.safe_nll_delta,
                    layer_group_size=args.profile_layer_group_size,
                    device=device,
                )
                records.append(
                    make_collection_record(
                        raw, request, manifest["manifest_sha256"], manifest["dataset"]
                    )
                )
                print(
                    f"{dataset_name} shard {shard_index:05d} "
                    f"{len(records)}/{len(expected_requests)} {request['request_id']}",
                    flush=True,
                )

            with temporary.open("w", encoding="ascii") as out:
                for record in records:
                    out.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")
                out.flush()
                os.fsync(out.fileno())
            validate_shard(
                temporary, expected_requests, manifest["manifest_sha256"], modes
            )
            os.replace(temporary, shard_path)
            ensure_shard_metadata(
                shard_path, expected_requests, manifest["manifest_sha256"], modes
            )
            print(f"{dataset_name} shard {shard_index:05d}: committed", flush=True)
            refresh_summaries(output_dir, manifests, modes, args.shard_size, run_manifest)

    return refresh_summaries(output_dir, manifests, modes, args.shard_size, run_manifest)
