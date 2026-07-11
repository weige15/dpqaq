"""Preregistered offline analysis for the immutable large request-demand collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import scipy
import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, cross_val_predict

from scripts.qaq_request_demand_protocol import (
    PROTOCOL_VERSION,
    RECORD_SCHEMA_VERSION,
    expected_mode_names,
    file_sha256,
    object_sha256,
    read_jsonl,
    validate_shard,
)

ANALYSIS_SCHEMA_VERSION = "qaq_request_demand_preregistered_analysis_v1"
PREDICTOR_SEEDS = (17, 29, 43)
BOOTSTRAP_SEED = 1729
DYNAMIC_MODES = ("dp_threshold_only", "mlp_multibit", "mlp_multibit_dp_guard")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen document-grouped request-demand analysis.")
    parser.add_argument("--collection_dir", required=True)
    parser.add_argument("--freeze_manifest", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--bootstrap_replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap_seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--predictor_seeds", type=int, nargs="+", default=list(PREDICTOR_SEEDS))
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--group_folds", type=int, default=5)
    return parser.parse_args()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def collection_file_manifest(root: Path) -> tuple[list[dict[str, Any]], str]:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        files.append({
            "path": str(path.relative_to(root)),
            "size": path.stat().st_size,
            "sha256": file_sha256(path),
        })
    return files, hashlib.sha256(canonical_json_bytes(files)).hexdigest()


def verify_freeze(collection_dir: Path, freeze_path: Path) -> dict[str, Any]:
    freeze = json.loads(freeze_path.read_text())
    files, tree_hash = collection_file_manifest(collection_dir)
    if freeze.get("validation_status") != "REAL_GPU_REQUEST_DEMAND_COMPLETE":
        raise RuntimeError("Freeze manifest does not describe a complete collection")
    if files != freeze.get("files") or tree_hash != freeze.get("collection_tree_sha256"):
        raise RuntimeError("Collection differs from the recursively frozen file manifest")
    return freeze


def load_collection(collection_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_manifest = json.loads((collection_dir / "run-manifest.json").read_text())
    combined = json.loads((collection_dir / "combined-summary.json").read_text())
    if combined.get("validation_status") != "REAL_GPU_REQUEST_DEMAND_COMPLETE":
        raise RuntimeError("Collection summary is not complete")
    if combined.get("collection_id") != run_manifest.get("collection_id"):
        raise RuntimeError("Collection ID mismatch")

    bits = sorted(int(bit) for bit in run_manifest["config"]["candidate_bits"])
    modes = expected_mode_names(bits, list(run_manifest["config"]["qaq_modes"]))
    shard_size = int(run_manifest["config"]["shard_size"])
    records: list[dict[str, Any]] = []
    manifests = {}
    for dataset_name in run_manifest["config"]["datasets"]:
        manifest_path = collection_dir / "manifests" / f"{dataset_name}.json"
        manifest = json.loads(manifest_path.read_text())
        stored_hash = manifest["manifest_sha256"]
        unhashed = dict(manifest)
        unhashed.pop("manifest_sha256")
        if object_sha256(unhashed) != stored_hash:
            raise RuntimeError(f"Manifest content hash mismatch: {dataset_name}")
        if stored_hash != run_manifest["config"]["manifest_hashes"][dataset_name]:
            raise RuntimeError(f"Run-manifest hash mismatch: {dataset_name}")
        manifests[dataset_name] = manifest
        dataset_records = []
        for start in range(0, len(manifest["requests"]), shard_size):
            index = start // shard_size
            expected = manifest["requests"][start:start + shard_size]
            shard = collection_dir / "datasets" / dataset_name / "shards" / f"shard-{index:05d}.jsonl"
            meta_path = shard.with_suffix(".meta.json")
            if not shard.exists() or not meta_path.exists():
                raise RuntimeError(f"Missing shard or sidecar: {shard}")
            actual_meta = validate_shard(shard, expected, stored_hash, modes)
            stored_meta = json.loads(meta_path.read_text())
            substantive_keys = [key for key in actual_meta if key != "path"]
            if any(stored_meta.get(key) != actual_meta[key] for key in substantive_keys):
                raise RuntimeError(f"Shard sidecar mismatch: {meta_path}")
            stored_shard_path = Path(stored_meta["path"])
            if not stored_shard_path.is_absolute():
                stored_shard_path = REPO_ROOT / stored_shard_path
            if stored_shard_path.resolve() != shard.resolve():
                raise RuntimeError(f"Shard sidecar path mismatch: {meta_path}")
            dataset_records.extend(read_jsonl(shard))
        if len(dataset_records) != manifest["request_count"]:
            raise RuntimeError(f"Record count mismatch: {dataset_name}")
        records.extend(dataset_records)
    return records, {"run_manifest": run_manifest, "combined_summary": combined, "manifests": manifests}


def feature_matrix(records: list[dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    names = sorted(records[0]["prompt_features"])
    if any(sorted(record["prompt_features"]) != names for record in records):
        raise ValueError("Prompt feature schemas differ")
    matrix = np.asarray([[float(record["prompt_features"][name]) for name in names] for record in records])
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite prompt feature")
    return matrix, names


def regression_metrics(target: np.ndarray, prediction: np.ndarray, multioutput: str = "uniform_average") -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": float(math.sqrt(mean_squared_error(target, prediction))),
        "r2": float(r2_score(target, prediction, multioutput=multioutput)),
    }


def classification_metrics(labels: np.ndarray, predictions: np.ndarray, classes: list[int]) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", labels=classes, zero_division=0)),
        "confusion_matrix_labels": classes,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=classes).tolist(),
    }


def expected_calibration_error(labels: np.ndarray, predictions: np.ndarray, confidence: np.ndarray) -> float:
    correct = predictions == labels
    total = len(labels)
    error = 0.0
    for low in np.linspace(0.0, 0.9, 10):
        high = low + 0.1
        mask = (confidence >= low) & (confidence < high if high < 1.0 else confidence <= high)
        if mask.any():
            error += mask.sum() / total * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(error)


def cluster_bootstrap(
    records: list[dict[str, Any]],
    statistic: Callable[[list[dict[str, Any]]], float],
    replicates: int,
    seed: int,
) -> np.ndarray:
    by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = f"{record['source']['dataset']}:{record['source']['document_id']}"
        by_document[key].append(record)
    clusters = list(by_document.values())
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = rng.integers(0, len(clusters), size=len(clusters))
        replicate = [record for cluster_index in sampled for record in clusters[cluster_index]]
        values[index] = statistic(replicate)
    return values


def safe_entropy(values: list[int]) -> float:
    counts = np.asarray(list(Counter(values).values()), dtype=np.float64)
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log(probabilities)).sum())


def demand_summary(records: list[dict[str, Any]], replicates: int, seed: int) -> dict[str, Any]:
    bits = [int(record["minimum_safe_precision"]["requested_bit"]) for record in records]
    distribution = Counter(bits)
    bootstrap = cluster_bootstrap(
        records,
        lambda sample: float(np.std([r["minimum_safe_precision"]["requested_bit"] for r in sample], ddof=0)),
        replicates,
        seed,
    )
    material_classes = sum(count / len(bits) >= 0.10 for count in distribution.values())
    return {
        "request_count": len(records),
        "document_count": len({record["source"]["document_id"] for record in records}),
        "minimum_safe_precision_counts": {str(bit): int(count) for bit, count in sorted(distribution.items())},
        "entropy_nats": safe_entropy(bits),
        "standard_deviation": float(np.std(bits, ddof=0)),
        "document_cluster_bootstrap_std_ci95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "at_least_two_classes_at_ten_percent": material_classes >= 2,
        "h1_pass": bool(material_classes >= 2 and np.quantile(bootstrap, 0.025) > 0.0),
    }


def quality_summary(records: list[dict[str, Any]], replicates: int, seed: int) -> dict[str, Any]:
    modes = sorted(records[0]["quality_by_mode"])
    output = {}
    for mode_index, mode in enumerate(modes):
        deltas = np.asarray([record["quality_by_mode"][mode]["nll_delta_vs_fixed_high"] for record in records])
        failures = deltas > 0.02
        mean_boot = cluster_bootstrap(
            records,
            lambda sample, m=mode: float(np.mean([r["quality_by_mode"][m]["nll_delta_vs_fixed_high"] for r in sample])),
            replicates,
            seed + 100 * mode_index,
        )
        failure_boot = cluster_bootstrap(
            records,
            lambda sample, m=mode: float(np.mean([r["quality_by_mode"][m]["nll_delta_vs_fixed_high"] > 0.02 for r in sample])),
            replicates,
            seed + 100 * mode_index + 1,
        )
        mean_delta = float(deltas.mean())
        failure_fraction = float(failures.mean())
        metrics = {
            "request_count": len(records),
            "mean_nll_delta_vs_fixed_high": mean_delta,
            "mean_nll_delta_one_sided_95_upper": float(np.quantile(mean_boot, 0.95)),
            "unsafe_request_fraction": failure_fraction,
            "unsafe_request_fraction_one_sided_95_upper": float(np.quantile(failure_boot, 0.95)),
            "mean_effective_bits": float(np.mean([record["quality_by_mode"][mode]["effective_bits"] for record in records])),
            "all_finite_logits": bool(all(record["quality_by_mode"][mode]["finite_logits"] for record in records)),
        }
        metrics["quality_gate_pass"] = bool(
            metrics["all_finite_logits"]
            and metrics["mean_nll_delta_one_sided_95_upper"] <= 0.02
            and failure_fraction <= 0.05
            and metrics["unsafe_request_fraction_one_sided_95_upper"] <= 0.10
        )
        output[mode] = metrics

    guarded = output["mlp_multibit_dp_guard"]
    unguarded = output["mlp_multibit"]
    return {
        "by_mode": output,
        "guard_efficacy_request_quality_pass": bool(
            guarded["unsafe_request_fraction"] <= unguarded["unsafe_request_fraction"]
        ),
        "guard_noncollapse_pass": bool(guarded["mean_effective_bits"] <= 5.90),
        "route_under_precision_gate": {
            "status": "UNAVAILABLE_FROM_COLLECTION",
            "reason": "Records contain executed route profiles but not per-decision real-output required-bit labels.",
        },
        "h3_status": (
            "FAIL_GUARDED_QUALITY_GATE_ROUTE_SAFETY_UNAVAILABLE"
            if not guarded["quality_gate_pass"]
            else "NOT_EVALUABLE_ROUTE_SAFETY_ENDPOINT_MISSING"
        ),
    }


def predictor_target_arrays(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    safe = np.asarray([record["minimum_safe_precision"]["requested_bit"] for record in records], dtype=np.int64)
    effective = np.asarray([record["observed_qaq_profiles"]["mlp_multibit_dp_guard"]["effective_bits"] for record in records])
    profiles = np.asarray([record["observed_qaq_profiles"]["mlp_multibit_dp_guard"]["group_expected_bits"] for record in records])
    return safe, effective, profiles


def evaluate_predictors(
    train_records: list[dict[str, Any]],
    calibration_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    feature_names: list[str],
    seed: int,
    trees: int,
) -> dict[str, Any]:
    train_x, _ = feature_matrix(train_records)
    calibration_x, _ = feature_matrix(calibration_records)
    test_x, _ = feature_matrix(test_records)
    train_safe, train_effective, train_profiles = predictor_target_arrays(train_records)
    calibration_safe, _, _ = predictor_target_arrays(calibration_records)
    test_safe, test_effective, test_profiles = predictor_target_arrays(test_records)
    classes = sorted(set(train_safe.tolist()) | set(test_safe.tolist()))

    classifier = RandomForestClassifier(
        n_estimators=trees, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1
    ).fit(train_x, train_safe)
    calibration_prediction = classifier.predict(calibration_x)
    calibration_confidence = classifier.predict_proba(calibration_x).max(axis=1)
    cutoff = float(np.quantile(calibration_confidence, 0.10, method="lower"))
    test_prediction = classifier.predict(test_x)
    test_confidence = classifier.predict_proba(test_x).max(axis=1)
    majority = min(label for label, count in Counter(train_safe.tolist()).items() if count == max(Counter(train_safe.tolist()).values()))
    baseline_prediction = np.full_like(test_safe, majority)
    classification = classification_metrics(test_safe, test_prediction, classes)
    baseline_classification = classification_metrics(test_safe, baseline_prediction, classes)
    classification.update({
        "constant_baseline": baseline_classification,
        "calibration_ece": expected_calibration_error(calibration_safe, calibration_prediction, calibration_confidence),
        "uncertainty_cutoff_for_90pct_calibration_coverage": cutoff,
        "calibration_coverage": float(np.mean(calibration_confidence >= cutoff)),
        "test_coverage": float(np.mean(test_confidence >= cutoff)),
    })
    classification["pass"] = bool(
        classification["balanced_accuracy"] > baseline_classification["balanced_accuracy"]
        and classification["macro_f1"] > baseline_classification["macro_f1"]
    )

    scalar = RandomForestRegressor(
        n_estimators=trees, min_samples_leaf=2, random_state=seed, n_jobs=-1
    ).fit(train_x, train_effective)
    scalar_prediction = scalar.predict(test_x)
    scalar_baseline_prediction = np.full_like(test_effective, float(train_effective.mean()))
    scalar_metrics = regression_metrics(test_effective, scalar_prediction)
    scalar_baseline = regression_metrics(test_effective, scalar_baseline_prediction)
    scalar_metrics["constant_baseline"] = scalar_baseline
    scalar_metrics["mae_improvement_vs_constant_fraction"] = float(
        (scalar_baseline["mae"] - scalar_metrics["mae"]) / scalar_baseline["mae"]
    ) if scalar_baseline["mae"] else 0.0
    scalar_metrics["pass"] = bool(
        scalar_metrics["mae_improvement_vs_constant_fraction"] >= 0.10 and scalar_metrics["r2"] >= 0.10
    )

    profile = RandomForestRegressor(
        n_estimators=trees, min_samples_leaf=2, random_state=seed, n_jobs=-1
    ).fit(train_x, train_profiles)
    profile_prediction = profile.predict(test_x)
    profile_baseline_prediction = np.repeat(train_profiles.mean(axis=0, keepdims=True), len(test_records), axis=0)
    profile_metrics = regression_metrics(test_profiles, profile_prediction, multioutput="variance_weighted")
    profile_baseline = regression_metrics(test_profiles, profile_baseline_prediction, multioutput="variance_weighted")
    profile_metrics["constant_baseline"] = profile_baseline
    profile_metrics["mae_improvement_vs_constant_fraction"] = float(
        (profile_baseline["mae"] - profile_metrics["mae"]) / profile_baseline["mae"]
    ) if profile_baseline["mae"] else 0.0
    profile_metrics["component_mae"] = np.mean(np.abs(test_profiles - profile_prediction), axis=0).tolist()
    profile_metrics["pass"] = bool(
        profile_metrics["mae_improvement_vs_constant_fraction"] >= 0.10 and profile_metrics["r2"] >= 0.10
    )

    predictions = [
        {
            "request_id": record["request_id"],
            "minimum_safe_precision": int(test_safe[index]),
            "predicted_minimum_safe_precision": int(test_prediction[index]),
            "classification_confidence": float(test_confidence[index]),
            "predicted_effective_bits": float(scalar_prediction[index]),
            "predicted_group_profile": profile_prediction[index].tolist(),
        }
        for index, record in enumerate(test_records)
    ]
    return {
        "minimum_safe_precision_classifier": classification,
        "effective_bits_regressor": scalar_metrics,
        "group_profile_regressor": profile_metrics,
        "all_endpoints_pass": bool(classification["pass"] and scalar_metrics["pass"] and profile_metrics["pass"]),
        "feature_importance": {
            "classifier": dict(zip(feature_names, classifier.feature_importances_.tolist())),
            "effective_bits": dict(zip(feature_names, scalar.feature_importances_.tolist())),
            "group_profile": dict(zip(feature_names, profile.feature_importances_.tolist())),
        },
        "predictions": predictions,
    }


def grouped_development_diagnostic(records: list[dict[str, Any]], seed: int, trees: int, folds: int) -> dict[str, Any]:
    x, _ = feature_matrix(records)
    safe, effective, profiles = predictor_target_arrays(records)
    groups = np.asarray([f"{r['source']['dataset']}:{r['source']['document_id']}" for r in records])
    splitter = GroupKFold(n_splits=folds)
    splits = list(splitter.split(x, safe, groups))
    classifier = RandomForestClassifier(
        n_estimators=trees, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1
    )
    scalar = RandomForestRegressor(n_estimators=trees, min_samples_leaf=2, random_state=seed, n_jobs=-1)
    profile = RandomForestRegressor(n_estimators=trees, min_samples_leaf=2, random_state=seed, n_jobs=-1)
    safe_prediction = cross_val_predict(classifier, x, safe, cv=splits, method="predict")
    scalar_prediction = cross_val_predict(scalar, x, effective, cv=splits, method="predict")
    profile_prediction = cross_val_predict(profile, x, profiles, cv=splits, method="predict")
    return {
        "status": "GROUPED_DEVELOPMENT_DIAGNOSTIC_ONLY",
        "folds": folds,
        "document_groups": int(len(set(groups.tolist()))),
        "minimum_safe_precision": classification_metrics(safe, safe_prediction, sorted(set(safe.tolist()))),
        "effective_bits": regression_metrics(effective, scalar_prediction),
        "group_profile": regression_metrics(profiles, profile_prediction, multioutput="variance_weighted"),
    }


def metrics_by_length_cell(
    train: list[dict[str, Any]], calibration: list[dict[str, Any]], test: list[dict[str, Any]],
    names: list[str], seed: int, trees: int,
) -> dict[str, Any]:
    output = {}
    cells = sorted({(r["prompt_length_tokens"], r["continuation_length_tokens"]) for r in test})
    for prompt, continuation in cells:
        cell_test = [r for r in test if (r["prompt_length_tokens"], r["continuation_length_tokens"]) == (prompt, continuation)]
        output[f"{prompt}p:{continuation}c"] = evaluate_predictors(train, calibration, cell_test, names, seed, trees)
    return output


def aggregate_seed_metrics(seed_results: dict[str, Any]) -> dict[str, Any]:
    paths = {
        "balanced_accuracy": ("minimum_safe_precision_classifier", "balanced_accuracy"),
        "macro_f1": ("minimum_safe_precision_classifier", "macro_f1"),
        "effective_bits_mae": ("effective_bits_regressor", "mae"),
        "effective_bits_r2": ("effective_bits_regressor", "r2"),
        "group_profile_mae": ("group_profile_regressor", "mae"),
        "group_profile_r2": ("group_profile_regressor", "r2"),
    }
    output = {}
    for name, (section, metric) in paths.items():
        values = np.asarray([result[section][metric] for result in seed_results.values()], dtype=np.float64)
        output[name] = {"mean": float(values.mean()), "standard_deviation": float(values.std(ddof=0))}
    return output


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 1 or args.trees < 1 or args.group_folds < 2:
        raise ValueError("Replicates/trees must be positive and group_folds >= 2")
    collection_dir = Path(args.collection_dir).resolve()
    freeze_path = Path(args.freeze_manifest).resolve()
    freeze = verify_freeze(collection_dir, freeze_path)
    records, metadata = load_collection(collection_dir)
    if any(record.get("schema_version") != RECORD_SCHEMA_VERSION for record in records):
        raise RuntimeError("Unexpected record schema")
    if any(record.get("quality_scope") != "continuation_only_teacher_forced" for record in records):
        raise RuntimeError("Non-continuation-only quality record")

    development = [r for r in records if r["source"]["partition"] == "development"]
    calibration = [r for r in records if r["source"]["partition"] == "calibration"]
    test = [r for r in records if r["source"]["partition"] == "test"]
    if (len(development), len(calibration), len(test)) != (256, 64, 192):
        raise RuntimeError("Partition counts do not match preregistration")
    _, feature_names = feature_matrix(development)

    datasets = sorted({r["source"]["dataset"] for r in records})
    h1 = {}
    h3 = {}
    for dataset_index, dataset in enumerate(datasets):
        dataset_test = [r for r in test if r["source"]["dataset"] == dataset]
        h1[dataset] = demand_summary(dataset_test, args.bootstrap_replicates, args.bootstrap_seed + dataset_index)
        h1[dataset]["by_length_cell"] = {
            f"{prompt}p:{continuation}c": demand_summary(
                [r for r in dataset_test if (r["prompt_length_tokens"], r["continuation_length_tokens"]) == (prompt, continuation)],
                args.bootstrap_replicates,
                args.bootstrap_seed + 10 + dataset_index * 10 + cell_index,
            )
            for cell_index, (prompt, continuation) in enumerate(sorted({
                (r["prompt_length_tokens"], r["continuation_length_tokens"]) for r in dataset_test
            }))
        }
        h3[dataset] = quality_summary(dataset_test, args.bootstrap_replicates, args.bootstrap_seed + 1000 + dataset_index * 100)

    predictors = {"development_grouped_diagnostics": {}, "test_by_dataset": {name: {} for name in datasets}}
    for seed in args.predictor_seeds:
        predictors["development_grouped_diagnostics"][str(seed)] = grouped_development_diagnostic(
            development, seed, args.trees, args.group_folds
        )
        for dataset in datasets:
            dataset_test = [r for r in test if r["source"]["dataset"] == dataset]
            result = evaluate_predictors(development, calibration, dataset_test, feature_names, seed, args.trees)
            result["by_length_cell"] = metrics_by_length_cell(
                development, calibration, dataset_test, feature_names, seed, args.trees
            )
            predictors["test_by_dataset"][dataset][str(seed)] = result
    predictors["seed_aggregates"] = {
        dataset: aggregate_seed_metrics(results)
        for dataset, results in predictors["test_by_dataset"].items()
    }
    predictors["h2_pass"] = bool(all(
        result["all_endpoints_pass"]
        for dataset_results in predictors["test_by_dataset"].values()
        for result in dataset_results.values()
    ))

    output = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_status": "PREREGISTERED_COLLECTION_ANALYSIS_COMPLETE_WITH_UNAVAILABLE_ENDPOINTS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_version": PROTOCOL_VERSION,
        "collection_id": metadata["run_manifest"]["collection_id"],
        "collection_tree_sha256": freeze["collection_tree_sha256"],
        "collection_freeze_manifest": str(freeze_path),
        "collection_freeze_manifest_sha256": file_sha256(freeze_path),
        "analysis_source_sha256": file_sha256(Path(__file__)),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True).stdout.strip(),
        "git_worktree_dirty": bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True, capture_output=True
        ).stdout.strip()),
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "config": vars(args),
        "partition_counts": {"development": len(development), "calibration": len(calibration), "test": len(test)},
        "feature_names": feature_names,
        "h1_heterogeneous_demand": {"by_dataset": h1, "pass_both_datasets": bool(all(x["h1_pass"] for x in h1.values()))},
        "h2_predecode_predictability": predictors,
        "h3_guarded_mlp": {
            "by_dataset": h3,
            "overall_status": (
                "FAIL_GUARDED_QUALITY_GATE_ROUTE_SAFETY_UNAVAILABLE"
                if any(result["h3_status"].startswith("FAIL_") for result in h3.values())
                else "NOT_EVALUABLE_ROUTE_SAFETY_ENDPOINT_MISSING"
            ),
        },
        "h4_scheduling": {
            "status": "NOT_RUN_REQUIRES_REAL_ONLINE_GPU_REPLAY",
            "reason": "Teacher-forced request-demand records do not contain registered arrivals, shared-profile execution, queue delay, or synchronized scheduler timing.",
        },
        "multiple_testing": {
            "status": "NOT_APPLIED_NO_CONFIRMATORY_P_VALUES",
            "reason": "This stage reports preregistered bootstrap safety bounds and predictive thresholds; missing H3 route-safety and H4 endpoints prevent full hypothesis-family testing.",
        },
        "provenance_deviation": {
            "status": "DISCLOSED_DIRTY_PRECOMMIT_COLLECTION",
            "strictly_preregistration_clean": False,
            "recorded_collection_git_commit": metadata["run_manifest"]["source_provenance"]["git_commit"],
            "recorded_collection_worktree_dirty": metadata["run_manifest"]["source_provenance"]["git_worktree_dirty"],
        },
        "contains_raw_text": False,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="ascii")
    temporary.replace(output_path)
    verify_freeze(collection_dir, freeze_path)
    print(json.dumps({
        "analysis_status": output["analysis_status"],
        "output_json": str(output_path),
        "collection_tree_sha256": freeze["collection_tree_sha256"],
        "h1_pass_both_datasets": output["h1_heterogeneous_demand"]["pass_both_datasets"],
        "h2_pass": predictors["h2_pass"],
        "h3_status": output["h3_guarded_mlp"]["overall_status"],
        "h4_status": output["h4_scheduling"]["status"],
    }, indent=2))


if __name__ == "__main__":
    main()
