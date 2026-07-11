"""Analyze request-level QAQ demand for scheduling oracle value and pre-decode predictability."""

import argparse
import hashlib
import itertools
import json
import math
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import scipy
import sklearn
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix
from sklearn.dummy import DummyRegressor
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
from sklearn.model_selection import KFold, cross_val_predict


ANALYSIS_SCHEMA_VERSION = "qaq_request_demand_analysis_v1"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an exact observed-profile batching oracle and cross-validated "
            "pre-decode demand/profile predictors."
        )
    )
    parser.add_argument("--dataset_jsonl", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument(
        "--profile_mode",
        choices=["dp_threshold_only", "mlp_multibit", "mlp_multibit_dp_guard"],
        default="mlp_multibit_dp_guard",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--milp_time_limit_s", type=float, default=120.0)
    parser.add_argument("--material_oracle_advantage", type=float, default=0.01)
    parser.add_argument("--predictability_mae_improvement", type=float, default=0.10)
    parser.add_argument("--predictability_r2", type=float, default=0.10)
    return parser.parse_args()


def load_records(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open() as source:
        for line_no, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("schema_version") != "qaq_request_demand_v1":
                raise ValueError(f"Unsupported record schema on line {line_no}")
            records.append(record)
    if not records:
        raise ValueError("Request-demand dataset is empty")
    return records


def profile_matrix(records: list[dict[str, Any]], profile_mode: str) -> np.ndarray:
    vectors = [
        record["observed_qaq_profiles"][profile_mode]["group_expected_bits"]
        for record in records
    ]
    width = len(vectors[0])
    if width == 0 or any(len(vector) != width for vector in vectors):
        raise ValueError("Observed group profiles must be non-empty and equal width")
    return np.asarray(vectors, dtype=np.float64)


def feature_matrix(records: list[dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    names = sorted(records[0]["prompt_features"])
    if any(sorted(record["prompt_features"]) != names for record in records):
        raise ValueError("Prompt feature schemas differ across requests")
    matrix = np.asarray(
        [[float(record["prompt_features"][name]) for name in names] for record in records],
        dtype=np.float64,
    )
    if not np.isfinite(matrix).all():
        raise ValueError("Prompt features contain non-finite values")
    return matrix, names


def batch_precision_work(profile_rows: np.ndarray) -> float:
    return float(profile_rows.shape[0] * np.max(profile_rows, axis=0).mean())


def partition_cost(profiles: np.ndarray, batches: list[tuple[int, ...]]) -> float:
    return sum(batch_precision_work(profiles[list(batch)]) for batch in batches)


def fixed_order_batches(order: list[int], batch_size: int) -> list[tuple[int, ...]]:
    if len(order) % batch_size:
        raise ValueError("Request count must be divisible by batch_size for exact comparison")
    return [
        tuple(order[start:start + batch_size])
        for start in range(0, len(order), batch_size)
    ]


def exact_profile_oracle(
    profiles: np.ndarray,
    batch_size: int,
    time_limit_s: float,
) -> dict[str, Any]:
    request_count = profiles.shape[0]
    if batch_size < 2 or request_count % batch_size:
        raise ValueError("batch_size must be >= 2 and evenly divide request count")
    candidates = list(itertools.combinations(range(request_count), batch_size))
    costs = np.asarray(
        [batch_precision_work(profiles[list(batch)]) for batch in candidates],
        dtype=np.float64,
    )
    row_indices = []
    col_indices = []
    for column, batch in enumerate(candidates):
        for request_index in batch:
            row_indices.append(request_index)
            col_indices.append(column)
    matrix = csc_matrix(
        (
            np.ones(len(row_indices), dtype=np.float64),
            (np.asarray(row_indices), np.asarray(col_indices)),
        ),
        shape=(request_count, len(candidates)),
    )
    result = milp(
        c=costs,
        integrality=np.ones(len(candidates), dtype=np.int8),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(matrix, 1.0, 1.0),
        options={"time_limit": float(time_limit_s)},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Exact profile grouping MILP failed: {result.message}")
    selected = [
        candidates[index] for index, value in enumerate(result.x) if value > 0.5
    ]
    if len(selected) != request_count // batch_size:
        raise RuntimeError("MILP returned an invalid number of batches")
    return {
        "solver": "scipy.optimize.milp_highs",
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "candidate_batch_count": len(candidates),
        "optimal_precision_work": float(result.fun),
        "batches": [list(batch) for batch in selected],
    }


def scheduler_oracle_analysis(
    records: list[dict[str, Any]],
    profiles: np.ndarray,
    profile_mode: str,
    batch_size: int,
    time_limit_s: float,
) -> dict[str, Any]:
    count = len(records)
    fcfs_batches = fixed_order_batches(list(range(count)), batch_size)
    scalar_order = sorted(
        range(count),
        key=lambda index: (
            records[index]["observed_qaq_profiles"][profile_mode]["effective_bits"],
            records[index]["request_id"],
        ),
    )
    scalar_batches = fixed_order_batches(scalar_order, batch_size)
    oracle = exact_profile_oracle(profiles, batch_size, time_limit_s)
    oracle_batches = [tuple(batch) for batch in oracle["batches"]]

    independent_lower_bound = float(np.mean(profiles, axis=1).sum())
    fcfs_cost = partition_cost(profiles, fcfs_batches)
    scalar_cost = partition_cost(profiles, scalar_batches)
    oracle_cost = partition_cost(profiles, oracle_batches)

    def overhead(cost):
        return cost / independent_lower_bound - 1.0 if independent_lower_bound > 0 else 0.0

    return {
        "status": "EXACT_OBSERVED_PROFILE_ORACLE",
        "cost_metric": (
            "sum over batches of batch_size * mean(componentwise max observed group bit); "
            "a precision-work proxy, not measured latency"
        ),
        "profile_mode": profile_mode,
        "profile_dimension": int(profiles.shape[1]),
        "request_count": count,
        "batch_size": batch_size,
        "independent_execution_lower_bound": independent_lower_bound,
        "fcfs_precision_work": fcfs_cost,
        "scalar_sorted_precision_work": scalar_cost,
        "oracle_precision_work": oracle_cost,
        "oracle_advantage_vs_fcfs_fraction": (fcfs_cost - oracle_cost) / fcfs_cost,
        "oracle_advantage_vs_scalar_fraction": (scalar_cost - oracle_cost) / scalar_cost,
        "fcfs_padding_overhead_fraction": overhead(fcfs_cost),
        "scalar_padding_overhead_fraction": overhead(scalar_cost),
        "oracle_padding_overhead_fraction": overhead(oracle_cost),
        "fcfs_batches": [list(batch) for batch in fcfs_batches],
        "scalar_sorted_batches": [list(batch) for batch in scalar_batches],
        **oracle,
    }


def fold_mean_predictions(target: np.ndarray, splits) -> np.ndarray:
    predictions = np.empty_like(target, dtype=np.float64)
    for train, test in splits:
        predictions[test] = np.mean(target[train], axis=0)
    return predictions


def classification_analysis(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    splits,
    seed: int,
    trees: int,
) -> dict[str, Any]:
    distribution = {str(label): int(count) for label, count in sorted(Counter(labels.tolist()).items())}
    if len(distribution) < 2:
        return {
            "status": "SINGLE_CLASS_TARGET",
            "class_distribution": distribution,
            "predictable_before_decode": False,
            "reason": "All requests have the same minimum safe precision.",
        }
    model = RandomForestClassifier(
        n_estimators=trees,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    predictions = cross_val_predict(model, features, labels, cv=splits, method="predict")
    model.fit(features, labels)
    majority_accuracy = max(distribution.values()) / len(labels)
    classes = sorted(int(value) for value in np.unique(labels))
    accuracy = float(accuracy_score(labels, predictions))
    return {
        "status": "CROSS_VALIDATED",
        "model": "RandomForestClassifier",
        "class_distribution": distribution,
        "accuracy": accuracy,
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "majority_baseline_accuracy": float(majority_accuracy),
        "accuracy_improvement_vs_majority": float(accuracy - majority_accuracy),
        "confusion_matrix_labels": classes,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=classes).tolist(),
        "feature_importance": {
            name: float(value)
            for name, value in sorted(
                zip(feature_names, model.feature_importances_),
                key=lambda item: item[1],
                reverse=True,
            )
        },
        "predictions": [int(value) for value in predictions],
        "predictable_before_decode": bool(accuracy > majority_accuracy),
    }


def regression_metrics(target: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(target, predictions)),
        "rmse": float(math.sqrt(mean_squared_error(target, predictions))),
        "r2": float(r2_score(target, predictions)),
    }


def predictor_analysis(
    records: list[dict[str, Any]],
    profiles: np.ndarray,
    profile_mode: str,
    cv_folds: int,
    seed: int,
    trees: int,
    minimum_mae_improvement: float = 0.10,
    minimum_r2: float = 0.10,
) -> dict[str, Any]:
    features, feature_names = feature_matrix(records)
    if len(records) < 8:
        raise ValueError("At least 8 requests are required for predictor evaluation")
    fold_count = min(cv_folds, len(records))
    if fold_count < 2:
        raise ValueError("cv_folds must be at least 2")
    splits = list(KFold(n_splits=fold_count, shuffle=True, random_state=seed).split(features))

    safe_bits = np.asarray(
        [record["minimum_safe_precision"]["requested_bit"] for record in records],
        dtype=np.int64,
    )
    classification = classification_analysis(
        features, safe_bits, feature_names, splits, seed, trees
    )

    effective_bits = np.asarray(
        [record["observed_qaq_profiles"][profile_mode]["effective_bits"] for record in records],
        dtype=np.float64,
    )
    scalar_model = RandomForestRegressor(
        n_estimators=trees, min_samples_leaf=2, random_state=seed, n_jobs=-1
    )
    scalar_predictions = cross_val_predict(
        scalar_model, features, effective_bits, cv=splits, method="predict"
    )
    scalar_baseline = fold_mean_predictions(effective_bits, splits)
    scalar_model.fit(features, effective_bits)
    scalar_metrics = regression_metrics(effective_bits, scalar_predictions)
    scalar_baseline_metrics = regression_metrics(effective_bits, scalar_baseline)

    profile_model = RandomForestRegressor(
        n_estimators=trees, min_samples_leaf=2, random_state=seed, n_jobs=-1
    )
    profile_predictions = cross_val_predict(
        profile_model, features, profiles, cv=splits, method="predict"
    )
    profile_baseline = fold_mean_predictions(profiles, splits)
    profile_model.fit(features, profiles)
    profile_mae = float(mean_absolute_error(profiles, profile_predictions))
    baseline_mae = float(mean_absolute_error(profiles, profile_baseline))
    component_mae = np.mean(np.abs(profiles - profile_predictions), axis=0)
    profile_rmse = float(math.sqrt(mean_squared_error(profiles, profile_predictions)))

    profile_r2_uniform = float(
        r2_score(profiles, profile_predictions, multioutput="uniform_average")
    )
    profile_r2_variance = float(
        r2_score(profiles, profile_predictions, multioutput="variance_weighted")
    )
    profile_improvement = (
        (baseline_mae - profile_mae) / baseline_mae if baseline_mae > 0 else 0.0
    )
    profile_supported = bool(
        profile_improvement >= minimum_mae_improvement
        and profile_r2_variance >= minimum_r2
    )
    profile_assessment = (
        "SUPPORTED" if profile_supported
        else "WEAK_SIGNAL" if profile_improvement > 0 and profile_r2_variance > 0
        else "NOT_SUPPORTED"
    )
    return {
        "status": "REQUEST_LEVEL_CROSS_VALIDATION",
        "feature_availability": "prompt tokens and fixed-high prompt prefill only; no continuation/decode features",
        "feature_names": feature_names,
        "folds": fold_count,
        "seed": seed,
        "minimum_safe_precision_classifier": classification,
        "effective_bits_regressor": {
            "model": "RandomForestRegressor",
            **scalar_metrics,
            "mean_baseline": scalar_baseline_metrics,
            "mae_improvement_vs_mean_fraction": (
                (scalar_baseline_metrics["mae"] - scalar_metrics["mae"]) / scalar_baseline_metrics["mae"]
                if scalar_baseline_metrics["mae"] > 0 else 0.0
            ),
            "feature_importance": {
                name: float(value)
                for name, value in sorted(
                    zip(feature_names, scalar_model.feature_importances_),
                    key=lambda item: item[1],
                    reverse=True,
                )
            },
            "predictions": scalar_predictions.tolist(),
        },
        "group_profile_regressor": {
            "model": "RandomForestRegressor_multioutput",
            "profile_mode": profile_mode,
            "profile_dimension": int(profiles.shape[1]),
            "mae": profile_mae,
            "rmse": profile_rmse,
            "r2_uniform_average": profile_r2_uniform,
            "r2_variance_weighted": profile_r2_variance,
            "predictability_assessment": profile_assessment,
            "mean_baseline_mae": baseline_mae,
            "mae_improvement_vs_mean_fraction": profile_improvement,
            "component_mae": component_mae.tolist(),
            "feature_importance": {
                name: float(value)
                for name, value in sorted(
                    zip(feature_names, profile_model.feature_importances_),
                    key=lambda item: item[1],
                    reverse=True,
                )
            },
            "predictions": profile_predictions.tolist(),
        },
        "profile_predictable_before_decode": profile_supported,
    }


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    args = parse_args()
    if args.batch_size < 2 or args.cv_folds < 2 or args.trees < 1:
        raise ValueError("batch_size/cv_folds must be >= 2 and trees must be positive")
    records = load_records(args.dataset_jsonl)
    profiles = profile_matrix(records, args.profile_mode)
    oracle = scheduler_oracle_analysis(
        records, profiles, args.profile_mode, args.batch_size, args.milp_time_limit_s
    )
    predictor = predictor_analysis(
        records, profiles, args.profile_mode, args.cv_folds, args.seed, args.trees,
        args.predictability_mae_improvement,
        args.predictability_r2,
    )

    output = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_status": "REAL_DATA_OFFLINE_ANALYSIS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
            capture_output=True, check=False,
        ).stdout.strip(),
        "git_worktree_dirty": bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True,
            capture_output=True, check=False,
        ).stdout.strip()),
        "analysis_source_sha256": hashlib.sha256(
            (REPO_ROOT / "scripts/analyze_qaq_request_demand.py").read_bytes()
        ).hexdigest(),
        "dataset_jsonl": str(Path(args.dataset_jsonl).resolve()),
        "dataset_sha256": file_sha256(args.dataset_jsonl),
        "request_ids": [record["request_id"] for record in records],
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "config": vars(args),
        "oracle_scheduler": oracle,
        "predecode_predictor": predictor,
        "conclusions": {
            "observed_profile_oracle_advantage": bool(
                oracle["oracle_advantage_vs_fcfs_fraction"] > 0
            ),
            "observed_profile_oracle_advantage_vs_scalar": bool(
                oracle["oracle_advantage_vs_scalar_fraction"] > 0
            ),
            "material_profile_oracle_advantage": bool(
                oracle["oracle_advantage_vs_fcfs_fraction"] >= args.material_oracle_advantage
            ),
            "material_profile_oracle_advantage_vs_scalar": bool(
                oracle["oracle_advantage_vs_scalar_fraction"] >= args.material_oracle_advantage
            ),
            "profile_predictable_before_decode": predictor["profile_predictable_before_decode"],
            "profile_predictability_assessment": predictor[
                "group_profile_regressor"
            ]["predictability_assessment"],
            "minimum_safe_precision_predictable_before_decode": predictor[
                "minimum_safe_precision_classifier"
            ]["predictable_before_decode"],
        },
        "limitations": [
            "Oracle costs are precision-work proxies from observed profiles, not measured latency or throughput.",
            "Predictor metrics are request-level cross-validation on one held-out subset, not external-dataset generalization.",
            "Observed profiles come from teacher-forced prompt+continuation execution; predictor features stop at prompt prefill.",
        ],
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w") as target:
        json.dump(output, target, indent=2)
    temporary.replace(output_path)
    print(json.dumps({
        "analysis_status": output["analysis_status"],
        "output_json": str(output_path),
        "request_count": len(records),
        "oracle_advantage_vs_fcfs_fraction": oracle["oracle_advantage_vs_fcfs_fraction"],
        "oracle_advantage_vs_scalar_fraction": oracle["oracle_advantage_vs_scalar_fraction"],
        "profile_predictable_before_decode": predictor["profile_predictable_before_decode"],
        "minimum_safe_precision_predictable_before_decode": predictor[
            "minimum_safe_precision_classifier"
        ]["predictable_before_decode"],
    }, indent=2))


if __name__ == "__main__":
    main()
