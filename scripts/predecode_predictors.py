"""Held-out training and evaluation for prompt/prefill-only QAQ predictors.

The preregistered request-demand collection already contains real targets and
source-document partitions.  This module deliberately keeps the split fixed:
development rows train the models, calibration rows set temperature,
conformal uncertainty, and fallback cutoffs, and test rows are evaluated once.
No continuation or observed-route field is admitted as a feature.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold


RESULT_SCHEMA_VERSION = "qaq_predecode_predictor_results_v2"
RECORD_SCHEMA_VERSION = "qaq_request_demand_v2"
PARTITIONS = ("development", "calibration", "test")
DEFAULT_SEEDS = (17, 29, 43)
FORBIDDEN_FEATURE_TERMS = (
    "continuation", "generated", "observed", "fallback", "guard", "route",
    "profile", "quality", "target", "safe", "delta",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_sha256(path: str | Path | list[str] | tuple[str, ...]) -> str:
    if isinstance(path, (list, tuple)):
        digest = hashlib.sha256()
        for item in path:
            digest.update(str(Path(item).resolve()).encode("utf-8"))
            digest.update(input_sha256(item).encode("ascii"))
        return digest.hexdigest()
    root = Path(path)
    if root.is_file():
        return _sha256_file(root)
    digest = hashlib.sha256()
    files = sorted(root.rglob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No JSONL records found under {root}")
    for item in files:
        digest.update(str(item.relative_to(root)).encode("utf-8"))
        digest.update(_sha256_file(item).encode("ascii"))
    return digest.hexdigest()


def _record_paths(path: str | Path | list[str] | tuple[str, ...]) -> list[Path]:
    if isinstance(path, (list, tuple)):
        paths = []
        for item in path:
            paths.extend(_record_paths(item))
        return paths
    root = Path(path)
    if root.is_file():
        return [root]
    preferred = sorted(root.glob("datasets/*/shards/*.jsonl"))
    return preferred or sorted(root.rglob("*.jsonl"))


def _validate_feature_names(names: Iterable[str]) -> None:
    for name in names:
        lowered = name.lower()
        if any(term in lowered for term in FORBIDDEN_FEATURE_TERMS):
            raise ValueError(f"Feature {name!r} is not available before decode")


def load_records(path: str | Path | list[str] | tuple[str, ...], required_datasets: Iterable[str] = ("wikitext2", "c4_new")) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record_path in _record_paths(path):
        with record_path.open(encoding="ascii") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("schema_version") != RECORD_SCHEMA_VERSION:
                    raise ValueError(
                        f"{record_path}:{line_number} is not {RECORD_SCHEMA_VERSION}"
                    )
                source_info = record.get("source", {})
                required = ("dataset", "document_id", "partition")
                if any(key not in source_info for key in required):
                    raise ValueError(f"{record_path}:{line_number} lacks source provenance")
                if source_info["partition"] not in PARTITIONS:
                    raise ValueError(f"Unknown partition {source_info['partition']!r}")
                if record.get("contains_raw_text"):
                    raise ValueError("Raw prompt/continuation text is forbidden")
                features = record.get("prompt_features")
                if not isinstance(features, dict) or not features:
                    raise ValueError("Every record needs numeric prompt_features")
                _validate_feature_names(features)
                if any(not np.isfinite(float(value)) for value in features.values()):
                    raise ValueError("Prompt features contain non-finite values")
                records.append(record)
    if not records:
        raise ValueError("Request-demand dataset is empty")
    request_ids = [record["request_id"] for record in records]
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("Duplicate request_id in request-demand dataset")
    _validate_partition_provenance(records, set(required_datasets))
    return records


def _group_key(record: dict[str, Any]) -> tuple[str, str]:
    source = record["source"]
    return str(source["dataset"]), str(source["document_id"])


def _validate_partition_provenance(records: list[dict[str, Any]], required_datasets: set[str]) -> None:
    groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in records:
        groups[_group_key(record)].add(record["source"]["partition"])
    if any(len(partitions) != 1 for partitions in groups.values()):
        raise ValueError("A source document appears in more than one partition")
    present_datasets = {record["source"]["dataset"] for record in records}
    if not required_datasets.issubset(present_datasets):
        raise ValueError(
            f"Held-out predictor analysis requires datasets {sorted(required_datasets)}, "
            f"found {sorted(present_datasets)}"
        )
    present = {record["source"]["partition"] for record in records}
    if present != set(PARTITIONS):
        raise ValueError(f"Expected all partitions {PARTITIONS}, found {sorted(present)}")


def feature_matrix(records: list[dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    names = sorted(records[0]["prompt_features"])
    _validate_feature_names(names)
    if any(sorted(record["prompt_features"]) != names for record in records):
        raise ValueError("Prompt feature schemas differ across records")
    matrix = np.asarray(
        [[float(record["prompt_features"][name]) for name in names] for record in records],
        dtype=np.float64,
    )
    if not np.isfinite(matrix).all():
        raise ValueError("Prompt feature matrix contains non-finite values")
    return matrix, names


def target_arrays(records: list[dict[str, Any]], profile_mode: str) -> dict[str, np.ndarray]:
    try:
        profiles = np.asarray(
            [record["observed_qaq_profiles"][profile_mode]["group_expected_bits"] for record in records],
            dtype=np.float64,
        )
        effective = np.asarray(
            [record["observed_qaq_profiles"][profile_mode]["effective_bits"] for record in records],
            dtype=np.float64,
        )
        guard_fraction = np.asarray(
            [record["quality_by_mode"][profile_mode]["dp_guard_trigger_fraction"] for record in records],
            dtype=np.float64,
        )
        safe_bits = np.asarray(
            [record["minimum_safe_precision"]["requested_bit"] for record in records],
            dtype=np.int64,
        )
        guard_triggered = np.asarray(
            [record["quality_by_mode"][profile_mode]["dp_guard_trigger_count"] > 0 for record in records],
            dtype=bool,
        )
    except KeyError as exc:
        raise ValueError(f"Missing target field for profile mode {profile_mode!r}: {exc}") from exc
    if profiles.ndim != 2 or profiles.shape[1] == 0:
        raise ValueError("Group profile target must be a non-empty matrix")
    if not all(np.isfinite(value).all() for value in (profiles, effective, guard_fraction)):
        raise ValueError("Targets contain non-finite values")
    return {
        "safe_bit": safe_bits,
        "effective_bits": effective,
        "group_profile": profiles,
        "guard_trigger_fraction": guard_fraction,
        "guard_triggered": guard_triggered,
    }


def partition_masks(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        partition: np.asarray(
            [record["source"]["partition"] == partition for record in records], dtype=bool
        )
        for partition in PARTITIONS
    }

def evaluation_masks(
    records: list[dict[str, Any]], holdout_dataset: str | None = None
) -> dict[str, np.ndarray]:
    """Return leakage-free train/calibration/test masks.

    With holdout_dataset set, every development and calibration row from that
    dataset is excluded, and the test mask contains only that dataset. This
    makes the LODO boundary explicit and reusable in tests.
    """
    masks = partition_masks(records)
    if holdout_dataset is None:
        return masks
    dataset_names = {str(record["source"]["dataset"]) for record in records}
    if holdout_dataset not in dataset_names:
        raise ValueError(
            f"Unknown holdout dataset {holdout_dataset!r}; found {sorted(dataset_names)}"
        )
    heldout = np.asarray(
        [record["source"]["dataset"] == holdout_dataset for record in records],
        dtype=bool,
    )
    result = {
        "development": masks["development"] & ~heldout,
        "calibration": masks["calibration"] & ~heldout,
        "test": masks["test"] & heldout,
    }
    if not result["development"].any() or not result["calibration"].any() or not result["test"].any():
        raise ValueError(f"LODO split for {holdout_dataset!r} has an empty partition")
    return result



def split_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for partition in PARTITIONS:
        rows = [record for record in records if record["source"]["partition"] == partition]
        summary[partition] = {
            "request_count": len(rows),
            "document_count": len({_group_key(record) for record in rows}),
            "by_dataset": {
                dataset: {
                    "request_count": sum(record["source"]["dataset"] == dataset for record in rows),
                    "document_count": len({_group_key(record) for record in rows if record["source"]["dataset"] == dataset}),
                }
                for dataset in sorted({record["source"]["dataset"] for record in rows})
            },
        }
    return summary


def grouped_development_folds(records: list[dict[str, Any]], folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    development = [record for record in records if record["source"]["partition"] == "development"]
    if len(development) < 2:
        raise ValueError("At least two development requests are required")
    group_keys = [_group_key(record) for record in development]
    groups = np.asarray([f"{dataset}\0{document}" for dataset, document in group_keys])
    unique_groups = list(dict.fromkeys(group_keys))
    fold_count = min(int(folds), len(unique_groups))
    if fold_count < 2:
        raise ValueError("At least two development source documents are required")
    splitter = GroupKFold(n_splits=fold_count)
    indices = np.arange(len(development))
    return list(splitter.split(indices, groups=groups))


def _forest_classifier(seed: int, trees: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=trees, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1
    )


def _forest_regressor(seed: int, trees: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=trees, min_samples_leaf=2, random_state=seed, n_jobs=-1
    )


def _align_probabilities(probabilities: np.ndarray, model_classes: np.ndarray, classes: np.ndarray) -> np.ndarray:
    aligned = np.full((len(probabilities), len(classes)), 1e-12, dtype=np.float64)
    for column, value in enumerate(model_classes.tolist()):
        aligned[:, int(np.flatnonzero(classes == value)[0])] = probabilities[:, column]
    return aligned / aligned.sum(axis=1, keepdims=True)


def fit_temperature(probabilities: np.ndarray, labels: np.ndarray, classes: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 1.0
    indices = np.asarray([int(np.flatnonzero(classes == label)[0]) for label in labels])
    logits = np.log(np.clip(probabilities, 1e-12, 1.0))

    def objective(log_temperature: float) -> float:
        scaled = softmax(logits / math.exp(float(log_temperature)), axis=1)
        return float(-np.log(np.clip(scaled[np.arange(len(indices)), indices], 1e-12, 1.0)).mean())

    result = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
    return float(math.exp(result.x)) if result.success else 1.0


def calibrated_probabilities(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probabilities, 1e-12, 1.0))
    return softmax(logits / max(float(temperature), 1e-8), axis=1)


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, classes: np.ndarray, bins: int = 10) -> float:
    predictions = classes[np.argmax(probabilities, axis=1)]
    confidence = probabilities.max(axis=1)
    correct = predictions == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for start, end in zip(edges[:-1], edges[1:]):
        selected = (confidence >= start) & (confidence <= end if end == 1.0 else confidence < end)
        if selected.any():
            error += float(selected.mean()) * abs(float(correct[selected].mean()) - float(confidence[selected].mean()))
    return float(error)


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray, classes: np.ndarray, high_bit: int, fallback_cutoff: float | None = None) -> dict[str, Any]:
    predictions = classes[np.argmax(probabilities, axis=1)].astype(np.int64)
    confidence = probabilities.max(axis=1)
    fallback = np.zeros(len(labels), dtype=bool) if fallback_cutoff is None else confidence < fallback_cutoff
    conservative = np.where(fallback, high_bit, predictions)
    one_hot = np.zeros_like(probabilities)
    for row, label in enumerate(labels):
        one_hot[row, int(np.flatnonzero(classes == label)[0])] = 1.0
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=classes, average="macro", zero_division=0)),
        "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "log_loss": float(log_loss(labels, probabilities, labels=classes)),
        "ece": expected_calibration_error(labels, probabilities, classes),
        "underprediction_rate": float(np.mean(predictions < labels)),
        "conservative_underprediction_rate": float(np.mean(conservative < labels)),
        "conservative_overprediction_rate": float(np.mean(conservative > labels)),
        "fallback_fraction": float(fallback.mean()),
        "mean_confidence": float(confidence.mean()),
        "predictions": predictions.tolist(),
        "conservative_predictions": conservative.tolist(),
    }


def _safe_r2(labels: np.ndarray, predictions: np.ndarray, multioutput: str = "uniform_average") -> float:
    return float(r2_score(labels, predictions, multioutput=multioutput, force_finite=True))


def regression_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(labels, predictions)),
        "rmse": float(math.sqrt(mean_squared_error(labels, predictions))),
        "r2": _safe_r2(labels, predictions),
    }


def profile_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(labels, predictions)),
        "rmse": float(math.sqrt(mean_squared_error(labels, predictions))),
        "r2_uniform_average": _safe_r2(labels, predictions, "uniform_average"),
        "r2_variance_weighted": _safe_r2(labels, predictions, "variance_weighted"),
    }


def forest_regression_prediction(model: RandomForestRegressor, features: np.ndarray, lower: float | None = None, upper: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    tree_predictions = np.stack([tree.predict(features) for tree in model.estimators_], axis=0)
    if lower is not None or upper is not None:
        tree_predictions = np.clip(tree_predictions, lower, upper)
    prediction = tree_predictions.mean(axis=0)
    uncertainty = tree_predictions.std(axis=0, ddof=1 if len(tree_predictions) > 1 else 0)
    return prediction, uncertainty


def conformal_scale(labels: np.ndarray, predictions: np.ndarray, raw_uncertainty: np.ndarray, alpha: float) -> float:
    scale = np.maximum(np.asarray(raw_uncertainty, dtype=np.float64), 1e-8)
    scores = np.abs(labels - predictions) / scale
    rank = min(max(int(math.ceil((len(scores) + 1) * (1.0 - alpha))) - 1, 0), len(scores) - 1)
    return float(np.sort(scores)[rank])


def conformal_profile_scale(labels: np.ndarray, predictions: np.ndarray, raw_uncertainty: np.ndarray, alpha: float) -> float:
    scale = np.maximum(np.asarray(raw_uncertainty, dtype=np.float64), 1e-8)
    scores = np.max(np.abs(labels - predictions) / scale, axis=1)
    rank = min(max(int(math.ceil((len(scores) + 1) * (1.0 - alpha))) - 1, 0), len(scores) - 1)
    return float(np.sort(scores)[rank])


def interval_metrics(labels: np.ndarray, predictions: np.ndarray, half_width: np.ndarray) -> dict[str, float]:
    lower = predictions - half_width
    upper = predictions + half_width
    return {"coverage": float(np.mean((labels >= lower) & (labels <= upper))), "mean_width": float(np.mean(2.0 * half_width))}


def _bootstrap_rows(records: list[dict[str, Any]], indices: np.ndarray, repetitions: int, seed: int) -> Iterable[np.ndarray]:
    clusters: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index in indices.tolist():
        clusters[_group_key(records[index])].append(index)
    keys = sorted(clusters)
    rng = np.random.default_rng(seed)
    for _ in range(repetitions):
        sampled = rng.choice(len(keys), size=len(keys), replace=True)
        yield np.asarray([row for key_index in sampled for row in clusters[keys[key_index]]], dtype=np.int64)


def cluster_bootstrap_ci(records: list[dict[str, Any]], indices: np.ndarray, metric: Callable[[np.ndarray], float], repetitions: int, seed: int) -> list[float]:
    values = np.asarray([metric(sampled) for sampled in _bootstrap_rows(records, indices, repetitions, seed)])
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return [float("nan"), float("nan")]
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def _metric_ci_bundle(records: list[dict[str, Any]], indices: np.ndarray, labels: np.ndarray, predictions: np.ndarray, baseline: np.ndarray | None, metric_functions: dict[str, Callable[[np.ndarray, np.ndarray], float]], repetitions: int, seed: int) -> dict[str, Any]:
    output: dict[str, Any] = {}
    row_positions = {int(index): position for position, index in enumerate(indices.tolist())}

    def local_positions(sampled: np.ndarray) -> np.ndarray:
        return np.asarray([row_positions[int(index)] for index in sampled], dtype=np.int64)

    for offset, (name, function) in enumerate(metric_functions.items()):
        output[name] = cluster_bootstrap_ci(records, indices, lambda sampled, f=function: f(labels[local_positions(sampled)], predictions[local_positions(sampled)]), repetitions, seed + offset)
        if baseline is not None:
            output[f"delta_{name}_vs_baseline"] = cluster_bootstrap_ci(records, indices, lambda sampled, f=function: f(labels[local_positions(sampled)], predictions[local_positions(sampled)]) - f(labels[local_positions(sampled)], baseline[local_positions(sampled)]), repetitions, seed + 100 + offset)
    return output


def _classification_endpoint(records: list[dict[str, Any]], test_indices: np.ndarray, labels: np.ndarray, probabilities: np.ndarray, classes: np.ndarray, majority: int, high_bit: int, fallback_cutoff: float, bootstrap_repetitions: int, seed: int) -> dict[str, Any]:
    metrics = classification_metrics(labels, probabilities, classes, high_bit, fallback_cutoff)
    baseline = np.full(len(labels), majority, dtype=np.int64)
    baseline_metrics = {
        "accuracy": float(accuracy_score(labels, baseline)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, baseline)),
        "macro_f1": float(f1_score(labels, baseline, labels=classes, average="macro", zero_division=0)),
        "underprediction_rate": float(np.mean(baseline < labels)),
    }
    ci = _metric_ci_bundle(records, test_indices, labels, np.asarray(metrics["predictions"], dtype=np.int64), baseline, {
        "balanced_accuracy": lambda y, p: balanced_accuracy_score(y, p),
        "macro_f1": lambda y, p: f1_score(y, p, labels=classes, average="macro", zero_division=0),
        "underprediction_rate": lambda y, p: float(np.mean(p < y)),
    }, bootstrap_repetitions, seed)
    return {
        "metrics": metrics,
        "majority_baseline": {"class": int(majority), "metrics": baseline_metrics},
        "confidence_intervals_95_cluster_bootstrap": ci,
        "fallback_policy": {"type": "fixed_high_on_low_calibrated_confidence", "confidence_cutoff": float(fallback_cutoff), "high_bit": int(high_bit)},
    }


def _regression_endpoint(records: list[dict[str, Any]], test_indices: np.ndarray, labels: np.ndarray, predictions: np.ndarray, half_width: np.ndarray, mean_baseline: np.ndarray, scalar_baseline: np.ndarray | None, fallback_cutoff: float, high_value: float | np.ndarray, bootstrap_repetitions: int, seed: int, profile: bool = False) -> dict[str, Any]:
    metric_function = profile_metrics if profile else regression_metrics
    metrics = metric_function(labels, predictions)
    mean_metrics = metric_function(labels, mean_baseline)
    uncertainty = np.max(half_width, axis=1) if half_width.ndim > 1 else half_width
    result: dict[str, Any] = {
        "metrics": metrics,
        "mean_baseline": {"metrics": mean_metrics},
        "conformal_interval": {
            **interval_metrics(labels, predictions, half_width),
            "fallback_uncertainty_cutoff": float(fallback_cutoff),
            "fallback_fraction": float(np.mean(uncertainty > fallback_cutoff)),
            "test_uncertainty_mean": float(np.mean(uncertainty)),
        },
    }
    row_positions = {int(index): position for position, index in enumerate(test_indices.tolist())}

    def local_positions(sampled: np.ndarray) -> np.ndarray:
        return np.asarray([row_positions[int(index)] for index in sampled], dtype=np.int64)

    if scalar_baseline is not None:
        result["scalar_baseline"] = {"metrics": metric_function(labels, scalar_baseline)}
    ci_metrics = {
        "mae": (lambda y, p: mean_absolute_error(y, p)),
        ("r2_variance_weighted" if profile else "r2"): (lambda y, p: r2_score(y, p, multioutput="variance_weighted" if profile else "uniform_average", force_finite=True)),
    }
    result["confidence_intervals_95_cluster_bootstrap"] = _metric_ci_bundle(records, test_indices, labels, predictions, mean_baseline, ci_metrics, bootstrap_repetitions, seed)
    if scalar_baseline is not None:
        result["confidence_intervals_95_cluster_bootstrap"]["delta_mae_vs_scalar_baseline"] = cluster_bootstrap_ci(records, test_indices, lambda sampled: metric_function(labels[local_positions(sampled)], predictions[local_positions(sampled)])["mae"] - metric_function(labels[local_positions(sampled)], scalar_baseline[local_positions(sampled)])["mae"], bootstrap_repetitions, seed + 300)
    fallback = uncertainty > fallback_cutoff
    conservative = np.asarray(predictions).copy()
    conservative[fallback] = high_value
    result["conservative_fallback"] = {
        "fallback_count": int(fallback.sum()),
        "fallback_fraction": float(fallback.mean()),
        "uncertainty_cutoff": float(fallback_cutoff),
        "metrics": metric_function(labels, conservative),
        "high_precision_value": np.asarray(high_value).tolist() if isinstance(high_value, np.ndarray) else float(high_value),
    }
    return result


def _endpoint_by_dataset(records: list[dict[str, Any]], test_mask: np.ndarray, endpoint_builder: Callable[[np.ndarray, str], dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for dataset in sorted({record["source"]["dataset"] for record in records}):
        selected = test_mask & np.asarray([record["source"]["dataset"] == dataset for record in records], dtype=bool)
        indices = np.flatnonzero(selected)
        if not len(indices):
            continue
        output[dataset] = endpoint_builder(indices, dataset)
    pooled_indices = np.flatnonzero(test_mask)
    if len(pooled_indices):
        output["pooled"] = endpoint_builder(pooled_indices, "pooled")
    return output


def train_one_seed(
    records: list[dict[str, Any]],
    profile_mode: str,
    seed: int,
    trees: int,
    alpha: float,
    bootstrap_repetitions: int,
    model_dir: str | Path | None = None,
    holdout_dataset: str | None = None,
) -> dict[str, Any]:
    features, feature_names = feature_matrix(records)
    targets = target_arrays(records, profile_mode)
    masks = evaluation_masks(records, holdout_dataset)
    train_indices = np.flatnonzero(masks["development"])
    calibration_indices = np.flatnonzero(masks["calibration"])
    test_mask = masks["test"]
    test_indices = np.flatnonzero(test_mask)
    candidate_bits = sorted(int(bit) for bit in records[0]["minimum_safe_precision"]["fixed_nll_deltas"])
    high_bit, min_bit = max(candidate_bits), min(candidate_bits)

    safe_model = _forest_classifier(seed, trees).fit(features[train_indices], targets["safe_bit"][train_indices])
    safe_classes = np.asarray(candidate_bits, dtype=np.int64)
    safe_calibration = _align_probabilities(safe_model.predict_proba(features[calibration_indices]), safe_model.classes_, safe_classes)
    safe_temperature = fit_temperature(safe_calibration, targets["safe_bit"][calibration_indices], safe_classes)
    safe_calibration = calibrated_probabilities(safe_calibration, safe_temperature)
    confidence_cutoff = float(np.quantile(safe_calibration.max(axis=1), 0.10))
    safe_test_probability = calibrated_probabilities(_align_probabilities(safe_model.predict_proba(features[test_mask]), safe_model.classes_, safe_classes), safe_temperature)
    majority = Counter(targets["safe_bit"][train_indices].tolist()).most_common(1)[0][0]
    safe_results = _endpoint_by_dataset(records, test_mask, lambda indices, _: _classification_endpoint(records, indices, targets["safe_bit"][indices], safe_test_probability[np.searchsorted(test_indices, indices)], safe_classes, int(majority), high_bit, confidence_cutoff, bootstrap_repetitions, seed))
    safe_results["calibration"] = classification_metrics(
        targets["safe_bit"][calibration_indices], safe_calibration, safe_classes, high_bit, confidence_cutoff
    )
    safe_results["calibration"]["coverage"] = float(np.mean(safe_calibration.max(axis=1) >= confidence_cutoff))

    effective_model = _forest_regressor(seed, trees).fit(features[train_indices], targets["effective_bits"][train_indices])
    effective_predictions: dict[str, np.ndarray] = {}
    effective_uncertainty: dict[str, np.ndarray] = {}
    for name, mask in (("calibration", masks["calibration"]), ("test", test_mask)):
        effective_predictions[name], effective_uncertainty[name] = forest_regression_prediction(effective_model, features[mask], float(min_bit), float(high_bit))
    effective_scale = conformal_scale(targets["effective_bits"][calibration_indices], effective_predictions["calibration"], effective_uncertainty["calibration"], alpha)
    effective_half_test = effective_scale * effective_uncertainty["test"]
    effective_cutoff = float(np.quantile(effective_scale * effective_uncertainty["calibration"], 0.90))
    mean_effective = float(np.mean(targets["effective_bits"][train_indices]))
    effective_results = _endpoint_by_dataset(records, test_mask, lambda indices, _: _regression_endpoint(records, indices, targets["effective_bits"][indices], effective_predictions["test"][np.searchsorted(test_indices, indices)], effective_half_test[np.searchsorted(test_indices, indices)], np.full(len(indices), mean_effective), None, effective_cutoff, float(high_bit), bootstrap_repetitions, seed + 10))
    effective_results["calibration"] = {
        **interval_metrics(targets["effective_bits"][calibration_indices], effective_predictions["calibration"], effective_scale * effective_uncertainty["calibration"]),
        "fallback_fraction": float(np.mean(effective_scale * effective_uncertainty["calibration"] > effective_cutoff)),
    }

    profile_model = _forest_regressor(seed, trees).fit(features[train_indices], targets["group_profile"][train_indices])
    profile_predictions: dict[str, np.ndarray] = {}
    profile_uncertainty: dict[str, np.ndarray] = {}
    for name, mask in (("calibration", masks["calibration"]), ("test", test_mask)):
        profile_predictions[name], profile_uncertainty[name] = forest_regression_prediction(profile_model, features[mask], float(min_bit), float(high_bit))
    profile_scale = conformal_profile_scale(targets["group_profile"][calibration_indices], profile_predictions["calibration"], profile_uncertainty["calibration"], alpha)
    profile_half_test = profile_scale * profile_uncertainty["test"]
    profile_cutoff = float(np.quantile(np.max(profile_scale * profile_uncertainty["calibration"], axis=1), 0.90))
    mean_profile = np.mean(targets["group_profile"][train_indices], axis=0)
    scalar_profile_test = np.repeat(effective_predictions["test"][:, None], targets["group_profile"].shape[1], axis=1)
    profile_results = _endpoint_by_dataset(records, test_mask, lambda indices, _: _regression_endpoint(records, indices, targets["group_profile"][indices], profile_predictions["test"][np.searchsorted(test_indices, indices)], profile_half_test[np.searchsorted(test_indices, indices)], np.repeat(mean_profile[None, :], len(indices), axis=0), scalar_profile_test[np.searchsorted(test_indices, indices)], profile_cutoff, np.full(targets["group_profile"].shape[1], high_bit), bootstrap_repetitions, seed + 20, profile=True))
    profile_results["calibration"] = {
        **interval_metrics(targets["group_profile"][calibration_indices], profile_predictions["calibration"], profile_scale * profile_uncertainty["calibration"]),
        "fallback_fraction": float(np.mean(np.max(profile_scale * profile_uncertainty["calibration"], axis=1) > profile_cutoff)),
    }

    guard_model = _forest_regressor(seed, trees).fit(features[train_indices], targets["guard_trigger_fraction"][train_indices])
    guard_predictions: dict[str, np.ndarray] = {}
    guard_uncertainty: dict[str, np.ndarray] = {}
    for name, mask in (("calibration", masks["calibration"]), ("test", test_mask)):
        guard_predictions[name], guard_uncertainty[name] = forest_regression_prediction(guard_model, features[mask], 0.0, 1.0)
    guard_scale = conformal_scale(targets["guard_trigger_fraction"][calibration_indices], guard_predictions["calibration"], guard_uncertainty["calibration"], alpha)
    guard_half_test = guard_scale * guard_uncertainty["test"]
    guard_cutoff = float(np.quantile(guard_scale * guard_uncertainty["calibration"], 0.90))
    mean_guard = float(np.mean(targets["guard_trigger_fraction"][train_indices]))
    guard_results = _endpoint_by_dataset(records, test_mask, lambda indices, _: _regression_endpoint(records, indices, targets["guard_trigger_fraction"][indices], guard_predictions["test"][np.searchsorted(test_indices, indices)], guard_half_test[np.searchsorted(test_indices, indices)], np.full(len(indices), mean_guard), None, guard_cutoff, 1.0, bootstrap_repetitions, seed + 30))
    guard_results["calibration"] = {
        **interval_metrics(targets["guard_trigger_fraction"][calibration_indices], guard_predictions["calibration"], guard_scale * guard_uncertainty["calibration"]),
        "fallback_fraction": float(np.mean(guard_scale * guard_uncertainty["calibration"] > guard_cutoff)),
    }

    binary_distribution = {str(value): int(count) for value, count in Counter(targets["guard_triggered"][train_indices].tolist()).items()}
    result = {
        "seed": int(seed),
        "holdout_dataset": holdout_dataset,
        "training_datasets": sorted({records[index]["source"]["dataset"] for index in train_indices}),
        "split_counts": {
            "development_training": int(len(train_indices)),
            "calibration": int(len(calibration_indices)),
            "test": int(len(test_indices)),
        },
        "feature_names": feature_names,
        "candidate_bits": candidate_bits,
        "safe_bit": {"classes": safe_classes.tolist(), "calibration_temperature": float(safe_temperature), "calibration_confidence_cutoff": confidence_cutoff, "test": safe_results},
        "effective_bits": {"conformal_scale": float(effective_scale), "test": effective_results},
        "group_profile": {"profile_dimension": int(targets["group_profile"].shape[1]), "conformal_scale": float(profile_scale), "test": profile_results},
        "guard_trigger_probability": {
            "target": "per-request DP-guard trigger fraction, treated as a probability",
            "conformal_scale": float(guard_scale),
            "test": guard_results,
            "binary_any_trigger_target": {"status": "SINGLE_CLASS_TARGET" if len(binary_distribution) == 1 else "NON_DEGENERATE", "training_distribution": binary_distribution, "majority_baseline": max(binary_distribution, key=binary_distribution.get)},
        },
    }
    if model_dir is not None:
        directory = Path(model_dir)
        directory.mkdir(parents=True, exist_ok=True)
        bundle = {
            "schema_version": "qaq_predecode_predictor_bundle_v1", "seed": int(seed), "feature_names": feature_names, "candidate_bits": candidate_bits, "profile_mode": profile_mode,
            "safe_model": safe_model, "safe_classes": safe_classes, "safe_temperature": safe_temperature, "safe_confidence_cutoff": confidence_cutoff,
            "effective_model": effective_model, "effective_conformal_scale": effective_scale, "effective_fallback_cutoff": effective_cutoff,
            "profile_model": profile_model, "profile_conformal_scale": profile_scale, "profile_fallback_cutoff": profile_cutoff,
            "guard_model": guard_model, "guard_conformal_scale": guard_scale, "guard_fallback_cutoff": guard_cutoff,
        }
        file_stem = f"{holdout_dataset}-seed-{seed}" if holdout_dataset else f"seed-{seed}"
        with (directory / f"{file_stem}.pkl").open("wb") as target:
            pickle.dump(bundle, target, protocol=pickle.HIGHEST_PROTOCOL)
    return result



def predict_request_conservatively(
    bundle: dict[str, Any], feature_values: dict[str, float] | np.ndarray
) -> dict[str, Any]:
    """Return a scheduler decision that routes high-uncertainty requests high.

    The bundle must be produced by ``train_one_seed(..., model_dir=...)``.
    This function consumes only the registered pre-decode feature vector.
    """
    names = list(bundle["feature_names"])
    if isinstance(feature_values, dict):
        if set(feature_values) != set(names):
            raise ValueError("Feature names do not match predictor bundle")
        vector = np.asarray([[float(feature_values[name]) for name in names]], dtype=np.float64)
    else:
        vector = np.asarray(feature_values, dtype=np.float64).reshape(1, -1)
        if vector.shape[1] != len(names):
            raise ValueError("Feature vector width does not match predictor bundle")
    if not np.isfinite(vector).all():
        raise ValueError("Pre-decode feature vector contains non-finite values")

    classes = np.asarray(bundle["safe_classes"], dtype=np.int64)
    safe_model = bundle["safe_model"]
    safe_prob = _align_probabilities(safe_model.predict_proba(vector), safe_model.classes_, classes)
    safe_prob = calibrated_probabilities(safe_prob, float(bundle["safe_temperature"]))[0]
    safe_index = int(np.argmax(safe_prob))
    safe_bit = int(classes[safe_index])
    safe_confidence = float(safe_prob[safe_index])

    effective_prediction, effective_std = forest_regression_prediction(
        bundle["effective_model"], vector, min(bundle["candidate_bits"]), max(bundle["candidate_bits"])
    )
    profile_prediction, profile_std = forest_regression_prediction(
        bundle["profile_model"], vector, min(bundle["candidate_bits"]), max(bundle["candidate_bits"])
    )
    guard_prediction, guard_std = forest_regression_prediction(
        bundle["guard_model"], vector, 0.0, 1.0
    )
    effective_half_width = float(bundle["effective_conformal_scale"] * effective_std[0])
    profile_half_width = bundle["profile_conformal_scale"] * profile_std[0]
    guard_half_width = float(bundle["guard_conformal_scale"] * guard_std[0])
    reasons = []
    if safe_confidence < float(bundle["safe_confidence_cutoff"]):
        reasons.append("safe_bit_confidence")
    if effective_half_width > float(bundle["effective_fallback_cutoff"]):
        reasons.append("effective_bits_uncertainty")
    if float(np.max(profile_half_width)) > float(bundle["profile_fallback_cutoff"]):
        reasons.append("profile_uncertainty")
    if guard_half_width > float(bundle["guard_fallback_cutoff"]):
        reasons.append("guard_probability_uncertainty")
    fallback = bool(reasons)
    high_bit = max(bundle["candidate_bits"])
    return {
        "lane": "fixed_high" if fallback else "predicted",
        "conservative_fallback": fallback,
        "fallback_reasons": reasons,
        "predicted_safe_bit": high_bit if fallback else safe_bit,
        "predicted_effective_bits": float(high_bit if fallback else effective_prediction[0]),
        "predicted_group_profile": (np.full_like(profile_prediction[0], high_bit) if fallback else profile_prediction[0]).tolist(),
        "predicted_guard_trigger_probability": float(1.0 if fallback else guard_prediction[0]),
        "uncertainty": {
            "safe_bit": float(1.0 - safe_confidence),
            "effective_bits_half_width": effective_half_width,
            "group_profile_max_half_width": float(np.max(profile_half_width)),
            "guard_probability_half_width": guard_half_width,
        },
    }

def _primary_metric_summary(seed_results: list[dict[str, Any]], datasets: Iterable[str]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for endpoint, metric_key in (("safe_bit", "balanced_accuracy"), ("effective_bits", "mae"), ("group_profile", "mae"), ("guard_trigger_probability", "mae")):
        for dataset in (*tuple(datasets), "pooled"):
            values = [result[endpoint]["test"][dataset]["metrics"][metric_key] for result in seed_results]
            output[f"{endpoint}.{dataset}.{metric_key}"] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, "values": [float(value) for value in values]}
    return output



def predictability_gate_summary(seed_results: list[dict[str, Any]], datasets: Iterable[str]) -> dict[str, Any]:
    """Apply the preregistered gates without looking at test results for tuning."""
    per_seed: list[dict[str, Any]] = []
    dataset_names = tuple(datasets)
    for result in seed_results:
        dataset_results: dict[str, Any] = {}
        for dataset in dataset_names:
            safe = result["safe_bit"]["test"][dataset]
            safe_baseline = safe["majority_baseline"]["metrics"]
            effective = result["effective_bits"]["test"][dataset]
            profile = result["group_profile"]["test"][dataset]
            effective_baseline_mae = effective["mean_baseline"]["metrics"]["mae"]
            profile_baseline_mae = profile["mean_baseline"]["metrics"]["mae"]
            effective_improvement = (effective_baseline_mae - effective["metrics"]["mae"]) / effective_baseline_mae
            profile_improvement = (profile_baseline_mae - profile["metrics"]["mae"]) / profile_baseline_mae
            safe_pass = (
                safe["metrics"]["balanced_accuracy"] > safe_baseline["balanced_accuracy"]
                and safe["metrics"]["macro_f1"] > safe_baseline["macro_f1"]
            )
            effective_pass = effective_improvement >= 0.10 and effective["metrics"]["r2"] >= 0.10
            profile_pass = profile_improvement >= 0.10 and profile["metrics"]["r2_variance_weighted"] >= 0.10
            dataset_results[dataset] = {
                "safe_bit": {
                    "balanced_accuracy": safe["metrics"]["balanced_accuracy"],
                    "majority_balanced_accuracy": safe_baseline["balanced_accuracy"],
                    "macro_f1": safe["metrics"]["macro_f1"],
                    "majority_macro_f1": safe_baseline["macro_f1"],
                    "passes": bool(safe_pass),
                },
                "effective_bits": {
                    "mae_improvement_vs_mean_fraction": float(effective_improvement),
                    "r2": effective["metrics"]["r2"],
                    "passes": bool(effective_pass),
                },
                "group_profile": {
                    "mae_improvement_vs_mean_fraction": float(profile_improvement),
                    "r2_variance_weighted": profile["metrics"]["r2_variance_weighted"],
                    "passes": bool(profile_pass),
                },
                "passes_all_endpoints": bool(safe_pass and effective_pass and profile_pass),
            }
        per_seed.append({"seed": result["seed"], "datasets": dataset_results, "passes_all_datasets": all(item["passes_all_endpoints"] for item in dataset_results.values())})
    return {
        "per_seed": per_seed,
        "requires_all_registered_seeds_and_datasets": True,
        "predictability_established": bool(all(item["passes_all_datasets"] for item in per_seed)),
    }

def run_analysis(dataset_path: str | Path | list[str] | tuple[str, ...], output_json: str | Path, profile_mode: str = "mlp_multibit_dp_guard", seeds: tuple[int, ...] = DEFAULT_SEEDS, trees: int = 300, alpha: float = 0.10, bootstrap_repetitions: int = 1000, model_dir: str | Path | None = None, required_datasets: tuple[str, ...] = ("wikitext2", "c4_new")) -> dict[str, Any]:
    if not seeds:
        raise ValueError("At least one predictor seed is required")
    if trees < 1 or bootstrap_repetitions < 20 or not 0.0 < alpha < 1.0:
        raise ValueError("trees/repetitions must be positive and alpha must be in (0,1)")
    records = load_records(dataset_path, required_datasets)
    features, _ = feature_matrix(records)
    targets = target_arrays(records, profile_mode)
    if len(records) != len(features) or len(targets["safe_bit"]) != len(records):
        raise ValueError("Inconsistent feature/target row counts")
    folds = grouped_development_folds(records, 5)
    development_indices = np.flatnonzero(partition_masks(records)["development"])
    fold_groups = [{
        "train_documents": len({_group_key(records[development_indices[index]]) for index in train}),
        "validation_documents": len({_group_key(records[development_indices[index]]) for index in validation}),
        "overlap": bool(
            {_group_key(records[development_indices[index]]) for index in train}
            & {_group_key(records[development_indices[index]]) for index in validation}
        ),
    } for train, validation in folds]
    seed_results = [train_one_seed(records, profile_mode, int(seed), trees, alpha, bootstrap_repetitions, model_dir) for seed in seeds]
    output = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "analysis_status": "REAL_DATA_HELDOUT_CALIBRATED_PREDICTOR_EVALUATION",
        "dataset_path": [str(Path(item).resolve()) for item in dataset_path] if isinstance(dataset_path, (list, tuple)) else str(Path(dataset_path).resolve()),
        "dataset_sha256": input_sha256(dataset_path),
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "profile_mode": profile_mode,
        "config": {"seeds": [int(seed) for seed in seeds], "trees": int(trees), "min_samples_leaf": 2, "alpha": float(alpha), "bootstrap_repetitions": int(bootstrap_repetitions), "bootstrap_unit": "(dataset, source_document_id)", "fallback_coverage_target": 0.90},
        "feature_availability": "prompt token and fixed-high prefill features only; continuation and observed-route fields are forbidden",
        "split_summary": split_summary(records),
        "development_grouped_cv": {"fold_count": len(folds), "group_key": "(dataset, source.document_id)", "folds": fold_groups},
        "seed_results": seed_results,
        "seed_summary": _primary_metric_summary(seed_results, required_datasets),
        "predictability_gates": predictability_gate_summary(seed_results, required_datasets),
        "conclusions": {"mae_alone_is_not_predictability_evidence": True, "test_metrics_are_held_out_after_calibration": True, "predictability_established": predictability_gate_summary(seed_results, required_datasets)["predictability_established"], "binary_guard_any_trigger_is_degenerate": all(result["guard_trigger_probability"]["binary_any_trigger_target"]["status"] == "SINGLE_CLASS_TARGET" for result in seed_results)},
        "provenance": {"git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip(), "source_script_sha256": _sha256_file(Path(__file__).resolve()), "numpy": np.__version__},
        "limitations": ["Test confidence intervals are document-cluster bootstrap intervals, not independent-request intervals.", "The binary any-trigger target is all-positive in the registered collection; its majority baseline is the only honest result.", "Conformal intervals calibrate marginal held-out coverage; they do not prove conditional coverage for every workload.", "Scheduler fallback is evaluated as a conservative decision rule on predictor uncertainty, not as measured end-to-end serving throughput."],
    }
    destination = Path(output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as target:
        json.dump(output, target, indent=2)
        target.write("\n")
    temporary.replace(destination)
    return output


def run_leave_one_dataset_out(
    dataset_path: str | Path | list[str] | tuple[str, ...],
    output_json: str | Path,
    profile_mode: str = "mlp_multibit_dp_guard",
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    trees: int = 300,
    alpha: float = 0.10,
    bootstrap_repetitions: int = 1000,
    model_dir: str | Path | None = None,
    required_datasets: tuple[str, ...] = ("wikitext2", "c4_new"),
) -> dict[str, Any]:
    """Evaluate transfer to each source with that source fully held out."""
    if len(required_datasets) < 2:
        raise ValueError("LODO requires at least two datasets")
    if len(set(required_datasets)) != len(required_datasets):
        raise ValueError("LODO dataset names must be unique")
    if not seeds:
        raise ValueError("At least one predictor seed is required")
    if trees < 1 or bootstrap_repetitions < 20 or not 0.0 < alpha < 1.0:
        raise ValueError("trees/repetitions must be positive and alpha must be in (0,1)")

    records = load_records(dataset_path, required_datasets)
    features, _ = feature_matrix(records)
    targets = target_arrays(records, profile_mode)
    if len(records) != len(features) or len(targets["safe_bit"]) != len(records):
        raise ValueError("Inconsistent feature/target row counts")

    lodo_results: dict[str, Any] = {}
    for holdout_dataset in required_datasets:
        masks = evaluation_masks(records, holdout_dataset)
        training_datasets = sorted({
            records[index]["source"]["dataset"]
            for index in np.flatnonzero(masks["development"])
        })
        calibration_datasets = sorted({
            records[index]["source"]["dataset"]
            for index in np.flatnonzero(masks["calibration"])
        })
        test_datasets = sorted({
            records[index]["source"]["dataset"]
            for index in np.flatnonzero(masks["test"])
        })
        training_records = [
            record for record in records
            if record["source"]["dataset"] != holdout_dataset
        ]
        folds = grouped_development_folds(training_records, 5)
        development_indices = np.flatnonzero(
            partition_masks(training_records)["development"]
        )
        fold_groups = [{
            "train_documents": len({
                _group_key(training_records[development_indices[index]])
                for index in train
            }),
            "validation_documents": len({
                _group_key(training_records[development_indices[index]])
                for index in validation
            }),
            "overlap": bool(
                {
                    _group_key(training_records[development_indices[index]])
                    for index in train
                }
                & {
                    _group_key(training_records[development_indices[index]])
                    for index in validation
                }
            ),
        } for train, validation in folds]
        seed_results = [
            train_one_seed(
                records,
                profile_mode,
                int(seed),
                trees,
                alpha,
                bootstrap_repetitions,
                model_dir,
                holdout_dataset,
            )
            for seed in seeds
        ]
        gates = predictability_gate_summary(seed_results, (holdout_dataset,))
        lodo_results[holdout_dataset] = {
            "holdout_dataset": holdout_dataset,
            "training_datasets": training_datasets,
            "calibration_datasets": calibration_datasets,
            "test_datasets": test_datasets,
            "split_integrity": {
                "holdout_absent_from_training": holdout_dataset not in training_datasets,
                "holdout_absent_from_calibration": holdout_dataset not in calibration_datasets,
                "test_contains_only_holdout": test_datasets == [holdout_dataset],
            },
            "training_split_summary": split_summary(training_records),
            "development_grouped_cv": {
                "fold_count": len(folds),
                "group_key": "(dataset, source.document_id)",
                "folds": fold_groups,
            },
            "seed_results": seed_results,
            "seed_summary": _primary_metric_summary(seed_results, (holdout_dataset,)),
            "predictability_gates": gates,
        }

    overall_predictability = all(
        result["predictability_gates"]["predictability_established"]
        for result in lodo_results.values()
    )
    binary_guard_degenerate = all(
        seed_result["guard_trigger_probability"]["binary_any_trigger_target"]["status"]
        == "SINGLE_CLASS_TARGET"
        for result in lodo_results.values()
        for seed_result in result["seed_results"]
    )
    output = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "analysis_status": "REAL_DATA_LEAVE_ONE_DATASET_OUT_HELDOUT_CALIBRATED_PREDICTOR_EVALUATION",
        "evaluation_mode": "leave_one_dataset_out",
        "dataset_path": [str(Path(item).resolve()) for item in dataset_path]
        if isinstance(dataset_path, (list, tuple))
        else str(Path(dataset_path).resolve()),
        "dataset_sha256": input_sha256(dataset_path),
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "datasets": list(required_datasets),
        "profile_mode": profile_mode,
        "config": {
            "seeds": [int(seed) for seed in seeds],
            "trees": int(trees),
            "min_samples_leaf": 2,
            "alpha": float(alpha),
            "bootstrap_repetitions": int(bootstrap_repetitions),
            "bootstrap_unit": "(dataset, source_document_id)",
            "fallback_coverage_target": 0.90,
        },
        "feature_availability": "prompt token and fixed-high prefill features only; continuation and observed-route fields are forbidden",
        "split_summary": split_summary(records),
        "lodo_results": lodo_results,
        "predictability_gates": {
            "requires_all_holdout_datasets_and_seeds": True,
            "per_holdout_dataset": {
                dataset: result["predictability_gates"]
                for dataset, result in lodo_results.items()
            },
            "predictability_established": bool(overall_predictability),
        },
        "conclusions": {
            "mae_alone_is_not_predictability_evidence": True,
            "test_metrics_are_held_out_after_calibration": True,
            "predictability_established": bool(overall_predictability),
            "binary_guard_any_trigger_is_degenerate": binary_guard_degenerate,
            "scheduler_integration_considered": False,
            "scheduler_integration_allowed": False,
        },
        "provenance": {
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip(),
            "source_script_sha256": _sha256_file(Path(__file__).resolve()),
            "numpy": np.__version__,
        },
        "limitations": [
            "Each LODO model trains and calibrates only on the other registered datasets.",
            "Test confidence intervals are document-cluster bootstrap intervals, not independent-request intervals.",
            "The binary any-trigger target is all-positive in the registered collection; its majority baseline is the only honest result.",
            "Conformal intervals calibrate marginal held-out coverage; they do not prove conditional coverage for every workload.",
            "No scheduler integration or end-to-end serving claim is made by this evaluation.",
        ],
    }
    destination = Path(output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as target:
        json.dump(output, target, indent=2)
        target.write("\n")
    temporary.replace(destination)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate prompt/prefill-only QAQ predictors with held-out calibration.")
    parser.add_argument("--dataset_path", nargs="+", required=True, help="one or more v2 JSONL files or artifact directories")
    parser.add_argument("--datasets", nargs="+", default=["wikitext2", "c4_new"], help="dataset names required in the input")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--profile_mode", default="mlp_multibit_dp_guard")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--bootstrap_repetitions", type=int, default=1000)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--evaluation_mode", choices=("pooled", "lodo"), default="pooled")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.evaluation_mode == "lodo":
        result = run_leave_one_dataset_out(args.dataset_path, args.output_json, args.profile_mode, tuple(args.seeds), args.trees, args.alpha, args.bootstrap_repetitions, args.model_dir, tuple(args.datasets))
    else:
        result = run_analysis(args.dataset_path, args.output_json, args.profile_mode, tuple(args.seeds), args.trees, args.alpha, args.bootstrap_repetitions, args.model_dir, tuple(args.datasets))
    print(json.dumps({"analysis_status": result["analysis_status"], "output_json": str(Path(args.output_json)), "request_count": sum(item["request_count"] for item in result["split_summary"].values()), "test_request_count": result["split_summary"]["test"]["request_count"], "seeds": args.seeds, "binary_guard_any_trigger_degenerate": result["conclusions"]["binary_guard_any_trigger_is_degenerate"]}, indent=2))


if __name__ == "__main__":
    main()
