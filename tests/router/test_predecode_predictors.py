import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.predecode_predictors import (
    _metric_ci_bundle,
    evaluation_masks,
    feature_matrix,
    grouped_development_folds,
    predict_request_conservatively,
    target_arrays,
)


def _record(dataset, document, partition, index):
    profile = [3.0 + (index % 3), 4.0 + (index % 2)]
    return {
        "schema_version": "qaq_request_demand_v2",
        "request_id": f"{dataset}-{partition}-{index}",
        "source": {
            "dataset": dataset,
            "document_id": document,
            "partition": partition,
        },
        "prompt_features": {
            "prompt_length_tokens": float(128 + index),
            "fixed_high_prompt_nll": float(index) / 10.0,
        },
        "minimum_safe_precision": {
            "requested_bit": int(3 + index % 3),
            "fixed_nll_deltas": {"3": 0.2, "4": 0.05, "5": 0.01, "6": 0.0},
        },
        "observed_qaq_profiles": {
            "mlp_multibit_dp_guard": {
                "group_expected_bits": profile,
                "effective_bits": float(np.mean(profile)),
            }
        },
        "quality_by_mode": {
            "mlp_multibit_dp_guard": {
                "dp_guard_trigger_fraction": float(index) / 100.0,
                "dp_guard_trigger_count": int(index > 0),
            }
        },
    }


def test_grouped_development_folds_keep_source_documents_disjoint():
    records = [
        _record("wikitext2", f"w-{index}", "development", index)
        for index in range(6)
    ] + [
        _record("c4_new", f"c-{index}", "development", index)
        for index in range(6)
    ]
    folds = grouped_development_folds(records, folds=3)

    assert len(folds) == 3
    for train, validation in folds:
        train_groups = {(records[index]["source"]["dataset"], records[index]["source"]["document_id"]) for index in train}
        validation_groups = {(records[index]["source"]["dataset"], records[index]["source"]["document_id"]) for index in validation}
        assert train_groups.isdisjoint(validation_groups)


def test_feature_matrix_rejects_post_decode_feature_names():
    records = [_record("wikitext2", "d", "development", 0)]
    records[0]["prompt_features"]["observed_route_fraction"] = 0.0

    try:
        feature_matrix(records)
    except ValueError as error:
        assert "not available before decode" in str(error)
    else:
        raise AssertionError("post-decode feature was accepted")


def test_target_arrays_include_non_degenerate_guard_fraction():
    records = [_record("wikitext2", "d", "development", 0)]
    targets = target_arrays(records, "mlp_multibit_dp_guard")

    assert targets["guard_trigger_fraction"].tolist() == [0.0]
    assert targets["guard_triggered"].tolist() == [False]


def test_cluster_bootstrap_ci_uses_global_record_indices():
    records = [
        _record("wikitext2", "d0", "test", 0),
        _record("wikitext2", "d1", "test", 1),
        _record("c4_new", "d2", "test", 2),
    ]
    indices = np.asarray([1, 2])
    labels = np.asarray([10.0, 20.0])
    predictions = np.asarray([11.0, 19.0])
    result = _metric_ci_bundle(
        records,
        indices,
        labels,
        predictions,
        None,
        {"mae": lambda y, p: float(np.mean(np.abs(y - p)))},
        repetitions=20,
        seed=0,
    )

    assert result["mae"][0] <= 1.0 <= result["mae"][1]


def test_scheduler_decision_falls_back_to_fixed_high_on_uncertainty():
    x = np.arange(20, dtype=np.float64).reshape(-1, 1)
    y_class = np.asarray([3, 4, 5, 6] * 5)
    safe_model = RandomForestClassifier(n_estimators=10, random_state=0).fit(x, y_class)
    effective_model = RandomForestRegressor(n_estimators=10, random_state=0).fit(x, y_class.astype(float))
    profile_model = RandomForestRegressor(n_estimators=10, random_state=0).fit(
        x, np.column_stack([y_class, y_class.astype(float)])
    )
    guard_model = RandomForestRegressor(n_estimators=10, random_state=0).fit(x, y_class / 10.0)
    bundle = {
        "feature_names": ["prompt_length_tokens"],
        "candidate_bits": [3, 4, 5, 6],
        "safe_model": safe_model,
        "safe_classes": np.asarray([3, 4, 5, 6]),
        "safe_temperature": 1.0,
        "safe_confidence_cutoff": 1.1,
        "effective_model": effective_model,
        "effective_conformal_scale": 1.0,
        "effective_fallback_cutoff": 0.0,
        "profile_model": profile_model,
        "profile_conformal_scale": 1.0,
        "profile_fallback_cutoff": 0.0,
        "guard_model": guard_model,
        "guard_conformal_scale": 1.0,
        "guard_fallback_cutoff": 0.0,
    }

    result = predict_request_conservatively(bundle, {"prompt_length_tokens": 3.0})

    assert result["conservative_fallback"] is True
    assert result["lane"] == "fixed_high"
    assert result["predicted_safe_bit"] == 6
    assert result["predicted_effective_bits"] == 6.0


def test_lodo_masks_exclude_heldout_dataset_from_training_and_calibration():
    records = [
        _record(dataset, f"{dataset}-{partition}", partition, index)
        for index, (dataset, partition) in enumerate(
            (("wikitext2", "development"), ("wikitext2", "calibration"), ("wikitext2", "test"),
             ("c4_new", "development"), ("c4_new", "calibration"), ("c4_new", "test"),
             ("fineweb_edu", "development"), ("fineweb_edu", "calibration"), ("fineweb_edu", "test"))
        )
    ]

    masks = evaluation_masks(records, "c4_new")
    train_datasets = {records[index]["source"]["dataset"] for index in np.flatnonzero(masks["development"])}
    calibration_datasets = {records[index]["source"]["dataset"] for index in np.flatnonzero(masks["calibration"])}
    test_datasets = {records[index]["source"]["dataset"] for index in np.flatnonzero(masks["test"])}

    assert train_datasets == {"wikitext2", "fineweb_edu"}
    assert calibration_datasets == {"wikitext2", "fineweb_edu"}
    assert test_datasets == {"c4_new"}
