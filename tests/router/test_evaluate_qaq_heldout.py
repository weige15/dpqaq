from pathlib import Path
import sys

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_qaq_heldout import (
    MODES,
    attach_nll_deltas,
    build_token_windows,
    precision_counts,
    smallest_safe_bits,
    summarize_precision_counts,
    validate_fixed_high_precision,
)


def test_build_token_windows_uses_documented_non_overlapping_subset():
    tokens = torch.arange(24)

    windows = build_token_windows(tokens, context_length=4, start=2, count=3)

    assert [window.tolist() for window in windows] == [
        [8, 9, 10, 11],
        [12, 13, 14, 15],
        [16, 17, 18, 19],
    ]


def test_build_token_windows_rejects_incomplete_requested_subset():
    with pytest.raises(ValueError, match="only 2 full windows"):
        build_token_windows(torch.arange(11), context_length=4, start=1, count=2)


def test_smallest_safe_bits_uses_real_relative_output_error():
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    weights = {
        3: torch.tensor([[0.95, 0.0], [0.0, 0.50]]),
        4: torch.tensor([[0.99, 0.0], [0.0, 0.95]]),
        6: torch.eye(2),
    }

    required = smallest_safe_bits(
        x,
        weights,
        reference_bit=6,
        error_threshold=0.02,
    )

    assert required.tolist() == [4, 6]


def test_precision_accounting_reports_under_over_exact_and_bit_gaps():
    counts = precision_counts(
        selected=torch.tensor([3, 6, 4, 6]),
        required=torch.tensor([4, 4, 4, 6]),
    )
    summary = summarize_precision_counts(counts)

    assert summary["decision_count"] == 4
    assert summary["under_precision_count"] == 1
    assert summary["over_precision_count"] == 1
    assert summary["exact_precision_count"] == 2
    assert summary["under_precision_rate"] == 0.25
    assert summary["over_precision_rate"] == 0.25
    assert summary["mean_signed_bit_gap"] == 0.25
    assert summary["mean_absolute_bit_gap"] == 0.75


def test_attach_nll_deltas_uses_same_example_fixed_high_baseline():
    mode_results = {}
    for mode in MODES:
        offset = 0.0 if mode == "fixed_high" else 0.2
        mode_results[mode] = {
            "mean_nll": 1.5 + offset,
            "perplexity": 4.0 + offset,
            "examples": [
                {
                    "example_index": 0,
                    "mean_nll": 1.0 + offset,
                    "perplexity": 3.0 + offset,
                },
                {
                    "example_index": 1,
                    "mean_nll": 2.0 + offset,
                    "perplexity": 5.0 + offset,
                },
            ],
        }

    attach_nll_deltas(mode_results)

    assert mode_results["fixed_high"]["mean_nll_delta_vs_fixed_high"] == 0.0
    assert mode_results["fixed_high"]["examples"][0]["nll_delta_vs_fixed_high"] == 0.0
    assert mode_results["mlp_multibit"]["mean_nll_delta_vs_fixed_high"] == pytest.approx(0.2)
    assert mode_results["mlp_multibit"]["examples"][1]["nll_delta_vs_fixed_high"] == pytest.approx(0.2)


def test_fixed_high_allows_over_precision_but_rejects_under_precision():
    validate_fixed_high_precision({
        "under_precision_count": 0,
        "over_precision_count": 10,
    })

    with pytest.raises(RuntimeError, match="executed below"):
        validate_fixed_high_precision({
            "under_precision_count": 1,
            "over_precision_count": 0,
        })


def test_all_five_required_modes_are_present():
    assert MODES == (
        "fixed_low",
        "fixed_high",
        "dp_threshold_only",
        "mlp_multibit",
        "mlp_multibit_dp_guard",
    )
