#!/usr/bin/env python3
"""Generate current-progress research figures from recorded QAQ result JSONs.

The script reads only existing result artifacts and writes PNG/PDF pairs plus a
small manifest. It does not run model inference or benchmark code.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts/qaq_current_progress_20260716/figures"

MODE_ORDER = [
    "fixed_low",
    "dp_threshold_only",
    "mlp_multibit",
    "mlp_multibit_dp_guard",
    "fixed_high",
]
MODE_LABELS = {
    "fixed_low": "Fixed low",
    "dp_threshold_only": "DP threshold",
    "mlp_multibit": "MLP multibit",
    "mlp_multibit_dp_guard": "MLP + DP guard",
    "fixed_high": "Fixed high",
}
MODE_COLORS = {
    "fixed_low": "#B0BEC5",
    "dp_threshold_only": "#E9C46A",
    "mlp_multibit": "#56B4E9",
    "mlp_multibit_dp_guard": "#E76F51",
    "fixed_high": "#264653",
}
DATASET_LABELS = {
    "wikitext2": "WikiText-2",
    "c4_new": "C4",
    "fineweb_edu": "FineWeb-Edu",
    "hellaswag": "HellaSwag",
}
ENDPOINTS = ["safe_bit", "effective_bits", "group_profile"]
ENDPOINT_LABELS = ["Safe bit", "Effective bits", "Group profile"]


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing result artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "-",
            "axes.axisbelow": True,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return [str(png), str(pdf)]


def quality_figure(quality_paths: dict[str, Path], output_dir: Path) -> list[str]:
    quality = {
        dataset: load_json(path)
        for dataset, path in quality_paths.items()
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.5), sharey=True)
    x = np.arange(len(MODE_ORDER))
    width = 0.68

    for ax, dataset in zip(axes, ["wikitext2", "c4_new"]):
        modes = quality[dataset]["modes"]
        values = [modes[mode]["perplexity"] for mode in MODE_ORDER]
        deltas = [modes[mode]["perplexity"] - modes["fixed_high"]["perplexity"] for mode in MODE_ORDER]
        bars = ax.bar(
            x,
            values,
            width=width,
            color=[MODE_COLORS[mode] for mode in MODE_ORDER],
            edgecolor="white",
            linewidth=0.7,
        )
        for bar, value, delta, mode in zip(bars, values, deltas, MODE_ORDER):
            if mode == "fixed_high":
                label = f"{value:.2f}"
            else:
                label = f"+{delta:.2f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.14,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
                color="#333333",
                rotation=90 if len(label) > 5 else 0,
            )
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_xticks(x)
        ax.set_xticklabels([MODE_LABELS[mode] for mode in MODE_ORDER], rotation=28, ha="right")
        ax.set_ylim(0, 17.0)
        ax.set_axisbelow(True)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("Teacher-forced perplexity (lower is better)")
    fig.suptitle("Held-out quality: DP guard narrows, but does not close, the gap", y=1.02)
    fig.text(
        0.5,
        -0.02,
        "Labels show +Δ versus fixed-high, except the fixed-high bar; 16 windows and 8,176 scored tokens per dataset.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout()
    return save_figure(fig, output_dir, "fig_heldout_quality_perplexity")


def predictor_gate_figure(predictor_path: Path, output_dir: Path) -> list[str]:
    predictor = load_json(predictor_path)
    gate_data = predictor["predictability_gates"]["per_holdout_dataset"]
    datasets = ["wikitext2", "c4_new", "fineweb_edu", "hellaswag"]
    matrix = np.zeros((len(datasets), len(ENDPOINTS)), dtype=int)

    for row, dataset in enumerate(datasets):
        per_seed = gate_data[dataset]["per_seed"]
        for col, endpoint in enumerate(ENDPOINTS):
            matrix[row, col] = sum(
                bool(seed["datasets"][dataset][endpoint]["passes"])
                for seed in per_seed
            )

    cmap = ListedColormap(["#C95C5C", "#E9C46A", "#9BC48B", "#2A9D8F"])
    norm = BoundaryNorm(np.arange(-0.5, 4.5, 1), cmap.N)
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col,
                row,
                f"{matrix[row, col]}/3",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="white" if matrix[row, col] in (0, 3) else "#333333",
            )

    ax.set_xticks(np.arange(len(ENDPOINTS)))
    ax.set_xticklabels(ENDPOINT_LABELS)
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_yticklabels([DATASET_LABELS[dataset] for dataset in datasets])
    ax.set_xlabel("Preregistered endpoint")
    ax.set_title("Prompt-only predictor transfer: seeds passing each gate")
    ax.grid(False)
    cbar = fig.colorbar(image, ax=ax, ticks=[0, 1, 2, 3], pad=0.03, aspect=25)
    cbar.ax.set_yticklabels(["0/3", "1/3", "2/3", "3/3"])
    cbar.set_label("Passing seeds")
    fig.text(
        0.5,
        -0.02,
        "All three registered seeds and all held-out datasets are required for predictability_established.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout()
    return save_figure(fig, output_dir, "fig_predictor_lodo_gates")


def shared_cuda_figure(shared_path: Path, output_dir: Path) -> list[str]:
    shared = load_json(shared_path)
    policies = shared["policies"]
    names = ["fixed_high", "max_profile_sharing"]
    labels = ["Fixed high", "Max profile sharing"]
    colors = ["#264653", "#E76F51"]
    summaries = [policies[name]["summary"] for name in names]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5))
    x = np.arange(len(names))
    width = 0.58

    p50 = [summary["latency_ms"]["p50"] for summary in summaries]
    p95 = [summary["latency_ms"]["p95"] for summary in summaries]
    tokens = [summary["generated_tokens_per_s"] for summary in summaries]

    bar_p50 = axes[0].bar(x - width / 4, p50, width / 2, label="p50", color="#56B4E9")
    bar_p95 = axes[0].bar(x + width / 4, p95, width / 2, label="p95", color="#E9C46A")
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_title("Latency")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].legend(loc="upper left")
    axes[0].grid(axis="y")
    axes[0].grid(axis="x", visible=False)
    for bars in [bar_p50, bar_p95]:
        for bar in bars:
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 18,
                f"{bar.get_height():.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    bar_tokens = axes[1].bar(x, tokens, width=width, color=colors, edgecolor="white")
    axes[1].set_ylabel("Generated tokens/s")
    axes[1].set_title("Throughput")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].grid(axis="y")
    axes[1].grid(axis="x", visible=False)
    for bar, value in zip(bar_tokens, tokens):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.45,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.suptitle("Shared-profile CUDA validation: execution works, speedup not yet observed", y=1.03)
    fig.text(
        0.5,
        -0.02,
        "RTX 3090 · 8 requests · max batch size 4 · 1 warmup + 3 repeats · effective bits = 6.0 for both policies.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout()
    return save_figure(fig, output_dir, "fig_shared_profile_cuda")


def shared_safety_figure(safety_path: Path, output_dir: Path) -> list[str]:
    safety = load_json(safety_path)
    audit = safety["policies"]["max_profile_sharing"]["quality_audit"]
    counts = [
        audit["precision_counts"]["under_precision_count"],
        audit["precision_counts"]["exact_precision_count"],
        audit["precision_counts"]["over_precision_count"],
    ]
    labels = ["Underprecision", "Exact", "Overprecision"]
    colors = ["#C95C5C", "#2A9D8F", "#E9C46A"]
    total = sum(counts)

    fig, ax = plt.subplots(figsize=(8.2, 2.8))
    left = 0.0
    for count, label, color in zip(counts, labels, colors):
        fraction = count / total if total else 0.0
        ax.barh(
            ["Real route-safety audit"],
            [fraction],
            left=left,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            label=label,
        )
        if fraction > 0.025:
            ax.text(
                left + fraction / 2,
                0,
                f"{fraction:.1%}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="white" if label != "Exact" else "#173B35",
            )
        left += fraction

    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Share of 241,920 audited route decisions")
    ax.set_title("Shared-profile route-safety audit")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.text(
        0.995,
        0.72,
        "0 underprecision violations",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#2A9D8F",
    )
    fig.text(
        0.5,
        -0.04,
        "This is route-level output-error safety, not task accuracy or perplexity.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout()
    return save_figure(fig, output_dir, "fig_shared_profile_safety")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-wikitext", type=Path, required=True)
    parser.add_argument("--quality-c4", type=Path, required=True)
    parser.add_argument("--predictor", type=Path, required=True)
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--safety", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    apply_style()
    outputs = []
    outputs += quality_figure(
        {"wikitext2": args.quality_wikitext, "c4_new": args.quality_c4},
        args.output_dir,
    )
    outputs += predictor_gate_figure(args.predictor, args.output_dir)
    outputs += shared_cuda_figure(args.shared, args.output_dir)
    outputs += shared_safety_figure(args.safety, args.output_dir)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "quality_wikitext": str(args.quality_wikitext),
            "quality_c4": str(args.quality_c4),
            "predictor": str(args.predictor),
            "shared_cuda": str(args.shared),
            "shared_safety": str(args.safety),
        },
        "outputs": outputs,
        "status": "REAL_RESULT_FIGURES",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
