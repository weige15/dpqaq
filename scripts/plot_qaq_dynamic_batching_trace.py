import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


WORKLOAD_ORDER = ["chat", "math", "code", "summarization"]
POLICY_LABELS = {
    "ordinary_dynamic_batching": "ordinary",
    "scalar_budget_batching": "scalar",
    "block_profile_batching": "block",
    "max_profile_sharing": "max share",
    "quantile_profile_sharing": "quantile share",
}
COLORS = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51", "#0072B2", "#56B4E9", "#8C8C8C"]
SIM_NOTE = "SIMULATED_ONLY"


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linestyle": "-",
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate first profile-diversity and fragmentation plots from a QAQ trace."
    )
    parser.add_argument("--trace_jsonl", required=True)
    parser.add_argument("--simulation_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--coarse_group_size", type=int, default=4)
    parser.add_argument("--coarse_bucket_size", type=float, default=0.25)
    parser.add_argument("--top_profiles", type=int, default=12)
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"No trace records found in {path}")
    return records


def route_sort_key(route: str) -> tuple[int, str]:
    layer, name = route.split(".", 1)
    return int(layer), name


def route_order(record: dict[str, Any]) -> list[str]:
    return sorted(record["per_layer_bit_counts"], key=route_sort_key)


def expected_bit_for_counts(counts: dict[str, int]) -> float:
    total = sum(int(count) for count in counts.values())
    if total <= 0:
        return 0.0
    return sum(int(bit) * int(count) for bit, count in counts.items()) / total


def majority_profile(record: dict[str, Any], routes: list[str]) -> tuple[int, ...]:
    bits = []
    for route in routes:
        counts = record["per_layer_bit_counts"][route]
        bit, _ = max(
            ((int(bit), int(count)) for bit, count in counts.items()),
            key=lambda item: (item[1], item[0]),
        )
        bits.append(bit)
    return tuple(bits)


def coarse_profile(
    record: dict[str, Any],
    routes: list[str],
    group_size: int,
    bucket_size: float,
) -> tuple[float, ...]:
    grouped = defaultdict(list)
    for route in routes:
        layer = int(route.split(".", 1)[0])
        bit = expected_bit_for_counts(record["per_layer_bit_counts"][route])
        grouped[layer // group_size].append(bit)
    profile = []
    for group_id in sorted(grouped):
        avg = statistics.fmean(grouped[group_id])
        profile.append(round(round(avg / bucket_size) * bucket_size, 2))
    return tuple(profile)


def profile_label(profile: tuple[float, ...]) -> str:
    payload = json.dumps(profile, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]


def scalar_bucket(value: float, bucket_size: float) -> float:
    return math.floor(value / bucket_size) * bucket_size


def workload_names(records: list[dict[str, Any]]) -> list[str]:
    present = {record["workload_type"] for record in records}
    ordered = [name for name in WORKLOAD_ORDER if name in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / f"{stem}.pdf"
    png = output_dir / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return [str(pdf), str(png)]


def plot_bits_by_workload(records: list[dict[str, Any]], output_dir: Path) -> list[str]:
    workloads = workload_names(records)
    values = [[float(r["average_selected_bit"]) for r in records if r["workload_type"] == workload] for workload in workloads]

    fig, ax = plt.subplots(figsize=(6.75, 2.75))
    box = ax.boxplot(values, patch_artist=True, tick_labels=workloads, widths=0.55, showfliers=False)
    for patch, color in zip(box["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.28)
        patch.set_edgecolor(color)
    for key in ["whiskers", "caps", "medians"]:
        for artist in box[key]:
            artist.set_color("#4A4A4A")

    rng = np.random.default_rng(7)
    for idx, workload_values in enumerate(values, start=1):
        jitter = rng.normal(0.0, 0.035, size=len(workload_values))
        ax.scatter(
            np.full(len(workload_values), idx) + jitter,
            workload_values,
            s=13,
            color=COLORS[(idx - 1) % len(COLORS)],
            alpha=0.55,
            linewidths=0,
        )

    ax.set_ylabel("Average selected bit")
    ax.set_title("QAQ Precision Budget by Workload")
    ax.set_ylim(5.55, 6.03)
    ax.text(0.99, 0.04, "Real single-request trace", transform=ax.transAxes, ha="right", va="bottom", color="#555")
    return save_figure(fig, output_dir, "fig_bits_by_workload")


def plot_profile_fragmentation(
    records: list[dict[str, Any]],
    routes: list[str],
    output_dir: Path,
    group_size: int,
    bucket_size: float,
) -> tuple[list[str], dict[str, Any]]:
    workloads = workload_names(records)
    categories = ["scalar 0.25b", "majority", "coarse expected"]
    metrics: dict[str, dict[str, int]] = {}
    for workload in workloads:
        subset = [r for r in records if r["workload_type"] == workload]
        metrics[workload] = {
            "scalar 0.25b": len({scalar_bucket(float(r["average_selected_bit"]), 0.25) for r in subset}),
            "majority": len({majority_profile(r, routes) for r in subset}),
            "coarse expected": len({coarse_profile(r, routes, group_size, bucket_size) for r in subset}),
        }

    fig, ax = plt.subplots(figsize=(6.75, 2.75))
    x = np.arange(len(workloads))
    width = 0.23
    for idx, category in enumerate(categories):
        offsets = x + (idx - 1) * width
        vals = [metrics[workload][category] for workload in workloads]
        bars = ax.bar(offsets, vals, width=width, color=COLORS[idx], label=category, edgecolor="white", linewidth=0.6)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.25, str(val), ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(workloads)
    ax.set_ylabel("Unique profiles")
    ax.set_title("Profile Fragmentation by Workload")
    ax.legend(ncol=3, loc="upper left")
    ax.set_ylim(0, max(max(v.values()) for v in metrics.values()) + 2.5)
    return save_figure(fig, output_dir, "fig_profile_fragmentation_by_workload"), metrics


def plot_coarse_profile_heatmap(
    records: list[dict[str, Any]],
    routes: list[str],
    output_dir: Path,
    group_size: int,
    bucket_size: float,
    top_profiles: int,
) -> tuple[list[str], dict[str, Any]]:
    workloads = workload_names(records)
    profile_by_record = {
        record["request_id"]: coarse_profile(record, routes, group_size, bucket_size)
        for record in records
    }
    counts = Counter(profile_by_record.values())
    selected_profiles = [profile for profile, _ in counts.most_common(top_profiles)]
    labels = [profile_label(profile) for profile in selected_profiles]
    if len(counts) > len(selected_profiles):
        labels.append("other")

    matrix = np.zeros((len(workloads), len(labels)), dtype=int)
    for row_i, workload in enumerate(workloads):
        for record in records:
            if record["workload_type"] != workload:
                continue
            profile = profile_by_record[record["request_id"]]
            if profile in selected_profiles:
                col_i = selected_profiles.index(profile)
            else:
                col_i = len(labels) - 1
            matrix[row_i, col_i] += 1

    fig, ax = plt.subplots(figsize=(6.75, 2.9))
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(workloads)))
    ax.set_yticklabels(workloads)
    ax.set_xlabel("Coarse profile id")
    ax.set_ylabel("Workload")
    ax.set_title("Coarse Profile Occupancy")
    for row_i in range(matrix.shape[0]):
        for col_i in range(matrix.shape[1]):
            value = matrix[row_i, col_i]
            if value:
                ax.text(col_i, row_i, str(value), ha="center", va="center", fontsize=7, color="#111")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Requests")

    metadata = {
        "top_profile_ids": labels,
        "top_profile_counts": {profile_label(profile): counts[profile] for profile in selected_profiles},
        "total_unique_coarse_profiles": len(counts),
    }
    return save_figure(fig, output_dir, "fig_coarse_profile_occupancy_heatmap"), metadata


def plot_simulated_tradeoff(simulation: dict[str, Any], output_dir: Path) -> tuple[list[str], str]:
    policies = simulation["policies"]
    names = list(policies)
    x = [float(policies[name]["requests_per_s"]) for name in names]
    y = [float(policies[name]["latency_ms"]["p95"]) / 1000.0 for name in names]
    sizes = [60 + 50 * float(policies[name]["mean_batch_size"]) for name in names]

    best_latency = min(names, key=lambda name: float(policies[name]["latency_ms"]["p95"]))
    best_rps = max(names, key=lambda name: float(policies[name]["requests_per_s"]))
    candidate = best_latency if best_latency == best_rps else best_rps

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for idx, name in enumerate(names):
        color = "#E76F51" if name == candidate else COLORS[idx % len(COLORS)]
        ax.scatter(x[idx], y[idx], s=sizes[idx], color=color, alpha=0.78, edgecolor="white", linewidth=0.8)
        ax.text(x[idx] + 0.006, y[idx], POLICY_LABELS.get(name, name), va="center", fontsize=8.2)
    ax.set_xlabel("Simulated requests/sec")
    ax.set_ylabel("Simulated p95 latency (s)")
    ax.set_title("Simulator Policy Tradeoff")
    ax.text(0.98, 0.96, SIM_NOTE, transform=ax.transAxes, ha="right", va="top", color="#B03A2E", fontweight="bold")
    return save_figure(fig, output_dir, "fig_simulated_policy_tradeoff"), candidate


def plot_lane_occupancy(simulation: dict[str, Any], output_dir: Path) -> list[str]:
    policies = simulation["policies"]
    names = list(policies)
    lane_names = sorted({lane for data in policies.values() for lane in data["lane_occupancy"]})
    y = np.arange(len(names))
    left = np.zeros(len(names))

    fig, ax = plt.subplots(figsize=(6.75, 3.0))
    for lane_i, lane in enumerate(lane_names):
        vals = [int(policies[name]["lane_occupancy"].get(lane, 0)) for name in names]
        ax.barh(y, vals, left=left, color=COLORS[lane_i % len(COLORS)], label=lane, edgecolor="white", linewidth=0.5)
        left += np.array(vals)

    ax.set_yticks(y)
    ax.set_yticklabels([POLICY_LABELS.get(name, name) for name in names])
    ax.invert_yaxis()
    ax.set_xlabel("Requests assigned to lane")
    ax.set_title("Lane Occupancy by Simulated Policy")
    ax.text(0.98, 0.04, SIM_NOTE, transform=ax.transAxes, ha="right", va="bottom", color="#B03A2E", fontweight="bold")
    ax.legend(ncol=2, bbox_to_anchor=(1.0, 1.02), loc="lower right")
    return save_figure(fig, output_dir, "fig_lane_occupancy_by_policy")


def write_manifest(
    output_dir: Path,
    records: list[dict[str, Any]],
    simulation: dict[str, Any],
    generated_files: list[str],
    fragmentation: dict[str, Any],
    heatmap_metadata: dict[str, Any],
    candidate: str,
) -> None:
    workloads = workload_names(records)
    summary = {
        "status": "PLOTS_COMPLETE_REAL_TRACE_SIMULATOR_ONLY_FOR_POLICY_FIGURES",
        "trace_record_count": len(records),
        "workload_counts": dict(Counter(record["workload_type"] for record in records)),
        "figures": generated_files,
        "profile_fragmentation_by_workload": fragmentation,
        "coarse_profile_heatmap": heatmap_metadata,
        "candidate_policy_for_real_replay": candidate,
        "candidate_basis": (
            "Selected from simulator aggregates only. This identifies a validation target, "
            "not a measured batching speedup."
        ),
        "simulated_policy_metrics": {
            name: {
                "batch_count": data["batch_count"],
                "mean_batch_size": data["mean_batch_size"],
                "p95_latency_ms": data["latency_ms"]["p95"],
                "requests_per_s": data["requests_per_s"],
                "lane_occupancy": data["lane_occupancy"],
            }
            for name, data in simulation["policies"].items()
        },
        "workload_average_selected_bit": {
            workload: summarize([
                float(record["average_selected_bit"])
                for record in records
                if record["workload_type"] == workload
            ])
            for workload in workloads
        },
        "limitations": [
            "Profile plots use real single-request QAQ trace records.",
            "Policy tradeoff and lane occupancy plots use SIMULATED_ONLY scheduler output.",
            "No quality metric, transfer-byte metric, or kernel-switch metric is validated by these plots.",
        ],
    }
    (output_dir / "plot_manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.trace_jsonl)
    simulation = json.loads(Path(args.simulation_json).read_text())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    routes = route_order(records[0])
    generated: list[str] = []
    generated.extend(plot_bits_by_workload(records, output_dir))
    files, fragmentation = plot_profile_fragmentation(
        records,
        routes,
        output_dir,
        group_size=args.coarse_group_size,
        bucket_size=args.coarse_bucket_size,
    )
    generated.extend(files)
    files, heatmap_metadata = plot_coarse_profile_heatmap(
        records,
        routes,
        output_dir,
        group_size=args.coarse_group_size,
        bucket_size=args.coarse_bucket_size,
        top_profiles=args.top_profiles,
    )
    generated.extend(files)
    files, candidate = plot_simulated_tradeoff(simulation, output_dir)
    generated.extend(files)
    generated.extend(plot_lane_occupancy(simulation, output_dir))
    write_manifest(output_dir, records, simulation, generated, fragmentation, heatmap_metadata, candidate)

    print(f"Wrote {len(generated)} figure files to {output_dir}")
    print(f"Candidate policy for real replay: {candidate} ({SIM_NOTE} basis)")


if __name__ == "__main__":
    main()
