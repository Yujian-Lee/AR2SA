import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def margin(d_a, d_s):
    return np.abs(d_a - d_s) / (d_a + d_s + 1e-6)


def load_records(raw_dir):
    raw_dir = Path(raw_dir)
    records = {}
    for file in sorted(raw_dir.glob("*.npz")):
        records[file.name] = np.load(file)
    if not records:
        raise FileNotFoundError(f"No .npz files found in {raw_dir}")
    return records


def per_file_stats(records, metric):
    if metric == "euclidean":
        margin_key_a, margin_key_s, grad_key = "euclidean_to_a", "euclidean_to_s", "euclidean_grad"
    elif metric == "hyperbolic":
        margin_key_a, margin_key_s, grad_key = "hyperbolic_to_a", "hyperbolic_to_s", "hyperbolic_grad"
    else:
        raise ValueError(f"Unknown metric: {metric}")

    stats = {}
    for name, rec in records.items():
        m = margin(rec[margin_key_a], rec[margin_key_s]).ravel()
        g = rec[grad_key].ravel()
        stats[name] = {
            "margin_values": m,
            "gradient_values": g,
            "margin_mean": float(m.mean()),
            "gradient_mean": float(g.mean()),
        }
    return stats


def concat(stats, key, names):
    return np.concatenate([stats[name][key] for name in names])


def write_summary(out_dir, common, e_stats, h_stats):
    e_margin = concat(e_stats, "margin_values", common)
    h_margin = concat(h_stats, "margin_values", common)
    e_grad = concat(e_stats, "gradient_values", common)
    h_grad = concat(h_stats, "gradient_values", common)
    e_margin_file = np.array([e_stats[name]["margin_mean"] for name in common])
    h_margin_file = np.array([h_stats[name]["margin_mean"] for name in common])
    e_grad_file = np.array([e_stats[name]["gradient_mean"] for name in common])
    h_grad_file = np.array([h_stats[name]["gradient_mean"] for name in common])

    rows = [
        ("paired_files", len(common), "", "", "", ""),
        (
            "matched_margin_pixel_mean",
            float(e_margin.mean()),
            float(h_margin.mean()),
            float(h_margin.mean() - e_margin.mean()),
            float(h_margin.mean() / (e_margin.mean() + 1e-12)),
            "",
        ),
        (
            "matched_margin_file_mean",
            float(e_margin_file.mean()),
            float(h_margin_file.mean()),
            float((h_margin_file - e_margin_file).mean()),
            float(h_margin_file.mean() / (e_margin_file.mean() + 1e-12)),
            float((h_margin_file > e_margin_file).mean()),
        ),
        (
            "matched_margin_file_median",
            float(np.median(e_margin_file)),
            float(np.median(h_margin_file)),
            float(np.median(h_margin_file - e_margin_file)),
            "",
            "",
        ),
        (
            "matched_gradient_pixel_mean",
            float(e_grad.mean()),
            float(h_grad.mean()),
            float(h_grad.mean() - e_grad.mean()),
            float(h_grad.mean() / (e_grad.mean() + 1e-12)),
            "",
        ),
        (
            "matched_gradient_file_mean",
            float(e_grad_file.mean()),
            float(h_grad_file.mean()),
            float((h_grad_file - e_grad_file).mean()),
            float(h_grad_file.mean() / (e_grad_file.mean() + 1e-12)),
            float((h_grad_file > e_grad_file).mean()),
        ),
        (
            "matched_gradient_file_median",
            float(np.median(e_grad_file)),
            float(np.median(h_grad_file)),
            float(np.median(h_grad_file - e_grad_file)),
            "",
            "",
        ),
    ]

    with open(out_dir / "geometry_comparison_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "metric",
                "euclidean_ckpt",
                "hard_ckpt",
                "hard_minus_euclidean",
                "hard_over_euclidean",
                "hard_higher_file_ratio",
            ]
        )
        writer.writerows(rows)


def plot_hist(e_values, h_values, xlabel, title, out_file):
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    bins = np.linspace(0, max(np.percentile(e_values, 99), np.percentile(h_values, 99)), 36)
    ax.hist(e_values, bins=bins, density=True, histtype="step", linewidth=2.4, label="Euclidean pull-push")
    ax.hist(h_values, bins=bins, density=True, histtype="step", linewidth=2.4, label="HARD")
    ax.axvline(e_values.mean(), linestyle="--", linewidth=1.8)
    ax.axvline(h_values.mean(), linestyle="--", linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_file, dpi=300)
    plt.close(fig)


def plot_violin(e_values, h_values, ylabel, title, out_file):
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    parts = ax.violinplot([e_values, h_values], showmeans=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_alpha(0.5)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Euclidean\npull-push", "HARD"])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out_file, dpi=300)
    plt.close(fig)


def plot_tradeoff(common, e_stats, h_stats, out_file):
    e_margin = np.array([e_stats[name]["margin_mean"] for name in common])
    h_margin = np.array([h_stats[name]["margin_mean"] for name in common])
    e_grad = np.array([e_stats[name]["gradient_mean"] for name in common])
    h_grad = np.array([h_stats[name]["gradient_mean"] for name in common])

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.scatter(e_margin, e_grad, s=28, alpha=0.55, label="Euclidean pull-push")
    ax.scatter(h_margin, h_grad, s=28, alpha=0.55, label="HARD")
    ax.scatter([e_margin.mean()], [e_grad.mean()], marker="*", s=180, edgecolor="black")
    ax.scatter([h_margin.mean()], [h_grad.mean()], marker="*", s=180, edgecolor="black")
    ax.set_xlabel("Per-sample normalized distance margin")
    ax.set_ylabel("Per-sample gradient norm")
    ax.set_title("Ambiguous-feature Margin vs. Optimization Signal")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_file, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--euclidean_raw", required=True)
    parser.add_argument("--hard_raw", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    e_records = load_records(args.euclidean_raw)
    h_records = load_records(args.hard_raw)
    common = sorted(set(e_records).intersection(h_records))
    if not common:
        raise ValueError("The two raw directories do not share any .npz filenames.")

    e_stats = per_file_stats({name: e_records[name] for name in common}, "euclidean")
    h_stats = per_file_stats({name: h_records[name] for name in common}, "hyperbolic")

    write_summary(out_dir, common, e_stats, h_stats)
    plot_hist(
        concat(e_stats, "margin_values", common),
        concat(h_stats, "margin_values", common),
        "Normalized distance margin",
        "Ambiguous-feature Distance Margin",
        out_dir / "matched_margin_hist.png",
    )
    plot_violin(
        concat(e_stats, "gradient_values", common),
        concat(h_stats, "gradient_values", common),
        "Gradient norm on ambiguous features",
        "Optimization Signal on Ambiguous Features",
        out_dir / "matched_gradient_violin.png",
    )
    plot_tradeoff(common, e_stats, h_stats, out_dir / "margin_gradient_tradeoff.png")


if __name__ == "__main__":
    main()
