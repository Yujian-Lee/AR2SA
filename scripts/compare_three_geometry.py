import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Global font
# ============================================================
plt.rcParams.update(
    {
        "font.family": "Comic Sans MS",
        "font.sans-serif": ["Comic Sans MS", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Comic Sans MS",
        "mathtext.it": "Comic Sans MS:italic",
        "mathtext.bf": "Comic Sans MS:bold",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# ============================================================
# High-contrast colors for three methods
# ============================================================
COLORS = [
    "#0072B2",  # Euclidean
    "#E69F00",  # Hybrid
    "#009E73",  # HARD / Hyperbolic
]


def _margin(d_a, d_s):
    return np.abs(d_a - d_s) / (d_a + d_s + 1e-6)


def _clean(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    return x[np.isfinite(x)]


def _label(name):
    return name.replace("\\n", "\n")


def _metric_values(rec, metric, hybrid_eu_weight, hybrid_hyp_weight, warnings):
    metric = metric.lower()
    if metric in {"euclidean", "eucli", "l2"}:
        m = _margin(rec["euclidean_to_a"], rec["euclidean_to_s"])
        g = rec["euclidean_grad"]
        return _clean(m), _clean(g)

    if metric in {"hard", "hyperbolic", "hyp"}:
        m = _margin(rec["hyperbolic_to_a"], rec["hyperbolic_to_s"])
        g = rec["hyperbolic_grad"]
        return _clean(m), _clean(g)

    if metric in {"hybrid", "mix", "hard_euclidean", "euclidean_hard"}:
        e_w = hybrid_eu_weight
        h_w = hybrid_hyp_weight
        if "hybrid_eu_weight" in rec.files:
            e_w = float(np.asarray(rec["hybrid_eu_weight"]).ravel()[0])
        if "hybrid_hyp_weight" in rec.files:
            h_w = float(np.asarray(rec["hybrid_hyp_weight"]).ravel()[0])
        w_sum = max(e_w + h_w, 1e-12)

        if "hybrid_margin" in rec.files:
            m = rec["hybrid_margin"]
        else:
            e_m = _margin(rec["euclidean_to_a"], rec["euclidean_to_s"])
            h_m = _margin(rec["hyperbolic_to_a"], rec["hyperbolic_to_s"])
            m = (e_w * e_m + h_w * h_m) / w_sum
            warnings.add("hybrid_margin_missing")

        if "hybrid_grad" in rec.files:
            g = rec["hybrid_grad"]
        else:
            g = (e_w * rec["euclidean_grad"] + h_w * rec["hyperbolic_grad"]) / w_sum
            warnings.add("hybrid_grad_missing")
        return _clean(m), _clean(g)

    raise ValueError(f"Unknown metric: {metric}")


def _raw_files(raw_dir):
    raw_dir = Path(raw_dir)
    files = {p.name: p for p in sorted(raw_dir.glob("*.npz"))}
    if not files:
        raise FileNotFoundError(f"No .npz files found in {raw_dir}")
    return files


def load_method_stats(methods, hybrid_eu_weight, hybrid_hyp_weight):
    method_files = []
    for method in methods:
        method_files.append(_raw_files(method["raw_dir"]))

    common = sorted(set.intersection(*(set(files.keys()) for files in method_files)))
    if not common:
        raise ValueError("Raw directories do not share any matched .npz filenames.")

    warnings = set()
    loaded = []
    for method, files in zip(methods, method_files):
        margin_values = []
        grad_values = []
        margin_file_mean = []
        grad_file_mean = []
        per_file = []
        for name in common:
            with np.load(files[name]) as rec:
                m, g = _metric_values(rec, method["metric"], hybrid_eu_weight, hybrid_hyp_weight, warnings)
            if m.size == 0 or g.size == 0:
                continue
            margin_values.append(m)
            grad_values.append(g)
            margin_file_mean.append(float(m.mean()))
            grad_file_mean.append(float(g.mean()))
            per_file.append((name, float(m.mean()), float(g.mean()), int(m.size), int(g.size)))

        if not margin_values or not grad_values:
            raise ValueError(f"No valid geometry values for method {method['name']}")

        loaded.append(
            {
                "name": method["name"],
                "metric": method["metric"],
                "raw_dir": str(method["raw_dir"]),
                "margin_values": np.concatenate(margin_values),
                "gradient_values": np.concatenate(grad_values),
                "margin_file_mean": np.asarray(margin_file_mean),
                "gradient_file_mean": np.asarray(grad_file_mean),
                "per_file": per_file,
            }
        )
    return common, loaded, warnings


def savefig(fig, out_dir, stem):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=800)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)
    print(f"[saved] {out_dir / (stem + '.png')}")


def write_summary(out_dir, common, stats):
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "geometry_comparison_summary.csv"
    rows = [
        ("paired_files", [len(common) for _ in stats]),
        ("matched_margin_pixel_mean", [float(s["margin_values"].mean()) for s in stats]),
        ("matched_margin_file_mean", [float(s["margin_file_mean"].mean()) for s in stats]),
        ("matched_margin_file_median", [float(np.median(s["margin_file_mean"])) for s in stats]),
        ("matched_gradient_pixel_mean", [float(s["gradient_values"].mean()) for s in stats]),
        ("matched_gradient_file_mean", [float(s["gradient_file_mean"].mean()) for s in stats]),
        ("matched_gradient_file_median", [float(np.median(s["gradient_file_mean"])) for s in stats]),
    ]
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric"] + [s["name"] for s in stats])
        for metric, values in rows:
            writer.writerow([metric] + values)

    per_file_path = out_dir / "per_file_geometry.csv"
    with open(per_file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "method", "metric", "margin_mean", "gradient_mean", "margin_points", "gradient_points"])
        for s in stats:
            for name, margin_mean, grad_mean, margin_n, grad_n in s["per_file"]:
                writer.writerow([name, s["name"], s["metric"], margin_mean, grad_mean, margin_n, grad_n])
    print(f"[saved] {summary_path}")
    print(f"[saved] {per_file_path}")


def plot_margin_hist(stats, out_dir):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    all_values = np.concatenate([s["margin_values"] for s in stats])
    upper = max(0.05, float(np.percentile(all_values, 99.5)))
    bins = np.linspace(0, upper, 42)
    for idx, s in enumerate(stats):
        color = COLORS[idx % len(COLORS)]
        values = s["margin_values"]
        ax.hist(values, bins=bins, density=True, histtype="step", linewidth=2.4, color=color, label=_label(s["name"]))
        ax.axvline(values.mean(), linestyle="--", linewidth=1.7, color=color, alpha=0.9)
    ax.set_title("Ambiguous-feature Distance Margin")
    ax.set_xlabel("Normalized distance margin")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    savefig(fig, out_dir, "matched_margin_hist")


def plot_gradient_violin(stats, out_dir):
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    values = [s["gradient_values"] for s in stats]
    parts = ax.violinplot(values, showmeans=True, showextrema=False)
    for idx, body in enumerate(parts["bodies"]):
        body.set_facecolor(COLORS[idx % len(COLORS)])
        body.set_alpha(0.48)
    parts["cmeans"].set_color("black")
    parts["cmeans"].set_linewidth(1.6)
    ax.set_xticks(np.arange(1, len(stats) + 1))
    ax.set_xticklabels([_label(s["name"]) for s in stats])
    ax.set_title("Optimization Signal on Ambiguous Features")
    ax.set_ylabel("Gradient norm on ambiguous features")
    savefig(fig, out_dir, "matched_gradient_violin")


def plot_tradeoff(stats, out_dir):
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    for idx, s in enumerate(stats):
        color = COLORS[idx % len(COLORS)]
        x = s["margin_file_mean"]
        y = s["gradient_file_mean"]
        ax.scatter(x, y, s=30, alpha=0.55, color=color, label=_label(s["name"]))
        ax.scatter([x.mean()], [y.mean()], marker="*", s=210, color=color, edgecolor="black", linewidth=1.2)
    ax.set_xlabel("Per-sample normalized distance margin")
    ax.set_ylabel("Per-sample gradient norm")
    ax.set_title("Ambiguous-feature Margin vs. Optimization Signal")
    ax.legend(frameon=False)
    savefig(fig, out_dir, "margin_gradient_tradeoff")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot matched geometry comparison for Euclidean, Hybrid, and HARD checkpoints.")
    parser.add_argument(
        "--raw",
        nargs=3,
        action="append",
        metavar=("NAME", "METRIC", "RAW_DIR"),
        required=True,
        help="One method spec. METRIC is euclidean, hybrid, or hyperbolic/hard.",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--hybrid_eu_weight", type=float, default=0.4)
    parser.add_argument("--hybrid_hyp_weight", type=float, default=0.6)
    return parser.parse_args()


def main():
    args = parse_args()
    methods = [
        {"name": name, "metric": metric, "raw_dir": Path(raw_dir)}
        for name, metric, raw_dir in args.raw
    ]
    out_dir = Path(args.out_dir)
    common, stats, warnings = load_method_stats(methods, args.hybrid_eu_weight, args.hybrid_hyp_weight)
    write_summary(out_dir, common, stats)
    plot_margin_hist(stats, out_dir)
    plot_gradient_violin(stats, out_dir)
    plot_tradeoff(stats, out_dir)
    print(f"[info] matched files: {len(common)}")
    if "hybrid_margin_missing" in warnings:
        print("[warn] hybrid_margin missing in some raw files; used weighted margin proxy.")
    if "hybrid_grad_missing" in warnings:
        print("[warn] hybrid_grad missing in some raw files; used weighted gradient-norm proxy.")


if __name__ == "__main__":
    main()