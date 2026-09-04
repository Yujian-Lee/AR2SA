import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ROWS = [
    {
        "method": "Euclidean",
        "label": "Euclidean\npull-push",
        "miou": "71.48",
        "fscore": "81.82",
        "mean": "76.65",
        "note": "train from scratch; replace with your exact best run if needed",
    },
    {
        "method": "Hybrid",
        "label": "Hybrid\nEuc.+HARD",
        "miou": "72.80",
        "fscore": "82.30",
        "mean": "77.55",
        "note": "HARD + Euclidean; replace with exact best run if needed",
    },
    {
        "method": "HARD",
        "label": "HARD\nhyperbolic",
        "miou": "74.11",
        "fscore": "82.40",
        "mean": "78.25",
        "note": "recover run best mIoU; replace with official reported value if needed",
    },
]

COLORS = {
    "Euclidean": "#4C78A8",
    "Hybrid": "#F58518",
    "HARD": "#E45756",
}


def _to_float(value, default=np.nan):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def write_default_metrics(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["method", "label", "miou", "fscore", "mean", "gradient", "margin", "note"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in DEFAULT_ROWS:
            out = {key: row.get(key, "") for key in fields}
            writer.writerow(out)


def read_metrics(path):
    if not path.exists():
        write_default_metrics(path)
        print(f"[init] wrote editable metrics csv: {path}")
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row.get("method", "").strip()
            if not method:
                continue
            rows.append(
                {
                    "method": method,
                    "label": row.get("label", method).replace("\\n", "\n"),
                    "miou": _to_float(row.get("miou")),
                    "fscore": _to_float(row.get("fscore")),
                    "mean": _to_float(row.get("mean")),
                    "gradient": _to_float(row.get("gradient")),
                    "margin": _to_float(row.get("margin")),
                    "note": row.get("note", ""),
                }
            )
    return rows


def read_compare_summary(path):
    if not path.exists():
        return {}
    out = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric = row.get("metric", "")
            out[metric] = row
    return out


def merge_geometry(rows, summary):
    if not summary:
        return rows
    aliases = {
        "Euclidean": "euclidean_ckpt",
        "HARD": "hard_ckpt",
    }
    grad = summary.get("matched_gradient_file_mean", {})
    margin = summary.get("matched_margin_file_mean", {})
    for row in rows:
        key = aliases.get(row["method"])
        if key:
            if np.isnan(row["gradient"]):
                row["gradient"] = _to_float(grad.get(key))
            if np.isnan(row["margin"]):
                row["margin"] = _to_float(margin.get(key))
    return rows


def _finish(fig, path, dpi):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[saved] {path}")


def plot_accuracy(rows, out_dir, dpi):
    labels = [r["label"] for r in rows]
    miou = np.array([r["miou"] for r in rows])
    fscore = np.array([r["fscore"] for r in rows])
    x = np.arange(len(rows))
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    b1 = ax.bar(x - width / 2, miou, width, label="mIoU", color="#4C78A8")
    b2 = ax.bar(x + width / 2, fscore, width, label="F-score", color="#E45756")
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x, labels)
    ax.set_ylim(max(0, np.nanmin(miou) - 4), min(100, np.nanmax(fscore) + 3))
    ax.set_title("Accuracy comparison of ambiguity resolution strategies")
    ax.legend(frameon=False, ncols=2)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            if np.isfinite(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.12, f"{h:.2f}", ha="center", va="bottom", fontsize=8)
    _finish(fig, out_dir / "strategy_accuracy_bars.png", dpi)


def plot_miou_gain(rows, out_dir, dpi):
    base = rows[0]["miou"]
    labels = [r["label"] for r in rows]
    gains = np.array([r["miou"] - base for r in rows])
    colors = [COLORS.get(r["method"], "#888888") for r in rows]

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    bars = ax.bar(np.arange(len(rows)), gains, color=colors, alpha=0.9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("mIoU gain over Euclidean (%)")
    ax.set_xticks(np.arange(len(rows)), labels)
    ax.set_title("Relative gain from Euclidean to hybrid and HARD")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, gain in zip(bars, gains):
        ax.text(bar.get_x() + bar.get_width() / 2, gain + 0.06, f"{gain:+.2f}", ha="center", va="bottom", fontsize=9)
    top = max(0.5, np.nanmax(gains) + 0.6)
    ax.set_ylim(min(-0.2, np.nanmin(gains) - 0.2), top)
    _finish(fig, out_dir / "strategy_miou_gain.png", dpi)


def plot_geometry(rows, out_dir, dpi):
    geo_rows = [r for r in rows if np.isfinite(r["gradient"]) or np.isfinite(r["margin"])]
    if len(geo_rows) < 2:
        print("[skip] geometry plot: fewer than two methods with gradient/margin values")
        return
    labels = [r["label"] for r in geo_rows]
    grad = np.array([r["gradient"] for r in geo_rows])
    margin = np.array([r["margin"] for r in geo_rows])
    x = np.arange(len(geo_rows))

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8))
    for ax, values, title, ylabel in [
        (axes[0], grad, "Matched gradient response", "Gradient magnitude"),
        (axes[1], margin, "Matched distance margin", "Normalized margin"),
    ]:
        bars = ax.bar(x, values, color=[COLORS.get(r["method"], "#888888") for r in geo_rows], alpha=0.9)
        ax.set_xticks(x, labels)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.text(bar.get_x() + bar.get_width() / 2, value + max(values[np.isfinite(values)]) * 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    _finish(fig, out_dir / "strategy_geometry_response.png", dpi)


def plot_tradeoff(rows, out_dir, dpi):
    pts = [r for r in rows if np.isfinite(r["gradient"]) and np.isfinite(r["miou"])]
    if len(pts) < 2:
        print("[skip] tradeoff plot: fewer than two methods with both gradient and mIoU")
        return
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for r in pts:
        ax.scatter(r["gradient"], r["miou"], s=95, color=COLORS.get(r["method"], "#888888"), edgecolor="black", linewidth=0.5)
        ax.annotate(r["method"], (r["gradient"], r["miou"]), xytext=(7, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Matched gradient response")
    ax.set_ylabel("mIoU (%)")
    ax.set_title("Accuracy aligns with stronger ambiguity-resolution response")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _finish(fig, out_dir / "strategy_gradient_miou_tradeoff.png", dpi)


def main():
    parser = argparse.ArgumentParser(description="Plot Euclidean vs Hybrid vs HARD strategy comparisons.")
    parser.add_argument("--compare_dir", default="VIS_PTH_COMPARE_20260729/COMPARE", help="Directory containing geometry_comparison_summary.csv")
    parser.add_argument("--metrics_csv", default="", help="CSV with columns: method,label,miou,fscore,mean,gradient,margin,note")
    parser.add_argument("--out_dir", default="", help="Output directory; defaults to compare_dir/strategy_plots")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    compare_dir = Path(args.compare_dir)
    out_dir = Path(args.out_dir) if args.out_dir else compare_dir / "strategy_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = Path(args.metrics_csv) if args.metrics_csv else out_dir / "strategy_metrics.csv"

    rows = read_metrics(metrics_csv)
    summary = read_compare_summary(compare_dir / "geometry_comparison_summary.csv")
    rows = merge_geometry(rows, summary)

    plot_accuracy(rows, out_dir, args.dpi)
    plot_miou_gain(rows, out_dir, args.dpi)
    plot_geometry(rows, out_dir, args.dpi)
    plot_tradeoff(rows, out_dir, args.dpi)

    print("\nEdit this file if you need exact values before rerunning:")
    print(metrics_csv)


if __name__ == "__main__":
    main()