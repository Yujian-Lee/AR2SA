import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def _sample_rows(array, max_rows):
    if array.shape[0] <= max_rows:
        return array
    idx = np.linspace(0, array.shape[0] - 1, max_rows).astype(np.int64)
    return array[idx]


def _margin(d_a, d_s):
    return np.abs(d_a - d_s) / (d_a + d_s + 1e-6)


def _poincare_distance_np(x, y, curvature=1.0, eps=1e-6):
    c = float(curvature)
    x_norm = np.clip(c * np.sum(x * x, axis=-1), None, 1 - eps)
    y_norm = np.clip(c * np.sum(y * y, axis=-1), None, 1 - eps)
    diff = np.sum((x - y) ** 2, axis=-1)
    z = 1 + 2 * c * diff / ((1 - x_norm) * (1 - y_norm) + eps)
    return np.arccosh(np.maximum(z, 1 + eps)) / np.sqrt(c)


def _project_to_ball_np(x, curvature=1.0, radius_ratio=0.8, eps=1e-9):
    max_radius = radius_ratio / np.sqrt(float(curvature))
    norm = np.linalg.norm(x, axis=-1, keepdims=True) + eps
    scale = np.tanh(norm / np.sqrt(x.shape[-1])) * max_radius
    return x / norm * scale


def _rec_array(rec, preferred, fallback):
    return rec[preferred] if preferred in rec.files else rec[fallback]


def load_npz(raw_dir):
    files = sorted(Path(raw_dir).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {raw_dir}")
    records = []
    for file in files:
        records.append(np.load(file))
    return records


def plot_feature_pca(records, out_dir, max_per_file):
    feats, labels = [], []
    centers = []
    for rec in records:
        for key, fallback, label in [("silence_ball", "silence", 0), ("audio_ball", "audio", 1), ("ambiguous_ball", "ambiguous", 2)]:
            x = _sample_rows(_rec_array(rec, key, fallback), max_per_file)
            feats.append(x)
            labels.append(np.full(x.shape[0], label))
        centers.append(_rec_array(rec, "silence_center_ball", "silence_center"))
        centers.append(_rec_array(rec, "audio_center_ball", "audio_center"))

    feat = np.concatenate(feats, axis=0)
    label = np.concatenate(labels, axis=0)
    center = np.concatenate(centers, axis=0)
    pca = PCA(n_components=2, random_state=0)
    xy = pca.fit_transform(feat)
    center_xy = pca.transform(center)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    colors = {0: "#4C78A8", 1: "#F58518", 2: "#E45756"}
    names = {0: "Silent", 1: "Audio-active", 2: "Ambiguous"}
    for label_id in [0, 1, 2]:
        mask = label == label_id
        ax.scatter(xy[mask, 0], xy[mask, 1], s=7, alpha=0.42, c=colors[label_id], label=names[label_id], linewidths=0)
    ax.scatter(center_xy[0::2, 0], center_xy[0::2, 1], marker="*", s=90, c=colors[0], edgecolors="black", linewidths=0.4)
    ax.scatter(center_xy[1::2, 0], center_xy[1::2, 1], marker="*", s=90, c=colors[1], edgecolors="black", linewidths=0.4)
    ax.set_title("Feature States in the ODS-HARD Space")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_pca_states.png", dpi=300)
    plt.close(fig)


def plot_margin(records, out_dir):
    e_margins, h_margins = [], []
    for rec in records:
        e_margins.append(_margin(rec["euclidean_to_a"], rec["euclidean_to_s"]))
        h_margins.append(_margin(rec["hyperbolic_to_a"], rec["hyperbolic_to_s"]))
    e = np.concatenate(e_margins)
    h = np.concatenate(h_margins)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    bins = np.linspace(0, 1, 41)
    ax.hist(e, bins=bins, density=True, histtype="step", linewidth=2.0, color="#4C78A8", label="Euclidean")
    ax.hist(h, bins=bins, density=True, histtype="step", linewidth=2.0, color="#E45756", label="Hyperbolic")
    ax.axvline(e.mean(), color="#4C78A8", linestyle="--", linewidth=1.2)
    ax.axvline(h.mean(), color="#E45756", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Normalized distance margin")
    ax.set_ylabel("Density")
    ax.set_title("Ambiguous Pixels Become More Decisive Under HARD")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "distance_margin_hist.png", dpi=300)
    plt.close(fig)
    return e, h


def plot_grad(records, out_dir):
    e_grad = np.concatenate([rec["euclidean_grad"] for rec in records])
    h_grad = np.concatenate([rec["hyperbolic_grad"] for rec in records])

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    parts = ax.violinplot([e_grad, h_grad], showmeans=True, showextrema=False)
    for body, color in zip(parts["bodies"], ["#4C78A8", "#E45756"]):
        body.set_facecolor(color)
        body.set_alpha(0.45)
    parts["cmeans"].set_color("black")
    ax.set_xticks([1, 2], ["Euclidean", "Hyperbolic"])
    ax.set_ylabel("Gradient norm on ambiguous features")
    ax.set_title("HARD Provides Stronger Optimization Signals")
    fig.tight_layout()
    fig.savefig(out_dir / "gradient_norm_violin.png", dpi=300)
    plt.close(fig)
    return e_grad, h_grad


def plot_curvature(records, out_dir, curvatures):
    means = []
    for c in curvatures:
        margins = []
        for rec in records:
            u = _project_to_ball_np(rec["ambiguous"], curvature=c)
            a = _project_to_ball_np(rec["audio_center"], curvature=c)
            s = _project_to_ball_np(rec["silence_center"], curvature=c)
            d_a = _poincare_distance_np(u, a, curvature=c)
            d_s = _poincare_distance_np(u, s, curvature=c)
            margins.append(_margin(d_a, d_s))
        means.append(np.concatenate(margins).mean())

    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    ax.plot(curvatures, means, marker="o", color="#E45756", linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("Curvature c")
    ax.set_ylabel("Mean normalized margin")
    ax.set_title("Curvature Sensitivity of Ambiguity Separation")
    fig.tight_layout()
    fig.savefig(out_dir / "curvature_margin_proxy.png", dpi=300)
    plt.close(fig)
    return means


def write_summary(out_dir, e_margin, h_margin, e_grad, h_grad, curvatures, curvature_means):
    with open(out_dir / "hard_geometry_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["euclidean_margin_mean", float(e_margin.mean())])
        writer.writerow(["hyperbolic_margin_mean", float(h_margin.mean())])
        writer.writerow(["euclidean_gradient_mean", float(e_grad.mean())])
        writer.writerow(["hyperbolic_gradient_mean", float(h_grad.mean())])
        for c, value in zip(curvatures, curvature_means):
            writer.writerow([f"curvature_{c}_margin_mean", float(value)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="./VIS_HARD_GEOM/raw")
    parser.add_argument("--out_dir", default="./VIS_HARD_GEOM/figs")
    parser.add_argument("--max_per_file", type=int, default=128)
    parser.add_argument("--curvatures", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_npz(args.raw_dir)
    plot_feature_pca(records, out_dir, args.max_per_file)
    e_margin, h_margin = plot_margin(records, out_dir)
    e_grad, h_grad = plot_grad(records, out_dir)
    curvature_means = plot_curvature(records, out_dir, args.curvatures)
    write_summary(out_dir, e_margin, h_margin, e_grad, h_grad, args.curvatures, curvature_means)
    print(f"Saved HARD geometry figures to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
