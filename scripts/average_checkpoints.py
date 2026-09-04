import argparse
import glob
import os
import re
from collections import OrderedDict

import torch


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def read_metric(path, sort_by):
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        if sort_by == "fscore":
            value = checkpoint.get("test_fscore")
            if isinstance(value, (int, float)):
                return float(value)
        if sort_by == "miou":
            value = checkpoint.get("test_miou") or checkpoint.get("best_miou") or checkpoint.get("miou")
            if isinstance(value, (int, float)):
                return float(value)
    name = os.path.basename(path)
    pattern = r"test([0-9.]+)\.pth" if sort_by in {"miou", "fscore"} else r"epoch_(\d+)"
    match = re.search(pattern, name)
    if match:
        return float(match.group(1))
    return float("-inf")


def collect_paths(args):
    paths = []
    if args.ckpt_dir:
        paths.extend(glob.glob(os.path.join(args.ckpt_dir, "epoch_*.pth")))
    paths.extend(args.ckpts)
    paths = [os.path.abspath(p) for p in paths]
    seen = set()
    unique = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    paths = [p for p in unique if os.path.exists(p)]
    if args.min_miou is not None:
        kept = []
        for path in paths:
            checkpoint = torch.load(path, map_location="cpu")
            miou = None
            if isinstance(checkpoint, dict):
                value = checkpoint.get("test_miou") or checkpoint.get("best_miou") or checkpoint.get("miou")
                if isinstance(value, (int, float)):
                    miou = float(value)
            if miou is None:
                match = re.search(r"test([0-9.]+)\.pth", os.path.basename(path))
                if match:
                    miou = float(match.group(1))
            if miou is not None and miou >= args.min_miou:
                kept.append(path)
        paths = kept
    if args.sort_by != "none":
        paths = sorted(paths, key=lambda p: read_metric(p, args.sort_by), reverse=True)
    if args.top_k > 0:
        paths = paths[: args.top_k]
    if len(paths) < 2:
        raise ValueError("Need at least two checkpoints to average.")
    return paths


def average_checkpoints(paths):
    avg_state = OrderedDict()
    ref_shapes = {}
    ref_keys = None
    n = len(paths)

    for idx, path in enumerate(paths):
        checkpoint = torch.load(path, map_location="cpu")
        state = extract_state_dict(checkpoint)
        if not isinstance(state, dict):
            raise TypeError(f"Checkpoint does not contain a state_dict: {path}")
        if ref_keys is None:
            ref_keys = list(state.keys())
            for key, value in state.items():
                if torch.is_tensor(value):
                    ref_shapes[key] = tuple(value.shape)
                    if torch.is_floating_point(value):
                        avg_state[key] = value.detach().cpu().float() / n
                    else:
                        avg_state[key] = value.detach().cpu().clone()
                else:
                    avg_state[key] = value
        else:
            keys = list(state.keys())
            if keys != ref_keys:
                missing = sorted(set(ref_keys) - set(keys))[:10]
                extra = sorted(set(keys) - set(ref_keys))[:10]
                raise ValueError(f"State dict keys differ in {path}; missing={missing}, extra={extra}")
            for key, value in state.items():
                if not torch.is_tensor(value):
                    continue
                if tuple(value.shape) != ref_shapes[key]:
                    raise ValueError(f"Shape mismatch for {key} in {path}: {tuple(value.shape)} != {ref_shapes[key]}")
                if torch.is_floating_point(value):
                    avg_state[key] += value.detach().cpu().float() / n
        print(f"[{idx + 1}/{n}] loaded {path}")
    return avg_state


def main():
    parser = argparse.ArgumentParser(description="Average AR2SA checkpoints into one state_dict.")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Directory containing epoch_*.pth files.")
    parser.add_argument("--ckpts", nargs="*", default=[], help="Explicit checkpoint paths.")
    parser.add_argument("--out", type=str, required=True, help="Output .pth path.")
    parser.add_argument("--sort_by", choices=["fscore", "miou", "none"], default="fscore")
    parser.add_argument("--top_k", type=int, default=0, help="Keep top-k checkpoints after sorting; 0 means use all.")
    parser.add_argument("--min_miou", type=float, default=None, help="Optional minimum test mIoU filter.")
    args = parser.parse_args()

    paths = collect_paths(args)
    print("Averaging checkpoints:")
    for path in paths:
        print("  ", path)
    avg_state = average_checkpoints(paths)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(avg_state, args.out)
    print(f"Saved averaged checkpoint to {args.out}")


if __name__ == "__main__":
    main()