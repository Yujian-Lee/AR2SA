import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from scripts.save_mask import save_batch_mask


def _env_float(name, default):
    return float(os.environ.get(name, str(default)))


def _mask_tensor(mask_like):
    if isinstance(mask_like, (list, tuple)):
        mask = torch.stack([m.detach().cpu() if torch.is_tensor(m) else torch.as_tensor(m) for m in mask_like], dim=0)
    elif torch.is_tensor(mask_like):
        mask = mask_like.detach().cpu()
    else:
        mask = torch.as_tensor(mask_like)

    while mask.ndim > 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    if mask.ndim == 4:
        mask = mask.argmax(1)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError(f"Unsupported mask shape for SFD memory: {tuple(mask.shape)}")

    if mask.dtype.is_floating_point:
        if float(mask.max()) <= 1.0:
            mask = mask > 0.5
        else:
            mask = mask > 0
    else:
        mask = mask != 0
    return mask.bool()


def _resize_mask(mask, target_hw):
    if tuple(mask.shape[-2:]) == tuple(target_hw):
        return mask
    mask_f = mask.float().unsqueeze(1)
    mask_f = F.interpolate(mask_f, size=target_hw, mode="nearest")
    return mask_f.squeeze(1).bool()


def _load_previous_mask(mask_dir, vid, shape):
    if not mask_dir:
        return None
    mask_dir = Path(mask_dir)
    frames = []
    for frame_idx in range(shape[0]):
        path = mask_dir / str(vid) / f"{frame_idx}.png"
        if not path.exists():
            return None
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = cv2.resize(img, (shape[2], shape[1]), interpolation=cv2.INTER_NEAREST)
        frames.append(torch.as_tensor(img > 0))
    return torch.stack(frames, dim=0).bool()


def _mean_pairwise_iou(mask):
    if mask.shape[0] < 2:
        return 1.0
    vals = []
    for idx in range(mask.shape[0] - 1):
        a = mask[idx]
        b = mask[idx + 1]
        union = torch.logical_or(a, b).sum().item()
        if union == 0:
            vals.append(1.0)
        else:
            inter = torch.logical_and(a, b).sum().item()
            vals.append(inter / union)
    return float(np.mean(vals))


def _iou(a, b):
    union = torch.logical_or(a, b).sum().item()
    if union == 0:
        return 1.0
    inter = torch.logical_and(a, b).sum().item()
    return inter / union


def mask_quality_pass(candidate, optical, previous=None):
    candidate = candidate.bool()
    optical = _resize_mask(optical.bool(), candidate.shape[-2:])
    previous = _resize_mask(previous.bool(), candidate.shape[-2:]) if previous is not None else None

    area_per_frame = candidate.float().mean(dim=(-1, -2))
    mean_area = float(area_per_frame.mean().item())
    nonempty_ratio = float((area_per_frame >= _env_float("AR2SA_SFD_MIN_FRAME_AREA", 0.0003)).float().mean().item())

    candidate_pixels = candidate.sum().item()
    optical_intersection = torch.logical_and(candidate, optical).sum().item()
    optical_precision = 0.0 if candidate_pixels == 0 else optical_intersection / candidate_pixels
    temporal_iou = _mean_pairwise_iou(candidate)

    min_area = _env_float("AR2SA_SFD_MIN_AREA", 0.0005)
    max_area = _env_float("AR2SA_SFD_MAX_AREA", 0.85)
    min_nonempty = _env_float("AR2SA_SFD_MIN_NONEMPTY_FRAMES", 0.2)
    min_optical_precision = _env_float("AR2SA_SFD_MIN_OPTICAL_PRECISION", 0.10)
    min_temporal_iou = _env_float("AR2SA_SFD_MIN_TEMPORAL_IOU", 0.0)
    max_area_delta = _env_float("AR2SA_SFD_MAX_AREA_DELTA", 0.70)

    passed = (
        min_area <= mean_area <= max_area
        and nonempty_ratio >= min_nonempty
        and optical_precision >= min_optical_precision
        and temporal_iou >= min_temporal_iou
    )

    previous_iou = None
    area_delta = None
    if previous is not None:
        previous_area = float(previous.float().mean().item())
        previous_iou = _iou(candidate, previous)
        area_delta = abs(mean_area - previous_area)
        passed = passed and area_delta <= max_area_delta

    metrics = {
        "mean_area": mean_area,
        "nonempty_ratio": nonempty_ratio,
        "optical_precision": optical_precision,
        "temporal_iou": temporal_iou,
        "previous_iou": previous_iou,
        "area_delta": area_delta,
    }
    return passed, metrics


def prepare_sfd_buffer(buffer_root, split):
    root = Path(buffer_root).resolve() / split
    current = (root / "current").resolve()
    if current.parent != root or current.name != "current":
        raise RuntimeError(f"Refusing to reset unexpected SFD buffer path: {current}")
    if current.exists():
        shutil.rmtree(current)
    current.mkdir(parents=True, exist_ok=True)
    return str(current)


def empty_sfd_stats():
    return {"accepted": 0, "rejected": 0}


def add_sfd_stats(total, update):
    total["accepted"] += update.get("accepted", 0)
    total["rejected"] += update.get("rejected", 0)
    return total


def format_sfd_stats(split, stats):
    return f"[SFD:{split}] accepted={stats.get('accepted', 0)} rejected={stats.get('rejected', 0)}"


def update_sfd_memory_batch(split, logits, vid, optical, active_dir, buffer_dir):
    candidate = _mask_tensor(logits)
    optical_mask = _mask_tensor(optical)
    previous = _load_previous_mask(active_dir, vid, candidate.shape)

    sample_buffer_dir = Path(buffer_dir) / str(vid)
    save_batch_mask(candidate, str(sample_buffer_dir))

    passed, metrics = mask_quality_pass(candidate, optical_mask, previous)
    stats = empty_sfd_stats()
    if passed:
        sample_active_dir = Path(active_dir) / str(vid)
        sample_active_dir.mkdir(parents=True, exist_ok=True)
        for file in sample_buffer_dir.glob("*.png"):
            shutil.copy2(file, sample_active_dir / file.name)
        stats["accepted"] = 1
    else:
        stats["rejected"] = 1

    stats.update({f"{split}_{k}": v for k, v in metrics.items() if v is not None})
    return stats
