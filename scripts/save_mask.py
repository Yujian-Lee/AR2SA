import os

import cv2
import numpy as np
import torch


def _prepare_mask_tensor(logits):
    if torch.is_tensor(logits):
        mask = logits.detach().cpu()
    else:
        mask = torch.as_tensor(logits)

    if mask.ndim == 4:
        # [T, C, H, W] class logits/probabilities.
        mask = mask.argmax(1)
    elif mask.ndim == 3:
        # [T, H, W] binary/class-index masks from post-processing.
        pass
    elif mask.ndim == 2:
        mask = mask.unsqueeze(0)
    else:
        raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")

    if mask.dtype.is_floating_point:
        if float(mask.max()) <= 1.0:
            mask = mask > 0.5
        else:
            mask = mask > 0
    else:
        mask = mask != 0
    return mask.to(torch.uint8).numpy() * 255


def save_batch_mask(logits, save_path):
    os.makedirs(save_path, exist_ok=True)
    mask = _prepare_mask_tensor(logits)
    for i in range(mask.shape[0]):
        cv2.imwrite(os.path.join(save_path, f"{i}.png"), np.ascontiguousarray(mask[i]))
