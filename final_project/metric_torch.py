"""PyTorch metrics/losses mirroring project TensorFlow metric semantics."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import hausdorff_distance as skimage_hausdorff


def dice_coef(y_true: torch.Tensor, y_pred: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Binary Dice coefficient with thresholded predictions."""
    y_true = y_true.float()
    y_pred = (y_pred > 0.5).float()
    intersection = torch.sum(y_true * y_pred)
    denom = torch.sum(y_true) + torch.sum(y_pred)
    return (2.0 * intersection + smooth) / (denom + smooth)


def soft_dice_loss(y_true: torch.Tensor, y_pred: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Differentiable Dice loss using raw probabilities."""
    y_true = y_true.float()
    y_pred = y_pred.float()

    if y_true.ndim < 3:
        intersection = torch.sum(y_true * y_pred)
        denom = torch.sum(y_true) + torch.sum(y_pred)
        return 1.0 - ((2.0 * intersection + smooth) / (denom + smooth))

    reduce_dims = tuple(range(2, y_true.ndim))
    intersection = torch.sum(y_true * y_pred, dim=reduce_dims)
    denom = torch.sum(y_true, dim=reduce_dims) + torch.sum(y_pred, dim=reduce_dims)
    return 1.0 - torch.mean((2.0 * intersection + smooth) / (denom + smooth))


def bce_dice_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Binary cross entropy plus soft Dice loss."""
    bce = F.binary_cross_entropy(y_pred.float(), y_true.float())
    return bce + soft_dice_loss(y_true, y_pred)


def hausdorff_distance(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Mean Hausdorff distance across a batch of binary 3D masks."""
    y_true_np = (y_true.detach().cpu().numpy() > 0.5)
    y_pred_np = (y_pred.detach().cpu().numpy() > 0.5)

    if y_true_np.ndim == 4:
        y_true_np = y_true_np[:, None, ...]
    if y_pred_np.ndim == 4:
        y_pred_np = y_pred_np[:, None, ...]

    scores = []
    for yt, yp in zip(y_true_np, y_pred_np):
        yt = yt.squeeze()
        yp = yp.squeeze()
        if not yt.any() and not yp.any():
            scores.append(0.0)
        elif not yt.any() or not yp.any():
            scores.append(float(max(yt.shape)))
        else:
            scores.append(float(skimage_hausdorff(yt, yp)))
    return float(np.mean(scores))
