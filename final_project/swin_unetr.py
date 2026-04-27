"""Custom Swin-UNETR model with Monte Carlo dropout for the uncertainty estimation."""

from __future__ import annotations # for lazy evaluation of type hints

import inspect
from typing import Any, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.layers import DropPath
from monai.networks.nets import SwinUNETR as MonaiSwinUNETR


__all__ = ["SwinUNETRWithMCDropout", "build_swin_unetr_mc"]


def _drop_path(
    x: torch.Tensor,
    drop_prob: float = 0.0,
    scale_by_keep: bool = True, # TODO: what does this do? should I delete?
) -> torch.Tensor:
    """Drop a skip-connection path with a probablity of `drop_prob`"""
    if drop_prob <= 0.0:
        return x

    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class MCDropout(nn.Module):
    """Monte Carlo dropout for a single neuron"""

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: switch training=True to training=self.training if you want MC dropout only at train-time.
        return F.dropout(x, p=self.p, training=True)


class MCDropPath(nn.Module):
    """Skip-connection path with Monte Carlo dropout"""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)
        self.scale_by_keep = bool(scale_by_keep)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: call _drop_path with drop_prob=0.0 or guard with self.training to disable eval-time MC behavior.
        return _drop_path(x, drop_prob=self.drop_prob, scale_by_keep=self.scale_by_keep)


def _replace_dropout_layers(module: nn.Module) -> None:
    """Recursively replace dropout-like modules with MC variants."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Dropout):
            # TODO: swap MCDropout back to nn.Dropout if you want standard deterministic eval mode.
            setattr(module, name, MCDropout(p=child.p))
            continue

        if isinstance(child, DropPath):
            # TODO: swap MCDropPath back to DropPath to disable eval-time stochastic depth.
            drop_prob = float(getattr(child, "drop_prob", 0.0))
            scale_by_keep = bool(getattr(child, "scale_by_keep", True))
            setattr(module, name, MCDropPath(drop_prob=drop_prob, scale_by_keep=scale_by_keep))
            continue

        _replace_dropout_layers(child)


class SwinUNETRWithMCDropout(nn.Module):
    """MONAI SwinUNETR wrapper with optional MC dropout."""

    def __init__(
        self,
        img_size: Union[Sequence[int], int],
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: Union[Tuple, str] = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample: Union[str, nn.Module] = "merging",
        force_mc_dropout: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.force_mc_dropout = bool(force_mc_dropout)
        init_kwargs = dict(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            depths=depths,
            num_heads=num_heads,
            feature_size=feature_size,
            norm_name=norm_name,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            normalize=normalize,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=downsample,
        )

        allowed = set(inspect.signature(MonaiSwinUNETR.__init__).parameters)
        filtered_kwargs = {k: v for k, v in init_kwargs.items() if k in allowed}

        self.model = MonaiSwinUNETR(**filtered_kwargs)

        if self.force_mc_dropout:
            _replace_dropout_layers(self.model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected 5D tensor, got shape {tuple(x.shape)}")

        if x.shape[1] != self.in_channels:
            if x.shape[-1] == self.in_channels:
                x = x.permute(0, 4, 1, 2, 3).contiguous()
            else:
                raise ValueError(
                    "Input channel mismatch. Expected either NCDHW or NDHWC with "
                    f"{self.in_channels} channels, got shape {tuple(x.shape)}"
                )

        return self.model(x)


def build_swin_unetr_mc(
    input_shape: Tuple[int, int, int, int] = (64, 64, 64, 4),
    out_channels: int = 1,
    feature_size: int = 24,
    drop_rate: float = 0.2,
    attn_drop_rate: float = 0.2,
    dropout_path_rate: float = 0.2,
    force_mc_dropout: bool = True,
    **kwargs: Any,
) -> SwinUNETRWithMCDropout:
    """Factory aligned with the 3D baseline input contract (D, H, W, C)."""
    if len(input_shape) != 4:
        raise ValueError("input_shape must be (D, H, W, C)")

    img_size = tuple(int(v) for v in input_shape[:3])
    in_channels = int(input_shape[3])

    return SwinUNETRWithMCDropout(
        img_size=img_size,
        in_channels=in_channels,
        out_channels=int(out_channels),
        feature_size=int(feature_size),
        drop_rate=float(drop_rate),
        attn_drop_rate=float(attn_drop_rate),
        dropout_path_rate=float(dropout_path_rate),
        force_mc_dropout=bool(force_mc_dropout),
        **kwargs,
    )
