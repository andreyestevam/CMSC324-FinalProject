"""Custom Swin-UNETR model with Monte Carlo dropout for the uncertainty estimation."""

from __future__ import annotations # for lazy evaluation of type hints

import inspect
from typing import Any, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.layers import DropPath
from monai.networks.nets import SwinUNETR as MonaiSwinUNETR


# explicitly export these definitions only
__all__ = ["SwinUNETRWithMCDropout", "build_swin_unetr_mc"]


def _drop_path(
    x: torch.Tensor,
    drop_probability: float = 0.0,
) -> torch.Tensor:
    """Drop a skip-connection path with a probablity of `drop_probability`"""        
    assert (0.0 <= drop_probability) and (drop_probability <= 1.0), ValueError(f"p is epxected to be between 0 and 1, got {drop_probability}")

    keep_probability = 1.0 - drop_probability
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape)
    random_tensor = random_tensor.bernoulli_(keep_probability) # fill the tensor with 0s and 1s with keep_probability
    if keep_probability > 0.0:
        random_tensor.div_(keep_probability) # this is to scale the activation to have the same expected value
    return x * random_tensor


class MCDropout(nn.Module):
    """Monte Carlo dropout for a single neuron with probability p"""
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        assert (0.0 <= p) and (p <= 1.0), ValueError(f"p is epxected to be between 0 and 1, got {p}")
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout(x, p=self.p, training=True) # training=True forces dropout to be active during both training and interence, which is MC dropout


class MCDropPath(nn.Module):
    """Skip-connection path with Monte Carlo dropout"""
    def __init__(self, drop_probability: float = 0.0) -> None:
        super().__init__()
        self.drop_probability = float(drop_probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _drop_path(x, drop_probability=self.drop_probability)


def _replace_dropout_layers(module: nn.Module) -> None:
    """Recursively replace dropout layers (both neuron and skip-connection) with their Monte Carlo counterparts."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Dropout): # if a single neuron
            setattr(module, name, MCDropout(p=child.p))
            continue

        if isinstance(child, DropPath): # if a skip-connection
            drop_probability = float(getattr(child, "drop_probability", 0.0))
            setattr(module, name, MCDropPath(drop_probability=drop_probability))
            continue

        _replace_dropout_layers(child) # recursive call


class SwinUNETRWithMCDropout(nn.Module):
    """MONAI SwinUNETR wrapper with MC dropout."""
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
        # clean the kwargs to pass along to the SwinUNETR constructor
        allowed = set(inspect.signature(MonaiSwinUNETR.__init__).parameters)
        filtered_kwargs = {k: v for k, v in init_kwargs.items() if k in allowed}

        # initialize the original SwinUNETR model
        self.model = MonaiSwinUNETR(**filtered_kwargs)

        # and apply the MC dropout replacement
        if self.force_mc_dropout:
            _replace_dropout_layers(self.model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NCDHW is expected
        return self.model(x)


def build_swin_unetr_mc(
    input_shape: Tuple[int, int, int, int] = (4, 64, 64, 64),
    out_channels: int = 1,
    feature_size: int = 24,
    drop_rate: float = 0.2,
    attn_drop_rate: float = 0.2,
    dropout_path_rate: float = 0.2,
    force_mc_dropout: bool = True,
    **kwargs: Any,
) -> SwinUNETRWithMCDropout:
    if len(input_shape) != 4:
        raise ValueError("input_shape must be (C, D, H, W)")

    in_channels = int(input_shape[0])
    img_size = tuple(int(v) for v in input_shape[1:])

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
