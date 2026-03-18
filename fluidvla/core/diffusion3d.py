"""
diffusion3d.py — 3D multi-scale diffusion operators for FluidVLA medical experiments.

This file extends the FluidVLA reaction-diffusion paradigm to volumetric data:
  - local propagation only (no attention)
  - multi-scale 3D Laplacian
  - optional signed diffusion for research
  - complexity proportional to the number of voxels times the number of scales

Input / output convention:
  x: (B, C, D, H, W)
  y: (B, C, D, H, W)
"""

from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_laplacian_kernel_3d(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    6-neighbour discrete 3D Laplacian kernel.

    Center = -6
    Direct neighbours along x/y/z = +1
    """
    k = torch.zeros((1, 1, 3, 3, 3), device=device, dtype=dtype)
    k[0, 0, 1, 1, 1] = -6.0
    k[0, 0, 0, 1, 1] = 1.0
    k[0, 0, 2, 1, 1] = 1.0
    k[0, 0, 1, 0, 1] = 1.0
    k[0, 0, 1, 2, 1] = 1.0
    k[0, 0, 1, 1, 0] = 1.0
    k[0, 0, 1, 1, 2] = 1.0
    return k


class Laplacian3D(nn.Module):
    """
    Multi-scale 3D Laplacian with learnable per-channel diffusion strengths.

    Each dilation gives a local propagation radius in voxel space.
    This is the direct volumetric analogue of FluidVLA's 2D diffusion.
    """

    def __init__(
        self,
        channels: int,
        dilations: Iterable[int] = (1, 2, 4),
        signed_diffusion: bool = False,
        diffusion_scale: float = 0.25,
    ):
        super().__init__()
        self.channels = int(channels)
        self.dilations: List[int] = [int(d) for d in dilations]
        self.signed_diffusion = bool(signed_diffusion)
        self.diffusion_scale = float(diffusion_scale)

        # One learnable diffusion coefficient tensor per dilation, per channel.
        self.diff_params = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.channels)) for _ in self.dilations]
        )

    def _coeff(self, idx: int) -> torch.Tensor:
        p = self.diff_params[idx]
        if self.signed_diffusion:
            return self.diffusion_scale * torch.tanh(p)

        # Positive coefficient in [0, diffusion_scale]
        return self.diffusion_scale * torch.sigmoid(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected x to have shape (B,C,D,H,W), got {tuple(x.shape)}")

        b, c, d, h, w = x.shape
        if c != self.channels:
            raise ValueError(f"Channel mismatch: module has {self.channels}, input has {c}")

        kernel = _make_laplacian_kernel_3d(x.device, x.dtype).repeat(c, 1, 1, 1, 1)
        out = torch.zeros_like(x)

        for i, dilation in enumerate(self.dilations):
            lap = F.conv3d(
                x,
                kernel,
                bias=None,
                stride=1,
                padding=dilation,
                dilation=dilation,
                groups=c,
            )
            coeff = self._coeff(i).view(1, c, 1, 1, 1)
            out = out + coeff * lap

        return out


if __name__ == "__main__":
    # Small self-test
    x = torch.randn(2, 8, 16, 32, 32)
    lap = Laplacian3D(channels=8, dilations=(1, 2, 4))
    y = lap(x)
    print("Input:", tuple(x.shape))
    print("Output:", tuple(y.shape))
