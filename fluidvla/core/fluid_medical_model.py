"""
fluid_medical_model.py — Medical 3D models built on FluidVLA's PDE paradigm.

Primary goal:
  - segment 3D medical volumes (CT / MRI)
  - preserve the no-attention, reaction-diffusion core
  - stay lightweight and memory-conscious
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .fluid_layer3d import FluidLayer3D, RMSNorm
except ImportError:
    from fluid_layer3d import FluidLayer3D, RMSNorm


class PatchEmbed3D(nn.Module):
    """
    Simple 3D patch embedding by strided Conv3D.
    """

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        patch_size: int | Tuple[int, int, int] = 2,
        norm_type: str = "rmsnorm",
    ):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )
        norm_type = norm_type.lower()
        if norm_type == "rmsnorm":
            self.norm = RMSNorm(d_model)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(d_model)
        else:
            raise ValueError("norm_type must be 'rmsnorm' or 'layernorm'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        b, c, d, h, w = x.shape
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        return x.permute(0, 4, 1, 2, 3).contiguous()


class SegHead3D(nn.Module):
    def __init__(self, d_model: int, n_classes: int):
        super().__init__()
        hidden = max(d_model // 2, 8)
        self.head = nn.Sequential(
            nn.Conv3d(d_model, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden, n_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, out_size: Tuple[int, int, int]) -> torch.Tensor:
        x = F.interpolate(x, size=out_size, mode="trilinear", align_corners=False)
        return self.head(x)


class FluidBotMedical3D(nn.Module):
    """
    Lightweight 3D medical segmentation model.

    Input:
      x: (B, C, D, H, W)
    Output:
      {
        "logits": (B, K, D, H, W),
        "features": latent volume,
        "info": per-layer diagnostics,
      }
    """

    def __init__(
        self,
        in_channels: int = 4,
        n_classes: int = 2,
        d_model: int = 48,
        n_layers: int = 3,
        patch_size: int | Tuple[int, int, int] = 2,
        dilations: Sequence[int] = (1, 2, 4),
        max_steps: int = 8,
        dt: float = 0.1,
        epsilon: float = 0.08,
        use_pde: bool = True,
        norm_type: str = "rmsnorm",
        norm_every: int = 2,
        local_memory_dhw: Tuple[int, int, int] = (4, 4, 4),
        signed_diffusion: bool = False,
        diffusion_scale: float = 0.25,
        stop_patience: int = 2,
        min_steps: int = 3,
        stop_probe_dhw: Tuple[int, int, int] = (4, 8, 8),
    ):
        super().__init__()
        self.patch_embed = PatchEmbed3D(
            in_channels=in_channels,
            d_model=d_model,
            patch_size=patch_size,
            norm_type=norm_type,
        )
        self.layers = nn.ModuleList(
            [
                FluidLayer3D(
                    channels=d_model,
                    dilations=list(dilations),
                    max_steps=max_steps,
                    dt=dt,
                    epsilon=epsilon,
                    use_pde=use_pde,
                    norm_type=norm_type,
                    norm_every=norm_every,
                    local_memory_dhw=local_memory_dhw,
                    signed_diffusion=signed_diffusion,
                    diffusion_scale=diffusion_scale,
                    stop_patience=stop_patience,
                    min_steps=min_steps,
                    stop_probe_dhw=stop_probe_dhw,
                )
                for _ in range(n_layers)
            ]
        )
        self.seg_head = SegHead3D(d_model=d_model, n_classes=n_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out_size = tuple(int(v) for v in x.shape[-3:])
        u = self.patch_embed(x)
        infos = []
        for layer in self.layers:
            u, info = layer(u)
            infos.append(info)
        logits = self.seg_head(u, out_size=out_size)
        return {
            "logits": logits,
            "features": u,
            "info": infos,
        }


if __name__ == "__main__":
    x = torch.randn(1, 4, 32, 64, 64)
    model = FluidBotMedical3D(in_channels=4, n_classes=2, d_model=32, n_layers=2)
    out = model(x)
    print("Input:", tuple(x.shape))
    print("Logits:", tuple(out["logits"].shape))
    print("Steps:", [i["steps_used"] for i in out["info"]])
