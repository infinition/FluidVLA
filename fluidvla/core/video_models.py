"""Video-oriented FluidVLA model components."""

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

try:
    from .fluid_layer import FluidLayerVideo
    from .vision_models import PatchEmbed
except ImportError:
    from fluid_layer import FluidLayerVideo
    from vision_models import PatchEmbed


class FluidBotVideo(nn.Module):
    """
    Video encoder with stacked FluidLayerVideo blocks.

    Output convention:
      out["features"] = latent video tensor of shape (B, C, T, H', W')
      out["info"]     = list of per-layer diagnostics
    """

    def __init__(
        self,
        in_channels: int = 1,
        d_model: int = 64,
        n_layers: int = 3,
        spatial_dilations: Sequence[int] = (1, 4, 16),
        temporal_dilations: Sequence[int] = (1, 2),
        max_steps: int = 12,
        dt: float = 0.1,
        epsilon: float = 0.08,
        patch_size: int = 4,
        num_classes: Optional[int] = None,
        causal_time: bool = True,
        use_pde: bool = True,
        norm_type: str = "rmsnorm",
        norm_every: int = 2,
        local_memory_hw: int = 4,
        signed_diffusion: bool = False,
        diffusion_scale: float = 0.25,
        temporal_mode: str = "backward_diff",
        stop_patience: int = 2,
        min_steps: int = 3,
        stop_probe_hw: int = 8,
        stop_probe_t: int = 2,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            d_model=d_model,
            patch_size=patch_size,
            norm_type=norm_type,
        )

        self.fluid_layers = nn.ModuleList(
            [
                FluidLayerVideo(
                    channels=d_model,
                    spatial_dilations=list(spatial_dilations),
                    temporal_dilations=list(temporal_dilations),
                    max_steps=max_steps,
                    dt=dt,
                    epsilon=epsilon,
                    causal_time=causal_time,
                    use_pde=use_pde,
                    norm_type=norm_type,
                    norm_every=norm_every,
                    local_memory_hw=local_memory_hw,
                    signed_diffusion=signed_diffusion,
                    diffusion_scale=diffusion_scale,
                    temporal_mode=temporal_mode,
                    stop_patience=stop_patience,
                    min_steps=min_steps,
                    stop_probe_hw=stop_probe_hw,
                    stop_probe_t=stop_probe_t,
                )
                for _ in range(n_layers)
            ]
        )

        self.num_classes = num_classes
        if num_classes is not None:
            self.cls_head = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
                nn.LayerNorm(d_model),
                nn.Linear(d_model, num_classes),
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, c, t, h, w = x.shape

        x_flat = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        u_flat = self.patch_embed(x_flat)

        _, d_model, hp, wp = u_flat.shape
        u = u_flat.reshape(b, t, d_model, hp, wp).permute(0, 2, 1, 3, 4).contiguous()

        info_list = []
        for layer in self.fluid_layers:
            u, info = layer(u)
            info_list.append(info)

        out: Dict[str, torch.Tensor] = {"features": u, "info": info_list}
        if self.num_classes is not None:
            out["logits"] = self.cls_head(u)
        return out
