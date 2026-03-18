"""
fluid_model.py — Complete FluidVLA models

FluidBotClassifier : image classification (Step 0)
FluidBotVideo      : video prediction (Step 1)
FluidBotVLA        : Vision-Language-Action robotics (Steps 2a→3)

══════════════════════════════════════════════════════════════
PDE ON / OFF
══════════════════════════════════════════════════════════════
All models propagate the use_pde flag to each FluidLayer.
See fluid_layer.py for full documentation.

Via CLI:
  --no_pde    → use_pde=False in all FluidLayers
  --pde       → use_pde=True  (default)

Via code:
  model = FluidBotVLA(..., use_pde=False)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

try:
    from .fluid_layer import FluidLayer2D, FluidLayerVideo
except ImportError:
    from fluid_layer import FluidLayer2D, FluidLayerVideo


class PatchEmbed(nn.Module):
    """
    Converts an image into non-overlapping patches projected to d_model.
    Identical to ViT — but ZERO attention follows.

    For a 64x64 image with patch_size=4:
      → 16x16 = 256 patches, each embedded in d_model channels
    """
    def __init__(self, in_channels: int, d_model: int, patch_size: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, d_model,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class FluidBotClassifier(nn.Module):
    """
    Image classifier with stacked FluidLayer2D — zero attention.

    Args:
        use_pde : propagated to all FluidLayer2D (default True)
    """
    CONFIGS = {
        'tiny' : dict(d_model=64,  n_layers=2, dilations=[1, 4],     max_steps=8,  patch_size=4),
        'small': dict(d_model=128, n_layers=3, dilations=[1, 4, 8],  max_steps=10, patch_size=4),
        'base' : dict(d_model=256, n_layers=4, dilations=[1, 4, 16], max_steps=12, patch_size=4),
    }

    def __init__(
        self,
        in_channels: int   = 3,
        num_classes : int   = 10,
        d_model     : int   = 128,
        n_layers    : int   = 3,
        dilations   : list  = [1, 4, 16],
        max_steps   : int   = 12,
        dt          : float = 0.1,
        epsilon     : float = 0.02,
        patch_size  : int   = 4,
        use_pde     : bool  = True,  # ← propagated to all layers
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(in_channels, d_model, patch_size)
        self.fluid_layers = nn.ModuleList([
            FluidLayer2D(
                channels=d_model, dilations=dilations,
                max_steps=max_steps, dt=dt, epsilon=epsilon,
                use_pde=use_pde,
            )
            for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )
        self._init_weights()

    @classmethod
    def from_config(cls, config_name: str, **kwargs):
        cfg = cls.CONFIGS[config_name].copy()
        cfg.update(kwargs)
        return cls(**cfg)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        u = self.patch_embed(x)
        all_info = []
        for layer in self.fluid_layers:
            u, info = layer(u)
            all_info.append(info)
        logits = self.head(u)
        return logits, {
            'avg_steps'  : sum(i['steps_used'] for i in all_info) / len(all_info),
            'layer_steps': [i['steps_used'] for i in all_info],
            'pde_active' : all_info[0]['pde_active'],
        }

    def count_parameters(self):
        t = sum(p.numel() for p in self.parameters())
        return {'total': t, 'trainable': sum(p.numel() for p in self.parameters() if p.requires_grad)}


class FluidBotVideo(nn.Module):
    """
    Video encoder with spatio-temporal FluidLayerVideo.

    Args:
        use_pde : propagated to all FluidLayerVideo
    """
    def __init__(
        self,
        in_channels       : int   = 1,
        d_model           : int   = 64,
        n_layers          : int   = 3,
        spatial_dilations : list  = [1, 4, 16],
        temporal_dilations: list  = [1, 2],
        max_steps         : int   = 12,
        dt                : float = 0.1,
        epsilon           : float = 0.02,
        patch_size        : int   = 4,
        num_classes       : Optional[int] = None,
        causal_time       : bool  = True,
        use_pde           : bool  = True,  # ← propagated
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(in_channels, d_model, patch_size)
        self.fluid_layers = nn.ModuleList([
            FluidLayerVideo(
                channels=d_model,
                spatial_dilations=spatial_dilations,
                temporal_dilations=temporal_dilations,
                max_steps=max_steps, dt=dt, epsilon=epsilon,
                causal_time=causal_time,
                use_pde=use_pde,
            )
            for _ in range(n_layers)
        ])
        self.num_classes = num_classes
        if num_classes is not None:
            self.cls_head = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
                nn.LayerNorm(d_model),
                nn.Linear(d_model, num_classes),
            )
        self.pred_head = nn.Conv3d(
            d_model, in_channels * patch_size * patch_size, kernel_size=1
        )

    def forward(self, x: torch.Tensor):
        B, C, T, H, W = x.shape
        p = self.patch_embed.patch_size
        x_flat = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        u_flat = self.patch_embed(x_flat)
        _, d, hp, wp = u_flat.shape
        u = u_flat.reshape(B, T, d, hp, wp).permute(0, 2, 1, 3, 4)
        info_list = []
        for layer in self.fluid_layers:
            u, info = layer(u)
            info_list.append(info)
        out = {'features': u, 'info': info_list}
        if self.num_classes is not None:
            out['logits'] = self.cls_head(u)
        return out


class ActionHead(nn.Module):
    """
    Converts visual features to robot actions.
    SO-101 (6-DOF): 6 joint angles + 1 gripper = 7 dims.
    """
    def __init__(self, d_model: int, action_dim: int = 7, proprio_dim: int = 0):
        super().__init__()
        self.action_dim  = action_dim
        self.proprio_dim = proprio_dim
        in_dim = d_model + proprio_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, features: torch.Tensor,
                proprio: Optional[torch.Tensor] = None) -> torch.Tensor:
        if proprio is not None and self.proprio_dim > 0:
            features = torch.cat([features, proprio], dim=-1)
        return self.mlp(features)


class FluidBotVLA(nn.Module):
    """
    Vision-Language-Action model for robotics.
    Complete pipeline: camera frames → robot actions.

    Steps 2a → 3 of the roadmap.

    Args:
        use_pde : if False, disables the Laplacian in all FluidLayers.
                  Useful for diagnosis or stable deployment on hardware.
                  See fluid_layer.py for complete documentation.

    Adaptive compute :
        epsilon=0.02  → calm scenes ≈ 3-6 PDE steps
        epsilon=0.0   → always max_steps (more robust, slower)
        epsilon=1e9   → always 1 step (ultra-fast)
    """
    def __init__(
        self,
        image_size  : int   = 224,
        in_channels : int   = 3,
        d_model     : int   = 256,
        n_layers    : int   = 4,
        patch_size  : int   = 16,
        action_dim  : int   = 7,
        proprio_dim : int   = 7,
        max_steps   : int   = 12,
        dt          : float = 0.1,
        epsilon     : float = 0.02,
        n_frames    : int   = 4,
        use_pde     : bool  = True,  # ← main flag
    ):
        super().__init__()
        self.image_size = image_size
        self.n_frames   = n_frames
        self.d_model    = d_model
        self.use_pde    = use_pde  # stored for logs / introspection

        self.visual = FluidBotVideo(
            in_channels=in_channels, d_model=d_model, n_layers=n_layers,
            spatial_dilations=[1, 4, 16], temporal_dilations=[1, 2],
            max_steps=max_steps, dt=dt, epsilon=epsilon,
            patch_size=patch_size, causal_time=True,
            use_pde=use_pde,
        )
        self.pool        = nn.AdaptiveAvgPool3d(1)
        self.action_head = ActionHead(d_model, action_dim, proprio_dim)

    def forward(
        self,
        frames : torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
    ) -> dict:
        vis_out  = self.visual(frames)
        features = vis_out['features']
        pooled   = self.pool(features).flatten(1)
        actions  = self.action_head(pooled, proprio)
        return {
            'actions' : actions,
            'features': pooled,
            'info'    : vis_out['info'],
        }

    def count_parameters(self):
        t = sum(p.numel() for p in self.parameters())
        return {'total': t, 'M': t / 1e6}


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    for use_pde in [True, False]:
        tag = "PDE ON" if use_pde else "PDE OFF"
        vla = FluidBotVLA(image_size=64, d_model=128, n_layers=3,
                          patch_size=4, use_pde=use_pde).to(device)
        p = vla.count_parameters()
        frames  = torch.randn(2, 3, 4, 64, 64, device=device)
        proprio = torch.randn(2, 7, device=device)
        vla.eval()
        with torch.no_grad():
            out = vla(frames, proprio)
        steps = sum(i['steps_used'] for i in out['info']) / len(out['info'])
        print(f"[{tag}] {p['M']:.2f}M params | actions={tuple(out['actions'].shape)} | steps={steps:.1f}")
    print("✅ OK")