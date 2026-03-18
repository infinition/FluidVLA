"""Action heads and VLA wrappers built on top of FluidBotVideo.

V2 changes vs original:
  1. Spatial-preserving pooling: AdaptiveAvgPool3d(1, S, S) keeps S×S spatial grid
     instead of crushing everything to a single vector.
  2. Action chunking: predict chunk_size future actions in one forward pass.
  3. Larger action MLP with LayerNorm to handle the richer input.
  4. Delta-action mode flag (actual delta logic lives in train/inference scripts).
  5. Full backward compatibility: spatial_pool_size=1, chunk_size=1 recovers V1.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

try:
    from .video_models import FluidBotVideo
except ImportError:
    from video_models import FluidBotVideo


# ---------------------------------------------------------------------------
# Legacy action head — kept for checkpoint loading compatibility
# ---------------------------------------------------------------------------

class ActionHead(nn.Module):
    """Maps pooled visual features to robot actions (V1 — global pool)."""

    def __init__(self, d_model: int, action_dim: int = 7, proprio_dim: int = 0):
        super().__init__()
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        in_dim = d_model + proprio_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, features: torch.Tensor, proprio: Optional[torch.Tensor] = None) -> torch.Tensor:
        if proprio is not None and self.proprio_dim > 0:
            features = torch.cat([features, proprio], dim=-1)
        return self.mlp(features)


# ---------------------------------------------------------------------------
# V2 action head — spatial-aware with optional chunking
# ---------------------------------------------------------------------------

class SpatialActionHead(nn.Module):
    """Maps spatially-preserved visual features to robot actions.

    Instead of receiving a single (B, d_model) vector, this head receives
    (B, d_model * S * S) where S = spatial_pool_size.  This preserves WHERE
    things are in the image, which is critical for manipulation tasks.

    Optional action chunking: predict `chunk_size` future actions at once
    to give the policy a short planning horizon.
    """

    def __init__(
        self,
        d_model: int,
        spatial_pool_size: int = 4,
        action_dim: int = 6,
        proprio_dim: int = 0,
        chunk_size: int = 1,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self.chunk_size = chunk_size
        self.spatial_pool_size = spatial_pool_size

        in_dim = d_model * spatial_pool_size * spatial_pool_size + proprio_dim
        out_dim = action_dim * chunk_size

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, features: torch.Tensor, proprio: Optional[torch.Tensor] = None) -> torch.Tensor:
        if proprio is not None and self.proprio_dim > 0:
            features = torch.cat([features, proprio], dim=-1)
        out = self.mlp(features)
        if self.chunk_size > 1:
            return out.view(out.shape[0], self.chunk_size, self.action_dim)
        return out


# ---------------------------------------------------------------------------
# FluidBotVLA V2
# ---------------------------------------------------------------------------

class FluidBotVLA(nn.Module):
    """
    Vision → latent fluid dynamics → action head.

    V2 improvements:
      - spatial_pool_size > 1 preserves spatial layout before the action head
      - chunk_size > 1 enables short-horizon action prediction
      - Backward compatible: spatial_pool_size=1, chunk_size=1 = V1 behavior
    """

    def __init__(
        self,
        image_size: int = 224,
        in_channels: int = 3,
        d_model: int = 256,
        n_layers: int = 4,
        patch_size: int = 16,
        action_dim: int = 7,
        proprio_dim: int = 7,
        max_steps: int = 12,
        dt: float = 0.1,
        epsilon: float = 0.08,
        n_frames: int = 4,
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
        # ── V2 additions ──
        spatial_pool_size: int = 1,   # 1 = V1 global pool, 4 = recommended V2
        chunk_size: int = 1,          # 1 = single-step, 4-8 = recommended V2
    ):
        super().__init__()
        self.image_size = image_size
        self.n_frames = n_frames
        self.d_model = d_model
        self.use_pde = use_pde
        self.spatial_pool_size = spatial_pool_size
        self.chunk_size = chunk_size

        self.visual = FluidBotVideo(
            in_channels=in_channels,
            d_model=d_model,
            n_layers=n_layers,
            spatial_dilations=(1, 4, 16),
            temporal_dilations=(1, 2),
            max_steps=max_steps,
            dt=dt,
            epsilon=epsilon,
            patch_size=patch_size,
            causal_time=True,
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

        # ── Pooling strategy ──
        # V1: AdaptiveAvgPool3d(1) → (B, d_model)
        # V2: AdaptiveAvgPool3d((1, S, S)) → (B, d_model * S * S)
        self.pool = nn.AdaptiveAvgPool3d((1, spatial_pool_size, spatial_pool_size))

        # ── Action head ──
        if spatial_pool_size == 1 and chunk_size == 1:
            # V1-compatible path: use legacy ActionHead for checkpoint loading
            self.action_head = ActionHead(d_model, action_dim, proprio_dim)
            self._use_spatial_head = False
        else:
            self.action_head = SpatialActionHead(
                d_model=d_model,
                spatial_pool_size=spatial_pool_size,
                action_dim=action_dim,
                proprio_dim=proprio_dim,
                chunk_size=chunk_size,
            )
            self._use_spatial_head = True

    def forward(self, frames: torch.Tensor, proprio: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        vis_out = self.visual(frames)
        features = vis_out["features"]   # (B, d_model, T, H', W')

        pooled = self.pool(features)     # (B, d_model, 1, S, S)
        pooled = pooled.flatten(1)       # (B, d_model * S * S)  or (B, d_model) if S=1

        actions = self.action_head(pooled, proprio)

        return {
            "actions": actions,
            "features": pooled,
            "info": vis_out["info"],
        }

    def count_parameters(self) -> Dict[str, float]:
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "M": total / 1e6}
