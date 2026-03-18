"""
fluid_layer3d.py — 3D reaction-diffusion fluid layer for medical volumes.

This is the volumetric extension of FluidVLA's core paradigm:
  - 3D local diffusion across voxels
  - per-voxel reaction MLP
  - global memory pump + cheap local low-resolution memory
  - adaptive compute by equilibrium / turbulence

Input / output convention:
  x: (B, C, D, H, W)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .diffusion3d import Laplacian3D
except ImportError:
    from diffusion3d import Laplacian3D


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


class MemoryPump(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)

    def forward(self, h: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return h + torch.sigmoid(self.gate(u)) * torch.tanh(self.value(u))


class ReactionMLP(nn.Module):
    def __init__(self, channels: int, expansion: int = 2):
        super().__init__()
        hidden = max(channels * expansion, channels)
        self.net = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.net(u)


class FluidLayer3D(nn.Module):
    """
    Volumetric fluid layer.

    Update rule:
      u <- u + dt * [ diffusion(u) + reaction(u) + alpha_global * h + alpha_local * low_res_context ]

    Adaptive stopping is enabled in eval mode only.
    """

    def __init__(
        self,
        channels: int,
        dilations: List[int] = [1, 2, 4],
        max_steps: int = 10,
        dt: float = 0.1,
        epsilon: float = 0.08,
        alpha: float = 0.1,
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
        self.channels = int(channels)
        self.max_steps = int(max_steps)
        self.epsilon = float(epsilon)
        self.alpha_global = nn.Parameter(torch.tensor(float(alpha)))
        self.alpha_local = nn.Parameter(torch.tensor(float(alpha) * 0.5))
        self.use_pde = bool(use_pde)
        self.norm_every = max(1, int(norm_every))
        self.stop_patience = max(1, int(stop_patience))
        self.min_steps = max(1, int(min_steps))
        self.stop_probe_dhw = tuple(int(v) for v in stop_probe_dhw)
        self.local_memory_dhw = tuple(int(v) for v in local_memory_dhw)

        self.reaction = ReactionMLP(channels)
        self.memory = MemoryPump(channels)
        self.diffusion = Laplacian3D(
            channels=channels,
            dilations=dilations,
            signed_diffusion=signed_diffusion,
            diffusion_scale=diffusion_scale,
        )
        self.local_proj = nn.Conv3d(channels, channels, kernel_size=1)
        self.log_dt = nn.Parameter(torch.log(torch.tensor(float(dt))))

        norm_type = norm_type.lower()
        if norm_type == "rmsnorm":
            self.norm = RMSNorm(channels)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(channels)
        else:
            raise ValueError("norm_type must be 'rmsnorm' or 'layernorm'")

    def _dt(self) -> torch.Tensor:
        return self.log_dt.exp().clamp(0.005, 0.35)

    def _make_stop_probe(self, x: torch.Tensor) -> torch.Tensor:
        d = min(self.stop_probe_dhw[0], x.shape[2])
        h = min(self.stop_probe_dhw[1], x.shape[3])
        w = min(self.stop_probe_dhw[2], x.shape[4])
        return F.adaptive_avg_pool3d(x, output_size=(d, h, w))

    def _should_stop(self, history: List[float], step_idx: int) -> bool:
        if self.training or self.epsilon <= 0.0:
            return False
        if (step_idx + 1) < self.min_steps:
            return False
        if len(history) < self.stop_patience:
            return False
        return max(history[-self.stop_patience:]) < self.epsilon

    def forward(self, u: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        if u.ndim != 5:
            raise ValueError(f"Expected u to have shape (B,C,D,H,W), got {tuple(u.shape)}")

        b, c, d, h, w = u.shape
        n = d * h * w
        device = u.device

        h_state = torch.zeros(b, c, device=device, dtype=u.dtype)
        stop_history: List[float] = []
        diff_turbulences: List[torch.Tensor] = []
        equilibrium_step = self.max_steps
        prev_probe = self._make_stop_probe(u).detach()

        for step in range(self.max_steps):
            diff = self.diffusion(u) if self.use_pde else torch.zeros_like(u)

            u_flat = u.permute(0, 2, 3, 4, 1).reshape(b, n, c)
            diff_flat = diff.permute(0, 2, 3, 4, 1).reshape(b, n, c)
            react = self.reaction(u_flat)

            pooled = u_flat.mean(dim=1)
            h_state = self.memory(h_state, pooled)
            h_global = h_state.unsqueeze(1).expand(-1, n, -1)

            local_d = min(self.local_memory_dhw[0], d)
            local_h = min(self.local_memory_dhw[1], h)
            local_w = min(self.local_memory_dhw[2], w)
            local_mem = F.adaptive_avg_pool3d(u, output_size=(local_d, local_h, local_w))
            local_mem = self.local_proj(local_mem)
            local_mem = F.interpolate(local_mem, size=(d, h, w), mode="trilinear", align_corners=False)
            local_mem = local_mem.permute(0, 2, 3, 4, 1).reshape(b, n, c)

            dt = self._dt()
            alpha_g = F.softplus(self.alpha_global)
            alpha_l = F.softplus(self.alpha_local)

            du = diff_flat + react + alpha_g * h_global + alpha_l * local_mem
            u_candidate = u_flat + dt * du

            u_flat_next = u_candidate
            if (step + 1) % self.norm_every == 0:
                u_flat_next = self.norm(u_flat_next)

            u_candidate_vol = u_candidate.reshape(b, d, h, w, c).permute(0, 4, 1, 2, 3).contiguous()
            u_next = u_flat_next.reshape(b, d, h, w, c).permute(0, 4, 1, 2, 3).contiguous()

            current_probe = self._make_stop_probe(u_candidate_vol).detach()
            stop_turb = (current_probe - prev_probe).abs().mean() / (prev_probe.abs().mean() + 1e-8)

            step_energy = du.abs().mean()
            lap_energy = diff_flat.abs().mean() if self.use_pde else torch.zeros((), device=device, dtype=u.dtype)
            diff_turb = stop_turb + 0.05 * step_energy + 0.01 * lap_energy
            diff_turbulences.append(diff_turb)

            stop_val = float(stop_turb.item())
            stop_history.append(stop_val)
            if stop_val < self.epsilon and equilibrium_step == self.max_steps:
                equilibrium_step = step + 1

            u = u_next
            prev_probe = current_probe

            if self._should_stop(stop_history, step):
                break

        diff_turb_mean = torch.stack(diff_turbulences).mean() if diff_turbulences else torch.zeros((), device=device, dtype=u.dtype)
        return u, {
            "steps_used": len(stop_history),
            "stop_history": stop_history,
            "equilibrium_step": equilibrium_step,
            "final_turbulence": stop_history[-1] if stop_history else 0.0,
            "min_turbulence": min(stop_history) if stop_history else 0.0,
            "diff_turbulence": diff_turb_mean,
            "pde_active": self.use_pde,
        }


if __name__ == "__main__":
    x = torch.randn(1, 16, 24, 32, 32)
    layer = FluidLayer3D(channels=16)
    y, info = layer(x)
    print("Input:", tuple(x.shape))
    print("Output:", tuple(y.shape))
    print("Info:", info)
