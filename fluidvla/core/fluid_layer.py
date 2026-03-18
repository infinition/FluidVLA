"""
fluid_layer.py — The core FluidLayer

One FluidLayer = one integration of the full PDE:
  u_{t+1} = LayerNorm(u_t + Δt · [diffusion(u_t) + reaction(u_t) + α·h_t])

══════════════════════════════════════════════════════════════
PDE ON / OFF  —  `use_pde` flag
══════════════════════════════════════════════════════════════

Each FluidLayer2D and FluidLayerVideo accepts a `use_pde: bool` argument.

  use_pde=True  (DEFAULT)
    Complete PDE integration with multi-scale Laplacian + early-stopping.
    FluidVLA's unique claim: O(N) memory, O(1) h-state, Turing Equilibrium.

  use_pde=False  (FALLBACK / DIAGNOSTIC MODE)
    The Laplacian diffusion term is DISABLED.
    Each step is reduced to:
      u_{t+1} = LayerNorm(u_t + dt · [reaction(u_t) + α·h_t])
    This is essentially a deep residual network (stacked MLPs) —
    always valid, always fast, but without spatial propagation.

  WHY THIS FLAG EXISTS:
    On the real robot, if you observe chaotic movements, diverging losses,
    or unstable behavior, the key question is:
    "Is it the Laplacian or the rest of the network causing the issue?"

    Recommended diagnostic workflow:
      1. Run with --no_pde → clean baseline without diffusion
      2. Confirm that actions are coherent and stable
      3. Reactivate the PDE (--pde, default) → if stable too, perfect
      4. If unstable with PDE: adjust dilations, dt, or epsilon

    This flag also allows quick testing on hardware without
    retraining, in inference-only mode.

  CHECKPOINT COMPATIBILITY:
    The flag is NOT in the state_dict — it's a constructor argument.
    A checkpoint trained with use_pde=True can be loaded into a
    model instantiated with use_pde=False (the D weights of the Laplacian are
    ignored at inference). The reverse is also possible.

  LIGHTWEIGHT ALTERNATIVE (without disabling the PDE):
    epsilon=1e9 → Turing Equilibrium always stops after 1 step.
    The Laplacian still runs 1 time — less savings but
    the weights remain active.

══════════════════════════════════════════════════════════════
Adaptive compute (Turing Equilibrium)
══════════════════════════════════════════════════════════════
  turbulence = mean(|u_t - u_{t-1}| / (|u_{t-1}| + ε))
  if turbulence < epsilon → STOP (eval only)

  epsilon=0.02 → default, stops on calm scenes in 3-6 steps
  epsilon=0.0  → disables early stopping, always max_steps
  epsilon=1e9  → almost always stops after 1 step
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

try:
    from .diffusion import Laplacian1D, Laplacian2D, LaplacianSpatioTemporal
except ImportError:
    from diffusion import Laplacian1D, Laplacian2D, LaplacianSpatioTemporal


class MemoryPump(nn.Module):
    """
    Gated memory accumulator — the h-state.

    h_t = h_{t-1} + σ(W_gate · u) ⊙ tanh(W_value · u)

    Captures what happened during integration steps.
    DOES NOT persist between two forward() calls — automatically reset on each call.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.gate  = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)

    def forward(self, h: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return h + torch.sigmoid(self.gate(u)) * torch.tanh(self.value(u))


class ReactionMLP(nn.Module):
    """
    Per-position reaction function R(u, θ).
    2-layer MLP applied independently to each spatial position.
    Creates new combinations of features — without this, diffusion
    alone smoothes everything towards a uniform average.
    """
    def __init__(self, channels: int, expansion: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(channels, channels * expansion),
            nn.GELU(),
            nn.Linear(channels * expansion, channels),
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.net(u)


class FluidLayer2D(nn.Module):
    """
    Complete FluidLayer for 2D images.

    T integration steps of:
      u_{t+1} = LayerNorm(u_t + Δt · [Σ D_k·∇²(u_t) + R(u_t,θ) + α·h_t])

    With adaptive early-stopping (Turing Equilibrium) in eval mode.

    Args:
        channels  : feature dimension (d_model)
        dilations : dilation scales for the multi-resolution Laplacian
        max_steps : maximum integration steps per forward pass
        dt        : initial integration step (learnable)
        epsilon   : Turing Equilibrium threshold (0.02 default)
        alpha     : initial weight of the memory pump
        use_pde   : if False, disables the Laplacian term (see module docstring)
    """
    def __init__(
        self,
        channels : int,
        dilations: list  = [1, 4, 16],
        max_steps: int   = 12,
        dt       : float = 0.1,
        epsilon  : float = 0.02,
        alpha    : float = 0.1,
        use_pde  : bool  = True,
    ):
        super().__init__()
        self.channels   = channels
        self.max_steps  = max_steps
        self.epsilon    = epsilon
        self.use_pde    = use_pde

        # Always instantiated even if use_pde=False
        # → weights exist to be able to reactivate without changing the architecture
        self.diffusion   = Laplacian2D(channels, dilations)
        self.reaction    = ReactionMLP(channels)
        self.memory      = MemoryPump(channels)
        self.norm        = nn.LayerNorm(channels)
        self.alpha_param = nn.Parameter(torch.tensor(alpha))
        self.log_dt      = nn.Parameter(torch.log(torch.tensor(dt)))

    def forward(self, u: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        u: (B, C, H, W)
        returns: (u_final, info_dict)
          info_dict keys:
            steps_used         : steps actually executed
            turbulence_history : list of turbulence values per step
            equilibrium_step   : first step below epsilon
            final_turbulence   : last turbulence value
            diff_turbulence    : differentiable turbulence for eq_loss
            pde_active         : bool, indicates if the Laplacian was active
        """
        B, C, H, W = u.shape
        device = u.device

        h             = torch.zeros(B, C, device=device, dtype=u.dtype)
        turbulence_history = []
        equilibrium_step   = self.max_steps
        u_prev_diff        = u
        diff_turbulences   = []

        for step in range(self.max_steps):

            # ── Diffusion term (conditional) ─────────────────────────
            if self.use_pde:
                diff = self.diffusion(u)    # Active Laplacian: (B, C, H, W)
            else:
                diff = torch.zeros_like(u)  # Disabled Laplacian: zero
            # ─────────────────────────────────────────────────────────────

            u_flat    = u.permute(0, 2, 3, 1).reshape(B, H * W, C)
            diff_flat = diff.permute(0, 2, 3, 1).reshape(B, H * W, C)

            react     = self.reaction(u_flat)
            u_pooled  = u_flat.mean(dim=1)
            h         = self.memory(h, u_pooled)
            h_spatial = h.unsqueeze(1).expand(-1, H * W, -1)

            alpha = F.softplus(self.alpha_param)
            dt    = self.log_dt.exp().clamp(0.01, 0.2)
            du    = diff_flat + react + alpha * h_spatial
            u_flat_new = self.norm(u_flat + dt * du)

            u = u_flat_new.reshape(B, H, W, C).permute(0, 3, 1, 2)

            diff_turb = (
                (u - u_prev_diff).abs().mean()
                / (u_prev_diff.detach().abs().mean() + 1e-8)
            )
            diff_turbulences.append(diff_turb)
            u_prev_diff = u

            with torch.no_grad():
                turbulence_val = diff_turb.item()
            turbulence_history.append(turbulence_val)

            if turbulence_val < self.epsilon and equilibrium_step == self.max_steps:
                equilibrium_step = step + 1

            # Early stop in eval only
            if not self.training and turbulence_val < self.epsilon:
                break

        diff_turb_mean = (
            torch.stack(diff_turbulences).mean()
            if diff_turbulences
            else torch.tensor(0.0, device=device)
        )

        return u, {
            'steps_used'        : len(turbulence_history),
            'turbulence_history': turbulence_history,
            'equilibrium_step'  : equilibrium_step,
            'final_turbulence'  : turbulence_history[-1] if turbulence_history else 0.0,
            'diff_turbulence'   : diff_turb_mean,
            'pde_active'        : self.use_pde,
        }


class FluidLayerVideo(nn.Module):
    """
    Complete FluidLayer for video — spatio-temporal diffusion.

    Same PDE structure as FluidLayer2D but with the 3D Laplacian,
    allowing motion to naturally propagate between frames.

    Input/output layout: (B, C, T, H, W)

    Args:
        channels           : feature dimension
        spatial_dilations  : spatial scales of the 2D Laplacian
        temporal_dilations : temporal scales of the Laplacian
        max_steps          : maximum integration steps
        dt                 : initial integration step (learnable)
        epsilon            : Turing Equilibrium threshold
        causal_time        : if True, only looks at past frames
                             (mandatory for real-time deployment)
        use_pde            : if False, disables the Laplacian term
                             See FluidLayer2D and module docstring.
    """
    def __init__(
        self,
        channels           : int,
        spatial_dilations  : list  = [1, 4, 16],
        temporal_dilations : list  = [1, 2],
        max_steps          : int   = 12,
        dt                 : float = 0.1,
        epsilon            : float = 0.02,
        causal_time        : bool  = True,
        use_pde            : bool  = True,
    ):
        super().__init__()
        self.max_steps = max_steps
        self.epsilon   = epsilon
        self.use_pde   = use_pde

        self.diffusion = LaplacianSpatioTemporal(
            channels, spatial_dilations, temporal_dilations, causal_time,
        )
        self.reaction  = ReactionMLP(channels)
        self.memory    = MemoryPump(channels)
        self.norm      = nn.LayerNorm(channels)
        self.alpha     = nn.Parameter(torch.tensor(0.1))
        self.log_dt    = nn.Parameter(torch.log(torch.tensor(dt)))

    def forward(self, u: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        B, C, T, H, W = u.shape
        device = u.device
        N = T * H * W

        h = torch.zeros(B, C, device=device, dtype=u.dtype)
        turbulence_history = []
        equilibrium_step   = self.max_steps
        u_prev_diff        = u
        diff_turbulences   = []

        for step in range(self.max_steps):

            # ── Diffusion term (conditional) ─────────────────────────
            if self.use_pde:
                diff = self.diffusion(u)
            else:
                diff = torch.zeros_like(u)
            # ─────────────────────────────────────────────────────────────

            u_flat    = u.permute(0, 2, 3, 4, 1).reshape(B, N, C)
            diff_flat = diff.permute(0, 2, 3, 4, 1).reshape(B, N, C)

            react     = self.reaction(u_flat)
            u_pooled  = u_flat.mean(dim=1)
            h         = self.memory(h, u_pooled)
            h_spatial = h.unsqueeze(1).expand(-1, N, -1)

            alpha = F.softplus(self.alpha)
            dt    = self.log_dt.exp().clamp(0.01, 0.2)
            du    = diff_flat + react + alpha * h_spatial
            u_flat_new = self.norm(u_flat + dt * du)

            u = u_flat_new.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)

            diff_turb = (
                (u - u_prev_diff).abs().mean()
                / (u_prev_diff.detach().abs().mean() + 1e-8)
            )
            diff_turbulences.append(diff_turb)
            u_prev_diff = u

            with torch.no_grad():
                turbulence_val = diff_turb.item()
            turbulence_history.append(turbulence_val)

            if turbulence_val < self.epsilon and equilibrium_step == self.max_steps:
                equilibrium_step = step + 1

            if not self.training and turbulence_val < self.epsilon:
                break

        diff_turb_mean = (
            torch.stack(diff_turbulences).mean()
            if diff_turbulences
            else torch.tensor(0.0, device=device)
        )

        return u, {
            'steps_used'        : len(turbulence_history),
            'turbulence_history': turbulence_history,
            'equilibrium_step'  : equilibrium_step,
            'final_turbulence'  : turbulence_history[-1] if turbulence_history else 0.0,
            'diff_turbulence'   : diff_turb_mean,
            'pde_active'        : self.use_pde,
        }


if __name__ == '__main__':
    print("=" * 60)
    print("FluidVLA — FluidLayer Sanity Check (PDE ON/OFF)")
    print("=" * 60)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    for use_pde in [True, False]:
        tag = "PDE ON" if use_pde else "PDE OFF"
        layer = FluidLayer2D(channels=64, max_steps=12, use_pde=use_pde).to(device)
        layer.eval()
        x = torch.randn(2, 64, 32, 32, device=device)
        with torch.no_grad():
            out, info = layer(x)
        print(f"[{tag}] shape={tuple(out.shape)} steps={info['steps_used']} pde={info['pde_active']}")

    print("\n✅ Checks passed.")