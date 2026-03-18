"""
diffusion.py — Multi-scale Laplacian operators for FluidBot

Implements discrete Laplacian convolutions in 1D, 2D, and 3D (spatio-temporal)
with learnable diffusion coefficients and multiple dilation scales.

Mathematical basis:
  1D: ∇²u(i)     = u(i-d) - 2·u(i) + u(i+d)
  2D: ∇²u(x,y)   = u(x+d,y) + u(x-d,y) + u(x,y+d) + u(x,y-d) - 4·u(x,y)
  3D: ∇²u(x,y,t) = [2D spatial] + D_t·(u(x,y,t+1) + u(x,y,t-1) - 2·u(x,y,t))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class Laplacian1D(nn.Module):
    """
    1D discrete Laplacian with multi-scale dilations.
    Used for sequential data (text, 1D sensor streams).
    
    Kernel: [1, -2, 1] applied at dilation d
    Causal variant: left-only padding for autoregressive use.
    """
    def __init__(self, channels: int, dilations: List[int] = [1, 4, 16], causal: bool = False):
        super().__init__()
        self.dilations = dilations
        self.causal = causal
        # Learnable diffusion coefficient per scale — initialized small for stability
        self.D = nn.Parameter(torch.ones(len(dilations), channels) * 0.1)
        
        # Fixed Laplacian kernel — NOT learned, this is physics
        kernel = torch.tensor([1.0, -2.0, 1.0])
        self.register_buffer('kernel', kernel)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        u: (B, C, L) — batch, channels, length
        returns: (B, C, L) — Laplacian-diffused state
        """
        B, C, L = u.shape
        out = torch.zeros_like(u)
        
        for i, d in enumerate(self.dilations):
            # Pad for dilation
            if self.causal:
                # Left-only padding — no future leakage
                padded = F.pad(u, (2 * d, 0))
            else:
                padded = F.pad(u, (d, d), mode='replicate')
            
            # Apply [1, -2, 1] kernel via grouped convolution
            k = self.kernel.view(1, 1, 3).expand(C, 1, 3)
            lap = F.conv1d(padded, k, dilation=d, groups=C)
            
            # Trim to original length if needed
            if lap.shape[-1] > L:
                lap = lap[..., :L]
            
            # Scale by learnable D_k (positive via softplus for stability)
            D_k = F.softplus(self.D[i]).unsqueeze(0).unsqueeze(-1)  # (1, C, 1)
            out = out + D_k * lap
        
        return out


class Laplacian2D(nn.Module):
    """
    2D discrete Laplacian with multi-scale dilations.
    Primary operator for image processing.
    
    Kernel (5-point stencil at dilation d):
      u(x+d,y) + u(x-d,y) + u(x,y+d) + u(x,y-d) - 4·u(x,y)
    
    This is equivalent to the sum of two 1D Laplacians along each axis.
    """
    def __init__(self, channels: int, dilations: List[int] = [1, 4, 16]):
        super().__init__()
        self.dilations = dilations
        self.D = nn.Parameter(torch.ones(len(dilations), channels) * 0.1)
        
        # 2D Laplacian kernel (5-point stencil)
        kernel_2d = torch.zeros(3, 3)
        kernel_2d[1, 0] = 1.0   # left
        kernel_2d[1, 2] = 1.0   # right
        kernel_2d[0, 1] = 1.0   # up
        kernel_2d[2, 1] = 1.0   # down
        kernel_2d[1, 1] = -4.0  # center
        self.register_buffer('kernel_2d', kernel_2d)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        u: (B, C, H, W) — batch, channels, height, width
        returns: (B, C, H, W)
        """
        B, C, H, W = u.shape
        out = torch.zeros_like(u)
        
        for i, d in enumerate(self.dilations):
            # Replicate padding to handle borders naturally
            padded = F.pad(u, (d, d, d, d), mode='replicate')
            
            # Apply 2D Laplacian kernel via grouped depthwise conv
            k = self.kernel_2d.view(1, 1, 3, 3).expand(C, 1, 3, 3)
            lap = F.conv2d(padded, k, dilation=d, groups=C)
            
            # Trim to original spatial size
            lap = lap[:, :, :H, :W]
            
            D_k = F.softplus(self.D[i]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            out = out + D_k * lap
        
        return out


class LaplacianSpatioTemporal(nn.Module):
    """
    3D Spatio-Temporal Laplacian for video processing.
    
    ∇²u(x,y,t) = ∇²_spatial(u) + D_t · ∇²_temporal(u)
    
    where:
      ∇²_spatial  = 2D Laplacian at each frame
      ∇²_temporal = u(x,y,t+1) + u(x,y,t-1) - 2·u(x,y,t)
    
    This means motion between frames propagates via diffusion —
    no optical flow required, motion is an emergent property.
    
    Causal mode: temporal padding only looks backward (no future frames)
    which is required for real-time robot deployment.
    """
    def __init__(
        self,
        channels: int,
        spatial_dilations: List[int] = [1, 4, 16],
        temporal_dilations: List[int] = [1, 2],
        causal_time: bool = True
    ):
        super().__init__()
        self.spatial_dilations = spatial_dilations
        self.temporal_dilations = temporal_dilations
        self.causal_time = causal_time
        
        # Spatial diffusion coefficients
        self.D_spatial = nn.Parameter(torch.ones(len(spatial_dilations), channels) * 0.1)
        # Temporal diffusion coefficients (often smaller — time axis is finer)
        self.D_temporal = nn.Parameter(torch.ones(len(temporal_dilations), channels) * 0.05)
        
        kernel_2d = torch.zeros(3, 3)
        kernel_2d[1, 0] = 1.0
        kernel_2d[1, 2] = 1.0
        kernel_2d[0, 1] = 1.0
        kernel_2d[2, 1] = 1.0
        kernel_2d[1, 1] = -4.0
        self.register_buffer('kernel_2d', kernel_2d)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        u: (B, C, T, H, W) — batch, channels, time, height, width
        returns: (B, C, T, H, W)
        """
        B, C, T, H, W = u.shape
        out = torch.zeros_like(u)
        
        # --- Spatial diffusion (applied frame-by-frame) ---
        u_flat = u.reshape(B * T, C, H, W)
        spatial_lap = torch.zeros_like(u_flat)
        
        for i, d in enumerate(self.spatial_dilations):
            padded = F.pad(u_flat, (d, d, d, d), mode='replicate')
            k = self.kernel_2d.view(1, 1, 3, 3).expand(C, 1, 3, 3)
            lap = F.conv2d(padded, k, dilation=d, groups=C)[:, :, :H, :W]
            D_k = F.softplus(self.D_spatial[i]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            spatial_lap = spatial_lap + D_k * lap
        
        out = out + spatial_lap.view(B, C, T, H, W)
        
        # --- Temporal diffusion ---
        for i, dt in enumerate(self.temporal_dilations):
            if self.causal_time:
                # Only look at past frames (real-time safe)
                u_past = F.pad(u, (0, 0, 0, 0, dt, 0))[:, :, :T, :, :]
                # Future is approximated as current (best guess with causal constraint)
                temporal_lap = u_past - 2 * u + u  # simplifies to u_past - u
                temporal_lap = u_past - u
            else:
                # Non-causal: look both past and future (for offline training)
                u_past   = F.pad(u, (0, 0, 0, 0, dt, 0))[:, :, :T, :, :]
                u_future = F.pad(u, (0, 0, 0, 0, 0, dt))[:, :, dt:dt+T, :, :]
                temporal_lap = u_past + u_future - 2 * u
            
            D_t = F.softplus(self.D_temporal[i]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            out = out + D_t * temporal_lap
        
        return out


# ─────────────────────────────────────────
# Quick sanity test
# ─────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("FluidBot — Diffusion Operators Sanity Check")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # --- 1D ---
    print("\n[1D Laplacian]")
    lap1d = Laplacian1D(channels=64, dilations=[1, 4, 16]).to(device)
    x1d = torch.randn(4, 64, 128, device=device)  # B=4, C=64, L=128
    y1d = lap1d(x1d)
    assert y1d.shape == x1d.shape, f"Shape mismatch: {y1d.shape}"
    print(f"  Input:  {tuple(x1d.shape)}")
    print(f"  Output: {tuple(y1d.shape)} ✓")
    print(f"  D values (first 3): {F.softplus(lap1d.D[:, 0]).detach().cpu().tolist()}")
    
    # --- 2D ---
    print("\n[2D Laplacian]")
    lap2d = Laplacian2D(channels=64, dilations=[1, 4, 16]).to(device)
    x2d = torch.randn(4, 64, 32, 32, device=device)  # B=4, C=64, H=32, W=32
    y2d = lap2d(x2d)
    assert y2d.shape == x2d.shape, f"Shape mismatch: {y2d.shape}"
    print(f"  Input:  {tuple(x2d.shape)}")
    print(f"  Output: {tuple(y2d.shape)} ✓")
    
    # --- 3D (video) ---
    print("\n[Spatio-Temporal Laplacian]")
    lap3d = LaplacianSpatioTemporal(channels=32, causal_time=True).to(device)
    x3d = torch.randn(2, 32, 8, 16, 16, device=device)  # B=2, C=32, T=8, H=16, W=16
    y3d = lap3d(x3d)
    assert y3d.shape == x3d.shape, f"Shape mismatch: {y3d.shape}"
    print(f"  Input:  {tuple(x3d.shape)}")
    print(f"  Output: {tuple(y3d.shape)} ✓")
    
    # --- Memory scaling test (the key claim) ---
    print("\n[Memory Scaling — O(N) claim]")
    print("  Measuring VRAM vs sequence length...")
    if torch.cuda.is_available():
        sizes = [32, 64, 128, 256]
        for s in sizes:
            torch.cuda.reset_peak_memory_stats()
            x = torch.randn(1, 64, s, s, device=device)
            y = lap2d(x)
            mem = torch.cuda.max_memory_allocated() / 1e6
            n = s * s
            print(f"  H=W={s:3d} | N={n:6d} pixels | VRAM: {mem:.1f} MB")
    else:
        print("  (CUDA not available — run on GPU for memory stats)")
    
    print("\n✅ All checks passed.")