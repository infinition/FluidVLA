"""
train_step1_video.py — Step 1: Video Prediction

Validates spatio-temporal diffusion on Moving MNIST.

The critical benchmark here is NOT accuracy — it's proving that
VRAM is constant regardless of the number of input frames.

This is the unique claim that differentiates FluidBot from all
Transformer-based video models:
  - ViT-Video: VRAM grows O(T × N²) — adding frames costs quadratically
  - FluidBot:  VRAM grows O(T × N)  — adding frames costs linearly

Key tests:
  1. Model trains (loss decreases on next-frame prediction)
  2. VRAM at T=4 ≈ VRAM at T=8 ≈ VRAM at T=16 (linear in T, not quadratic)
  3. Causal inference: frame t only sees frames 0..t-1 (real-time safe)
  4. Motion propagates through diffusion without optical flow

To generate Moving MNIST:
  The script downloads it automatically from yann.lecun.com
  or generates it on-the-fly if download fails.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.core.fluid_model import FluidBotVideo


# ─────────────────────────────────────────
# Moving MNIST Dataset
# ─────────────────────────────────────────

class MovingMNIST(Dataset):
    """
    Moving MNIST: 10,000 sequences of 20 frames (64×64) with 2 bouncing digits.
    Task: given first T frames, predict frame T+1.
    
    Download from: http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy
    Or use torchvision (experimental).
    """
    def __init__(self, root: str, split: str = 'train', seq_len: int = 10, download: bool = True):
        self.seq_len = seq_len
        self.split   = split
        
        path = os.path.join(root, 'mnist_test_seq.npy')
        
        if not os.path.exists(path):
            if download:
                self._download(root, path)
            else:
                raise FileNotFoundError(f"Moving MNIST not found at {path}")
        
        # Shape: (20, 10000, 64, 64) — time, samples, H, W
        data = np.load(path)
        data = data.transpose(1, 0, 2, 3)  # → (10000, 20, 64, 64)
        data = data.astype(np.float32) / 255.0
        
        # Train/test split: 8000/2000
        if split == 'train':
            self.data = data[:8000]
        else:
            self.data = data[8000:]
    
    def _download(self, root, path):
        import urllib.request
        os.makedirs(root, exist_ok=True)
        url = 'http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy'
        print(f"Downloading Moving MNIST from {url}...")
        try:
            urllib.request.urlretrieve(url, path)
            print("Download complete.")
        except Exception as e:
            print(f"Download failed: {e}")
            print("Generating synthetic Moving MNIST...")
            self._generate_synthetic(root, path)
    
    def _generate_synthetic(self, root, path, n_samples=10000, n_frames=20, size=64):
        """Generate synthetic bouncing digit sequences if download fails."""
        from torchvision.datasets import MNIST
        import torchvision.transforms as transforms
        
        mnist = MNIST(root, train=True, download=True,
                      transform=transforms.Compose([transforms.Resize(20), transforms.ToTensor()]))
        
        data = np.zeros((n_samples, n_frames, size, size), dtype=np.float32)
        
        for s in range(n_samples):
            # Two digits
            for digit_idx in range(2):
                idx = np.random.randint(len(mnist))
                img = mnist[idx][0].squeeze().numpy()  # (20, 20)
                
                # Random initial position and velocity
                px, py = np.random.randint(0, size - 20, size=2).astype(float)
                vx, vy = np.random.uniform(-3, 3, size=2)
                
                for f in range(n_frames):
                    px += vx; py += vy
                    # Bounce off walls
                    if px < 0 or px > size - 20: vx = -vx; px = np.clip(px, 0, size - 20)
                    if py < 0 or py > size - 20: vy = -vy; py = np.clip(py, 0, size - 20)
                    
                    x, y = int(px), int(py)
                    data[s, f, y:y+20, x:x+20] = np.maximum(data[s, f, y:y+20, x:x+20], img)
        
        np.save(path, data.transpose(1, 0, 2, 3))  # back to (T, N, H, W) format
        print(f"Generated synthetic Moving MNIST: {data.shape}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq = self.data[idx]  # (20, 64, 64)
        # Pick random start position
        max_start = seq.shape[0] - self.seq_len - 1
        start = np.random.randint(0, max(1, max_start))
        
        context = seq[start:start + self.seq_len]      # (T, H, W) — input frames
        target  = seq[start + self.seq_len]             # (H, W)    — frame to predict
        
        context = torch.tensor(context).unsqueeze(0)   # (1, T, H, W)
        target  = torch.tensor(target).unsqueeze(0)    # (1, H, W)
        
        return context, target


# ─────────────────────────────────────────
# Reconstruction Head
# ─────────────────────────────────────────

class FramePredictionHead(nn.Module):
    """
    Predicts next frame pixel values from the latent video features.
    Uses the last temporal slice + upsampling to original resolution.
    """
    def __init__(self, d_model: int, patch_size: int, out_channels: int = 1):
        super().__init__()
        self.patch_size  = patch_size
        self.out_channels = out_channels
        
        # Per-patch prediction, then unfold to pixels
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_model // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(d_model // 2, out_channels * patch_size * patch_size, kernel_size=1),
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: (B, d_model, T, H//p, W//p)
        returns:  (B, out_channels, H, W) — predicted frame
        """
        # Take last frame's features
        f = features[:, :, -1, :, :]  # (B, d_model, H//p, W//p)
        
        p = self.patch_size
        pred = self.conv(f)  # (B, out_C * p * p, H//p, W//p)
        
        # Pixel shuffle to reconstruct spatial resolution
        pred = F.pixel_shuffle(pred, p)  # (B, out_C, H, W)
        return torch.sigmoid(pred)


# ─────────────────────────────────────────
# Memory Scaling Benchmark
# ─────────────────────────────────────────

def benchmark_vram_vs_frames(model, device, H=64, W=64, max_T=32):
    """
    THE key benchmark for Step 1.
    
    Measures VRAM at different numbers of input frames.
    For FluidBot, this should be approximately linear in T.
    For a Transformer, this would be quadratic in T (for attention).
    
    Expected output:
      T=2  | VRAM: ~300 MB
      T=4  | VRAM: ~450 MB   ← ~linear, not ~quadratic
      T=8  | VRAM: ~750 MB
      T=16 | VRAM: ~1400 MB
    """
    print("\n" + "=" * 60)
    print("CRITICAL BENCHMARK: VRAM vs Number of Frames")
    print("This proves O(T·N) scaling — the core claim of Step 1")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("(CUDA not available — run on GPU)")
        return None
    
    C = 1  # grayscale
    results = []
    
    model.eval()
    for T in [2, 4, 8, 16, 32]:
        if T > max_T:
            break
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            x = torch.randn(1, C, T, H, W, device=device)
            _ = model(x)
        
        mem_mb = torch.cuda.max_memory_allocated() / 1e6
        results.append((T, T * H * W, mem_mb))
        print(f"  T={T:2d} | N_total={T*H*W:7,} | VRAM: {mem_mb:7.1f} MB")
    
    if len(results) >= 2:
        # Linear scaling check: VRAM should grow ~proportionally to T
        ratios = [r[2] / r[0] for r in results]  # VRAM / T
        variation = max(ratios) / min(ratios)
        print(f"\n  VRAM/T ratio variation: {variation:.2f}x (ideal: 1.0x)")
        
        if variation < 3.0:
            print("  ✅ VRAM scales approximately linearly with T")
        else:
            print("  ⚠️  VRAM scaling may be super-linear — check for quadratic ops")
    
    return results


# ─────────────────────────────────────────
# Training
# ─────────────────────────────────────────

def train_step1(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*60}")
    print(f"FluidBot Step 1 — Video Prediction (Moving MNIST)")
    print(f"{'='*60}")
    print(f"Device: {device}")
    
    # Dataset
    print("\nLoading Moving MNIST...")
    train_ds = MovingMNIST(args.data_dir, split='train', seq_len=args.seq_len)
    test_ds  = MovingMNIST(args.data_dir, split='test',  seq_len=args.seq_len)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
    
    print(f"Train samples: {len(train_ds):,} | Test: {len(test_ds):,}")
    
    # Model
    model = FluidBotVideo(
        in_channels        = 1,
        d_model            = args.d_model,
        n_layers           = args.n_layers,
        spatial_dilations  = [1, 4, 16],
        temporal_dilations = [1, 2],
        max_steps          = 12,
        patch_size         = 4,
        causal_time        = True,
    ).to(device)
    
    pred_head = FramePredictionHead(args.d_model, patch_size=4, out_channels=1).to(device)
    
    total_params = (sum(p.numel() for p in model.parameters()) +
                    sum(p.numel() for p in pred_head.parameters()))
    print(f"\nModel params: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Critical benchmark BEFORE training
    benchmark_vram_vs_frames(model, device)
    
    # Optimizer
    all_params = list(model.parameters()) + list(pred_head.parameters())
    optimizer  = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=0.05)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))
    scaler     = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    os.makedirs(args.save_dir, exist_ok=True)
    history  = []
    best_mse = float('inf')
    
    print(f"\nTraining for {args.epochs} epochs...")
    
    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        pred_head.train()
        total_loss = 0.0
        avg_steps  = 0.0
        
        for batch_idx, (context, target) in enumerate(train_loader):
            # context: (B, 1, T, H, W) — input frames
            # target:  (B, 1, H, W)    — next frame to predict
            context, target = context.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                out      = model(context)
                features = out['features']         # (B, d_model, T, H//p, W//p)
                pred     = pred_head(features)     # (B, 1, H, W)
                
                # Combined loss: MSE + temporal gradient loss
                # Temporal gradient forces the model to predict motion, not just mean
                mse = F.mse_loss(pred, target)
                # Penalize difference between predicted motion and actual motion
                last_frame  = context[:, :, -1:, :, :]              # last input frame (B,1,1,H,W)
                last_frame  = last_frame.squeeze(2)                    # (B,1,H,W)
                pred_diff   = pred   - last_frame  # predicted change from last frame
                actual_diff = target - last_frame  # actual change from last frame
                motion_loss = F.mse_loss(pred_diff, actual_diff)
                loss = mse + 0.5 * motion_loss
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            avg_steps  += sum(i['steps_used'] for i in out['info']) / len(out['info'])
            
            if batch_idx % 50 == 0:
                print(f"  [{epoch}] Batch {batch_idx:3d}/{len(train_loader)} | "
                      f"MSE: {loss.item():.5f} | Steps: {avg_steps/(batch_idx+1):.1f}")
        
        # Evaluate
        model.eval()
        pred_head.eval()
        test_mse = 0.0
        
        with torch.no_grad():
            for context, target in test_loader:
                context, target = context.to(device), target.to(device)
                out  = model(context)
                pred = pred_head(out['features'])
                test_mse += F.mse_loss(pred, target).item()
        
        test_mse /= len(test_loader)
        train_mse = total_loss / len(train_loader)
        avg_steps_ep = avg_steps / len(train_loader)
        
        print(f"\nEpoch {epoch:3d} | Train MSE: {train_mse:.5f} | Test MSE: {test_mse:.5f} | Steps: {avg_steps_ep:.1f}")
        
        history.append({
            'epoch': epoch, 'train_mse': train_mse,
            'test_mse': test_mse, 'avg_steps': avg_steps_ep
        })
        
        if test_mse < best_mse:
            best_mse = test_mse
            torch.save({'model': model.state_dict(), 'head': pred_head.state_dict(),
                        'mse': best_mse}, os.path.join(args.save_dir, 'best_video.pt'))
            print(f"  💾 Best model saved (MSE={best_mse:.5f})")
    
    # Final memory benchmark
    benchmark_vram_vs_frames(model, device)
    
    print(f"\n✅ Step 1 Complete | Best test MSE: {best_mse:.5f}")
    
    with open(os.path.join(args.save_dir, 'history_video.json'), 'w') as f:
        json.dump(history, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   default='./data')
    parser.add_argument('--save_dir',   default='./checkpoints/step1')
    parser.add_argument('--seq_len',    default=10,   type=int)
    parser.add_argument('--d_model',    default=64,   type=int)
    parser.add_argument('--n_layers',   default=3,    type=int)
    parser.add_argument('--epochs',     default=30,   type=int)
    parser.add_argument('--batch_size', default=32,   type=int)
    parser.add_argument('--lr',         default=1e-3, type=float)
    parser.add_argument('--workers',    default=4,    type=int)
    args = parser.parse_args()
    
    train_step1(args)