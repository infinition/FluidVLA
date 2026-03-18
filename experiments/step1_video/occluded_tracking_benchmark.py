"""
occluded_tracking_benchmark.py

Fast synthetic benchmark aligned with FluidVLA's strengths:
local spatio-temporal propagation, robustness to occlusion, and linear-ish video scaling.

Task:
  - Observe T frames of a moving dot
  - A central occluder hides part of the trajectory
  - Predict the next position (x, y) normalized in [0, 1]

Run:
  python occluded_tracking_benchmark.py --epochs 10 --seq_len 8
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from fluidvla.core import FluidBotVideo


class OccludedMovingDot(Dataset):
    def __init__(self, n_samples=5000, seq_len=8, size=64, occluder_size=16):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.size = size
        self.occluder_size = occluder_size

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        T = self.seq_len
        S = self.size
        occ = self.occluder_size
        frames = np.zeros((1, T, S, S), dtype=np.float32)
        x = np.random.uniform(8, S - 8)
        y = np.random.uniform(8, S - 8)
        vx = np.random.uniform(-2.5, 2.5)
        vy = np.random.uniform(-2.5, 2.5)
        cx0 = (S - occ) // 2
        cx1 = cx0 + occ

        next_x = x
        next_y = y
        for t in range(T + 1):
            x += vx
            y += vy
            if x < 2 or x > S - 3:
                vx *= -1
                x = np.clip(x, 2, S - 3)
            if y < 2 or y > S - 3:
                vy *= -1
                y = np.clip(y, 2, S - 3)
            if t < T:
                xi, yi = int(round(x)), int(round(y))
                if not (cx0 <= xi < cx1 and cx0 <= yi < cx1):
                    frames[0, t, yi-1:yi+2, xi-1:xi+2] = 1.0
                frames[0, t, cx0:cx1, cx0:cx1] = 0.25
            else:
                next_x, next_y = x, y

        target = np.array([next_x / (S - 1), next_y / (S - 1)], dtype=np.float32)
        return torch.from_numpy(frames), torch.from_numpy(target)


class PositionHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

    def forward(self, x):
        return torch.sigmoid(self.mlp(self.pool(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--seq_len', type=int, default=8)
    ap.add_argument('--size', type=int, default=64)
    ap.add_argument('--d_model', type=int, default=96)
    ap.add_argument('--n_layers', type=int, default=3)
    ap.add_argument('--max_steps', type=int, default=8)
    ap.add_argument('--no_pde', action='store_true')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds = OccludedMovingDot(n_samples=4000, seq_len=args.seq_len, size=args.size)
    val_ds = OccludedMovingDot(n_samples=1000, seq_len=args.seq_len, size=args.size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    backbone = FluidBotVideo(
        in_channels=1,
        d_model=args.d_model,
        n_layers=args.n_layers,
        max_steps=args.max_steps,
        patch_size=4,
        causal_time=True,
        use_pde=not args.no_pde,
    ).to(device)
    head = PositionHead(args.d_model).to(device)
    params = list(backbone.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=0.01)

    for epoch in range(1, args.epochs + 1):
        backbone.train(); head.train()
        total = 0.0
        for frames, target in train_loader:
            frames, target = frames.to(device), target.to(device)
            opt.zero_grad()
            out = backbone(frames)
            pred = head(out['features'])
            loss = F.mse_loss(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            total += loss.item()

        backbone.eval(); head.eval()
        val = 0.0
        with torch.no_grad():
            for frames, target in val_loader:
                frames, target = frames.to(device), target.to(device)
                pred = head(backbone(frames)['features'])
                val += F.mse_loss(pred, target).item()
        print(f"Epoch {epoch:02d} | train={total/len(train_loader):.5f} | val={val/len(val_loader):.5f} | mode={'PDE OFF' if args.no_pde else 'PDE ON'}")


if __name__ == '__main__':
    main()
