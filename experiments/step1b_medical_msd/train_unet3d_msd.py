"""
train_unet3d_msd.py
===================
3D U-Net baseline on any MSD task — mirror of train_fluidvla_msd.py.

IMPORTANT : uses the exact same MSDDataset with the same seed=42 split,
so train/val cases are identical between FluidVLA and U-Net. No bias.

Checkpoints saved to:
    ./checkpoints/unet3d/<task>/best_unet3d_tiny.pt   (features=4)
    ./checkpoints/unet3d/<task>/best_unet3d_std.pt    (features=32)
    ./checkpoints/unet3d/<task>/history_tiny.json
    ./checkpoints/unet3d/<task>/history_std.json

Run from FluidVLA-main/ :

    python experiments/step1b_medical_msd/train_unet3d_msd.py ^
        --task Task09_Spleen ^
        --data_dir ./data/step1b_medical_msd/Task09_Spleen ^
        --binary --epochs 5 --batch_size 1 ^
        --max_train_samples 16 --max_val_samples 4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from experiments.step1b_medical_msd.msd_dataset import MSDDataset, get_task_meta


# ── Architecture ──────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, features=32):
        super().__init__()
        f = features
        self.enc1 = ConvBlock(in_channels, f)
        self.enc2 = ConvBlock(f,     f*2)
        self.enc3 = ConvBlock(f*2,   f*4)
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = ConvBlock(f*4, f*8)
        self.up3  = nn.ConvTranspose3d(f*8, f*4, 2, stride=2)
        self.dec3 = ConvBlock(f*8, f*4)
        self.up2  = nn.ConvTranspose3d(f*4, f*2, 2, stride=2)
        self.dec2 = ConvBlock(f*4, f*2)
        self.up1  = nn.ConvTranspose3d(f*2, f,   2, stride=2)
        self.dec1 = ConvBlock(f*2, f)
        self.head = nn.Conv3d(f, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


# ── Loss / metric ─────────────────────────────────────────────────────────────

def soft_dice_loss(logits, target, num_classes, eps=1e-6):
    probs = F.softmax(logits, dim=1)
    target_1h = F.one_hot(target, num_classes).permute(0,4,1,2,3).float()
    dices = []
    for c in range(1, num_classes):
        p = probs[:,c]; t = target_1h[:,c]
        dices.append(1.0 - (2*(p*t).sum()+eps)/(p.sum()+t.sum()+eps))
    return torch.stack(dices).mean() if dices else logits.new_zeros(())

def combined_loss(logits, target, num_classes):
    return F.cross_entropy(logits, target) + soft_dice_loss(logits, target, num_classes)

def dice_score(logits, target, num_classes, eps=1e-6):
    pred = logits.argmax(dim=1)
    dices = []
    for c in range(1, num_classes):
        inter = ((pred==c)&(target==c)).float().sum()
        denom = (pred==c).float().sum() + (target==c).float().sum()
        dices.append(((2*inter+eps)/(denom+eps)).item())
    return sum(dices)/max(len(dices),1)


# ── Train one model ───────────────────────────────────────────────────────────

def train_one(model, train_loader, val_loader, epochs, device, num_classes,
              name, save_path):
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*62}")
    print(f"  {name}")
    print(f"  Params: {n_params:,} ({n_params/1e6:.4f}M) | save -> {save_path}")
    print(f"{'='*62}")

    model = model.to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    best_dice = -1.0
    history = []

    for epoch in range(1, epochs+1):
        model.train()
        t0 = time.perf_counter()
        tr_l, tr_d = [], []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = combined_loss(out, y, num_classes)
            loss.backward(); opt.step()
            tr_l.append(loss.item())
            tr_d.append(dice_score(out, y, num_classes))
        sched.step()

        model.eval()
        vd = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                vd += dice_score(model(x), y, num_classes)
        vd /= max(len(val_loader), 1)

        n = lambda l: sum(l)/max(len(l),1)
        if vd > best_dice:
            best_dice = vd
            torch.save({"model": model.state_dict(), "best_val_dice": best_dice,
                        "n_params": n_params}, save_path)
            print(f"  [ckpt] best saved  val_dice={best_dice:.4f}")

        elapsed = time.perf_counter() - t0
        print(f"  Epoch {epoch:02d}/{epochs}  loss={n(tr_l):.4f}  "
              f"train_dice={n(tr_d):.4f}  val_dice={vd:.4f}  ({elapsed:.1f}s)")
        history.append({"epoch": epoch, "train_loss": n(tr_l), "train_dice": n(tr_d),
                        "val_dice": vd, "best_val_dice": best_dice})

    Path(save_path).with_suffix(".json").write_text(json.dumps(history, indent=2))

    # CPU latency
    model.eval().cpu()
    sample = next(iter(val_loader))[0][:1]
    import time as _t; lats = []
    with torch.no_grad():
        for _ in range(5):
            t0 = _t.perf_counter(); model(sample)
            lats.append((_t.perf_counter()-t0)*1000)
    import numpy as np
    lat_cpu = float(np.median(lats))

    # VRAM
    vram = 0.0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        model.to(device).eval()
        with torch.no_grad(): model(next(iter(val_loader))[0][:1].to(device))
        vram = torch.cuda.max_memory_allocated()/(1024**2)

    return best_dice, lat_cpu, n_params, vram


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task",      required=True)
    p.add_argument("--data_dir",  required=True)
    p.add_argument("--save_dir",  default="./checkpoints/unet3d")
    p.add_argument("--binary",    action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--epochs",    type=int, default=5)
    p.add_argument("--batch_size",type=int, default=1)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_val_samples",   type=int, default=None)
    p.add_argument("--depth",  type=int, default=128)
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--width",  type=int, default=128)
    p.add_argument("--seed",   type=int, default=42)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_ch, n_cls_full, default_crop, desc = get_task_meta(args.data_dir)
    num_classes = 2 if args.binary else n_cls_full
    crop = (args.depth, args.height, args.width)

    print(f"\n{'='*62}")
    print(f"  UNet3D Baseline MSD - {args.task}")
    print(f"  {desc}  |  in_ch={in_ch}  |  classes={num_classes}  |  device={device}")
    print(f"  crop={crop}  |  crop_mode={args.crop_mode}  |  fg_prob={args.foreground_prob:.2f}")
    print(f"{'='*62}")

    # Same split as FluidVLA (seed=42) - guaranteed no bias
    train_ds = MSDDataset(args.data_dir, "train", max_samples=args.max_train_samples,
                          crop_shape=crop, binary=args.binary, seed=args.seed,
                          crop_mode=args.crop_mode, foreground_prob=args.foreground_prob,
                          jitter_frac=args.crop_jitter)
    val_ds   = MSDDataset(args.data_dir, "val",   max_samples=args.max_val_samples,
                          crop_shape=crop, binary=args.binary, seed=args.seed,
                          crop_mode=("foreground" if args.crop_mode in ("foreground", "mixed") else "center"),
                          foreground_prob=1.0, jitter_frac=0.0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=1,               shuffle=False, num_workers=0)

    save_dir = Path(args.save_dir) / args.task
    save_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for features, tag in [(4, "tiny"), (32, "std")]:
        model = UNet3D(in_channels=in_ch, num_classes=num_classes, features=features)
        label = f"UNet3D_{tag.capitalize()} (features={features})"
        save_path = str(save_dir / f"best_unet3d_{tag}.pt")
        best_dice, lat_cpu, n_params, vram = train_one(
            model, train_loader, val_loader, args.epochs, device,
            num_classes, label, save_path)
        results[tag] = {"best_val_dice": best_dice, "lat_cpu_ms": lat_cpu,
                        "n_params": n_params, "vram_mib": vram}

    print(f"\n{'='*62}")
    print(f"  RESULTS - {args.task}")
    print(f"{'='*62}")
    for tag, r in results.items():
        print(f"  UNet3D_{tag:<6}  dice={r['best_val_dice']:.4f}  "
              f"lat={r['lat_cpu_ms']:.0f}ms  vram={r['vram_mib']:.0f}MiB")
    (save_dir / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
