"""
train_fluidvla_msd.py
=====================
FluidVLA PDE training on any MSD task.

Run from FluidVLA-main/ :

    python experiments/step1b_medical_msd/train_fluidvla_msd.py ^
        --task Task09_Spleen ^
        --data_dir ./data/step1b_medical_msd/Task09_Spleen ^
        --binary --epochs 5 --batch_size 1 ^
        --max_train_samples 16 --max_val_samples 4

Checkpoints saved to:
    ./checkpoints/fluidvla/<task>/best_fluidvla.pt
    ./checkpoints/fluidvla/<task>/history.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from fluidvla.core import FluidBotMedical3D
from experiments.step1b_medical_msd.msd_dataset import MSDDataset, get_task_meta


# ── Losses / metrics ──────────────────────────────────────────────────────────

def soft_dice_loss(logits, target, num_classes, eps=1e-6):
    probs = torch.softmax(logits, dim=1)
    target_1h = F.one_hot(target, num_classes).permute(0, 4, 1, 2, 3).float()
    dices = []
    for c in range(1, num_classes):
        p = probs[:, c]; t = target_1h[:, c]
        dices.append(1.0 - (2*(p*t).sum() + eps) / (p.sum() + t.sum() + eps))
    return torch.stack(dices).mean() if dices else logits.new_zeros(())


def dice_score(logits, target, num_classes, eps=1e-6):
    pred = logits.argmax(dim=1)
    dices = []
    for c in range(1, num_classes):
        inter = ((pred == c) & (target == c)).float().sum()
        denom = (pred == c).float().sum() + (target == c).float().sum()
        dices.append(((2*inter + eps) / (denom + eps)).item())
    return sum(dices) / max(len(dices), 1)


def combined_loss(logits, target, num_classes):
    return F.cross_entropy(logits, target) + soft_dice_loss(logits, target, num_classes)


# ── Evaluate ──────────────────────────────────────────────────────────────────

def evaluate(model, loader, device, num_classes):
    model.eval()
    losses, dices, steps_list = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            logits = out["logits"]
            losses.append(combined_loss(logits, y, num_classes).item())
            dices.append(dice_score(logits, y, num_classes))
            info = out.get("info", [])
            if info:
                steps_list.append(sum(i["steps_used"] for i in info) / len(info))
    n = lambda lst: sum(lst) / max(len(lst), 1)
    return n(losses), n(dices), n(steps_list) if steps_list else float("nan")


# ── Train ─────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    in_ch, n_cls_full, default_crop, desc = get_task_meta(args.data_dir)
    num_classes = 2 if args.binary else n_cls_full
    crop = (args.depth, args.height, args.width)

    print(f"\n{'='*62}")
    print(f"  FluidVLA MSD - {args.task}")
    print(f"  {desc}  |  in_ch={in_ch}  |  classes={num_classes}  |  device={device}")
    print(f"  crop={crop}  |  crop_mode={args.crop_mode}  |  fg_prob={args.foreground_prob:.2f}")
    print(f"{'='*62}")

    max_train = args.max_train_samples if args.max_train_samples and args.max_train_samples > 0 else None
    max_val = args.max_val_samples if args.max_val_samples and args.max_val_samples > 0 else None
    train_ds = MSDDataset(args.data_dir, "train", max_samples=max_train,
                          crop_shape=crop, binary=args.binary, seed=args.seed,
                          crop_mode=args.crop_mode, foreground_prob=args.foreground_prob,
                          jitter_frac=args.crop_jitter)
    val_ds   = MSDDataset(args.data_dir, "val",   max_samples=max_val,
                          crop_shape=crop, binary=args.binary, seed=args.seed,
                          crop_mode=("foreground" if args.crop_mode in ("foreground", "mixed") else "center"),
                          foreground_prob=1.0, jitter_frac=0.0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=1,               shuffle=False, num_workers=0)

    model = FluidBotMedical3D(
        in_channels=in_ch,
        n_classes=num_classes,
        d_model=args.d_model,
        n_layers=args.n_layers,
        patch_size=args.patch_size,
        dilations=args.dilations,
        max_steps=args.max_steps,
        dt=args.dt,
        epsilon=args.epsilon,
        use_pde=not args.no_pde,
        norm_type=args.norm_type,
        norm_every=args.norm_every,
        local_memory_dhw=(args.local_memory_d, args.local_memory_h, args.local_memory_w),
        signed_diffusion=args.signed_diffusion,
        diffusion_scale=args.diffusion_scale,
        stop_patience=args.stop_patience,
        min_steps=args.min_steps,
        stop_probe_dhw=(args.stop_probe_d, args.stop_probe_h, args.stop_probe_w),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,} ({n_params/1e6:.4f}M)")

    save_dir = Path(args.save_dir) / args.task
    save_dir.mkdir(parents=True, exist_ok=True)

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total = max(args.epochs * len(train_loader), 1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_dice = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tr_losses, tr_dices, tr_steps = [], [], []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            ctx = torch.amp.autocast("cuda", enabled=use_amp) if use_amp else torch.amp.autocast("cpu", enabled=False)
            with ctx:
                out = model(x)
                logits = out["logits"]
                loss = combined_loss(logits, y, num_classes)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt); scaler.update(); sched.step()
            tr_losses.append(loss.item())
            tr_dices.append(dice_score(logits, y, num_classes))
            info = out.get("info", [])
            if info:
                tr_steps.append(sum(i["steps_used"] for i in info) / len(info))

        val_loss, val_dice, val_steps = evaluate(model, val_loader, device, num_classes)
        n = lambda l: sum(l)/max(len(l),1)
        elapsed = time.time() - t0

        print(f"  Epoch {epoch:02d}/{args.epochs}  "
              f"loss={n(tr_losses):.4f}  train_dice={n(tr_dices):.4f}  "
              f"val_dice={val_dice:.4f}  steps={val_steps:.1f}  "
              f"({elapsed:.1f}s)")

        rec = {"epoch": epoch, "train_loss": n(tr_losses), "train_dice": n(tr_dices),
               "val_loss": val_loss, "val_dice": val_dice, "val_steps": val_steps}
        history.append(rec)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "best_val_dice": best_val_dice, "n_params": n_params,
                        "in_channels": in_ch, "num_classes": num_classes,
                        "crop": list(crop)},
                       save_dir / "best_fluidvla.pt")
            print(f"  [ckpt] best saved  val_dice={best_val_dice:.4f}")

        (save_dir / "history.json").write_text(json.dumps(history, indent=2))

    print(f"\n  Done - Best Val Dice: {best_val_dice:.4f}")
    print(f"  Checkpoint : {save_dir / 'best_fluidvla.pt'}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task",      required=True, help="e.g. Task09_Spleen")
    p.add_argument("--data_dir",  required=True)
    p.add_argument("--save_dir",  default="./checkpoints/fluidvla")
    p.add_argument("--binary",    action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--epochs",    type=int,   default=5)
    p.add_argument("--batch_size",type=int,   default=1)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_val_samples",   type=int, default=None)
    p.add_argument("--depth",  type=int, default=128)
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--width",  type=int, default=128)
    p.add_argument("--d_model",   type=int, default=32)
    p.add_argument("--n_layers",  type=int, default=2)
    p.add_argument("--patch_size",type=int, default=2)
    p.add_argument("--max_steps", type=int, default=6)
    p.add_argument("--dilations", nargs="+", type=int, default=[1, 2, 4])
    p.add_argument("--dt",        type=float, default=0.1)
    p.add_argument("--epsilon",   type=float, default=0.08)
    p.add_argument("--norm_type", default="rmsnorm")
    p.add_argument("--norm_every",type=int, default=2)
    p.add_argument("--local_memory_d", type=int, default=4)
    p.add_argument("--local_memory_h", type=int, default=4)
    p.add_argument("--local_memory_w", type=int, default=4)
    p.add_argument("--stop_probe_d",   type=int, default=4)
    p.add_argument("--stop_probe_h",   type=int, default=8)
    p.add_argument("--stop_probe_w",   type=int, default=8)
    p.add_argument("--stop_patience",  type=int, default=2)
    p.add_argument("--min_steps",      type=int, default=3)
    p.add_argument("--crop_mode", choices=["center", "foreground", "mixed"], default="center")
    p.add_argument("--foreground_prob", type=float, default=0.7)
    p.add_argument("--crop_jitter", type=float, default=0.10)
    p.add_argument("--signed_diffusion", action="store_true")
    p.add_argument("--diffusion_scale", type=float, default=0.08)
    p.add_argument("--no_pde",  action="store_true")
    p.add_argument("--seed",    type=int, default=42)
    args = p.parse_args()
    train(args)
