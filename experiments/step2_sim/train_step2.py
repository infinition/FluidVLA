"""
train_step2.py — Step 2: Imitation Learning on Pick & Place

Trains FluidVLA on demonstrations collected by isaac_env.py.
Works on dataset from both SyntheticPickPlace and Isaac Sim.

RTX 4070 Ti 12GB : ~2-3 hours for 50 epochs on 1000 demos
32GB RAM         : comfortable, no swap needed

Key metrics to watch:
  1. Action loss decreasing          → model learning to imitate oracle
  2. Adaptive compute < max_steps    → model stabilizing on calm scenes
  3. Latency <50ms at eval           → ready for Step 3 hardware
  4. Turbulence decreasing           → PDE converging (equilibrium loss working)

Equilibrium regularization (Option B):
  loss = action_loss + eq_weight * mean_turbulence

  This is analogous to Graves (2016) Adaptive Computation Time penalty.
  Without it, the model has no incentive to stabilize its internal PDE
  dynamics — it will always use max_steps. The penalty creates pressure
  toward equilibrium, so calm scenes genuinely converge in fewer steps.

  Default eq_weight=0.01 — mild pressure, won't hurt action quality.
  Increase to 0.05 for stronger early-stopping, decrease to 0.001 to
  prioritize action accuracy over compute savings.

Dataset format (produced by isaac_env.py):
  data/step2/
    episode_0000.npz   # frames=(steps, 3, T, H, W), proprios=(steps, 8), actions=(steps, 7), reward=(1,)
    episode_0001.npz
    ...
    metadata.json      # optional metadata

Usage:
  # Collect synthetic demos
  python experiments/step2_sim/isaac_env.py --mode synthetic --episodes 1000 --image_size 64

  # Train with equilibrium regularization
  python experiments/step2_sim/train_step2.py --dataset ./data/step2 --epochs 50 --batch_size 32 --d_model 128

  # Stronger equilibrium pressure (more aggressive early-stopping)
  python experiments/step2_sim/train_step2.py --dataset ./data/step2 --epochs 50 --eq_weight 0.05
"""

import os
import sys
import time
import json
import glob
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.core.fluid_model import FluidBotVLA


# ─────────────────────────────────────────
# Dataset — reads .npz episodes from isaac_env.py
# ─────────────────────────────────────────

class PickPlaceDataset(Dataset):
    """
    Loads demonstrations saved by isaac_env.py as individual .npz files.

    Each episode_XXXX.npz contains:
        frames  : (steps, 3, T, H, W)  float32
        proprios: (steps, 8)            float32
        actions : (steps, 7)            float32
        reward  : (1,)                  float32

    We flatten all episodes into individual (frame, proprio, action) samples.
    Optionally filter to successful episodes only (reward > 0).
    """

    def __init__(self, data_dir: str, success_only: bool = True,
                 augment: bool = True, max_episodes: Optional[int] = None):
        self.augment = augment
        data_path = Path(data_dir)

        # Find all episode files
        npz_files = sorted(glob.glob(str(data_path / "episode_*.npz")))
        assert len(npz_files) > 0, (
            f"No episode_*.npz files found in {data_path}\n"
            f"Run first: python experiments/step2_sim/isaac_env.py "
            f"--mode synthetic --episodes 1000"
        )

        if max_episodes is not None:
            npz_files = npz_files[:max_episodes]

        # Load and flatten
        self.frames_list   = []
        self.proprios_list = []
        self.actions_list  = []

        n_total    = 0
        n_success  = 0
        n_filtered = 0

        for f in npz_files:
            data = np.load(f)
            reward = float(data['reward'][0])
            n_total += 1

            if success_only and reward <= 0:
                n_filtered += 1
                continue

            n_success += 1
            frames  = data['frames']    # (steps, 3, T, H, W)
            proprios = data['proprios'] # (steps, 8)
            actions = data['actions']   # (steps, 7)

            n_steps = len(actions)
            for i in range(n_steps):
                self.frames_list.append(frames[i])    # (3, T, H, W)
                self.proprios_list.append(proprios[i]) # (8,)
                self.actions_list.append(actions[i])   # (7,)

        if success_only:
            print(f"  Episodes: {n_success} successful / {n_total} total "
                  f"({n_filtered} filtered)")
        else:
            print(f"  Episodes: {n_total} loaded")

        print(f"  Training samples: {len(self.frames_list):,}")

        # Detect image size and frame count from data
        if len(self.frames_list) > 0:
            sample = self.frames_list[0]
            print(f"  Frame shape: {sample.shape} "
                  f"(C={sample.shape[0]}, T={sample.shape[1]}, "
                  f"H={sample.shape[2]}, W={sample.shape[3]})")

    def __len__(self):
        return len(self.frames_list)

    def __getitem__(self, idx):
        frames  = torch.from_numpy(self.frames_list[idx]).float()   # (3, T, H, W)
        proprio = torch.from_numpy(self.proprios_list[idx]).float() # (8,)
        action  = torch.from_numpy(self.actions_list[idx]).float()  # (7,)

        # Data augmentation: mild color jitter
        if self.augment and torch.rand(1).item() > 0.5:
            noise = torch.randn_like(frames) * 0.02
            frames = (frames + noise).clamp(0.0, 1.0)

        return frames, proprio, action


# ─────────────────────────────────────────
# Equilibrium loss — the key addition
# ─────────────────────────────────────────

def compute_equilibrium_loss(info_list: List[dict]) -> torch.Tensor:
    """
    Compute mean DIFFERENTIABLE turbulence across all FluidLayer outputs.

    This acts as a regularizer analogous to Graves (2016) ACT ponder cost:
    it penalizes the model for NOT converging, creating gradient pressure
    toward internal equilibrium.

    Unlike the previous version (which used detached scalars and had NO
    gradient flow), this version uses diff_turbulence tensors computed
    inside FluidLayer with gradients attached. The gradient flows through:
      turbulence → u_t - u_{t-1} → LayerNorm → dt * du → reaction/diffusion weights

    This means the model actually receives a signal to learn stable
    representations that converge quickly.
    """
    if not info_list:
        return torch.tensor(0.0)

    diff_turbs = []
    for info in info_list:
        dt = info.get('diff_turbulence', None)
        if dt is not None and isinstance(dt, torch.Tensor):
            diff_turbs.append(dt)

    if not diff_turbs:
        return torch.tensor(0.0)

    return torch.stack(diff_turbs).mean()


# ─────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler, scaler, device,
                    epoch, eq_weight=0.01):
    model.train()
    total_loss  = 0.0
    total_mse   = 0.0
    total_eq    = 0.0
    avg_steps   = 0.0
    avg_turb    = 0.0

    for batch_idx, (frames, proprio, actions) in enumerate(loader):
        frames  = frames.to(device)    # (B, 3, T, H, W)
        proprio = proprio.to(device)   # (B, 8)
        actions = actions.to(device)   # (B, 7)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            out          = model(frames, proprio)
            pred_actions = out['actions']  # (B, 7)

            # ── Action imitation loss ──
            mse_loss = F.mse_loss(pred_actions, actions)
            l1_loss  = F.l1_loss(pred_actions, actions)

            # ── Equilibrium regularization ──
            # Penalizes high turbulence → model learns to stabilize
            eq_loss = compute_equilibrium_loss(out['info']).to(device)

            loss = mse_loss + 0.1 * l1_loss + eq_weight * eq_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()
        total_mse  += mse_loss.item()
        total_eq   += eq_loss.item()

        # Track PDE steps and turbulence
        if 'info' in out and out['info']:
            n_layers = len(out['info'])
            avg_steps += sum(
                i.get('steps_used', 0) for i in out['info']
            ) / max(n_layers, 1)
            avg_turb += sum(
                i.get('final_turbulence', 0.0) for i in out['info']
            ) / max(n_layers, 1)

        if batch_idx % 50 == 0:
            n = max(batch_idx + 1, 1)
            print(f"  [E{epoch}] Batch {batch_idx:4d}/{len(loader)} | "
                  f"Loss: {loss.item():.5f} | "
                  f"MSE: {mse_loss.item():.5f} | "
                  f"Turb: {eq_loss.item():.4f} | "
                  f"Steps: {avg_steps/n:.1f}/12")

    n = max(len(loader), 1)
    return {
        'loss':      total_loss / n,
        'mse':       total_mse  / n,
        'eq_loss':   total_eq   / n,
        'avg_steps': avg_steps  / n,
        'avg_turb':  avg_turb   / n,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_mse  = 0.0
    avg_steps  = 0.0
    avg_turb   = 0.0
    latencies  = []

    for frames, proprio, actions in loader:
        frames  = frames.to(device)
        proprio = proprio.to(device)
        actions = actions.to(device)

        t0  = time.perf_counter()
        out = model(frames, proprio)
        dt  = (time.perf_counter() - t0) * 1000
        latencies.append(dt / max(frames.shape[0], 1))  # per-sample

        total_mse += F.mse_loss(out['actions'], actions).item()

        if 'info' in out and out['info']:
            n_layers = len(out['info'])
            avg_steps += sum(
                i.get('steps_used', 0) for i in out['info']
            ) / max(n_layers, 1)
            avg_turb += sum(
                i.get('final_turbulence', 0.0) for i in out['info']
            ) / max(n_layers, 1)

    n = max(len(loader), 1)
    return {
        'mse':         total_mse / n,
        'avg_steps':   avg_steps / n,
        'avg_turb':    avg_turb  / n,
        'latency_ms':  float(np.mean(latencies)) if latencies else 0,
        'latency_p95': float(np.percentile(latencies, 95)) if latencies else 0,
    }


@torch.no_grad()
def benchmark_latency(model, device, image_size=64, n_frames=4, n_runs=200):
    """
    Standalone latency benchmark with synthetic input.
    Target: <50ms on RTX 4070 Ti, <50ms on Jetson Orin Nano.
    """
    model.eval()
    frames  = torch.randn(1, 3, n_frames, image_size, image_size, device=device)
    proprio = torch.randn(1, 8, device=device)

    # Warmup GPU
    for _ in range(20):
        _ = model(frames, proprio)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    latencies = []
    for _ in range(n_runs):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(frames, proprio)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    # PDE steps from last run
    pde_steps = 0
    pde_turb  = 0
    if 'info' in out and out['info']:
        n_layers = len(out['info'])
        pde_steps = sum(
            i.get('steps_used', 0) for i in out['info']
        ) / max(n_layers, 1)
        pde_turb = sum(
            i.get('final_turbulence', 0.0) for i in out['info']
        ) / max(n_layers, 1)

    mean_lat = np.mean(latencies)
    p50_lat  = np.percentile(latencies, 50)
    p95_lat  = np.percentile(latencies, 95)
    p99_lat  = np.percentile(latencies, 99)
    fps      = 1000.0 / mean_lat if mean_lat > 0 else 0

    print(f"\n{'='*55}")
    print(f"LATENCY BENCHMARK — FluidVLA Step 2")
    print(f"  Input: (1, 3, {n_frames}, {image_size}, {image_size})")
    print(f"{'='*55}")
    print(f"  Mean       : {mean_lat:.2f} ms")
    print(f"  p50        : {p50_lat:.2f} ms")
    print(f"  p95        : {p95_lat:.2f} ms")
    print(f"  p99        : {p99_lat:.2f} ms")
    print(f"  FPS        : {fps:.1f}")
    print(f"  PDE steps  : {pde_steps:.1f}/12")
    print(f"  Turbulence : {pde_turb:.4f}")

    target = 50.0
    status = "✅" if mean_lat < target else "❌"
    print(f"  {status} RTX target (<{target}ms): {mean_lat:.2f}ms")

    return {
        'mean_ms':    float(mean_lat),
        'p50_ms':     float(p50_lat),
        'p95_ms':     float(p95_lat),
        'p99_ms':     float(p99_lat),
        'fps':        float(fps),
        'pde_steps':  float(pde_steps),
        'turbulence': float(pde_turb),
    }


@torch.no_grad()
def benchmark_adaptive_compute(model, device, image_size=64, n_frames=4):
    """
    Test adaptive compute on static vs dynamic scenes.
    This is the key claim: calm scenes should use fewer PDE steps.
    """
    model.eval()
    proprio = torch.randn(1, 8, device=device)

    # Static scene: all frames identical gray
    static = torch.ones(1, 3, n_frames, image_size, image_size, device=device) * 0.5
    out_s = model(static, proprio)
    steps_s = sum(i['steps_used'] for i in out_s['info']) / len(out_s['info'])
    turb_s  = sum(i['final_turbulence'] for i in out_s['info']) / len(out_s['info'])

    # Dynamic scene: random noise (every pixel different, every frame different)
    dynamic = torch.randn(1, 3, n_frames, image_size, image_size, device=device)
    out_d = model(dynamic, proprio)
    steps_d = sum(i['steps_used'] for i in out_d['info']) / len(out_d['info'])
    turb_d  = sum(i['final_turbulence'] for i in out_d['info']) / len(out_d['info'])

    # Smooth scene: gradients but no noise
    t = torch.linspace(0, 1, image_size, device=device)
    smooth = t.view(1, 1, 1, 1, -1).expand(1, 3, n_frames, image_size, -1)
    out_m = model(smooth, proprio)
    steps_m = sum(i['steps_used'] for i in out_m['info']) / len(out_m['info'])
    turb_m  = sum(i['final_turbulence'] for i in out_m['info']) / len(out_m['info'])

    print(f"\n{'='*55}")
    print(f"ADAPTIVE COMPUTE BENCHMARK")
    print(f"{'='*55}")
    print(f"  Static scene  : {steps_s:.1f}/12 steps | turb={turb_s:.4f}")
    print(f"  Smooth scene  : {steps_m:.1f}/12 steps | turb={turb_m:.4f}")
    print(f"  Dynamic scene : {steps_d:.1f}/12 steps | turb={turb_d:.4f}")

    if steps_s < steps_d:
        saved = (1 - steps_s / steps_d) * 100
        print(f"  ✅ Adaptive compute active: {saved:.0f}% fewer steps on static")
    else:
        print(f"  ⏳ Adaptive compute not yet differentiating (needs more training)")

    return {
        'static_steps':  float(steps_s),
        'smooth_steps':  float(steps_m),
        'dynamic_steps': float(steps_d),
        'static_turb':   float(turb_s),
        'dynamic_turb':  float(turb_d),
    }


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='FluidVLA Step 2 — Imitation Learning (Pick & Place)'
    )
    parser.add_argument('--dataset',    required=True,
                        help='Path to dataset dir from isaac_env.py')
    parser.add_argument('--epochs',     default=50,    type=int)
    parser.add_argument('--batch_size', default=16,    type=int)
    parser.add_argument('--lr',         default=3e-4,  type=float)
    parser.add_argument('--d_model',    default=256,   type=int)
    parser.add_argument('--n_layers',   default=4,     type=int)
    parser.add_argument('--max_steps',  default=12,    type=int)
    parser.add_argument('--epsilon',    default=0.02,  type=float,
                        help='Turing Equilibrium threshold for early stopping')
    parser.add_argument('--eq_weight',  default=0.01,  type=float,
                        help='Equilibrium regularization weight (0=disabled, '
                             '0.01=mild, 0.05=strong)')
    parser.add_argument('--image_size', default=None,  type=int,
                        help='Image size (auto-detected from data if omitted)')
    parser.add_argument('--save_dir',   default='./checkpoints/step2')
    parser.add_argument('--benchmark',  action='store_true',
                        help='Run latency benchmark only')
    parser.add_argument('--checkpoint', default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--success_only', action='store_true', default=True,
                        help='Train only on successful episodes')
    parser.add_argument('--all_episodes', action='store_true',
                        help='Train on all episodes (overrides success_only)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Auto-detect image size from data ──
    image_size = args.image_size
    n_frames   = 4  # default

    if image_size is None:
        npz_files = sorted(glob.glob(str(Path(args.dataset) / "episode_*.npz")))
        if npz_files:
            sample = np.load(npz_files[0])
            # frames shape: (steps, 3, T, H, W)
            _, C, T, H, W = sample['frames'].shape
            image_size = H
            n_frames   = T
            print(f"Auto-detected: image_size={image_size}, n_frames={n_frames}")
        else:
            image_size = 64
            print(f"No data found, defaulting to image_size={image_size}")

    # ── Load metadata if available ──
    meta_path = Path(args.dataset) / 'metadata.json'
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"Metadata: {json.dumps(meta, indent=2)}")

    print(f"\n{'='*60}")
    print(f"FluidVLA Step 2 — Imitation Learning (Pick & Place)")
    print(f"{'='*60}")

    # ── Resume checkpoint (loaded before model init) ──
    start_epoch = 1
    ckpt = None
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"  Resumed from: {args.checkpoint} (epoch {start_epoch - 1})")
        if 'config' in ckpt:
            cfg = ckpt['config']
            args.d_model   = cfg.get('d_model', args.d_model)
            args.n_layers  = cfg.get('n_layers', args.n_layers)
            args.max_steps = cfg.get('max_steps', args.max_steps)
            args.epsilon   = cfg.get('epsilon', args.epsilon)
            if 'image_size' in cfg and image_size != cfg['image_size']:
                image_size = cfg['image_size']
            if 'n_frames' in cfg and n_frames != cfg['n_frames']:
                n_frames = cfg['n_frames']

    print(f"  Device     : {device}")
    print(f"  Dataset    : {args.dataset}")
    print(f"  Image size : {image_size}")
    print(f"  Frames     : {n_frames}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch      : {args.batch_size}")
    print(f"  LR         : {args.lr}")
    print(f"  d_model    : {args.d_model}")
    print(f"  n_layers   : {args.n_layers}")
    print(f"  max_steps  : {args.max_steps}")
    print(f"  epsilon    : {args.epsilon}")
    print(f"  eq_weight  : {args.eq_weight}")

    # ── Model ──
    model = FluidBotVLA(
        image_size  = image_size,
        in_channels = 3,
        d_model     = args.d_model,
        n_layers    = args.n_layers,
        patch_size  = min(16, image_size // 4),  # adapt patch size to image
        action_dim  = 7,
        proprio_dim = 8,
        max_steps   = args.max_steps,
        epsilon     = args.epsilon,
        n_frames    = n_frames,
    ).to(device)

    if ckpt:
        model.load_state_dict(ckpt['model'])

    p = model.count_parameters()
    print(f"  Params     : {p['total']:,} ({p['M']:.2f}M)")

    # ── Benchmark only ──
    if args.benchmark:
        benchmark_latency(model, device, image_size=image_size, n_frames=n_frames)
        benchmark_adaptive_compute(model, device, image_size=image_size, n_frames=n_frames)
        return

    # ── Dataset ──
    success_only = args.success_only and not args.all_episodes
    dataset = PickPlaceDataset(
        args.dataset,
        success_only=success_only,
        augment=True,
    )

    if len(dataset) == 0:
        print("\n❌ No training samples! Check your dataset or use --all_episodes")
        return

    # Train/val split (90/10)
    n_val   = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == 'cuda'),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == 'cuda'),
    )

    print(f"  Train : {n_train:,} samples ({len(train_loader)} batches)")
    print(f"  Val   : {n_val:,} samples")

    # ── Optimizer ──
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    total_steps = args.epochs * len(train_loader)
    scheduler   = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler      = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # Resume optimizer if available
    if args.checkpoint and 'optimizer' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer'])
            print("  Optimizer state restored")
        except Exception:
            print("  ⚠️ Could not restore optimizer state, starting fresh")

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_mse = float('inf')
    history      = []

    # ── Pre-training adaptive compute check ──
    print(f"\n{'─'*60}")
    print("Pre-training adaptive compute check...")
    benchmark_adaptive_compute(model, device, image_size=image_size, n_frames=n_frames)

    print(f"\n{'─'*60}")
    print("Training...")
    print(f"{'─'*60}")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device,
            epoch, eq_weight=args.eq_weight,
        )
        val_metrics = evaluate(model, val_loader, device)

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch:3d}/{start_epoch + args.epochs - 1} | "
              f"Loss: {train_metrics['loss']:.5f} | "
              f"MSE: {val_metrics['mse']:.5f} | "
              f"Turb: {val_metrics['avg_turb']:.4f} | "
              f"Steps(eval): {val_metrics['avg_steps']:.1f}/12 | "
              f"Lat: {val_metrics['latency_ms']:.1f}ms | "
              f"LR: {lr_now:.2e} | "
              f"{elapsed:.0f}s")

        # Adaptive compute check
        if val_metrics['avg_steps'] < args.max_steps * 0.8:
            saved = (1 - val_metrics['avg_steps'] / args.max_steps) * 100
            print(f"  ✅ Adaptive compute: {val_metrics['avg_steps']:.1f}/12 "
                  f"({saved:.0f}% compute saved)")

        # Latency check
        if val_metrics['latency_ms'] < 50.0:
            print(f"  ✅ Latency: {val_metrics['latency_ms']:.1f}ms < 50ms")

        # History
        record = {
            'epoch':          epoch,
            'train_loss':     train_metrics['loss'],
            'train_mse':      train_metrics['mse'],
            'train_eq_loss':  train_metrics['eq_loss'],
            'train_steps':    train_metrics['avg_steps'],
            'train_turb':     train_metrics['avg_turb'],
            'val_mse':        val_metrics['mse'],
            'val_steps':      val_metrics['avg_steps'],
            'val_turb':       val_metrics['avg_turb'],
            'val_latency_ms': val_metrics['latency_ms'],
            'lr':             lr_now,
        }
        history.append(record)

        # Save best
        if val_metrics['mse'] < best_val_mse:
            best_val_mse = val_metrics['mse']
            save_path = os.path.join(args.save_dir, 'best.pt')
            torch.save({
                'epoch':     epoch,
                'model':     model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_mse':   best_val_mse,
                'config': {
                    'image_size':  image_size,
                    'n_frames':    n_frames,
                    'd_model':     args.d_model,
                    'n_layers':    args.n_layers,
                    'max_steps':   args.max_steps,
                    'epsilon':     args.epsilon,
                    'eq_weight':   args.eq_weight,
                    'action_dim':  7,
                    'proprio_dim': 8,
                },
            }, save_path)
            print(f"  💾 Best model saved (val_mse={best_val_mse:.5f})")

        # Periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            ckpt_path = os.path.join(args.save_dir, f'epoch_{epoch:03d}.pt')
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_mse': val_metrics['mse'],
            }, ckpt_path)

    # ── Final benchmarks ──
    print(f"\n{'='*60}")
    print("FINAL BENCHMARKS")
    print(f"{'='*60}")
    bench = benchmark_latency(model, device, image_size=image_size, n_frames=n_frames)
    adapt = benchmark_adaptive_compute(model, device, image_size=image_size, n_frames=n_frames)

    # ── Save history ──
    history_path = os.path.join(args.save_dir, 'history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Best val MSE      : {best_val_mse:.5f}")
    print(f"  Final latency     : {bench['mean_ms']:.2f}ms ({bench['fps']:.0f} FPS)")
    print(f"  PDE steps (rand)  : {bench['pde_steps']:.1f}/12")
    print(f"  PDE steps (static): {adapt['static_steps']:.1f}/12")
    print(f"  PDE steps (dynamic): {adapt['dynamic_steps']:.1f}/12")
    print(f"  Turbulence        : {bench['turbulence']:.4f}")
    print(f"  History           : {history_path}")
    print(f"  Best model        : {args.save_dir}/best.pt")

    # Key claim validation
    if adapt['static_steps'] < adapt['dynamic_steps']:
        ratio = adapt['dynamic_steps'] / max(adapt['static_steps'], 1)
        print(f"\n  🎯 ADAPTIVE COMPUTE VALIDATED: "
              f"static={adapt['static_steps']:.1f} vs dynamic={adapt['dynamic_steps']:.1f} "
              f"({ratio:.1f}x ratio)")
    else:
        print(f"\n  ⏳ Adaptive compute not yet visible — try more epochs or higher eq_weight")

    print(f"\nNext steps:")
    print(f"  Evaluate: python experiments/step2_sim/isaac_env.py "
          f"--mode eval --checkpoint {args.save_dir}/best.pt")
    print(f"  Step 3:   python experiments/step3_lerobot/lerobot_inference.py "
          f"--mode benchmark")


if __name__ == '__main__':
    main()