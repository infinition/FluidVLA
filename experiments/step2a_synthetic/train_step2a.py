"""
train_step2a.py — Step 2a : Imitation Learning on synthetic data
────────────────────────────────────────────────────────────────────────
Trains FluidBotVLA on demos collected by synthetic_env.py.

DATA   → data/step2a_synthetic/
CHECKPOINTS → checkpoints/step2a/

PROVEN RESULTS (Step 2a) :
  Validation MSE : 0.01345  (epoch 48/50)
  Latence        : ~4.1ms   (~244 FPS sur RTX 4070 Ti)
  Adaptive compute : 1/12 steps, 92% compute saved (eq_weight=0.1)

IMPORTANT CLI FLAGS :
  --no_pde      Disables the Laplacian (see fluid_layer.py)
                Useful if training is unstable or for diagnostics.
                Actions are still predicted — only spatial diffusion
                is disabled. Result: pure residual MLP.

  --eq_weight   Equilibrium regularization weight (default 0.1)
                0.0  → no pressure towards early stopping
                0.01 → light pressure
                0.1  → strong pressure (paper result)

  --epsilon     Turing Equilibrium threshold (default 0.02)
                0.0  → early stopping disabled (always max_steps)
                1e9  → stops after 1 step (max speed)

EXAMPLES :
  # Collect synthetic demos (from this directory)
  python ../step2a_synthetic/synthetic_env.py --episodes 1000

  # Training standard
  python train_step2a.py --dataset ../../data/step2a_synthetic --epochs 50

  # Training without PDE (diagnostic)
  python train_step2a.py --dataset ../../data/step2a_synthetic --epochs 50 --no_pde

  # Fine-tune from step2a checkpoint to step2c
  python ../step2c_isaac_collect/train_step2c.py \
      --dataset ../../data/step2c_isaac \
      --checkpoint ../../checkpoints/step2a/best.pt
"""

import os, sys, time, json, glob, argparse
import numpy as np
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.core.fluid_model import FluidBotVLA

# Reuse constants from the synthetic env
sys.path.insert(0, str(Path(__file__).parent))
from synthetic_env import ACTION_DIM, PROPRIO_DIM, N_FRAMES

# ─── Dataset ────────────────────────────────────────────────────────────────

class PickPlaceDataset(Dataset):
    """
    Loads .npz episodes produced by synthetic_env.py or isaac_env.py.

    Expected format per episode:
        frames   : (steps, 3, T, H, W)  float16 ou float32
        proprios : (steps, 8)            float32
        actions  : (steps, 7)            float32
        reward   : (1,)                  float32
    """
    def __init__(self, data_dir: str, success_only: bool = True,
                 augment: bool = True, max_episodes: Optional[int] = None):
        self.augment = augment
        data_path = Path(data_dir)

        npz_files = sorted(glob.glob(str(data_path / "episode_*.npz")))
        assert len(npz_files) > 0, (
            f"No episodes found in {data_path}\n"
            f"Run first: python synthetic_env.py --episodes 1000"
        )
        if max_episodes:
            npz_files = npz_files[:max_episodes]

        self.frames_list   = []
        self.proprios_list = []
        self.actions_list  = []
        n_total = n_success = 0

        for f in npz_files:
            data   = np.load(f)
            reward = float(data['reward'][0])
            n_total += 1
            if success_only and reward <= 0:
                continue
            n_success += 1
            frames  = data['frames'].astype(np.float32)   # force float32
            proprios = data['proprios'].astype(np.float32)
            actions = data['actions'].astype(np.float32)
            for i in range(len(actions)):
                self.frames_list.append(frames[i])
                self.proprios_list.append(proprios[i])
                self.actions_list.append(actions[i])

        print(f"  Episodes : {n_success}/{n_total} | Samples : {len(self.frames_list):,}")

    def __len__(self): return len(self.frames_list)

    def __getitem__(self, idx):
        frames  = torch.from_numpy(self.frames_list[idx]).float()
        proprio = torch.from_numpy(self.proprios_list[idx]).float()
        action  = torch.from_numpy(self.actions_list[idx]).float()
        if self.augment and torch.rand(1).item() > 0.5:
            frames = (frames + torch.randn_like(frames) * 0.02).clamp(0, 1)
        return frames, proprio, action


# ─── Equilibrium loss ────────────────────────────────────────────────────────

def compute_equilibrium_loss(info_list) -> torch.Tensor:
    """
    Penalizes the average turbulence (differentiable).
    Forces the model to converge fast → effective early-stopping.
    Analogous to the ponder cost in Graves (2016) ACT.
    """
    diff_turbs = [
        i['diff_turbulence'] for i in info_list
        if 'diff_turbulence' in i and isinstance(i['diff_turbulence'], torch.Tensor)
    ]
    return torch.stack(diff_turbs).mean() if diff_turbs else torch.tensor(0.0)


# ─── Training loop ──────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, epoch, eq_weight):
    model.train()
    total_loss = total_mse = total_eq = avg_steps = 0.0

    for batch_idx, (frames, proprio, actions) in enumerate(loader):
        frames  = frames.to(device)
        proprio = proprio.to(device)
        actions = actions.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            out     = model(frames, proprio)
            mse     = F.mse_loss(out['actions'], actions)
            l1      = F.l1_loss(out['actions'], actions)
            eq_loss = compute_equilibrium_loss(out['info']).to(device)
            loss    = mse + 0.1 * l1 + eq_weight * eq_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler: scheduler.step()

        total_loss += loss.item()
        total_mse  += mse.item()
        total_eq   += eq_loss.item()
        if out['info']:
            avg_steps += sum(i['steps_used'] for i in out['info']) / len(out['info'])

        if batch_idx % 50 == 0:
            n = max(batch_idx + 1, 1)
            pde_tag = "PDE ON" if out['info'][0].get('pde_active', True) else "PDE OFF"
            print(f"  [E{epoch}] {batch_idx:4d}/{len(loader)} | "
                  f"Loss:{loss.item():.5f} MSE:{mse.item():.5f} "
                  f"Steps:{avg_steps/n:.1f} [{pde_tag}]")

    n = max(len(loader), 1)
    return {'loss': total_loss/n, 'mse': total_mse/n, 'eq': total_eq/n, 'steps': avg_steps/n}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_mse = avg_steps = 0.0
    latencies = []
    for frames, proprio, actions in loader:
        frames = frames.to(device); proprio = proprio.to(device); actions = actions.to(device)
        t0 = time.perf_counter()
        out = model(frames, proprio)
        latencies.append((time.perf_counter() - t0) * 1000 / frames.shape[0])
        total_mse += F.mse_loss(out['actions'], actions).item()
        if out['info']:
            avg_steps += sum(i['steps_used'] for i in out['info']) / len(out['info'])
    n = max(len(loader), 1)
    return {
        'mse': total_mse/n, 'steps': avg_steps/n,
        'latency_ms': float(np.mean(latencies)) if latencies else 0,
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='FluidVLA Step 2a — Synthetic Imitation Learning')
    parser.add_argument('--dataset',       required=True,
                        help='Path to data/step2a_synthetic/')
    parser.add_argument('--save_dir',      default='../../checkpoints/step2a')
    parser.add_argument('--epochs',        default=50,   type=int)
    parser.add_argument('--batch_size',    default=32,   type=int)
    parser.add_argument('--lr',            default=3e-4, type=float)
    parser.add_argument('--d_model',       default=128,  type=int)
    parser.add_argument('--n_layers',      default=3,    type=int)
    parser.add_argument('--max_steps',     default=12,   type=int)
    parser.add_argument('--eq_weight',     default=0.1,  type=float,
                        help='Equilibrium weight (0=disabled, 0.1=paper result)')
    parser.add_argument('--epsilon',       default=0.02, type=float,
                        help='Turing Equilibrium threshold (0=always max_steps)')
    parser.add_argument('--image_size',    default=None, type=int)
    # ── PDE ON/OFF ────────────────────────────────────────────────────────────
    parser.add_argument('--no_pde', action='store_true',
                        help='Disables the Laplacian term (diagnostic/fallback). '
                             'See fluid_layer.py for complete documentation.')
    # ─────────────────────────────────────────────────────────────────────────
    parser.add_argument('--checkpoint',    default=None)
    parser.add_argument('--all_episodes',  action='store_true')
    args = parser.parse_args()

    use_pde = not args.no_pde  # True by default

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Auto-detect image_size from data
    image_size = args.image_size
    n_frames   = N_FRAMES
    if image_size is None:
        npz_files = sorted(glob.glob(str(Path(args.dataset) / "episode_*.npz")))
        if npz_files:
            s = np.load(npz_files[0])
            _, C, T, H, W = s['frames'].shape
            image_size = H; n_frames = T
    image_size = image_size or 64

    print(f"\n{'='*60}")
    print(f"FluidVLA Step 2a — Synthetic Imitation Learning")
    print(f"  use_pde    : {use_pde}  {'(Laplacian active)' if use_pde else '(PDE DISABLED — diagnostic mode)'}")
    print(f"  image_size : {image_size}  | n_frames : {n_frames}")
    print(f"  d_model    : {args.d_model} | n_layers : {args.n_layers}")
    print(f"  eq_weight  : {args.eq_weight} | epsilon : {args.epsilon}")
    print(f"  save_dir   : {args.save_dir}")
    print(f"{'='*60}")

    model = FluidBotVLA(
        image_size=image_size, in_channels=3,
        d_model=args.d_model, n_layers=args.n_layers,
        patch_size=min(16, image_size // 4),
        action_dim=ACTION_DIM, proprio_dim=PROPRIO_DIM,
        max_steps=args.max_steps, epsilon=args.epsilon,
        n_frames=n_frames, use_pde=use_pde,
    ).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        print(f"  Checkpoint loaded: {args.checkpoint}")

    p = model.count_parameters()
    print(f"  Params : {p['total']:,} ({p['M']:.2f}M)")

    dataset = PickPlaceDataset(args.dataset, success_only=not args.all_episodes)
    assert len(dataset) > 0, "Empty dataset ! Run synthetic_env.py --episodes 1000 first."

    n_val   = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"  Train : {n_train:,} | Val : {n_val:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader)
    )
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    os.makedirs(args.save_dir, exist_ok=True)
    best_mse = float('inf')
    history  = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tm = train_one_epoch(model, train_loader, optimizer, scheduler,
                             scaler, device, epoch, args.eq_weight)
        vm = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        print(f"\nEpoch {epoch:3d}/{args.epochs} | "
              f"Train MSE:{tm['mse']:.5f} | Val MSE:{vm['mse']:.5f} | "
              f"Steps(eval):{vm['steps']:.1f}/12 | Lat:{vm['latency_ms']:.1f}ms | {elapsed:.0f}s")

        if vm['steps'] < args.max_steps * 0.8:
            saved = (1 - vm['steps'] / args.max_steps) * 100
            print(f"  ✅ Adaptive compute : {vm['steps']:.1f}/12 ({saved:.0f}% compute saved)")

        history.append({'epoch': epoch, **tm, 'val_mse': vm['mse'],
                        'val_steps': vm['steps'], 'val_lat': vm['latency_ms']})

        if vm['mse'] < best_mse:
            best_mse = vm['mse']
            torch.save({
                'epoch': epoch, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'val_mse': best_mse,
                'config': {
                    'image_size': image_size, 'n_frames': n_frames,
                    'd_model': args.d_model, 'n_layers': args.n_layers,
                    'max_steps': args.max_steps, 'epsilon': args.epsilon,
                    'action_dim': ACTION_DIM, 'proprio_dim': PROPRIO_DIM,
                    'use_pde': use_pde,
                }
            }, os.path.join(args.save_dir, 'best.pt'))
            print(f"  💾 Best model saved (val_mse={best_mse:.5f})")

    with open(os.path.join(args.save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE — Step 2a")
    print(f"  Best val MSE : {best_mse:.5f}")
    print(f"  use_pde      : {use_pde}")
    print(f"  Checkpoint   : {args.save_dir}/best.pt")
    print(f"\nNext → Step 2b : Isaac Sim camera validation")
    print(f"  python ../step2b_isaac_validate/camera_check.py --episodes 10")
    print(f"Next → Step 2c : Isaac Sim physical collection")
    print(f"  python ../step2c_isaac_collect/train_step2c.py \\")
    print(f"      --dataset ../../data/step2c_isaac \\")
    print(f"      --checkpoint {args.save_dir}/best.pt")

if __name__ == '__main__':
    main()