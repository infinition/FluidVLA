"""Train FluidVLA on real SO-101 episodes converted from LeRobot.

V2 changes vs original:
  1. Loads norm_stats.json for normalized targets (converter already normalizes).
  2. Supports action chunking: predicts chunk_size future actions per step.
  3. Saves V2 config in checkpoint (spatial_pool_size, chunk_size, delta_actions).
  4. Improved loss: weighted MSE per joint is no longer needed (data is pre-normalized).
  5. Optional cosine-similarity loss to encourage directional correctness.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from fluidvla.core import FluidBotVLA


# ---------------------------------------------------------------------------
# Dataset — now supports action chunking
# ---------------------------------------------------------------------------

class RealRobotEpisodeDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        success_only: bool = False,
        augment: bool = True,
        max_episodes: Optional[int] = None,
        chunk_size: int = 1,
    ):
        self.augment = augment
        self.chunk_size = chunk_size
        data_path = Path(data_dir)
        self.frames_list: list[np.ndarray] = []
        self.proprios_list: list[np.ndarray] = []
        self.actions_list: list[np.ndarray] = []

        npz_files = sorted(glob.glob(str(data_path / "episode_*.npz")))
        if not npz_files:
            raise FileNotFoundError(f"No episode_*.npz files found in {data_path}")
        if max_episodes is not None:
            npz_files = npz_files[:max_episodes]

        n_total = 0
        n_kept = 0
        for npz_path in npz_files:
            data = np.load(npz_path)
            reward = float(np.asarray(data["reward"]).reshape(-1)[0])
            n_total += 1
            if success_only and reward <= 0:
                continue
            n_kept += 1

            frames = np.asarray(data["frames"], dtype=np.float32)
            proprios = np.asarray(data["proprios"], dtype=np.float32)
            actions = np.asarray(data["actions"], dtype=np.float32)

            ep_len = actions.shape[0]

            if chunk_size <= 1:
                # Single-step: one target per frame (V1 behavior)
                for idx in range(ep_len):
                    self.frames_list.append(frames[idx])
                    self.proprios_list.append(proprios[idx])
                    self.actions_list.append(actions[idx])
            else:
                # Chunked: target is a window of chunk_size future actions
                for idx in range(ep_len):
                    chunk_actions = []
                    for k in range(chunk_size):
                        future_idx = min(idx + k, ep_len - 1)
                        chunk_actions.append(actions[future_idx])
                    self.frames_list.append(frames[idx])
                    self.proprios_list.append(proprios[idx])
                    self.actions_list.append(np.stack(chunk_actions, axis=0))

        print(f"  Episodes : {n_kept}/{n_total} | Samples : {len(self.frames_list):,}")
        if self.frames_list:
            sample = self.frames_list[0]
            print(
                f"  Frame shape   : {sample.shape} "
                f"(C={sample.shape[0]}, T={sample.shape[1]}, "
                f"H={sample.shape[2]}, W={sample.shape[3]})"
            )
            asample = self.actions_list[0]
            print(f"  Action shape  : {asample.shape}")

    def __len__(self) -> int:
        return len(self.frames_list)

    def __getitem__(self, idx: int):
        frames = torch.from_numpy(self.frames_list[idx]).float()
        proprio = torch.from_numpy(self.proprios_list[idx]).float()
        action = torch.from_numpy(self.actions_list[idx]).float()

        if self.augment and torch.rand(1).item() > 0.5:
            frames = (frames + torch.randn_like(frames) * 0.01).clamp(0.0, 1.0)
        return frames, proprio, action


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def compute_equilibrium_loss(info_list) -> torch.Tensor:
    diff_turbs = [
        item["diff_turbulence"]
        for item in info_list
        if "diff_turbulence" in item and isinstance(item["diff_turbulence"], torch.Tensor)
    ]
    if not diff_turbs:
        return torch.tensor(0.0)
    return torch.stack(diff_turbs).mean()


def cosine_direction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Penalize wrong direction of movement — critical for delta-action training.

    Returns 1 - cos_sim averaged over the batch, so 0 = perfect alignment.
    """
    cos = F.cosine_similarity(pred, target, dim=-1)
    return (1.0 - cos).mean()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scaler, device, epoch, eq_weight, use_cosine_loss):
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_eq = 0.0
    avg_steps = 0.0

    for batch_idx, (frames, proprio, actions) in enumerate(loader):
        frames = frames.to(device)
        proprio = proprio.to(device)
        actions = actions.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out = model(frames, proprio)
            pred = out["actions"]

            # Handle chunked vs single-step
            if pred.ndim == 2 and actions.ndim == 2:
                mse = F.mse_loss(pred, actions)
                l1 = F.l1_loss(pred, actions)
            elif pred.ndim == 3 and actions.ndim == 3:
                mse = F.mse_loss(pred, actions)
                l1 = F.l1_loss(pred, actions)
            else:
                # Shape mismatch — squeeze to match
                mse = F.mse_loss(pred.reshape_as(actions), actions)
                l1 = F.l1_loss(pred.reshape_as(actions), actions)

            eq_loss = compute_equilibrium_loss(out["info"]).to(device)
            loss = mse + 0.1 * l1 + eq_weight * eq_loss

            # Cosine direction loss — helps especially with delta-actions
            if use_cosine_loss:
                flat_pred = pred.reshape(pred.shape[0], -1)
                flat_target = actions.reshape(actions.shape[0], -1)
                cos_loss = cosine_direction_loss(flat_pred, flat_target)
                loss = loss + 0.2 * cos_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_mse += mse.item()
        total_eq += eq_loss.item()
        if out["info"]:
            avg_steps += sum(item.get("steps_used", 0) for item in out["info"]) / len(out["info"])

        if batch_idx % 50 == 0:
            n = max(batch_idx + 1, 1)
            print(
                f"  [E{epoch}] {batch_idx:4d}/{len(loader)} | "
                f"Loss:{loss.item():.5f} MSE:{mse.item():.5f} "
                f"Steps:{avg_steps / n:.1f}"
            )

    n = max(len(loader), 1)
    return {
        "loss": total_loss / n,
        "mse": total_mse / n,
        "eq": total_eq / n,
        "steps": avg_steps / n,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_mse = 0.0
    total_l1 = 0.0
    avg_steps = 0.0
    avg_turb = 0.0
    latencies = []

    for frames, proprio, actions in loader:
        frames = frames.to(device)
        proprio = proprio.to(device)
        actions = actions.to(device)

        t0 = time.perf_counter()
        out = model(frames, proprio)
        latencies.append((time.perf_counter() - t0) * 1000 / max(frames.shape[0], 1))

        pred = out["actions"]
        if pred.shape != actions.shape:
            pred = pred.reshape_as(actions)
        total_mse += F.mse_loss(pred, actions).item()
        total_l1 += F.l1_loss(pred, actions).item()
        if out["info"]:
            avg_steps += sum(item.get("steps_used", 0) for item in out["info"]) / len(out["info"])
            avg_turb += sum(item.get("final_turbulence", 0.0) for item in out["info"]) / len(out["info"])

    n = max(len(loader), 1)
    return {
        "mse": total_mse / n,
        "l1": total_l1 / n,
        "steps": avg_steps / n,
        "turbulence": avg_turb / n,
        "latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "latency_p95_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
    }


@torch.no_grad()
def benchmark_latency(model, device, image_size, n_frames, proprio_dim, runs=200):
    model.eval()
    frames = torch.randn(1, 3, n_frames, image_size, image_size, device=device)
    proprio = torch.randn(1, proprio_dim, device=device)

    for _ in range(20):
        _ = model(frames, proprio)
    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies = []
    out = None
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(frames, proprio)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    avg_steps = 0.0
    avg_turb = 0.0
    if out and out["info"]:
        avg_steps = sum(item.get("steps_used", 0) for item in out["info"]) / len(out["info"])
        avg_turb = sum(item.get("final_turbulence", 0.0) for item in out["info"]) / len(out["info"])

    return {
        "mean_ms": float(np.mean(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "fps": float(1000.0 / np.mean(latencies)) if latencies else 0.0,
        "avg_steps": float(avg_steps),
        "avg_turbulence": float(avg_turb),
    }


@torch.no_grad()
def benchmark_adaptive_compute(model, device, image_size, n_frames, proprio_dim):
    model.eval()
    frames = torch.randn(1, 3, n_frames, image_size, image_size, device=device)
    proprio = torch.randn(1, proprio_dim, device=device)

    # Static input — should converge fast
    static_out = model(frames, proprio)
    static_steps = sum(i.get("steps_used", 0) for i in static_out["info"]) / max(len(static_out["info"]), 1)

    # Dynamic input — changing frames should need more steps
    dynamic_steps_list = []
    for _ in range(20):
        frames = torch.randn(1, 3, n_frames, image_size, image_size, device=device)
        out = model(frames, proprio)
        dynamic_steps_list.append(
            sum(i.get("steps_used", 0) for i in out["info"]) / max(len(out["info"]), 1)
        )

    return {
        "static_steps": float(static_steps),
        "dynamic_steps": float(np.mean(dynamic_steps_list)),
    }


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train FluidVLA Step 3 (V2)")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--save_dir", default="checkpoints/step3_lerobot/v2")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=12)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--eq_weight", type=float, default=0.01)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--override_max_steps", default=None, type=int)
    parser.add_argument("--override_epsilon", default=None, type=float)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--all_episodes", action="store_true")
    parser.add_argument("--max_episodes", default=None, type=int)
    # ── V2 additions ──
    parser.add_argument(
        "--spatial_pool_size", type=int, default=4,
        help="Spatial grid size kept after pooling. 1 = V1, 4 = recommended V2.",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=1,
        help="Number of future actions predicted per step. 1 = V1, 4-8 recommended.",
    )
    parser.add_argument(
        "--cosine_loss", action="store_true", default=False,
        help="Add cosine-similarity direction loss (recommended for delta-actions).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    npz_files = sorted(glob.glob(str(Path(args.dataset) / "episode_*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"No episode_*.npz files found in {args.dataset}")

    sample = np.load(npz_files[0])
    _, channels, n_frames, image_size, _ = sample["frames"].shape
    action_dim = int(sample["actions"].shape[-1])
    proprio_dim = int(sample["proprios"].shape[-1])

    # Load norm stats if available
    norm_stats_path = Path(args.dataset) / "norm_stats.json"
    norm_stats = None
    if norm_stats_path.exists():
        with norm_stats_path.open("r", encoding="utf-8") as f:
            norm_stats = json.load(f)
        print(f"Loaded normalization stats from {norm_stats_path}")
        print(f"  delta_actions: {norm_stats.get('delta_actions', False)}")

    # Determine if we're using cosine loss
    use_cosine_loss = args.cosine_loss
    if norm_stats and norm_stats.get("delta_actions", False) and not args.cosine_loss:
        print("  [INFO] Delta-action mode detected — enabling cosine direction loss")
        use_cosine_loss = True

    ckpt = None
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})
        args.d_model = cfg.get("d_model", args.d_model)
        args.n_layers = cfg.get("n_layers", args.n_layers)
        args.max_steps = cfg.get("max_steps", args.max_steps)
        args.epsilon = cfg.get("epsilon", args.epsilon)
        args.spatial_pool_size = cfg.get("spatial_pool_size", args.spatial_pool_size)
        args.chunk_size = cfg.get("chunk_size", args.chunk_size)
        image_size = int(cfg.get("image_size", image_size))
        n_frames = int(cfg.get("n_frames", n_frames))
        action_dim = int(cfg.get("action_dim", action_dim))
        proprio_dim = int(cfg.get("proprio_dim", proprio_dim))
        if args.override_max_steps is not None:
            args.max_steps = args.override_max_steps
        if args.override_epsilon is not None:
            args.epsilon = args.override_epsilon

    meta_path = Path(args.dataset) / "metadata.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        print(json.dumps(metadata.get("source", {}), indent=2))

    model = FluidBotVLA(
        image_size=image_size,
        in_channels=channels,
        d_model=args.d_model,
        n_layers=args.n_layers,
        patch_size=min(16, image_size // 4),
        action_dim=action_dim,
        proprio_dim=proprio_dim,
        max_steps=args.max_steps,
        epsilon=args.epsilon,
        n_frames=n_frames,
        spatial_pool_size=args.spatial_pool_size,
        chunk_size=args.chunk_size,
    ).to(device)

    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        print(f"Loaded checkpoint: {args.checkpoint}")

    print("=" * 60)
    print("FluidVLA Step 3 - SO-101 from LeRobot (V2)")
    print(f"  device            : {device}")
    print(f"  dataset           : {args.dataset}")
    print(f"  image_size        : {image_size}")
    print(f"  n_frames          : {n_frames}")
    print(f"  action_dim        : {action_dim}")
    print(f"  proprio_dim       : {proprio_dim}")
    print(f"  params            : {model.count_parameters()['M']:.2f}M")
    print(f"  max_steps         : {args.max_steps}")
    print(f"  epsilon           : {args.epsilon}")
    print(f"  spatial_pool_size : {args.spatial_pool_size}")
    print(f"  chunk_size        : {args.chunk_size}")
    print(f"  cosine_loss       : {use_cosine_loss}")
    if norm_stats:
        print(f"  delta_actions     : {norm_stats.get('delta_actions', False)}")
    print("=" * 60)

    if args.benchmark:
        latency = benchmark_latency(model, device, image_size, n_frames, proprio_dim)
        adaptive = benchmark_adaptive_compute(model, device, image_size, n_frames, proprio_dim)
        print(json.dumps({"latency": latency, "adaptive_compute": adaptive}, indent=2))
        return

    if args.eval_only:
        dataset = RealRobotEpisodeDataset(
            args.dataset,
            success_only=not args.all_episodes,
            augment=False,
            max_episodes=args.max_episodes,
            chunk_size=args.chunk_size,
        )
        if len(dataset) == 0:
            raise RuntimeError("Converted dataset is empty")

        n_val = max(1, int(len(dataset) * 0.1))
        n_train = len(dataset) - n_val
        _, val_ds = torch.utils.data.random_split(
            dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        val_metrics = evaluate(model, val_loader, device)
        latency = benchmark_latency(model, device, image_size, n_frames, proprio_dim)
        adaptive = benchmark_adaptive_compute(model, device, image_size, n_frames, proprio_dim)
        print(json.dumps({
            "validation": val_metrics,
            "latency": latency,
            "adaptive_compute": adaptive,
        }, indent=2))
        return

    dataset = RealRobotEpisodeDataset(
        args.dataset,
        success_only=not args.all_episodes,
        max_episodes=args.max_episodes,
        chunk_size=args.chunk_size,
    )
    if len(dataset) == 0:
        raise RuntimeError("Converted dataset is empty")

    n_val = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    os.makedirs(args.save_dir, exist_ok=True)
    best_mse = float("inf")
    history = []

    # Build config dict for checkpoint
    save_config = {
        "image_size": image_size,
        "n_frames": n_frames,
        "action_dim": action_dim,
        "proprio_dim": proprio_dim,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "max_steps": args.max_steps,
        "epsilon": args.epsilon,
        "spatial_pool_size": args.spatial_pool_size,
        "chunk_size": args.chunk_size,
    }

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            epoch,
            args.eq_weight,
            use_cosine_loss,
        )
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train MSE:{train_metrics['mse']:.5f} | "
            f"Val MSE:{val_metrics['mse']:.5f} | "
            f"Val L1:{val_metrics['l1']:.5f} | "
            f"Steps:{val_metrics['steps']:.1f}/12 | "
            f"Lat:{val_metrics['latency_ms']:.2f}ms | {elapsed:.0f}s"
        )

        entry = {
            "epoch": epoch,
            **train_metrics,
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(entry)

        if val_metrics["mse"] < best_mse:
            best_mse = val_metrics["mse"]
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_mse": best_mse,
                    "config": save_config,
                    "norm_stats": norm_stats,
                },
                os.path.join(args.save_dir, "best.pt"),
            )
            print(f"  [SAVE] best.pt updated (val_mse={best_mse:.5f})")

    latency = benchmark_latency(model, device, image_size, n_frames, proprio_dim)
    adaptive = benchmark_adaptive_compute(model, device, image_size, n_frames, proprio_dim)

    with open(os.path.join(args.save_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(args.save_dir, "benchmark.json"), "w", encoding="utf-8") as f:
        json.dump({"latency": latency, "adaptive_compute": adaptive}, f, indent=2)

    print("=" * 60)
    print("TRAINING COMPLETE - Step 3 SO-101 (V2)")
    print(f"  Best val MSE : {best_mse:.5f}")
    print(f"  Mean latency : {latency['mean_ms']:.2f} ms")
    print(f"  p95 latency  : {latency['p95_ms']:.2f} ms")
    print(f"  FPS          : {latency['fps']:.1f}")
    print(
        "  Adaptive     : "
        f"static={adaptive['static_steps']:.1f} steps, "
        f"dynamic={adaptive['dynamic_steps']:.1f} steps"
    )
    print(f"  Outputs      : {args.save_dir}")


if __name__ == "__main__":
    main()
