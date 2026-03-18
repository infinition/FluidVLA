"""
train_step1_video.py — Step 1: next-frame video prediction on Moving MNIST

What this script demonstrates:
  1. The model learns a non-trivial video task.
  2. Memory growth with the number of frames is moderate.
  3. The causal reaction-diffusion core works without attention or optical flow.
  4. Adaptive compute can be inspected explicitly through turbulence metrics.

Important correction:
  - the old "motion loss" was mathematically identical to plain MSE
  - it is replaced here by a real spatial-gradient loss
"""

import argparse
import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from fluidvla.core import FluidBotVideo


# ─────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_mean(values: List[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def make_autocast(device: torch.device, enabled: bool = True):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", enabled=enabled)
    return torch.amp.autocast(device_type="cpu", enabled=False)


# ─────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────

class MovingMNIST(Dataset):
    """
    Moving MNIST: 10k sequences, 20 frames each, grayscale 64x64.

    Task:
      input  = first T frames
      target = next frame
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        seq_len: int = 10,
        download: bool = True,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.split = split

        if self.seq_len < 1:
            raise ValueError("seq_len must be >= 1")

        path = os.path.join(root, "mnist_test_seq.npy")
        if not os.path.exists(path):
            if download:
                self._download(root, path)
            else:
                raise FileNotFoundError(f"Moving MNIST not found at {path}")

        data = np.load(path)  # (20, 10000, 64, 64)
        data = data.transpose(1, 0, 2, 3).astype(np.float32) / 255.0  # -> (10000, 20, 64, 64)

        if data.shape[1] <= self.seq_len:
            raise ValueError("Need at least seq_len + 1 frames per sequence")

        if split == "train":
            data = data[:8000]
        elif split == "test":
            data = data[8000:]
        else:
            raise ValueError("split must be 'train' or 'test'")

        if max_samples is not None:
            data = data[:max_samples]

        self.data = data

    def _download(self, root: str, path: str) -> None:
        os.makedirs(root, exist_ok=True)
        url = "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy"
        print(f"Downloading Moving MNIST from {url}...")
        try:
            urllib.request.urlretrieve(url, path)
            print("Download complete.")
        except Exception as e:
            print(f"Download failed: {e}")
            print("Falling back to synthetic Moving MNIST generation...")
            self._generate_synthetic(root, path)

    def _generate_synthetic(
        self,
        root: str,
        path: str,
        n_samples: int = 10000,
        n_frames: int = 20,
        size: int = 64,
    ) -> None:
        from torchvision.datasets import MNIST
        import torchvision.transforms as transforms

        mnist = MNIST(
            root=root,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(20),
                transforms.ToTensor(),
            ]),
        )

        data = np.zeros((n_samples, n_frames, size, size), dtype=np.float32)
        print(f"Generating synthetic Moving MNIST: {n_samples} samples x {n_frames} frames")
        t0 = time.time()

        for s in range(n_samples):
            for _ in range(2):
                idx = np.random.randint(len(mnist))
                img = mnist[idx][0].squeeze(0).numpy()

                px, py = np.random.randint(0, size - 20, size=2).astype(np.float32)
                vx, vy = np.random.uniform(-3.0, 3.0, size=2).astype(np.float32)

                # Avoid almost-static digits
                if abs(vx) < 0.5:
                    vx = 0.5 if vx >= 0 else -0.5
                if abs(vy) < 0.5:
                    vy = 0.5 if vy >= 0 else -0.5

                for f in range(n_frames):
                    px += vx
                    py += vy

                    if px < 0 or px > size - 20:
                        vx = -vx
                        px = np.clip(px, 0, size - 20)
                    if py < 0 or py > size - 20:
                        vy = -vy
                        py = np.clip(py, 0, size - 20)

                    x, y = int(px), int(py)
                    data[s, f, y:y + 20, x:x + 20] = np.maximum(
                        data[s, f, y:y + 20, x:x + 20],
                        img,
                    )

            if (s + 1) % 250 == 0 or s == 0:
                elapsed = time.time() - t0
                rate = (s + 1) / max(elapsed, 1e-6)
                eta_sec = (n_samples - (s + 1)) / max(rate, 1e-6)
                print(
                    f"  [{s+1:5d}/{n_samples}] "
                    f"{100.0*(s+1)/n_samples:5.1f}% | "
                    f"{rate:6.1f} seq/s | ETA {eta_sec/60.0:5.1f} min"
                )

        np.save(path, data.transpose(1, 0, 2, 3))
        print(f"Generated synthetic Moving MNIST: {data.shape}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.data[idx]  # (20, 64, 64)
        max_start = seq.shape[0] - self.seq_len - 1
        start = np.random.randint(0, max_start + 1) if max_start > 0 else 0

        context = seq[start:start + self.seq_len]
        target = seq[start + self.seq_len]

        context = torch.from_numpy(context).unsqueeze(0)  # (1, T, H, W)
        target = torch.from_numpy(target).unsqueeze(0)    # (1, H, W)
        return context, target


# ─────────────────────────────────────────
# Heads / baselines
# ─────────────────────────────────────────

class FramePredictionHead(nn.Module):
    """
    Reconstruct next frame from the final latent slice.

    Predict patch-grid logits, then upscale with pixel_shuffle.
    """

    def __init__(self, d_model: int, patch_size: int, out_channels: int = 1):
        super().__init__()
        self.patch_size = patch_size
        hidden = max(d_model // 2, 8)

        self.conv = nn.Sequential(
            nn.Conv2d(d_model, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, out_channels * patch_size * patch_size, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Use the last latent time slice
        f = features[:, :, -1, :, :]
        pred = self.conv(f)
        pred = F.pixel_shuffle(pred, self.patch_size)
        return torch.sigmoid(pred)


class TinyConvGRUVideoBaseline(nn.Module):
    """Tiny baseline for rough comparison, not SOTA."""

    def __init__(self, in_channels: int = 1, hidden: int = 64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.temporal = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Conv2d(hidden, max(hidden // 2, 8), kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(max(hidden // 2, 8), in_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, c, t, h, w = x.shape
        feats = []

        for i in range(t):
            f = self.enc(x[:, :, i])
            feats.append(f.mean(dim=(2, 3)))

        seq = torch.stack(feats, dim=1)
        y, _ = self.temporal(seq)
        last = y[:, -1].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
        pred = torch.sigmoid(self.head(last))
        return {"pred": pred}


# ─────────────────────────────────────────
# Losses
# ─────────────────────────────────────────

def spatial_gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Structural image loss based on spatial gradients.

    This is a real extra signal, unlike the old "motion loss"
    that collapsed to plain MSE.
    """
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    tgt_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    tgt_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, tgt_dx) + F.l1_loss(pred_dy, tgt_dy)


# ─────────────────────────────────────────
# Benchmarks
# ─────────────────────────────────────────

def benchmark_vram_and_time_vs_frames(
    model: nn.Module,
    device: torch.device,
    h: int = 64,
    w: int = 64,
    max_t: int = 32,
    frame_counts: Optional[List[int]] = None,
    label: str = "FluidBot",
) -> Optional[List[Dict[str, float]]]:
    """
    Measure peak VRAM and forward time.

    Important:
    absolute VRAM contains a large constant allocator/cache offset.
    The meaningful thing is the incremental growth, not VRAM/T.
    """
    print("\n" + "=" * 60)
    print(f"BENCHMARK: VRAM + Forward Time vs Frames [{label}]")
    print("Goal: check incremental growth as T increases")
    print("=" * 60)

    if device.type != "cuda" or not torch.cuda.is_available():
        print("(CUDA not available - benchmark skipped)")
        return None

    model.eval()
    frame_counts = frame_counts or [2, 4, 8, 16, 32]
    results: List[Dict[str, float]] = []

    for t in frame_counts:
        if t > max_t:
            break

        x = torch.randn(1, 1, t, h, w, device=device)

        # Warmup pass
        with torch.no_grad():
            _ = model(x)
        torch.cuda.synchronize()

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        starter.record()
        with torch.no_grad():
            _ = model(x)
        ender.record()
        torch.cuda.synchronize()

        mem_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)
        elapsed_ms = starter.elapsed_time(ender)
        n_total = t * h * w
        ms_per_frame = elapsed_ms / max(t, 1)

        results.append({
            "T": t,
            "N_total": n_total,
            "peak_vram_mib": mem_mib,
            "forward_ms": elapsed_ms,
            "ms_per_frame": ms_per_frame,
        })

        print(
            f"  T={t:2d} | N_total={n_total:7,} | "
            f"Peak VRAM: {mem_mib:8.2f} MiB | Forward: {elapsed_ms:7.2f} ms | "
            f"ms/frame: {ms_per_frame:6.2f}"
        )

    if len(results) >= 2:
        t0 = results[0]["T"]
        m0 = results[0]["peak_vram_mib"]

        mem_slopes = []
        for r in results[1:]:
            d_t = r["T"] - t0
            mem_slopes.append((r["peak_vram_mib"] - m0) / max(d_t, 1e-8))

        mem_var = max(mem_slopes) / max(min(mem_slopes), 1e-8) if mem_slopes else 1.0

        print(f"\n  Incremental VRAM slope variation: {mem_var:.2f}x")
        print("  Forward time is shown directly above (absolute timing is often noisy).")

        if mem_var < 2.5:
            print("  [OK] Memory growth looks reasonably consistent")
        else:
            print("  [WARN] Memory growth varies noticeably; inspect overhead or hidden scaling")

    return results


# ─────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────

def evaluate(
    model: nn.Module,
    pred_head: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    model.eval()
    pred_head.eval()

    losses = []
    step_stats = []
    final_turbs = []
    min_turbs = []

    with torch.no_grad():
        for context, target in loader:
            context = context.to(device, non_blocking=(device.type == "cuda"))
            target = target.to(device, non_blocking=(device.type == "cuda"))

            out = model(context)
            pred = pred_head(out["features"])
            losses.append(F.mse_loss(pred, target).item())

            info = out.get("info", [])
            if isinstance(info, list) and len(info) > 0:
                if "steps_used" in info[0]:
                    step_stats.append(sum(i["steps_used"] for i in info) / len(info))
                if "final_turbulence" in info[0]:
                    final_turbs.append(sum(i["final_turbulence"] for i in info) / len(info))
                if "min_turbulence" in info[0]:
                    min_turbs.append(sum(i["min_turbulence"] for i in info) / len(info))

    return (
        safe_mean(losses),
        safe_mean(step_stats),
        safe_mean(final_turbs),
        safe_mean(min_turbs),
    )


# ─────────────────────────────────────────
# Training
# ─────────────────────────────────────────

def train_step1(args) -> None:
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")

    print(f"\n{'=' * 60}")
    print("FluidBot Step 1 - Video Prediction (Moving MNIST)")
    print(f"{'=' * 60}")
    print(f"Device: {device}")

    print("\nLoading Moving MNIST...")
    train_ds = MovingMNIST(
        args.data_dir,
        split="train",
        seq_len=args.seq_len,
        download=True,
        max_samples=args.max_train_samples,
    )
    test_ds = MovingMNIST(
        args.data_dir,
        split="test",
        seq_len=args.seq_len,
        download=True,
        max_samples=args.max_test_samples,
    )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": (device.type == "cuda"),
    }
    if args.workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    print(f"Train samples: {len(train_ds):,} | Test: {len(test_ds):,}")

    model = FluidBotVideo(
        in_channels=1,
        d_model=args.d_model,
        n_layers=args.n_layers,
        spatial_dilations=(1, 4, 16),
        temporal_dilations=(1, 2),
        max_steps=args.max_steps,
        dt=args.dt,
        epsilon=args.epsilon,
        patch_size=args.patch_size,
        causal_time=True,
        use_pde=not args.no_pde,
        norm_type=args.norm_type,
        norm_every=args.norm_every,
        local_memory_hw=args.local_memory_hw,
        signed_diffusion=args.signed_diffusion,
        diffusion_scale=args.diffusion_scale,
        temporal_mode="backward_diff",
        stop_patience=args.stop_patience,
        min_steps=args.min_steps,
        stop_probe_hw=args.stop_probe_hw,
        stop_probe_t=args.stop_probe_t,
    ).to(device)

    pred_head = FramePredictionHead(
        d_model=args.d_model,
        patch_size=args.patch_size,
        out_channels=1,
    ).to(device)

    total_params = (
        sum(p.numel() for p in model.parameters()) +
        sum(p.numel() for p in pred_head.parameters())
    )
    print(f"\nModel params: {total_params:,} ({total_params / 1e6:.3f}M)")

    baseline = None
    if args.run_baseline:
        baseline = TinyConvGRUVideoBaseline(in_channels=1, hidden=args.d_model).to(device)
        baseline_params = sum(p.numel() for p in baseline.parameters())
        print(f"Baseline params: {baseline_params:,} ({baseline_params / 1e6:.3f}M)")

    pre_bench = benchmark_vram_and_time_vs_frames(
        model,
        device,
        max_t=max(32, args.seq_len),
        label="FluidBot",
    )
    if baseline is not None:
        benchmark_vram_and_time_vs_frames(
            baseline,
            device,
            max_t=max(32, args.seq_len),
            label="TinyConvGRU",
        )

    all_params = list(model.parameters()) + list(pred_head.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(args.epochs * len(train_loader), 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(args.save_dir, exist_ok=True)
    history = []
    best_test_mse = float("inf")

    print(f"\nTraining for {args.epochs} epoch(s)...")

    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.time()
        model.train()
        pred_head.train()

        train_losses = []
        train_mses = []
        train_grad_losses = []
        train_steps_used = []

        for batch_idx, (context, target) in enumerate(train_loader):
            context = context.to(device, non_blocking=use_amp)
            target = target.to(device, non_blocking=use_amp)

            optimizer.zero_grad(set_to_none=True)

            with make_autocast(device, enabled=use_amp):
                out = model(context)
                features = out["features"]
                pred = pred_head(features)

                mse = F.mse_loss(pred, target)
                grad_loss = spatial_gradient_loss(pred, target)
                loss = mse + args.grad_loss_weight * grad_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(all_params, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_losses.append(loss.item())
            train_mses.append(mse.item())
            train_grad_losses.append(grad_loss.item())

            info = out.get("info", [])
            if isinstance(info, list) and len(info) > 0 and "steps_used" in info[0]:
                mean_steps = sum(i["steps_used"] for i in info) / len(info)
                train_steps_used.append(mean_steps)

                mean_final_turb = sum(i.get("final_turbulence", 0.0) for i in info) / len(info)
                mean_min_turb = sum(i.get("min_turbulence", 0.0) for i in info) / len(info)
            else:
                mean_steps = float("nan")
                mean_final_turb = float("nan")
                mean_min_turb = float("nan")

            if batch_idx % args.log_every == 0:
                elapsed = time.time() - epoch_t0
                it_per_sec = (batch_idx + 1) / max(elapsed, 1e-6)
                remaining = len(train_loader) - (batch_idx + 1)
                eta_sec = remaining / max(it_per_sec, 1e-6)

                print(
                    f"  [{epoch}] Batch {batch_idx:4d}/{len(train_loader)} | "
                    f"Loss: {loss.item():.5f} | MSE: {mse.item():.5f} | "
                    f"Grad: {grad_loss.item():.5f} | Steps: {mean_steps:.1f} | "
                    f"Final turb: {mean_final_turb:.4f} | Min turb: {mean_min_turb:.4f} | "
                    f"ETA: {eta_sec/60.0:.1f} min"
                )

        train_loss = safe_mean(train_losses)
        train_mse = safe_mean(train_mses)
        train_grad = safe_mean(train_grad_losses)
        train_avg_steps = safe_mean(train_steps_used)

        test_mse, test_avg_steps, test_final_turb, test_min_turb = evaluate(
            model, pred_head, test_loader, device
        )
        epoch_time = time.time() - epoch_t0

        print(
            f"\nEpoch {epoch:3d} | "
            f"Train Loss: {train_loss:.5f} | Train MSE: {train_mse:.5f} | "
            f"Train Grad: {train_grad:.5f} | Test MSE: {test_mse:.5f} | "
            f"Train Steps: {train_avg_steps:.2f} | Test Steps: {test_avg_steps:.2f} | "
            f"Test Final Turb: {test_final_turb:.4f} | Test Min Turb: {test_min_turb:.4f} | "
            f"Time: {epoch_time/60.0:.1f} min"
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mse": train_mse,
            "train_grad_loss": train_grad,
            "test_mse": test_mse,
            "train_avg_steps": train_avg_steps,
            "test_avg_steps": test_avg_steps,
            "test_final_turbulence": test_final_turb,
            "test_min_turbulence": test_min_turb,
            "epoch_time_sec": epoch_time,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(record)

        if test_mse < best_test_mse:
            best_test_mse = test_mse
            torch.save(
                {
                    "model": model.state_dict(),
                    "head": pred_head.state_dict(),
                    "args": vars(args),
                    "best_test_mse": best_test_mse,
                },
                os.path.join(args.save_dir, "best_video.pt"),
            )
            print(f"  [SAVE] Best model saved (Test MSE={best_test_mse:.5f})")

        with open(os.path.join(args.save_dir, "history_video.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print("\n[Post-training benchmark - absolute VRAM includes allocator/cache effects]")
    post_bench = benchmark_vram_and_time_vs_frames(
        model,
        device,
        max_t=max(32, args.seq_len),
        label="FluidBot-post",
    )

    summary = {
        "best_test_mse": best_test_mse,
        "pre_benchmark": pre_bench,
        "post_benchmark": post_bench,
    }
    with open(os.path.join(args.save_dir, "summary_video.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[OK] Step 1 Complete | Best test MSE: {best_test_mse:.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./data/step1_video")
    parser.add_argument("--save_dir", default="./checkpoints/step1")

    parser.add_argument("--seq_len", default=10, type=int)
    parser.add_argument("--d_model", default=64, type=int)
    parser.add_argument("--n_layers", default=3, type=int)
    parser.add_argument("--patch_size", default=4, type=int)
    parser.add_argument("--max_steps", default=12, type=int)

    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=0.05, type=float)
    parser.add_argument("--grad_clip", default=1.0, type=float)
    parser.add_argument("--grad_loss_weight", default=0.10, type=float)

    parser.add_argument("--dt", default=0.1, type=float)
    parser.add_argument("--epsilon", default=0.20, type=float)
    parser.add_argument("--norm_type", default="rmsnorm")
    parser.add_argument("--norm_every", default=2, type=int)
    parser.add_argument("--local_memory_hw", default=4, type=int)

    # Early-stop calibration
    parser.add_argument("--stop_patience", default=2, type=int)
    parser.add_argument("--min_steps", default=3, type=int)
    parser.add_argument("--stop_probe_hw", default=8, type=int)
    parser.add_argument("--stop_probe_t", default=2, type=int)

    parser.add_argument("--signed_diffusion", action="store_true")
    parser.add_argument("--diffusion_scale", default=0.25, type=float)
    parser.add_argument("--no_pde", action="store_true")

    parser.add_argument("--workers", default=0, type=int)
    parser.add_argument("--log_every", default=10, type=int)
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)

    parser.add_argument("--run_baseline", action="store_true")

    args = parser.parse_args()
    train_step1(args)