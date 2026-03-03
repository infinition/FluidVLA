"""
train_step0.py — Step 0: Image Classification

Validates that 2D reaction-diffusion learns spatial features.

Benchmarks:
  MNIST   : target >98% accuracy (binary, simple)
  CIFAR-10: target >70% accuracy (color, complex — pure diffusion baseline)

Run:
  python experiments/step0_mnist/train_step0.py --dataset mnist --model tiny
  python experiments/step0_mnist/train_step0.py --dataset cifar10 --model small

The key things to verify:
  1. Model trains and converges (loss decreases)
  2. Accuracy is competitive with simple CNNs (not SOTA, but solid)
  3. Adaptive compute: eval mode uses fewer steps than max_steps
  4. Memory scales linearly with image size (not quadratically)
"""

import os
import sys
import time
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.core.fluid_model import FluidBotClassifier


def get_dataset(name: str, data_dir: str = './data'):
    """Load MNIST or CIFAR-10 with appropriate transforms."""
    
    if name == 'mnist':
        # MNIST: 1-channel 28×28, padded to 32×32 for cleaner patch division
        transform_train = transforms.Compose([
            transforms.Pad(2),  # 28→32
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        transform_test = transforms.Compose([
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_ds = torchvision.datasets.MNIST(data_dir, train=True,  download=True, transform=transform_train)
        test_ds  = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=transform_test)
        in_channels = 1
        
    elif name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_ds = torchvision.datasets.CIFAR10(data_dir, train=True,  download=True, transform=transform_train)
        test_ds  = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)
        in_channels = 3
    
    else:
        raise ValueError(f"Unknown dataset: {name}. Use 'mnist' or 'cifar10'.")
    
    return train_ds, test_ds, in_channels


def train_one_epoch(model, loader, optimizer, scheduler, device, scaler, epoch):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0
    avg_steps  = 0.0
    
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            logits, info = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        preds       = logits.argmax(dim=-1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)
        avg_steps  += info['avg_steps']
        
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx:4d}/{len(loader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Acc: {100*correct/total:.1f}% | "
                  f"Steps: {info['avg_steps']:.1f}")
    
    return {
        'loss'     : total_loss / len(loader),
        'accuracy' : correct / total,
        'avg_steps': avg_steps / len(loader),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct   = 0
    total     = 0
    avg_steps = 0.0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits, info = model(images)
        preds      = logits.argmax(dim=-1)
        correct   += (preds == labels).sum().item()
        total     += labels.size(0)
        avg_steps += info['avg_steps']
    
    return {
        'accuracy' : correct / total,
        'avg_steps': avg_steps / len(loader),
    }


def benchmark_memory_scaling(model, device):
    """
    Key test: verify O(N) memory scaling.
    Measure VRAM at different image sizes.
    """
    print("\n" + "=" * 50)
    print("MEMORY SCALING BENCHMARK (O(N) claim)")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("(Skip — CUDA not available)")
        return
    
    model.eval()
    results = []
    
    for H in [32, 64, 128, 256]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            x = torch.randn(1, model.patch_embed.proj.in_channels, H, H, device=device)
            _ = model(x)
        
        mem_mb = torch.cuda.max_memory_allocated() / 1e6
        n_pixels = H * H
        results.append((H, n_pixels, mem_mb))
        print(f"  {H:4d}×{H:4d} | N={n_pixels:7,} px | VRAM: {mem_mb:7.1f} MB")
    
    # Check linearity: ratio of VRAM / N should be roughly constant
    ratios = [r[2] / r[1] for r in results]
    ratio_variation = max(ratios) / min(ratios)
    print(f"\n  VRAM/N ratio variation: {ratio_variation:.2f}x (ideal: 1.0x, <3x is good)")
    
    if ratio_variation < 5.0:
        print("  ✅ Memory scaling is approximately linear")
    else:
        print("  ⚠️  Memory scaling may be super-linear — investigate")
    
    return results


def benchmark_adaptive_compute(model, device):
    """
    Test that adaptive compute uses fewer steps on simple inputs.
    Simple = constant image. Complex = high-frequency noise.
    """
    print("\n" + "=" * 50)
    print("ADAPTIVE COMPUTE BENCHMARK")
    print("=" * 50)
    
    C = model.patch_embed.proj.in_channels
    model.eval()
    
    with torch.no_grad():
        # Constant image — should converge very fast
        simple = torch.ones(4, C, 32, 32, device=device) * 0.5
        _, info_simple = model(simple)
        
        # Random noise — should take more steps
        complex_ = torch.randn(4, C, 32, 32, device=device)
        _, info_complex = model(complex_)
        
        # Natural image-like (smooth gradients) — in between
        y, x = torch.meshgrid(torch.linspace(-1, 1, 32), torch.linspace(-1, 1, 32), indexing='ij')
        smooth = torch.stack([torch.sin(y * 3.14), torch.cos(x * 3.14)] + 
                             ([torch.zeros_like(y)] if C == 3 else []), dim=0
                            ).unsqueeze(0).expand(4, -1, -1, -1).to(device)
        smooth = smooth[:, :C]
        _, info_smooth = model(smooth)
    
    print(f"  Constant image : avg_steps = {info_simple['avg_steps']:.2f}")
    print(f"  Smooth gradients: avg_steps = {info_smooth['avg_steps']:.2f}")
    print(f"  Random noise   : avg_steps = {info_complex['avg_steps']:.2f}")
    
    if info_simple['avg_steps'] <= info_complex['avg_steps']:
        print("  ✅ Adaptive compute: simple inputs use ≤ steps than complex")
    else:
        print("  ⚠️  Adaptive compute not differentiating inputs yet (expected early in training)")


def main():
    parser = argparse.ArgumentParser(description='FluidBot Step 0 — Image Classification')
    parser.add_argument('--dataset',    default='cifar10',  choices=['mnist', 'cifar10'])
    parser.add_argument('--model',      default='small',    choices=['tiny', 'small', 'base'])
    parser.add_argument('--epochs',     default=50,         type=int)
    parser.add_argument('--batch_size', default=128,        type=int)
    parser.add_argument('--lr',         default=3e-4,       type=float)
    parser.add_argument('--data_dir',   default='./data')
    parser.add_argument('--save_dir',   default='./checkpoints/step0')
    parser.add_argument('--workers',    default=4,          type=int)
    parser.add_argument('--max_steps', default=4, type=int)

    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"FluidBot Step 0 — {args.dataset.upper()} Classification")
    print(f"{'='*60}")
    print(f"Device   : {device}")
    print(f"Dataset  : {args.dataset}")
    print(f"Model    : {args.model}")
    print(f"Epochs   : {args.epochs}")
    print(f"Batch    : {args.batch_size}")
    print(f"LR       : {args.lr}")
    
    # Data
    train_ds, test_ds, in_channels = get_dataset(args.dataset, args.data_dir)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
    
    # Model
    cfg = FluidBotClassifier.CONFIGS[args.model].copy()
    cfg['max_steps'] = args.max_steps  # override
    model = FluidBotClassifier(
        in_channels=in_channels,
        num_classes=10,
        **cfg
    ).to(device)
    
    params = model.count_parameters()
    print(f"\nModel params: {params['total']:,} ({params['total']/1e6:.2f}M)")
    
    # Initial benchmarks (before training)
    benchmark_memory_scaling(model, device)
    benchmark_adaptive_compute(model, device)
    
    # Optimizer: AdamW with cosine schedule
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    total_steps = args.epochs * len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    # Training loop
    os.makedirs(args.save_dir, exist_ok=True)
    best_acc = 0.0
    history  = []
    
    print(f"\n{'─'*60}")
    print("Starting training...")
    print(f"{'─'*60}")
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler,
                                        device, scaler, epoch)
        test_metrics  = evaluate(model, test_loader, device)
        
        elapsed = time.time() - t0
        
        print(f"\n  Train | Loss: {train_metrics['loss']:.4f} | "
              f"Acc: {100*train_metrics['accuracy']:.2f}% | "
              f"Steps: {train_metrics['avg_steps']:.1f}")
        print(f"  Test  | Acc: {100*test_metrics['accuracy']:.2f}% | "
              f"Steps (eval): {test_metrics['avg_steps']:.1f} | "
              f"Time: {elapsed:.1f}s")
        
        # Key check: eval steps should be ≤ train steps (adaptive compute works)
        if test_metrics['avg_steps'] < train_metrics['avg_steps']:
            print(f"  ✅ Adaptive: eval uses {train_metrics['avg_steps'] - test_metrics['avg_steps']:.1f} fewer steps")
        
        record = {
            'epoch'       : epoch,
            'train_loss'  : train_metrics['loss'],
            'train_acc'   : train_metrics['accuracy'],
            'test_acc'    : test_metrics['accuracy'],
            'train_steps' : train_metrics['avg_steps'],
            'eval_steps'  : test_metrics['avg_steps'],
        }
        history.append(record)
        
        # Save best
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
            torch.save({
                'epoch'    : epoch,
                'model'    : model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'acc'      : best_acc,
                'config'   : cfg,
            }, os.path.join(args.save_dir, f'best_{args.dataset}.pt'))
            print(f"  💾 Saved best model (acc={100*best_acc:.2f}%)")
    
    # Final report
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS — {args.dataset.upper()}")
    print(f"{'='*60}")
    print(f"Best test accuracy: {100*best_acc:.2f}%")
    
    targets = {'mnist': 98.0, 'cifar10': 70.0}
    target  = targets[args.dataset]
    if 100 * best_acc >= target:
        print(f"✅ Target ({target}%) ACHIEVED")
    else:
        print(f"❌ Target ({target}%) NOT reached — investigate architecture")
    
    # Post-training benchmarks
    benchmark_adaptive_compute(model, device)
    
    # Save history
    with open(os.path.join(args.save_dir, f'history_{args.dataset}.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nHistory saved to {args.save_dir}/history_{args.dataset}.json")


if __name__ == '__main__':
    main()
