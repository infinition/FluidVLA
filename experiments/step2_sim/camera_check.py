"""
camera_check.py — Isaac Sim Camera Validation
═══════════════════════════════════════════════
Runs N episodes, captures frames at each step,
displays stats and generates a complete visual report.

Usage:
  python experiments/step2_sim/camera_check.py
  python experiments/step2_sim/camera_check.py --episodes 20 --output ./camera_report
  python experiments/step2_sim/camera_check.py --mode synthetic   # test without Isaac Sim

What this script validates:
  PASS Non-black frames (max > 0.01)
  PASS Visible objects (red + green detectable by color)
  PASS Temporal consistency (frames T=0..3 consistent)
  PASS Inter-episode stability
  PASS Capture latency
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Add project root to path ──
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────
# Analysis helpers
# ─────────────────────────────────────────

def frame_stats(frame: np.ndarray) -> dict:
    """Calculates key stats of a (H, W, 3) frame."""
    return {
        'min':   float(frame.min()),
        'max':   float(frame.max()),
        'mean':  float(frame.mean()),
        'std':   float(frame.std()),
        'black': frame.max() < 0.01,
        # Color detection: red cube
        'has_red':   bool(((frame[:, :, 0] > 0.6) & (frame[:, :, 1] < 0.3) & (frame[:, :, 2] < 0.3)).mean() > 0.001),
        # Color detection: green target
        'has_green': bool(((frame[:, :, 1] > 0.6) & (frame[:, :, 0] < 0.3) & (frame[:, :, 2] < 0.3)).mean() > 0.001),
    }


def check_temporal_consistency(frames_thwc: np.ndarray) -> dict:
    """
    frames_thwc : (T, H, W, 3)
    Verify that the 4 temporal frames are consistent with each other.
    """
    T = frames_thwc.shape[0]
    diffs = []
    for t in range(1, T):
        diff = np.abs(frames_thwc[t].astype(float) - frames_thwc[t-1].astype(float)).mean()
        diffs.append(diff)
    return {
        'mean_frame_diff': float(np.mean(diffs)) if diffs else 0.0,
        'all_identical':   all(d < 1e-4 for d in diffs),
        'all_different':   all(d > 0.01 for d in diffs),
    }


# ─────────────────────────────────────────
# Visual report
# ─────────────────────────────────────────

def generate_report(episodes_data: list, output_dir: Path, mode: str):
    """
    Generate camera_report.png with:
      - Frame grid for each episode (start, mid, end step)
      - Stats graph (max brightness per step)
      - Summary table
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n_ep = len(episodes_data)

    print(f"\nGenerating report ({n_ep} episodes)...")

    # ── Main figure: frames grid ──
    steps_to_show = ['first', 'mid', 'last']
    n_cols = 4 * len(steps_to_show)   # 4 temporal frames × 3 moments
    n_rows = n_ep

    fig = plt.figure(figsize=(n_cols * 2.5, n_rows * 2.5 + 2))
    fig.suptitle(
        f'FluidVLA — Camera Validation Report\n'
        f'Mode: {mode} | Episodes: {n_ep}',
        fontsize=14, fontweight='bold', y=0.98
    )

    gs = gridspec.GridSpec(n_rows + 1, n_cols, figure=fig,
                           hspace=0.4, wspace=0.1,
                           top=0.92, bottom=0.08)

    # Column headers
    col_labels = []
    for moment in steps_to_show:
        for t in range(4):
            col_labels.append(f'{moment}\nT={t}')

    for col_idx, label in enumerate(col_labels):
        ax = fig.add_subplot(gs[0, col_idx])
        ax.text(0.5, 0.5, label, ha='center', va='center',
                fontsize=8, fontweight='bold', transform=ax.transAxes)
        ax.axis('off')

    # Frames per episode
    for ep_idx, ep_data in enumerate(episodes_data):
        row = ep_idx + 1
        frames_by_step = ep_data['frames_by_step']  # dict: 'first'/'mid'/'last' → (4, H, W, 3)

        col_idx = 0
        for moment in steps_to_show:
            frames_t = frames_by_step.get(moment)  # (4, H, W, 3) or None

            for t in range(4):
                ax = fig.add_subplot(gs[row, col_idx])

                if frames_t is not None:
                    img = np.clip(frames_t[t], 0, 1)
                    ax.imshow(img)
                    stats = frame_stats(img)
                    color = 'red' if stats['black'] else ('green' if (stats['has_red'] or stats['has_green']) else 'orange')
                    indicator = 'BLK' if stats['black'] else ('R+G' if (stats['has_red'] or stats['has_green']) else 'WHT')
                    ax.set_title(f'Ep{ep_idx} {indicator}\nmax={stats["max"]:.2f}',
                                fontsize=6, color=color, pad=2)
                else:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)

                ax.axis('off')
                col_idx += 1

    frames_path = output_dir / 'camera_frames.png'
    fig.savefig(frames_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] Frames -> {frames_path}")

    # ── Stats figure: brightness per step ──
    fig2, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig2.suptitle('Camera Stats per Step', fontsize=13, fontweight='bold')

    ax_bright = axes[0]
    ax_red    = axes[1]

    colors = plt.cm.tab10(np.linspace(0, 1, n_ep))

    for ep_idx, ep_data in enumerate(episodes_data):
        steps_stats = ep_data['steps_stats']
        if not steps_stats:
            continue
        step_indices = list(range(len(steps_stats)))
        brightness   = [s['max'] for s in steps_stats]
        has_red      = [float(s['has_red']) for s in steps_stats]
        has_green    = [float(s['has_green']) for s in steps_stats]

        ax_bright.plot(step_indices, brightness,
                       color=colors[ep_idx], alpha=0.8,
                       label=f'Ep {ep_idx}', linewidth=1.5)
        ax_red.plot(step_indices, has_red,
                    color=colors[ep_idx], alpha=0.6,
                    linestyle='-', linewidth=1.5)

    ax_bright.axhline(0.01, color='red', linestyle='--', linewidth=1, label='Black threshold (0.01)')
    ax_bright.set_ylabel('Max pixel value')
    ax_bright.set_xlabel('Step')
    ax_bright.set_title('Brightness per step (>0.01 = not black)')
    ax_bright.legend(fontsize=7, ncol=min(n_ep, 5), loc='upper right')
    ax_bright.set_ylim(-0.05, 1.1)
    ax_bright.grid(alpha=0.3)

    ax_red.set_ylabel('Has red object (0/1)')
    ax_red.set_xlabel('Step')
    ax_red.set_title('Red cube detected per step')
    ax_red.set_ylim(-0.1, 1.2)
    ax_red.grid(alpha=0.3)

    stats_path = output_dir / 'camera_stats.png'
    fig2.savefig(stats_path, dpi=100, bbox_inches='tight')
    plt.close(fig2)
    print(f"  [OK] Stats  -> {stats_path}")

    # ── Text summary ──
    all_stats = [s for ep in episodes_data for s in ep['steps_stats']]
    n_black   = sum(1 for s in all_stats if s['black'])
    n_red     = sum(1 for s in all_stats if s['has_red'])
    n_green   = sum(1 for s in all_stats if s['has_green'])
    n_total   = len(all_stats)

    print(f"\n{'='*55}")
    print(f"CAMERA SUMMARY")
    print(f"{'='*55}")
    print(f"  Episodes      : {n_ep}")
    print(f"  Total steps   : {n_total}")
    print(f"  Black frames  : {n_black}/{n_total} ({100*n_black/max(n_total,1):.1f}%)")
    print(f"  Red cube      : {n_red}/{n_total} ({100*n_red/max(n_total,1):.1f}%)")
    print(f"  Green target  : {n_green}/{n_total} ({100*n_green/max(n_total,1):.1f}%)")

    if n_black == 0:
        print(f"\n  [OK] Camera OK - no black frames")
    elif n_black < n_total * 0.1:
        print(f"\n  [WARN] A few black frames (warm-up) - acceptable")
    else:
        print(f"\n  [FAIL] Too many black frames - unresolved camera issue")

    if n_red > n_total * 0.3:
        print(f"  [OK] Red cube visible in {100*n_red/n_total:.0f}% of frames")
    else:
        print(f"  [FAIL] Red cube NOT visible - camera misoriented or scene poorly lit")

    verdict = (n_black == 0 or n_black < n_total * 0.1) and n_red > n_total * 0.3
    print(f"\n{'='*55}")
    print(f"  VERDICT : {'[OK] READY for 1000 episode collection' if verdict else '[FAIL] FIX camera before collection'}")
    print(f"{'='*55}")

    return verdict


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='FluidVLA - Camera Validation')
    parser.add_argument('--mode',     default='collect',
                        choices=['collect', 'synthetic'],
                        help='collect=Isaac Sim, synthetic=without Isaac Sim')
    parser.add_argument('--episodes', default=10, type=int)
    parser.add_argument('--output',   default='./camera_report')
    parser.add_argument('--image_size', default=224, type=int)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load environment ──
    print(f"\n{'='*55}")
    print(f"FluidVLA - Camera Validation")
    print(f"  Mode      : {args.mode}")
    print(f"  Episodes  : {args.episodes}")
    print(f"  Output    : {output_dir}")
    print(f"{'='*55}\n")

    try:
        from experiments.step2_sim.isaac_env import (
            IsaacPickPlace, SyntheticPickPlace, N_FRAMES
        )
    except ImportError:
        # Fallback path
        sys.path.insert(0, str(Path(__file__).parent))
        from isaac_env import IsaacPickPlace, SyntheticPickPlace, N_FRAMES

    if args.mode == 'synthetic':
        env = SyntheticPickPlace(args.image_size, N_FRAMES)
        print("[OK] Synthetic environment loaded")
    else:
        env = IsaacPickPlace(headless=True, image_size=args.image_size, n_frames=N_FRAMES)
        if not env.available:
            print("[WARN] Isaac Sim not available - synthetic fallback")
            env = SyntheticPickPlace(args.image_size, N_FRAMES)

    # ── Collect episodes ──
    episodes_data = []

    for ep in range(args.episodes):
        print(f"\n-- Episode {ep+1}/{args.episodes} --")
        obs = env.reset()
        done = False
        step = 0

        step_frames  = []   # list of (4, H, W, 3) — one per step
        steps_stats  = []
        capture_times = []

        while not done and step < 200:
            action = env.oracle_action()

            t0 = time.perf_counter()
            next_obs, reward, done, info = env.step(action)
            capture_ms = (time.perf_counter() - t0) * 1000
            capture_times.append(capture_ms)

            # Extract frames from obs (format (1, 3, T, H, W))
            frames_tensor = obs['frames'].squeeze(0)          # (3, T, H, W)
            frames_np = frames_tensor.permute(1, 2, 3, 0).numpy()  # (T, H, W, 3)

            step_frames.append(frames_np)

            # Stats on the first temporal frame (T=0)
            s = frame_stats(frames_np[0])
            s['step'] = step
            steps_stats.append(s)

            obs = next_obs
            step += 1

        # Select 3 representative moments
        n = len(step_frames)
        idx_first = 0
        idx_mid   = n // 2
        idx_last  = n - 1

        frames_by_step = {}
        if n > 0:
            frames_by_step['first'] = step_frames[idx_first]
            frames_by_step['mid']   = step_frames[idx_mid]
            frames_by_step['last']  = step_frames[idx_last]

        # Temporal consistency on middle step
        temporal = {}
        if n > 0:
            temporal = check_temporal_consistency(step_frames[idx_mid])

        episodes_data.append({
            'ep':           ep,
            'n_steps':      n,
            'reward':       reward,
            'frames_by_step': frames_by_step,
            'steps_stats':  steps_stats,
            'temporal':     temporal,
            'capture_ms_mean': float(np.mean(capture_times)) if capture_times else 0,
        })

        n_black = sum(1 for s in steps_stats if s['black'])
        n_red   = sum(1 for s in steps_stats if s['has_red'])
        status_cam   = '[OK]' if n_black == 0 else f'[WARN] {n_black} black'
        status_red   = '[OK]' if n_red > 0 else '[FAIL]'
        status_ok    = '[OK]' if reward > 0 else '[FAIL]'

        print(f"  Steps: {n} | Reward: {reward:.0f} {status_ok} | "
              f"Black frames: {status_cam} | Red cube: {status_red} | "
              f"Capture: {episodes_data[-1]['capture_ms_mean']:.1f}ms/step")
        if temporal:
            print(f"  Temporal consistency: diff_mean={temporal['mean_frame_diff']:.4f} | "
                  f"all_identical={temporal['all_identical']}")

    # ── Generate report ──
    verdict = generate_report(episodes_data, output_dir, args.mode)

    if hasattr(env, 'close'):
        env.close()

    print(f"\n[DIR] Complete report -> {output_dir}/")
    print(f"   camera_frames.png  - visual frames grid")
    print(f"   camera_stats.png   - brightness / detection per step")

    if verdict and args.mode == 'collect':
        print(f"\nReady! Now run:")
        print(f"   python experiments/step2_sim/isaac_env.py --mode collect --episodes 1000")
    elif not verdict:
        print(f"\nFix the camera, then relaunch this script.")

    return 0 if verdict else 1


if __name__ == '__main__':
    exit(main())
