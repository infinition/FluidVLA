"""Convert a local LeRobot dataset into FluidVLA step-style episode files.

V2 changes vs original:
  1. Computes and saves per-joint normalization stats (mean/std) for actions and proprios.
  2. Optional delta-action mode: stores (action - proprio) instead of absolute positions.
  3. Optional static-frame subsampling: drops frames where |action - proprio| < threshold.
  4. Saves norm_stats.json alongside metadata.json for use by train and inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a LeRobot dataset into FluidVLA .npz episodes (V2)"
    )
    parser.add_argument(
        "--lerobot-root",
        required=True,
        help="Path to the LeRobot repository root containing src/lerobot",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="LeRobot dataset repo id, for example local/so101_balle_bol_test",
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Local LeRobot dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory where episode_XXXX.npz files will be written",
    )
    parser.add_argument(
        "--camera-key",
        default=None,
        help="Exact LeRobot camera feature to use. Defaults to the first camera key",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Target square resolution for FluidVLA inputs",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=4,
        help="Temporal window size stacked per sample",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap on the number of converted episodes",
    )
    parser.add_argument(
        "--reward",
        type=float,
        default=1.0,
        help="Reward stored for each converted episode",
    )
    parser.add_argument(
        "--video-backend",
        default=None,
        help="Optional LeRobot video backend override",
    )
    # ── V2 additions ──
    parser.add_argument(
        "--delta-actions",
        action="store_true",
        default=False,
        help="Store delta-actions (action - proprio) instead of absolute positions. "
             "This removes the bias toward static poses.",
    )
    parser.add_argument(
        "--filter-static",
        type=float,
        default=0.0,
        help="Drop frames where max |action - proprio| < this threshold (degrees). "
             "0 = keep all frames. Recommended: 0.5 to 1.0 for SO-101.",
    )
    parser.add_argument(
        "--subsample-static",
        type=int,
        default=1,
        help="Instead of dropping static frames, keep every Nth one. "
             "Applied only to frames below --filter-static threshold. "
             "Example: --filter-static 1.0 --subsample-static 4 keeps 1 in 4 static frames.",
    )
    return parser.parse_args()


def add_lerobot_to_path(lerobot_root: Path) -> None:
    src_dir = lerobot_root / "src"
    if not src_dir.exists():
        raise FileNotFoundError(f"LeRobot src directory not found: {src_dir}")
    sys.path.insert(0, str(src_dir))
    sys.path.insert(0, str(lerobot_root))


def to_chw_float(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError(f"Expected 3D image tensor, got shape={tuple(image.shape)}")
    if image.shape[0] == 3:
        chw = image.float()
    elif image.shape[-1] == 3:
        chw = image.permute(2, 0, 1).float()
    else:
        raise ValueError(f"Unsupported image layout: shape={tuple(image.shape)}")

    if chw.max().item() > 1.0:
        chw = chw / 255.0
    return chw.clamp(0.0, 1.0)


def resize_image(image: torch.Tensor, image_size: int) -> torch.Tensor:
    if image.shape[-2:] == (image_size, image_size):
        return image
    return F.interpolate(
        image.unsqueeze(0),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def build_temporal_windows(images: list[torch.Tensor], n_frames: int) -> np.ndarray:
    windows = []
    for frame_idx in range(len(images)):
        start_idx = max(0, frame_idx - n_frames + 1)
        history = images[start_idx : frame_idx + 1]
        if len(history) < n_frames:
            pad = [history[0]] * (n_frames - len(history))
            history = pad + history
        windows.append(torch.stack(history, dim=1).cpu().numpy())
    return np.asarray(windows, dtype=np.float32)


def summarize_metadata(ds: Any, camera_key: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "source": {
            "repo_id": args.repo_id,
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "lerobot_root": str(Path(args.lerobot_root).resolve()),
            "robot_type": ds.meta.robot_type,
            "fps": ds.meta.fps,
            "camera_key": camera_key,
        },
        "fluidvla": {
            "image_size": args.image_size,
            "n_frames": args.n_frames,
            "action_dim": int(ds.meta.shapes["action"][0]),
            "proprio_dim": int(ds.meta.shapes["observation.state"][0]),
            "reward_value": args.reward,
            "delta_actions": args.delta_actions,
            "filter_static": args.filter_static,
            "subsample_static": args.subsample_static,
        },
        "episodes": [],
    }


def main() -> None:
    args = parse_args()
    lerobot_root = Path(args.lerobot_root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    add_lerobot_to_path(lerobot_root)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(
        args.repo_id,
        root=dataset_root,
        video_backend=args.video_backend,
    )

    if not ds.meta.camera_keys:
        raise ValueError("Dataset has no camera keys; FluidVLA requires image observations")

    camera_key = args.camera_key or ds.meta.camera_keys[0]
    if camera_key not in ds.meta.camera_keys:
        raise ValueError(
            f"Camera key {camera_key!r} not found. Available: {ds.meta.camera_keys}"
        )

    meta = summarize_metadata(ds, camera_key, args)
    total_episodes = ds.meta.total_episodes
    episode_indices = list(range(total_episodes))
    if args.max_episodes is not None:
        episode_indices = episode_indices[: args.max_episodes]

    print(f"Converting {len(episode_indices)} episode(s) from {dataset_root}")
    print(f"  camera_key       : {camera_key}")
    print(f"  image_size       : {args.image_size}")
    print(f"  n_frames         : {args.n_frames}")
    print(f"  delta_actions    : {args.delta_actions}")
    print(f"  filter_static    : {args.filter_static}")
    print(f"  subsample_static : {args.subsample_static}")
    print(f"  output_dir       : {output_dir}")

    # ── First pass: collect all proprios and actions for normalization stats ──
    all_proprios = []
    all_actions = []
    all_deltas = []

    episode_data_cache: dict[int, dict] = {}

    for episode_index in episode_indices:
        episode_meta = ds.meta.episodes[episode_index]
        start_idx = int(episode_meta["dataset_from_index"])
        end_idx = int(episode_meta["dataset_to_index"])

        ep_proprios = []
        ep_actions = []
        ep_images = []

        for sample_idx in range(start_idx, end_idx):
            sample = ds[sample_idx]
            image = resize_image(to_chw_float(sample[camera_key]), args.image_size)
            ep_images.append(image)
            p = sample["observation.state"].detach().cpu().numpy().astype(np.float32)
            a = sample["action"].detach().cpu().numpy().astype(np.float32)
            ep_proprios.append(p)
            ep_actions.append(a)

        if not ep_images:
            continue

        proprios_np = np.asarray(ep_proprios, dtype=np.float32)
        actions_np = np.asarray(ep_actions, dtype=np.float32)
        deltas_np = actions_np - proprios_np

        all_proprios.append(proprios_np)
        all_actions.append(actions_np)
        all_deltas.append(deltas_np)

        episode_data_cache[episode_index] = {
            "images": ep_images,
            "proprios": proprios_np,
            "actions": actions_np,
            "deltas": deltas_np,
            "task": sample.get("task", "unknown"),
        }

    # ── Compute normalization statistics ──
    all_proprios_cat = np.concatenate(all_proprios, axis=0)
    all_actions_cat = np.concatenate(all_actions, axis=0)
    all_deltas_cat = np.concatenate(all_deltas, axis=0)

    norm_stats = {
        "proprio_mean": all_proprios_cat.mean(axis=0).tolist(),
        "proprio_std": all_proprios_cat.std(axis=0).tolist(),
        "action_mean": all_actions_cat.mean(axis=0).tolist(),
        "action_std": all_actions_cat.std(axis=0).tolist(),
        "delta_mean": all_deltas_cat.mean(axis=0).tolist(),
        "delta_std": all_deltas_cat.std(axis=0).tolist(),
        "delta_actions": args.delta_actions,
    }

    # Replace zero stds with 1.0 to avoid division by zero
    for key in ["proprio_std", "action_std", "delta_std"]:
        norm_stats[key] = [max(v, 1e-6) for v in norm_stats[key]]

    print("\n── Normalization statistics ──")
    print(f"  proprio mean : {np.round(all_proprios_cat.mean(axis=0), 2)}")
    print(f"  proprio std  : {np.round(all_proprios_cat.std(axis=0), 2)}")
    print(f"  action mean  : {np.round(all_actions_cat.mean(axis=0), 2)}")
    print(f"  action std   : {np.round(all_actions_cat.std(axis=0), 2)}")
    print(f"  delta mean   : {np.round(all_deltas_cat.mean(axis=0), 2)}")
    print(f"  delta std    : {np.round(all_deltas_cat.std(axis=0), 2)}")

    # Report static frame proportion
    max_abs_delta = np.abs(all_deltas_cat).max(axis=1)
    if args.filter_static > 0:
        n_static = (max_abs_delta < args.filter_static).sum()
        print(f"\n  Static frames (max|delta| < {args.filter_static}°): "
              f"{n_static}/{len(max_abs_delta)} ({100*n_static/len(max_abs_delta):.1f}%)")

    # ── Second pass: write episodes with filtering and normalization ──
    print()
    total_kept = 0
    total_dropped = 0

    # Normalization arrays for vectorized operations
    if args.delta_actions:
        target_mean = np.array(norm_stats["delta_mean"], dtype=np.float32)
        target_std = np.array(norm_stats["delta_std"], dtype=np.float32)
    else:
        target_mean = np.array(norm_stats["action_mean"], dtype=np.float32)
        target_std = np.array(norm_stats["action_std"], dtype=np.float32)

    proprio_mean = np.array(norm_stats["proprio_mean"], dtype=np.float32)
    proprio_std = np.array(norm_stats["proprio_std"], dtype=np.float32)

    for episode_index in episode_indices:
        if episode_index not in episode_data_cache:
            continue

        ep = episode_data_cache[episode_index]
        images = ep["images"]
        proprios = ep["proprios"]
        actions = ep["actions"]
        deltas = ep["deltas"]

        # ── Static frame filtering / subsampling ──
        keep_mask = np.ones(len(actions), dtype=bool)
        if args.filter_static > 0:
            max_delta_per_frame = np.abs(deltas).max(axis=1)
            is_static = max_delta_per_frame < args.filter_static

            if args.subsample_static > 1:
                # Keep every Nth static frame instead of dropping all
                static_indices = np.where(is_static)[0]
                drop_indices = static_indices[np.arange(len(static_indices)) % args.subsample_static != 0]
                keep_mask[drop_indices] = False
            else:
                keep_mask[is_static] = False

        kept_indices = np.where(keep_mask)[0]
        n_dropped = int((~keep_mask).sum())
        total_dropped += n_dropped
        total_kept += len(kept_indices)

        if len(kept_indices) == 0:
            continue

        # Filter arrays
        kept_images = [images[i] for i in kept_indices]
        kept_proprios = proprios[kept_indices]
        kept_actions = actions[kept_indices]
        kept_deltas = deltas[kept_indices]

        # ── Build targets ──
        if args.delta_actions:
            targets = (kept_deltas - target_mean) / target_std
        else:
            targets = (kept_actions - target_mean) / target_std

        # Normalize proprios too
        kept_proprios_norm = (kept_proprios - proprio_mean) / proprio_std

        # ── Build temporal windows from kept frames ──
        frames = build_temporal_windows(kept_images, args.n_frames)

        reward_np = np.asarray([args.reward], dtype=np.float32)

        episode_path = output_dir / f"episode_{episode_index:04d}.npz"
        np.savez_compressed(
            episode_path,
            frames=frames,
            proprios=kept_proprios_norm.astype(np.float32),
            actions=targets.astype(np.float32),
            reward=reward_np,
            # Also store raw values for debugging
            proprios_raw=kept_proprios.astype(np.float32),
            actions_raw=kept_actions.astype(np.float32),
        )

        meta["episodes"].append(
            {
                "episode_index": episode_index,
                "num_steps": int(targets.shape[0]),
                "num_steps_original": int(actions.shape[0]),
                "num_dropped": n_dropped,
                "task": ep["task"],
                "path": episode_path.name,
            }
        )
        print(
            f"  episode {episode_index:03d} -> {episode_path.name} "
            f"steps={targets.shape[0]} (dropped {n_dropped})"
        )

    # ── Save metadata and norm stats ──
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    with (output_dir / "norm_stats.json").open("w", encoding="utf-8") as f:
        json.dump(norm_stats, f, indent=2)

    print(f"\nConversion complete")
    print(f"  Total kept    : {total_kept}")
    print(f"  Total dropped : {total_dropped}")
    print(f"  metadata      : {output_dir / 'metadata.json'}")
    print(f"  norm_stats    : {output_dir / 'norm_stats.json'}")


if __name__ == "__main__":
    main()
