"""
isaac_env.py — Step 2: Isaac Sim Pick & Place Environment
─────────────────────────────────────────────────────────
Fixes vs original:
  • Camera warm-up increased (5 → 30 steps) + explicit frame validation
  • _capture_frame() retries up to 5× if frame is black/None
  • Added DEBUG_CAMERA flag for diagnosing render issues
  • Camera orientation uses explicit quaternion (avoids euler ambiguity)
  • Synthetic env uses the same obs format for consistency
  • collect_demonstrations saves metadata (env type, image_size, etc.)
  • evaluate_policy handles variable action dims gracefully
  • General cleanup: no duplicate torch import, type hints, docstrings
"""

import os
import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.core.fluid_model import FluidBotVLA

# ─────────────────────────────────────────
# Constants
# ─────────────────────────────────────────
CAMERA_H       = 224
CAMERA_W       = 224
N_FRAMES       = 4
ACTION_DIM     = 7
PROPRIO_DIM    = 8
SUCCESS_THRESH = 0.03

WORKSPACE = {
    'x': (-0.3, 0.3),
    'y': (-0.3, 0.3),
    'z': ( 0.0, 0.4),
}

# Set True to print frame stats on every capture
DEBUG_CAMERA = False

# Isaac Sim warm-up steps (RTX renderer needs time to initialize)
WARMUP_STEPS = 30

# Max retries when camera returns black/empty frames
CAPTURE_RETRIES = 5


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────

def _clip_to_workspace(pos: np.ndarray) -> np.ndarray:
    """Clip a 3D position to the workspace bounds."""
    return np.clip(
        pos,
        [WORKSPACE['x'][0], WORKSPACE['y'][0], 0.0],
        [WORKSPACE['x'][1], WORKSPACE['y'][1], WORKSPACE['z'][1]],
    )


def _proportional_action(current: np.ndarray, target: np.ndarray,
                          gain: float = 5.0) -> np.ndarray:
    """Proportional controller: returns clipped [-1, 1] action."""
    return np.clip((target - current) * gain, -1, 1)


def _make_frames_tensor(frame_buffer: List[np.ndarray]) -> torch.Tensor:
    """
    Convert list of (H, W, 3) float32 frames → (1, 3, T, H, W) tensor.
    """
    frames = np.stack(frame_buffer, axis=0)                # (T, H, W, 3)
    return (
        torch.from_numpy(frames)
        .permute(3, 0, 1, 2)     # (3, T, H, W)
        .unsqueeze(0)            # (1, 3, T, H, W)
        .float()
    )


def _make_proprio(ee_pos: np.ndarray, obj_pos: np.ndarray,
                  tgt_pos: np.ndarray, gripper: float) -> torch.Tensor:
    """Build (1, 8) proprioception tensor."""
    proprio = np.concatenate([
        ee_pos,
        obj_pos[:2],
        tgt_pos[:2],
        [gripper],
    ]).astype(np.float32)
    return torch.from_numpy(proprio).unsqueeze(0)


# ─────────────────────────────────────────
# Oracle controller (shared by both envs)
# ─────────────────────────────────────────

class OraclePickPlace:
    """
    5-phase proportional oracle:
      0 — approach XY above object
      1 — descend to object, open gripper → close
      2 — lift
      3 — move XY to target
      4 — descend & release
    """

    def __init__(self):
        self.phase   = 0
        self.gripper = 0.0

    def reset(self):
        self.phase   = 0
        self.gripper = 0.0

    def __call__(self, ee: np.ndarray, obj: np.ndarray,
                 tgt: np.ndarray) -> np.ndarray:
        action = np.zeros(ACTION_DIM, dtype=np.float32)

        if self.phase == 0:
            # Approach XY above object
            action[:2] = _proportional_action(ee[:2], obj[:2])
            action[2]  = np.clip((0.15 - ee[2]) * 3.0, -1, 1)
            action[6]  = 0.0
            if np.linalg.norm(ee[:2] - obj[:2]) < 0.05:
                self.phase = 1

        elif self.phase == 1:
            # Descend to object
            action[:3] = _proportional_action(ee[:3], obj[:3])
            action[6]  = 1.0
            if ee[2] < obj[2] + 0.03:
                self.gripper = 1.0
                self.phase = 2

        elif self.phase == 2:
            # Lift
            action[2] = 1.0
            action[6] = 1.0
            if ee[2] > 0.15:
                self.phase = 3

        elif self.phase == 3:
            # Move XY to target
            action[:2] = _proportional_action(ee[:2], tgt[:2])
            action[6]  = 1.0
            if np.linalg.norm(ee[:2] - tgt[:2]) < 0.05:
                self.phase = 4

        elif self.phase == 4:
            # Lower & release
            place_z = tgt[2] + 0.02
            action[2] = np.clip((place_z - ee[2]) * 5.0, -1, 1)
            action[6] = 0.0
            self.gripper = 0.0

        return np.clip(action, -1.0, 1.0)


# ─────────────────────────────────────────
# Synthetic Environment (no Isaac Sim)
# ─────────────────────────────────────────

class SyntheticPickPlace:
    """
    Lightweight pick-and-place with procedural rendering.
    Produces identical obs format to IsaacPickPlace.
    """

    def __init__(self, image_size: int = 224, n_frames: int = 4, seed: int = 42):
        self.image_size   = image_size
        self.n_frames     = n_frames
        self.rng          = np.random.default_rng(seed)
        self.step_count   = 0
        self.object_pos   = self._random_pos()
        self.target_pos   = self._random_pos()
        self.ee_pos       = np.array([0.0, 0.0, 0.2])
        self.oracle       = OraclePickPlace()
        self.frame_buffer: List[np.ndarray] = []

    def _random_pos(self) -> np.ndarray:
        return np.array([
            self.rng.uniform(*WORKSPACE['x']),
            self.rng.uniform(*WORKSPACE['y']),
            0.02,
        ])

    def _render_frame(self) -> np.ndarray:
        """Returns (H, W, 3) float32 in [0, 1]."""
        H = W = self.image_size
        frame = np.full((H, W, 3), 0.75, dtype=np.float32)

        def world_to_pixel(pos):
            px = int((pos[0] - WORKSPACE['x'][0]) /
                     (WORKSPACE['x'][1] - WORKSPACE['x'][0]) * W)
            py = int((pos[1] - WORKSPACE['y'][0]) /
                     (WORKSPACE['y'][1] - WORKSPACE['y'][0]) * H)
            return np.clip(px, 0, W - 1), np.clip(py, 0, H - 1)

        def draw_circle(img, pos, color, radius=12):
            cx, cy = world_to_pixel(pos)
            y_idx, x_idx = np.ogrid[:H, :W]
            mask = (x_idx - cx) ** 2 + (y_idx - cy) ** 2 < radius ** 2
            img[mask] = color

        draw_circle(frame, self.target_pos, [0.1, 0.8, 0.1], radius=16)
        draw_circle(frame, self.object_pos, [0.9, 0.1, 0.1], radius=12)
        draw_circle(frame, self.ee_pos,     [0.1, 0.1, 0.9], radius=6)
        return frame

    def reset(self) -> Dict[str, Any]:
        self.object_pos = self._random_pos()
        self.target_pos = self._random_pos()
        self.ee_pos     = np.array([0.0, 0.0, 0.2])
        self.step_count = 0
        self.oracle.reset()

        frame = self._render_frame()
        self.frame_buffer = [frame.copy() for _ in range(self.n_frames)]
        return self._get_obs()

    def _get_obs(self) -> Dict[str, Any]:
        return {
            'frames':     _make_frames_tensor(self.frame_buffer),
            'proprio':    _make_proprio(self.ee_pos, self.object_pos,
                                        self.target_pos, self.oracle.gripper),
            'object_pos': self.object_pos.copy(),
            'target_pos': self.target_pos.copy(),
            'ee_pos':     self.ee_pos.copy(),
        }

    def oracle_action(self) -> np.ndarray:
        return self.oracle(self.ee_pos, self.object_pos, self.target_pos)

    def step(self, action: np.ndarray):
        # Apply action
        self.ee_pos += action[:3] * 0.08
        self.ee_pos  = _clip_to_workspace(self.ee_pos)
        self.oracle.gripper = float(action[6] > 0.5)

        # Gripper attachment logic
        if (self.oracle.gripper > 0.5 and
                np.linalg.norm(self.ee_pos[:2] - self.object_pos[:2]) < 0.12):
            self.object_pos       = self.ee_pos.copy()
            self.object_pos[2]    = max(0.0, self.ee_pos[2] - 0.02)

        # Update frame buffer
        frame = self._render_frame()
        self.frame_buffer.pop(0)
        self.frame_buffer.append(frame.copy())
        self.step_count += 1

        dist    = np.linalg.norm(self.object_pos[:2] - self.target_pos[:2])
        success = dist < SUCCESS_THRESH
        done    = success or self.step_count >= 200

        return self._get_obs(), float(success), done, {
            'success': success,
            'dist_to_target': float(dist),
            'step': self.step_count,
        }


# ─────────────────────────────────────────
# Isaac Sim Environment
# ─────────────────────────────────────────

class IsaacPickPlace:
    """
    Isaac Sim 4.x wrapper with:
      • Robust camera warm-up + retry logic
      • Explicit quaternion orientation (no euler ambiguity)
      • Distant light for uniform illumination
      • Debug diagnostics for black-frame issues
    """

    def __init__(self, headless: bool = True, image_size: int = 224,
                 n_frames: int = 4):
        self.image_size   = image_size
        self.n_frames     = n_frames
        self.available    = False
        self.frame_buffer: List[np.ndarray] = []
        self.step_count   = 0

        self.ee_pos  = np.array([0.0, 0.0, 0.2])
        self.obj_pos = np.zeros(3)
        self.tgt_pos = np.zeros(3)
        self.oracle  = OraclePickPlace()

        try:
            self._init_isaac(headless)
        except Exception as e:
            import traceback
            print(f"\n[CRITICAL ERROR] Isaac Sim failed to initialize:")
            traceback.print_exc()
            print("\n[Fallback] Switching to synthetic environment...")

    def _init_isaac(self, headless: bool):
            from isaacsim import SimulationApp
            self.sim_app = SimulationApp({'headless': headless})

            from omni.isaac.core import World
            from omni.isaac.core.objects import DynamicCuboid, VisualSphere
            from omni.isaac.sensor import Camera
            import omni.isaac.core.utils.prims as prim_utils
            import numpy as np

            self.world = World(stage_units_in_meters=1.0)
            self.world.scene.add_default_ground_plane()

            # ── 1. Lighting (The good old create_prim, but without USD bug) ──
            light_quat = np.array([0.7071068, 0.0, 0.7071068, 0.0]) # w, x, y, z
            prim_utils.create_prim(
                prim_path='/World/DistantLight',
                prim_type='DistantLight',
                orientation=light_quat, # Isaac handles the conversion!
                attributes={
                    'inputs:intensity': 3000.0,
                    'inputs:color': (1.0, 1.0, 1.0),
                }
            )

            # ── 2. Scene objects ──
            self.cube = self.world.scene.add(DynamicCuboid(
                prim_path='/World/cube', name='cube',
                position=np.array([0.1, 0.0, 0.02]),
                scale=np.array([0.04, 0.04, 0.04]),
                color=np.array([0.9, 0.1, 0.1]),
            ))
            self.target_marker = self.world.scene.add(VisualSphere(
                prim_path='/World/target', name='target',
                position=np.array([-0.1, 0.1, 0.01]),
                radius=0.03,
                color=np.array([0.1, 0.8, 0.1]),
            ))

            # ── 3. Camera: Top-down ──
            self.camera = Camera(
                prim_path='/World/camera_topdown',
                position=np.array([0.0, 0.0, 1.0]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                frequency=30,
                resolution=(self.image_size, self.image_size),
            )

            self.world.reset()
            self.camera.initialize()

            # ── 4. Warm-up ──
            print(f"[Isaac Sim] Warming up renderer ({WARMUP_STEPS} steps)...")
            for i in range(WARMUP_STEPS):
                self.world.step(render=True)

            test_frame = self._capture_frame(retries=CAPTURE_RETRIES)
            frame_max  = test_frame.max()
            print(f"[Isaac Sim] Post-warmup frame check: max={frame_max:.4f}")
            
            self.available = True
            print(f"[Isaac Sim] ✅ Initialized (camera {'OK' if frame_max > 0.01 else 'BLACK'})")

    def _capture_frame(self, retries: int = 1) -> np.ndarray:
        """
        Capture camera → (H, W, 3) float32 in [0, 1].
        Retries up to `retries` times if the frame is black or None.
        """
        blank = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)

        for attempt in range(retries):
            if not hasattr(self, 'camera') or self.camera is None:
                return blank

            try:
                rgba = self.camera.get_rgba()
            except Exception as e:
                if DEBUG_CAMERA:
                    print(f"[Camera] get_rgba() exception: {e}")
                if attempt < retries - 1:
                    self.world.step(render=True)
                    continue
                return blank

            # Validate rgba
            if rgba is None:
                if DEBUG_CAMERA:
                    print(f"[Camera] get_rgba() returned None (attempt {attempt+1})")
                if attempt < retries - 1:
                    self.world.step(render=True)
                    continue
                return blank

            if not isinstance(rgba, np.ndarray) or rgba.size == 0:
                if DEBUG_CAMERA:
                    print(f"[Camera] Invalid rgba: type={type(rgba)}, "
                          f"size={rgba.size if isinstance(rgba, np.ndarray) else 'N/A'}")
                if attempt < retries - 1:
                    self.world.step(render=True)
                    continue
                return blank

            # Handle flat array (some Isaac versions return 1D)
            if len(rgba.shape) == 1:
                try:
                    res = self.camera.get_resolution()
                    rgba = rgba.reshape((res[1], res[0], 4))
                except Exception:
                    if attempt < retries - 1:
                        self.world.step(render=True)
                        continue
                    return blank

            rgb = rgba[:, :, :3]

            # Resize if needed
            if rgb.shape[0] != self.image_size or rgb.shape[1] != self.image_size:
                try:
                    import cv2
                    rgb = cv2.resize(
                        rgb, (self.image_size, self.image_size),
                        interpolation=cv2.INTER_AREA,
                    )
                except ImportError:
                    # Fallback: nearest-neighbor via numpy
                    h, w = rgb.shape[:2]
                    row_idx = (np.arange(self.image_size) * h // self.image_size).astype(int)
                    col_idx = (np.arange(self.image_size) * w // self.image_size).astype(int)
                    rgb = rgb[np.ix_(row_idx, col_idx)]

            # Normalize to [0, 1]
            if rgb.dtype == np.uint8:
                frame = rgb.astype(np.float32) / 255.0
            elif rgb.max() > 1.0:
                frame = np.clip(rgb.astype(np.float32) / 255.0, 0, 1)
            else:
                frame = np.clip(rgb.astype(np.float32), 0, 1)

            # Check if frame is black
            if frame.max() < 0.01 and attempt < retries - 1:
                if DEBUG_CAMERA:
                    print(f"[Camera] Black frame (max={frame.max():.4f}), "
                          f"retry {attempt+1}/{retries}")
                self.world.step(render=True)
                continue

            if DEBUG_CAMERA:
                print(f"[Camera] OK: shape={frame.shape}, "
                      f"min={frame.min():.3f}, max={frame.max():.3f}, "
                      f"mean={frame.mean():.3f}")

            return frame

        return blank

    def reset(self) -> Dict[str, Any]:
        if not self.available:
            raise RuntimeError("Isaac Sim not available")

        rng = np.random.default_rng()
        self.obj_pos = np.array([
            rng.uniform(*WORKSPACE['x']),
            rng.uniform(*WORKSPACE['y']),
            0.02,
        ])
        self.tgt_pos = np.array([
            rng.uniform(*WORKSPACE['x']),
            rng.uniform(*WORKSPACE['y']),
            0.01,
        ])

        self.cube.set_world_pose(position=self.obj_pos)
        self.target_marker.set_world_pose(position=self.tgt_pos)

        # Reset state
        self.ee_pos     = np.array([0.0, 0.0, 0.2])
        self.step_count = 0
        self.oracle.reset()

        # Stabilize physics + renderer (2 steps is too few for RTX)
        for _ in range(5):
            self.world.step(render=True)

        frame = self._capture_frame(retries=CAPTURE_RETRIES)
        self.frame_buffer = [frame.copy() for _ in range(self.n_frames)]
        return self._make_obs()

    def _make_obs(self) -> Dict[str, Any]:
        return {
            'frames':     _make_frames_tensor(self.frame_buffer),
            'proprio':    _make_proprio(self.ee_pos, self.obj_pos,
                                        self.tgt_pos, self.oracle.gripper),
            'object_pos': self.obj_pos.copy(),
            'target_pos': self.tgt_pos.copy(),
            'ee_pos':     self.ee_pos.copy(),
        }

    def oracle_action(self) -> np.ndarray:
        return self.oracle(self.ee_pos, self.obj_pos, self.tgt_pos)

    def step(self, action: np.ndarray):
        # Apply action
        self.ee_pos += action[:3] * 0.08
        self.ee_pos  = _clip_to_workspace(self.ee_pos)
        self.oracle.gripper = float(action[6] > 0.5)

        # Gripper attachment
        if (self.oracle.gripper > 0.5 and
                np.linalg.norm(self.ee_pos[:2] - self.obj_pos[:2]) < 0.12):
            self.obj_pos    = self.ee_pos.copy()
            self.obj_pos[2] = max(0.0, self.ee_pos[2] - 0.02)
            self.cube.set_world_pose(position=self.obj_pos)

        self.world.step(render=True)
        self.step_count += 1

        # Read back actual poses from sim
        obj_pos_real, _ = self.cube.get_world_pose()
        tgt_pos_real, _ = self.target_marker.get_world_pose()
        self.obj_pos = np.array(obj_pos_real)
        self.tgt_pos = np.array(tgt_pos_real)

        # Capture with retry
        frame = self._capture_frame(retries=2)
        self.frame_buffer.pop(0)
        self.frame_buffer.append(frame.copy())

        dist    = np.linalg.norm(self.obj_pos[:2] - self.tgt_pos[:2])
        success = dist < SUCCESS_THRESH
        done    = success or self.step_count >= 200

        return self._make_obs(), float(success), done, {
            'success': success,
            'dist_to_target': float(dist),
            'step': self.step_count,
        }

    def close(self):
        if hasattr(self, 'sim_app'):
            self.sim_app.close()


# ─────────────────────────────────────────
# Demo Collection
# ─────────────────────────────────────────

# def collect_demonstrations(env, num_episodes: int = 100,
#                            save_dir: str = "./data/step2"):
#     save_path = Path(save_dir)
#     save_path.mkdir(parents=True, exist_ok=True)

#     env_type = type(env).__name__
#     print(f"\n{'='*55}")
#     print(f"Collecting {num_episodes} demonstrations → {save_dir}")
#     print(f"Environment: {env_type}")
#     print(f"{'='*55}")

#     success_count = 0
#     total_steps   = 0
#     all_rewards   = []

#     for ep in range(num_episodes):
#         obs          = env.reset()
#         episode_data = []
#         done         = False
#         step         = 0
#         reward       = 0.0

#         while not done and step < 200:
#             action = env.oracle_action()
#             next_obs, reward, done, info = env.step(action)

#             episode_data.append({
#                 'frames':  obs['frames'].squeeze(0).numpy(),   # (3, T, H, W)
#                 'proprio': obs['proprio'].squeeze(0).numpy(),  # (8,)
#                 'action':  action.astype(np.float32),          # (7,)
#             })

#             obs    = next_obs
#             step  += 1
#             total_steps += 1

#         if reward > 0:
#             success_count += 1
#         all_rewards.append(reward)

#         # Save episode
#         ep_file  = save_path / f"episode_{ep:04d}.npz"
#         frames   = np.stack([d['frames']  for d in episode_data])
#         proprios = np.stack([d['proprio'] for d in episode_data])
#         actions  = np.stack([d['action']  for d in episode_data])
#         np.savez_compressed(
#             ep_file,
#             frames=frames,
#             proprios=proprios,
#             actions=actions,
#             reward=np.array([reward]),
#         )

#         sr = (success_count / (ep + 1)) * 100
#         print(f"\r  Ep {ep+1:3d}/{num_episodes} | "
#               f"Success: {sr:.1f}% | Steps: {step:3d} | "
#               f"Saved {ep_file.name}     ", end="", flush=True)

#     print()

#     # Save metadata
#     meta = {
#         'env_type':       env_type,
#         'num_episodes':   num_episodes,
#         'success_rate':   success_count / num_episodes,
#         'total_steps':    total_steps,
#         'image_size':     getattr(env, 'image_size', CAMERA_H),
#         'n_frames':       getattr(env, 'n_frames', N_FRAMES),
#         'action_dim':     ACTION_DIM,
#         'proprio_dim':    PROPRIO_DIM,
#     }
#     with open(save_path / 'metadata.json', 'w') as f:
#         json.dump(meta, f, indent=2)

#     print(f"\n✅ Done | Success: {meta['success_rate']*100:.1f}% | "
#           f"Total steps: {total_steps}")
#     print(f"   Metadata: {save_path / 'metadata.json'}")
#     print(f"   Train:    python experiments/step2_sim/train_step2.py "
#           f"--dataset {save_dir}")

#     return meta
def collect_demonstrations(env, num_episodes: int = 100, save_dir: str = "./data/step2"):
    import gc
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    env_type = type(env).__name__
    print(f"\n{'='*55}")
    print(f"Collecting {num_episodes} demonstrations -> {save_dir}")
    print(f"Environment: {env_type}")
    print(f"{'='*55}")

    success_count = 0
    total_steps   = 0
    all_rewards   = []

    for ep in range(num_episodes):
        obs          = env.reset()
        episode_data = []
        done         = False
        step         = 0
        reward       = 0.0

        while not done and step < 200:
            action = env.oracle_action()
            
            # Compatible Isaac / Synthetic handling for step() return
            step_result = env.step(action)
            if len(step_result) == 4:
                next_obs, reward, done, info = step_result
            else:
                next_obs, reward, done = step_result
                info = {}

            # ----------------------------------------------------
            # Data extraction (The fix is here)
            # ----------------------------------------------------
            
            # Image extraction (expected format: C, T, H, W)
            # The environment returns 'frames' in a tensor (1, C, T, H, W)
            if isinstance(obs, dict) and 'frames' in obs:
                 # Remove batch dimension (1) for saving
                frame_data = obs['frames'].squeeze(0).numpy()
            else:
                 # Safety fallback (should not happen)
                 print("Warning: 'frames' missing from observation")
                 frame_data = np.zeros((3, N_FRAMES, CAMERA_H, CAMERA_W), dtype=np.float32)

            # Proprioception extraction
            if isinstance(obs, dict) and 'proprio' in obs:
                 # Remove batch dimension (1) for saving
                proprio_data = obs['proprio'].squeeze(0).numpy()
            else:
                 proprio_data = np.zeros(PROPRIO_DIM, dtype=np.float32)

            episode_data.append({
                'frames':  frame_data.astype(np.float16), 
                'proprio': proprio_data.astype(np.float32),  
                'action':  action.astype(np.float32),          
            })

            obs    = next_obs
            step  += 1
            total_steps += 1

        if reward > 0:
            success_count += 1
        all_rewards.append(reward)

        # Save episode
        ep_file  = save_path / f"episode_{ep:04d}.npz"
        frames   = np.stack([d['frames']  for d in episode_data])
        proprios = np.stack([d['proprio'] for d in episode_data])
        actions  = np.stack([d['action']  for d in episode_data])
        np.savez_compressed(
            ep_file,
            frames=frames,
            proprios=proprios,
            actions=actions,
            reward=np.array([reward]),
        )

        sr = (success_count / (ep + 1)) * 100
        print(f"\r  Ep {ep+1:3d}/{num_episodes} | "
              f"Success: {sr:.1f}% | Steps: {step:3d} | "
              f"Saved {ep_file.name}     ", end="", flush=True)

        # Aggressive RAM release to avoid Out of Memory
        del episode_data, frames, proprios, actions
        gc.collect()

    print()

    # Save metadata
    meta = {
        'env_type':       env_type,
        'num_episodes':   num_episodes,
        'success_rate':   success_count / num_episodes,
        'total_steps':    total_steps,
        'image_size':     getattr(env, 'image_size', CAMERA_H),
        'n_frames':       getattr(env, 'n_frames', N_FRAMES),
        'action_dim':     ACTION_DIM,
        'proprio_dim':    PROPRIO_DIM,
    }
    with open(save_path / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Done | Success: {meta['success_rate']*100:.1f}% | "
          f"Total steps: {total_steps}")
    print(f"   Metadata: {save_path / 'metadata.json'}")
    print(f"   Train:    python experiments/step2_sim/train_step2.py "
          f"--dataset {save_dir}")

    return meta

# ─────────────────────────────────────────
# Debug Visualization
# ─────────────────────────────────────────

def debug_visualize(save_dir: str, episode: int = 0, output: str = "debug_frames.png"):
    """
    Load an episode and display frames to diagnose black-frame issues.
    Correctly handles the (steps, 3, T, H, W) format.
    """
    ep_file = Path(save_dir) / f"episode_{episode:04d}.npz"
    if not ep_file.exists():
        print(f"❌ File not found: {ep_file}")
        return

    data    = np.load(ep_file)
    frames  = data['frames']   # (steps, 3, T, H, W)
    actions = data['actions']
    reward  = data['reward']

    print(f"Episode {episode}: frames.shape={frames.shape}, "
          f"actions.shape={actions.shape}, reward={reward}")
    print(f"  Frame stats: min={frames.min():.4f}, max={frames.max():.4f}, "
          f"mean={frames.mean():.4f}")

    if frames.max() < 0.01:
        print("  ⚠️  ALL FRAMES ARE BLACK — camera capture failed during collection")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Show frames from 4 evenly spaced time steps
    n_steps = len(frames)
    indices = [0, n_steps // 3, 2 * n_steps // 3, n_steps - 1]
    n_temporal = frames.shape[2]  # T

    fig, axes = plt.subplots(len(indices), n_temporal,
                             figsize=(4 * n_temporal, 4 * len(indices)))
    fig.suptitle(f'Episode {episode} — reward={float(reward):.1f}\n'
                 f'frames shape: {frames.shape}', fontsize=14)

    for row, step_idx in enumerate(indices):
        for t in range(n_temporal):
            ax = axes[row, t] if len(indices) > 1 else axes[t]
            # frames[step_idx] is (3, T, H, W)
            # We want channel-last: (H, W, 3)
            img = frames[step_idx, :, t, :, :].transpose(1, 2, 0)
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.set_title(f'Step {step_idx}, T={t}\n'
                         f'max={img.max():.3f}', fontsize=9)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(output, dpi=120, bbox_inches='tight')
    print(f"  Saved → {output}")


# ─────────────────────────────────────────
# Policy Evaluation
# ─────────────────────────────────────────

@torch.no_grad()
def evaluate_policy(env, model: torch.nn.Module, n_episodes: int,
                    device: torch.device) -> Dict[str, float]:
    model.eval()
    successes  = 0
    latencies  = []
    steps_list = []

    print(f"\n{'='*55}")
    print(f"Evaluating FluidVLA — {n_episodes} episodes")
    print(f"{'='*55}")

    for ep in range(n_episodes):
        obs  = env.reset()
        done = False
        step = 0

        while not done and step < 200:
            frames  = obs['frames'].to(device)
            proprio = obs['proprio'].to(device)

            t0  = time.perf_counter()
            out = model(frames, proprio)
            latencies.append((time.perf_counter() - t0) * 1000)

            action = out['actions'].cpu().numpy()[0]

            # Track PDE steps if available
            if 'info' in out and out['info']:
                avg_steps = sum(
                    i.get('steps_used', 0) for i in out['info']
                ) / max(len(out['info']), 1)
                steps_list.append(avg_steps)

            obs, _, done, info = env.step(action)
            step += 1
            if info['success']:
                successes += 1
                break

        if (ep + 1) % 10 == 0:
            sr  = successes / (ep + 1) * 100
            lat = np.mean(latencies[-50:]) if latencies else 0
            stp = np.mean(steps_list[-50:]) if steps_list else 0
            print(f"  Ep {ep+1:3d}/{n_episodes} | "
                  f"Success: {sr:.1f}% | "
                  f"Latency: {lat:.1f}ms | "
                  f"PDE steps: {stp:.1f}/12")

    results = {
        'success_rate':    successes / max(n_episodes, 1),
        'latency_mean_ms': float(np.mean(latencies)) if latencies else 0,
        'latency_p95_ms':  float(np.percentile(latencies, 95)) if latencies else 0,
        'avg_pde_steps':   float(np.mean(steps_list)) if steps_list else 0,
        'n_episodes':      n_episodes,
    }

    sr  = results['success_rate'] * 100
    lat = results['latency_mean_ms']
    stp = results['avg_pde_steps']
    print(f"\n{'='*55}")
    print(f"  Success rate : {sr:.1f}%  {'✅' if sr >= 70 else '❌'}  (target >70%)")
    print(f"  Latency mean : {lat:.1f}ms  {'✅' if lat <= 50 else '❌'}  (target <50ms)")
    print(f"  Latency p95  : {results['latency_p95_ms']:.1f}ms")
    if steps_list:
        print(f"  PDE steps    : {stp:.1f}/12  "
              f"{'✅' if stp <= 6 else '⏳'}  (target <6 on calm)")

    return results


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FluidVLA Step 2 — Pick & Place Environment"
    )
    parser.add_argument('--mode', default='synthetic',
                        choices=['collect', 'eval', 'synthetic', 'debug'])
    parser.add_argument('--episodes',   default=500,  type=int)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--save_dir',   default='./data/step2')
    parser.add_argument('--image_size', default=224,  type=int)
    parser.add_argument('--show_gui',   action='store_true',
                        help='Show Isaac Sim 3D window (disables headless mode)')
    parser.add_argument('--debug_ep',   default=0,    type=int,
                        help='Episode index for debug visualization')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"FluidVLA — Step 2 Pick & Place")
    print(f"  Mode:   {args.mode}")
    print(f"  Device: {device}")
    print(f"  Dir:    {args.save_dir}")
    print(f"  GUI:    {args.show_gui}")
    print(f"{'='*60}")

    # ── Debug mode: just visualize existing data ──
    if args.mode == 'debug':
        debug_visualize(args.save_dir, args.debug_ep,
                        output=str(Path(args.save_dir) / 'debug_frames.png'))
        return

    # ── Create environment ──
    headless = not args.show_gui
    
    if args.mode == 'synthetic':
        env = SyntheticPickPlace(args.image_size, N_FRAMES)
    else:
        env = IsaacPickPlace(headless, args.image_size, N_FRAMES)
        if not env.available:
            print("[Fallback] Using synthetic environment")
            env = SyntheticPickPlace(args.image_size, N_FRAMES)

    # ── Collect demonstrations ──
    if args.mode in ('collect', 'synthetic'):
        collect_demonstrations(env, args.episodes, args.save_dir)

    # ── Evaluate policy ──
    elif args.mode == 'eval':
        assert args.checkpoint, "Need --checkpoint for eval mode"
        model = FluidBotVLA(
            image_size=args.image_size,
            in_channels=3,
            d_model=128,
            n_layers=4,
            patch_size=16,
            action_dim=ACTION_DIM,
            proprio_dim=PROPRIO_DIM,
            n_frames=N_FRAMES,
        ).to(device)

        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        print(f"Loaded checkpoint: {args.checkpoint}")

        results = evaluate_policy(env, model, args.episodes, device)

        out_file = Path(args.save_dir) / 'eval_results.json'
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved → {out_file}")

    # ── Cleanup ──
    if hasattr(env, 'close'):
        env.close()


if __name__ == '__main__':
    main()