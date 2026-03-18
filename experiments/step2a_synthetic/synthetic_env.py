"""
synthetic_env.py — Step 2a: Synthetic Pick & Place Environment
────────────────────────────────────────────────────────────────────
Light procedural environment (no real physics).
Produces exactly the same observation format as Isaac Sim.

ROLE IN THE ROADMAP:
  Step 2a — Validate the complete VLA architecture before Isaac Sim.
  If it works here → pipeline OK → move to step2b/2c with Isaac.

DATA PRODUCED → data/step2a_synthetic/
CHECKPOINTS   → checkpoints/step2a_synthetic/

OBS FORMAT (identical to step2b/2c/3):
  frames  : torch.Tensor (1, 3, T, H, W)   float32  [0,1]
  proprio : torch.Tensor (1, 8)             float32
  object_pos, target_pos, ee_pos : np.ndarray (3,)
"""

import numpy as np
import torch
from typing import Dict, Any, List

# ─── Shared constants (same as step2c_isaac) ───────────────────────────
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


def clip_to_workspace(pos: np.ndarray) -> np.ndarray:
    return np.clip(
        pos,
        [WORKSPACE['x'][0], WORKSPACE['y'][0], 0.0],
        [WORKSPACE['x'][1], WORKSPACE['y'][1], WORKSPACE['z'][1]],
    )


def make_frames_tensor(frame_buffer: List[np.ndarray]) -> torch.Tensor:
    """List of (H,W,3) → (1, 3, T, H, W) tensor."""
    frames = np.stack(frame_buffer, axis=0)        # (T, H, W, 3)
    return (
        torch.from_numpy(frames)
        .permute(3, 0, 1, 2)   # (3, T, H, W)
        .unsqueeze(0)          # (1, 3, T, H, W)
        .float()
    )


def make_proprio(ee_pos, obj_pos, tgt_pos, gripper) -> torch.Tensor:
    """→ (1, 8)"""
    v = np.concatenate([ee_pos, obj_pos[:2], tgt_pos[:2], [gripper]]).astype(np.float32)
    return torch.from_numpy(v).unsqueeze(0)


# ─── Shared Oracle (reused by step2c) ──────────────────────────────────

class OraclePickPlace:
    """
    Proportional 5-phase controller:
      0 — XY approach above the object
      1 — descent + close gripper
      2 — ascent
      3 — XY movement to target
      4 — descent + open gripper
    """
    def __init__(self):
        self.phase   = 0
        self.gripper = 0.0

    def reset(self):
        self.phase   = 0
        self.gripper = 0.0

    @staticmethod
    def _prop(current, target, gain=5.0):
        return np.clip((target - current) * gain, -1, 1)

    def __call__(self, ee, obj, tgt) -> np.ndarray:
        a = np.zeros(ACTION_DIM, dtype=np.float32)

        if self.phase == 0:
            a[:2] = self._prop(ee[:2], obj[:2])
            a[2]  = np.clip((0.15 - ee[2]) * 3.0, -1, 1)
            a[6]  = 0.0
            if np.linalg.norm(ee[:2] - obj[:2]) < 0.05:
                self.phase = 1

        elif self.phase == 1:
            a[:3] = self._prop(ee[:3], obj[:3])
            a[6]  = 1.0
            if ee[2] < obj[2] + 0.03:
                self.gripper = 1.0
                self.phase   = 2

        elif self.phase == 2:
            a[2] = 1.0
            a[6] = 1.0
            if ee[2] > 0.15:
                self.phase = 3

        elif self.phase == 3:
            a[:2] = self._prop(ee[:2], tgt[:2])
            a[6]  = 1.0
            if np.linalg.norm(ee[:2] - tgt[:2]) < 0.05:
                self.phase = 4

        elif self.phase == 4:
            place_z = tgt[2] + 0.02
            a[2] = np.clip((place_z - ee[2]) * 5.0, -1, 1)
            a[6] = 0.0
            self.gripper = 0.0

        return np.clip(a, -1.0, 1.0)


# ─── Synthetic environment ───────────────────────────────────────────────

class SyntheticPickPlace:
    """
    Pick & Place with procedural rendering (colored circles on gray background).
    Obs format identical to IsaacPickPlace to ensure compatibility.

    red cube   = object to grasp
    green circle = drop target
    blue circle  = effector (gripper)
    """

    def __init__(self, image_size: int = 224, n_frames: int = N_FRAMES, seed: int = 42):
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
        """Procedural top-down rendering. Returns (H, W, 3) float32 [0,1]."""
        H = W = self.image_size
        frame = np.full((H, W, 3), 0.75, dtype=np.float32)  # light gray background

        def world_to_pixel(pos):
            px = int((pos[0] - WORKSPACE['x'][0]) / (WORKSPACE['x'][1] - WORKSPACE['x'][0]) * W)
            py = int((pos[1] - WORKSPACE['y'][0]) / (WORKSPACE['y'][1] - WORKSPACE['y'][0]) * H)
            return np.clip(px, 0, W - 1), np.clip(py, 0, H - 1)

        def draw_circle(img, pos, color, radius=12):
            cx, cy = world_to_pixel(pos)
            y_idx, x_idx = np.ogrid[:H, :W]
            mask = (x_idx - cx) ** 2 + (y_idx - cy) ** 2 < radius ** 2
            img[mask] = color

        draw_circle(frame, self.target_pos, [0.1, 0.8, 0.1], radius=16)  # green = target
        draw_circle(frame, self.object_pos, [0.9, 0.1, 0.1], radius=12)  # red = cube
        draw_circle(frame, self.ee_pos,     [0.1, 0.1, 0.9], radius=6)   # blue = gripper
        return frame

    def reset(self) -> Dict[str, Any]:
        self.object_pos   = self._random_pos()
        self.target_pos   = self._random_pos()
        self.ee_pos       = np.array([0.0, 0.0, 0.2])
        self.step_count   = 0
        self.oracle.reset()
        frame             = self._render_frame()
        self.frame_buffer = [frame.copy() for _ in range(self.n_frames)]
        return self._get_obs()

    def _get_obs(self) -> Dict[str, Any]:
        return {
            'frames'    : make_frames_tensor(self.frame_buffer),
            'proprio'   : make_proprio(self.ee_pos, self.object_pos,
                                       self.target_pos, self.oracle.gripper),
            'object_pos': self.object_pos.copy(),
            'target_pos': self.target_pos.copy(),
            'ee_pos'    : self.ee_pos.copy(),
        }

    def oracle_action(self) -> np.ndarray:
        return self.oracle(self.ee_pos, self.object_pos, self.target_pos)

    def step(self, action: np.ndarray):
        self.ee_pos += action[:3] * 0.08
        self.ee_pos  = clip_to_workspace(self.ee_pos)
        self.oracle.gripper = float(action[6] > 0.5)

        # Gripper attachment logic
        if (self.oracle.gripper > 0.5 and
                np.linalg.norm(self.ee_pos[:2] - self.object_pos[:2]) < 0.12):
            self.object_pos    = self.ee_pos.copy()
            self.object_pos[2] = max(0.0, self.ee_pos[2] - 0.02)

        frame = self._render_frame()
        self.frame_buffer.pop(0)
        self.frame_buffer.append(frame.copy())
        self.step_count += 1

        dist    = np.linalg.norm(self.object_pos[:2] - self.target_pos[:2])
        success = dist < SUCCESS_THRESH
        done    = success or self.step_count >= 200

        return self._get_obs(), float(success), done, {
            'success': success, 'dist_to_target': float(dist), 'step': self.step_count
        }