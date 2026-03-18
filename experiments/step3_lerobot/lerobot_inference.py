"""
lerobot_inference.py — Step 3: Real Hardware (SO-101 LeRobot) — V2

V2 changes vs original:
  1. Loads norm_stats from checkpoint or JSON for proper denormalization.
  2. Delta-action mode: model predicts normalized deltas, inference converts to absolute.
  3. Action chunking execution: model predicts chunk_size actions, executed sequentially.
  4. Compatible with V2 spatial-pooling FluidBotVLA architecture.
  5. Keeps full backward compatibility with V1 checkpoints.

Pipeline:
  Camera frame(s)
       │
  [Frame Buffer]   — maintain rolling window of N_FRAMES
       │
  [FluidBotVLA V2] — spatial-aware diffusion vision → action chunk
       │
  [Denormalize]    — undo dataset normalization
       │
  [Delta → Abs]    — convert delta-action to absolute joint target
       │
  [Action Filter]  — smooth + safety clip
       │
  [SO-101 Motors]  — execute joint positions
"""

import os
import sys
import time
import threading
import json
import argparse
import numpy as np
from pathlib import Path
from collections import deque
from typing import Optional, List, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from fluidvla.core import FluidBotVLA


def maybe_add_lerobot_to_syspath(lerobot_root: Optional[str]) -> Optional[Path]:
    if not lerobot_root:
        return None
    root = Path(lerobot_root)
    src_dir = root / 'src'
    if src_dir.exists():
        sys.path.insert(0, str(src_dir))
        return src_dir
    return None


def apply_runtime_overrides(model, override_epsilon=None, override_max_steps=None):
    """Apply runtime adaptive-compute overrides to all fluid layers."""
    layers = getattr(getattr(model, 'visual', None), 'fluid_layers', [])
    for layer in layers:
        if override_epsilon is not None:
            layer.epsilon = float(override_epsilon)
        if override_max_steps is not None:
            layer.max_steps = int(override_max_steps)


# ─────────────────────────────────────────
# Action Denormalization (V2)
# ─────────────────────────────────────────

class ActionDenormalizer:
    """Converts model output (normalized, possibly delta) back to absolute joint degrees.

    Handles three cases based on checkpoint config:
      1. V1 (no norm_stats): pass-through, model outputs raw absolute positions.
      2. V2 absolute: denormalize with action_mean/action_std.
      3. V2 delta: denormalize with delta_mean/delta_std, then add current proprio.
    """

    def __init__(self, norm_stats: Optional[dict] = None):
        self.active = norm_stats is not None
        self.is_delta = False

        if self.active:
            self.is_delta = norm_stats.get("delta_actions", False)

            if self.is_delta:
                self.target_mean = np.array(norm_stats["delta_mean"], dtype=np.float32)
                self.target_std = np.array(norm_stats["delta_std"], dtype=np.float32)
            else:
                self.target_mean = np.array(norm_stats["action_mean"], dtype=np.float32)
                self.target_std = np.array(norm_stats["action_std"], dtype=np.float32)

            self.proprio_mean = np.array(norm_stats["proprio_mean"], dtype=np.float32)
            self.proprio_std = np.array(norm_stats["proprio_std"], dtype=np.float32)

    def __call__(self, raw_model_output: np.ndarray, current_joints: np.ndarray, reference_joints: Optional[np.ndarray] = None) -> np.ndarray:
        """Convert model output to absolute joint positions.

        Args:
            raw_model_output: (action_dim,) or (chunk_size, action_dim) from model
            current_joints: (action_dim,) current joint readings in degrees (for absolute mode)
            reference_joints: (action_dim,) base joint positions (e.g. last target) to add deltas to. Defaults to current_joints.
        Returns:
            Absolute joint positions in degrees, same shape as input.
        """
        if not self.active:
            return raw_model_output
        
        if reference_joints is None:
            reference_joints = current_joints

        # Denormalize: undo the (x - mean) / std from converter
        denormed = raw_model_output * self.target_std + self.target_mean

        if self.is_delta:
            # Delta mode: denormed is displacement in degrees, add to reference position
            if denormed.ndim == 1:
                return reference_joints + denormed
            else:
                # Chunk: cumulative application of deltas
                result = np.zeros_like(denormed)
                pos = reference_joints.copy()
                for i in range(denormed.shape[0]):
                    pos = pos + denormed[i]
                    result[i] = pos
                return result
        else:
            # Absolute mode: denormed is already the target position
            return denormed

    def normalize_proprio(self, joints: np.ndarray) -> np.ndarray:
        """Normalize proprioception to match training distribution."""
        if not self.active:
            return joints
        return (joints - self.proprio_mean) / self.proprio_std


# ─────────────────────────────────────────
# Camera Interface
# ─────────────────────────────────────────

try:
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera, OpenCVCameraConfig
except ImportError:
    OpenCVCamera, OpenCVCameraConfig = None, None

class CameraStream:
    """USB camera interface for SO-101."""

    def __init__(
        self,
        device_id  : int = 0,
        width      : int = 224,
        height     : int = 224,
        fps        : int = 30,
        buffer_size: int = 4,
        backend    : str = 'auto',
        fourcc     : Optional[str] = None,
        allow_synthetic: bool = True,
    ):
        self.width       = width
        self.height      = height
        self.buffer_size = buffer_size
        self.allow_synthetic = allow_synthetic
        self.frame_buffer = deque(maxlen=buffer_size)
        self.available = False
        self.camera = None

        if OpenCVCameraConfig is None:
            print("[Camera] OpenCVCamera not installed/found - using synthetic frames")
            self.available = False
            return

        try:
            # Construct the config object used by Lerobot
            # We explicitly do NOT pass width/height here because Windows/dshow
            # throws an exception if the hardware doesn't natively support exactly 224x224
            cam_config = OpenCVCameraConfig(
                index_or_path=device_id,
                fps=fps,
                fourcc=fourcc if fourcc and len(fourcc) == 4 else None
            )
            self.camera = OpenCVCamera(cam_config)
            self.camera.connect(warmup=True)
            self.available = True
            print(f"[Camera] device={device_id} connected via async OpenCVCamera: size={width}x{height} fps={fps} fourcc={fourcc}")
        except Exception as e:
            print(f"[Camera] Failed to connect using LeRobot OpenCVCamera: {e}")
            if allow_synthetic:
                print("[Camera] Falling back to synthetic frames")
            self.available = False

    def read_frame(self) -> Optional[np.ndarray]:
        """Read one frame, return (H, W, 3) uint8 or None."""
        if not self.available:
            if not self.allow_synthetic:
                raise RuntimeError("Camera is not available and synthetic frames are disabled")
            # Generate moving synthetic gradient if camera fails
            t = time.time()
            H, W = self.height, self.width
            frame = np.zeros((H, W, 3), dtype=np.uint8)
            frame[:, :, 0] = (np.sin(np.linspace(0, 2*np.pi, W) + t) * 127 + 127).astype(np.uint8)
            frame[:, :, 1] = (np.cos(np.linspace(0, 2*np.pi, H) + t * 0.7)[:, None] * 127 + 127).astype(np.uint8)
            return frame

        # Use non-blocking read_latest to fetch the background frame instantly
        try:
            frame = self.camera.read_latest()
            return frame
        except Exception as e:
            print(f"[WARN] Camera peek failed: {e}")
            return None

    def get_tensor_batch(self, device='cpu') -> Optional[torch.Tensor]:
        """Read a new frame, add to buffer, return full buffer as tensor.
        Returns: (1, 3, T, H, W) or None if buffer not full yet.
        """
        frame = self.read_frame()
        if frame is None:
            return None

        # Resize if dims do not match what model expects
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            import cv2
            frame = cv2.resize(frame, (self.width, self.height))

        # Match training preprocessing: scale to [0, 1] only.
        frame_t = torch.tensor(frame.copy(), dtype=torch.float32).permute(2, 0, 1) / 255.0

        self.frame_buffer.append(frame_t)

        if len(self.frame_buffer) < self.buffer_size:
            return None

        frames = torch.stack(list(self.frame_buffer), dim=0)  # (T, C, H, W)
        frames = frames.permute(1, 0, 2, 3)                   # (C, T, H, W)
        return frames.unsqueeze(0).to(device)                  # (1, C, T, H, W)

    def close(self):
        if self.available and self.camera is not None:
            self.camera.disconnect()


# ─────────────────────────────────────────
# SO-101 Robot Interface
# ─────────────────────────────────────────

class SO101Interface:
    """Interface to the SO-101 6-DOF robot arm."""

    JOINT_LIMITS = {
        'shoulder_pan' : (-120.0, 120.0),
        'shoulder_lift': (-90.0, 90.0),
        'elbow_flex'   : (-120.0, 120.0),
        'wrist_flex'   : (-90.0, 90.0),
        'wrist_roll'   : (-180.0, 180.0),
        'gripper'      : (0.0, 100.0),
    }

    MAX_DELTA = 3.0

    JOINT_ORDER = [
        'shoulder_pan',
        'shoulder_lift',
        'elbow_flex',
        'wrist_flex',
        'wrist_roll',
        'gripper',
    ]

    def __init__(self, port: str = '/dev/ttyUSB0', mock: bool = False,
                 lerobot_root: Optional[str] = None, robot_id: Optional[str] = None,
                 max_delta: Optional[float] = None):
        self.mock = mock
        self.current_joints = np.zeros(6, dtype=np.float32)
        self.connected = False
        self.robot = None
        self.lerobot_src = maybe_add_lerobot_to_syspath(lerobot_root)
        if max_delta is not None:
            self.MAX_DELTA = float(max_delta)

        if not mock:
            try:
                from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
                from lerobot.robots.so_follower.so_follower import SOFollower

                robot_cfg = SOFollowerRobotConfig(
                    port=port,
                    id=robot_id,
                    cameras={},
                    max_relative_target=None,  # We handle safety clamping inside our inference loop directly via args.max_delta
                    use_degrees=True,
                )
                self.robot = SOFollower(robot_cfg)
                print(f"[SO-101] Calibration file: {self.robot.calibration_fpath}")
                self.robot.connect(calibrate=False)
                self.connected = True
                print(f"[SO-101] Connected on {port} via LeRobot")
            except Exception as e:
                print(f"[SO-101] Connection failed: {e}")
                print("[SO-101] Falling back to mock mode")
                self.mock = True
        else:
            print("[SO-101] Mock mode - no hardware commands sent")

    def get_joint_positions(self) -> np.ndarray:
        """Read current joint angles in degrees. Shape: (6,)"""
        if self.mock or not self.connected:
            return self.current_joints.copy()

        obs = self.robot.get_observation()
        joints = np.array([obs[f'{joint}.pos'] for joint in self.JOINT_ORDER], dtype=np.float32)
        self.current_joints = joints.copy()
        return joints

    def set_joint_positions(self, targets: np.ndarray) -> np.ndarray:
        """Send joint position targets with safety limits."""
        targets = np.array(targets, dtype=np.float32)

        for i, (lo, hi) in enumerate(self.JOINT_LIMITS.values()):
            targets[i] = np.clip(targets[i], lo, hi)

        # Let the FluidVLA logic handle accumulated delta constraints via args.max_delta bounds
        # Remove inner constraints like self.MAX_DELTA to let it traverse fast if approved.

        if not self.mock and self.connected:
            action = {f'{joint}.pos': float(targets[idx]) for idx, joint in enumerate(self.JOINT_ORDER)}
            sent_action = self.robot.send_action(action)
            targets = np.array([sent_action[f'{joint}.pos'] for joint in self.JOINT_ORDER], dtype=np.float32)

        self.current_joints = targets.copy()
        return targets

    def emergency_stop(self):
        print("[WARN] EMERGENCY STOP")
        if not self.mock and self.connected:
            try:
                self.robot.bus.disable_torque()
            except:
                pass

    def disconnect(self):
        if not self.mock and self.connected:
            self.robot.disconnect()
            print("[SO-101] Disconnected")


# ─────────────────────────────────────────
# Action Post-Processing
# ─────────────────────────────────────────

class ActionFilter:
    """EMA smoothing for predicted actions."""

    def __init__(self, action_dim: int = 6, alpha: float = 0.3):
        self.alpha   = alpha
        self.history = None

    def __call__(self, action: np.ndarray) -> np.ndarray:
        if self.history is None:
            self.history = action.copy()

        self.history = self.alpha * action + (1 - self.alpha) * self.history
        return self.history.copy()


# ─────────────────────────────────────────
# Latency Benchmark
# ─────────────────────────────────────────

def benchmark_latency(model, device, n_runs=100):
    print("\n" + "=" * 60)
    print("LATENCY BENCHMARK - Target: <33ms for 30fps")
    print("=" * 60)

    model.eval()
    proprio_dim = getattr(getattr(model, 'action_head', None), 'proprio_dim', 0)

    with torch.inference_mode():
        dummy = torch.randn(1, 3, 4, 224, 224, device=device)
        prop = torch.randn(1, proprio_dim, device=device) if proprio_dim > 0 else None
        for _ in range(10):
            _ = model(dummy, prop)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    latencies = []
    steps_used = []

    with torch.inference_mode():
        for _ in range(n_runs):
            frames = torch.randn(1, 3, 4, 224, 224, device=device)
            proprio = torch.randn(1, proprio_dim, device=device) if proprio_dim > 0 else None

            t0 = time.perf_counter()
            out = model(frames, proprio)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            latencies.append((t1 - t0) * 1000)
            steps_used.append(sum(i['steps_used'] for i in out['info']) / len(out['info']))

    latencies  = np.array(latencies)
    steps_used = np.array(steps_used)

    print(f"  Latency (ms): mean={latencies.mean():.1f} | "
          f"p50={np.percentile(latencies, 50):.1f} | "
          f"p95={np.percentile(latencies, 95):.1f} | "
          f"p99={np.percentile(latencies, 99):.1f}")
    print(f"  Avg integration steps: {steps_used.mean():.1f}")
    print(f"  Effective FPS: {1000/latencies.mean():.1f}")

    target_ms = 33.0
    if latencies.mean() < target_ms:
        print(f"  [OK] Meets 30fps target (<{target_ms}ms)")
    else:
        print(f"  [WARN] Exceeds {target_ms}ms - consider: fewer layers, smaller d_model, or TorchScript export")

    return latencies


# ─────────────────────────────────────────
# Diagnostic Replay (feeds training data through model)
# ─────────────────────────────────────────

def run_diagnose(args):
    """Replay training .npz episodes through the model to verify predictions.

    This checks whether the model produces reasonable outputs with training data.
    If outputs are garbage here, the model is undertrained.
    If outputs are good here but bad with live camera, there's a preprocessing mismatch.
    """
    import glob

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if not args.checkpoint or not os.path.exists(args.checkpoint):
        print("[ERROR] --checkpoint required for diagnose mode")
        return

    ckpt = torch.load(args.checkpoint, map_location=device)
    model_cfg = ckpt.get('config', {})
    model = FluidBotVLA(**model_cfg).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  Config: {json.dumps(model_cfg, indent=2)}")

    norm_stats = ckpt.get('norm_stats', None)
    if norm_stats is None and args.norm_stats:
        with open(args.norm_stats, 'r') as f:
            norm_stats = json.load(f)

    if norm_stats is None:
        print("[ERROR] No norm_stats found in checkpoint or --norm_stats")
        return

    denorm = ActionDenormalizer(norm_stats)
    print(f"  delta_actions: {denorm.is_delta}")
    print(f"  delta_mean: {np.round(denorm.target_mean, 3)}")
    print(f"  delta_std:  {np.round(denorm.target_std, 3)}")
    print(f"  proprio_mean: {np.round(denorm.proprio_mean, 3)}")
    print(f"  proprio_std:  {np.round(denorm.proprio_std, 3)}")

    # Find training data
    data_dir = args.dataset
    if not data_dir:
        print("[ERROR] --dataset required for diagnose mode (path to training data dir with .npz files)")
        return

    npz_files = sorted(glob.glob(os.path.join(data_dir, "episode_*.npz")))
    if not npz_files:
        print(f"[ERROR] No episode_*.npz files in {data_dir}")
        return

    print(f"\nFound {len(npz_files)} episodes in {data_dir}")

    # ── Test 1: Frame statistics from training data ──
    print("\n" + "=" * 60)
    print("TEST 1: Training data frame statistics")
    print("=" * 60)

    ep0 = np.load(npz_files[0])
    frames = ep0['frames']  # (N, C, T, H, W)
    proprios = ep0['proprios']  # (N, 6) - NORMALIZED
    actions = ep0['actions']  # (N, 6) - NORMALIZED delta targets

    print(f"  frames shape:  {frames.shape} dtype={frames.dtype}")
    print(f"  frames range:  [{frames.min():.4f}, {frames.max():.4f}]")
    print(f"  frames mean:   {frames.mean():.4f}")
    print(f"  frames std:    {frames.std():.4f}")
    print(f"  proprios shape: {proprios.shape} (NORMALIZED)")
    print(f"  proprios range: [{proprios.min():.3f}, {proprios.max():.3f}]")
    print(f"  proprios mean:  {np.round(proprios.mean(axis=0), 3)}")
    print(f"  actions shape:  {actions.shape} (NORMALIZED deltas)")
    print(f"  actions range:  [{actions.min():.3f}, {actions.max():.3f}]")
    print(f"  actions mean:   {np.round(actions.mean(axis=0), 3)}")
    print(f"  actions std:    {np.round(actions.std(axis=0), 3)}")

    if 'proprios_raw' in ep0:
        proprios_raw = ep0['proprios_raw']
        print(f"  proprios_raw mean: {np.round(proprios_raw.mean(axis=0), 3)}")

    # ── Test 2: Feed training data through model ──
    print("\n" + "=" * 60)
    print("TEST 2: Model predictions on training data")
    print("=" * 60)

    n_test = min(20, len(frames))
    test_indices = np.linspace(0, len(frames) - 1, n_test, dtype=int)

    all_raw_outputs = []
    all_expected = []
    all_errors = []

    with torch.inference_mode():
        for idx in test_indices:
            frame_t = torch.from_numpy(frames[idx]).float().unsqueeze(0).to(device)
            proprio_t = torch.from_numpy(proprios[idx]).float().unsqueeze(0).to(device)

            out = model(frame_t, proprio_t)
            raw_output = out['actions'].cpu().numpy().squeeze()
            expected = actions[idx]

            error = np.abs(raw_output - expected)
            all_raw_outputs.append(raw_output)
            all_expected.append(expected)
            all_errors.append(error)

            avg_steps = sum(i['steps_used'] for i in out['info']) / len(out['info'])
            print(f"  Step {idx:4d} | "
                  f"pred={np.round(raw_output, 3)} | "
                  f"expected={np.round(expected, 3)} | "
                  f"err={np.round(error, 3)} | "
                  f"PDE steps={avg_steps:.1f}")

    all_raw_outputs = np.array(all_raw_outputs)
    all_expected = np.array(all_expected)
    all_errors = np.array(all_errors)

    print(f"\n  Mean absolute error per joint: {np.round(all_errors.mean(axis=0), 4)}")
    print(f"  Mean abs error overall:        {all_errors.mean():.4f}")
    print(f"  Predicted range:  [{all_raw_outputs.min():.3f}, {all_raw_outputs.max():.3f}]")
    print(f"  Expected range:   [{all_expected.min():.3f}, {all_expected.max():.3f}]")
    print(f"  Predicted mean:   {np.round(all_raw_outputs.mean(axis=0), 3)}")
    print(f"  Predicted std:    {np.round(all_raw_outputs.std(axis=0), 3)}")

    # ── Test 3: Denormalized delta magnitudes ──
    print("\n" + "=" * 60)
    print("TEST 3: Denormalized delta magnitudes (what motors would see)")
    print("=" * 60)

    for i, idx in enumerate(test_indices[:5]):
        raw = all_raw_outputs[i]
        denormed_delta = raw * denorm.target_std + denorm.target_mean
        print(f"  Step {idx:4d} | "
              f"raw_norm={np.round(raw, 3)} → "
              f"delta_deg={np.round(denormed_delta, 3)}")

    # ── Test 4: Check if live camera matches training distribution ──
    print("\n" + "=" * 60)
    print("TEST 4: Live camera frame comparison (if camera available)")
    print("=" * 60)

    try:
        cam = CameraStream(
            device_id=args.camera_id, width=224, height=224,
            fps=args.camera_fps, fourcc=args.camera_fourcc,
            allow_synthetic=True,
        )
        frame = cam.read_frame()
        if frame is not None:
            print(f"  Live frame shape: {frame.shape} dtype={frame.dtype}")
            print(f"  Live frame range: [{frame.min()}, {frame.max()}]")
            frame_f = frame.astype(np.float32) / 255.0
            print(f"  After /255:  range=[{frame_f.min():.4f}, {frame_f.max():.4f}] "
                  f"mean={frame_f.mean():.4f} std={frame_f.std():.4f}")
            print(f"  Training:    range=[{frames.min():.4f}, {frames.max():.4f}] "
                  f"mean={frames.mean():.4f} std={frames.std():.4f}")

            diff_mean = abs(frame_f.mean() - frames.mean())
            diff_std = abs(frame_f.std() - frames.std())
            if diff_mean > 0.1 or diff_std > 0.1:
                print(f"  [WARN] Significant distribution shift: "
                      f"mean_diff={diff_mean:.3f} std_diff={diff_std:.3f}")
            else:
                print(f"  [OK] Frame distributions roughly match")
        cam.close()
    except Exception as e:
        print(f"  Camera not available: {e}")

    print("\n" + "=" * 60)
    print("DIAGNOSIS SUMMARY")
    print("=" * 60)
    mae = all_errors.mean()
    pred_range = all_raw_outputs.max() - all_raw_outputs.min()
    if mae < 0.3 and pred_range > 0.1:
        print("  Model appears to produce reasonable predictions on training data.")
        print("  If live inference is bad, the issue is likely a preprocessing mismatch")
        print("  between camera frames and training frames.")
    elif pred_range < 0.05:
        print("  [PROBLEM] Model outputs near-constant values (mode collapse).")
        print("  The model likely needs more training or architecture changes.")
    else:
        print(f"  [PROBLEM] Model MAE={mae:.4f} on training data.")
        print("  If MAE > 0.5, the model is likely undertrained.")
        print("  Consider training for more epochs or checking the training pipeline.")


# ─────────────────────────────────────────
# Main Inference Loop (V2)
# ─────────────────────────────────────────

def run_inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    norm_stats = None
    chunk_size = 1

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model_cfg = ckpt.get('config', {})
        chunk_size = model_cfg.get('chunk_size', 1)

        model = FluidBotVLA(**model_cfg).to(device)
        model.load_state_dict(ckpt['model'])
        apply_runtime_overrides(model, args.override_epsilon, args.override_max_steps)
        print(f"Loaded checkpoint: {args.checkpoint}")
        print(f"  Config: {json.dumps(model_cfg, indent=2)}")

        # Load norm stats from checkpoint or external file
        norm_stats = ckpt.get('norm_stats', None)
        if norm_stats is None and args.norm_stats:
            with open(args.norm_stats, 'r') as f:
                norm_stats = json.load(f)
            print(f"  Loaded external norm_stats: {args.norm_stats}")

        if norm_stats:
            print(f"  Normalization: active | delta_actions={norm_stats.get('delta_actions', False)}")
        else:
            print(f"  Normalization: none (V1 mode)")

        print(
            f"  Runtime overrides: epsilon={args.override_epsilon or model_cfg.get('epsilon')} | "
            f"max_steps={args.override_max_steps or model_cfg.get('max_steps')}"
        )
    else:
        print("No checkpoint found - using untrained model (for testing only)")
        model = FluidBotVLA(
            image_size=224, in_channels=3, d_model=128, n_layers=3,
            patch_size=16, action_dim=6, proprio_dim=6, n_frames=4,
            spatial_pool_size=4, chunk_size=1,
        ).to(device)
        apply_runtime_overrides(model, args.override_epsilon, args.override_max_steps)

    model.eval()

    # Setup denormalizer
    denorm = ActionDenormalizer(norm_stats)

    # Latency benchmark
    benchmark_latency(model, device)

    # Hardware
    camera = CameraStream(
        device_id=args.camera_id,
        width=224,
        height=224,
        fps=args.camera_fps,
        buffer_size=getattr(model, 'n_frames', 4),
        backend=args.camera_backend,
        fourcc=args.camera_fourcc,
        allow_synthetic=args.mock_robot,
    )
    robot = SO101Interface(
        port=args.robot_port,
        mock=args.mock_robot,
        lerobot_root=args.lerobot_root,
        robot_id=args.robot_id,
        max_delta=args.max_delta,
    )
    print(f"[DEBUG] mock_robot={args.mock_robot} | real_robot={getattr(args, 'real_robot', False)} | robot.mock={robot.mock}")
    if robot.mock:
        print("[WARNING] Robot is in mock mode: no real commands will be sent!")

    # ── Threaded architecture: decouple inference (~60ms) from motor control (30fps) ──
    #
    # Thread 1 (background): camera → model → publish target position
    # Thread 2 (main, 30fps): interpolate smoothly toward target → send to robot
    #
    # This ensures the robot receives steady, smooth commands even while
    # the model is computing. Without this, motors get 0 commands during
    # the 60ms inference window, causing visible saccades.

    robot_lock = threading.Lock()
    target_lock = threading.Lock()
    stop_event = threading.Event()

    shared_state = {
        'target': None,           # latest absolute target from model
        'joints_raw': None,       # physical joints at inference time
        'infer_ms': 0.0,
        'avg_steps': 0.0,
        'infer_count': 0,
        'denorm_target': None,
        'raw_delta': None,
    }

    def _inference_loop():
        """Background thread: camera → model → publish target."""
        last_target_joints = None
        wait_start = time.perf_counter()
        _diag_count = 0

        try:
            while not stop_event.is_set():
                # ── Get camera frames ──
                frames = camera.get_tensor_batch(device=device)
                if frames is None:
                    if (time.perf_counter() - wait_start) > args.camera_timeout:
                        print(f"[ERROR] No frames from camera_id={args.camera_id} "
                              f"after {args.camera_timeout:.1f}s")
                        stop_event.set()
                        return
                    time.sleep(0.005)
                    continue
                wait_start = time.perf_counter()

                # ── Get proprioception (serial access under lock) ──
                with robot_lock:
                    joints_raw = robot.get_joint_positions()

                if last_target_joints is None:
                    last_target_joints = joints_raw.copy()

                # ── Diagnostic logging for first 3 iterations ──
                if _diag_count < 3:
                    f_np = frames.cpu().numpy()
                    print(f"[DIAG {_diag_count}] frames: shape={f_np.shape} "
                          f"range=[{f_np.min():.4f}, {f_np.max():.4f}] "
                          f"mean={f_np.mean():.4f} std={f_np.std():.4f}")
                    print(f"[DIAG {_diag_count}] joints_raw: {np.round(joints_raw, 3)}")

                # ── Normalize proprio (with OOD clamping) ──
                # If the robot drifts to positions unseen during training,
                # the normalized proprio becomes extreme (e.g., -8σ), causing
                # the model to output huge deltas → positive feedback loop.
                # Clamp to ±3σ (training range was approx [-2.2, 2.6]).
                joints_for_model = denorm.normalize_proprio(joints_raw)
                joints_for_model = np.clip(joints_for_model, -3.0, 3.0)
                proprio = torch.tensor(joints_for_model,
                                       dtype=torch.float32).unsqueeze(0).to(device)

                if _diag_count < 3:
                    raw_norm = denorm.normalize_proprio(joints_raw)
                    print(f"[DIAG {_diag_count}] joints_normalized: {np.round(raw_norm, 3)} "
                          f"(clamped to: {np.round(joints_for_model, 3)})")

                # ── Model inference ──
                t_infer = time.perf_counter()
                with torch.inference_mode():
                    out = model(frames, proprio)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                infer_ms = (time.perf_counter() - t_infer) * 1000

                # ── Post-process: denormalize + delta-to-absolute ──
                raw_output = out['actions'].cpu().numpy().squeeze()

                # Clamp normalized output to training range.
                # Training actions had std ~0.9, range [-4.5, 4.8].
                # Model predictions on training data were in [-2.5, 2.3].
                # Clamp to ±3σ to prevent amplified-by-delta_std blowups.
                raw_output_clamped = np.clip(raw_output, -3.0, 3.0)

                if _diag_count < 3:
                    avg_s = sum(i['steps_used'] for i in out['info']) / len(out['info'])
                    denormed_delta = raw_output_clamped * denorm.target_std + denorm.target_mean
                    print(f"[DIAG {_diag_count}] raw_model_output: {np.round(raw_output, 4)} "
                          f"→ clamped: {np.round(raw_output_clamped, 4)} "
                          f"(PDE steps={avg_s:.1f})")
                    print(f"[DIAG {_diag_count}] denormed_delta:   {np.round(denormed_delta, 3)}")
                    _diag_count += 1
                reference = last_target_joints if denorm.is_delta else joints_raw
                abs_actions = denorm(raw_output_clamped, joints_raw,
                                     reference_joints=reference)

                # Anti-windup clamp — default 5° matches typical training delta magnitude.
                # Previous 20° was far too large and allowed the feedback loop to run away.
                if denorm.is_delta:
                    max_d = args.max_delta if args.max_delta is not None else 5.0
                    abs_actions = np.clip(abs_actions,
                                          joints_raw - max_d, joints_raw + max_d)

                # Handle chunk_size > 1 (take first action)
                if abs_actions.ndim == 2:
                    abs_actions = abs_actions[0]

                # ── Gentle homing correction ──
                # When the robot has drifted >2σ from the training mean position,
                # blend in a small pull back toward the mean. This prevents the
                # robot from getting stuck at joint limits where the model has
                # never seen data and can only output garbage.
                if denorm.is_delta and denorm.active:
                    home_pos = denorm.proprio_mean  # training mean in degrees
                    home_std = denorm.proprio_std
                    deviation = (joints_raw - home_pos) / home_std  # how many σ away
                    # Only activate beyond 2σ, strength grows linearly after that
                    homing_strength = 0.3  # degrees per σ beyond threshold
                    threshold = 2.0
                    correction = np.zeros_like(joints_raw)
                    for j in range(len(deviation)):
                        if abs(deviation[j]) > threshold:
                            excess = abs(deviation[j]) - threshold
                            correction[j] = -np.sign(deviation[j]) * excess * homing_strength
                    if np.any(correction != 0):
                        abs_actions = abs_actions + correction
                        if _diag_count <= 3:
                            print(f"[HOMING] deviation_σ={np.round(deviation, 2)} "
                                  f"correction={np.round(correction, 2)}")

                # Anchor absolute accumulated targets strictly against current reality
                # (Prevents violent integral wind-up when motors lag behind model)
                last_target_joints = joints_raw.copy()
                raw_delta = abs_actions - joints_raw
                avg_steps = (sum(i['steps_used'] for i in out['info'])
                             / len(out['info']))

                # ── Publish to control thread ──
                with target_lock:
                    shared_state['target'] = abs_actions.copy()
                    shared_state['joints_raw'] = joints_raw.copy()
                    shared_state['infer_ms'] = infer_ms
                    shared_state['avg_steps'] = avg_steps
                    shared_state['infer_count'] += 1
                    shared_state['denorm_target'] = abs_actions.copy()
                    shared_state['raw_delta'] = raw_delta.copy()

        except Exception as e:
            print(f"[INFERENCE ERROR] {e}")
            import traceback
            traceback.print_exc()
            stop_event.set()

    # ── Start background inference ──
    inf_thread = threading.Thread(target=_inference_loop, daemon=True)
    inf_thread.start()

    # ── Smooth control loop (main thread) ──
    control_fps = args.control_fps
    interp_alpha = args.interp_alpha
    last_sent = None
    step = 0
    loop_times = []
    last_printed_infer = -1

    print(f"\n{'-'*50}")
    print(f"Starting inference loop - Ctrl+C to stop")
    print(f"  chunk_size    : {chunk_size}")
    print(f"  delta_actions : {denorm.is_delta}")
    print(f"  normalization : {'active' if denorm.active else 'off (V1)'}")
    print(f"  control_fps   : {control_fps}")
    print(f"  interp_alpha  : {interp_alpha}")
    print(f"{'-'*50}\n")

    try:
        while not stop_event.is_set():
            t_loop = time.perf_counter()

            # ── Read latest target from inference thread ──
            with target_lock:
                target = shared_state['target']
                infer_count = shared_state['infer_count']
                infer_ms = shared_state['infer_ms']
                avg_steps = shared_state['avg_steps']
                denorm_target = shared_state['denorm_target']
                raw_delta = shared_state['raw_delta']

            if target is None:
                time.sleep(0.01)
                continue

            target = target.copy()

            if last_sent is None:
                last_sent = target.copy()

            # ── Exponential interpolation: smooth approach to target ──
            # Each step moves interp_alpha fraction of remaining distance.
            # At 30fps with alpha=0.5: 95% convergence in ~100ms (≈ 2 inferences)
            next_pos = last_sent + interp_alpha * (target - last_sent)

            # ── Send to robot (serial access under lock) ──
            with robot_lock:
                sent = robot.set_joint_positions(next_pos)
            last_sent = sent.copy()

            loop_ms = (time.perf_counter() - t_loop) * 1000
            loop_times.append(loop_ms)

            # ── Status logging ──
            if (args.print_actions_every > 0
                    and step % args.print_actions_every == 0
                    and infer_count != last_printed_infer):
                print(f"Step {step:5d} | Infer: {infer_ms:5.1f}ms | "
                      f"Ctrl: {loop_ms:5.1f}ms | PDE: {avg_steps:.1f} | "
                      f"Sent: {np.round(sent, 3)}")
                if raw_delta is not None:
                    print(f"  target={np.round(denorm_target, 3)} | "
                          f"delta={np.round(raw_delta, 3)}")
                last_printed_infer = infer_count

            step += 1
            elapsed = time.perf_counter() - t_loop
            time.sleep(max(0, 1.0 / control_fps - elapsed))

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        stop_event.set()
        inf_thread.join(timeout=3.0)
        camera.close()
        robot.disconnect()

        if loop_times:
            arr = np.array(loop_times)
            print(f"\nSession: {step} control steps | "
                  f"Mean: {arr.mean():.1f}ms | "
                  f"p95: {np.percentile(arr, 95):.1f}ms | "
                  f"Effective ctrl FPS: {1000/arr.mean():.1f}")


# ─────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FluidBot SO-101 LeRobot Interface (V2)')
    parser.add_argument('--mode',       default='benchmark',
                        choices=['infer', 'collect', 'benchmark', 'diagnose'],
                        help='Run mode')
    parser.add_argument('--checkpoint', default=None,
                        help='Path to trained model checkpoint')
    parser.add_argument('--camera_id',  default=0,   type=int)
    parser.add_argument('--camera_fps', default=30,  type=int)
    parser.add_argument('--camera_backend', default='auto', choices=['auto', 'dshow', 'msmf'])
    parser.add_argument('--camera_fourcc', default=None)
    parser.add_argument('--camera_timeout', default=5.0, type=float)
    parser.add_argument('--robot_port', default='/dev/ttyUSB0')
    parser.add_argument('--robot_id', default=None)
    parser.add_argument('--max_delta', default=None, type=float)
    parser.add_argument('--lerobot_root', default=None,
                        help='Path to lerobot repository root (e.g. /path/to/lerobot)')
    parser.add_argument('--mock_robot', action='store_true', default=False)
    parser.add_argument('--real_robot', action='store_true', default=False)
    parser.add_argument('--override_epsilon', default=None, type=float)
    parser.add_argument('--override_max_steps', default=None, type=int)
    parser.add_argument('--filter_alpha', default=0.4, type=float)
    parser.add_argument('--control_fps', default=30, type=int,
                        help='Motor command frequency in Hz (decoupled from inference)')
    parser.add_argument('--interp_alpha', default=0.5, type=float,
                        help='Interpolation smoothing toward target (0=frozen, 1=instant)')
    parser.add_argument('--print_actions_every', default=30, type=int)
    parser.add_argument('--task',       default='pick_place')
    parser.add_argument('--save_dir',   default='./data/step3_lerobot')
    # ── V2 additions ──
    parser.add_argument('--norm_stats', default=None,
                        help='Path to norm_stats.json if not embedded in checkpoint')
    parser.add_argument('--dataset', default=None,
                        help='Path to training data dir (for diagnose mode)')
    args = parser.parse_args()

    if args.real_robot and args.mock_robot:
        parser.error('Use either --real_robot or --mock_robot, not both.')
    if not args.real_robot:
        args.mock_robot = True

    if args.mode == 'benchmark':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if args.checkpoint and os.path.exists(args.checkpoint):
            ckpt = torch.load(args.checkpoint, map_location=device)
            model_cfg = ckpt.get('config', {})
            model = FluidBotVLA(**model_cfg).to(device)
            model.load_state_dict(ckpt['model'])
            apply_runtime_overrides(model, args.override_epsilon, args.override_max_steps)
            print(f"Loaded checkpoint: {args.checkpoint}")
        else:
            model = FluidBotVLA(
                image_size=224, d_model=128, n_layers=3,
                patch_size=16, action_dim=6, proprio_dim=6,
                spatial_pool_size=4, chunk_size=1,
            ).to(device)
            apply_runtime_overrides(model, args.override_epsilon, args.override_max_steps)
        model.eval()
        benchmark_latency(model, device)

    elif args.mode == 'infer':
        run_inference(args)

    elif args.mode == 'diagnose':
        run_diagnose(args)

    elif args.mode == 'collect':
        print("Teleoperation collection mode")
        print("Use your teleoperation device to move the robot.")
        print("Press Enter to start/end episodes, 'q' to quit.")
        print("(Full teleoperation requires LeRobot teleoperation device setup)")
        print("See: https://github.com/huggingface/lerobot")
