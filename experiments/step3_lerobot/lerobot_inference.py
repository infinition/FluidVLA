"""
lerobot_inference.py — Step 3: Real Hardware (SO-101 LeRobot)

Deploys FluidBot on the SO-101 arm with USB camera.
Targets real-time 30fps inference, fully on-device.

Hardware:
  - SO-101 6-DOF robot arm
  - USB camera (mounted on gripper or fixed overhead)
  - Host PC or Jetson AGX Orin for inference

Pipeline:
  Camera frame(s)
       │
  [Frame Buffer]   — maintain rolling window of N_FRAMES
       │
  [FluidBotVLA]    — diffusion-based vision → action
       │
  [Action Filter]  — smooth + safety clip
       │
  [SO-101 Motors]  — execute joint positions

Usage:
  # Teleoperation data collection
  python lerobot_inference.py --mode collect --task pick_place

  # Inference with trained model  
  python lerobot_inference.py --mode infer --checkpoint ./checkpoints/step3/best.pt

  # Benchmark latency
  python lerobot_inference.py --mode benchmark
"""

import os
import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path
from collections import deque
from typing import Optional, List, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.core.fluid_model import FluidBotVLA


# ─────────────────────────────────────────
# Camera Interface
# ─────────────────────────────────────────

class CameraStream:
    """
    USB camera interface for SO-101.
    Maintains a rolling buffer of recent frames for temporal context.
    """
    def __init__(
        self,
        device_id  : int = 0,
        width      : int = 224,
        height     : int = 224,
        fps        : int = 30,
        buffer_size: int = 4,    # number of frames to feed FluidBot
    ):
        self.width       = width
        self.height      = height
        self.buffer_size = buffer_size
        self.frame_buffer = deque(maxlen=buffer_size)
        
        try:
            import cv2
            self.cap = cv2.VideoCapture(device_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            self.cv2 = cv2
            self.available = self.cap.isOpened()
        except ImportError:
            print("[Camera] OpenCV not installed — using synthetic frames")
            self.available = False
    
    def read_frame(self) -> Optional[np.ndarray]:
        """Read one frame, return (H, W, 3) uint8 or None."""
        if not self.available:
            # Synthetic: slow-moving gradient (simulates a real scene)
            t = time.time()
            H, W = self.height, self.width
            frame = np.zeros((H, W, 3), dtype=np.uint8)
            frame[:, :, 0] = (np.sin(np.linspace(0, 2*np.pi, W) + t) * 127 + 127).astype(np.uint8)
            frame[:, :, 1] = (np.cos(np.linspace(0, 2*np.pi, H) + t * 0.7)[:, None] * 127 + 127).astype(np.uint8)
            return frame
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        frame = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        frame = self.cv2.resize(frame, (self.width, self.height))
        return frame
    
    def get_tensor_batch(self, device='cpu') -> Optional[torch.Tensor]:
        """
        Read a new frame, add to buffer, return full buffer as tensor.
        Returns: (1, 3, T, H, W) or None if buffer not full yet.
        """
        frame = self.read_frame()
        if frame is None:
            return None
        
        # Normalize to [0, 1] and convert to (C, H, W)
        frame_t = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1) / 255.0
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        frame_t = (frame_t - mean) / std
        
        self.frame_buffer.append(frame_t)
        
        if len(self.frame_buffer) < self.buffer_size:
            return None  # Not enough frames yet
        
        # Stack to (T, C, H, W) then permute to (C, T, H, W)
        frames = torch.stack(list(self.frame_buffer), dim=0)  # (T, C, H, W)
        frames = frames.permute(1, 0, 2, 3)                   # (C, T, H, W)
        return frames.unsqueeze(0).to(device)                 # (1, C, T, H, W)
    
    def close(self):
        if self.available and hasattr(self, 'cap'):
            self.cap.release()


# ─────────────────────────────────────────
# SO-101 Robot Interface
# ─────────────────────────────────────────

class SO101Interface:
    """
    Interface to the SO-101 6-DOF robot arm.
    Uses the LeRobot library for motor control.
    
    Joint order: [shoulder_pan, shoulder_lift, elbow_flex, 
                  wrist_flex, wrist_roll, gripper_open]
    
    All angles in radians. Gripper: 0=closed, 1=open.
    """
    
    # Joint limits (radians) — CRITICAL for safety
    JOINT_LIMITS = {
        'shoulder_pan' : (-2.09, 2.09),    # ±120°
        'shoulder_lift': (-1.57, 1.57),    # ±90°
        'elbow_flex'   : (-2.09, 2.09),    # ±120°
        'wrist_flex'   : (-1.57, 1.57),    # ±90°
        'wrist_roll'   : (-3.14, 3.14),    # ±180°
        'gripper'      : (0.0,   1.0),     # open/close
    }
    
    # Maximum joint velocity per step (radians/step) — safety limit
    MAX_DELTA = 0.05  # ~3° per step at 30fps = reasonable speed
    
    def __init__(self, port: str = '/dev/ttyUSB0', mock: bool = False):
        self.mock = mock
        self.current_joints = np.zeros(6, dtype=np.float32)
        self.connected = False
        
        if not mock:
            try:
                from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus
                self.robot = DynamixelMotorsBus(port=port)
                self.robot.connect()
                self.connected = True
                print(f"[SO-101] Connected on {port}")
            except Exception as e:
                print(f"[SO-101] Connection failed: {e}")
                print("[SO-101] Falling back to mock mode")
                self.mock = True
        else:
            print("[SO-101] Mock mode — no hardware commands sent")
    
    def get_joint_positions(self) -> np.ndarray:
        """Read current joint angles in radians. Shape: (6,)"""
        if self.mock or not self.connected:
            return self.current_joints.copy()
        
        # Read from real hardware
        pos = self.robot.read_with_motor_ids('Present_Position', range(6))
        # Convert from Dynamixel ticks to radians (depends on motor model)
        return np.array(pos, dtype=np.float32) * (2 * np.pi / 4096)
    
    def set_joint_positions(self, targets: np.ndarray) -> np.ndarray:
        """
        Send joint position targets to motors.
        Applies safety limits before sending.
        Returns actually-sent positions.
        """
        targets = np.array(targets, dtype=np.float32)
        
        # 1. Hard joint limit clipping
        limit_names = list(self.JOINT_LIMITS.keys())
        for i, (lo, hi) in enumerate(self.JOINT_LIMITS.values()):
            targets[i] = np.clip(targets[i], lo, hi)
        
        # 2. Velocity limit — prevent sudden jumps
        delta = targets - self.current_joints
        delta = np.clip(delta, -self.MAX_DELTA, self.MAX_DELTA)
        targets = self.current_joints + delta
        
        # 3. Send to hardware
        if not self.mock and self.connected:
            ticks = (targets * 4096 / (2 * np.pi)).astype(int)
            self.robot.write_with_motor_ids('Goal_Position', range(6), ticks.tolist())
        
        self.current_joints = targets.copy()
        return targets
    
    def emergency_stop(self):
        """Immediately stop all motors."""
        print("⚠️  EMERGENCY STOP")
        if not self.mock and self.connected:
            try:
                self.robot.write_with_motor_ids('Torque_Enable', range(6), [0] * 6)
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
    """
    Smooths and validates predicted actions before sending to robot.
    
    Uses exponential moving average to prevent jerky motion.
    The alpha parameter controls smoothing strength:
      alpha=1.0: no smoothing (raw predictions)
      alpha=0.1: heavy smoothing
    """
    def __init__(self, action_dim: int = 6, alpha: float = 0.3):
        self.alpha   = alpha
        self.history = None
    
    def __call__(self, action: np.ndarray) -> np.ndarray:
        if self.history is None:
            self.history = action.copy()
        
        self.history = self.alpha * action + (1 - self.alpha) * self.history
        return self.history.copy()


# ─────────────────────────────────────────
# Data Collection (Teleoperation)
# ─────────────────────────────────────────

class TeleoperationRecorder:
    """
    Records camera frames + robot joint positions for imitation learning.
    
    Output format: LeRobot dataset format (HuggingFace datasets compatible)
    Each episode: sequence of (frame, joint_positions) pairs.
    """
    def __init__(self, save_dir: str, task_name: str):
        self.save_dir  = Path(save_dir)
        self.task_name = task_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.episodes = []
        self.current_episode = []
    
    def record_step(self, frame: np.ndarray, joints: np.ndarray, action: np.ndarray):
        self.current_episode.append({
            'frame' : frame,
            'joints': joints.tolist(),
            'action': action.tolist(),
        })
    
    def end_episode(self, success: bool = True):
        if len(self.current_episode) > 0:
            ep_id = len(self.episodes)
            ep_data = {
                'id'     : ep_id,
                'task'   : self.task_name,
                'success': success,
                'length' : len(self.current_episode),
                'steps'  : self.current_episode,
            }
            self.episodes.append(ep_data)
            
            # Save frames as numpy array
            frames = np.stack([s['frame'] for s in self.current_episode])
            np.save(self.save_dir / f'episode_{ep_id:04d}_frames.npy', frames)
            
            # Save metadata without frames
            meta = {k: v for k, v in ep_data.items() if k != 'steps'}
            meta['steps'] = [{k: v for k, v in s.items() if k != 'frame'}
                             for s in self.current_episode]
            with open(self.save_dir / f'episode_{ep_id:04d}_meta.json', 'w') as f:
                json.dump(meta, f, indent=2)
            
            print(f"Episode {ep_id} saved: {len(self.current_episode)} steps, success={success}")
            self.current_episode = []
    
    def save_dataset_index(self):
        index = {
            'task'          : self.task_name,
            'n_episodes'    : len(self.episodes),
            'total_steps'   : sum(ep['length'] for ep in self.episodes),
            'success_rate'  : sum(ep['success'] for ep in self.episodes) / max(1, len(self.episodes)),
        }
        with open(self.save_dir / 'dataset_index.json', 'w') as f:
            json.dump(index, f, indent=2)
        print(f"\nDataset saved: {index['n_episodes']} episodes, {index['total_steps']} steps")


# ─────────────────────────────────────────
# Latency Benchmark
# ─────────────────────────────────────────

def benchmark_latency(model, device, n_runs=100):
    """
    Measures end-to-end inference latency.
    Target: <33ms (30fps) on RTX GPU, <50ms on Jetson AGX Orin.
    """
    print("\n" + "=" * 60)
    print("LATENCY BENCHMARK — Target: <33ms for 30fps")
    print("=" * 60)
    
    model.eval()
    
    # Warmup
    with torch.no_grad():
        dummy = torch.randn(1, 3, 4, 224, 224, device=device)
        prop  = torch.randn(1, 7, device=device)
        for _ in range(10):
            _ = model(dummy, prop)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Timed runs
    latencies = []
    steps_used = []
    
    with torch.no_grad():
        for _ in range(n_runs):
            frames = torch.randn(1, 3, 4, 224, 224, device=device)
            proprio = torch.randn(1, 7, device=device)
            
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
        print(f"  ✅ Meets 30fps target (<{target_ms}ms)")
    else:
        print(f"  ⚠️  Exceeds {target_ms}ms — consider: fewer layers, smaller d_model, or TorchScript export")
    
    return latencies


# ─────────────────────────────────────────
# Main Inference Loop
# ─────────────────────────────────────────

def run_inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model_cfg = ckpt.get('config', {})
        model = FluidBotVLA(**model_cfg).to(device)
        model.load_state_dict(ckpt['model'])
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint found — using untrained model (for testing only)")
        model = FluidBotVLA(
            image_size=224, in_channels=3, d_model=128, n_layers=3,
            patch_size=16, action_dim=6, proprio_dim=6, n_frames=4
        ).to(device)
    
    model.eval()
    
    # Latency benchmark first
    benchmark_latency(model, device)
    
    # Hardware
    camera = CameraStream(device_id=args.camera_id, width=224, height=224,
                          buffer_size=4)
    robot  = SO101Interface(port=args.robot_port, mock=args.mock_robot)
    filt   = ActionFilter(action_dim=6, alpha=0.4)
    
    print(f"\n{'─'*50}")
    print("Starting inference loop — Ctrl+C to stop")
    print(f"{'─'*50}\n")
    
    loop_times = []
    step = 0
    
    try:
        while True:
            t_loop = time.perf_counter()
            
            # 1. Get camera frames
            frames = camera.get_tensor_batch(device=device)
            if frames is None:
                time.sleep(0.01)
                continue
            
            # 2. Get proprioception
            joints = robot.get_joint_positions()
            proprio = torch.tensor(joints, dtype=torch.float32).unsqueeze(0).to(device)
            
            # 3. Inference
            t_infer = time.perf_counter()
            with torch.no_grad():
                out = model(frames, proprio)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            infer_ms = (time.perf_counter() - t_infer) * 1000
            
            # 4. Post-process actions
            raw_actions = out['actions'].cpu().numpy().squeeze()  # (6,)
            actions = filt(raw_actions)
            
            # 5. Send to robot
            sent = robot.set_joint_positions(actions)
            
            # 6. Timing
            loop_ms = (time.perf_counter() - t_loop) * 1000
            loop_times.append(loop_ms)
            
            avg_steps = sum(i['steps_used'] for i in out['info']) / len(out['info'])
            
            if step % 30 == 0:  # Print every ~1 second at 30fps
                print(f"Step {step:5d} | Infer: {infer_ms:5.1f}ms | "
                      f"Loop: {loop_ms:5.1f}ms | "
                      f"PDE steps: {avg_steps:.1f} | "
                      f"Joints: {np.round(sent, 3)}")
            
            step += 1
            
            # Maintain ~30fps
            elapsed = time.perf_counter() - t_loop
            sleep = max(0, 1/30 - elapsed)
            time.sleep(sleep)
    
    except KeyboardInterrupt:
        print("\nStopped by user.")
    
    finally:
        camera.close()
        robot.disconnect()
        
        if loop_times:
            loop_arr = np.array(loop_times)
            print(f"\nSession stats: {step} steps | "
                  f"Loop mean: {loop_arr.mean():.1f}ms | "
                  f"p95: {np.percentile(loop_arr, 95):.1f}ms | "
                  f"Effective FPS: {1000/loop_arr.mean():.1f}")


# ─────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FluidBot SO-101 LeRobot Interface')
    parser.add_argument('--mode',       default='benchmark',
                        choices=['infer', 'collect', 'benchmark'],
                        help='Run mode')
    parser.add_argument('--checkpoint', default=None,
                        help='Path to trained model checkpoint')
    parser.add_argument('--camera_id',  default=0,   type=int)
    parser.add_argument('--robot_port', default='/dev/ttyUSB0')
    parser.add_argument('--mock_robot', action='store_true', default=True,
                        help='Use mock robot (no hardware)')
    parser.add_argument('--task',       default='pick_place')
    parser.add_argument('--save_dir',   default='./data/episodes')
    args = parser.parse_args()
    
    if args.mode == 'benchmark':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = FluidBotVLA(
            image_size=224, d_model=128, n_layers=3,
            patch_size=16, action_dim=6, proprio_dim=7
        ).to(device)
        model.eval()
        benchmark_latency(model, device)
    
    elif args.mode == 'infer':
        run_inference(args)
    
    elif args.mode == 'collect':
        print("Teleoperation collection mode")
        print("Use your teleoperation device to move the robot.")
        print("Press Enter to start/end episodes, 'q' to quit.")
        # Full teleoperation loop would integrate with LeRobot's teleoperation tools
        print("(Full teleoperation requires LeRobot teleoperation device setup)")
        print("See: https://github.com/huggingface/lerobot")