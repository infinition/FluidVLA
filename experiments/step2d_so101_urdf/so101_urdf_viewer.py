"""
so101_urdf_viewer.py — Step 2d : 3D Viewer SO-101 + Live Inference
────────────────────────────────────────────────────────────────────
Loads the SO-101 URDF into Isaac Sim and drives the arm in real
time with FluidBotVLA. Strong visual demo for arXiv / investors.

PREREQUISITES:
  - Isaac Sim 4.x installed (omniverse://localhost)
  - SO-101 URDF available (see README.md for download)
  - FluidVLA Checkpoint from Step 2a or 2c

ALTERNATIVES IF ISAAC SIM IS NOT AVAILABLE:
  - See README.md: Option B (Rerun), Option C (MuJoCo / PyBullet)

USAGE:
  # Full viewer mode (Isaac Sim GUI)
  python so101_urdf_viewer.py \\
      --checkpoint ../../checkpoints/step2c_isaac/best.pt \\
      --urdf /path/to/so101.urdf \\
      --episodes 5 \\
      --show_gui

  # Headless mode (log only, no GUI)
  python so101_urdf_viewer.py \\
      --checkpoint ../../checkpoints/step2a_synthetic/best.pt \\
      --urdf /path/to/so101.urdf \\
      --headless

  # Without checkpoint (random weights, pipeline test)
  python so101_urdf_viewer.py --urdf /path/to/so101.urdf --random_weights

FLAGS PDE ON/OFF :
  --no_pde    Disables the Laplacian when loading the model.
              Useful if trajectories are unstable in viewer.
              See fluid_layer.py for complete documentation.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, List

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from fluidvla.core import FluidBotVLA

# Shared constants with step2a / step2c
ACTION_DIM  = 7
PROPRIO_DIM = 8
N_FRAMES    = 4


# ─── Common Helpers ────────────────────────────────────────────────────────

def make_synthetic_obs(image_size: int = 64, n_frames: int = N_FRAMES,
                       device: str = 'cpu'):
    """
    Generates a synthetic observation (uniform gray scene).
    Used for warm-up and testing without environment.
    """
    frames  = torch.ones(1, 3, n_frames, image_size, image_size, device=device) * 0.5
    proprio = torch.zeros(1, PROPRIO_DIM, device=device)
    return frames, proprio


# ─── Isaac Sim Viewer ────────────────────────────────────────────────────────

class SO101IsaacViewer:
    """
    Loads the SO-101 URDF into Isaac Sim and drives the joints
    with the actions predicted by FluidBotVLA.

    Requires Isaac Sim 4.x. See README.md for alternatives.
    """

    def __init__(self, urdf_path: str, headless: bool = False,
                 image_size: int = 64):
        self.urdf_path  = urdf_path
        self.image_size = image_size
        self.available  = False
        self.robot_path = "/World/SO101"

        try:
            self._init_isaac(headless)
        except Exception as e:
            import traceback
            print(f"\n[Step 2d] Isaac Sim unavailable: {e}")
            print("[Step 2d] Use --fallback_rerun or --fallback_mujoco")
            print("[Step 2d] See README.md for alternatives")
            traceback.print_exc()

    def _init_isaac(self, headless: bool):
        from isaacsim import SimulationApp
        self.sim_app = SimulationApp({'headless': headless})

        from omni.isaac.core import World
        import omni.isaac.core.utils.prims as prim_utils
        from omni.isaac.urdf import _urdf as urdf_ext

        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()

        # ── Import URDF ─────────────────────────────────────────────────────
        # Isaac Sim 4.x natively supports URDF import
        # Doc: https://docs.omniverse.nvidia.com/isaacsim/latest/robot_setup/urdf_importer.html
        urdf_interface = urdf_ext.acquire_urdf_interface()
        config = urdf_ext.ImportConfig()
        config.merge_fixed_joints = False  # keeps all joints visible
        config.fix_base            = True  # base fixed to ground
        config.import_inertia_tensor = True
        config.self_collision_filter = True

        print(f"[Step 2d] Import URDF : {self.urdf_path}")
        result, prim_path = urdf_interface.import_urdf(
            self.urdf_path,
            self.robot_path,
            config,
        )
        if not result:
            raise RuntimeError(f"URDF import failed. Valid URDF path? {self.urdf_path}")

        print(f"[Step 2d] [OK] SO-101 imported -> {prim_path}")

        # Get articulation to control joints
        from omni.isaac.core.articulations import Articulation
        self.robot_art = Articulation(prim_path=self.robot_path)

        # Top-down camera for visual observation
        from omni.isaac.sensor import Camera
        self.camera = Camera(
            prim_path='/World/camera_topdown',
            position=np.array([0.0, 0.0, 1.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            frequency=30,
            resolution=(self.image_size, self.image_size),
        )

        self.world.reset()
        self.robot_art.initialize()
        self.camera.initialize()

        # Warm-up renderer RTX
        print("[Step 2d] Warm-up renderer (30 steps)...")
        for _ in range(30):
            self.world.step(render=True)

        self.available    = True
        self.n_joints     = self.robot_art.num_dof
        self.frame_buffer = []
        print(f"[Step 2d] [OK] Viewer ready | {self.n_joints} joints detected")

    def get_joint_positions(self) -> np.ndarray:
        """Reads current joint positions (radians)."""
        if not self.available:
            return np.zeros(ACTION_DIM)
        return self.robot_art.get_joint_positions().astype(np.float32)

    def set_joint_targets(self, actions: np.ndarray):
        """
        Applies predicted actions to the 3D model joints.
        actions: (7,) — 6 joints + 1 gripper, all in [-1, 1]
        """
        if not self.available:
            return
        # Converts normalized actions to joint positions
        joint_pos = actions[:6] * np.pi  # [-1,1] → [-π, π]
        self.robot_art.set_joint_position_targets(joint_pos)
        self.world.step(render=True)

    def capture_frame(self) -> np.ndarray:
        """Captures (H, W, 3) float32 [0,1] from the camera."""
        blank = np.full((self.image_size, self.image_size, 3), 0.75, dtype=np.float32)
        if not self.available:
            return blank
        try:
            rgba = self.camera.get_rgba()
            if rgba is None or rgba.max() < 0.01:
                self.world.step(render=True)
                rgba = self.camera.get_rgba()
            rgb = rgba[:, :, :3].astype(np.float32) / 255.0
            return np.clip(rgb, 0, 1)
        except Exception:
            return blank

    def get_obs(self, n_frames: int = N_FRAMES):
        """
        Returns full observation for FluidBotVLA.
        (1, 3, T, H, W) tensor + (1, 8) proprio tensor
        """
        frame = self.capture_frame()
        self.frame_buffer.append(frame)
        if len(self.frame_buffer) > n_frames:
            self.frame_buffer.pop(0)
        # Padding if buffer not yet full
        buf = self.frame_buffer
        while len(buf) < n_frames:
            buf = [buf[0]] + buf

        frames_np = np.stack(buf[-n_frames:], axis=0)  # (T, H, W, 3)
        frames_t  = (
            torch.from_numpy(frames_np)
            .permute(3, 0, 1, 2)  # (3, T, H, W)
            .unsqueeze(0)          # (1, 3, T, H, W)
            .float()
        )
        joints  = self.get_joint_positions()
        proprio = torch.from_numpy(
            np.concatenate([joints, [0.0]])[:PROPRIO_DIM]  # pad/trim à 8
        ).unsqueeze(0).float()

        return frames_t, proprio

    def close(self):
        if hasattr(self, 'sim_app'):
            self.sim_app.close()


# ─── Fallback Viewer (Rerun) ─────────────────────────────────────────────────

class SO101RerunViewer:
    """
    Lightweight viewer based on Rerun (no physics).
    Alternative if Isaac Sim is not available.

    pip install rerun-sdk
    """
    def __init__(self, urdf_path: str, image_size: int = 64):
        self.urdf_path  = urdf_path
        self.image_size = image_size
        self.available  = False
        try:
            import rerun as rr
            self.rr = rr
            rr.init("FluidVLA SO-101 Viewer", spawn=True)
            rr.log("robot/urdf", rr.Asset3D(path=urdf_path))
            self.available = True
            print("[Step 2d] [OK] Rerun viewer launched")
        except ImportError:
            print("[Step 2d] Rerun not installed. pip install rerun-sdk")
        except Exception as e:
            print(f"[Step 2d] Rerun init failed: {e}")

    def log_step(self, step: int, actions: np.ndarray, info: dict):
        if not self.available:
            return
        for i, a in enumerate(actions[:6]):
            self.rr.log(f"robot/joint_{i}", self.rr.Scalar(float(a)))
        self.rr.log("inference/steps_used",
                    self.rr.Scalar(float(info.get('steps_used', 0))))
        self.rr.log("inference/turbulence",
                    self.rr.Scalar(float(info.get('final_turbulence', 0))))

    def get_obs(self, n_frames=N_FRAMES):
        """Gray synthetic obs — no real camera in Rerun mode."""
        frames  = torch.ones(1, 3, n_frames, self.image_size, self.image_size) * 0.5
        proprio = torch.zeros(1, PROPRIO_DIM)
        return frames, proprio


# ─── Main Inference Loop ──────────────────────────────────────────

@torch.no_grad()
def run_inference_loop(viewer, model: FluidBotVLA, device: torch.device,
                       n_steps: int = 200, log_interval: int = 10):
    """
    Inference loop: FluidVLA predicts actions → 3D viewer applies.
    Logs key metrics at each step.
    """
    model.eval()
    latencies  = []
    steps_list = []
    all_actions = []

    print(f"\n{'='*55}")
    print(f"Live inference SO-101 - {n_steps} steps")
    print(f"{'='*55}")

    for step in range(n_steps):
        # Observation
        frames, proprio = viewer.get_obs()
        frames  = frames.to(device)
        proprio = proprio.to(device)

        # FluidVLA inference
        t0  = time.perf_counter()
        out = model(frames, proprio)
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)

        actions = out['actions'].cpu().numpy()[0]  # (7,)
        all_actions.append(actions.copy())

        # Stats PDE
        avg_steps = 0.0
        avg_turb  = 0.0
        if out['info']:
            avg_steps = sum(i['steps_used'] for i in out['info']) / len(out['info'])
            avg_turb  = sum(i['final_turbulence'] for i in out['info']) / len(out['info'])
        steps_list.append(avg_steps)

        # Apply to viewer
        if hasattr(viewer, 'set_joint_targets'):
            viewer.set_joint_targets(actions)
        elif hasattr(viewer, 'log_step'):
            viewer.log_step(step, actions, out['info'][0] if out['info'] else {})

        if step % log_interval == 0:
            pde_tag = "PDE ON" if out['info'][0].get('pde_active', True) else "PDE OFF"
            print(f"  Step {step:4d}/{n_steps} | "
                  f"Lat:{lat:.1f}ms | "
                  f"PDE steps:{avg_steps:.1f}/12 | "
                  f"Turb:{avg_turb:.4f} | "
                  f"Actions:{np.round(actions[:4], 2)} [{pde_tag}]")

    # Summary
    print(f"\n{'='*55}")
    print(f"SUMMARY Step 2d")
    print(f"  Avg latency      : {np.mean(latencies):.1f}ms")
    print(f"  Latency p95      : {np.percentile(latencies, 95):.1f}ms")
    print(f"  Effective FPS    : {1000/np.mean(latencies):.0f}")
    print(f"  Avg PDE steps    : {np.mean(steps_list):.1f}/12")
    print(f"  [OK]" if np.mean(latencies) < 50 else "  [FAIL]", "Latency <50ms")

    return {
        'latency_mean_ms': float(np.mean(latencies)),
        'latency_p95_ms' : float(np.percentile(latencies, 95)),
        'avg_pde_steps'  : float(np.mean(steps_list)),
        'n_steps'        : n_steps,
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='FluidVLA Step 2d - Viewer 3D SO-101 URDF'
    )
    parser.add_argument('--urdf', required=True,
                        help='Path to so101.urdf (see README.md)')
    parser.add_argument('--checkpoint', default=None,
                        help='Checkpoint FluidVLA (step2a ou step2c)')
    parser.add_argument('--show_gui',   action='store_true',
                        help='Shows Isaac Sim 3D GUI (otherwise headless)')
    parser.add_argument('--fallback_rerun', action='store_true',
                        help='Uses Rerun instead of Isaac Sim')
    parser.add_argument('--random_weights', action='store_true',
                        help='Random weights (pipeline test without checkpoint)')
    parser.add_argument('--n_steps',    default=200, type=int)
    parser.add_argument('--image_size', default=64,  type=int)
    parser.add_argument('--save_dir',   default='../../checkpoints/step2d_so101_urdf')
    # ── PDE ON/OFF ────────────────────────────────────────────────────────────
    parser.add_argument('--no_pde', action='store_true',
                        help='Disables Laplacian (diagnostic). '
                             'See fluid_layer.py for documentation.')
    # ─────────────────────────────────────────────────────────────────────────
    args = parser.parse_args()

    use_pde = not args.no_pde
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"FluidVLA Step 2d - Viewer 3D SO-101 URDF")
    print(f"  URDF      : {args.urdf}")
    print(f"  use_pde   : {use_pde}")
    print(f"  device    : {device}")
    print(f"{'='*60}")

    # ── Model loading ─────────────────────────────────────────────────
    if args.checkpoint and not args.random_weights:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        cfg  = ckpt.get('config', {})
        # Respect checkpoint use_pde unless --no_pde is explicitly used
        cfg_use_pde = cfg.get('use_pde', True) and use_pde
        model = FluidBotVLA(
            image_size  = cfg.get('image_size', args.image_size),
            d_model     = cfg.get('d_model', 128),
            n_layers    = cfg.get('n_layers', 3),
            patch_size  = min(16, cfg.get('image_size', args.image_size) // 4),
            action_dim  = cfg.get('action_dim', ACTION_DIM),
            proprio_dim = cfg.get('proprio_dim', PROPRIO_DIM),
            max_steps   = cfg.get('max_steps', 12),
            epsilon     = cfg.get('epsilon', 0.02),
            n_frames    = cfg.get('n_frames', N_FRAMES),
            use_pde     = cfg_use_pde,
        ).to(device)
        model.load_state_dict(ckpt['model'])
        print(f"  Checkpoint loaded: {args.checkpoint} | use_pde={cfg_use_pde}")
    else:
        # Random weights - pipeline test only
        model = FluidBotVLA(
            image_size=args.image_size, d_model=128, n_layers=3,
            patch_size=max(4, args.image_size // 16),
            action_dim=ACTION_DIM, proprio_dim=PROPRIO_DIM,
            use_pde=use_pde,
        ).to(device)
        print("  [WARN] Random weights - actions will not be meaningful")

    p = model.count_parameters()
    print(f"  Params : {p['M']:.2f}M")

    # ── Create viewer ───────────────────────────────────────────────────
    if args.fallback_rerun:
        viewer = SO101RerunViewer(args.urdf, args.image_size)
    else:
        viewer = SO101IsaacViewer(args.urdf, headless=not args.show_gui,
                                  image_size=args.image_size)
        if not viewer.available:
            print("[Step 2d] Isaac Sim unavailable -> Rerun fallback")
            viewer = SO101RerunViewer(args.urdf, args.image_size)

    if not viewer.available:
        print("[Step 2d] [FAIL] No viewer available.")
        print("[Step 2d] Install Isaac Sim or: pip install rerun-sdk")
        print("[Step 2d] See README.md for options.")
        return

    # ── Inference loop ───────────────────────────────────────────────────
    try:
        results = run_inference_loop(
            viewer, model, device,
            n_steps=args.n_steps, log_interval=10,
        )
    finally:
        if hasattr(viewer, 'close'):
            viewer.close()

    # ── Save results ─────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'inference_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved -> {args.save_dir}/inference_results.json")
    print(f"\nNext -> Step 3: real Jetson Orin Nano hardware")
    print(f"  python ../step3_lerobot/lerobot_inference.py \\")
    print(f"      --mode benchmark \\")
    print(f"      --checkpoint {args.save_dir}/../step2c_isaac/best.pt")


if __name__ == '__main__':
    main()