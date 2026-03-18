"""
SO-101 direct joint action test script (LeRobot).

- Connects to the robot
- Reads the current position
- Moves one joint slightly (shoulder_pan +10 deg)
- Prints the new position

Usage:
    python test_move_joint.py --robot_port COM6 --lerobot_root /path/to/lerobot
"""
import argparse
import sys
from pathlib import Path
import time
import numpy as np

def maybe_add_lerobot_to_syspath(lerobot_root):
    if not lerobot_root:
        return None
    root = Path(lerobot_root)
    src_dir = root / 'src'
    if src_dir.exists():
        sys.path.insert(0, str(src_dir))
        return src_dir
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot_port', default='COM6')
    parser.add_argument('--lerobot_root', default=None,
                        help='Path to lerobot repository root (e.g. /path/to/lerobot)')
    parser.add_argument('--robot_id', default=None)
    args = parser.parse_args()

    maybe_add_lerobot_to_syspath(args.lerobot_root)

    try:
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
        from lerobot.robots.so_follower.so_follower import SOFollower
    except Exception as e:
        print(f"[ERROR] Failed to import LeRobot: {e}")
        sys.exit(1)

    robot_cfg = SOFollowerRobotConfig(
        port=args.robot_port,
        id=args.robot_id,
        cameras={},
        max_relative_target=5.0,
        use_degrees=True,
    )
    robot = SOFollower(robot_cfg)
    print(f"[INFO] Calibration file: {robot.calibration_fpath}")
    robot.connect(calibrate=False)
    print("[INFO] Robot connected.")

    # Read current position
    obs = robot.get_observation()
    joints = np.array([obs[f'{joint}.pos'] for joint in ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']], dtype=np.float32)
    print(f"[INFO] Current position: {joints}")

    # Move the first joint (shoulder_pan) by +10 deg
    target = joints.copy()
    target[0] += 10.0
    print(f"[ACTION] Requesting shoulder_pan +10 deg -> {target}")
    action = {f'{joint}.pos': float(target[idx]) for idx, joint in enumerate(['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper'])}
    sent_action = robot.send_action(action)
    time.sleep(1.0)

    # Read new position
    obs2 = robot.get_observation()
    joints2 = np.array([obs2[f'{joint}.pos'] for joint in ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']], dtype=np.float32)
    print(f"[INFO] New position: {joints2}")

    robot.disconnect()
    print("[INFO] Test completed.")
