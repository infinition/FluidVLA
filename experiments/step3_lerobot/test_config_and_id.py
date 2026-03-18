"""
SO-101 calibration and robot ID verification script (LeRobot).

- Prints the calibration file path in use
- Prints the calibration file contents (JSON)
- Prints the detected robot ID
- Prints joint limits

Usage:
    python test_config_and_id.py --robot_port COM3 --lerobot_root /path/to/lerobot
"""
import argparse
import json
import sys
from pathlib import Path

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

    # Joint limits in degrees (hardcoded as in SO101Interface)
    JOINT_LIMITS_DEGREES = {
        'shoulder_pan' : (-120.0, 120.0),
        'shoulder_lift': (-90.0, 90.0),
        'elbow_flex'   : (-120.0, 120.0),
        'wrist_flex'   : (-90.0, 90.0),
        'wrist_roll'   : (-180.0, 180.0),
        'gripper'      : (0.0, 100.0),
    }

    robot_cfg = SOFollowerRobotConfig(
        port=args.robot_port,
        id=args.robot_id,
        cameras={},
        max_relative_target=5.0,
        use_degrees=True,
    )
    robot = SOFollower(robot_cfg)
    print(f"[INFO] Calibration file path: {robot.calibration_fpath}")
    try:
        with open(robot.calibration_fpath, 'r') as f:
            calib = json.load(f)
        print("[INFO] Calibration file content (per joint):")
        for joint, params in calib.items():
            print(f"  {joint}:")
            for k, v in params.items():
                print(f"    {k}: {v}")
    except Exception as e:
        print(f"[ERROR] Failed to read calibration file: {e}")

    print(f"[INFO] Robot ID: {robot_cfg.id}")
    print(f"[INFO] Joint limits (degrees, hardcoded):")
    for joint, (lo, hi) in JOINT_LIMITS_DEGREES.items():
        print(f"  {joint}: {lo} -> {hi}")
