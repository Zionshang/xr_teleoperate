#!/usr/bin/env python3
"""XR controller teleop for ARX5 arm (no image/data pipeline).

- Uses TeleVuer controller tracking.
- Uses arx5-sdk LCM client for arm control.
- Aligns XR wrist origin with robot EE pose at teleop start (recenter supported).
"""

import argparse
import time
from typing import Tuple
import numpy as np

from televuer import TeleVuerWrapper
from televuer.tv_wrapper import fast_mat_inv
from teleop.utils.pose_plotter import PosePlotter

from communication.lcm.lcm_client import Arx5LcmClient


def rpy2rotm(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ]
    )
    R_y = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )
    R_z = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ]
    )
    return R_z @ R_y @ R_x


def rotm2rpy(R: np.ndarray) -> np.ndarray:
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])


def pose6d_to_mat(pose_6d: np.ndarray) -> np.ndarray:
    mat = np.eye(4)
    mat[:3, :3] = rpy2rotm(pose_6d[3:])
    mat[:3, 3] = pose_6d[:3]
    return mat


def mat_to_pose6d(mat: np.ndarray) -> np.ndarray:
    pose_6d = np.zeros(6, dtype=np.float64)
    pose_6d[:3] = mat[:3, 3]
    pose_6d[3:] = rotm2rpy(mat[:3, :3])
    return pose_6d


def choose_wrist_pose(tele_data, hand: str) -> Tuple[np.ndarray, bool, bool, float]:
    if hand == "left":
        return (
            tele_data.left_wrist_pose,
            tele_data.left_ctrl_aButton,
            tele_data.left_ctrl_bButton,
            tele_data.left_ctrl_triggerValue,
        )
    return (
        tele_data.right_wrist_pose,
        tele_data.right_ctrl_aButton,
        tele_data.right_ctrl_bButton,
        tele_data.right_ctrl_triggerValue,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="ARX5 XR controller teleop (no images/data).")
    parser.add_argument("--frequency", type=float, default=60.0, help="Control frequency (Hz)")
    parser.add_argument(
        "--display-mode",
        type=str,
        choices=["immersive", "ego", "pass-through"],
        default="pass-through",
        help="XR display mode",
    )
    parser.add_argument("--hand", type=str, choices=["left", "right"], default="right")
    parser.add_argument("--gripper-max", type=float, default=0.09, help="Max gripper width (m)")
    parser.add_argument("--debug", action="store_true", help="Visualize target pose instead of sending commands")
    parser.add_argument("--lcm-url", type=str, default="", help="LCM URL, e.g. udpm://239.255.76.67:7667?ttl=1")
    parser.add_argument("--lcm-address", type=str, default="239.255.76.67")
    parser.add_argument("--lcm-port", type=int, default=7667)
    parser.add_argument("--lcm-ttl", type=int, default=1)
    args = parser.parse_args()

    tv = TeleVuerWrapper(
        use_hand_tracking=False,
        binocular=False,
        img_shape=(480, 1280),
        display_mode=args.display_mode,
        zmq=False,
        webrtc=False,
        webrtc_url=None,
        return_hand_rot_data=False,
        use_clutch=True,
    )

    client = Arx5LcmClient(url=args.lcm_url, address=args.lcm_address, port=args.lcm_port, ttl=args.lcm_ttl)
    client.reset_to_home()

    xr_origin = None
    robot_origin = None

    pose_plotter = PosePlotter() if args.debug else None

    dt = 1.0 / max(args.frequency, 1e-3)

    print("----------------------------------------------------------------")
    print("🟢  按下 A 重新对零(以当前腕部为零位)")
    print("🔴  按下 B 退出")
    print("⚠️  请保持安全距离")

    try:
        while True:
            t0 = time.time()
            tele_data = tv.get_tele_data()
            xr_pose, recenter_pressed, exit_pressed, trigger_value = choose_wrist_pose(
                tele_data, args.hand
            )

            if exit_pressed:
                break

            if xr_origin is None or recenter_pressed:
                xr_origin = xr_pose.copy()
                state = client.get_state()
                robot_origin = pose6d_to_mat(state["ee_pose"])

            if xr_origin is None or robot_origin is None:
                time.sleep(dt)
                continue

            # Relative motion from XR origin to current
            xr_rel = fast_mat_inv(xr_origin) @ xr_pose
            target_mat = robot_origin @ xr_rel
            target_pose = mat_to_pose6d(target_mat)
            gripper_pos = args.gripper_max * float(trigger_value) / 10.0

            if args.debug:
                pose_plotter.update_matrix(target_mat, label="target", draw_axes=True)
                print(f"Target pose: {target_pose}, gripper: {gripper_pos:.3f} m")
            else:
                client.set_ee_pose(target_pose, gripper_pos=gripper_pos)

            elapsed = time.time() - t0
            time.sleep(max(0.0, dt - elapsed))
    finally:
        tv.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
