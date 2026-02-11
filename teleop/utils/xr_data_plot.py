#!/usr/bin/env python3
"""XR data plotter for standalone device testing."""

import argparse
import time
import threading

from televuer import TeleVuerWrapper
from teleop.utils.pose_plotter import PosePlotter


def _parse_img_shape(value: str) -> tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("img-shape must be like '480,1280'")
    try:
        h = int(parts[0].strip())
        w = int(parts[1].strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("img-shape must be integers like '480,1280'") from exc
    return (h, w)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot XR wrist trajectories from TeleVuer.")
    parser.add_argument("--display-mode", choices=["pass-through"], default="pass-through")
    parser.add_argument("--hz", type=float, default=30.0, help="Sampling frequency")
    parser.add_argument("--window-sec", type=float, default=10.0, help="Plot window length in seconds")
    parser.add_argument("--img-shape", type=_parse_img_shape, default="480,1280")
    parser.add_argument("--binocular", action="store_true", help="Use binocular image shape semantics")
    parser.add_argument("--cert", type=str, default=None, help="SSL cert path")
    parser.add_argument("--key", type=str, default=None, help="SSL key path")
    args = parser.parse_args()

    sample_dt = 1.0 / max(args.hz, 1.0)
    maxlen = max(2, int(args.window_sec / sample_dt))

    data_lock = threading.Lock()

    tv = TeleVuerWrapper(
        use_hand_tracking=False,
        binocular=args.binocular,
        img_shape=args.img_shape,
        display_mode=args.display_mode,
        zmq=False,
        webrtc=False,
        webrtc_url=None,
        cert_file=args.cert,
        key_file=args.key,
        return_hand_rot_data=False,
        use_clutch=True
    )

    stop_event = threading.Event()
    latest_pose = {"lw": None, "rw": None}

    def sampler():
        while not stop_event.is_set():
            tele = tv.get_tele_data()

            with data_lock:
                latest_pose["lw"] = tele.left_wrist_pose.copy()
                latest_pose["rw"] = tele.right_wrist_pose.copy()

            time.sleep(sample_dt)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()

    plotter = PosePlotter(
        history=maxlen,
        title="XR Wrist Trajectories",
        axis_limits=((-0.2, 0.8), (-0.5, 0.5), (-0.5, 0.5)),
        view_init=(0, 200),
        axis_scale=0.08,
    )

    try:
        while True:
            with data_lock:
                lw_pose = latest_pose["lw"]
                rw_pose = latest_pose["rw"]

            if lw_pose is not None:
                plotter.update_matrix(lw_pose, label="left_wrist", draw_axes=True, draw=False)
            if rw_pose is not None:
                plotter.update_matrix(rw_pose, label="right_wrist", draw_axes=True, draw=False)

            plotter.draw()
            time.sleep(sample_dt)
    finally:
        stop_event.set()
        thread.join(timeout=1.0)
        tv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
