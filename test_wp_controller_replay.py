"""
test_controller_replay.py

Replays a logged CSV through the controller. Prints what the controller
WOULD have commanded at each timestep. Useful for tuning on real data
without a live robot.

Usage:
    python test_wp_controller_replay.py runs/forward_01.csv --target 1.0 0.0
"""

import argparse
import math
import sys
from pathlib import Path

from jidenna.jidenna_analyze import load_csv
from jidenna.controllers import WaypointController
from jidenna.controllers.base import ControlOutput
from dataclasses import dataclass


@dataclass
class Pose:
    x: float = 0.0
    y: float = 0.0
    th: float = 0.0
    v: float = 0.0
    w: float = 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="input CSV with EKF columns")
    ap.add_argument("--target", nargs=2, type=float, required=True,
                    metavar=("X", "Y"), help="waypoint target")
    args = ap.parse_args()

    path = Path(args.csv)
    if not path.exists():
        print(f"ERROR: {path} not found")
        return 1

    data = load_csv(path)

    # Prefer EKF columns if present; fall back to raw odom
    if "ekf_x" in data and data["ekf_x"].size > 0:
        xs, ys, ths = data["ekf_x"], data["ekf_y"], data["ekf_th"]
        source = "EKF"
    else:
        xs, ys, ths = data["x"], data["y"], -data["th"]
        source = "raw odom"

    ctrl = WaypointController()
    ctrl.set_target(args.target[0], args.target[1])

    print(f"Replaying {path.name} ({source} pose), "
          f"target = ({args.target[0]:+.2f}, {args.target[1]:+.2f})")
    print()
    print("  t [s]    pose                          ctrl output              state")
    print("  " + "-"*78)

    t = data["t_host"] - data["t_host"][0]

    for k in range(len(t)):
        pose = Pose(x=xs[k], y=ys[k], th=ths[k])
        out = ctrl.compute(pose)

        if k % 5 == 0 or out.done:
            print(f"  {t[k]:5.2f}  "
                  f"({pose.x:+.2f},{pose.y:+.2f},{math.degrees(pose.th):+6.1f}°)  "
                  f"v={out.v:+.2f} w={out.w:+.2f}  "
                  f"{out.info.get('state','?'):7s}  "
                  f"dist={out.info.get('distance',0):.3f}")

        if out.done:
            print(f"\nController would report DONE at t = {t[k]:.2f}s")
            break
    else:
        print(f"\nController never reached the target (ran out of data).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())