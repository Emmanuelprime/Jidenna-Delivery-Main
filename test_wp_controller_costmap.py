"""
test_wp_controller_with_map.py

Drive waypoints while the costmap builds. Uses JidennaRuntime.
"""

import math
import time
from jidenna.jidenna_runtime     import JidennaRuntime, RuntimeConfig
from jidenna.jidenna_controller  import ControllerManager
from jidenna.controllers         import WaypointController


def main():
    cfg = RuntimeConfig(
        nano_port="/dev/ttyUSB0",
        lidar_port="/dev/ttyUSB1",
    )

    with JidennaRuntime(cfg) as rt:
        if not rt.is_ready():
            print("Runtime not ready")
            return

        ctrl = WaypointController(v_max=0.15, w_max=0.50, goal_tolerance=0.10)
        mgr  = ControllerManager(rt.bridge, rt.pose, ctrl, rate_hz=20.0)

        try:
            while True:
                line = input("target> ").strip()
                if line.lower() in ("q", "quit"):
                    break
                if line.lower() == "map":
                    # Print number of occupied cells
                    grid = rt.costmap.snapshot()
                    occupied = int((grid > 0.4).sum())
                    print(f"  costmap: {occupied} occupied cells")
                    continue
                parts = line.split()
                if len(parts) != 2:
                    print("  use: x y")
                    continue
                tx, ty = float(parts[0]), float(parts[1])

                ctrl.set_target(tx, ty)
                mgr.start()
                t0 = time.time()
                while not mgr.wait_until_done(timeout=0.5):
                    if time.time() - t0 > 30:
                        print("  timeout")
                        break
                mgr.stop()
                p = rt.pose.get_pose()
                print(f"  reached at ({p.x:+.3f}, {p.y:+.3f})")
        except KeyboardInterrupt:
            print("\nInterrupted")
        finally:
            mgr.stop()