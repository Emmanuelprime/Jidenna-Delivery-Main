"""
test_wp_controller_with_map_debug.py

Diagnostic version. Prints progress at every step.
"""

import time
from jidenna.jidenna_runtime    import JidennaRuntime, RuntimeConfig
from jidenna.jidenna_controller import ControllerManager
from jidenna.controllers        import WaypointController


def main():
    print("[1/6] Constructing runtime...")
    cfg = RuntimeConfig(
        nano_port="/dev/ttyUSB0",
        lidar_port="/dev/ttyUSB1",
    )
    rt = JidennaRuntime(cfg)
    print("      runtime object created")

    print("[2/6] Starting runtime (waiting for Nano + LiDAR)...")
    ok = rt.start(timeout_s=20.0)
    print(f"      start() returned: {ok}")
    print(f"      bridge connected: {rt.bridge.is_connected()}")
    print(f"      lidar  connected: {rt.lidar.is_connected()}")
    print(f"      pose available:   {rt.pose.get_pose()}")

    if not rt.is_ready():
        print("      Runtime not ready — aborting")
        rt.stop()
        return

    print("[3/6] Creating controller...")
    ctrl = WaypointController(v_max=0.15, w_max=0.50, goal_tolerance=0.10)
    mgr  = ControllerManager(rt.bridge, rt.pose, ctrl, rate_hz=20.0)
    print("      controller + manager ready")

    print("[4/6] Entering main loop. Type 'x y' to drive, 'q' to quit.")
    try:
        while True:
            print("[5/6] Waiting for input...")
            line = input("target> ").strip()
            print(f"      got input: '{line}'")

            if line.lower() in ("q", "quit"):
                print("      quitting")
                break

            if line.lower() == "map":
                grid = rt.costmap.snapshot()
                occupied = int((grid > 0.4).sum())
                print(f"      costmap: {occupied} occupied cells")
                continue

            parts = line.split()
            if len(parts) != 2:
                print("      use: x y")
                continue

            try:
                tx, ty = float(parts[0]), float(parts[1])
            except ValueError:
                print("      bad numbers")
                continue

            print(f"[6/6] Driving to ({tx:+.2f}, {ty:+.2f})...")
            ctrl.set_target(tx, ty)
            print(f"      target set on controller: {ctrl.get_target()}")

            mgr.start()
            print(f"      manager started, thread alive: {mgr._thread is not None}")

            t0 = time.time()
            while not mgr.wait_until_done(timeout=0.5):
                elapsed = time.time() - t0
                if elapsed > 30:
                    print(f"      timeout after {elapsed:.1f}s")
                    break
                # Print current state every 2 s
                if int(elapsed) % 2 == 0 and int(elapsed * 2) % 4 == 0:
                    p = rt.pose.get_pose()
                    o = mgr.get_last_output()
                    print(f"      t={elapsed:4.1f}s  pose=({p.x:+.2f},{p.y:+.2f})  "
                          f"v={o.v:+.2f} w={o.w:+.2f}  state={o.info.get('state','?')}")

            mgr.stop()
            p = rt.pose.get_pose()
            print(f"      reached at ({p.x:+.3f}, {p.y:+.3f})")

    except KeyboardInterrupt:
        print("\n      Ctrl-C — shutting down")
    finally:
        print("Shutting down...")
        mgr.stop()
        rt.stop()
        print("Done.")


if __name__ == "__main__":
    main()