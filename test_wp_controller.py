"""
test_wp_controller.py

Single-waypoint live test. Press Play in VSCode to run.

Edit the CONFIG block below, then hit Run. Ctrl-C to abort at any time.
"""

import math
import time
from pathlib import Path

from jidenna.jidenna_bridge      import JidennaBridge
from jidenna.jidenna_pose        import JidennaPose
from jidenna.jidenna_controller  import ControllerManager
from jidenna.controllers         import WaypointController
from jidenna.jidenna_logger      import CsvLogger


# =============================================================================
# CONFIG — edit these
# =============================================================================

PORT = "COM19"

TARGET_X = 01.0                      # waypoint x (m, relative to start)
TARGET_Y = 0.00                      # waypoint y (m, relative to start)

V_MAX = 0.15                         # max forward speed (m/s)
W_MAX = 0.50                         # max turn rate (rad/s)

GOAL_TOLERANCE = 0.10                # "reached" radius (m)

LOG_PATH = "runs/live_test_01.csv"   # set to None to disable logging

RUN_TIMEOUT_S = 30.0                 # hard timeout for the whole run
PRINT_HZ      = 4

# =============================================================================
# DO NOT EDIT BELOW
# =============================================================================


def main() -> int:
    print("=" * 70)
    print(f"Jidenna single-waypoint live test")
    print(f"  port      : {PORT}")
    print(f"  target    : ({TARGET_X:+.2f}, {TARGET_Y:+.2f})")
    print(f"  v_max     : {V_MAX} m/s")
    print(f"  w_max     : {W_MAX} rad/s")
    print(f"  tolerance : {GOAL_TOLERANCE} m")
    print(f"  log       : {LOG_PATH}")
    print("=" * 70)

    # ---- Construct everything ------------------------------------------
    bridge = JidennaBridge(PORT)
    pose   = JidennaPose(bridge)

    ctrl = WaypointController(
        v_max=V_MAX,
        w_max=W_MAX,
        k_v=1.0,
        k_w=2.0,
        goal_tolerance=GOAL_TOLERANCE,
        heading_gate=0.20,
        heading_gate_wide=0.60,
    )
    mgr = ControllerManager(bridge, pose, ctrl, rate_hz=20.0)

    logger = None
    if LOG_PATH:
        logger = CsvLogger(Path(LOG_PATH),
                           comment=f"live test to ({TARGET_X},{TARGET_Y})")
        bridge.add_telemetry_callback(logger)
        pose.add_pose_callback(logger.on_pose)

    # ---- Start ----------------------------------------------------------
    print("Opening port...")
    bridge.start()
    pose.start()

    # Wait for connection (Nano takes ~6 s: bootloader + gyro calibration)
    t0 = time.time()
    while not bridge.is_connected() and time.time() - t0 < 15.0:
        time.sleep(0.1)

    if not bridge.is_connected():
        print("ERROR: no telemetry from Nano. Check cable/port.")
        bridge.stop()
        if logger: logger.close()
        return 1

    p0 = pose.get_pose()
    print(f"Connected. Initial pose: x={p0.x:+.3f} y={p0.y:+.3f} th={p0.th:+.3f}")

    # ---- Countdown ------------------------------------------------------
    print("Starting in 3 s... (Ctrl-C to abort)")
    for i in [3, 2, 1]:
        print(f"  {i}...")
        time.sleep(1.0)

    # ---- Set target BEFORE starting the manager -------------------------
    print(f"GO: target ({TARGET_X:+.2f}, {TARGET_Y:+.2f})")
    ctrl.set_target(TARGET_X, TARGET_Y)
    mgr.start()

    # ---- Drive ----------------------------------------------------------
    t_start = time.time()
    last_print = 0.0
    print_period = 1.0 / PRINT_HZ
    reached = False

    try:
        while True:
            now = time.time()
            elapsed = now - t_start

            if elapsed > RUN_TIMEOUT_S:
                print(f"TIMEOUT after {elapsed:.1f}s. Aborting.")
                break

            if now - last_print >= print_period:
                last_print = now
                p = pose.get_pose()
                o = mgr.get_last_output()
                state = o.info.get("state", "?")
                dist  = o.info.get("distance", 0.0)
                print(f"  t={elapsed:5.1f}s  "
                      f"pose=({p.x:+.3f},{p.y:+.3f},{math.degrees(p.th):+6.1f}°)  "
                      f"v={o.v:+.2f} w={o.w:+.2f}  "
                      f"state={state:7s}  dist={dist:.3f}")

            # Check for done
            if mgr.wait_until_done(timeout=0.1):
                # Verify by actual distance, not just the done signal
                p = pose.get_pose()
                err = math.hypot(TARGET_X - p.x, TARGET_Y - p.y)
                if err < GOAL_TOLERANCE * 1.5:
                    reached = True
                else:
                    print(f"  WARNING: controller reported done but error is "
                          f"{err:.3f} m (tolerance {GOAL_TOLERANCE})")
                break
    except KeyboardInterrupt:
        print("\nCtrl-C received. Stopping.")
    finally:
        try:
            bridge.set_velocity(0.0, 0.0)
        except Exception:
            pass
        mgr.stop()
        bridge.stop()
        if logger:
            logger.close()

    # ---- Summary --------------------------------------------------------
    print()
    print("=" * 70)
    pf = pose.get_pose()
    err = math.hypot(TARGET_X - pf.x, TARGET_Y - pf.y)
    print(f"Final pose: x={pf.x:+.3f}  y={pf.y:+.3f}  th={math.degrees(pf.th):+6.1f}°")
    print(f"Target    : x={TARGET_X:+.3f}  y={TARGET_Y:+.3f}")
    print(f"Error     : {err:.3f} m")
    print(f"Result    : {'REACHED' if reached else 'NOT REACHED'}")
    if logger:
        print(f"Log saved : {LOG_PATH}  ({logger.count} samples)")
    print("=" * 70)

    return 0 if reached else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)