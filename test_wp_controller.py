"""
test_wp_controller.py

Interactive single-waypoint live test. Press Play in VSCode to run.

Type a target as "x y" (e.g. "0.3 0.5"), press Enter, and the robot
drives there. When it arrives, type the next target.

Commands:
    <x> <y>     drive to (x, y)
    r           reset pose estimate to (0, 0, 0) and re-anchor IMU
    s / stop    emergency stop (zero velocity, keep session alive)
    q / quit    quit cleanly
    h / help    show this help
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

PORT = "/dev/ttyUSB0"
# PORT = "COM19"                       # or "/dev/ttyUSB0" on Linux

V_MAX = 0.15                         # max forward speed (m/s)
W_MAX = 0.50                         # max turn rate (rad/s)

GOAL_TOLERANCE = 0.10                # "reached" radius (m)

LOG_PATH = "runs/live_session_01.csv"   # set to None to disable

RUN_TIMEOUT_S = 60.0                 # per-waypoint timeout
PRINT_HZ      = 4                    # live pose print rate

# =============================================================================
# DO NOT EDIT BELOW
# =============================================================================


def parse_target(line: str):
    """Parse 'x y' into (x, y). Returns None on failure."""
    parts = line.replace(",", " ").split()
    if len(parts) != 2:
        return None
    try:
        return (float(parts[0]), float(parts[1]))
    except ValueError:
        return None


def print_help():
    print("  Commands:")
    print("    <x> <y>     drive to (x, y)")
    print("    r           reset pose estimate to (0, 0, 0)")
    print("    s / stop    emergency stop")
    print("    q / quit    quit cleanly")
    print("    h / help    show this help")


def run_waypoint(mgr, pose, ctrl, target_x, target_y,
                 goal_tolerance, timeout_s, print_hz):
    """Drive to one waypoint. Returns True if reached."""
    print(f"\n>>> target ({target_x:+.3f}, {target_y:+.3f})")
    ctrl.set_target(target_x, target_y)

    t_start = time.time()
    last_print = 0.0
    print_period = 1.0 / print_hz

    while True:
        now = time.time()
        elapsed = now - t_start

        if elapsed > timeout_s:
            print(f"TIMEOUT after {elapsed:.1f}s")
            return False

        if now - last_print >= print_period:
            last_print = now
            p = pose.get_pose()
            o = mgr.get_last_output()
            state = o.info.get("state", "?")
            dist  = o.info.get("distance", 0.0)
            herr  = math.degrees(o.info.get("heading_error", 0.0))
            print(f"  t={elapsed:5.1f}s  "
                  f"pose=({p.x:+.3f},{p.y:+.3f},{math.degrees(p.th):+6.1f}°)  "
                  f"v={o.v:+.2f} w={o.w:+.2f}  "
                  f"state={state:7s}  "
                  f"dist={dist:.3f}  herr={herr:+6.1f}°")

        if mgr.wait_until_done(timeout=0.1):
            p = pose.get_pose()
            err = math.hypot(target_x - p.x, target_y - p.y)
            ok = err < goal_tolerance * 1.5
            if ok:
                print(f"  ✓ reached ({p.x:+.3f}, {p.y:+.3f})  err={err:.3f} m")
            else:
                print(f"  WARNING: controller reported done but error is "
                      f"{err:.3f} m")
            return ok


def main() -> int:
    print("=" * 70)
    print(f"Jidenna interactive waypoint controller")
    print(f"  port      : {PORT}")
    print(f"  v_max     : {V_MAX} m/s")
    print(f"  w_max     : {W_MAX} rad/s")
    print(f"  tolerance : {GOAL_TOLERANCE} m")
    print(f"  log       : {LOG_PATH}")
    print("=" * 70)
    print_help()
    print()

    # ---- Construct ------------------------------------------------------
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
        logger = CsvLogger(Path(LOG_PATH), comment="interactive session")
        bridge.add_telemetry_callback(logger)
        pose.add_pose_callback(logger.on_pose)

    # ---- Start ----------------------------------------------------------
    print("Opening port...")
    bridge.start()
    pose.start()

    t0 = time.time()
    while not bridge.is_connected() and time.time() - t0 < 15.0:
        time.sleep(0.1)

    if not bridge.is_connected():
        print("ERROR: no telemetry from Nano. Check cable/port.")
        bridge.stop()
        if logger: logger.close()
        return 1

    p0 = pose.get_pose()
    print(f"Connected. Initial pose: x={p0.x:+.3f} y={p0.y:+.3f} "
          f"th={math.degrees(p0.th):+.1f}°")
    print()

    # ---- Interactive loop -----------------------------------------------
    try:
        while True:
            try:
                line = input("target> ").strip()
            except EOFError:
                print("\nEOF, quitting.")
                break

            if not line:
                continue

            low = line.lower()
            if low in ("q", "quit", "exit"):
                print("Quitting.")
                break
            if low in ("h", "help", "?"):
                print_help()
                continue
            if low in ("s", "stop"):
                print("Emergency stop.")
                bridge.set_velocity(0.0, 0.0)
                continue
            if low == "r":
                pose.reset(0.0, 0.0, 0.0)
                bridge.reset_odometry()
                time.sleep(0.3)
                p = pose.get_pose()
                print(f"Reset. Pose now: x={p.x:+.3f} y={p.y:+.3f} "
                      f"th={math.degrees(p.th):+.1f}°")
                continue

            tgt = parse_target(line)
            if tgt is None:
                print(f"  Could not parse '{line}'. Type two numbers "
                      f"(e.g. '0.3 0.5') or 'h' for help.")
                continue

            tx, ty = tgt
            p = pose.get_pose()
            dist = math.hypot(tx - p.x, ty - p.y)
            print(f"Current pose: ({p.x:+.3f}, {p.y:+.3f}, "
                  f"{math.degrees(p.th):+.1f}°)")
            print(f"Target      : ({tx:+.3f}, {ty:+.3f})  dist={dist:.3f} m")

            mgr.start()   # no-op if already running
            run_waypoint(mgr, pose, ctrl, tx, ty,
                         GOAL_TOLERANCE, RUN_TIMEOUT_S, PRINT_HZ)

    except KeyboardInterrupt:
        print("\nCtrl-C received. Stopping.")

    finally:
        print("Shutting down...")
        try:
            bridge.set_velocity(0.0, 0.0)
        except Exception:
            pass
        mgr.stop()
        bridge.stop()
        if logger:
            logger.close()
            print(f"Log saved: {LOG_PATH}  ({logger.count} samples)")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)