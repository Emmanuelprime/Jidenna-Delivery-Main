"""
drive_and_map.py

Drive Jidenna with the waypoint controller while the costmap builds.

One process owns everything:
    - Nano bridge + EKF pose
    - LiDAR + costmap
    - Waypoint controller

Type targets at the prompt, watch the map fill in live.
Ctrl-C to stop.

Commands:
    <x> <y>       drive to (x, y)
    m             print costmap stats
    s / stop      emergency stop (zero velocity, session stays alive)
    r             reset pose to (0, 0, 0)
    q / quit      exit cleanly
    h / help      show help
"""

import math
import time
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from jidenna.jidenna_bridge      import JidennaBridge
from jidenna.jidenna_pose        import JidennaPose
from jidenna.jidenna_lidar       import Lidar
from jidenna.jidenna_controller  import ControllerManager
from jidenna.controllers         import WaypointController
from jidenna.perception.costmap  import Costmap


# =============================================================================
# CONFIG — edit these
# =============================================================================

NANO_PORT  = "/dev/ttyUSB0"
LIDAR_PORT = "/dev/ttyUSB1"

LIDAR_DX   = -0.432
LIDAR_DY   =  0.000

V_MAX = 0.15
W_MAX = 0.50
GOAL_TOLERANCE = 0.10

RESOLUTION_M    = 0.05
SIZE_X_M        = 20.0
SIZE_Y_M        = 20.0
LIDAR_MIN_RANGE = 0.20
LIDAR_MAX_RANGE = 8.00

PLOT_HZ  = 5
X_RANGE  = 6.0
Y_RANGE  = 6.0

# =============================================================================


def print_help():
    print("  Commands:")
    print("    <x> <y>     drive to (x, y)")
    print("    m           print costmap stats")
    print("    s / stop    emergency stop")
    print("    r           reset pose estimate to (0, 0, 0)")
    print("    q / quit    quit cleanly")
    print("    h / help    show this help")


def main():
    print("=" * 70)
    print("Jidenna — drive + costmap")
    print("=" * 70)
    print_help()
    print()

    # ---- Construct ------------------------------------------------------
    print("[1] Constructing bridge, pose, lidar, costmap, controller...")
    bridge = JidennaBridge(NANO_PORT)
    pose   = JidennaPose(bridge)
    lidar  = Lidar(LIDAR_PORT)
    cm     = Costmap(
        resolution=RESOLUTION_M,
        size_x=SIZE_X_M, size_y=SIZE_Y_M,
        lidar_dx=LIDAR_DX, lidar_dy=LIDAR_DY,
        lidar_min_range=LIDAR_MIN_RANGE,
        lidar_max_range=LIDAR_MAX_RANGE,
    )

    ctrl = WaypointController(
        v_max=V_MAX, w_max=W_MAX,
        k_v=1.0, k_w=2.0,
        goal_tolerance=GOAL_TOLERANCE,
        heading_gate=0.20, heading_gate_wide=0.60,
    )
    mgr = ControllerManager(bridge, pose, ctrl, rate_hz=20.0)

    # ---- Wire LiDAR -> costmap -----------------------------------------
    print("[2] Wiring LiDAR scans into costmap...")
    def on_scan(scan):
        p = pose.get_pose()
        cm.integrate_scan(scan, p)
    lidar.add_scan_callback(on_scan)

    # ---- Start ---------------------------------------------------------
    print("[3] Starting bridge, pose, lidar...")
    bridge.start()
    pose.start()
    lidar.start()

    print("[4] Waiting for connection...")
    t0 = time.time()
    while time.time() - t0 < 15.0:
        if bridge.is_connected() and lidar.is_connected():
            break
        time.sleep(0.1)

    if not bridge.is_connected() or not lidar.is_connected():
        print("ERROR: not connected")
        print(f"  bridge: {bridge.is_connected()}")
        print(f"  lidar:  {lidar.is_connected()}")
        lidar.stop(); bridge.stop()
        return

    p0 = pose.get_pose()
    print(f"    Connected. Pose: ({p0.x:+.3f}, {p0.y:+.3f}, "
          f"{math.degrees(p0.th):+.1f}°)")
    print()

    # ---- Plot setup ----------------------------------------------------
    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    extent = [cm.origin_x, cm.origin_x + cm.size_x,
              cm.origin_y, cm.origin_y + cm.size_y]

    grid_img = ax.imshow(
        np.zeros_like(cm.grid),
        origin="lower", extent=extent,
        cmap="Greys", vmin=0.0, vmax=1.0,
        interpolation="nearest", alpha=0.9,
    )
    scan_scatter = ax.scatter([], [], s=2, c="r", alpha=0.6,
                              label="LiDAR scan")
    robot_arrow = ax.arrow(0, 0, 0.3, 0, head_width=0.08,
                           head_length=0.12, fc="b", ec="b",
                           zorder=10, label="robot")
    target_marker, = ax.plot([], [], "g*", markersize=15,
                             markeredgewidth=1, label="target")
    ax.plot(0, 0, "g+", markersize=12, markeredgewidth=2, alpha=0.5)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("Costmap + scan + robot  (type targets in terminal)")
    ax.legend(loc="upper right")

    last_draw = 0.0
    draw_period = 1.0 / PLOT_HZ

    # ---- Driving loop --------------------------------------------------
    # We drive in the main thread when a target is active, and only
    # return to the input prompt when the goal is reached or the user
    # hits Ctrl-C. The plot updates on every tick of the drive loop.
    try:
        while True:
            try:
                line = input("target> ").strip()
            except EOFError:
                break

            if not line:
                continue

            low = line.lower()
            if low in ("q", "quit", "exit"):
                break
            if low in ("h", "help", "?"):
                print_help(); continue
            if low in ("s", "stop"):
                bridge.set_velocity(0.0, 0.0)
                print("  emergency stop")
                continue
            if low == "r":
                pose.reset(0.0, 0.0, 0.0)
                bridge.reset_odometry()
                time.sleep(0.3)
                p = pose.get_pose()
                print(f"  reset. pose = ({p.x:+.3f}, {p.y:+.3f})")
                continue
            if low == "m":
                grid = cm.snapshot()
                touched = int((grid != 0).sum())
                occupied = int((grid > 0.4).sum())
                print(f"  costmap: {touched} touched, {occupied} occupied cells")
                continue

            # Parse target
            parts = line.replace(",", " ").split()
            if len(parts) != 2:
                print("  use: x y")
                continue
            try:
                tx, ty = float(parts[0]), float(parts[1])
            except ValueError:
                print("  bad numbers")
                continue

            print(f">>> target ({tx:+.2f}, {ty:+.2f})")
            ctrl.set_target(tx, ty)
            target_marker.set_data([tx], [ty])
            mgr.start()

            # Drive loop: update plot while waiting for done
            t_start = time.time()
            reached = False
            while True:
                now = time.time()
                elapsed = now - t_start
                if elapsed > 60.0:
                    print(f"  TIMEOUT after {elapsed:.1f}s")
                    break

                # Draw
                if now - last_draw >= draw_period:
                    last_draw = now
                    grid_img.set_data(cm.probability_grid())

                    p = pose.get_pose()
                    robot_arrow.remove()
                    dx = 0.35 * math.cos(p.th)
                    dy = 0.35 * math.sin(p.th)
                    robot_arrow = ax.arrow(
                        p.x, p.y, dx, dy,
                        head_width=0.08, head_length=0.12,
                        fc="b", ec="b", zorder=10)

                    # scan points in odom frame
                    o = mgr.get_last_output()
                    ax.set_xlim(p.x - X_RANGE, p.x + X_RANGE)
                    ax.set_ylim(p.y - Y_RANGE, p.y + Y_RANGE)
                    plt.draw()
                    plt.pause(0.001)

                if mgr.wait_until_done(timeout=0.05):
                    reached = True
                    break

            mgr.stop()
            p = pose.get_pose()
            err = math.hypot(tx - p.x, ty - p.y)
            tag = "✓ reached" if reached else "✗ timeout"
            print(f"  {tag} ({p.x:+.3f}, {p.y:+.3f})  err={err:.3f} m")

    except KeyboardInterrupt:
        print("\nCtrl-C — shutting down")
    finally:
        print("Stopping...")
        try:
            bridge.set_velocity(0.0, 0.0)
        except Exception:
            pass
        mgr.stop()
        lidar.stop()
        bridge.stop()
        plt.close("all")

        # Save final map
        try:
            fig2, ax2 = plt.subplots(figsize=(10, 10))
            ax2.imshow(cm.probability_grid(), origin="lower",
                       extent=extent, cmap="Greys",
                       vmin=0, vmax=1, interpolation="nearest")
            ax2.set_aspect("equal")
            ax2.set_xlabel("x [m]"); ax2.set_ylabel("y [m]")
            ax2.set_title("Final costmap")
            fig2.savefig("costmap_final.png", dpi=150)
            print("Saved costmap_final.png")
        except Exception as e:
            print(f"save failed: {e}")


if __name__ == "__main__":
    main()