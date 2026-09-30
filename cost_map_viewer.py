"""
test_costmap_only.py

Build a costmap from LiDAR + EKF while pushing the robot by hand.
No controller. No waypoints. Just map-building and visualization.

Press Ctrl-C to stop.
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
from jidenna.perception.costmap  import Costmap


# =============================================================================
# CONFIG — edit these
# =============================================================================

NANO_PORT  = "/dev/ttyUSB0"      # Nano (telemetry, odometry)
LIDAR_PORT = "/dev/ttyUSB1"      # YDLIDAR

# LiDAR mount in robot frame (meters)
LIDAR_DX = -0.432
LIDAR_DY =  0.000

# Costmap
RESOLUTION_M = 0.05
SIZE_X_M     = 20.0
SIZE_Y_M     = 20.0
LIDAR_MIN_RANGE = 0.20
LIDAR_MAX_RANGE = 8.00

# Display
PLOT_HZ  = 5
X_RANGE  = 8.0
Y_RANGE  = 8.0

# =============================================================================


def main():
    print("[1] Creating bridge, pose, lidar, costmap...")
    bridge = JidennaBridge(NANO_PORT)
    pose   = JidennaPose(bridge)
    lidar  = Lidar(LIDAR_PORT)
    cm     = Costmap(
        resolution=RESOLUTION_M,
        size_x=SIZE_X_M,
        size_y=SIZE_Y_M,
        lidar_dx=LIDAR_DX,
        lidar_dy=LIDAR_DY,
        lidar_min_range=LIDAR_MIN_RANGE,
        lidar_max_range=LIDAR_MAX_RANGE,
    )

    print("[2] Wiring LiDAR -> costmap (using current pose)...")
    def on_scan(scan):
        p = pose.get_pose()
        cm.integrate_scan(scan, p)
    lidar.add_scan_callback(on_scan)

    print("[3] Starting bridge, pose, lidar...")
    bridge.start()
    pose.start()
    lidar.start()

    print("[4] Waiting for both to connect...")
    t0 = time.time()
    while time.time() - t0 < 15.0:
        b_ok = bridge.is_connected()
        l_ok = lidar.is_connected()
        if b_ok and l_ok:
            break
        time.sleep(0.1)

    print(f"    bridge connected: {bridge.is_connected()}")
    print(f"    lidar  connected: {lidar.is_connected()}")

    if not bridge.is_connected():
        print("ERROR: Nano not producing telemetry. Check port.")
        lidar.stop(); bridge.stop()
        return
    if not lidar.is_connected():
        print("ERROR: LiDAR not connected. Check port.")
        lidar.stop(); bridge.stop()
        return

    # Sanity check: is the pose updating?
    print("[5] Checking pose is alive (2 s)...")
    t0 = time.time()
    p0 = pose.get_pose()
    while time.time() - t0 < 2.0:
        time.sleep(0.2)
    p1 = pose.get_pose()
    if p0.x == p1.x and p0.y == p1.y and p0.th == p1.th and \
       abs(p1.P_xx - p0.P_xx) < 1e-9:
        print("    WARNING: pose not updating. EKF may not be receiving telemetry.")
        print(f"    p0 = {p0}")
        print(f"    p1 = {p1}")
    else:
        print(f"    pose alive. p0=({p0.x:+.3f},{p0.y:+.3f})  "
              f"p1=({p1.x:+.3f},{p1.y:+.3f})")

    print("[6] Building costmap. Push the robot by hand. Ctrl-C to stop.")

    # ---- Live plot ------------------------------------------------------
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
    scan_scatter = ax.scatter([], [], s=2, c="r", alpha=0.6)
    robot_arrow  = ax.arrow(0, 0, 0.3, 0, head_width=0.08, head_length=0.12,
                            fc="b", ec="b", zorder=10)
    ax.plot(0, 0, "g+", markersize=12, markeredgewidth=2)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Costmap (grayscale) + current scan (red) + robot (blue)")

    last_draw = 0.0
    draw_period = 1.0 / PLOT_HZ

    try:
        while True:
            now = time.time()
            if now - last_draw < draw_period:
                time.sleep(0.02)
                continue
            last_draw = now

            grid_img.set_data(cm.probability_grid())

            p = pose.get_pose()
            robot_arrow.remove()
            dx = 0.35 * math.cos(p.th)
            dy = 0.35 * math.sin(p.th)
            robot_arrow = ax.arrow(p.x, p.y, dx, dy,
                                   head_width=0.08, head_length=0.12,
                                   fc="b", ec="b", zorder=10)

            ax.set_xlim(p.x - X_RANGE, p.x + X_RANGE)
            ax.set_ylim(p.y - Y_RANGE, p.y + Y_RANGE)

            plt.draw()
            plt.pause(0.001)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        lidar.stop()
        bridge.stop()
        plt.close("all")

        # Save a snapshot of the costmap
        try:
            import matplotlib.image as mpimg
            fig2, ax2 = plt.subplots(figsize=(10, 10))
            ax2.imshow(cm.probability_grid(), origin="lower",
                       extent=extent, cmap="Greys", vmin=0, vmax=1)
            ax2.set_aspect("equal")
            ax2.set_xlabel("x [m]")
            ax2.set_ylabel("y [m]")
            ax2.set_title("Final costmap snapshot")
            fig2.savefig("costmap_final.png", dpi=100)
            print("Saved costmap_final.png")
        except Exception as e:
            print(f"Could not save snapshot: {e}")

        print(f"Costmap cells: {int((cm.snapshot() > 0).sum())} touched")


if __name__ == "__main__":
    main()