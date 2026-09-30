"""
jidenna_costmap_viewer.py

Live view of the costmap + current LiDAR scan + robot pose.

Requires both the LiDAR and the EKF (so the Jidenna Nano must be on).
Press Ctrl-C to stop.
"""

import math
import time
import numpy as np

import matplotlib
matplotlib.use("TkAgg")   # interactive; change if you don't have Tk
import matplotlib.pyplot as plt

from jidenna.jidenna_lidar           import Lidar
from jidenna.jidenna_pose            import JidennaPose
from jidenna.jidenna_bridge          import JidennaBridge
from jidenna.perception.costmap      import Costmap


# =============================================================================
# CONFIG
# =============================================================================

LIDAR_PORT = "/dev/ttyUSB1"
NANO_PORT  = "/dev/ttyUSB0"      # or COMx

# LiDAR mounting (measured earlier)
LIDAR_DX = -0.432
LIDAR_DY =  0.000

# Costmap parameters
RESOLUTION_M = 0.05               # 5 cm per cell
SIZE_X_M = 20.0
SIZE_Y_M = 20.0

LIDAR_MIN_RANGE = 0.20
LIDAR_MAX_RANGE = 8.0

# Display
PLOT_HZ = 5
X_RANGE = 8.0                    # show ±8 m around the robot
Y_RANGE = 8.0

# =============================================================================


def main():
    print("Starting LiDAR...")
    lidar = Lidar(LIDAR_PORT)

    print("Starting Nano bridge + pose...")
    bridge = JidennaBridge(NANO_PORT)
    pose   = JidennaPose(bridge)

    print("Creating costmap...")
    cm = Costmap(
        resolution=RESOLUTION_M,
        size_x=SIZE_X_M,
        size_y=SIZE_Y_M,
        lidar_dx=LIDAR_DX,
        lidar_dy=LIDAR_DY,
        lidar_min_range=LIDAR_MIN_RANGE,
        lidar_max_range=LIDAR_MAX_RANGE,
    )

    # Latest scan for display
    latest = {"scan": None, "t": 0.0}

    def on_scan(scan):
        p = pose.get_pose()
        cm.integrate_scan(scan, p)
        latest["scan"] = scan
        latest["t"] = time.time()

    lidar.add_scan_callback(on_scan)

    # Start everything
    bridge.start()
    pose.start()
    lidar.start()

    # Wait for both to be ready
    print("Waiting for Nano and LiDAR...")
    t0 = time.time()
    while time.time() - t0 < 15.0:
        if bridge.is_connected() and lidar.is_connected():
            break
        time.sleep(0.1)

    if not bridge.is_connected():
        print("ERROR: Nano not connected")
        lidar.stop(); bridge.stop()
        return
    if not lidar.is_connected():
        print("ERROR: LiDAR not connected")
        lidar.stop(); bridge.stop()
        return

    print("Both connected. Press Ctrl-C to stop.")

    # ---- Live plot ------------------------------------------------------
    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Occupancy image
    # We show the log-odds grid as a grayscale heatmap in world coordinates.
    extent = [cm.origin_x, cm.origin_x + cm.size_x,
              cm.origin_y, cm.origin_y + cm.size_y]
    grid_img = ax.imshow(
        np.zeros_like(cm.grid),
        origin="lower", extent=extent,
        cmap="Greys", vmin=0.0, vmax=1.0,
        interpolation="nearest", alpha=0.8,
    )

    # Scan scatter
    scan_scatter = ax.scatter([], [], s=2, c="r", alpha=0.5,
                              label="current scan")

    # Robot pose arrow
    robot_arrow = ax.arrow(0, 0, 0.3, 0, head_width=0.1, head_length=0.15,
                           fc="b", ec="b", zorder=10)

    # Target / origin markers
    ax.plot(0, 0, "g+", markersize=12, markeredgewidth=2, label="odom origin")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Costmap (grayscale) + current scan (red) + robot (blue)")
    ax.legend(loc="upper right")

    last_draw = 0.0
    draw_period = 1.0 / PLOT_HZ

    try:
        while True:
            now = time.time()
            if now - last_draw < draw_period:
                time.sleep(0.02)
                continue
            last_draw = now

            # Update occupancy image
            prob = cm.probability_grid()
            grid_img.set_data(prob)

            # Update scan scatter
            scan = latest["scan"]
            p = pose.get_pose()
            if scan is not None and len(scan) > 0:
                c = math.cos(p.th)
                s = math.sin(p.th)
                x_robot = scan.points_xy[:, 0] + LIDAR_DX
                y_robot = scan.points_xy[:, 1] + LIDAR_DY
                xs = x_robot * c - y_robot * s + p.x
                ys = x_robot * s + y_robot * c + p.y
                scan_scatter.set_offsets(np.column_stack((xs, ys)))

            # Robot arrow: clear and redraw
            robot_arrow.remove()
            length = 0.35
            dx = length * math.cos(p.th)
            dy = length * math.sin(p.th)
            robot_arrow = ax.arrow(p.x, p.y, dx, dy,
                                   head_width=0.08, head_length=0.10,
                                   fc="b", ec="b", zorder=10)

            # Auto-follow the robot
            ax.set_xlim(p.x - X_RANGE, p.x + X_RANGE)
            ax.set_ylim(p.y - Y_RANGE, p.y + Y_RANGE)

            plt.draw()
            plt.pause(0.001)

    except KeyboardInterrupt:
        print("\nCtrl-C received. Shutting down...")
    finally:
        lidar.stop()
        bridge.stop()
        plt.close("all")


if __name__ == "__main__":
    main()