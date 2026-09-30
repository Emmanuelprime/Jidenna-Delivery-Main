"""
jidenna_costmap_viewer.py

Live costmap viewer. Uses JidennaRuntime so the Nano port is opened once
and shared with any other part of the same process.

Run:  python jidenna_costmap_viewer.py
Press Ctrl-C to stop.
"""

import math
import time
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from jidenna.jidenna_runtime import JidennaRuntime, RuntimeConfig


# =============================================================================
# CONFIG
# =============================================================================

PLOT_HZ  = 5
X_RANGE  = 8.0
Y_RANGE  = 8.0


def main():
    cfg = RuntimeConfig(
        nano_port="/dev/ttyUSB0",
        lidar_port="/dev/ttyUSB1",
        lidar_dx=-0.432,
        lidar_dy=0.000,
        costmap_resolution=0.05,
        costmap_size_x=20.0,
        costmap_size_y=20.0,
        lidar_min_range=0.20,
        lidar_max_range=8.0,
    )

    with JidennaRuntime(cfg) as rt:
        if not rt.is_ready():
            print("ERROR: runtime not ready")
            return

        print("Runtime ready. Press Ctrl-C to stop.")

        # ---- Plot setup -------------------------------------------------
        plt.ion()
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        cm = rt.costmap
        extent = [cm.origin_x, cm.origin_x + cm.size_x,
                  cm.origin_y, cm.origin_y + cm.size_y]

        grid_img = ax.imshow(
            np.zeros_like(cm.grid),
            origin="lower", extent=extent,
            cmap="Greys", vmin=0.0, vmax=1.0,
            interpolation="nearest", alpha=0.8,
        )

        scan_scatter = ax.scatter([], [], s=2, c="r", alpha=0.5)
        robot_arrow  = ax.arrow(0, 0, 0.3, 0, head_width=0.1,
                                head_length=0.15, fc="b", ec="b", zorder=10)
        ax.plot(0, 0, "g+", markersize=12, markeredgewidth=2)

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title("Costmap + scan + robot  (Ctrl-C to stop)")

        last_draw = 0.0
        draw_period = 1.0 / PLOT_HZ

        try:
            while True:
                now = time.time()
                if now - last_draw < draw_period:
                    time.sleep(0.02)
                    continue
                last_draw = now

                # Update occupancy
                grid_img.set_data(cm.probability_grid())

                # Update robot arrow
                p = rt.pose.get_pose()
                robot_arrow.remove()
                dx = 0.35 * math.cos(p.th)
                dy = 0.35 * math.sin(p.th)
                robot_arrow = ax.arrow(p.x, p.y, dx, dy,
                                       head_width=0.08, head_length=0.10,
                                       fc="b", ec="b", zorder=10)

                # Auto-follow
                ax.set_xlim(p.x - X_RANGE, p.x + X_RANGE)
                ax.set_ylim(p.y - Y_RANGE, p.y + Y_RANGE)

                plt.draw()
                plt.pause(0.001)

        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            plt.close("all")


if __name__ == "__main__":
    main()