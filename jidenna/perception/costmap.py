"""
perception/costmap.py

2D occupancy grid in the odom frame.

Consumes LidarScan and PoseEstimate, produces a probabilistic occupancy
grid. Uses log-odds so repeated observations build/decay confidence.

Frame:
    - LiDAR frame: 0° = forward along LiDAR's 0° axis, CCW+ (confirmed)
    - Robot frame: origin at wheel-axle midpoint, +x forward, +y left
    - Odom frame:  fixed, origin = robot's pose at EKF start

Transform chain per point:
    (r, θ_lidar) → (x_lidar, y_lidar)  [done in lidar driver]
                 → (x_robot, y_robot)  [apply LiDAR mount offset]
                 → (x_odom,  y_odom)   [apply EKF pose]
                 → (i, j)              [grid indices]
"""

from __future__ import annotations

import math
import threading
import numpy as np
from dataclasses import dataclass
from typing import Optional

from jidenna.jidenna_lidar import LidarScan
from jidenna.jidenna_pose  import PoseEstimate


# ---------------------------------------------------------------------------
# Log-odds constants
# ---------------------------------------------------------------------------

# Probability of the cell being occupied, given a "hit" observation.
# Not the same as the true probability — these are inverse-sensor-model
# constants that control how fast the map changes.
P_OCC = 0.70      # probability of "occupied" given a hit
P_FREE = 0.30     # probability of "occupied" given a pass-through

L_OCC  = math.log(P_OCC  / (1.0 - P_OCC))    # ≈ +0.847
L_FREE = math.log(P_FREE / (1.0 - P_FREE))   # ≈ -0.847

# Log-odds clamp — keeps values in a sane range so cells can't get stuck.
L_MIN = -5.0
L_MAX = +5.0


def _logodds_to_prob(l: float) -> float:
    """Convert log-odds to probability."""
    return 1.0 - 1.0 / (1.0 + math.exp(l))


# ---------------------------------------------------------------------------
# Costmap
# ---------------------------------------------------------------------------

class Costmap:
    """
    2D occupancy grid in the odom frame.

    Usage:
        cm = Costmap(resolution=0.05, size_x=20.0, size_y=20.0,
                     lidar_dx=-0.432, lidar_dy=0.0)
        cm.integrate_scan(scan, pose)
        prob = cm.get_occupancy(x, y)     # [0..1]
        occupied = cm.is_occupied(x, y)   # bool
    """

    def __init__(self,
                 resolution: float = 0.05,
                 size_x: float = 20.0,
                 size_y: float = 20.0,
                 lidar_dx: float = 0.0,
                 lidar_dy: float = 0.0,
                 lidar_min_range: float = 0.20,
                 lidar_max_range: float = 8.0):
        self.resolution = float(resolution)
        self.size_x = float(size_x)
        self.size_y = float(size_y)

        # Grid dimensions
        self.nx = int(round(size_x / resolution))
        self.ny = int(round(size_y / resolution))

        # Grid origin: world (0, 0) maps to grid center
        self.origin_x = -size_x / 2.0
        self.origin_y = -size_y / 2.0

        # LiDAR mount offset (meters, robot frame)
        self.lidar_dx = float(lidar_dx)
        self.lidar_dy = float(lidar_dy)

        # Range filter
        self.lidar_min_range = float(lidar_min_range)
        self.lidar_max_range = float(lidar_max_range)

        # The grid: log-odds values, zero = unknown (p=0.5)
        self.grid = np.zeros((self.ny, self.nx), dtype=np.float32)

        # Thread safety
        self._lock = threading.Lock()

    # ---- public API -----------------------------------------------------

    def integrate_scan(self, scan: LidarScan, pose: PoseEstimate) -> None:
        """
        Add a LiDAR scan to the costmap, using the given pose to place it
        in the odom frame.
        """
        if len(scan) == 0:
            return

        # LiDAR frame -> robot frame (translation only; rotation is 0)
        x_robot = scan.points_xy[:, 0] + self.lidar_dx
        y_robot = scan.points_xy[:, 1] + self.lidar_dy

        # Robot frame -> odom frame
        c = math.cos(pose.th)
        s = math.sin(pose.th)
        x_odom = x_robot * c - y_robot * s + pose.x
        y_odom = x_robot * s + y_robot * c + pose.y

        # LiDAR position in odom frame (start of each ray)
        lx_odom = self.lidar_dx * c - self.lidar_dy * s + pose.x
        ly_odom = self.lidar_dx * s + self.lidar_dy * c + pose.y

        with self._lock:
            for i in range(len(x_odom)):
                r = scan.ranges_m[i]
                if r < self.lidar_min_range or r > self.lidar_max_range:
                    continue
                self._integrate_ray(lx_odom, ly_odom,
                                    x_odom[i], y_odom[i])

    def get_occupancy(self, x: float, y: float) -> float:
        """Return occupancy probability in [0, 1] at world (x, y)."""
        i, j = self.world_to_grid(x, y)
        if not self._in_bounds(i, j):
            return 0.5
        with self._lock:
            return _logodds_to_prob(float(self.grid[j, i]))

    def is_occupied(self, x: float, y: float, threshold: float = 0.6) -> bool:
        """Convenience: occupancy > threshold."""
        return self.get_occupancy(x, y) > threshold

    def snapshot(self) -> np.ndarray:
        """Return a copy of the raw log-odds grid (ny, nx)."""
        with self._lock:
            return self.grid.copy()

    def probability_grid(self) -> np.ndarray:
        """Return a copy of the grid as probabilities in [0, 1]."""
        with self._lock:
            # Vectorized log-odds -> probability
            return 1.0 - 1.0 / (1.0 + np.exp(self.grid))

    def reset(self) -> None:
        """Clear the grid back to unknown."""
        with self._lock:
            self.grid.fill(0.0)

    # ---- coordinate conversions -----------------------------------------

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        i = int(math.floor((x - self.origin_x) / self.resolution))
        j = int(math.floor((y - self.origin_y) / self.resolution))
        return i, j

    def grid_to_world(self, i: int, j: int) -> tuple[float, float]:
        x = (i + 0.5) * self.resolution + self.origin_x
        y = (j + 0.5) * self.resolution + self.origin_y
        return x, y

    def _in_bounds(self, i: int, j: int) -> bool:
        return 0 <= i < self.nx and 0 <= j < self.ny

    # ---- internal: ray casting ------------------------------------------

    def _integrate_ray(self,
                       x0: float, y0: float,
                       x1: float, y1: float) -> None:
        """
        Walk a ray from (x0, y0) to (x1, y1) in world coordinates.
        Marks all cells along the ray as free, and the final cell as occupied.
        Uses a grid-based Bresenham-like walk.
        """
        i0, j0 = self.world_to_grid(x0, y0)
        i1, j1 = self.world_to_grid(x1, y1)

        # Skip ray if the endpoint is far out of bounds
        if not self._in_bounds(i1, j1):
            return

        # Bresenham line
        di = abs(i1 - i0)
        dj = abs(j1 - j0)
        si = 1 if i1 > i0 else -1
        sj = 1 if j1 > j0 else -1
        err = di - dj

        i, j = i0, j0
        # Mark every cell along the ray as free, except the last one.
        while True:
            if i == i1 and j == j1:
                break
            if self._in_bounds(i, j):
                self._update_cell(i, j, L_FREE)
            e2 = 2 * err
            if e2 > -dj:
                err -= dj
                i += si
            if e2 < di:
                err += di
                j += sj

        # Mark the endpoint as occupied.
        self._update_cell(i1, j1, L_OCC)

    def _update_cell(self, i: int, j: int, dlog: float) -> None:
        """Add dlog to the cell's log-odds value, clamped to [L_MIN, L_MAX]."""
        new_val = self.grid[j, i] + dlog
        if new_val < L_MIN:
            new_val = L_MIN
        elif new_val > L_MAX:
            new_val = L_MAX
        self.grid[j, i] = new_val