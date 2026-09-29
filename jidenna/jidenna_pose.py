"""
jidenna_pose.py

Live EKF wrapper for Jidenna. Attaches to a JidennaBridge and maintains
a real-time pose estimate by fusing wheel odometry and IMU yaw.

Public API:
    pose = JidennaPose(bridge)          # attaches to bridge telemetry
    pose.get_pose()                     # -> PoseEstimate (latest)
    pose.add_pose_callback(fn)          # called on every EKF update
    pose.reset(x=0, y=0, th=0)          # reset the filter state

Conventions:
    theta  : radians, CCW+ (standard math/ROS convention)
    omega  : radians/sec, CCW+
    v      : m/s, forward+
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .jidenna_ekf import EKF, EKFConfig
from .jidenna_bridge import JidennaBridge, Telemetry


# ---------------------------------------------------------------------------
# PoseEstimate
# ---------------------------------------------------------------------------

@dataclass
class PoseEstimate:
    t_host: float = 0.0
    x: float = 0.0
    y: float = 0.0
    th: float = 0.0          # rad, CCW+
    v: float = 0.0           # m/s
    w: float = 0.0           # rad/s, CCW+
    P_xx: float = 0.0        # covariance of x
    P_yy: float = 0.0
    P_thth: float = 0.0
    covariance: Optional[np.ndarray] = None

    def __str__(self) -> str:
        return (f"x={self.x:+.3f} y={self.y:+.3f} th={self.th:+.3f} "
                f"v={self.v:+.3f} w={self.w:+.3f} "
                f"σ(x,y,θ)=({math.sqrt(self.P_xx):.3f},"
                f"{math.sqrt(self.P_yy):.3f},"
                f"{math.degrees(math.sqrt(self.P_thth)):.2f}°)")


# ---------------------------------------------------------------------------
# JidennaPose
# ---------------------------------------------------------------------------

class JidennaPose:
    """
    Live EKF pose estimator. Fuses wheel odometry + IMU yaw from a
    JidennaBridge into a single best-estimate pose.

    Usage:
        bridge = JidennaBridge("COM19")
        pose   = JidennaPose(bridge)
        bridge.start()

        # ... drive ...

        p = pose.get_pose()
        print(p)

        bridge.stop()
    """

    def __init__(self, bridge: JidennaBridge, cfg: EKFConfig | None = None):
        self._bridge = bridge
        self._ekf = EKF(cfg or EKFConfig())

        self._lock = threading.Lock()
        self._latest = PoseEstimate()
        self._last_t = None
        self._imu_offset = None      # set on first sample
        self._callbacks: list[Callable[[PoseEstimate], None]] = []
        self._started = False

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def start(self) -> None:
        """Begin listening to bridge telemetry."""
        if self._started:
            return
        self._bridge.add_telemetry_callback(self._on_telemetry)
        self._started = True

    def stop(self) -> None:
        """Stop listening (does not stop the bridge)."""
        # The bridge doesn't expose remove_callback; we use a flag instead.
        self._started = False

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def get_pose(self) -> PoseEstimate:
        with self._lock:
            p = self._latest
            p.covariance = self._ekf.covariance()   # copy
            return p

    def add_pose_callback(self, fn: Callable[[PoseEstimate], None]) -> None:
        """Called on every EKF update. Runs on the bridge reader thread."""
        self._callbacks.append(fn)

    def reset(self, x: float = 0.0, y: float = 0.0, th: float = 0.0) -> None:
        """Reset the filter to a known pose and re-anchor the IMU offset."""
        with self._lock:
            self._ekf = EKF(self._ekf.cfg)
            self._ekf.x[0] = x
            self._ekf.x[1] = y
            self._ekf.x[2] = th
            self._imu_offset = None
            self._last_t = None
            self._latest = PoseEstimate(
                x=x, y=y, th=th,
                P_xx=self._ekf.P[0, 0],
                P_yy=self._ekf.P[1, 1],
                P_thth=self._ekf.P[2, 2],
            )

    # -----------------------------------------------------------------------
    # Telemetry handler (runs on the bridge reader thread)
    # -----------------------------------------------------------------------

    def _on_telemetry(self, t: Telemetry) -> None:
        if not self._started:
            return

        now = t.t_host

        # First sample: initialize the filter from raw odom + anchor IMU yaw
        if self._last_t is None:
            # Bootstrap the EKF at the raw odom pose (Nano's th is CW+; flip it)
            with self._lock:
                self._ekf.x[0] = t.x
                self._ekf.x[1] = t.y
                self._ekf.x[2] = -t.th
                # Anchor IMU: we compute the constant offset so imu_yaw
                # matches odom's heading at t=0.
                imu_rad = math.radians(-t.imu_yaw)
                self._imu_offset = self._ekf.x[2] - imu_rad
                self._last_t = now
            return

        # Compute dt from host clock (robust to bridge jitter)
        dt = now - self._last_t
        self._last_t = now

        # Guard against unreasonable dt (bridge stall, reconnect)
        if dt <= 0.0 or dt > 0.5:
            return

        with self._lock:
            # 1) Predict using last known velocities
            self._ekf.predict(dt)

            # 2) Wheel update — Nano's vL, vR are already physical m/s,
            #    forward positive. EKF expects them in that same convention.
            self._ekf.update_wheels(t.vL, t.vR)

            # 3) IMU yaw update — apply the anchor offset, flip sign
            if self._imu_offset is None:
                imu_rad = math.radians(-t.imu_yaw)
                self._imu_offset = self._ekf.x[2] - imu_rad
            imu_yaw_corrected_deg = math.degrees(
                math.radians(-t.imu_yaw) + self._imu_offset
            )
            self._ekf.update_imu_yaw(imu_yaw_corrected_deg)

            # Snapshot
            self._latest = PoseEstimate(
                t_host=now,
                x=float(self._ekf.x[0]),
                y=float(self._ekf.x[1]),
                th=float(self._ekf.x[2]),
                v=float(self._ekf.x[3]),
                w=float(self._ekf.x[4]),
                P_xx=float(self._ekf.P[0, 0]),
                P_yy=float(self._ekf.P[1, 1]),
                P_thth=float(self._ekf.P[2, 2]),
                covariance=self._ekf.covariance(),
            )
            snapshot = self._latest

        # Callbacks fire outside the lock
        for cb in self._callbacks:
            try:
                cb(snapshot)
            except Exception as e:
                print(f"[jidenna_pose] callback error: {e}")