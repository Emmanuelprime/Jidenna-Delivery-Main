"""
jidenna_ekf.py

Extended Kalman Filter for Jidenna.
Fuses wheel odometry (vL, vR) and IMU yaw.

State: x = [x, y, theta, v, omega]
  x, y    : position (m)
  theta   : heading (rad, CCW+)
  v       : linear velocity (m/s)
  omega   : angular velocity (rad/s, CCW+)

Measurements:
  z_wheel = [v_wheel, omega_wheel]  from (vL, vR)
  z_imu   = [theta_imu]             from imu_yaw

Note: the Nano reports theta and IMU yaw with the opposite sign
convention (CW+). We flip signs on ingestion so the EKF works in the
standard CCW+ frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WHEEL_SEPARATION_M = 0.521
WHEEL_RADIUS_M     = 0.165 / 2.0


@dataclass
class EKFConfig:
    # Process noise (how much we distrust the constant-velocity model)
    # Units: [m², m², rad², (m/s)², (rad/s)²] per second
    q_xy:     float = 0.01     # position noise
    q_theta:  float = 0.005    # heading noise
    q_v:      float = 0.05     # linear velocity noise
    q_omega:  float = 0.10     # angular velocity noise

    # Measurement noise
    r_wheel_v:     float = 0.05 ** 2   # (m/s)²
    r_wheel_omega: float = 0.10 ** 2   # (rad/s)²
    r_imu_yaw:     float = 0.02 ** 2   # (rad)²  — IMU is fairly precise short-term

    # Initial covariance (big = don't trust the initial state)
    p0_xy:    float = 1.0
    p0_theta: float = 1.0
    p0_v:     float = 1.0
    p0_omega: float = 1.0


# ---------------------------------------------------------------------------
# EKF
# ---------------------------------------------------------------------------

@dataclass
class EKF:
    cfg: EKFConfig = field(default_factory=EKFConfig)

    # State and covariance
    x: np.ndarray = field(default_factory=lambda: np.zeros(5))
    P: np.ndarray = field(default_factory=lambda: np.eye(5))

    def __post_init__(self):
        self.x = np.zeros(5)
        self.P = np.diag([
            self.cfg.p0_xy,
            self.cfg.p0_xy,
            self.cfg.p0_theta,
            self.cfg.p0_v,
            self.cfg.p0_omega,
        ])

    # ---- prediction step ------------------------------------------------

    def predict(self, dt: float) -> None:
        """Advance the state by dt using the constant-velocity motion model."""
        if dt <= 0.0:
            return

        x, y, th, v, w = self.x

        # Nonlinear motion model (unicycle)
        # x_{k+1} = x + v*cos(th)*dt
        # y_{k+1} = y + v*sin(th)*dt
        # th_{k+1}= th + w*dt
        # v_{k+1} = v        (constant velocity)
        # w_{k+1} = w
        self.x = np.array([
            x + v * math.cos(th) * dt,
            y + v * math.sin(th) * dt,
            _wrap_angle(th + w * dt),
            v,
            w,
        ])

        # Jacobian F = ∂f/∂x
        F = np.eye(5)
        F[0, 3] = math.cos(th) * dt
        F[0, 2] = -v * math.sin(th) * dt
        F[1, 3] = math.sin(th) * dt
        F[1, 2] =  v * math.cos(th) * dt
        F[2, 4] = dt

        # Process noise covariance
        Q = np.diag([
            self.cfg.q_xy    * dt,
            self.cfg.q_xy    * dt,
            self.cfg.q_theta * dt,
            self.cfg.q_v     * dt,
            self.cfg.q_omega * dt,
        ])

        # Covariance propagation
        self.P = F @ self.P @ F.T + Q

    # ---- wheel measurement update --------------------------------------

    def update_wheels(self, vL: float, vR: float) -> None:
        """
        vL, vR: wheel linear velocities in m/s (from Nano telemetry,
        same sign convention as Nano: both positive = forward).
        """
        v_wheel     = 0.5 * (vL + vR)
        omega_wheel = -(vR - vL) / WHEEL_SEPARATION_M   # CCW+ in EKF frame

        # Measurement vector: [v, omega]
        z = np.array([v_wheel, omega_wheel])

        # Measurement model: we observe (v, omega) directly
        H = np.zeros((2, 5))
        H[0, 3] = 1.0
        H[1, 4] = 1.0

        R = np.diag([self.cfg.r_wheel_v, self.cfg.r_wheel_omega])

        self._kalman_update(z, H, R)

    # ---- IMU yaw measurement update ------------------------------------

    def update_imu_yaw(self, imu_yaw_deg: float) -> None:
        """
        imu_yaw_deg: MPU6050 integrated yaw in degrees, as reported by Nano.
        Nano convention: th decreases for w>0 (CW+). We flip to CCW+ here.
        """
        theta_meas = math.radians(-imu_yaw_deg)   # CCW+ in EKF frame

        z = np.array([theta_meas])

        # Measurement model: we observe theta directly
        H = np.zeros((1, 5))
        H[0, 2] = 1.0

        R = np.array([[self.cfg.r_imu_yaw]])

        # Handle angular wrap: since theta is bounded, we use the
        # innovation wrap below in _kalman_update for row index 2.
        self._kalman_update(z, H, R, angle_indices=[0])

    # ---- Generic Kalman update ------------------------------------------

    def _kalman_update(self, z, H, R, angle_indices=None):
        """
        z: measurement (m,)
        H: measurement Jacobian (m, n)
        R: measurement noise (m, m)
        angle_indices: list of measurement-row indices that represent angles
                       and need innovation wrapping.
        """
        z = np.atleast_1d(z)
        y = z - H @ self.x

        # Wrap angular innovation
        if angle_indices:
            for i in angle_indices:
                y[i] = _wrap_angle(y[i])

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.x[2] = _wrap_angle(self.x[2])   # keep theta wrapped

        I = np.eye(self.P.shape[0])
        self.P = (I - K @ H) @ self.P

    # ---- Convenience ----------------------------------------------------

    def state(self):
        return {
            "x": float(self.x[0]),
            "y": float(self.x[1]),
            "th": float(self.x[2]),
            "v": float(self.x[3]),
            "w": float(self.x[4]),
        }

    def covariance(self):
        return self.P.copy()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wrap_angle(a: float) -> float:
    while a >  math.pi: a -= 2.0 * math.pi
    while a < -math.pi: a += 2.0 * math.pi
    return a


def _wrap_angle_np(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi