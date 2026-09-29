"""
controllers/waypoint.py

Simple proportional waypoint follower with heading gate.

Behavior:
    1. If heading error is large -> rotate in place until aligned.
    2. Else -> drive toward the goal, correcting heading with a
       proportional term.
    3. When within goal_tolerance -> stop and report done.

Conventions:
    - Works in the EKF/ROS frame: theta CCW+, w CCW+.
    - The ControllerManager is responsible for translating w to the
      hardware frame (Nano expects CW+), so this controller stays in
      standard robotics convention.
"""

from __future__ import annotations

import math
from typing import Optional

from jidenna.jidenna_pose import PoseEstimate
from jidenna.controllers.base import ControlOutput, clamp, wrap_angle


class WaypointController:

    ROTATE = "ROTATE"
    DRIVE  = "DRIVE"
    DONE   = "DONE"

    def __init__(self,
                 v_max: float = 0.4,
                 w_max: float = 1.0,
                 k_v: float = 1.0,
                 k_w: float = 2.0,
                 goal_tolerance: float = 0.10,
                 heading_gate: float = 0.20,
                 heading_gate_wide: float = 0.60):
        self.v_max = v_max
        self.w_max = w_max
        self.k_v = k_v
        self.k_w = k_w
        self.goal_tolerance = goal_tolerance
        self.heading_gate = heading_gate
        self.heading_gate_wide = heading_gate_wide

        self._x_goal: Optional[float] = None
        self._y_goal: Optional[float] = None
        self._state = self.DONE

    # ---- public API -----------------------------------------------------

    def reset(self) -> None:
        self._x_goal = None
        self._y_goal = None
        self._state = self.DONE

    def set_target(self, x: float, y: float) -> None:
        self._x_goal = float(x)
        self._y_goal = float(y)
        self._state = self.ROTATE

    def has_target(self) -> bool:
        return self._x_goal is not None

    def get_target(self) -> Optional[tuple]:
        if self._x_goal is None:
            return None
        return (self._x_goal, self._y_goal)

    def is_done(self) -> bool:
        return self._state == self.DONE

    # ---- control law ----------------------------------------------------

    def compute(self, pose: PoseEstimate) -> ControlOutput:
        if self._x_goal is None:
            return ControlOutput(v=0.0, w=0.0, done=False,
                                 info={"state": "NO_TARGET"})

        dx = self._x_goal - pose.x
        dy = self._y_goal - pose.y
        distance = math.hypot(dx, dy)

        # Goal reached?
        if distance < self.goal_tolerance and self._state != self.ROTATE:
            self._state = self.DONE
            return ControlOutput(v=0.0, w=0.0, done=True,
                                 info={"state": self.DONE,
                                       "distance": distance})

        heading_to_goal = math.atan2(dy, dx)
        heading_error = wrap_angle(heading_to_goal - pose.th)

        # State transitions
        if self._state == self.ROTATE:
            if abs(heading_error) < self.heading_gate:
                self._state = self.DRIVE
        elif self._state == self.DRIVE:
            # Only re-enter ROTATE when we're far enough away that the
            # target direction is meaningful. Prevents spinning at the
            # goal where atan2(dy, dx) is dominated by noise.
            if (distance > 2.0 * self.goal_tolerance and
                abs(heading_error) > self.heading_gate_wide):
                self._state = self.ROTATE

        # Control law
        if self._state == self.ROTATE:
            v = 0.0
            w = clamp(self.k_w * heading_error, -self.w_max, self.w_max)
        elif self._state == self.DRIVE:
            v = clamp(self.k_v * distance, -self.v_max, self.v_max)
            # Fade w down near the goal so tiny geometry errors don't
            # cause last-second spins.
            w_scale = min(1.0, distance / (4.0 * self.goal_tolerance))
            w = clamp(self.k_w * heading_error * w_scale,
                      -self.w_max, self.w_max)
        else:
            v = 0.0
            w = 0.0

        return ControlOutput(
            v=v, w=w, done=(self._state == self.DONE),
            info={
                "state": self._state,
                "distance": distance,
                "heading_error": heading_error,
            },
        )