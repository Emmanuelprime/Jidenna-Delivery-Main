"""
jidenna_controller.py

Runs a controller at a fixed rate against a live JidennaPose,
sending (v, w) to the bridge.

Frame translation:
    Controllers work in the EKF/ROS frame (theta CCW+, w CCW+).
    The Nano's set_velocity expects CW+ w (positive = right turn).
    The manager flips w here, so every controller can use the
    standard robotics convention.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from jidenna.jidenna_bridge import JidennaBridge
from jidenna.jidenna_pose   import JidennaPose
from jidenna.controllers.base import ControlOutput, Controller


class ControllerManager:

    def __init__(self,
                 bridge: JidennaBridge,
                 pose: JidennaPose,
                 controller: Controller,
                 rate_hz: float = 20.0):
        self._bridge = bridge
        self._pose = pose
        self._controller = controller
        self._period = 1.0 / rate_hz

        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._done_evt = threading.Event()
        self._last_output: ControlOutput = ControlOutput()
        self._last_output_lock = threading.Lock()

    # ---- lifecycle ------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_evt.clear()
        self._done_evt.clear()
        # NOTE: do NOT call controller.reset() here. The caller sets
        # the target before start(), and reset() would wipe it.
        self._thread = threading.Thread(
            target=self._loop, name="jidenna-ctrl", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        try:
            self._bridge.set_velocity(0.0, 0.0)
        except Exception:
            pass

    def wait_until_done(self, timeout: Optional[float] = None) -> bool:
        return self._done_evt.wait(timeout=timeout)

    def get_last_output(self) -> ControlOutput:
        with self._last_output_lock:
            return self._last_output

    # ---- main loop ------------------------------------------------------

    def _loop(self) -> None:
        next_t = time.time()
        while not self._stop_evt.is_set():
            now = time.time()
            if now < next_t:
                self._stop_evt.wait(min(0.005, next_t - now))
                continue
            next_t += self._period

            pose = self._pose.get_pose()
            output = self._controller.compute(pose)

            # Controllers work in EKF/ROS frame (CCW+ w).
            # The Nano's set_velocity expects CW+ w. Flip here.
            w_hw = -output.w

            try:
                self._bridge.set_velocity(output.v, w_hw)
            except Exception as e:
                print(f"[ControllerManager] set_velocity error: {e}")

            with self._last_output_lock:
                self._last_output = output

            if output.done:
                self._done_evt.set()
            else:
                self._done_evt.clear()

        try:
            self._bridge.set_velocity(0.0, 0.0)
        except Exception:
            pass