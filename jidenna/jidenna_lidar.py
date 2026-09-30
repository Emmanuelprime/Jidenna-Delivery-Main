"""
jidenna_lidar.py

LiDAR driver for Jidenna. Wraps the YDLIDAR SDK, runs the scan loop in a
background thread, and delivers each scan to registered callbacks as a
numpy array of shape (N, 2): [angle_rad, range_m].

Public API:
    lidar = Lidar(port="/dev/ttyUSB1")
    lidar.add_scan_callback(fn)
    lidar.start()
    ...
    lidar.stop()

Callback signature:
    fn(scan: LidarScan)

Where LidarScan is:
    angles_rad : np.ndarray  shape (N,)   angle in radians (CCW+)
    ranges_m   : np.ndarray  shape (N,)   range in meters
    t_host     : float                    time.time() when received
    points_xy  : np.ndarray  shape (N, 2) [x_lidar, y_lidar] in meters

Notes:
    - The YDLIDAR SDK reports angles as floats; we convert to radians.
    - Points with range <= 0 are dropped (invalid returns).
    - The LiDAR is expected to have 0° pointing along robot +x
      (confirmed empirically). No mounting offset is applied here —
      that belongs in the consumer.
    - This class only reads LiDAR data. It does NOT know about the robot,
      the pose, or the bridge.

IMPORTANT (GIL note):
    The YDLIDAR SDK's `doProcessSimple()` call does not release the Python
    GIL during its execution. When the LiDAR is between scans, the SDK
    returns False immediately, and a naive `if not ok: continue` becomes
    a tight busy loop that starves every other thread in the process.

    We insert `self._stop_evt.wait(0.005)` in the not-ok branch to yield
    the GIL for 5 ms. This is invisible to the scan rate (100 ms/scan)
    but lets the bridge reader thread, the controller thread, and the
    pose estimator all keep running.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import ydlidar


# ---------------------------------------------------------------------------
# Scan dataclass
# ---------------------------------------------------------------------------

@dataclass
class LidarScan:
    t_host: float                    # time.time() on host when received
    angles_rad: np.ndarray           # shape (N,), CCW+
    ranges_m: np.ndarray             # shape (N,), meters
    points_xy: np.ndarray            # shape (N, 2), LiDAR-frame (x,y) meters

    def __len__(self) -> int:
        return len(self.ranges_m)

    def __repr__(self) -> str:
        return (f"LidarScan(n={len(self)}, "
                f"r=[{self.ranges_m.min():.2f}..{self.ranges_m.max():.2f}]m)")


# ---------------------------------------------------------------------------
# Lidar class
# ---------------------------------------------------------------------------

class Lidar:
    """
    Background-threaded YDLIDAR driver.

    Usage:
        lidar = Lidar("/dev/ttyUSB1")
        lidar.add_scan_callback(lambda s: print(len(s)))
        lidar.start()
        ...
        lidar.stop()
    """

    def __init__(self,
                 port: str,
                 baud: int = 115200,
                 scan_hz: float = 10.0,
                 sample_rate_khz: int = 3,
                 min_range_m: float = 0.08,
                 max_range_m: float = 8.0,
                 min_angle_deg: float = -180.0,
                 max_angle_deg: float =  180.0,
                 reconnect_interval_s: float = 1.0,
                 idle_sleep_s: float = 0.005):
        self._port = port
        self._baud = baud
        self._scan_hz = scan_hz
        self._sample_rate_khz = sample_rate_khz
        self._min_range = min_range_m
        self._max_range = max_range_m
        self._min_angle = min_angle_deg
        self._max_angle = max_angle_deg
        self._reconnect_interval = reconnect_interval_s
        self._idle_sleep = idle_sleep_s

        # SDK objects (created/recreated on connect)
        self._laser: Optional[ydlidar.CYdLidar] = None
        self._scan_buffer = None

        # Threading
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

        # Callbacks
        self._callbacks: list[Callable[[LidarScan], None]] = []

        # State
        self._connected = False
        self._last_scan_time = 0.0
        self._scan_count = 0

    # ---- public API -----------------------------------------------------

    def add_scan_callback(self, fn: Callable[[LidarScan], None]) -> None:
        """Register a callback fired on every valid scan. Runs on the reader thread."""
        self._callbacks.append(fn)

    def start(self) -> None:
        if self._thread is not None:
            return
        ydlidar.os_init()
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._loop, name="jidenna-lidar", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        self._disconnect()

    def is_connected(self) -> bool:
        return self._connected

    def last_scan_age_s(self) -> float:
        if self._last_scan_time == 0.0:
            return float("inf")
        return time.time() - self._last_scan_time

    def scan_count(self) -> int:
        return self._scan_count

    # ---- connection management ------------------------------------------

    def _connect(self) -> bool:
        laser = ydlidar.CYdLidar()
        laser.setlidaropt(ydlidar.LidarPropSerialPort, self._port)
        laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, self._baud)
        laser.setlidaropt(ydlidar.LidarPropLidarType, ydlidar.TYPE_TRIANGLE)
        laser.setlidaropt(ydlidar.LidarPropDeviceType, ydlidar.YDLIDAR_TYPE_SERIAL)
        laser.setlidaropt(ydlidar.LidarPropSampleRate, self._sample_rate_khz)
        laser.setlidaropt(ydlidar.LidarPropScanFrequency, self._scan_hz)
        laser.setlidaropt(ydlidar.LidarPropSingleChannel, True)
        laser.setlidaropt(ydlidar.LidarPropMaxAngle, self._max_angle)
        laser.setlidaropt(ydlidar.LidarPropMinAngle, self._min_angle)
        laser.setlidaropt(ydlidar.LidarPropMaxRange, self._max_range)
        laser.setlidaropt(ydlidar.LidarPropMinRange, self._min_range)

        if not laser.initialize():
            print(f"[lidar] initialize() failed on {self._port}")
            return False

        time.sleep(0.5)

        if not laser.turnOn():
            print(f"[lidar] turnOn() failed on {self._port}")
            try:
                laser.disconnecting()
            except Exception:
                pass
            return False

        self._laser = laser
        self._scan_buffer = ydlidar.LaserScan()
        self._connected = True
        print(f"[lidar] connected on {self._port}")
        return True

    def _disconnect(self) -> None:
        self._connected = False
        if self._laser is not None:
            try:
                self._laser.turnOff()
            except Exception:
                pass
            try:
                self._laser.disconnecting()
            except Exception:
                pass
        self._laser = None
        self._scan_buffer = None

    # ---- reader loop ----------------------------------------------------

    def _loop(self) -> None:
        while not self._stop_evt.is_set():
            # Ensure we're connected
            if self._laser is None:
                if not self._connect():
                    self._stop_evt.wait(self._reconnect_interval)
                    continue

            try:
                ok = self._laser.doProcessSimple(self._scan_buffer)
            except Exception as e:
                print(f"[lidar] read error: {e}")
                self._disconnect()
                self._stop_evt.wait(self._reconnect_interval)
                continue

            if not ok:
                # Critical: yield the GIL. Without this, the busy loop
                # starves every other thread in the process.
                self._stop_evt.wait(self._idle_sleep)
                continue

            if not ydlidar.os_isOk():
                print("[lidar] os_isOk() returned False, reconnecting")
                self._disconnect()
                self._stop_evt.wait(self._reconnect_interval)
                continue

            scan = self._build_scan(self._scan_buffer)
            if scan is None:
                # No valid points this cycle; also yield.
                self._stop_evt.wait(self._idle_sleep)
                continue

            self._last_scan_time = scan.t_host
            self._scan_count += 1

            for cb in self._callbacks:
                try:
                    cb(scan)
                except Exception as e:
                    print(f"[lidar] callback error: {e}")

            # Small yield at the end of every successful iteration too,
            # so callbacks don't hog the GIL either.
            self._stop_evt.wait(0.001)

        self._disconnect()

    def _build_scan(self, sdk_scan) -> Optional[LidarScan]:
        pts = sdk_scan.points
        if not pts:
            return None

        angles = np.empty(len(pts), dtype=np.float64)
        ranges = np.empty(len(pts), dtype=np.float64)

        n_valid = 0
        for p in pts:
            r = p.range
            if r <= 0.0 or r < self._min_range or r > self._max_range:
                continue
            angles[n_valid] = p.angle
            ranges[n_valid] = r
            n_valid += 1

        if n_valid == 0:
            return None

        angles = angles[:n_valid]
        ranges = ranges[:n_valid]

        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)
        points_xy = np.column_stack((xs, ys))

        return LidarScan(
            t_host=time.time(),
            angles_rad=angles,
            ranges_m=ranges,
            points_xy=points_xy,
        )