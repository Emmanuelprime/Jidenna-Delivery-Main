"""
jidenna_runtime.py

Single-process owner of all Jidenna hardware.

Opens:
    - Nano serial (bridge) -> JidennaPose (EKF)
    - LiDAR serial -> Lidar

Wires:
    - Lidar scans -> Costmap (using current EKF pose)

Usage:
    with JidennaRuntime("/dev/ttyUSB0", "/dev/ttyUSB1") as rt:
        # rt.bridge, rt.pose, rt.lidar, rt.costmap are ready
        rt.bridge.set_velocity(0.2, 0.0)
        time.sleep(2)
        rt.bridge.set_velocity(0.0, 0.0)
        print(rt.pose.get_pose())

Any process that wants to talk to Jidenna should use exactly one
JidennaRuntime. The serial ports can only be opened once.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from jidenna.jidenna_bridge     import JidennaBridge
from jidenna.jidenna_pose       import JidennaPose
from jidenna.jidenna_lidar      import Lidar
from jidenna.perception.costmap import Costmap


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RuntimeConfig:
    # Serial ports
    nano_port:  str = "/dev/ttyUSB0"     # or "COMx" on Windows
    lidar_port: str = "/dev/ttyUSB1"

    # LiDAR mounting (robot frame, meters)
    lidar_dx: float = -0.432
    lidar_dy: float =  0.000

    # Costmap
    costmap_resolution: float = 0.05     # 5 cm per cell
    costmap_size_x:     float = 20.0
    costmap_size_y:     float = 20.0
    lidar_min_range:    float = 0.20
    lidar_max_range:    float = 8.00


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

class JidennaRuntime:

    def __init__(self, cfg: RuntimeConfig | None = None,log_path=None):
        self.cfg = cfg or RuntimeConfig()

        # Core modules
        self.bridge = JidennaBridge(self.cfg.nano_port)
        self.pose   = JidennaPose(self.bridge)
        self.lidar  = Lidar(self.cfg.lidar_port)

        # Costmap, with LiDAR mounting offset configured
        self.costmap = Costmap(
            resolution=self.cfg.costmap_resolution,
            size_x=self.cfg.costmap_size_x,
            size_y=self.cfg.costmap_size_y,
            lidar_dx=self.cfg.lidar_dx,
            lidar_dy=self.cfg.lidar_dy,
            lidar_min_range=self.cfg.lidar_min_range,
            lidar_max_range=self.cfg.lidar_max_range,
        )

        # Wire LiDAR scans into the costmap, using the current pose
        self.lidar.add_scan_callback(self._on_lidar_scan)

        # Internal state
        self._started = False
        self.logger = None

        if log_path:
            from jidenna.jidenna_logger import CsvLogger
            from pathlib import Path
            self.logger = CsvLogger(Path(log_path), comment="runtime session")
            self.bridge.add_telemetry_callback(self.logger)
            self.pose.add_pose_callback(self.logger.on_pose)

    # ---- lifecycle ------------------------------------------------------

    def start(self, timeout_s: float = 20.0) -> bool:
        """
        Start the bridge, pose, and LiDAR; block until both are connected
        (or timeout_s elapses). Returns True on success.
        """
        if self._started:
            return True

        self.bridge.start()
        self.pose.start()
        self.lidar.start()

        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self.bridge.is_connected() and self.lidar.is_connected():
                self._started = True
                return True
            time.sleep(0.1)

        # Partial success is still useful; report what's connected
        print(f"[runtime] start timeout: "
              f"nano_connected={self.bridge.is_connected()} "
              f"lidar_connected={self.lidar.is_connected()}")
        self._started = True
        return False

    def stop(self, timeout: float = 2.0) -> None:
        """Stop all threads and close all ports. Safe to call multiple times."""
        try:
            self.bridge.set_velocity(0.0, 0.0)
        except Exception:
            pass
        try:
            self.lidar.stop(timeout=timeout)
        except Exception:
            pass
        try:
            self.pose.stop()
        except Exception:
            pass
        try:
            self.bridge.stop(timeout=timeout)
        except Exception:
            pass
        self._started = False

    def is_ready(self) -> bool:
        return self._started and self.bridge.is_connected()

    # ---- context manager -------------------------------------------------

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    # ---- internals ------------------------------------------------------

    def _on_lidar_scan(self, scan) -> None:
        """Called from the LiDAR reader thread on every scan."""
        p = self.pose.get_pose()
        try:
            self.costmap.integrate_scan(scan, p)
        except Exception as e:
            # Never let a costmap bug kill the LiDAR thread
            print(f"[runtime] costmap.integrate_scan error: {e}")


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def open_runtime(nano_port: str = "/dev/ttyUSB0",
                 lidar_port: str = "/dev/ttyUSB1",
                 **kwargs) -> JidennaRuntime:
    """
    Shorthand constructor with the common case. Auto-starts.

    Example:
        rt = open_runtime(nano_port="/dev/ttyUSB0")
        # rt is already started
    """
    cfg = RuntimeConfig(nano_port=nano_port, lidar_port=lidar_port, **kwargs)
    rt = JidennaRuntime(cfg)
    rt.start()
    return rt