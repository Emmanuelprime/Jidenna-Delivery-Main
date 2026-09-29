"""
jidenna_bridge.py

Pi/Jetson-side serial bridge to the Jidenna hoverboard interface Nano.

The Nano is a deterministic hardware bridge. This module is the Pi's
software-side counterpart: it opens the serial port, sends velocity
commands at a fixed rate (keeping the Nano's watchdog fed), reads
telemetry in the background, and exposes a clean Python API.

Public API:
    JidennaBridge(port, baud=115200)
        .start()                         # begin threads
        .stop()                          # shut down cleanly
        .set_velocity(v, w)              # v m/s (fwd+), w rad/s (CCW+)
        .reset_odometry()
        .get_telemetry()                 # latest Telemetry snapshot
        .add_telemetry_callback(fn)
        .is_connected()                  # bool
        .is_watchdog_expired()           # bool, mirrors Nano's wd flag

Telemetry fields are exposed via the Telemetry dataclass.

Wire protocol (matches Nano firmware):
    Host -> Nano:  "<v>,<w>\\n"        v in [-0.8, 0.8] m/s
                                       w in [-3.0, 3.0] rad/s (CCW+)
                   "r\\n"              reset odometry

    Nano -> Host:  CSV line, 13 fields:
        x,y,th,vL,vR,wL,wR,bat,temp,fb_age,wd,or,imu_yaw
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import serial


# ---------------------------------------------------------------------------
# Constants — must match the Nano firmware
# ---------------------------------------------------------------------------

DEFAULT_BAUD       = 115200
DEFAULT_CMD_HZ     = 10           # matches firmware's TIME_SEND = 100 ms
DEFAULT_CMD_PERIOD = 1.0 / DEFAULT_CMD_HZ

# Firmware limits (parseIncoming rejects anything outside these)
V_MAX_REAL = 0.8                  # m/s
W_MAX_REAL = 3.0                  # rad/s (CCW+)

NUM_FIELDS = 13                   # x,y,th,vL,vR,wL,wR,bat,temp,fb_age,wd,or,imu_yaw

# If no telemetry arrives for this long, consider the link dead
TELEMETRY_TIMEOUT_S = 1.0

# Reconnect retry interval when the link is down
RECONNECT_INTERVAL_S = 1.0


# ---------------------------------------------------------------------------
# Telemetry dataclass
# ---------------------------------------------------------------------------

@dataclass
class Telemetry:
    """One snapshot of Nano telemetry."""
    t_host: float = 0.0           # time.time() when received on the Pi
    x: float = 0.0                # odometry position, meters
    y: float = 0.0                # odometry position, meters
    th: float = 0.0               # odometry heading, radians (CCW negative — see Nano docstring)
    vL: float = 0.0               # left wheel linear velocity, m/s
    vR: float = 0.0               # right wheel linear velocity, m/s
    wL: float = 0.0               # left wheel angular velocity, rad/s
    wR: float = 0.0               # right wheel angular velocity, rad/s
    bat: float = 0.0              # battery voltage, V
    temp: float = 0.0             # hoverboard board temperature, C
    fb_age: int = 9999            # ms since last valid hoverboard feedback
    wd: int = 0                   # watchdog flag: 1 if Nano is in timeout
    odom_reset_count: int = 0     # increments on each 'r' command
    imu_yaw: float = 0.0          # MPU6050 integrated yaw, degrees
    raw_line: str = ""            # original line for debugging

    def __str__(self) -> str:
        return (
            f"x={self.x:+.3f} y={self.y:+.3f} th={self.th:+.3f} "
            f"vL={self.vL:+.3f} vR={self.vR:+.3f} "
            f"bat={self.bat:.2f}V temp={self.temp:.1f}C "
            f"fb_age={self.fb_age:4d}ms wd={self.wd} "
            f"or={self.odom_reset_count} yaw={self.imu_yaw:+.1f}deg"
        )


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------

class JidennaBridge:
    """
    Thread-safe serial bridge to the Jidenna hoverboard Nano.

    Usage:
        bridge = JidennaBridge("COM19")     # or "/dev/ttyUSB0"
        bridge.add_telemetry_callback(lambda t: print(t))
        bridge.start()
        bridge.set_velocity(0.3, 0.0)
        time.sleep(5)
        bridge.set_velocity(0.0, 0.0)
        bridge.stop()
    """

    def __init__(self,
                 port: str,
                 baud: int = DEFAULT_BAUD,
                 cmd_hz: float = DEFAULT_CMD_HZ,
                 read_timeout_s: float = 0.05,
                 connect_settle_s: float = 6.0):
        """
        port: serial port name (e.g. "COM19" or "/dev/ttyUSB0")
        baud: must match Nano (115200)
        cmd_hz: command send rate; firmware watchdog = 1 s, so 10 Hz is safe
        read_timeout_s: pyserial read timeout (small so threads stay responsive)
        connect_settle_s: seconds to wait after opening the port before
                          sending the first command. Covers the Nano's
                          bootloader + MPU6050 gyro calibration (~4-5 s).
        """
        self._port = port
        self._baud = baud
        self._cmd_period = 1.0 / cmd_hz
        self._read_timeout_s = read_timeout_s
        self._connect_settle_s = connect_settle_s

        # Serial handle — owned by the read thread once started
        self._ser: Optional[serial.Serial] = None

        # Threads
        self._reader_thread: Optional[threading.Thread] = None
        self._writer_thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

        # Command state
        self._cmd_lock = threading.Lock()
        self._target_v = 0.0
        self._target_w = 0.0
        self._reset_pending = False

        # Telemetry state
        self._telem_lock = threading.Lock()
        self._latest_telemetry = Telemetry()
        self._last_telem_time = 0.0
        self._callbacks: list[Callable[[Telemetry], None]] = []

        # Connection state
        self._connected = False
        self._last_reconnect_attempt = 0.0

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def start(self) -> None:
        """Start the reader and writer threads."""
        if self._reader_thread is not None:
            return  # already started

        self._stop_evt.clear()

        self._reader_thread = threading.Thread(
            target=self._reader_loop, name="jidenna-reader", daemon=True)
        self._writer_thread = threading.Thread(
            target=self._writer_loop, name="jidenna-writer", daemon=True)

        self._reader_thread.start()
        self._writer_thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Stop threads and send a final zero command."""
        # Try to send a stop command before tearing down
        try:
            if self._ser is not None and self._ser.is_open:
                self._ser.write(b"0,0\n")
                self._ser.flush()
                time.sleep(0.05)
        except Exception:
            pass

        self._stop_evt.set()

        if self._reader_thread is not None:
            self._reader_thread.join(timeout=timeout)
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=timeout)

        self._reader_thread = None
        self._writer_thread = None
        self._close_serial()

    # -----------------------------------------------------------------------
    # Public command API
    # -----------------------------------------------------------------------

    def set_velocity(self, v: float, w: float) -> None:
        """
        Set target velocity. v in m/s (forward+), w in rad/s (CCW+).
        Values outside the firmware's limits are clamped (with a warning).
        Safe to call from any thread.
        """
        v = _clamp(v, -V_MAX_REAL, V_MAX_REAL)
        w = _clamp(w, -W_MAX_REAL, W_MAX_REAL)
        with self._cmd_lock:
            self._target_v = float(v)
            self._target_w = float(w)

    def reset_odometry(self) -> None:
        """Send the 'r' command to reset the Nano's odometry to (0,0,0)."""
        with self._cmd_lock:
            self._reset_pending = True

    # -----------------------------------------------------------------------
    # Public telemetry API
    # -----------------------------------------------------------------------

    def get_telemetry(self) -> Telemetry:
        """Return the latest telemetry snapshot (copy)."""
        with self._telem_lock:
            # Return a copy so callers can't mutate our state
            return Telemetry(**self._latest_telemetry.__dict__)

    def add_telemetry_callback(self, fn: Callable[[Telemetry], None]) -> None:
        """
        Register a callback called on every valid telemetry line.
        Callback runs on the reader thread — keep it fast, or hand off
        to a queue/thread of your own.
        """
        self._callbacks.append(fn)

    def is_connected(self) -> bool:
        """True if the serial port is open AND telemetry has arrived recently."""
        if not self._connected or self._ser is None or not self._ser.is_open:
            return False
        age = time.time() - self._last_telem_time
        return age < TELEMETRY_TIMEOUT_S

    def is_watchdog_expired(self) -> bool:
        """Mirrors the Nano's own wd flag from the last telemetry line."""
        with self._telem_lock:
            return bool(self._latest_telemetry.wd)

    # -----------------------------------------------------------------------
    # Reader thread
    # -----------------------------------------------------------------------

    def _reader_loop(self) -> None:
        while not self._stop_evt.is_set():
            # Ensure port is open (reconnect if needed)
            if self._ser is None or not self._ser.is_open:
                if not self._try_open():
                    self._stop_evt.wait(RECONNECT_INTERVAL_S)
                    continue

            try:
                line = self._ser.readline()
            except serial.SerialException:
                # Cable unplugged, port gone, etc.
                self._close_serial()
                continue

            if not line:
                continue

            self._handle_line(line)

    def _try_open(self) -> bool:
        now = time.time()
        if now - self._last_reconnect_attempt < RECONNECT_INTERVAL_S:
            return False
        self._last_reconnect_attempt = now

        try:
            self._ser = serial.Serial(
                self._port, self._baud,
                timeout=self._read_timeout_s,
                write_timeout=0.5,
            )
        except serial.SerialException:
            self._ser = None
            self._connected = False
            return False

        # Wait for Nano bootloader + MPU6050 gyro calibration.
        # We sleep here in the reader thread; the writer thread also
        # needs to wait. Use a shared "connected_at" timestamp.
        self._connected_at = time.time()
        time.sleep(self._connect_settle_s)

        # Flush any boot-time garbage
        try:
            self._ser.reset_input_buffer()
            self._ser.reset_output_buffer()
        except Exception:
            pass

        self._connected = True
        self._last_telem_time = time.time()
        return True

    def _close_serial(self) -> None:
        self._connected = False
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None

    def _handle_line(self, raw: bytes) -> None:
        try:
            text = raw.decode("ascii", errors="ignore").strip()
        except Exception:
            return

        if not text or text.count(",") != NUM_FIELDS - 1:
            return  # header or malformed

        parts = text.split(",")
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            return

        # Field order must match firmware:
        # x,y,th,vL,vR,wL,wR,bat,temp,fb_age,wd,or,imu_yaw
        (x, y, th, vL, vR, wL, wR, bat, temp,
         fb_age, wd, orr, imu_yaw) = vals

        t = Telemetry(
            t_host=time.time(),
            x=x, y=y, th=th,
            vL=vL, vR=vR, wL=wL, wR=wR,
            bat=bat, temp=temp,
            fb_age=int(fb_age), wd=int(wd), odom_reset_count=int(orr),
            imu_yaw=imu_yaw,
            raw_line=text,
        )

        with self._telem_lock:
            self._latest_telemetry = t
            self._last_telem_time = t.t_host

        for cb in self._callbacks:
            try:
                cb(t)
            except Exception as e:
                # Never let a bad callback kill the reader thread
                print(f"[jidenna] telemetry callback error: {e}")

    # -----------------------------------------------------------------------
    # Writer thread
    # -----------------------------------------------------------------------

    def _writer_loop(self) -> None:
        # Wait until the reader thread has established a connection
        while not self._stop_evt.is_set():
            if self.is_connected():
                break
            self._stop_evt.wait(0.1)

        next_send = time.time()

        while not self._stop_evt.is_set():
            now = time.time()
            if now < next_send:
                self._stop_evt.wait(min(0.01, next_send - now))
                continue
            next_send += self._cmd_period

            # Read current command
            with self._cmd_lock:
                v = self._target_v
                w = self._target_w
                do_reset = self._reset_pending
                self._reset_pending = False

            # Reset odometry if requested
            if do_reset:
                try:
                    self._ser.write(b"r\n")
                except Exception:
                    self._close_serial()
                    continue

            # Send velocity
            line = f"{v:.4f},{w:.4f}\n".encode("ascii")
            try:
                self._ser.write(line)
            except Exception:
                self._close_serial()
                continue

        # Final safe stop
        try:
            if self._ser is not None and self._ser.is_open:
                self._ser.write(b"0,0\n")
                self._ser.flush()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x