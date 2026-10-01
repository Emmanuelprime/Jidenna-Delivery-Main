"""
jidenna/joy_receiver.py

UDP joystick receiver for Jidenna.

Listens for JSON packets emitted by the ESP32 joystick firmware
(joy_firmware.ino) and exposes them as JoyState snapshots.

Packet format (one UDP datagram per packet):
    {"t": <millis>, "lx": <f>, "ly": <f>, "rx": <f>, "ry": <f>}

All axes are in [-1.0, 1.0] and already deadzoned + normalized by the
firmware. This module does not re-apply any stick shaping; it only
clamps to the legal range as a defensive measure.

Public API:
    rx = JoyReceiver(udp_port=4210)
    rx.add_callback(fn)         # fn(JoyState), runs on the receiver thread
    rx.start()
    ...
    st = rx.get_state()         # latest snapshot (thread-safe)
    age = rx.last_rx_age_s()    # seconds since last packet, inf if never
    rx.stop()

Stick -> (v, w) helper:
    v, w = sticks_to_vw(st, scheme="arcade", v_max=0.4, w_max=1.0)

Conventions for (v, w):
    v : m/s, forward positive
    w : rad/s, CCW positive (EKF / ROS frame)

Consumers are responsible for flipping w into the Nano's CW+ convention
(JidennaBridge.set_velocity expects CW+). The ControllerManager already
does this flip, so a Controller wrapper around JoyReceiver does not
need to.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

@dataclass
class JoyState:
    """One snapshot of joystick axes."""
    t_joy_ms: float = 0.0     # firmware millis() timestamp
    t_host: float = 0.0       # time.time() on receipt
    lx: float = 0.0           # left stick X,  [-1, 1]
    ly: float = 0.0           # left stick Y,  [-1, 1]
    rx: float = 0.0           # right stick X, [-1, 1]
    ry: float = 0.0           # right stick Y, [-1, 1]

    def __str__(self) -> str:
        return (f"lx={self.lx:+.2f} ly={self.ly:+.2f} "
                f"rx={self.rx:+.2f} ry={self.ry:+.2f}")


# ---------------------------------------------------------------------------
# Receiver
# ---------------------------------------------------------------------------

class JoyReceiver:
    """
    Background-threaded UDP receiver for joystick packets.

    Usage:
        rx = JoyReceiver(udp_port=4210)
        rx.add_callback(lambda st: print(st))
        rx.start()
        ...
        st = rx.get_state()
        print(rx.last_rx_age_s())
        rx.stop()

    Thread-safety:
        - get_state(), last_rx_age_s(), packet_count(), bad_packet_count()
          are safe to call from any thread.
        - Callbacks run on the receiver thread; keep them fast or hand
          work off to a queue.
    """

    # Packets are tiny (~80 bytes); this is generous.
    RECV_BUFFER = 1024

    def __init__(self,
                 udp_port: int = 4210,
                 bind_addr: str = "0.0.0.0",
                 recv_timeout_s: float = 0.1,
                 max_axis: float = 1.0):
        """
        udp_port       : UDP port to listen on (must match firmware).
        bind_addr      : interface to bind. "0.0.0.0" accepts broadcasts.
        recv_timeout_s : how long recvfrom() blocks before the loop checks
                         the stop flag. Lower = snappier stop, higher =
                         less CPU. 100 ms is fine.
        max_axis       : clamp incoming axes to [-max_axis, max_axis].
                         The firmware already clamps to [-1, 1], but be safe.
        """
        self._port = int(udp_port)
        self._bind_addr = bind_addr
        self._recv_timeout = float(recv_timeout_s)
        self._max_axis = float(max_axis)

        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

        self._lock = threading.Lock()
        self._state = JoyState()
        self._last_rx_time = 0.0
        self._packet_count = 0
        self._bad_packet_count = 0

        self._callbacks: list[Callable[[JoyState], None]] = []

        # Set to True if bind() failed for any reason; the thread will exit.
        self._bind_failed = False
        self._bind_error: Optional[str] = None

    # ---- lifecycle ------------------------------------------------------

    def start(self) -> None:
        """Start the background receiver thread. Idempotent."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._bind_failed = False
        self._bind_error = None
        self._thread = threading.Thread(
            target=self._loop, name="joy-receiver", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        """Stop the receiver thread and close the socket."""
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        self._close_socket()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def bind_failed(self) -> bool:
        return self._bind_failed

    def bind_error(self) -> Optional[str]:
        return self._bind_error

    # ---- callbacks ------------------------------------------------------

    def add_callback(self, fn: Callable[[JoyState], None]) -> None:
        """
        Register a callback invoked on every valid packet.
        Runs on the receiver thread — do not block.
        """
        self._callbacks.append(fn)

    def remove_callback(self, fn: Callable[[JoyState], None]) -> None:
        try:
            self._callbacks.remove(fn)
        except ValueError:
            pass

    # ---- polling --------------------------------------------------------

    def get_state(self) -> JoyState:
        """Latest snapshot (copy)."""
        with self._lock:
            return JoyState(**self._state.__dict__)

    def last_rx_age_s(self) -> float:
        """Seconds since last valid packet, or inf if none yet."""
        with self._lock:
            if self._last_rx_time == 0.0:
                return float("inf")
            return time.time() - self._last_rx_time

    def packet_count(self) -> int:
        with self._lock:
            return self._packet_count

    def bad_packet_count(self) -> int:
        with self._lock:
            return self._bad_packet_count

    # ---- internals ------------------------------------------------------

    def _loop(self) -> None:
        # Set up the socket. If bind fails, record the error and bail out;
        # start() callers can check bind_failed()/bind_error().
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            self._sock.bind((self._bind_addr, self._port))
            self._sock.settimeout(self._recv_timeout)
        except OSError as e:
            self._bind_failed = True
            self._bind_error = f"{self._bind_addr}:{self._port}: {e}"
            self._close_socket()
            return

        while not self._stop_evt.is_set():
            try:
                data, _addr = self._sock.recvfrom(self.RECV_BUFFER)
            except socket.timeout:
                continue
            except OSError:
                # Socket was closed from another thread (stop()).
                break

            self._handle_packet(data)

        self._close_socket()

    def _handle_packet(self, data: bytes) -> None:
        # Decode
        try:
            text = data.decode("ascii", errors="ignore").strip()
        except Exception:
            self._bump_bad()
            return

        # Cheap structural check before handing off to json.loads
        if not text or not text.startswith("{") or not text.endswith("}"):
            self._bump_bad()
            return

        try:
            obj = json.loads(text)
            lx = float(obj["lx"])
            ly = float(obj["ly"])
            rx = float(obj["rx"])
            ry = float(obj["ry"])
            t_joy = float(obj.get("t", 0.0))
        except (ValueError, KeyError, TypeError):
            self._bump_bad()
            return

        # Reject NaN/inf without polluting the state
        if not all(map(_is_finite, (lx, ly, rx, ry))):
            self._bump_bad()
            return

        m = self._max_axis
        st = JoyState(
            t_joy_ms=t_joy,
            t_host=time.time(),
            lx=_clamp(lx, -m, m),
            ly=_clamp(ly, -m, m),
            rx=_clamp(rx, -m, m),
            ry=_clamp(ry, -m, m),
        )

        with self._lock:
            self._state = st
            self._last_rx_time = st.t_host
            self._packet_count += 1

        # Snapshot the callback list so callbacks added/removed mid-flight
        # don't corrupt iteration.
        for cb in list(self._callbacks):
            try:
                cb(st)
            except Exception as e:
                print(f"[joy] callback error: {e}")

    def _bump_bad(self) -> None:
        with self._lock:
            self._bad_packet_count += 1

    def _close_socket(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None


# ---------------------------------------------------------------------------
# Stick -> (v, w)
# ---------------------------------------------------------------------------

def sticks_to_vw(st: JoyState,
                 scheme: str = "arcade",
                 v_max: float = 0.4,
                 w_max: float = 1.0,
                 invert_lx: bool = False,
                 invert_ly: bool = False,
                 invert_rx: bool = False,
                 invert_ry: bool = False) -> tuple[float, float]:
    """
    Map a JoyState to (v, w) in the EKF frame: v forward+, w CCW+.

    schemes:
        arcade      v = ly * v_max       w = lx * w_max
        tank        v = 0.5*(ly+ry)*vmax w = 0.5*(lx-rx)*wmax
        left_only   v = ly * v_max       w = lx * w_max   (same as arcade)
        right_only  v = ry * v_max       w = rx * w_max

    invert_* flags flip the sign of the named axis before mapping. Use
    them to correct for physical stick wiring / firmware invert flags
    without editing the firmware.

    Result is clamped to [-v_max, v_max] x [-w_max, w_max].
    """
    lx = -st.lx if invert_lx else st.lx
    ly = -st.ly if invert_ly else st.ly
    rx = -st.rx if invert_rx else st.rx
    ry = -st.ry if invert_ry else st.ry

    if scheme == "arcade" or scheme == "left_only":
        v = ly * v_max
        w = lx * w_max
    elif scheme == "tank":
        v = 0.5 * (ly + ry) * v_max
        w = 0.5 * (lx - rx) * w_max
    elif scheme == "right_only":
        v = ry * v_max
        w = rx * w_max
    else:
        # Unknown scheme: return zeros rather than guess.
        return 0.0, 0.0

    return _clamp(v, -v_max, v_max), _clamp(w, -w_max, w_max)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _is_finite(x: float) -> bool:
    # math.isfinite is available on 3.2+; keep it simple.
    return x == x and x not in (float("inf"), float("-inf"))