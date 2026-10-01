"""
gui.py

Live costmap viewer with joystick control.

- Left panel: live costmap + trajectory overlay (only UI element besides status).
- Joystick (UDP from joy_firmware.ino) drives the robot automatically.
- Deadman: if no UDP packets for --joy-timeout seconds, robot stops.

Keyboard:
    R   : reset costmap
    Esc : quit

Usage:
    python gui.py --port /dev/ttyUSB0 --lidar /dev/ttyUSB1 --udp-port 4210
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox
from typing import Optional

import numpy as np

from jidenna.jidenna_bridge import JidennaBridge
from jidenna.jidenna_lidar  import Lidar, LidarScan
from jidenna.jidenna_pose   import JidennaPose, PoseEstimate
from jidenna.perception.costmap import Costmap
from jidenna.controllers.joy_receiver import JoyReceiver, sticks_to_vw


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class CostmapGUI:

    REFRESH_MS = 100
    MAX_TRAJ = 5000

    def __init__(self,
                 root: tk.Tk,
                 bridge: JidennaBridge,
                 pose: JidennaPose,
                 lidar: Lidar,
                 costmap: Costmap,
                 joy_rx: JoyReceiver,
                 joy_v_max: float = 0.30,
                 joy_w_max: float = 0.80,
                 joy_timeout_s: float = 0.5,
                 joy_scheme: str = "arcade",
                 joy_invert_lx: bool = False):
        self.root = root
        self.bridge = bridge
        self.pose = pose
        self.lidar = lidar
        self.cm = costmap
        self.joy_rx = joy_rx

        self.joy_v_max = joy_v_max
        self.joy_w_max = joy_w_max
        self.joy_timeout_s = joy_timeout_s
        self.joy_scheme = joy_scheme
        self.joy_invert_lx = joy_invert_lx
        self._joy_last_send = 0.0

        self.traj: list[tuple[float, float, float]] = []
        self.traj_lock = threading.Lock()
        self.latest_pose = PoseEstimate()
        self.pose_lock = threading.Lock()

        self._img: Optional[tk.PhotoImage] = None
        self._last_stats_time = 0.0

        self._build_ui()
        self._bind_keys()

        self.pose.add_pose_callback(self._on_pose)
        self.lidar.add_scan_callback(self._on_scan)

        self.root.after(self.REFRESH_MS, self._tick)

    # ---- UI ---------------------------------------------------------------

    def _build_ui(self) -> None:
        self.root.title("Jidenna Costmap")
        self.root.geometry("1000x900")
        self.root.minsize(600, 600)

        self.canvas = tk.Canvas(self.root, background="#202020",
                                highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value="starting…")
        status = ttk.Label(self.root, textvariable=self.status_var,
                           relief=tk.SUNKEN, anchor="w", padding=4)
        status.pack(side=tk.BOTTOM, fill=tk.X)

    def _bind_keys(self) -> None:
        self.root.focus_set()
        self.root.bind("<KeyPress-r>", lambda e: self._reset_map())
        self.root.bind("<Escape>", lambda e: self._quit())

    # ---- joystick --------------------------------------------------------

    def _update_joystick(self) -> None:
        """Called from _tick. Sends at most 20 Hz."""
        now = time.time()
        if now - self._joy_last_send < 0.05:
            return
        self._joy_last_send = now

        if self.joy_rx.last_rx_age_s() > self.joy_timeout_s:
            try:
                self.bridge.set_velocity(0.0, 0.0)
            except Exception:
                pass
            return

        st = self.joy_rx.get_state()
        v, w = sticks_to_vw(st, self.joy_scheme,
                            self.joy_v_max, self.joy_w_max,
                            invert_lx=self.joy_invert_lx)
        try:
            self.bridge.set_velocity(v, -w)   # Nano expects CW+
        except Exception as e:
            self._set_status(f"joy set_velocity failed: {e}")

    # ---- bridge/lidar callbacks -----------------------------------------

    def _on_pose(self, p: PoseEstimate) -> None:
        with self.pose_lock:
            self.latest_pose = p
        with self.traj_lock:
            self.traj.append((p.x, p.y, p.th))
            if len(self.traj) > self.MAX_TRAJ:
                del self.traj[:len(self.traj) // 10]

    def _on_scan(self, scan: LidarScan) -> None:
        with self.pose_lock:
            p = self.latest_pose
        try:
            self.cm.integrate_scan(scan, p)
        except Exception as e:
            print(f"[costmap] integrate_scan error: {e}")

    # ---- refresh --------------------------------------------------------

    def _tick(self) -> None:
        try:
            self._redraw()
            self._update_joystick()
            self._update_status()
        except Exception as e:
            print(f"[gui] tick error: {e}")
        self.root.after(self.REFRESH_MS, self._tick)

    def _redraw(self) -> None:
        W = self.canvas.winfo_width()
        H = self.canvas.winfo_height()
        if W < 20 or H < 20:
            return

        side = min(W, H) - 20
        if side < 50:
            return

        prob = self.cm.probability_grid()
        ny, nx = prob.shape

        target = 500
        if nx > target or ny > target:
            sx = max(1, nx // target)
            sy = max(1, ny // target)
            small = prob[::sy, ::sx]
        else:
            small = prob

        sh, sw = small.shape
        v = np.clip(prob_to_gray(small), 0, 255).astype(np.uint8)
        rgb = np.empty((sh, sw, 3), dtype=np.uint8)
        rgb[..., 0] = v
        rgb[..., 1] = v
        rgb[..., 2] = v

        ppm = f"P6\n{sw} {sh}\n255\n".encode("ascii") + rgb.tobytes()

        if self._img is None:
            self._img = tk.PhotoImage(data=ppm, format="PPM")
        else:
            try:
                self._img.configure(data=ppm, format="PPM")
            except tk.TclError:
                self._img = tk.PhotoImage(data=ppm, format="PPM")

        self.canvas.delete("all")
        cx, cy = W // 2, H // 2
        self.canvas.create_image(cx, cy, image=self._img, anchor=tk.CENTER)

        px_per_m = side / self.cm.size_x
        ox = cx - side / 2
        oy = cy - side / 2

        def w2c(x, y):
            return (ox + (x - self.cm.origin_x) * px_per_m,
                    oy + side - (y - self.cm.origin_y) * px_per_m)

        with self.traj_lock:
            traj = list(self.traj)

        if len(traj) >= 2:
            coords = []
            for (x, y, _) in traj:
                px, py = w2c(x, y)
                coords.extend([px, py])
            self.canvas.create_line(*coords, fill="#ff3030", width=2)

        if traj:
            step = max(1, len(traj) // 25)
            for (x, y, th) in traj[::step]:
                px, py = w2c(x, y)
                L = 0.25 * px_per_m
                self.canvas.create_line(px, py,
                                        px + L * math.cos(th),
                                        py - L * math.sin(th),
                                        fill="#30a0ff", width=2,
                                        arrow=tk.LAST)

            x0, y0, _ = traj[0]
            px0, py0 = w2c(x0, y0)
            self.canvas.create_oval(px0 - 5, py0 - 5, px0 + 5, py0 + 5,
                                    fill="#00ff00", outline="")

            xN, yN, thN = traj[-1]
            pxN, pyN = w2c(xN, yN)
            self.canvas.create_oval(pxN - 6, pyN - 6, pxN + 6, pyN + 6,
                                    fill="#ffff00", outline="")
            L = 0.30 * px_per_m
            self.canvas.create_line(pxN, pyN,
                                    pxN + L * math.cos(thN),
                                    pyN - L * math.sin(thN),
                                    fill="#ffff00", width=3)

        # Odom origin
        px0, py0 = w2c(0.0, 0.0)
        self.canvas.create_oval(px0 - 3, py0 - 3, px0 + 3, py0 + 3,
                                fill="#c080ff", outline="")

    def _update_status(self) -> None:
        now = time.time()
        if now - self._last_stats_time < 0.25:
            return
        self._last_stats_time = now

        prob = self.cm.probability_grid()
        occ = int(np.sum(prob > 0.6))
        free = int(np.sum(prob < 0.4))

        with self.pose_lock:
            p = self.latest_pose

        st = self.joy_rx.get_state()
        age = self.joy_rx.last_rx_age_s()
        if age == float("inf"):
            joy = "no pkt"
        elif age > self.joy_timeout_s:
            joy = f"STALE {age*1000:.0f}ms"
        else:
            joy = f"ly={st.ly:+.2f} lx={st.lx:+.2f}"

        nano = "OK" if self.bridge.is_connected() else "DOWN"
        lidar = "OK" if self.lidar.is_connected() else "DOWN"

        self._set_status(
            f"nano:{nano}  lidar:{lidar}  joy:{joy}  "
            f"pose x={p.x:+.2f} y={p.y:+.2f} "
            f"th={math.degrees(p.th):+.1f}°  "
            f"occ:{occ} free:{free}  "
            f"scans:{self.lidar.scan_count()}   [R]eset  [Esc]quit"
        )

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _reset_map(self) -> None:
        self.cm.reset()
        with self.traj_lock:
            self.traj.clear()
        self._set_status("reset")

    def _quit(self) -> None:
        self.root.after(50, self.root.destroy)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def prob_to_gray(prob: np.ndarray) -> np.ndarray:
    p = np.clip(prob, 0.0, 1.0)
    return np.where(
        p <= 0.5,
        230.0 - p * (230.0 - 70.0) / 0.5,
        70.0  - (p - 0.5) * 70.0 / 0.5,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Costmap viewer + joystick")
    ap.add_argument("--port",  required=True)
    ap.add_argument("--lidar", required=True)
    ap.add_argument("--baud",  type=int, default=115200)
    ap.add_argument("--udp-port", type=int, default=4210)
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--scheme", default="arcade",
                    choices=["arcade", "tank", "left_only", "right_only"])
    ap.add_argument("--joy-v-max", type=float, default=0.30)
    ap.add_argument("--joy-w-max", type=float, default=0.80)
    ap.add_argument("--joy-timeout", type=float, default=0.5)
    ap.add_argument("--joy-invert-lx", action="store_true")
    ap.add_argument("--size", type=float, default=15.0)
    ap.add_argument("--resolution", type=float, default=0.05)
    ap.add_argument("--lidar-dx", type=float, default=-0.432)
    ap.add_argument("--lidar-dy", type=float, default=0.0)
    args = ap.parse_args()

    cm = Costmap(
        resolution=args.resolution,
        size_x=args.size, size_y=args.size,
        lidar_dx=args.lidar_dx, lidar_dy=args.lidar_dy,
        lidar_min_range=0.20, lidar_max_range=8.0,
    )

    bridge = JidennaBridge(args.port, baud=args.baud)
    pose   = JidennaPose(bridge)
    lidar  = Lidar(args.lidar, baud=args.baud)
    joy_rx = JoyReceiver(udp_port=args.udp_port, bind_addr=args.bind)

    root = tk.Tk()
    gui = CostmapGUI(root, bridge, pose, lidar, cm, joy_rx,
                     joy_v_max=args.joy_v_max,
                     joy_w_max=args.joy_w_max,
                     joy_timeout_s=args.joy_timeout,
                     joy_scheme=args.scheme,
                     joy_invert_lx=args.joy_invert_lx)

    def handle_sigint(sig, frame):
        root.after(0, gui._quit)
    signal.signal(signal.SIGINT, handle_sigint)

    def startup():
        gui._set_status(f"opening bridge {args.port}…")
        bridge.start()
        t0 = time.time()
        while not bridge.is_connected() and time.time() - t0 < 10.0:
            time.sleep(0.1)
        if not bridge.is_connected():
            root.after(0, lambda: messagebox.showerror(
                "Bridge", f"No telemetry on {args.port}"))
            root.after(0, gui._quit)
            return

        pose.start()
        time.sleep(0.5)

        gui._set_status(f"opening lidar {args.lidar}…")
        lidar.start()
        t0 = time.time()
        while not lidar.is_connected() and time.time() - t0 < 5.0:
            time.sleep(0.1)
        if not lidar.is_connected():
            root.after(0, lambda: messagebox.showerror(
                "LiDAR", f"No data on {args.lidar}"))
            root.after(0, gui._quit)
            return

        gui._set_status("opening UDP…")
        joy_rx.start()
        time.sleep(0.3)
        if joy_rx.bind_failed():
            root.after(0, lambda: messagebox.showerror(
                "Joystick UDP", f"bind failed: {joy_rx.bind_error()}"))
            root.after(0, gui._quit)
            return

        gui._set_status("ready — move the joystick")

    threading.Thread(target=startup, daemon=True).start()

    def on_close():
        try:
            joy_rx.stop()
        except Exception:
            pass
        try:
            bridge.set_velocity(0.0, 0.0)
        except Exception:
            pass
        try:
            lidar.stop()
        except Exception:
            pass
        try:
            pose.stop()
        except Exception:
            pass
        try:
            bridge.stop()
        except Exception:
            pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    try:
        root.mainloop()
    finally:
        try:
            bridge.set_velocity(0.0, 0.0)
        except Exception:
            pass
        try:
            bridge.stop()
        except Exception:
            pass
        try:
            lidar.stop()
        except Exception:
            pass
        try:
            joy_rx.stop()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())