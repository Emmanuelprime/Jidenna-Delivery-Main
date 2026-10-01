"""
test_costmap_gui.py

Interactive Tkinter GUI for testing costmap generation while driving.

Layout:
    +--------------------------------------------------+----------------+
    |                                                  |  CONTROLS      |
    |                Live costmap canvas               |  [Forward]     |
    |          (robot pose + trajectory overlay)       |  [Back]        |
    |                                                  |  [Left] [Right]|
    |                                                  |  [Spin CCW]    |
    |                                                  |  [Spin CW]     |
    |                                                  |  [STOP]        |
    |                                                  |                |
    |                                                  |  v: 0.25 m/s   |
    |                                                  |  w: 0.40 r/s   |
    |                                                  |  [Reset map]   |
    |                                                  |  [Save PNG]    |
    |                                                  |  [Save NPY]    |
    +--------------------------------------------------+----------------+
    status bar: bridge / lidar / pose / map stats

Keyboard (when window has focus):
    W / S   : forward / backward
    A / D   : turn left / right
    Q / E   : spin in place CCW / CW
    Space   : stop
    R       : reset costmap
    Esc     : quit

Usage:
    python gui.py --port /dev/ttyUSB0 --lidar /dev/ttyUSB1
"""

from __future__ import annotations

import argparse
import math
import queue
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

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class CostmapGUI:

    # Canvas refresh period (ms). 100 ms = 10 Hz, plenty for watching a map.
    REFRESH_MS = 100

    # Max trajectory points to keep in the history overlay.
    MAX_TRAJ = 5000

    def __init__(self,
                 root: tk.Tk,
                 bridge: JidennaBridge,
                 pose: JidennaPose,
                 lidar: Lidar,
                 costmap: Costmap,
                 out_prefix: Path):
        self.root = root
        self.bridge = bridge
        self.pose = pose
        self.lidar = lidar
        self.cm = costmap
        self.out_prefix = out_prefix

        # Motion state
        self.v_cmd = 0.0
        self.w_cmd = 0.0   # EKF frame (CCW+)
        self.v_step = 0.25
        self.w_step = 0.40

        # Trajectory overlay
        self.traj: list[tuple[float, float, float]] = []   # x, y, th
        self.traj_lock = threading.Lock()
        self.latest_pose = PoseEstimate()
        self.pose_lock = threading.Lock()

        # Scan callback may run on the lidar thread — queue for the GUI.
        self.pose_queue: queue.Queue[PoseEstimate] = queue.Queue()

        # Costmap image buffer (RGBA, updated in place)
        self._img: Optional[tk.PhotoImage] = None
        self._img_id: Optional[int] = None

        # Debounce for saving / stats
        self._last_map_stats = (0, 0, 0)
        self._last_stats_time = 0.0

        self._build_ui()
        self._bind_keys()

        # Wire up callbacks
        self.pose.add_pose_callback(self._on_pose)
        self.lidar.add_scan_callback(self._on_scan)

        # Start the periodic refresh loop
        self.root.after(self.REFRESH_MS, self._tick)

    # ---- UI construction -------------------------------------------------

    def _build_ui(self) -> None:
        self.root.title("Jidenna Costmap Test")
        self.root.geometry("1200x820")
        self.root.minsize(800, 600)

        # ---- left: canvas ----
        left = ttk.Frame(self.root, padding=4)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(left, background="#202020",
                                highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # ---- right: controls ----
        right = ttk.Frame(self.root, padding=8, width=240)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        ttk.Label(right, text="Drive", font=("TkDefaultFont", 12, "bold")) \
            .pack(pady=(0, 6))

        grid = ttk.Frame(right)
        grid.pack()

        def btn(text, r, c, cmd, width=6):
            b = ttk.Button(grid, text=text, command=cmd, width=width)
            b.grid(row=r, column=c, padx=2, pady=2, sticky="nsew")
            return b

        btn("↑", 0, 1, lambda: self._set_motion(self.v_step, 0.0))
        btn("←", 1, 0, lambda: self._set_motion(0.0,  self.w_step))
        btn("STOP", 1, 1, self._stop)
        btn("→", 1, 2, lambda: self._set_motion(0.0, -self.w_step))
        btn("↓", 2, 1, lambda: self._set_motion(-self.v_step, 0.0))

        for i in range(3):
            grid.columnconfigure(i, weight=1)

        ttk.Separator(right).pack(fill=tk.X, pady=10)

        ttk.Label(right, text="In-place spin",
                  font=("TkDefaultFont", 11, "bold")).pack()
        spin = ttk.Frame(right)
        spin.pack()
        ttk.Button(spin, text="⟲ CCW",
                   command=lambda: self._set_motion(0.0, self.w_step)) \
            .grid(row=0, column=0, padx=2, pady=4)
        ttk.Button(spin, text="CW ⟳",
                   command=lambda: self._set_motion(0.0, -self.w_step)) \
            .grid(row=0, column=1, padx=2, pady=4)

        ttk.Separator(right).pack(fill=tk.X, pady=10)

        # ---- speed sliders ----
        # NOTE: labels are created BEFORE the scales, because ttk.Scale fires
        # its `command` callback during construction and on `.set()`. If the
        # labels didn't exist yet, the callbacks would crash with
        # AttributeError.
        ttk.Label(right, text="Linear speed (m/s)").pack(anchor="w")
        self.v_label = ttk.Label(right, text="0.25")
        self.v_label.pack(anchor="e")
        self.v_scale = ttk.Scale(right, from_=0.05, to=0.60,
                                 orient=tk.HORIZONTAL,
                                 command=self._on_v_scale)
        self.v_scale.set(0.25)
        self.v_scale.pack(fill=tk.X)

        ttk.Label(right, text="Angular speed (rad/s)").pack(anchor="w",
                                                            pady=(8, 0))
        self.w_label = ttk.Label(right, text="0.40")
        self.w_label.pack(anchor="e")
        self.w_scale = ttk.Scale(right, from_=0.10, to=1.50,
                                 orient=tk.HORIZONTAL,
                                 command=self._on_w_scale)
        self.w_scale.set(0.40)
        self.w_scale.pack(fill=tk.X)

        ttk.Separator(right).pack(fill=tk.X, pady=10)

        # ---- map controls ----
        ttk.Button(right, text="Reset costmap",
                   command=self._reset_map).pack(fill=tk.X, pady=2)
        ttk.Button(right, text="Reset trajectory",
                   command=self._reset_traj).pack(fill=tk.X, pady=2)
        ttk.Button(right, text="Save PNG",
                   command=self._save_png).pack(fill=tk.X, pady=2)
        ttk.Button(right, text="Save NPY",
                   command=self._save_npy).pack(fill=tk.X, pady=2)

        ttk.Separator(right).pack(fill=tk.X, pady=10)

        # ---- stats ----
        ttk.Label(right, text="Map stats",
                  font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        self.stats_label = ttk.Label(right, text="—", justify="left")
        self.stats_label.pack(anchor="w", pady=4)

        # ---- bottom: status bar ----
        self.status_var = tk.StringVar(value="starting…")
        status = ttk.Label(self.root, textvariable=self.status_var,
                           relief=tk.SUNKEN, anchor="w", padding=4)
        status.pack(side=tk.BOTTOM, fill=tk.X)

    def _bind_keys(self) -> None:
        # Don't force focus to the canvas only; bind on root so WASD works
        # regardless of which widget has focus (except entries, of which
        # there are none here).
        self.root.focus_set()

        def bind(key, fn):
            self.root.bind(key, lambda e: fn())

        bind("<KeyPress-w>", lambda: self._set_motion(self.v_step, 0.0))
        bind("<KeyPress-s>", lambda: self._set_motion(-self.v_step, 0.0))
        bind("<KeyPress-a>", lambda: self._set_motion(0.0,  self.w_step))
        bind("<KeyPress-d>", lambda: self._set_motion(0.0, -self.w_step))
        bind("<KeyPress-q>", lambda: self._set_motion(0.0,  self.w_step))
        bind("<KeyPress-e>", lambda: self._set_motion(0.0, -self.w_step))
        bind("<KeyPress-space>", self._stop)
        bind("<KeyPress-r>", self._reset_map)
        bind("<Escape>", self._quit)

        # Key-release also stops for momentary control feel.
        self.root.bind("<KeyRelease-w>", lambda e: self._stop())
        self.root.bind("<KeyRelease-s>", lambda e: self._stop())
        self.root.bind("<KeyRelease-a>", lambda e: self._stop())
        self.root.bind("<KeyRelease-d>", lambda e: self._stop())
        self.root.bind("<KeyRelease-q>", lambda e: self._stop())
        self.root.bind("<KeyRelease-e>", lambda e: self._stop())

    # ---- motion commands -------------------------------------------------

    def _set_motion(self, v: float, w_ccw: float) -> None:
        self.v_cmd = v
        self.w_cmd = w_ccw
        # Nano expects CW+ w; EKF frame is CCW+.
        try:
            self.bridge.set_velocity(v, -w_ccw)
        except Exception as e:
            self._set_status(f"set_velocity failed: {e}")

    def _stop(self) -> None:
        self._set_motion(0.0, 0.0)

    # ---- slider callbacks -----------------------------------------------

    def _on_v_scale(self, val: str) -> None:
        self.v_step = float(val)
        # Guard in case the callback fires before the label exists.
        if hasattr(self, "v_label"):
            self.v_label.config(text=f"{self.v_step:.2f}")

    def _on_w_scale(self, val: str) -> None:
        self.w_step = float(val)
        if hasattr(self, "w_label"):
            self.w_label.config(text=f"{self.w_step:.2f}")

    # ---- bridge / lidar callbacks (fire on background threads) ----------

    def _on_pose(self, p: PoseEstimate) -> None:
        with self.pose_lock:
            self.latest_pose = p
        with self.traj_lock:
            self.traj.append((p.x, p.y, p.th))
            if len(self.traj) > self.MAX_TRAJ:
                # Drop oldest 10%
                del self.traj[:len(self.traj) // 10]

    def _on_scan(self, scan: LidarScan) -> None:
        with self.pose_lock:
            p = self.latest_pose
        try:
            self.cm.integrate_scan(scan, p)
        except Exception as e:
            print(f"[costmap] integrate_scan error: {e}")

    # ---- periodic GUI refresh -------------------------------------------

    def _tick(self) -> None:
        try:
            self._redraw()
            self._update_status()
        except Exception as e:
            print(f"[gui] tick error: {e}")
        self.root.after(self.REFRESH_MS, self._tick)

    def _redraw(self) -> None:
        W = self.canvas.winfo_width()
        H = self.canvas.winfo_height()
        if W < 20 or H < 20:
            return

        # The costmap is square; use the smaller canvas dimension.
        side = min(W, H) - 20
        if side < 50:
            return

        # Snapshot the probability grid and turn it into an RGB image.
        prob = self.cm.probability_grid()   # (ny, nx), float in [0,1]
        ny, nx = prob.shape

        # Downsample to keep Tk's PhotoImage happy. Target ~400 px.
        target = 400
        if nx > target or ny > target:
            step_x = max(1, nx // target)
            step_y = max(1, ny // target)
            small = prob[::step_y, ::step_x]
        else:
            small = prob

        sh, sw = small.shape

        # Build a PPM (P6) byte string — Tkinter can read this directly.
        v = np.clip(prob_to_gray(small), 0, 255).astype(np.uint8)
        rgb = np.empty((sh, sw, 3), dtype=np.uint8)
        rgb[..., 0] = v
        rgb[..., 1] = v
        rgb[..., 2] = v

        ppm_header = f"P6\n{sw} {sh}\n255\n".encode("ascii")
        ppm_data = ppm_header + rgb.tobytes()

        if self._img is None:
            self._img = tk.PhotoImage(data=ppm_data, format="PPM")
        else:
            try:
                self._img.configure(data=ppm_data, format="PPM")
            except tk.TclError:
                self._img = tk.PhotoImage(data=ppm_data, format="PPM")

        # Clear canvas
        self.canvas.delete("all")

        # Place image centered
        cx, cy = W // 2, H // 2
        self.canvas.create_image(cx, cy, image=self._img, anchor=tk.CENTER)

        # Overlay trajectory. Map world -> canvas.
        px_per_m = side / self.cm.size_x
        ox = cx - side / 2
        oy = cy - side / 2

        def world_to_canvas(x: float, y: float) -> tuple[float, float]:
            # +y is up in world; canvas y is down, so flip.
            cx_ = ox + (x - self.cm.origin_x) * px_per_m
            cy_ = oy + side - (y - self.cm.origin_y) * px_per_m
            return cx_, cy_

        with self.traj_lock:
            traj = list(self.traj)

        if len(traj) >= 2:
            coords = []
            for (x, y, _) in traj:
                px, py = world_to_canvas(x, y)
                coords.extend([px, py])
            self.canvas.create_line(*coords, fill="#ff3030", width=2)

        # Draw heading arrows every N points
        if traj:
            step = max(1, len(traj) // 25)
            for (x, y, th) in traj[::step]:
                px, py = world_to_canvas(x, y)
                L = 0.25 * px_per_m
                ex = px + L * math.cos(th)
                ey = py - L * math.sin(th)
                self.canvas.create_line(px, py, ex, ey,
                                        fill="#30a0ff", width=2,
                                        arrow=tk.LAST)

            # Start / current markers
            x0, y0, _ = traj[0]
            px0, py0 = world_to_canvas(x0, y0)
            self.canvas.create_oval(px0 - 5, py0 - 5, px0 + 5, py0 + 5,
                                    fill="#00ff00", outline="")

            xN, yN, _ = traj[-1]
            pxN, pyN = world_to_canvas(xN, yN)
            self.canvas.create_oval(pxN - 6, pyN - 6, pxN + 6, pyN + 6,
                                    fill="#ffff00", outline="")

            # Robot body: a small circle with a heading line
            L = 0.30 * px_per_m
            ex = pxN + L * math.cos(traj[-1][2])
            ey = pyN - L * math.sin(traj[-1][2])
            self.canvas.create_oval(pxN - 8, pyN - 8, pxN + 8, pyN + 8,
                                    outline="#ffff00", width=2)
            self.canvas.create_line(pxN, pyN, ex, ey,
                                    fill="#ffff00", width=3)

        # Odom origin marker
        px0, py0 = world_to_canvas(0.0, 0.0)
        self.canvas.create_oval(px0 - 3, py0 - 3, px0 + 3, py0 + 3,
                                fill="#c080ff", outline="")

    def _update_status(self) -> None:
        now = time.time()
        if now - self._last_stats_time < 0.5:
            return
        self._last_stats_time = now

        prob = self.cm.probability_grid()
        occ = int(np.sum(prob > 0.6))
        free = int(np.sum(prob < 0.4))
        unk = prob.size - occ - free
        self._last_map_stats = (occ, free, unk)

        self.stats_label.config(
            text=(f"occupied: {occ:>7d}\n"
                  f"free    : {free:>7d}\n"
                  f"unknown : {unk:>7d}\n"
                  f"total   : {prob.size:>7d}")
        )

        with self.pose_lock:
            p = self.latest_pose

        link = "OK" if self.bridge.is_connected() else "DOWN"
        lidar = "OK" if self.lidar.is_connected() else "DOWN"
        wd = "EXPIRED" if self.bridge.is_watchdog_expired() else "ok"

        self._set_status(
            f"Nano: {link}   LiDAR: {lidar}   wd: {wd}   "
            f"cmd: v={self.v_cmd:+.2f} w={self.w_cmd:+.2f}   "
            f"pose: x={p.x:+.2f} y={p.y:+.2f} "
            f"th={math.degrees(p.th):+.1f}°   "
            f"scans: {self.lidar.scan_count()}"
        )

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    # ---- actions --------------------------------------------------------

    def _reset_map(self) -> None:
        self.cm.reset()
        self._set_status("costmap reset")

    def _reset_traj(self) -> None:
        with self.traj_lock:
            self.traj.clear()
        self._set_status("trajectory reset")

    def _save_png(self) -> None:
        if not HAVE_MPL:
            messagebox.showerror("Save PNG",
                                 "matplotlib not available")
            return
        path = self.out_prefix.with_suffix(".png")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._render_save_png(path)
            self._set_status(f"saved {path}")
        except Exception as e:
            messagebox.showerror("Save PNG", str(e))

    def _render_save_png(self, path: Path) -> None:
        prob = self.cm.probability_grid()
        with self.traj_lock:
            traj = list(self.traj)

        fig, ax = plt.subplots(figsize=(10, 10))
        extent = [self.cm.origin_x, self.cm.origin_x + self.cm.size_x,
                  self.cm.origin_y, self.cm.origin_y + self.cm.size_y]
        ax.imshow(prob, origin="lower", extent=extent,
                  cmap="gray_r", vmin=0.0, vmax=1.0,
                  interpolation="nearest")
        if traj:
            xs = [t[0] for t in traj]
            ys = [t[1] for t in traj]
            ax.plot(xs, ys, "-r", linewidth=2, label="trajectory")
            ax.plot(xs[0], ys[0], "go", markersize=10, label="start")
            ax.plot(xs[-1], ys[-1], "ro", markersize=10, label="end")
            step = max(1, len(traj) // 30)
            for (x, y, th) in traj[::step]:
                ax.arrow(x, y, 0.15 * math.cos(th), 0.15 * math.sin(th),
                         head_width=0.05, head_length=0.05,
                         fc="blue", ec="blue", alpha=0.6)
            ax.legend(loc="upper right")
        ax.set_xlabel("x [m] (odom)")
        ax.set_ylabel("y [m] (odom)")
        ax.set_title("Jidenna costmap")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)

    def _save_npy(self) -> None:
        path = self.out_prefix.with_suffix(".npy")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.cm.snapshot())
        self._set_status(f"saved {path}")

    def _quit(self) -> None:
        self._stop()
        self.root.after(100, self.root.destroy)


# ---------------------------------------------------------------------------
# Colour mapping helper
# ---------------------------------------------------------------------------

def prob_to_gray(prob: np.ndarray) -> np.ndarray:
    """
    Map occupancy probability [0..1] to a grayscale value [0..255].

        0.0 (free)      -> bright (~230)
        0.5 (unknown)   -> mid gray (~ 70)
        1.0 (occupied)  -> black  (  0)

    Unknown cells render as dark gray, which lets you distinguish
    "not yet seen" from "seen and free".
    """
    p = np.clip(prob, 0.0, 1.0)
    out = np.where(
        p <= 0.5,
        230.0 - p * (230.0 - 70.0) / 0.5,
        70.0  - (p - 0.5) * 70.0 / 0.5,
    )
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Interactive costmap GUI")
    ap.add_argument("--port",  required=True, help="Nano serial port")
    ap.add_argument("--lidar", required=True, help="LiDAR serial port")
    ap.add_argument("--baud",  type=int, default=115200)
    ap.add_argument("--out",   default="runs/costmap_gui",
                    help="output prefix for saved PNG/NPY")
    ap.add_argument("--size",  type=float, default=15.0)
    ap.add_argument("--resolution", type=float, default=0.05)
    ap.add_argument("--lidar-dx", type=float, default=-0.432)
    ap.add_argument("--lidar-dy", type=float, default=0.0)
    args = ap.parse_args()

    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # --- costmap ---
    cm = Costmap(
        resolution=args.resolution,
        size_x=args.size, size_y=args.size,
        lidar_dx=args.lidar_dx, lidar_dy=args.lidar_dy,
        lidar_min_range=0.20, lidar_max_range=8.0,
    )

    # --- bridge / pose / lidar ---
    bridge = JidennaBridge(args.port, baud=args.baud)
    pose   = JidennaPose(bridge)
    lidar  = Lidar(args.lidar, baud=args.baud)

    # --- Tk root ---
    root = tk.Tk()
    gui = CostmapGUI(root, bridge, pose, lidar, cm, out_prefix)

    # Ctrl-C in the terminal closes the GUI cleanly.
    def handle_sigint(sig, frame):
        root.after(0, gui._quit)
    signal.signal(signal.SIGINT, handle_sigint)

    # --- background connections (so the GUI appears immediately) ---
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

        gui._set_status("connected — use WASD / buttons to drive")

    threading.Thread(target=startup, daemon=True, name="startup").start()

    # --- clean shutdown when window closes ---
    def on_close():
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
        # Belt and braces — ensure the robot is stopped no matter how we exit.
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

    return 0


if __name__ == "__main__":
    sys.exit(main())