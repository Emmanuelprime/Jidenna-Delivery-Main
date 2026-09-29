"""
jidenna_logger.py

Attaches to JidennaBridge as a telemetry callback and logs every sample
to a CSV file. Runs until Ctrl-C, then cleanly stops the robot and closes
the log.

Usage:
    python jidenna_logger.py --port COM19 --out run_001.csv
    python jidenna_logger.py --port /dev/ttyUSB0 --out run_001.csv --comment "arc test at w=0.2"

While running, you can drive the robot from another script (the bridge
is not exclusive to the logger — but only one process can open the
serial port). To combine control + logging in one process, see the
`with_control` example at the bottom.

CSV columns:
    t_host, x, y, th, vL, vR, wL, wR, bat, temp, fb_age, wd, or, imu_yaw
"""

from __future__ import annotations

import argparse
import csv
import signal
import sys
import time
from pathlib import Path

from .jidenna_bridge import JidennaBridge, Telemetry
import threading


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class CsvLogger:
    """Writes Telemetry samples + EKF pose to a CSV file."""

    COLUMNS = [
        "t_host", "x", "y", "th",
        "vL", "vR", "wL", "wR",
        "bat", "temp", "fb_age", "wd", "or", "imu_yaw",
        # EKF pose columns (appended)
        "ekf_x", "ekf_y", "ekf_th", "ekf_v", "ekf_w",
        "ekf_Pxx", "ekf_Pyy", "ekf_Pthth",
    ]

    def __init__(self, path: Path, comment: str = ""):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", newline="")
        self._writer = csv.writer(self._file)

        if comment:
            self._file.write(f"# {comment}\n")
            self._file.write(f"# started {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        self._writer.writerow(self.COLUMNS)
        self._file.flush()

        # Latest EKF pose, updated via the pose callback
        self._ekf_row = {
            "ekf_x": 0.0, "ekf_y": 0.0, "ekf_th": 0.0,
            "ekf_v": 0.0, "ekf_w": 0.0,
            "ekf_Pxx": 0.0, "ekf_Pyy": 0.0, "ekf_Pthth": 0.0,
        }
        self._ekf_lock = threading.Lock()

        self.count = 0
        self.t_first = None
        self.t_last = None

    def on_pose(self, p) -> None:
        """Register this with JidennaPose.add_pose_callback."""
        with self._ekf_lock:
            self._ekf_row["ekf_x"] = p.x
            self._ekf_row["ekf_y"] = p.y
            self._ekf_row["ekf_th"] = p.th
            self._ekf_row["ekf_v"] = p.v
            self._ekf_row["ekf_w"] = p.w
            self._ekf_row["ekf_Pxx"] = p.P_xx
            self._ekf_row["ekf_Pyy"] = p.P_yy
            self._ekf_row["ekf_Pthth"] = p.P_thth

    def __call__(self, t: Telemetry) -> None:
        with self._ekf_lock:
            ekf = dict(self._ekf_row)

        row = [
            f"{t.t_host:.6f}",
            f"{t.x:.6f}", f"{t.y:.6f}", f"{t.th:.6f}",
            f"{t.vL:.4f}", f"{t.vR:.4f}",
            f"{t.wL:.4f}", f"{t.wR:.4f}",
            f"{t.bat:.2f}", f"{t.temp:.1f}",
            t.fb_age, t.wd, t.odom_reset_count,
            f"{t.imu_yaw:.2f}",
            f"{ekf['ekf_x']:.6f}", f"{ekf['ekf_y']:.6f}", f"{ekf['ekf_th']:.6f}",
            f"{ekf['ekf_v']:.4f}",  f"{ekf['ekf_w']:.4f}",
            f"{ekf['ekf_Pxx']:.6f}", f"{ekf['ekf_Pyy']:.6f}", f"{ekf['ekf_Pthth']:.6f}",
        ]
        self._writer.writerow(row)
        self.count += 1
        if self.t_first is None:
            self.t_first = t.t_host
        self.t_last = t.t_host

        if self.count % 20 == 0:
            self._file.flush()

    def close(self) -> None:
        try:
            self._file.flush()
            self._file.close()
        except Exception:
            pass

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Log Jidenna telemetry to CSV.")
    ap.add_argument("--port", required=True, help="serial port (COM19 or /dev/ttyUSB0)")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--comment", default="", help="optional note written to CSV header")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--print-every", type=int, default=10,
                    help="print every Nth sample to stdout (0 = silent)")
    args = ap.parse_args()

    bridge = JidennaBridge(args.port, baud=args.baud)
    logger = CsvLogger(Path(args.out), comment=args.comment)

    counter = {"n": 0}

    def on_telemetry(t: Telemetry) -> None:
        logger(t)
        if args.print_every > 0:
            counter["n"] += 1
            if counter["n"] % args.print_every == 0:
                print(t)

    bridge.add_telemetry_callback(on_telemetry)

    # Ctrl-C handler: mark stop, but let cleanup happen below
    stop = {"flag": False}
    def handle_sigint(sig, frame):
        stop["flag"] = True
    signal.signal(signal.SIGINT, handle_sigint)

    print(f"Opening {args.port}...")
    bridge.start()

    # Wait for connection
    t0 = time.time()
    while not bridge.is_connected() and time.time() - t0 < 10.0:
        time.sleep(0.1)
    if not bridge.is_connected():
        print("ERROR: no telemetry. Check cable/port/Nano.")
        bridge.stop()
        logger.close()
        return 1

    print(f"Connected. Logging to {args.out}")
    print("Ctrl-C to stop.")
    try:
        while not stop["flag"]:
            time.sleep(0.2)
    finally:
        print("\nStopping...")
        bridge.set_velocity(0.0, 0.0)
        time.sleep(0.2)
        bridge.stop()
        logger.close()

    if logger.t_first is not None:
        dur = logger.t_last - logger.t_first
        print(f"Logged {logger.count} samples over {dur:.2f} s "
              f"({logger.count/dur:.1f} Hz avg).")
    else:
        print("No samples logged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())