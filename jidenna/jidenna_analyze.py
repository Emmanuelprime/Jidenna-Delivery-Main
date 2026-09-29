"""
jidenna_analyze.py

Reads a Jidenna telemetry CSV, runs it through the EKF (from jidenna_ekf),
and produces:
    - A summary printed to stdout
    - Optional plots saved as PNG

Usage:
    python jidenna_analyze.py runs/forward_01.csv
    python jidenna_analyze.py runs/forward_01.csv --plot runs/forward_01.png
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

from .jidenna_ekf import EKF, EKFConfig   # <-- import, don't duplicate

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLUMNS = ["t_host", "x", "y", "th", "vL", "vR", "wL", "wR",
           "bat", "temp", "fb_age", "wd", "or", "imu_yaw"]

WHEEL_SEPARATION_M = 0.521


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_csv(path: Path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            rows.append(line.strip())

    reader = csv.DictReader(rows)
    data = {col: [] for col in COLUMNS}
    for row in reader:
        for col in COLUMNS:
            try:
                data[col].append(float(row[col]))
            except (KeyError, ValueError):
                data[col].append(float("nan"))

    return {k: np.array(v, dtype=float) for k, v in data.items()}


# ---------------------------------------------------------------------------
# EKF runner
# ---------------------------------------------------------------------------

def run_ekf_on_data(data) -> dict:
    """Feed a logged CSV through the EKF. Returns arrays for plotting."""
    t = data["t_host"] - data["t_host"][0]
    n = len(t)

    ekf = EKF(EKFConfig())
    ex  = np.zeros(n); ey  = np.zeros(n)
    eth = np.zeros(n); ev  = np.zeros(n); ew = np.zeros(n)

    ekf.x[0] = data["x"][0]
    ekf.x[1] = data["y"][0]
    ekf.x[2] = -data["th"][0]

    # Anchor IMU yaw to odom heading at t=0 (IMU has arbitrary offset).
    imu_yaw0 = math.radians(-data["imu_yaw"][0])
    imu_correction = ekf.x[2] - imu_yaw0

    for k in range(n):
        if k > 0:
            dt = t[k] - t[k-1]
            if 0.0 < dt < 0.5:
                ekf.predict(dt)

        ekf.update_wheels(data["vL"][k], data["vR"][k])

        imu_yaw_deg_corrected = math.degrees(
            math.radians(-data["imu_yaw"][k]) + imu_correction
        )
        ekf.update_imu_yaw(imu_yaw_deg_corrected)

        ex[k]  = ekf.x[0]
        ey[k]  = ekf.x[1]
        eth[k] = ekf.x[2]
        ev[k]  = ekf.x[3]
        ew[k]  = ekf.x[4]

    return {
        "t": t,
        "ekf_x": ex, "ekf_y": ey, "ekf_th": eth,
        "ekf_v": ev, "ekf_w": ew,
        "odom_th": -data["th"],
        "imu_th":  np.deg2rad(-data["imu_yaw"]) - imu_correction,   # <-- FIXED
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze(data):
    t = data["t_host"]
    if len(t) < 2:
        return {"error": "not enough samples"}

    t = t - t[0]
    dt = np.diff(t)
    duration = t[-1]
    n = len(t)
    rate = n / duration if duration > 0 else 0.0

    vL = data["vL"]; vR = data["vR"]
    v_mean = 0.5 * (vL + vR)

    x = data["x"]; y = data["y"]; th = data["th"]
    dx = np.diff(x); dy = np.diff(y)
    dist_odom = float(np.sum(np.hypot(dx, dy)))

    th_unwrapped = np.unwrap(th)
    total_th = float(th_unwrapped[-1] - th_unwrapped[0])

    imu_yaw = np.deg2rad(data["imu_yaw"])
    imu_yaw_unwrapped = np.unwrap(imu_yaw)
    total_imu_yaw = float(imu_yaw_unwrapped[-1] - imu_yaw_unwrapped[0])

    straight_mask = np.abs(vL - vR) < 0.02
    straight_samples = int(np.sum(straight_mask))
    if straight_samples > 5:
        v_mismatch = float(np.mean(vL[straight_mask] - vR[straight_mask]))
    else:
        v_mismatch = float("nan")

    fb_age = data["fb_age"]; wd = data["wd"]
    fb_age_median = float(np.median(fb_age))
    fb_age_max = int(np.max(fb_age))
    fb_age_stale = int(np.sum(fb_age > 300))
    wd_count = int(np.sum(wd > 0))

    bat_first = float(data["bat"][0]); bat_last = float(data["bat"][-1])
    temp_first = float(data["temp"][0]); temp_last = float(data["temp"][-1])

    if len(t) > 10:
        dth_odom = np.diff(th_unwrapped) / dt
        dth_imu  = np.diff(imu_yaw_unwrapped) / dt
        moving = np.abs(v_mean[:-1]) > 0.05
        if np.sum(moving) > 5:
            yaw_rate_err = dth_odom[moving] - dth_imu[moving]
            yaw_rate_err_rms = float(np.sqrt(np.mean(yaw_rate_err ** 2)))
        else:
            yaw_rate_err_rms = float("nan")
    else:
        yaw_rate_err_rms = float("nan")

    # --- EKF analysis ---
    ekf_results = run_ekf_on_data(data)
    ekf_th_unwrap = np.unwrap(ekf_results["ekf_th"])

    ekf_total_th = float(ekf_th_unwrap[-1] - ekf_th_unwrap[0])
    ekf_dist = float(np.sum(np.hypot(np.diff(ekf_results["ekf_x"]),
                                     np.diff(ekf_results["ekf_y"]))))

    odom_th_ekf_frame = np.unwrap(ekf_results["odom_th"])
    imu_th_ekf_frame  = np.unwrap(ekf_results["imu_th"])

    odom_vs_imu_rms = float(np.sqrt(np.mean(
        (odom_th_ekf_frame - imu_th_ekf_frame) ** 2)))
    ekf_vs_imu_rms = float(np.sqrt(np.mean(
        (ekf_th_unwrap - imu_th_ekf_frame) ** 2)))

    return {
        "duration_s": duration,
        "samples": n,
        "rate_hz": rate,
        "dist_odom_m": dist_odom,
        "dist_ekf_m": ekf_dist,
        "total_th_rad": total_th,
        "total_imu_yaw_rad": total_imu_yaw,
        "total_ekf_th_rad": ekf_total_th,
        "v_mismatch_straight_mps": v_mismatch,
        "fb_age_median_ms": fb_age_median,
        "fb_age_max_ms": fb_age_max,
        "fb_age_stale_count": fb_age_stale,
        "watchdog_events": wd_count,
        "bat_start_v": bat_first,
        "bat_end_v": bat_last,
        "temp_start_c": temp_first,
        "temp_end_c": temp_last,
        "yaw_rate_rms_err_radps": yaw_rate_err_rms,
        "odom_vs_imu_th_rms_rad": odom_vs_imu_rms,
        "ekf_vs_imu_th_rms_rad":  ekf_vs_imu_rms,
        "_ekf_results": ekf_results,
    }


def print_summary(summary: dict, path: Path) -> None:
    print(f"\n=== Summary for {path.name} ===")
    print(f"Duration        : {summary['duration_s']:.2f} s")
    print(f"Samples         : {summary['samples']}  ({summary['rate_hz']:.1f} Hz)")
    print(f"Distance (odom) : {summary['dist_odom_m']:.3f} m")
    print(f"Distance (EKF)  : {summary['dist_ekf_m']:.3f} m")
    print()
    print(f"Total th  (odom): {summary['total_th_rad']:+.4f} rad "
          f"({math.degrees(summary['total_th_rad']):+.2f} deg)")
    print(f"Total th  (IMU) : {summary['total_imu_yaw_rad']:+.4f} rad "
          f"({math.degrees(summary['total_imu_yaw_rad']):+.2f} deg)")
    print(f"Total th  (EKF) : {summary['total_ekf_th_rad']:+.4f} rad "
          f"({math.degrees(summary['total_ekf_th_rad']):+.2f} deg)")
    print()
    print(f"RMS(th_odom - th_imu): "
          f"{math.degrees(summary['odom_vs_imu_th_rms_rad']):.3f} deg")
    print(f"RMS(th_ekf  - th_imu): "
          f"{math.degrees(summary['ekf_vs_imu_th_rms_rad']):.3f} deg")
    print()
    print(f"vL - vR (straight): {summary['v_mismatch_straight_mps']:+.4f} m/s")
    print(f"fb_age median/max : {summary['fb_age_median_ms']:.1f} / "
          f"{summary['fb_age_max_ms']} ms  "
          f"(stale>300ms: {summary['fb_age_stale_count']})")
    print(f"Watchdog events   : {summary['watchdog_events']}")
    print(f"Battery           : {summary['bat_start_v']:.2f} -> "
          f"{summary['bat_end_v']:.2f} V")
    print(f"Temperature       : {summary['temp_start_c']:.1f} -> "
          f"{summary['temp_end_c']:.1f} C")
    print(f"Yaw-rate RMS err  : {summary['yaw_rate_rms_err_radps']:.4f} rad/s "
          f"(odom vs IMU, while moving)")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_all(data, summary, path_out: Path) -> None:
    if not HAVE_MPL:
        print("matplotlib not available, skipping plots.")
        return

    ekf = summary["_ekf_results"]
    t = ekf["t"]

    th_unwrapped  = np.unwrap(data["th"])
    imu_unwrapped = np.unwrap(np.deg2rad(data["imu_yaw"]))
    ekf_unwrapped = np.unwrap(ekf["ekf_th"])

    fig, axes = plt.subplots(5, 1, figsize=(10, 15), sharex=True)

    # 1. Trajectory
    axes[0].plot(data["x"], data["y"], "-b", label="raw odom", alpha=0.6)
    axes[0].plot(ekf["ekf_x"], ekf["ekf_y"], "-r", label="EKF", linewidth=2)
    axes[0].plot(data["x"][0], data["y"][0], "go", label="start")
    axes[0].plot(data["x"][-1], data["y"][-1], "ro", label="end")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].set_title("Trajectory: raw odom vs EKF")
    axes[0].legend()
    axes[0].grid(True)

    # 2. Wheel velocities
    axes[1].plot(t, data["vL"], label="vL")
    axes[1].plot(t, data["vR"], label="vR")
    axes[1].set_ylabel("wheel v [m/s]")
    axes[1].set_title("Wheel linear velocities")
    axes[1].legend()
    axes[1].grid(True)

    # 3. Heading: odom vs IMU vs EKF
    axes[2].plot(t, np.rad2deg(th_unwrapped), label="odom th", alpha=0.7)
    axes[2].plot(t, np.rad2deg(imu_unwrapped), label="imu_yaw", alpha=0.7)
    axes[2].plot(t, np.rad2deg(ekf_unwrapped), label="EKF th", linewidth=2)
    axes[2].set_ylabel("heading [deg]")
    axes[2].set_title("Heading: odom vs IMU vs EKF")
    axes[2].legend()
    axes[2].grid(True)

    # 4. Heading error vs IMU
    odom_err = np.rad2deg(th_unwrapped - imu_unwrapped)
    ekf_err  = np.rad2deg(ekf_unwrapped - imu_unwrapped)
    axes[3].plot(t, odom_err, label="odom - imu", alpha=0.7)
    axes[3].plot(t, ekf_err,  label="EKF  - imu", linewidth=2)
    axes[3].axhline(0, color="k", linewidth=0.5)
    axes[3].set_ylabel("heading error [deg]")
    axes[3].set_title("Heading error vs IMU (lower = better)")
    axes[3].legend()
    axes[3].grid(True)

    # 5. Feedback health
    ax5 = axes[4]
    ax5.plot(t, data["fb_age"], label="fb_age [ms]")
    ax5.set_ylabel("fb_age [ms]")
    ax5.set_xlabel("time [s]")
    ax5.set_title("Hoverboard feedback health")
    ax5.grid(True)
    ax5b = ax5.twinx()
    ax5b.plot(t, data["wd"], "r--", label="wd")
    ax5b.set_ylabel("wd (0/1)")
    ax5b.set_ylim(-0.1, 1.1)

    plt.tight_layout()
    fig.savefig(path_out, dpi=120)
    print(f"Plot saved to {path_out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="input CSV from jidenna_logger.py")
    ap.add_argument("--plot", default=None, help="output PNG path (optional)")
    args = ap.parse_args()

    path = Path(args.csv)
    if not path.exists():
        print(f"ERROR: {path} not found")
        return 1

    data = load_csv(path)
    summary = analyze(data)
    if "error" in summary:
        print(f"ERROR: {summary['error']}")
        return 1

    print_summary(summary, path)

    if args.plot:
        plot_all(data, summary, Path(args.plot))

    return 0


if __name__ == "__main__":
    sys.exit(main())