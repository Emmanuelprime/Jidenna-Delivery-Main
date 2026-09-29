"""
analyze_run.py

Analyze one Jidenna telemetry CSV and save a plot (including EKF comparison).

Usage:
    python analyze_run.py runs/straight_01.csv
    python analyze_run.py runs/straight_01.csv --plot runs/straight_01.png
    python analyze_run.py runs/straight_01.csv --plot out.png --no-print
"""

import argparse
import sys
from pathlib import Path

from jidenna.jidenna_analyze import load_csv, analyze, print_summary, plot_all


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze one Jidenna telemetry run.")
    ap.add_argument("csv", help="input CSV from jidenna_logger.py")
    ap.add_argument("--plot", default=None,
                    help="output PNG path (omit to skip plotting)")
    ap.add_argument("--no-print", action="store_true",
                    help="skip printing the summary")
    args = ap.parse_args()

    path = Path(args.csv)
    if not path.exists():
        print(f"ERROR: {path} not found")
        return 1

    data = load_csv(path)
    summary = analyze(data)                  # analyze() now runs the EKF internally
    if "error" in summary:
        print(f"ERROR: {summary['error']}")
        return 1

    if not args.no_print:
        print_summary(summary, path)         # includes EKF lines

    if args.plot:
        plot_all(data, summary, Path(args.plot))   # <-- pass summary too

    return 0


if __name__ == "__main__":
    sys.exit(main())