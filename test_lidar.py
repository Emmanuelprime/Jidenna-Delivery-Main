"""
test_lidar.py

Standalone LiDAR driver test. Prints scan stats at 2 Hz.
"""

import time
from jidenna.jidenna_lidar import Lidar


PORT = "/dev/ttyUSB1"   # or COMx on Windows


def on_scan(scan):
    n = len(scan)
    rmin = scan.ranges_m.min()
    rmax = scan.ranges_m.max()
    amin = scan.angles_rad.min()
    amax = scan.angles_rad.max()
    print(f"scan #{lidar.scan_count():5d}  "
          f"n={n:4d}  "
          f"range=[{rmin:5.2f}..{rmax:5.2f}]m  "
          f"angle=[{amin:+6.2f}..{amax:+6.2f}]rad")


lidar = Lidar(PORT)
lidar.add_scan_callback(on_scan)
lidar.start()

print("Reading LiDAR. Ctrl-C to stop.")
try:
    while True:
        time.sleep(1.0)
        age = lidar.last_scan_age_s()
        if age > 0.5:
            print(f"  (no scan for {age:.2f}s — check connection)")
except KeyboardInterrupt:
    print("\nstopping")
finally:
    lidar.stop()