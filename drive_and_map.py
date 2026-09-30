"""
test_pose_with_lidar.py

Diagnostic: start bridge + pose + lidar, drive forward 3 s, watch pose.

This is the minimal reproduction of the "pose frozen when LiDAR is running"
bug. It prints a lot so we can see exactly what's happening.
"""

import time

# -----------------------------------------------------------------------------
# Monkey-patch JidennaPose._on_telemetry to print every call
# -----------------------------------------------------------------------------
from jidenna.jidenna_pose import JidennaPose

_original_on_telemetry = JidennaPose._on_telemetry
_cb_count = [0]

def _debug_on_telemetry(self, t):
    _cb_count[0] += 1
    n = _cb_count[0]
    # Print every call for the first 5 calls, then every 20th
    if n <= 5 or n % 20 == 0:
        print(f"  [pose_cb #{n:4d}]  vL={t.vL:+.3f} vR={t.vR:+.3f}  "
              f"started={self._started}  last_t={self._last_t}")
    return _original_on_telemetry(self, t)

JidennaPose._on_telemetry = _debug_on_telemetry


# -----------------------------------------------------------------------------
# Now the rest of the test
# -----------------------------------------------------------------------------
from jidenna.jidenna_bridge import JidennaBridge
from jidenna.jidenna_lidar  import Lidar

NANO_PORT  = "/dev/ttyUSB0"
LIDAR_PORT = "/dev/ttyUSB1"


print("=" * 70)
print("Constructing objects...")
print("=" * 70)
bridge = JidennaBridge(NANO_PORT)
pose   = JidennaPose(bridge)
lidar  = Lidar(LIDAR_PORT)

print(f"  pose._started before start(): {pose._started}")
print(f"  bridge._callbacks before start(): {len(bridge._callbacks)}")

# Optional no-op LiDAR callback so the LiDAR thread runs
lidar.add_scan_callback(lambda s: None)

print()
print("=" * 70)
print("Starting bridge and pose...")
print("=" * 70)
bridge.start()
pose.start()

print(f"  pose._started after start(): {pose._started}")
print(f"  bridge._callbacks after start(): {len(bridge._callbacks)}")
print(f"  callbacks: {[getattr(cb, '__qualname__', repr(cb)) for cb in bridge._callbacks]}")

print()
print("=" * 70)
print("Starting LiDAR...")
print("=" * 70)
lidar.start()

print(f"  pose._started after lidar.start(): {pose._started}")
print(f"  bridge._callbacks after lidar.start(): {len(bridge._callbacks)}")

print()
print("Waiting for connection (bridge + lidar)...")
t0 = time.time()
while time.time() - t0 < 15.0:
    b = bridge.is_connected()
    l = lidar.is_connected()
    if b and l:
        break
    time.sleep(0.1)

print(f"  bridge.is_connected(): {bridge.is_connected()}")
print(f"  lidar.is_connected():  {lidar.is_connected()}")
print(f"  pose._started:         {pose._started}")
print(f"  bridge._callbacks:     {len(bridge._callbacks)}")

p = pose.get_pose()
print(f"  pose.get_pose(): {p}")

print()
print("=" * 70)
print("Driving forward 0.15 m/s for 3 s...")
print("=" * 70)
bridge.set_velocity(0.15, 0.0)

t0 = time.time()
while time.time() - t0 < 3.0:
    p = pose.get_pose()
    print(f"  t={time.time()-t0:4.1f}s  "
          f"pose=({p.x:+.3f}, {p.y:+.3f}, {p.th:+.3f})  "
          f"v={p.v:+.3f}  σxy=({p.P_xx**0.5:.3f},{p.P_yy**0.5:.3f})  "
          f"cb_count={_cb_count[0]}")
    time.sleep(0.3)

print()
print("Stopping...")
bridge.set_velocity(0.0, 0.0)
time.sleep(0.5)

p = pose.get_pose()
print(f"Final pose: ({p.x:+.3f}, {p.y:+.3f}, {p.th:+.3f})")
print(f"Total telemetry callbacks received: {_cb_count[0]}")

lidar.stop()
bridge.stop()