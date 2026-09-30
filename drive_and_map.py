"""
test_pose_with_lidar.py

Minimal: start bridge + pose + lidar, drive forward 1 m, watch pose.
"""

import time
from jidenna.jidenna_bridge import JidennaBridge
from jidenna.jidenna_pose   import JidennaPose
from jidenna.jidenna_lidar  import Lidar

NANO_PORT = "/dev/ttyUSB0"
LIDAR_PORT = "/dev/ttyUSB1"

bridge = JidennaBridge(NANO_PORT)
pose = JidennaPose(bridge)
lidar = Lidar(LIDAR_PORT)

# Optional: wire a no-op callback so the LiDAR thread runs
lidar.add_scan_callback(lambda s: None)

bridge.start()
pose.start()
lidar.start()

# Wait for connection
print("Waiting for connection...")
t0 = time.time()
while time.time() - t0 < 15:
    if bridge.is_connected() and lidar.is_connected():
        break
    time.sleep(0.1)
print(f"  bridge: {bridge.is_connected()}  lidar: {lidar.is_connected()}")

# Record pose before
p0 = pose.get_pose()
print(f"Initial pose: ({p0.x:+.3f}, {p0.y:+.3f}, {p0.th:+.3f})")

print("Driving forward 0.15 m/s for 3 s...")
bridge.set_velocity(0.15, 0.0)
t0 = time.time()
while time.time() - t0 < 3.0:
    p = pose.get_pose()
    print(f"  t={time.time()-t0:4.1f}s  pose=({p.x:+.3f}, {p.y:+.3f}, {p.th:+.3f})  "
          f"v={p.v:+.3f}  σxy=({p.P_xx**0.5:.3f},{p.P_yy**0.5:.3f})")
    time.sleep(0.3)

print("Stopping...")
bridge.set_velocity(0.0, 0.0)
time.sleep(0.5)

p = pose.get_pose()
print(f"Final pose: ({p.x:+.3f}, {p.y:+.3f}, {p.th:+.3f})")

lidar.stop()
bridge.stop()