import time
from pathlib import Path

from jidenna.jidenna_bridge import JidennaBridge
from jidenna.jidenna_logger import CsvLogger
from jidenna.jidenna_pose   import JidennaPose


PORT = "COM19"
OUT  = Path("runs/ekf_demo_01.csv")

bridge = JidennaBridge(PORT)
logger = CsvLogger(OUT, comment="forward + slight turn, with live EKF")
pose   = JidennaPose(bridge)

# Wire callbacks
bridge.add_telemetry_callback(logger)   # raw telemetry → CSV
pose.add_pose_callback(logger.on_pose)  # EKF pose   → CSV
pose.add_pose_callback(lambda p: print(p))  # live print

# Start
bridge.start()
pose.start()

# Wait for connection
t0 = time.time()
while not bridge.is_connected() and time.time() - t0 < 10.0:
    time.sleep(0.1)
if not bridge.is_connected():
    print("No telemetry. Abort.")
    bridge.stop(); logger.close(); raise SystemExit(1)

print("Connected. Starting in 2 s...")
time.sleep(2)

print("Forward 0.3 m/s, slight turn w=0.2, for 5 s")
bridge.set_velocity(0.3, 0.2)
time.sleep(5)

print("Stop")
bridge.set_velocity(0.0, 0.0)
time.sleep(0.5)

bridge.stop()
logger.close()
print(f"Done. {logger.count} samples to {OUT}")