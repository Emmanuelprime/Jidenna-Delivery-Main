import time
from pathlib import Path
from jidenna.jidenna_bridge import JidennaBridge
from jidenna.jidenna_logger import CsvLogger

bridge = JidennaBridge("COM19")
logger = CsvLogger(Path("runs/forward_01.csv"), comment="forward 0.3 m/s, 5 s")
bridge.add_telemetry_callback(logger)
bridge.start()

# Wait for connection...
while not bridge.is_connected():
    time.sleep(0.1)

# Drive
bridge.set_velocity(0.3, 0.0)
time.sleep(5)
bridge.set_velocity(0.0, 0.0)
time.sleep(0.5)

bridge.stop()
logger.close()
print(f"Logged {logger.count} samples.")