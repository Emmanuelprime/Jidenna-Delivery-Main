"""
Minimal demo: connect, drive forward for 3 s, stop.
Run with wheels off the ground for the first test.
"""
import time
from jidenna.jidenna_bridge import JidennaBridge

PORT = "COM19"   # or "/dev/ttyUSB0" on Linux

def on_telem(t):
    print(t)   # the Telemetry.__str__ is nicely formatted

def main():
    bridge = JidennaBridge(PORT)
    bridge.add_telemetry_callback(on_telem)

    print(f"Opening {PORT}...")
    bridge.start()

    # Wait for telemetry
    t0 = time.time()
    while not bridge.is_connected() and time.time() - t0 < 10:
        time.sleep(0.1)

    if not bridge.is_connected():
        print("ERROR: no telemetry. Check cable/port/Nano.")
        bridge.stop()
        return

    print("Connected. Starting in 3 s...")
    time.sleep(3)

    print("Forward 0.3 m/s for 3 s")
    bridge.set_velocity(0.3, 0.0)
    time.sleep(3)

    print("Stop")
    bridge.set_velocity(0.0, 0.0)
    time.sleep(0.5)

    print("Final telemetry:", bridge.get_telemetry())
    bridge.stop()
    print("done.")

if __name__ == "__main__":
    main()