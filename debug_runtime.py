"""
debug_pose_in_runtime.py

Minimal test: does the EKF update when running inside JidennaRuntime?
"""

import time
from jidenna.jidenna_runtime import JidennaRuntime, RuntimeConfig


def main():
    cfg = RuntimeConfig(
        nano_port="/dev/ttyUSB0",
        lidar_port="/dev/ttyUSB1",
    )
    rt = JidennaRuntime(cfg)
    rt.start()

    print("Started. Watching pose for 5 seconds...")
    t0 = time.time()
    while time.time() - t0 < 5.0:
        p = rt.pose.get_pose()
        print(f"  t={time.time()-t0:5.2f}s  "
              f"pose=({p.x:+.3f}, {p.y:+.3f}, {p.th:+.3f})  "
              f"v={p.v:+.3f} w={p.w:+.3f}  "
              f"σxy=({p.P_xx**0.5:.3f},{p.P_yy**0.5:.3f})")
        time.sleep(0.5)

    # Now check the bridge's telemetry directly
    print("\nBridge telemetry:")
    for i in range(5):
        t = rt.bridge.get_telemetry()
        print(f"  x={t.x:+.3f} y={t.y:+.3f} vL={t.vL:+.3f} vR={t.vR:+.3f}")
        time.sleep(0.5)

    rt.stop()


if __name__ == "__main__":
    main()