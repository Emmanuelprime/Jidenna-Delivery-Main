"""
test_joystick.py

Minimal joystick → robot test. No GUI, no costmap — just drive.

Left stick Y = forward/back, left stick X = turn (arcade).
Starts disarmed: press Enter to arm, Ctrl-C to quit.

Usage:
    python -m jidenna.test_joystick --serial /dev/ttyUSB0 --udp-port 4210
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

from jidenna.jidenna_bridge import JidennaBridge
from jidenna.controllers.joy_receiver import JoyReceiver, sticks_to_vw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", required=True)
    ap.add_argument("--udp-port", type=int, default=4210)
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--v-max", type=float, default=0.25)
    ap.add_argument("--w-max", type=float, default=0.8)
    ap.add_argument("--timeout", type=float, default=0.5)
    ap.add_argument("--scheme", default="arcade",
                    choices=["arcade", "tank", "left_only", "right_only"])
    args = ap.parse_args()

    # --- bridge ---
    bridge = JidennaBridge(args.serial)
    print(f"opening {args.serial}...")
    bridge.start()
    t0 = time.time()
    while not bridge.is_connected() and time.time() - t0 < 10.0:
        time.sleep(0.1)
    if not bridge.is_connected():
        print("no telemetry from Nano")
        bridge.stop()
        return 1
    print("bridge ok")

    # --- joystick ---
    rx = JoyReceiver(udp_port=args.udp_port, bind_addr=args.bind)
    rx.start()
    time.sleep(0.2)
    if rx.bind_failed():
        print(f"udp bind failed: {rx.bind_error()}")
        bridge.stop()
        return 1
    print(f"listening on UDP {args.bind}:{args.udp_port}")

    # --- arm/disarm via Enter ---
    armed = {"v": False}
    stop  = {"v": False}

    def key_loop():
        while not stop["v"]:
            line = sys.stdin.readline()
            if not line:
                time.sleep(0.2)
                continue
            armed["v"] = not armed["v"]
            print(f"\n{'ARMED' if armed['v'] else 'DISARMED'}")
            if not armed["v"]:
                try:
                    bridge.set_velocity(0.0, 0.0)
                except Exception:
                    pass

    threading.Thread(target=key_loop, daemon=True).start()

    print("press Enter to ARM. Ctrl-C to quit.")

    # --- main loop: 20 Hz ---
    try:
        while True:
            if not armed["v"]:
                bridge.set_velocity(0.0, 0.0)
            elif rx.last_rx_age_s() > args.timeout:
                bridge.set_velocity(0.0, 0.0)
            else:
                st = rx.get_state()
                v, w = sticks_to_vw(st, args.scheme,
                                    args.v_max, args.w_max,
                                    invert_lx=True)
                # Nano expects CW+ w; sticks_to_vw returns CCW+.
                bridge.set_velocity(v, -w)

            st = rx.get_state()
            age_ms = rx.last_rx_age_s() * 1000.0
            print(f"\r[{'ARM' if armed['v'] else 'dis'}] "
                  f"lx={st.lx:+.2f} ly={st.ly:+.2f} "
                  f"rx={st.rx:+.2f} ry={st.ry:+.2f}  "
                  f"pkt={rx.packet_count():6d} age={age_ms:6.0f}ms",
                  end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()
    finally:
        stop["v"] = True
        try:
            bridge.set_velocity(0.0, 0.0)
        except Exception:
            pass
        rx.stop()
        time.sleep(0.1)
        bridge.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())