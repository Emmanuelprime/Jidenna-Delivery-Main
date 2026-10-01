"""
test_joystick.py

Minimal joystick UDP test. No robot — just listen and print.

Usage:
    python -m jidenna.test_joystick --udp-port 4210
"""

from __future__ import annotations

import argparse
import sys
import time

from jidenna.controllers.joy_receiver import JoyReceiver


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--udp-port", type=int, default=4210)
    ap.add_argument("--bind", default="0.0.0.0")
    args = ap.parse_args()

    rx = JoyReceiver(udp_port=args.udp_port, bind_addr=args.bind)
    rx.start()
    time.sleep(0.2)

    if rx.bind_failed():
        print(f"bind failed: {rx.bind_error()}")
        return 1

    print(f"listening on UDP {args.bind}:{args.udp_port}  (Ctrl-C to quit)")
    try:
        while True:
            st = rx.get_state()
            age_ms = rx.last_rx_age_s() * 1000.0
            print(f"\rlx={st.lx:+.2f} ly={st.ly:+.2f} "
                  f"rx={st.rx:+.2f} ry={st.ry:+.2f}  "
                  f"pkt={rx.packet_count():6d} "
                  f"bad={rx.bad_packet_count():4d} "
                  f"age={age_ms:6.0f}ms", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()
    finally:
        rx.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())