import serial
import time
import sys

PORT = "/dev/ttyUSB0"
BAUD = 115200

V_MAX = 0.8   # must match firmware V_MAX_REAL
W_MAX = 3.0   # must match firmware W_MAX_REAL

CMD_HZ = 10
CMD_PERIOD = 1.0 / CMD_HZ

# Firmware telemetry: x,y,th,vL,vR,wL,wR,bat,temp,fb_age,wd,or  -> 12 fields
NUM_FIELDS = 13


def send(ser, v, w):
    ser.write(f"{v},{w}\n".encode())


def drain_telemetry(ser, log_lines):
    """Read all currently available lines without blocking long."""
    while ser.in_waiting:
        try:
            raw = ser.readline()
        except serial.SerialException:
            return
        if not raw:
            break
        line = raw.decode(errors='ignore').strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) != NUM_FIELDS:
            continue  # header or partial
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            continue
        log_lines.append((time.time(), vals))
        (x, y, th, vL, vR, wL, wR, bat, temp, fb_age, wd, orr, imu_yaw) = vals
        print(f"x={x:+.3f} y={y:+.3f} th={th:+.3f} "
            f"vL={vL:+.3f} vR={vR:+.3f} "
            f"bat={bat:.2f}V temp={temp:.1f}C "
            f"fb_age={int(fb_age):4d}ms wd={int(wd)} or={int(orr)} "
            f"imu_yaw={imu_yaw:+.1f}deg")


def main():
    v = float(input(f"v (m/s, |v|<={V_MAX}): "))
    w = float(input(f"w (rad/s CW+, |w|<={W_MAX}): "))
    duration = float(input("duration (s): "))

    if abs(v) > V_MAX or abs(w) > W_MAX:
        print(f"ERROR: |v|<={V_MAX}, |w|<={W_MAX}. Aborting.")
        sys.exit(1)

    print(f"Opening {PORT} @ {BAUD}...")
    ser = serial.Serial(PORT, BAUD, timeout=4)
    time.sleep(4)                # wait for Nano bootloader
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    # Verify telemetry is arriving (data lines have NUM_FIELDS-1 commas)
    print("Waiting for telemetry...")
    deadline = time.time() + 3.0
    got = False
    while time.time() < deadline:
        if ser.in_waiting:
            line = ser.readline().decode(errors='ignore').strip()
            if line and line.count(',') == NUM_FIELDS - 1:
                print(f"  got: {line}")
                got = True
                break
        time.sleep(0.05)
    if not got:
        print("ERROR: no telemetry from Nano. Check port/cable/firmware.")
        ser.close()
        sys.exit(1)

    print(f"Starting in 3 s: v={v}, w={w}, duration={duration}s")
    time.sleep(3)

    log = []
    t0 = time.time()
    next_cmd_t = t0

    try:
        while time.time() - t0 < duration:
            now = time.time()
            if now >= next_cmd_t:
                send(ser, v, w)
                next_cmd_t += CMD_PERIOD
            drain_telemetry(ser, log)
            time.sleep(0.005)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        try:
            send(ser, 0, 0)
            ser.flush()
            time.sleep(0.3)
            drain_telemetry(ser, log)
        except Exception:
            pass
        ser.close()

    print(f"done. {len(log)} telemetry samples logged.")
    # Optional: save to file
    # with open("run_log.csv", "w") as f:
    #     for t, vals in log:
    #         f.write(f"{t:.3f}," + ",".join(f"{val:.4f}" for val in vals) + "\n")


if __name__ == "__main__":
    main()