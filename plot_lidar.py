import serial
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from collections import deque

# ----------------------------
# Configuration
# ----------------------------
PORT = "/dev/ttyUSB1"
BAUDRATE = 115200
DISTANCE_SCALE = 0.001

# Visualization
MAX_RANGE   = 8.0        # meters
FADE_TIME   = 1.5        # seconds a point stays on screen
UPDATE_MS   = 20         # animation interval
NUM_RINGS   = 4
BG_COLOR    = "#04120a"
GRID_COLOR  = "#0d3b22"
SWEEP_COLOR = "#00ff88"
POINT_COLOR = "#39ff14"


# ----------------------------
# Packet parsing
# ----------------------------
def parse_packet(packet_bytes):
    if len(packet_bytes) < 10:
        return None, []
    lsn = packet_bytes[3]
    fsa_raw = packet_bytes[4] | (packet_bytes[5] << 8)
    lsa_raw = packet_bytes[6] | (packet_bytes[7] << 8)
    start_angle = (fsa_raw >> 1) / 64.0
    end_angle   = (lsa_raw >> 1) / 64.0
    angle_diff = end_angle - start_angle
    if angle_diff < 0:
        angle_diff += 360
    angle_step = angle_diff / (lsn - 1) if lsn > 1 else 0

    points = []
    for i in range(lsn):
        idx = 10 + i * 3
        if idx + 2 < len(packet_bytes):
            dist_raw = packet_bytes[idx] | (packet_bytes[idx + 1] << 8)
            quality  = packet_bytes[idx + 2]
            distance = dist_raw * 0.25 * DISTANCE_SCALE
            angle    = (start_angle + i * angle_step) % 360
            points.append((angle, distance, quality))
    return start_angle, points


def stream_lidar(serial_port):
    while True:
        byte = serial_port.read(1)
        if not byte:
            continue
        if byte[0] == 0xAA:
            second = serial_port.read(1)
            if not second or second[0] != 0x55:
                continue
            header = serial_port.read(8)
            if len(header) != 8:
                continue
            lsn = header[1]
            data_bytes = lsn * 3
            data = serial_port.read(data_bytes)
            if len(data) != data_bytes:
                continue
            full = bytearray([0xAA, 0x55]) + header + data
            _, points = parse_packet(full)
            for p in points:
                yield p


# ----------------------------
# Set up the LiDAR display
# ----------------------------
ser = serial.Serial(PORT, BAUDRATE, timeout=1)

fig, ax = plt.subplots(figsize=(9, 9), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)
ax.set_xlim(-MAX_RANGE, MAX_RANGE)
ax.set_ylim(-MAX_RANGE, MAX_RANGE)
ax.set_aspect('equal')
ax.axis('off')

# Range rings + labels
for r in np.linspace(MAX_RANGE / NUM_RINGS, MAX_RANGE, NUM_RINGS):
    ax.add_patch(Circle((0, 0), r, fill=False,
                        edgecolor=GRID_COLOR, linewidth=0.8, alpha=0.7))
    ax.text(r, 0.15, f"{r:.0f} m", color=GRID_COLOR,
            fontsize=8, ha='center', va='bottom')

# Crosshair axes
for ang in range(0, 360, 30):
    rad = math.radians(ang)
    ax.plot([0, MAX_RANGE * math.cos(rad)],
            [0, MAX_RANGE * math.sin(rad)],
            color=GRID_COLOR, linewidth=0.5, alpha=0.5)
    ax.text(MAX_RANGE * 1.02 * math.cos(rad),
            MAX_RANGE * 1.02 * math.sin(rad),
            f"{ang}°", color=GRID_COLOR,
            fontsize=7, ha='center', va='center')

# The rotating sweep line
sweep_line, = ax.plot([0, 0], [0, 0], color=SWEEP_COLOR,
                      linewidth=2, alpha=0.9, zorder=4)

# Fading trail behind the sweep (a translucent wedge)
trail_wedge = plt.matplotlib.patches.Wedge(
    (0, 0), MAX_RANGE, 0, 0,
    color=SWEEP_COLOR, alpha=0.15, zorder=2)
ax.add_patch(trail_wedge)

# Scatter for the points (we update colors to fade old ones)
scatter = ax.scatter([], [], s=6, c=[], cmap='Greens',
                     vmin=0, vmax=1, zorder=3)

# HUD text
hud = ax.text(-MAX_RANGE * 0.98, MAX_RANGE * 0.97, "",
              color=SWEEP_COLOR, fontsize=10,
              family='monospace', va='top')

# Marker at origin (the sensor)
ax.plot(0, 0, marker='o', color=SWEEP_COLOR,
        markersize=8, markeredgecolor='white', zorder=6)

# Rolling point buffer: list of (angle_deg, distance, timestamp)
points_buf = deque(maxlen=8000)


# ----------------------------
# Animation
# ----------------------------
current_sweep_angle = 0.0
last_angle = None
rpm = 0.0

def update(_frame):
    global current_sweep_angle, last_angle, rpm

    # Drain a batch of serial points
    batch = 0
    now = time.time()
    for angle, distance, quality in stream_lidar(ser):
        if 0 < distance <= MAX_RANGE and quality > 0:
            points_buf.append((angle, distance, now))
            current_sweep_angle = angle
        batch += 1
        if batch >= 400:
            break

    # Drop expired points
    cutoff = now - FADE_TIME
    while points_buf and points_buf[0][2] < cutoff:
        points_buf.popleft()

    # --- Compute point positions & alpha (fade with age) ---
    if points_buf:
        angles   = np.array([p[0] for p in points_buf])
        dists    = np.array([p[1] for p in points_buf])
        times    = np.array([p[2] for p in points_buf])
        ages     = (now - times) / FADE_TIME        # 0 fresh -> 1 expired
        alphas   = np.clip(1.0 - ages, 0, 1) ** 1.5  # ease-out fade

        rads = np.radians(angles)
        xs   = dists * np.cos(rads)
        ys   = dists * np.sin(rads)

        # RGBA colors: bright green fading toward black
        colors = np.zeros((len(xs), 4))
        colors[:, 0] = 0.22 * alphas          # R
        colors[:, 1] = 1.00 * alphas          # G
        colors[:, 2] = 0.35 * alphas          # B
        colors[:, 3] = alphas                  # A

        scatter.set_offsets(np.column_stack([xs, ys]))
        scatter.set_color(colors)
        scatter.set_sizes(np.full(len(xs), 7.0))

    # --- Sweep line ---
    rad_sweep = math.radians(current_sweep_angle)
    sweep_line.set_data(
        [0, MAX_RANGE * math.cos(rad_sweep)],
        [0, MAX_RANGE * math.sin(rad_sweep)],
    )

    # --- Sweep trail wedge (30° behind the line) ---
    trail_wedge.set_theta1(current_sweep_angle - 30)
    trail_wedge.set_theta2(current_sweep_angle)

    # --- RPM estimate ---
    if last_angle is not None:
        delta = (current_sweep_angle - last_angle) % 360
        if delta > 0:
            # smooth it
            rpm = 0.9 * rpm + 0.1 * (delta / 360.0) * (60000.0 / UPDATE_MS)
    last_angle = current_sweep_angle

    # --- HUD ---
    hud.set_text(
        f"SCAN  {current_sweep_angle:6.1f}°\n"
        f"RPM   {rpm:6.0f}\n"
        f"PTS   {len(points_buf):5d}"
    )

    return scatter, sweep_line, trail_wedge, hud


ani = animation.FuncAnimation(
    fig, update,
    interval=UPDATE_MS,
    blit=False,
    cache_frame_data=False,
)

plt.tight_layout()
plt.show()
ser.close()