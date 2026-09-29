import serial
import time

# ----------------------------
# Configuration
# ----------------------------
PORT = "/dev/ttyUSB1"
BAUDRATE = 115200
PACKET_HEADER = 0xAA  # X3 start byte
DISTANCE_SCALE = 0.001  # mm to meters
POINTS_PER_PACKET = 12  # X3 sends 12 measurements per packet

# ----------------------------
# Helper function to parse a single packet (X3 protocol)
# ----------------------------
def parse_packet(packet_bytes):
    # X3 packet: 0xAA 0x55 CT LSN FSA(2) LSA(2) CS(2) [Data(3*LSN)]
    if len(packet_bytes) < 10:
        return None, []
    
    lsn = packet_bytes[3]  # Number of samples
    
    fsa_raw = packet_bytes[4] | (packet_bytes[5] << 8)
    lsa_raw = packet_bytes[6] | (packet_bytes[7] << 8)
    
    start_angle = (fsa_raw >> 1) / 64.0
    end_angle = (lsa_raw >> 1) / 64.0
    
    angle_diff = end_angle - start_angle
    if angle_diff < 0:
        angle_diff += 360
    
    angle_step = angle_diff / (lsn - 1) if lsn > 1 else 0
    
    points = []
    data_start = 10
    
    for i in range(lsn):
        idx = data_start + i * 3
        if idx + 2 < len(packet_bytes):
            dist_raw = packet_bytes[idx] | (packet_bytes[idx+1] << 8)
            quality = packet_bytes[idx+2]
            distance = dist_raw * 0.25 * DISTANCE_SCALE
            angle = (start_angle + i * angle_step) % 360
            points.append((angle, distance, quality))
    
    return start_angle, points

# ----------------------------
# Generator for streaming LiDAR points
# ----------------------------
def stream_lidar(serial_port, points_per_packet=POINTS_PER_PACKET):
    while True:
        byte = serial_port.read(1)
        if not byte:
            continue
        
        if byte[0] == 0xAA:
            second_byte = serial_port.read(1)
            if not second_byte or second_byte[0] != 0x55:
                continue
            
            header = serial_port.read(8)
            if len(header) != 8:
                continue
            
            lsn = header[1]
            data_bytes = lsn * 3
            data = serial_port.read(data_bytes)
            if len(data) != data_bytes:
                continue
            
            full_packet = bytearray([0xAA, 0x55]) + header + data
            start_angle, points = parse_packet(full_packet)
            
            if start_angle is not None:
                for angle, distance, quality in points:
                    yield angle, distance, quality

# ----------------------------
# Usage example
# ----------------------------
if __name__ == "__main__":
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    print("Streaming LiDAR points... Press Ctrl+C to stop.")
    try:
        for angle, distance, quality in stream_lidar(ser):
            # Process each point immediately
            print(f"Angle: {angle:.1f}°, Distance: {distance:.3f} m, Quality: {quality}")
            # Optional: add a small sleep if your processing is too fast
            # time.sleep(0.001)
    except KeyboardInterrupt:
        print("Stopping LiDAR...")
        ser.close()