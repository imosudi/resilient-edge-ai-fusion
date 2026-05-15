import serial
import time

PORT = "/dev/ttyACM0"
BAUDRATE = 115200

try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)

    print("LiDAR connected successfully")

    while True:

        line = ser.readline().decode(errors="ignore").strip()

        if line:
            timestamp = time.time()

            print(f"[{timestamp}] {line}")

except serial.SerialException as e:
    print(f"Serial connection error: {e}")

except KeyboardInterrupt:
    print("\nLiDAR capture stopped")

finally:
    if 'ser' in locals():
        ser.close()