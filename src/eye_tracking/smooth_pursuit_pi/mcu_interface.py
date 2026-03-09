"""
Optional MCU (RP2040) UART interface.
Provides start/stop commands and receives battery/IMU status.
The system works without the MCU - this is a thin integration layer.
"""

import time
import threading
from config import MCUConfig

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


class MCUInterface:
    """
    Thin UART wrapper for communicating with the RP2040 MCU.
    Handles: start/stop commands, battery voltage reading, IMU data (optional).
    """

    def __init__(self, config: MCUConfig):
        self.config = config
        self._serial = None
        self._listener_thread = None
        self._running = False
        self._battery_voltage = 0.0
        self._last_imu_data = None

    @property
    def available(self) -> bool:
        return self.config.enabled and SERIAL_AVAILABLE and self._serial is not None

    def connect(self) -> bool:
        """Attempt to connect to the MCU. Returns True if successful."""
        if not self.config.enabled:
            print("MCU interface disabled in config")
            return False

        if not SERIAL_AVAILABLE:
            print("WARNING: pyserial not installed. MCU interface unavailable.")
            return False

        try:
            self._serial = serial.Serial(
                port=self.config.port,
                baudrate=self.config.baudrate,
                timeout=self.config.timeout,
            )
            self._running = True
            self._listener_thread = threading.Thread(target=self._listen, daemon=True)
            self._listener_thread.start()
            print(f"MCU connected on {self.config.port}")
            return True
        except Exception as e:
            print(f"MCU connection failed: {e}")
            self._serial = None
            return False

    def send_command(self, command: str):
        """Send a command string to the MCU."""
        if not self.available:
            return
        try:
            self._serial.write(f"{command}\n".encode())
        except Exception as e:
            print(f"MCU send error: {e}")

    def start_test(self):
        """Signal the MCU that a test session is starting."""
        self.send_command("START_TEST")

    def stop_test(self):
        """Signal the MCU that a test session has ended."""
        self.send_command("STOP_TEST")

    def get_battery_voltage(self) -> float:
        """Return the last known battery voltage."""
        return self._battery_voltage

    def _listen(self):
        """Background thread: read messages from MCU."""
        while self._running and self._serial:
            try:
                line = self._serial.readline().decode().strip()
                if not line:
                    continue
                self._parse_message(line)
            except Exception:
                pass

    def _parse_message(self, message: str):
        """Parse an incoming message from the MCU."""
        if message.startswith("BATTERY:"):
            try:
                self._battery_voltage = float(message.split(":")[1])
            except (ValueError, IndexError):
                pass
        elif message.startswith("IMU:"):
            self._last_imu_data = message

    def close(self):
        """Disconnect from the MCU."""
        self._running = False
        self.send_command("STOP_TEST")
        if self._serial:
            self._serial.close()
            self._serial = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()
