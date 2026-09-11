"""
alerts.py — Buzzer + cancel-button GPIO interface, with a mock fallback for
non-Pi development. GPIO/buzzer logic is safety-critical and left as-is.
"""

import time
import threading

try:
    import RPi.GPIO as GPIO
    HAS_GPIO = True
except ImportError:
    GPIO = None
    HAS_GPIO = False


class HardwareController:
    def __init__(self, buzzer_pin: int = 18, button_pin: int = 23):
        self.buzzer_pin = buzzer_pin
        self.button_pin = button_pin
        self.is_beeping = False
        self.beep_thread = None
        self._mock_button_pressed = False

        if HAS_GPIO:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.buzzer_pin, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(self.button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            print(f"[HardwareController] Initialized RPi.GPIO (Buzzer: Pin {self.buzzer_pin}, Button: Pin {self.button_pin})")
        else:
            print("[HardwareController] RPi.GPIO not available. Running in Mock/Simulation mode.")

    def _beep_loop(self, interval_s: float):
        while self.is_beeping:
            if HAS_GPIO:
                GPIO.output(self.buzzer_pin, GPIO.HIGH)
                time.sleep(0.1)
                GPIO.output(self.buzzer_pin, GPIO.LOW)
                time.sleep(max(0.0, interval_s - 0.1))
            else:
                time.sleep(interval_s)

    def start_warning_beep(self, interval_s: float = 0.5):
        """Rhythmic warning beep during the 30-second countdown."""
        if not self.is_beeping:
            self.is_beeping = True
            self.beep_thread = threading.Thread(target=self._beep_loop, args=(interval_s,), daemon=True)
            self.beep_thread.start()

    def start_alarm_sound(self):
        """Continuous final alarm after countdown expires."""
        self.stop_sound()
        if HAS_GPIO:
            GPIO.output(self.buzzer_pin, GPIO.HIGH)
        print("[HARDWARE] Final continuous alarm ACTIVE!")

    def stop_sound(self):
        self.is_beeping = False
        if self.beep_thread and self.beep_thread.is_alive():
            self.beep_thread.join(timeout=0.2)
        if HAS_GPIO:
            GPIO.output(self.buzzer_pin, GPIO.LOW)

    def is_button_pressed(self) -> bool:
        if self._mock_button_pressed:
            self._mock_button_pressed = False
            return True
        if HAS_GPIO:
            return GPIO.input(self.button_pin) == GPIO.LOW  # active LOW, pull-up
        return False

    def simulate_button_press(self):
        """Helper for testing in non-GPIO/PC environments."""
        self._mock_button_pressed = True

    def cleanup(self):
        self.stop_sound()
        if HAS_GPIO:
            GPIO.cleanup()
