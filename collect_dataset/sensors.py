"""
sensors.py — All sensor I/O for the fall-detection wearable.

Contains:
  - BNO055Reader        : real IMU driver (I2C, 200Hz, ACCGYRO mode)
  - MockBNO055Reader     : IMU simulator for development without hardware
  - BMP280Reader         : real barometer driver (I2C)
  - MockBMP280Reader     : barometer simulator
  - KalmanFilter1D       : smooths noisy raw pressure samples
  - DriftingBaseline     : slow-adapting reference pressure that absorbs
                            weather-driven drift WITHOUT absorbing a real fall

Why the two-filter barometer design matters
---------------------------------------------
A 1m fall changes pressure by only ~0.12 hPa, but the sensor's own noise
floor is ~0.02-0.05 hPa and normal weather drift is ~1-3 hPa over a few
hours. Using the raw sample directly (as earlier versions of this project
did) means:
  1. Noise alone can occasionally look like a "drop" -> false positives.
  2. A reference pressure calibrated once at startup slowly goes stale as
     the weather changes -> false positives AND false negatives, growing
     the longer the device has been worn.

The fix used here is standard in barometric altimetry:
  - KalmanFilter1D removes sample-to-sample sensor noise (fast timescale).
  - DriftingBaseline tracks the SLOW-moving "resting" pressure with a long
    time constant (minutes), and is only updated while the wearer is in
    normal MONITORING (never during a fall verification window), so a real
    fall can never be absorbed into the baseline.
"""

import time
import numpy as np

try:
    import smbus2
except ImportError:
    smbus2 = None


# =====================================================================
# Filtering primitives
# =====================================================================

class KalmanFilter1D:
    """Scalar Kalman filter for smoothing a single noisy measurement stream.

    Parameters
    ----------
    initial_value : float
        Starting estimate (e.g. the first raw pressure reading).
    r : float
        Measurement noise variance (hPa^2). Measure this empirically with
        `calibrate_and_collect.py noise` on a stationary sensor; default is
        a realistic BMP280 noise-floor estimate (std ~0.03 hPa).
    q : float
        Process noise variance — how much we expect the true value to move
        between samples. Small for a mostly-still-at-rest wearable.
    """

    def __init__(self, initial_value: float, r: float = 0.03 ** 2, q: float = 1e-5):
        self.x = float(initial_value)
        self.p = 1.0
        self.r = r
        self.q = q

    def update(self, measurement: float) -> float:
        if measurement is None or not np.isfinite(measurement):
            return self.x  # ignore corrupt reads, hold last good estimate
        # Predict
        self.p += self.q
        # Update
        k = self.p / (self.p + self.r)
        self.x += k * (measurement - self.x)
        self.p *= (1.0 - k)
        return self.x


class DriftingBaseline:
    """Very slow exponential tracker used as the altitude "zero point".

    Call `.update()` continuously while the wearer is in normal monitoring.
    Do NOT call it during fall verification / alarm states — that pause is
    what stops a real fall from ever averaging itself back into "normal".
    """

    def __init__(self, initial_value: float, tau_s: float = 300.0, sample_dt_s: float = 0.05):
        self.value = float(initial_value)
        self.alpha = sample_dt_s / (tau_s + sample_dt_s)

    def update(self, measurement: float) -> float:
        if measurement is None or not np.isfinite(measurement):
            return self.value
        self.value += self.alpha * (measurement - self.value)
        return self.value


def altitude_from_pressure(pressure_hpa: float, reference_hpa: float) -> float:
    """Barometric formula: altitude (m) relative to a local reference pressure."""
    return 44330.0 * (1.0 - (pressure_hpa / reference_hpa) ** (1.0 / 5.255))


# =====================================================================
# BNO055 IMU (accelerometer + gyroscope)
# =====================================================================

_REG_PAGE_ID = 0x07
_REG_OPR_MODE = 0x3D
_REG_PWR_MODE = 0x3E
_REG_UNIT_SEL = 0x3B
_REG_ACC_DATA_X_LSB = 0x08   # page 0
_REG_GYR_DATA_X_LSB = 0x14   # page 0
_REG_ACC_CONFIG = 0x08       # page 1
_REG_GYR_CONFIG_0 = 0x0A     # page 1

_OPR_MODE_CONFIG = 0x00
_OPR_MODE_ACCGYRO = 0x05     # accel + gyro only, no sensor fusion
_PWR_MODE_NORMAL = 0x00
_ACC_RANGE_16G = 0b11
_ACC_BANDWIDTH_250HZ = 0b101
_GYR_RANGE_2000DPS = 0b000
_GYR_BANDWIDTH_523HZ = 0b000
_UNIT_SEL_MG_DPS = 0b00000001

_ACCEL_LSB_TO_MG = 1.0
_GYRO_LSB_TO_DPS = 1.0 / 16.0


class BNO055Reader:
    """Real BNO055 IMU over I2C. Matches SisFall's sensor setup: +-16g accel,
    +-2000dps gyro, ACCGYRO mode (no onboard fusion)."""

    def __init__(self, bus: int = 1, address: int = 0x28):
        if smbus2 is None:
            raise ImportError("smbus2 is required: pip install smbus2 --break-system-packages")
        self.bus = smbus2.SMBus(bus)
        self.addr = address
        # Sample-and-hold state for the implausible-read guard in read_raw().
        self._last_good_accel_g = np.array([0.0, 0.0, -1.0])
        self._last_good_gyro_dps = np.array([0.0, 0.0, 0.0])
        self._glitch_streak = 0
        self._glitch_total = 0
        self._reads_total = 0

    def _write(self, reg, val, retries=3):
        last_err = None
        for _ in range(retries):
            try:
                self.bus.write_byte_data(self.addr, reg, val)
                time.sleep(0.01)
                return
            except OSError as e:
                last_err = e
                time.sleep(0.02)
        raise last_err

    def _read_block(self, reg, length, retries=3):
        last_err = None
        for _ in range(retries):
            try:
                return self.bus.read_i2c_block_data(self.addr, reg, length)
            except OSError as e:
                last_err = e
                time.sleep(0.005)
        raise last_err

    def _set_page(self, page):
        self._write(_REG_PAGE_ID, page)

    def configure(self):
        self._set_page(0)
        self._write(_REG_OPR_MODE, _OPR_MODE_CONFIG)
        time.sleep(0.025)
        self._write(_REG_PWR_MODE, _PWR_MODE_NORMAL)
        self._write(_REG_UNIT_SEL, _UNIT_SEL_MG_DPS)
        self._set_page(1)
        self._write(_REG_ACC_CONFIG, (_ACC_BANDWIDTH_250HZ << 2) | _ACC_RANGE_16G)
        self._write(_REG_GYR_CONFIG_0, (_GYR_BANDWIDTH_523HZ << 3) | _GYR_RANGE_2000DPS)
        self._set_page(0)
        self._write(_REG_OPR_MODE, _OPR_MODE_ACCGYRO)
        time.sleep(0.02)

    # Some BNO055 (esp. clone) boards occasionally return implausible values
    # on specific channels for BURSTS of several consecutive samples (not
    # single isolated glitches) when polled continuously at 200Hz — a single
    # immediate retry lands back inside the same bad burst and doesn't help.
    # Rather than propagate garbage into the fall-detection window (which
    # forces validate_window() to discard the whole 2s window every time),
    # we hold the last known-good value PER AXIS through a bad burst, the
    # same sample-and-hold pattern KalmanFilter1D already uses for NaN
    # barometer reads. A short settle delay between the accel and gyro
    # block reads also reduces the chance of catching the chip mid-update.
    _RAW_ACCEL_SANITY_G = 16.0     # sensor is configured for +-16g full scale
    _RAW_GYRO_SANITY_DPS = 2000.0  # sensor is configured for +-2000dps full scale
    _INTER_BLOCK_SETTLE_S = 0.0003

    def read_raw(self):
        acc = self._read_block(_REG_ACC_DATA_X_LSB, 6)
        time.sleep(self._INTER_BLOCK_SETTLE_S)
        gyr = self._read_block(_REG_GYR_DATA_X_LSB, 6)

        def to_signed16(lsb, msb):
            val = (msb << 8) | lsb
            return val - 65536 if val > 32767 else val

        ax = to_signed16(acc[0], acc[1])
        ay = to_signed16(acc[2], acc[3])
        az = to_signed16(acc[4], acc[5])
        gx = to_signed16(gyr[0], gyr[1])
        gy = to_signed16(gyr[2], gyr[3])
        gz = to_signed16(gyr[4], gyr[5])

        accel_g = np.array([ax, ay, az]) * _ACCEL_LSB_TO_MG / 1000.0
        gyro_dps = np.array([gx, gy, gz]) * _GYRO_LSB_TO_DPS

        self._reads_total += 1
        accel_bad = np.abs(accel_g) > self._RAW_ACCEL_SANITY_G
        gyro_bad = np.abs(gyro_dps) > self._RAW_GYRO_SANITY_DPS

        if np.any(accel_bad) or np.any(gyro_bad):
            self._glitch_streak += 1
            self._glitch_total += 1
            # Per-axis hold: only replace the specific implausible axis with
            # its last known-good value, leave the clean axes as freshly
            # read (accel_x/y and gyro_z have been observed clean even
            # during a bad burst — no need to throw that real data away).
            accel_g = np.where(accel_bad, self._last_good_accel_g, accel_g)
            gyro_dps = np.where(gyro_bad, self._last_good_gyro_dps, gyro_dps)
        else:
            self._glitch_streak = 0

        self._last_good_accel_g = accel_g
        self._last_good_gyro_dps = gyro_dps

        return accel_g, gyro_dps

    def glitch_stats(self):
        """Diagnostic: fraction of reads that needed sample-and-hold, and
        the longest consecutive bad-read streak seen so far."""
        rate = self._glitch_total / self._reads_total if self._reads_total else 0.0
        return {"reads_total": self._reads_total, "glitches_total": self._glitch_total,
                "glitch_rate": rate, "current_streak": self._glitch_streak}


class MockBNO055Reader:
    """IMU simulator for development without hardware.

    Rest state: gravity on Y axis (accel_y_g ~= -1.0), matching how the
    real sensor reads when worn upright at the waist. Periodically injects
    a simulated fall impact (counter 5000-5200) for demo/testing.
    """

    def __init__(self):
        self.simulated_fall_counter = 0
        self._counter_obj = None

    def configure(self):
        print("[MockBNO055Reader] Simulation mode active.")

    def set_counter(self, counter_obj):
        """Share a mutable [n] counter so IMU + barometer mocks stay in sync."""
        self._counter_obj = counter_obj

    def read_raw(self):
        self.simulated_fall_counter += 1
        if self._counter_obj is not None:
            self._counter_obj[0] = self.simulated_fall_counter
        counter = self.simulated_fall_counter

        ax = np.random.normal(0.0, 0.05)
        ay = np.random.normal(-1.0, 0.05)
        az = np.random.normal(0.0, 0.05)
        gx = np.random.normal(0.0, 2.0)
        gy = np.random.normal(0.0, 2.0)
        gz = np.random.normal(0.0, 2.0)

        if 5000 < counter < 5200:
            ax = np.random.normal(0.8, 0.3)
            ay = np.random.normal(0.3, 0.4)   # largest deviation from rest (-1.0 -> ~0.3)
            az = np.random.normal(0.5, 0.3)
            gx = np.random.normal(200.0, 30.0)

        return np.array([ax, ay, az]), np.array([gx, gy, gz])


# =====================================================================
# BMP280 barometer
# =====================================================================

_REG_CHIP_ID = 0xD0
_REG_SOFT_RESET = 0xE0
_REG_CTRL_MEAS = 0xF4
_REG_CONFIG = 0xF5
_REG_PRESS_MSB = 0xF7
_REG_DIG_T1 = 0x88
_REG_DIG_P1 = 0x8E

_BMP280_CHIP_ID = 0x58
_CTRL_MEAS_T1_P4_NORMAL = 0b00100111
_STANDBY_0_5_MS = 0x00
_IIR_FILTER_COEFF_4 = 0x08


def _compensate_temperature(adc_T, dig_T1, dig_T2, dig_T3):
    """Bosch reference temperature compensation. Returns (temp_C, t_fine)."""
    var1 = (adc_T / 16384.0 - dig_T1 / 1024.0) * dig_T2
    var2 = ((adc_T / 131072.0 - dig_T1 / 8192.0) ** 2) * dig_T3
    t_fine = var1 + var2
    return t_fine / 5120.0, t_fine


def _compensate_pressure(adc_P, dig_P1, dig_P2, dig_P3, dig_P4, dig_P5,
                          dig_P6, dig_P7, dig_P8, dig_P9, t_fine):
    """Bosch reference pressure compensation (returns Pa). Direct translation
    of the datasheet's compensate_P_double — do not "simplify" the algebra."""
    var1 = (t_fine / 2.0) - 64000.0
    var2 = var1 * var1 * dig_P6 / 32768.0
    var2 = var2 + var1 * dig_P5 * 2.0
    var2 = (var2 / 4.0) + (dig_P4 * 65536.0)
    var1 = (dig_P3 * var1 * var1 / 524288.0 + dig_P2 * var1) / 524288.0
    var1 = (1.0 + var1 / 32768.0) * dig_P1
    if var1 == 0.0:
        return 0.0  # caller treats 0 Pa as an invalid reading
    p = 1048576.0 - adc_P
    p = (p - (var2 / 4096.0)) * 6250.0 / var1
    var1 = dig_P9 * p * p / 2147483648.0
    var2 = p * dig_P8 / 32768.0
    p = p + (var1 + var2 + dig_P7) / 16.0
    return p


class BMP280Reader:
    """Real BMP280 barometer over I2C. Returns RAW compensated (pressure_hpa,
    temp_c) — noise filtering is handled separately by KalmanFilter1D so it
    can be tuned/tested independently of the hardware driver."""

    def __init__(self, bus: int = 1, address: int = 0x76):
        if smbus2 is None:
            raise ImportError("smbus2 is required: pip install smbus2 --break-system-packages")
        self.bus = smbus2.SMBus(bus)
        self.addr = address
        self._dig = {}

    def _write(self, reg, val, retries=3):
        last_err = None
        for _ in range(retries):
            try:
                self.bus.write_byte_data(self.addr, reg, val)
                time.sleep(0.005)
                return
            except OSError as e:
                last_err = e
                time.sleep(0.01)
        raise last_err

    def _block(self, reg, length, retries=3):
        last_err = None
        for _ in range(retries):
            try:
                return self.bus.read_i2c_block_data(self.addr, reg, length)
            except OSError as e:
                last_err = e
                time.sleep(0.005)
        raise last_err

    @staticmethod
    def _u16(block, i=0):
        return block[i] | (block[i + 1] << 8)

    @staticmethod
    def _s16(block, i=0):
        val = block[i] | (block[i + 1] << 8)
        return val - 65536 if val >= 32768 else val

    def configure(self):
        chip_id = self.bus.read_byte_data(self.addr, _REG_CHIP_ID)
        if chip_id != _BMP280_CHIP_ID:
            raise ValueError(f"Expected BMP280 chip ID 0x{_BMP280_CHIP_ID:02X}, got 0x{chip_id:02X}")
        self._write(_REG_SOFT_RESET, 0xB6)
        time.sleep(0.01)

        d = self._dig
        d["T1"] = self._u16(self._block(0x88, 2))
        d["T2"] = self._s16(self._block(0x8A, 2))
        d["T3"] = self._s16(self._block(0x8C, 2))
        d["P1"] = self._u16(self._block(0x8E, 2))
        d["P2"] = self._s16(self._block(0x90, 2))
        d["P3"] = self._s16(self._block(0x92, 2))
        d["P4"] = self._s16(self._block(0x94, 2))
        d["P5"] = self._s16(self._block(0x96, 2))
        d["P6"] = self._s16(self._block(0x98, 2))
        d["P7"] = self._s16(self._block(0x9A, 2))
        d["P8"] = self._s16(self._block(0x9C, 2))
        d["P9"] = self._s16(self._block(0x9E, 2))

        self._write(_REG_CTRL_MEAS, _CTRL_MEAS_T1_P4_NORMAL)
        time.sleep(0.01)
        self._write(_REG_CONFIG, _STANDBY_0_5_MS | _IIR_FILTER_COEFF_4)
        time.sleep(0.01)

    def read_raw(self):
        raw = self._block(_REG_PRESS_MSB, 6)
        p_msb, p_lsb, p_xlsb, t_msb, t_lsb, t_xlsb = raw

        pres_raw = ((p_msb << 12) | (p_lsb << 4) | (p_xlsb >> 4)) & 0xFFFFF
        if pres_raw >= 0x80000:
            pres_raw -= 0x100000
        temp_raw = ((t_msb << 12) | (t_lsb << 4) | (t_xlsb >> 4)) & 0xFFFFF
        if temp_raw >= 0x80000:
            temp_raw -= 0x100000

        d = self._dig
        temp_c, t_fine = _compensate_temperature(temp_raw, d["T1"], d["T2"], d["T3"])
        pressure_pa = _compensate_pressure(
            pres_raw, d["P1"], d["P2"], d["P3"], d["P4"], d["P5"],
            d["P6"], d["P7"], d["P8"], d["P9"], t_fine,
        )
        return pressure_pa / 100.0, temp_c


class MockBMP280Reader:
    """Barometer simulator: stable ~1013 hPa baseline with realistic noise
    (std ~0.03 hPa), plus a synchronized pressure rise during the same
    simulated-fall window used by MockBNO055Reader (counter 5000-5200)."""

    def __init__(self):
        self.pressure_hpa = 1013.0
        self.temperature_c = 25.0
        self._counter_obj = None

    def set_counter(self, counter_obj):
        self._counter_obj = counter_obj

    def configure(self):
        print("[MockBMP280Reader] Simulation mode active.")

    def read_raw(self):
        counter = self._counter_obj[0] if self._counter_obj is not None else 0
        self.pressure_hpa = 1013.0 + np.random.normal(0.0, 0.03)
        self.temperature_c = 25.0 + np.random.normal(0.0, 0.1)
        if 5000 < counter < 5200:
            self.pressure_hpa += 0.135  # ~1m equivalent pressure rise on falling
        return self.pressure_hpa, self.temperature_c
