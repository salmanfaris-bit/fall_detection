# Wearable Fall Detection System

A waist-worn fall detector for Raspberry Pi 4. It fuses a 200Hz IMU
(BNO055), a barometric altimeter (BMP280), and a trained RandomForest
classifier into one decision, then gives the wearer a 30-second
cancellable warning before raising a final alarm.

## How a fall is detected

```
BNO055 IMU (200Hz) ──► 2.0s sliding window ──► RandomForest ──► fall probability
                                                                       │
                                          2 consecutive windows ≥0.65 │
                                                                       ▼
                              ┌───────────────── VERIFYING (2.0s) ─────────────────┐
                              │  Post-impact stillness check (accel/gyro variance)  │
                              │  Barometric altitude drop (Kalman-filtered BMP280)  │
                              └──────────────────────────┬──────────────────────────┘
                                                          ▼
                          fuse_decision(): 0.55·ML + 0.25·stillness + 0.20·altitude
                                                          │
                                          fused_score ≥ 0.65 ?
                                          ├── no  → back to MONITORING (false alarm)
                                          └── yes → 30s cancellable countdown → alarm
```

## Why this version is more reliable than a naive implementation

1. **Barometer drift is actively compensated, not just calibrated once.**
   A 1m fall changes pressure by only ~0.12 hPa — smaller than normal
   weather drift over a few hours. `sensors.py` uses two filters:
   - `KalmanFilter1D` removes sample-to-sample sensor noise.
   - `DriftingBaseline` slowly tracks the "resting" reference pressure
     (minutes-long time constant), and is **frozen** the instant a fall
     candidate starts verifying — so a real fall can never be averaged
     away, but weather drift never accumulates into a false trigger either.

2. **Fusion inputs are clamped and bounded.** `fuse_decision()` in
   `detection.py` clamps `ml_prob` and `altitude_score` to `[0, 1]` before
   combining them, so a corrupted or out-of-range sensor reading can never
   produce an unbounded, unpredictable fused score.

3. **Glitch-resistant triggering.** The ML model must stay above threshold
   for `ML_CONSECUTIVE_TRIGGERS_REQUIRED` (2) consecutive 0.5s windows
   before verification starts — filters out single-window sensor noise
   without adding meaningful latency to a real fall.

4. **Input validation.** `validate_window()` rejects windows containing
   NaN/Inf or physically impossible readings (beyond the sensor's ±16g /
   ±2000°/s range) before they reach the model, instead of letting a
   corrupted I2C frame produce a silently wrong prediction.

5. **Event logging.** Every confirmed fall, suppressed false alarm, and
   user cancellation is appended to `events.csv` for post-incident review.

6. **Honest mock data.** Hardware drivers always try real I2C first and
   fall back to simulation only on failure — mock data is always tagged
   and never silently saved or used as if it were real.

## Files

| File | Purpose |
|---|---|
| `main.py` | Production entry point — the live state machine |
| `sensors.py` | IMU + barometer drivers, mocks, Kalman filter, drift-resistant baseline |
| `detection.py` | Feature extraction, stillness/altitude checks, fusion decision |
| `alerts.py` | Buzzer + cancel-button GPIO controller |
| `fall_rf.joblib` | Frozen RandomForest model (18 features — do not modify/retrain in place) |
| `demo_simulation.py` | End-to-end demo with mock sensors, no hardware required |
| `calibrate_and_collect.py` | CLI: measure barometer noise, or record a labeled trial |
| `tests.py` | Automated test suite (27 checks) |

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install numpy pandas joblib smbus2   # smbus2 only needed on real hardware
# RPi.GPIO only on an actual Raspberry Pi:
#   pip install RPi.GPIO
```

## Running

```bash
python3 tests.py               # run the full test suite
python3 demo_simulation.py     # end-to-end demo, no hardware needed
python3 main.py                # live pipeline (auto-falls back to mocks off-Pi)
python3 calibrate_and_collect.py noise    # measure real BMP280 noise floor
python3 calibrate_and_collect.py collect  # record a labeled fall/ADL trial
```

## Constraints (do not modify)

- `fall_rf.joblib` — trained on the exact 18 features in `detection.py::extract_features()`.
  Changing feature names/order/count breaks the model without retraining.
- GPIO/buzzer logic in `alerts.py` is safety-critical and left as originally verified.

## Tuning

- `FUSION_WEIGHTS` / `FUSION_THRESHOLD` — in `detection.py`.
- `ML_CONSECUTIVE_TRIGGERS_REQUIRED` — in `detection.py`.
- Kalman filter noise (`r`) — measure with `calibrate_and_collect.py noise` on your
  actual sensor and update the `KalmanFilter1D(...)` call in `main.py`.
- `DriftingBaseline(tau_s=...)` — larger = slower drift tracking, more resistant
  to being fooled by a real fall; smaller = adapts to weather faster.
