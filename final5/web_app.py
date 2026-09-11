"""
web_app.py — Local browser dashboard for the fall-detection pipeline.

Runs the REAL pipeline from main.py (real BNO055 + BMP280 over I2C, the
frozen RandomForest model, real consecutive-trigger / stillness / altitude
checks) in a background thread, and exposes it as a live flow-diagram
dashboard in the browser. It only advances past MONITORING when the actual
sensors produce a fall-like signal — nothing is scripted or auto-injected.

This file imports main.py's own functions (init_sensors, load_model,
calibrate_barometer, buffer_to_df) instead of re-implementing them, so
there is one source of truth for the pipeline logic. main.py itself is
untouched — importing it as a module only defines functions/constants, it
does not run the pipeline (that only happens inside main.py's own
`if __name__ == "__main__":` guard).

The "Cancel Alert" / "Reset Alarm" button lives on the WEBSITE, not on any
GPIO pin. alerts.py's HardwareController already runs in mock mode whenever
real RPi.GPIO isn't available, and mock mode is driven by
HardwareController.simulate_button_press() / is_button_pressed() — exactly
what main.py's live pipeline calls from a real GPIO interrupt. This file
just calls that same mock method from a Flask route instead of a wire, so
nothing in alerts.py had to change. On the Pi, if you HAVE real GPIO.PUD_UP
button wiring, that will still work too — is_button_pressed() checks the
real pin first and the mock flag doesn't interfere with it.

Run (on the Raspberry Pi, with the BNO055 + BMP280 wired up):
    python3 web_app.py
Then open, from any device on the same network:
    http://<pi-ip-address>:5000
or on the Pi itself:
    http://localhost:5000
"""

import collections
import threading
import time

import numpy as np
from flask import Flask, jsonify, render_template_string

import main as pipeline  # reuses main.py's own init_sensors/load_model/etc.
from sensors import (
    MockBNO055Reader, MockBMP280Reader,
    KalmanFilter1D, DriftingBaseline, altitude_from_pressure,
)
from detection import (
    extract_features, check_stillness, check_altitude_drop, fuse_decision,
    validate_window, FUSION_THRESHOLD, ML_CONSECUTIVE_TRIGGERS_REQUIRED,
)
from alerts import HardwareController

app = Flask(__name__)

_lock = threading.Lock()
STATE = {
    "phase": "STARTING",   # STARTING | MONITORING | CANDIDATE | VERIFYING | FUSION | WARNING | CANCELLED | ALARM
    "ml_prob": 0.0,
    "is_still": None,
    "altitude_score": None,
    "fused_score": None,
    "countdown_remaining": None,
    "using_real_hardware": None,
    "log": [],
}
_current_hw = {"hw": None}


def _set_state(**kwargs):
    with _lock:
        STATE.update(kwargs)


def _log(msg: str):
    with _lock:
        STATE["log"].insert(0, f"{time.strftime('%H:%M:%S')}  {msg}")
        STATE["log"] = STATE["log"][:14]


def simulation_loop():
    """Runs main.py's actual pipeline logic, driven by real sensors when
    available. Publishes state to STATE for the dashboard instead of
    printing to the console the way main.py's own main() does."""
    _log("Loading model...")
    try:
        model, feature_cols = pipeline.load_model(pipeline.MODEL_PATH)
    except Exception as e:
        _log(f"FATAL: could not load {pipeline.MODEL_PATH}: {e}")
        return
    _log(f"Model loaded ({len(feature_cols)} features)")

    hw = HardwareController(buzzer_pin=18, button_pin=23)
    _current_hw["hw"] = hw

    imu, baro = pipeline.init_sensors()  # real hardware first, mock fallback with a printed notice — see sensors.py
    used_mock = isinstance(imu, MockBNO055Reader) or isinstance(baro, MockBMP280Reader)
    _set_state(using_real_hardware=not used_mock)
    if used_mock:
        _log("Real IMU/barometer not found on I2C \u2014 falling back to MOCK sensors (see terminal)")
    else:
        _log("Connected to real BNO055 + BMP280 over I2C")

    reference_hpa = pipeline.calibrate_barometer(baro)
    kalman = KalmanFilter1D(initial_value=reference_hpa)
    baseline = DriftingBaseline(initial_value=reference_hpa, tau_s=300.0,
                                 sample_dt_s=pipeline.ALT_SAMPLE_RATE_DIVISOR / pipeline.TARGET_HZ)

    buffer = collections.deque(maxlen=pipeline.WINDOW_SAMPLES)
    stillness_buffer = []
    altitude_deque = collections.deque(maxlen=500)

    state = "MONITORING"
    samples_since_eval = 0
    consecutive_triggers = 0
    countdown_start = 0.0
    fall_prob = 0.0
    altitude_baseline_m = 0.0
    verify_tick = 0

    interval = 1.0 / pipeline.TARGET_HZ
    next_t = time.time()

    _set_state(phase="MONITORING", ml_prob=0.0, is_still=None, altitude_score=None,
               fused_score=None, countdown_remaining=None)
    _log("Monitoring started")

    while True:
        try:
            accel_g, gyro_dps = imu.read_raw()
        except Exception as e:
            _log(f"IMU read failed, skipping sample: {e}")
            next_t += interval
            time.sleep(max(0, next_t - time.time()))
            continue

        sample = {
            "accel_x_g": accel_g[0], "accel_y_g": accel_g[1], "accel_z_g": accel_g[2],
            "gyro_x_dps": gyro_dps[0], "gyro_y_dps": gyro_dps[1], "gyro_z_dps": gyro_dps[2],
        }

        # ------------------------------------------------- MONITORING
        if state == "MONITORING":
            buffer.append(sample)
            samples_since_eval += 1

            if samples_since_eval % pipeline.ALT_SAMPLE_RATE_DIVISOR == 0:
                try:
                    p_raw, _ = baro.read_raw()
                    p_filtered = kalman.update(p_raw)
                    reference_hpa = baseline.update(p_filtered)
                    alt_m = altitude_from_pressure(p_filtered, reference_hpa)
                    altitude_deque.append(alt_m)
                except Exception:
                    pass

            if len(buffer) == pipeline.WINDOW_SAMPLES and samples_since_eval >= pipeline.STEP_SAMPLES:
                samples_since_eval = 0
                window_df = pipeline.buffer_to_df(list(buffer), pipeline.TARGET_HZ)

                if not validate_window(window_df):
                    _log("Corrupted sensor window skipped")
                else:
                    feats = extract_features(window_df)
                    x = np.array([[feats[c] for c in feature_cols]])
                    pred = model.predict(x)[0]
                    prob = model.predict_proba(x)[0]
                    fall_prob = float(prob[1] if len(prob) > 1 else prob[0])
                    _set_state(ml_prob=fall_prob)

                    if pred == 1 and fall_prob >= pipeline.FALL_PROB_THRESHOLD:
                        consecutive_triggers += 1
                    else:
                        consecutive_triggers = 0

                    if consecutive_triggers >= ML_CONSECUTIVE_TRIGGERS_REQUIRED:
                        alt_window = list(altitude_deque)[-20:]
                        altitude_baseline_m = float(np.median(alt_window)) if alt_window else 0.0

                        _set_state(phase="CANDIDATE")
                        _log(f"Candidate fall (ML prob {fall_prob:.2f}, {consecutive_triggers} consecutive windows). Verifying...")

                        stillness_buffer = []
                        altitude_deque.clear()
                        altitude_deque.append(altitude_baseline_m)
                        state = "VERIFYING"
                        verify_tick = 0
                        consecutive_triggers = 0
                        _set_state(phase="VERIFYING")

        # ------------------------------------------------- VERIFYING
        elif state == "VERIFYING":
            stillness_buffer.append(sample)
            verify_tick += 1
            if verify_tick % pipeline.ALT_SAMPLE_RATE_DIVISOR == 0:
                try:
                    p_raw, _ = baro.read_raw()
                    p_filtered = kalman.update(p_raw)
                    # baseline frozen during verification, same as main.py
                    alt_m = altitude_from_pressure(p_filtered, reference_hpa)
                    altitude_deque.append(alt_m)
                except Exception:
                    pass

            if len(stillness_buffer) >= pipeline.STILLNESS_SAMPLES:
                stillness_df = pipeline.buffer_to_df(stillness_buffer, pipeline.TARGET_HZ)
                stillness_res = check_stillness(stillness_df, accel_std_threshold=0.15, gyro_std_threshold=15.0)

                alt_samples = list(altitude_deque)
                altitude_after_m = float(np.median(alt_samples[-20:])) if alt_samples else 0.0
                altitude_res = check_altitude_drop(altitude_baseline_m, altitude_after_m,
                                                    drop_threshold_m=pipeline.ALT_DROP_THRESHOLD_M)

                fuse_result = fuse_decision(ml_prob=fall_prob, is_still=stillness_res["is_still"],
                                             altitude_score=altitude_res["altitude_score"])

                _set_state(phase="FUSION", is_still=stillness_res["is_still"],
                           altitude_score=altitude_res["altitude_score"], fused_score=fuse_result["fused_score"])
                _log(f"Fused score {fuse_result['fused_score']:.2f} (threshold {FUSION_THRESHOLD}) "
                     f"\u2192 {'CONFIRMED' if fuse_result['confirm_fall'] else 'SUPPRESSED'}")

                if fuse_result["confirm_fall"]:
                    state = "WARNING_COUNTDOWN"
                    countdown_start = time.time()
                    hw.start_warning_beep(interval_s=0.5)
                    _set_state(phase="WARNING", countdown_remaining=pipeline.WARNING_COUNTDOWN_S)
                    _log("WARNING countdown started \u2014 cancel on the dashboard to stop it")
                else:
                    _set_state(phase="NORMAL")
                    _log("Suppressed as false alarm \u2014 back to monitoring")
                    state = "MONITORING"
                    buffer.clear()
                    altitude_deque.clear()
                    stillness_buffer = []
                    fall_prob = 0.0
                    time.sleep(1.0)
                    _set_state(phase="MONITORING")

        # ------------------------------------------------- WARNING COUNTDOWN
        elif state == "WARNING_COUNTDOWN":
            elapsed = time.time() - countdown_start
            remaining = max(0.0, pipeline.WARNING_COUNTDOWN_S - elapsed)
            _set_state(countdown_remaining=remaining)

            if hw.is_button_pressed():
                hw.stop_sound()
                _set_state(phase="CANCELLED", countdown_remaining=0)
                _log("Cancelled from the web dashboard")
                state = "MONITORING"
                buffer.clear()
                fall_prob = 0.0
                time.sleep(1.5)
                _set_state(phase="MONITORING", countdown_remaining=None)
            elif remaining <= 0:
                hw.start_alarm_sound()
                _set_state(phase="ALARM", countdown_remaining=0)
                _log("FINAL ALARM \u2014 no response within 30s")
                state = "ALARM_ACTIVE"

        # ------------------------------------------------- ALARM ACTIVE
        elif state == "ALARM_ACTIVE":
            if hw.is_button_pressed():
                hw.stop_sound()
                _log("Alarm reset from the web dashboard")
                state = "MONITORING"
                buffer.clear()
                fall_prob = 0.0
                _set_state(phase="MONITORING", countdown_remaining=None)

        next_t += interval
        sleep_time = next_t - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)


@app.route("/api/state")
def api_state():
    with _lock:
        return jsonify(dict(STATE))


@app.route("/api/cancel", methods=["POST"])
def api_cancel():
    """Website-triggered cancel/reset. Calls the SAME mock button method
    main.py's live pipeline would call from a real GPIO interrupt."""
    hw = _current_hw["hw"]
    if hw is not None:
        hw.simulate_button_press()
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "pipeline not started yet"}), 409


PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Wearable Fall Detection \u2014 Live Pipeline</title>
<style>
  :root {
    --blue: #2f6fb3; --blue-bg: #dbe9fb; --blue-border: #9dc2ec;
    --purple-bg: #ece7fb; --purple-border: #cbb9f3;
    --green: #1f8a4c; --green-bg: #dff5e6; --green-border: #97dcb0;
    --orange: #b3760b; --orange-bg: #fdefd2; --orange-border: #f0c876;
    --red: #b3311f; --red-bg: #fbe0dd; --red-border: #f0a99e;
    --ink: #1b2733; --muted: #6b7a89;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
    background: #f5f7fa; color: var(--ink); padding: 24px 16px 60px;
  }
  h1 { text-align: center; color: var(--blue); margin: 0 0 4px; font-size: 1.6rem; }
  .subtitle { text-align: center; color: var(--muted); margin: 0 0 6px; font-size: .9rem; }
  #hwBadge { text-align: center; margin: 0 0 24px; font-size: .8rem; font-weight: 600; }
  .diagram { max-width: 760px; margin: 0 auto; display: flex; flex-direction: column; align-items: center; }
  .node {
    width: 100%; border: 2px solid; border-radius: 10px; padding: 12px 16px;
    margin: 6px 0; background: #fff; transition: all .25s ease; opacity: .55;
  }
  .node b { display: block; font-size: .95rem; }
  .node span { font-size: .8rem; color: var(--muted); }
  .node.active { opacity: 1; box-shadow: 0 0 0 4px rgba(47,111,179,.15); transform: scale(1.02); }
  .blue { background: var(--blue-bg); border-color: var(--blue-border); }
  .purple { background: var(--purple-bg); border-color: var(--purple-border); }
  .green { background: var(--green-bg); border-color: var(--green-border); }
  .orange { background: var(--orange-bg); border-color: var(--orange-border); }
  .red { background: var(--red-bg); border-color: var(--red-border); }
  .diamond { text-align: center; }
  .arrow { color: var(--muted); font-size: 1.1rem; margin: 2px 0; }
  .parallel { display: flex; gap: 12px; width: 100%; }
  .parallel .node { width: 50%; }
  .branch { display: flex; gap: 12px; width: 100%; }
  .branch .node { width: 50%; }
  #cancelBtn {
    display: none; margin-top: 10px; background: #fff; border: 2px solid var(--orange);
    color: var(--orange); font-weight: 700; padding: 10px 22px; border-radius: 8px;
    cursor: pointer; font-size: .95rem;
  }
  #cancelBtn:hover { background: var(--orange-bg); }
  #countdown { font-size: 1.4rem; font-weight: 700; }
  .log-box {
    max-width: 760px; margin: 28px auto 0; background: #fff; border: 1px solid #e1e6ea;
    border-radius: 10px; padding: 12px 16px; font-family: ui-monospace, Menlo, monospace;
    font-size: .78rem; color: var(--muted); max-height: 220px; overflow-y: auto;
  }
  .log-box div.title { color: var(--ink); font-family: -apple-system, sans-serif; font-weight: 600; margin-bottom: 6px; }
</style>
</head>
<body>
  <h1>Wearable Fall Detection \u2014 Live Pipeline</h1>
  <p class="subtitle">Driven by main.py's real pipeline: real BNO055/BMP280 sensors when connected, the trained RandomForest model, and live fusion.</p>
  <p id="hwBadge"></p>

  <div class="diagram">
    <div class="node blue" id="n-imu"><b>BNO055 IMU</b><span>Accelerometer + gyroscope, waist-mounted, 200Hz</span></div>
    <div class="arrow">&#8595;</div>
    <div class="node blue" id="n-window"><b>Windowing + Feature Extraction</b><span>2.0s sliding window &middot; 18 kinematic features</span></div>
    <div class="arrow">&#8595;</div>
    <div class="node blue" id="n-classifier"><b>Random Forest Fall Classifier</b><span id="ml-val">Fall probability: \u2014</span></div>
    <div class="arrow">&#8595;</div>
    <div class="node purple diamond" id="n-candidate"><b>Candidate Fall</b><span>2 consecutive triggers required, flagged for verification</span></div>
    <div class="arrow">&#8595;</div>
    <div class="parallel">
      <div class="node purple" id="n-altitude"><b>BMP280 Barometric Drop</b><span id="alt-val">Altitude score: \u2014</span></div>
      <div class="node purple" id="n-stillness"><b>Post-fall Stillness</b><span id="still-val">Still: \u2014</span></div>
    </div>
    <div class="arrow">&#8595;</div>
    <div class="node blue" id="n-fusion"><b>Fusion / Rules</b><span id="fusion-val">0.55&middot;ML + 0.25&middot;stillness + 0.20&middot;altitude</span></div>
    <div class="arrow">&#8595;</div>
    <div class="node blue diamond" id="n-threshold"><b>Score vs. Threshold (0.65)</b><span id="threshold-val">&mdash;</span></div>
    <div class="arrow">&#8595;</div>
    <div class="branch">
      <div class="node green" id="n-normal"><b>Normal &mdash; no action</b><span>Continue monitoring</span></div>
      <div class="node orange" id="n-warning">
        <b>30-second Warning</b>
        <span>Buzzer counts down &mdash; cancel below</span>
        <div id="countdown"></div>
        <button id="cancelBtn" onclick="cancelAlert()">Cancel Alert</button>
      </div>
    </div>
    <div class="arrow">&#8595;</div>
    <div class="branch">
      <div class="node green" id="n-cancelled"><b>Cancel Alert</b><span>Logged as user-cancelled from dashboard</span></div>
      <div class="node red" id="n-alarm"><b>Final Fall Event</b><span>No response in 30s &mdash; caregiver notified</span></div>
    </div>
  </div>

  <div class="log-box">
    <div class="title">Event log</div>
    <div id="log"></div>
  </div>

<script>
async function cancelAlert() {
  await fetch('/api/cancel', { method: 'POST' });
}

function clearActive() {
  document.querySelectorAll('.node').forEach(n => n.classList.remove('active'));
}

function setText(id, text) {
  document.getElementById(id).textContent = text;
}

async function tick() {
  try {
    const r = await fetch('/api/state');
    const s = await r.json();
    clearActive();

    const phase = s.phase;
    const btn = document.getElementById('cancelBtn');
    if (phase === 'WARNING') {
      btn.style.display = 'inline-block';
      btn.textContent = 'Cancel Alert';
    } else if (phase === 'ALARM') {
      btn.style.display = 'inline-block';
      btn.textContent = 'Reset Alarm';
    } else {
      btn.style.display = 'none';
    }

    if (s.using_real_hardware !== null && s.using_real_hardware !== undefined) {
      const badge = document.getElementById('hwBadge');
      if (s.using_real_hardware) {
        badge.textContent = '\\u25cf Real BNO055 + BMP280 connected over I2C';
        badge.style.color = '#1f8a4c';
      } else {
        badge.textContent = '\\u25cf Real hardware not found \\u2014 running on MOCK sensors (see terminal)';
        badge.style.color = '#b3311f';
      }
    }

    if (s.ml_prob !== null && s.ml_prob !== undefined) {
      setText('ml-val', 'Fall probability: ' + s.ml_prob.toFixed(2));
    }
    if (s.altitude_score !== null && s.altitude_score !== undefined) {
      setText('alt-val', 'Altitude score: ' + s.altitude_score.toFixed(2));
    }
    if (s.is_still !== null && s.is_still !== undefined) {
      setText('still-val', 'Still: ' + (s.is_still ? 'yes' : 'no'));
    }
    if (s.fused_score !== null && s.fused_score !== undefined) {
      setText('threshold-val', 'Fused score: ' + s.fused_score.toFixed(2));
    }
    if (s.countdown_remaining !== null && s.countdown_remaining !== undefined) {
      document.getElementById('countdown').textContent = Math.ceil(s.countdown_remaining) + 's';
    } else {
      document.getElementById('countdown').textContent = '';
    }

    const active = {
      STARTING:   [],
      MONITORING: ['n-imu', 'n-window', 'n-classifier'],
      CANDIDATE:  ['n-classifier', 'n-candidate'],
      VERIFYING:  ['n-candidate', 'n-altitude', 'n-stillness'],
      FUSION:     ['n-altitude', 'n-stillness', 'n-fusion'],
      NORMAL:     ['n-fusion', 'n-threshold', 'n-normal'],
      WARNING:    ['n-threshold', 'n-warning'],
      CANCELLED:  ['n-warning', 'n-cancelled'],
      ALARM:      ['n-warning', 'n-alarm'],
    }[phase] || [];
    active.forEach(id => document.getElementById(id).classList.add('active'));

    const logDiv = document.getElementById('log');
    logDiv.innerHTML = (s.log || []).map(l => '<div>' + l + '</div>').join('');
  } catch (e) {
    // server not ready yet, ignore and retry
  }
}

setInterval(tick, 300);
tick();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE)


if __name__ == "__main__":
    t = threading.Thread(target=simulation_loop, daemon=True)
    t.start()
    print("Dashboard running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
