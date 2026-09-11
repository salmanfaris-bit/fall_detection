"""
web_app.py — Local browser dashboard for the fall-detection pipeline.

Runs the SAME pipeline logic already in sensors.py / detection.py / alerts.py
(mock sensors, no hardware needed) in a background thread, forever cycling
through MONITORING -> CANDIDATE -> VERIFYING -> FUSION -> WARNING ->
CANCELLED/ALARM, and exposes it as a flow-diagram dashboard in the browser.

The "Cancel" button lives on the WEBSITE, not on any GPIO pin. That works
for free: alerts.py's HardwareController already runs in mock mode whenever
real RPi.GPIO isn't available (i.e. on a laptop), and mock mode is driven by
HardwareController.simulate_button_press() / is_button_pressed() — exactly
what main.py's live pipeline would use if you *had* wired a physical button.
This file just calls that same mock method from a Flask route instead of
from a GPIO pin, so nothing in alerts.py had to change.

Run:
    python3 web_app.py
Then open:
    http://localhost:5000
"""

import threading
import time

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string

from sensors import (
    MockBNO055Reader, MockBMP280Reader,
    KalmanFilter1D, DriftingBaseline, altitude_from_pressure,
)
from detection import check_stillness, check_altitude_drop, fuse_decision, FUSION_THRESHOLD
from alerts import HardwareController

TARGET_HZ = 200
STILLNESS_WINDOW_S = 2.0
STILLNESS_SAMPLES = int(STILLNESS_WINDOW_S * TARGET_HZ)
WARNING_COUNTDOWN_S = 30.0
# Demo pacing only (not a detection parameter): how many simulated IMU
# samples into each cycle the mock "fall" is injected, so the dashboard
# doesn't sit in MONITORING for a long time between demonstrations.
TRIGGER_AFTER_SAMPLES = 5 * TARGET_HZ

app = Flask(__name__)

_lock = threading.Lock()
STATE = {
    "phase": "MONITORING",   # MONITORING | CANDIDATE | VERIFYING | FUSION | WARNING | CANCELLED | ALARM
    "ml_prob": 0.0,
    "is_still": None,
    "altitude_score": None,
    "fused_score": None,
    "countdown_remaining": None,
    "cycle": 0,
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
    """Continuously cycles the pipeline using mock sensors, mirroring
    demo_simulation.py's state machine but running forever for the dashboard
    instead of two fixed scenarios."""
    cycle = 0
    while True:
        cycle += 1
        shared_counter = [0]
        imu = MockBNO055Reader()
        baro = MockBMP280Reader()
        imu.set_counter(shared_counter)
        baro.set_counter(shared_counter)
        imu.configure()
        baro.configure()

        kalman = KalmanFilter1D(initial_value=1013.0)
        baseline = DriftingBaseline(initial_value=1013.0, tau_s=300.0, sample_dt_s=0.05)
        hw = HardwareController(buzzer_pin=18, button_pin=23)
        _current_hw["hw"] = hw

        _set_state(phase="MONITORING", ml_prob=0.0, is_still=None, altitude_score=None,
                   fused_score=None, countdown_remaining=None, cycle=cycle)
        _log(f"Cycle {cycle}: monitoring")

        buffer, altitude_deque = [], []
        fall_prob, altitude_baseline_m = 0.0, 0.0
        countdown_start = None
        state = "MONITORING"

        while True:
            shared_counter[0] += 1
            accel_g, gyro_dps = imu.read_raw()
            p_raw, _ = baro.read_raw()

            if state == "MONITORING":
                if shared_counter[0] % 10 == 0:
                    p_f = kalman.update(p_raw)
                    ref = baseline.update(p_f)
                    altitude_deque.append(altitude_from_pressure(p_f, ref))

                if shared_counter[0] > TRIGGER_AFTER_SAMPLES and fall_prob == 0.0:
                    fall_prob = 0.85  # simulated ML trigger, same value demo_simulation.py uses
                    altitude_baseline_m = float(np.median(altitude_deque[-20:])) if altitude_deque else 0.0
                    buffer, altitude_deque = [], [altitude_baseline_m]
                    _set_state(phase="CANDIDATE", ml_prob=fall_prob)
                    _log(f"Cycle {cycle}: candidate fall flagged (ML prob {fall_prob:.2f})")
                    time.sleep(0.6)  # let the diagram show the candidate node before moving on
                    state = "VERIFYING"
                    _set_state(phase="VERIFYING")

            elif state == "VERIFYING":
                buffer.append({
                    "accel_x_g": accel_g[0], "accel_y_g": accel_g[1], "accel_z_g": accel_g[2],
                    "gyro_x_dps": gyro_dps[0], "gyro_y_dps": gyro_dps[1], "gyro_z_dps": gyro_dps[2],
                })
                if shared_counter[0] % 10 == 0:
                    p_f = kalman.update(p_raw)
                    ref = baseline.value  # frozen during verification, same as main.py
                    altitude_deque.append(altitude_from_pressure(p_f, ref))

                if len(buffer) >= STILLNESS_SAMPLES:
                    df = pd.DataFrame(buffer)
                    df["t"] = np.arange(len(df)) / TARGET_HZ
                    still_res = check_stillness(df, accel_std_threshold=0.15, gyro_std_threshold=15.0)
                    alt_after = float(np.median(altitude_deque[-20:])) if altitude_deque else 0.0
                    alt_res = check_altitude_drop(altitude_baseline_m, alt_after, drop_threshold_m=0.3)
                    fuse = fuse_decision(ml_prob=fall_prob, is_still=still_res["is_still"],
                                          altitude_score=alt_res["altitude_score"])

                    _set_state(phase="FUSION", is_still=still_res["is_still"],
                               altitude_score=alt_res["altitude_score"], fused_score=fuse["fused_score"])
                    _log(f"Cycle {cycle}: fused score {fuse['fused_score']:.2f} (threshold {FUSION_THRESHOLD})")
                    time.sleep(0.8)

                    if fuse["confirm_fall"]:
                        state = "WARNING_COUNTDOWN"
                        countdown_start = time.time()
                        hw.start_warning_beep(interval_s=0.5)
                        _set_state(phase="WARNING", countdown_remaining=WARNING_COUNTDOWN_S)
                        _log(f"Cycle {cycle}: WARNING countdown started \u2014 cancel on the dashboard to stop it")
                    else:
                        _set_state(phase="NORMAL")
                        _log(f"Cycle {cycle}: suppressed as false alarm")
                        time.sleep(1.5)
                        state = "DONE"

            elif state == "WARNING_COUNTDOWN":
                elapsed = time.time() - countdown_start
                remaining = max(0.0, WARNING_COUNTDOWN_S - elapsed)
                _set_state(countdown_remaining=remaining)

                if hw.is_button_pressed():
                    hw.stop_sound()
                    _set_state(phase="CANCELLED", countdown_remaining=0)
                    _log(f"Cycle {cycle}: cancelled from the web dashboard")
                    time.sleep(2.5)
                    state = "DONE"
                elif remaining <= 0:
                    hw.start_alarm_sound()
                    _set_state(phase="ALARM", countdown_remaining=0)
                    _log(f"Cycle {cycle}: FINAL ALARM \u2014 no response within 30s")
                    time.sleep(3.0)
                    hw.stop_sound()
                    state = "DONE"

            if state == "DONE":
                hw.cleanup()
                break

            time.sleep(0.0008)  # accelerated pacing for the demo, same idea as demo_simulation.py

        time.sleep(1.2)  # brief pause between cycles


@app.route("/api/state")
def api_state():
    with _lock:
        return jsonify(dict(STATE))


@app.route("/api/cancel", methods=["POST"])
def api_cancel():
    """Website-triggered cancel. Calls the SAME mock button method main.py's
    live pipeline would call from a real GPIO interrupt — no button wired."""
    hw = _current_hw["hw"]
    if hw is not None:
        hw.simulate_button_press()
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "no active cycle"}), 409


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
  .subtitle { text-align: center; color: var(--muted); margin: 0 0 28px; font-size: .9rem; }
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
  .metrics { font-size: .78rem; margin-top: 4px; color: var(--ink); }
  .metrics b { font-size: .78rem; }
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
  <p class="subtitle">Simulated sensors, running locally. This mirrors main.py's state machine in real time.</p>

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
    document.getElementById('cancelBtn').style.display = (phase === 'WARNING') ? 'inline-block' : 'none';

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
