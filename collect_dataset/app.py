"""
app.py — Continuous-session web dashboard for fall/ADL calibration data
collection.

WHY THIS EXISTS (read this before running it)
-----------------------------------------------
The earlier version of this dashboard recorded one clean, isolated clip per
activity: pick a label, press Start, do ONE thing, press Stop, save. That
produces data shaped exactly like SisFall — clean, separated clips — which
is the same shape mismatch that caused your POC's false positives, because
your real inference pipeline slides a 2-second window across a CONTINUOUS
live stream that includes natural transitions between activities.

This version fixes that: you press Start ONCE per session (5-10 minutes),
the sensor keeps streaming the whole time, and you just tap a button each
time the activity changes. Nothing stops or restarts — you're marking
boundaries inside one continuous recording, not cutting separate files.

Each saved session gives you two files:
    - session_XXX_raw.csv     one continuous stream, every sample already
                               tagged with whatever activity was current
                               at that instant (label + category columns)
    - session_XXX_events.csv  the clean segment boundaries derived from
                               your marker taps (label, category, start_s,
                               end_s, duration_s) — useful for building your
                               2s training windows with correct labels,
                               including the boundary/transition windows.

Run:
    pip install flask pandas numpy
    python3 app.py
Then open http://localhost:5000 in a browser on the same machine (or your
LAN if you pass host="0.0.0.0" and know what you're doing security-wise).

This file does NOT touch sensors.py, detection.py, alerts.py, or your
existing calibrate_and_collect.py / collect_my_dataset.py scripts — it only
imports the same reader classes those scripts already use, so the sensor
driver, axis convention (Y = vertical), and glitch handling are identical
everywhere.
"""

import os
import csv
import time
import threading

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template

from sensors import BNO055Reader, MockBNO055Reader, BMP280Reader, MockBMP280Reader

OUTPUT_DIR = "recorded_sessions"
MANIFEST_PATH = os.path.join(OUTPUT_DIR, "session_manifest.csv")
MANIFEST_FIELDS = [
    "session_num", "filename_raw", "filename_events", "source",
    "duration_s", "n_events", "n_falls", "peak_accel_g", "timestamp",
]
TARGET_HZ = 200

# Quick-tap presets shown as buttons in the UI. Edit this list to match your
# hard-negative / false-positive catalog — it only controls what shows up as
# one-tap buttons, you can always type a custom label instead.
PRESET_ADLS = [
    "idle_standing", "walking", "sitting_slow", "sitting_fast", "standing_up",
    "lying_down", "bending", "picking_object", "stairs_up", "stairs_down",
    "jumping", "running", "stumble_recover", "coughing_sneezing",
]
PRESET_FALLS = [
    "fall_forward", "fall_backward", "fall_lateral_left",
    "fall_lateral_right", "fall_sitting",
]

app = Flask(__name__)


# ---------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------
def load_manifest():
    if not os.path.exists(MANIFEST_PATH):
        return []
    with open(MANIFEST_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["session_num"] = int(r["session_num"])
    return rows


def append_manifest(row):
    is_new = not os.path.exists(MANIFEST_PATH)
    with open(MANIFEST_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow(row)


def next_session_number(manifest):
    if not manifest:
        return 1
    return max(r["session_num"] for r in manifest) + 1


# ---------------------------------------------------------------------
# Recorder — owns the hardware handles, the sampling loop, and the running
# list of event-segment boundaries for the CURRENT session.
# ---------------------------------------------------------------------
class Recorder:
    def __init__(self):
        self.lock = threading.Lock()
        self.imu = None
        self.baro = None
        self.imu_is_real = False
        self.baro_is_real = False

        self.thread = None
        self.stop_event = threading.Event()
        self.recording = False
        self.buffer = []
        self.events = []          # closed segments: {label, category, start_s, end_s, duration_s}
        self.current_label = None
        self.current_category = None
        self.current_start_s = None
        self.start_time = None

        # Holds the most recently stopped-but-not-yet-saved session.
        self.pending = None  # {"df", "events_df", "duration", "peak", "n_falls"}

        self.manifest = load_manifest()
        self._init_sensors()

    def _init_sensors(self):
        try:
            self.imu = BNO055Reader(address=0x28)
            self.imu.configure()
            self.imu_is_real = True
        except Exception as e:
            print(f"[WARNING] Could not connect to real BNO055 hardware: {e}")
            self.imu = MockBNO055Reader()
            self.imu.configure()
            self.imu_is_real = False

        try:
            self.baro = BMP280Reader(address=0x76)
            self.baro.configure()
            self.baro_is_real = True
        except Exception as e:
            print(f"[WARNING] Could not connect to real BMP280 hardware: {e}")
            self.baro = MockBMP280Reader()
            self.baro.configure()
            self.baro_is_real = False

    def _elapsed(self):
        return (time.time() - self.start_time) if self.start_time else 0.0

    def status(self):
        with self.lock:
            return {
                "recording": self.recording,
                "elapsed_s": round(self._elapsed(), 1),
                "samples": len(self.buffer),
                "current_label": self.current_label,
                "current_category": self.current_category,
                "n_events_closed": len(self.events),
                "imu_is_real": self.imu_is_real,
                "baro_is_real": self.baro_is_real,
                "pending": self._pending_summary(),
            }

    def _pending_summary(self):
        if not self.pending:
            return None
        p = self.pending
        return {
            "duration_s": round(p["duration"], 2),
            "peak_accel_g": round(p["peak"], 3) if p["peak"] is not None else None,
            "n_events": len(p["events_df"]),
            "n_falls": p["n_falls"],
            "samples": len(p["df"]),
            "events": p["events_df"].to_dict("records"),
        }

    def start_session(self):
        with self.lock:
            if self.recording:
                raise RuntimeError("Already recording.")
            if self.pending:
                raise RuntimeError("Save or discard the pending session first.")
            self.buffer = []
            self.events = []
            self.current_label = "idle"
            self.current_category = "adl"
            self.current_start_s = 0.0
            self.stop_event.clear()
            self.recording = True
            self.start_time = time.time()
            self.thread = threading.Thread(target=self._record_loop, daemon=True)
            self.thread.start()

    def _record_loop(self):
        interval = 1.0 / TARGET_HZ
        next_t = time.time()
        i = 0
        while not self.stop_event.is_set():
            accel_g, gyro_dps = self.imu.read_raw()
            try:
                p, _ = self.baro.read_raw()
            except Exception:
                p = None
            with self.lock:
                label = self.current_label
                category = self.current_category
            self.buffer.append({
                "t": i / TARGET_HZ,
                "accel_x_g": accel_g[0], "accel_y_g": accel_g[1], "accel_z_g": accel_g[2],
                "gyro_x_dps": gyro_dps[0], "gyro_y_dps": gyro_dps[1], "gyro_z_dps": gyro_dps[2],
                "pressure_hpa": p,
                "label": label, "category": category,
            })
            i += 1
            next_t += interval
            sleep_time = next_t - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # Fell behind (slow sensor read) — resync instead of busy-looping.
                next_t = time.time()

    def mark_event(self, label, category):
        """Close the segment that was running and open a new one, WITHOUT
        stopping the recording thread. This is the core of continuous
        collection: the sensor never pauses, only the label changes."""
        with self.lock:
            if not self.recording:
                raise RuntimeError("Not recording.")
            if not label:
                raise RuntimeError("Label is required.")
            now = self._elapsed()
            self.events.append({
                "label": self.current_label, "category": self.current_category,
                "start_s": round(self.current_start_s, 3), "end_s": round(now, 3),
                "duration_s": round(now - self.current_start_s, 3),
            })
            self.current_label = label
            self.current_category = category or "adl"
            self.current_start_s = now
            return {"closed": self.events[-1], "now_recording": label}

    def stop_session(self):
        with self.lock:
            if not self.recording:
                raise RuntimeError("Not recording.")
            self.stop_event.set()
            thread = self.thread
        thread.join(timeout=5)
        with self.lock:
            now = self._elapsed()
            # Close the final open segment.
            self.events.append({
                "label": self.current_label, "category": self.current_category,
                "start_s": round(self.current_start_s, 3), "end_s": round(now, 3),
                "duration_s": round(now - self.current_start_s, 3),
            })
            self.recording = False
            duration = now
            df = pd.DataFrame(self.buffer)
            events_df = pd.DataFrame(self.events)
            peak = None
            if not df.empty:
                a_svm = np.sqrt(df["accel_x_g"] ** 2 + df["accel_y_g"] ** 2 + df["accel_z_g"] ** 2)
                peak = float(a_svm.max())
            n_falls = int((events_df["category"] == "fall").sum()) if not events_df.empty else 0
            self.pending = {
                "df": df, "events_df": events_df, "duration": duration,
                "peak": peak, "n_falls": n_falls,
            }
            return self._pending_summary()

    def save(self):
        with self.lock:
            if not self.pending:
                raise RuntimeError("Nothing pending to save.")
            p = self.pending
            session_num = next_session_number(self.manifest)
            marker = "real" if self.imu_is_real else "mock"

            filename_raw = f"session_{session_num:03d}_{marker}_raw.csv"
            filename_events = f"session_{session_num:03d}_{marker}_events.csv"
            filepath_raw = os.path.join(OUTPUT_DIR, filename_raw)
            filepath_events = os.path.join(OUTPUT_DIR, filename_events)

            df = p["df"].copy()
            df["session_num"] = session_num
            df["source"] = "real_hardware" if self.imu_is_real else "mock_simulation"
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            df.to_csv(filepath_raw, index=False)

            events_df = p["events_df"].copy()
            events_df["session_num"] = session_num
            events_df.to_csv(filepath_events, index=False)

            row = {
                "session_num": session_num,
                "filename_raw": filename_raw,
                "filename_events": filename_events,
                "source": df["source"].iloc[0] if not df.empty else "unknown",
                "duration_s": round(p["duration"], 2),
                "n_events": len(events_df),
                "n_falls": p["n_falls"],
                "peak_accel_g": round(p["peak"], 3) if p["peak"] is not None else "",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            append_manifest(row)
            self.manifest.append(row)
            self.pending = None
            return row

    def discard(self):
        with self.lock:
            if not self.pending:
                raise RuntimeError("Nothing pending to discard.")
            self.pending = None


recorder = Recorder()


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", preset_adls=PRESET_ADLS, preset_falls=PRESET_FALLS)


@app.route("/api/status")
def api_status():
    return jsonify(recorder.status())


@app.route("/api/manifest")
def api_manifest():
    manifest = recorder.manifest
    total_falls = sum(int(r.get("n_falls", 0) or 0) for r in manifest)
    total_duration = sum(float(r.get("duration_s", 0) or 0) for r in manifest)
    return jsonify({
        "sessions": manifest,
        "total_falls": total_falls,
        "total_duration_s": round(total_duration, 1),
    })


@app.route("/api/start", methods=["POST"])
def api_start():
    try:
        recorder.start_session()
        return jsonify(recorder.status())
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/mark", methods=["POST"])
def api_mark():
    data = request.get_json(force=True) or {}
    label = (data.get("label") or "").strip().replace(" ", "_")
    category = (data.get("category") or "adl").strip()
    try:
        result = recorder.mark_event(label, category)
        return jsonify(result)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/stop", methods=["POST"])
def api_stop():
    try:
        summary = recorder.stop_session()
        return jsonify(summary)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/save", methods=["POST"])
def api_save():
    try:
        row = recorder.save()
        return jsonify(row)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/discard", methods=["POST"])
def api_discard():
    try:
        recorder.discard()
        return jsonify({"ok": True})
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 64)
    print(" Wearable Fall/ADL Calibration Data Collector — Continuous Sessions")
    print("=" * 64)
    if not recorder.imu_is_real:
        print("[!] No real BNO055 detected — running on MOCK IMU data.")
        print("    Sessions recorded now will be tagged 'mock' and are NOT")
        print("    useful for calibration. Connect the real sensor and restart.")
    print(f" Open http://localhost:5000 in your browser.")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
