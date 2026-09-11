# fall_detection — Folder Guide

Your Drive folder `fall_detection` has 4 subfolders. This is a map of what's where, so you can jump straight to the file you need.

```
fall_detection/
├── final5/              ← the actual production system (start here)
├── collect_dataset/     ← Raspberry Pi web app used to record your own sensor data
├── own_dataset/         ← the real sessions you recorded with collect_dataset
└── SisFall_dataset/     ← public benchmark dataset + model training pipeline
```

---

## 1. `final5/` — Production fall-detection system

The finished, working version. If you want to run the detector, demo it, or explain how it works, everything is here.

| File | What it does |
|---|---|
| `README.md` | Full write-up of this system: architecture, fusion logic, setup/run instructions, tuning knobs |
| `Fall_Detection_System_Explainer.docx` | Plain-English explainer doc (good for presentations/non-technical readers) |
| `main.py` | Production entry point — the live state machine |
| `sensors.py` | IMU + barometer drivers, mocks, Kalman filter, drift-resistant baseline |
| `detection.py` | Feature extraction, stillness/altitude checks, fusion decision logic |
| `alerts.py` | Buzzer + cancel-button GPIO controller |
| `fall_rf.joblib` | Trained RandomForest model (18 features — frozen, don't retrain in place) |
| `demo_simulation.py` | End-to-end demo with mock sensors, no hardware required |
| `calibrate_and_collect.py` | CLI to measure barometer noise or record a labeled trial |
| `debug_validate_window.py` | Debug script for the window-validation logic |
| `tests.py` | Automated test suite (27 checks) |
| `web_app.py` / `web_app_demo.py` | Web dashboard versions (live / demo mode) |
| `WhatsApp Image 2026-09-02...jpeg` | Reference/setup photo |

**Quick run:**
```bash
python3 tests.py              # run test suite
python3 demo_simulation.py    # demo without hardware
python3 main.py               # live pipeline
```

---

## 2. `collect_dataset/` — Data-collection web app

The tool used to gather your own real-world sensor sessions (feeds into `own_dataset/`).

| File | What it does |
|---|---|
| `app.py` | Backend server for the data-collection app |
| `sensors.py` | Sensor driver code for live collection |
| `index.html` | Front-end UI for the collection app |
| `session details.txt` | Notes on session/recording setup |

---

## 3. `own_dataset/` — Your recorded sessions

Real sensor recordings you captured (sessions 001–010), each with a raw stream and a labeled events file.

| File pattern | What it is |
|---|---|
| `session_XXX_real_raw.csv` | Raw, full-rate sensor stream for that session (large files) |
| `session_XXX_real_events.csv` | Labeled fall/event markers for that session (small files) |
| `session_manifest.csv` | Index/summary of all sessions |

Sessions present: 001–010 (raw files for 001–010, event files for 001–010; note 007–010 raw files are the largest).

---

## 4. `SisFall_dataset/` — Public dataset + model training pipeline

The public SisFall benchmark dataset (subject folders) plus the scripts used to turn it into the trained model that `final5/` uses.

| File | What it does |
|---|---|
| `Readme.txt` | Original SisFall dataset documentation (subject/activity codes, format) |
| `Supplementary.pdf` | Supplementary paper/documentation for the dataset |
| `sisfall_loader.py` | Loads raw SisFall subject files into a usable format |
| `feature_extraction.py` | Extracts the 18 ML features from raw windows |
| `build_dataset.py` | Builds the combined training dataset |
| `train_rf.py` | Trains the RandomForest model |
| `features.csv` | Extracted feature table used for training |
| `fall_rf.joblib` | Trained model output (same model used in `final5/`) |
| `SA01–SA23`, `SE01–SE15` | Per-subject raw data folders from the SisFall dataset (SA = adult, SE = elderly, by convention) |
| `.venv/`, `__pycache__/` | Environment/cache — safe to ignore |

---

## Where do I go for...?
- **"How does the whole system work?"** → `final5/README.md` or `Fall_Detection_System_Explainer.docx`
- **"I want to run/demo it"** → `final5/demo_simulation.py` or `main.py`
- **"I want to retrain the model"** → `SisFall_dataset/train_rf.py` (+ `feature_extraction.py`, `build_dataset.py`)
- **"I want to record new real-world data"** → `collect_dataset/app.py`
- **"Where's my recorded data?"** → `own_dataset/session_XXX_*.csv`
