# Retraining Report — Fixing the SisFall→BNO055 Generalization Gap

## What changed

| | Old (`fall_rf.joblib`) | New (`fall_rf_v2.joblib`) |
|---|---|---|
| Training data | SisFall (different hardware) | Your 10 real BNO055 sessions, 4 subjects |
| Features | 18, signed raw axes (`ax_range`, `az_min/max`) | 22, **gravity-relative** (`vertical_*`, `horizontal_*`, `tilt`) + free-fall dip + post-window stillness |
| Ground truth | — | Fixed: uses the reliable raw per-sample labels (the `events.csv` misalignment bug from the earlier report is bypassed) |
| Validation | Unknown / not re-verified on your hardware | **Leave-one-subject-out** (train on 3 people, test on the 4th, rotated) — the honest test of generalizing to a new wearer |
| `FALL_PROB_THRESHOLD` | 0.65 | 0.32 (re-derived from an out-of-fold precision/recall curve) |

## Why the old model failed

Your resting `accel_y_g` measured **+1.0** in the data, but the code's own comments assumed **-1.0** ("Y is up... on this hardware mounting"). Checking further: the estimated gravity direction actually **flips sign on the dominant axis between session 1 and every other session** — the device was mounted/worn differently at least once. Since the old model's top two features (`ax_range`, `gyro_svm_max`) included a raw signed axis, that mounting inconsistency alone was enough to scramble its signal — on top of the base SisFall→your-hardware domain shift.

## Results — leave-one-subject-out (unseen-person) evaluation

| Held-out subject | Windows | Real falls | AUC (old model, in-sample) | AUC (new model, LOSO) |
|---|---:|---:|---:|---:|
| Damien | 1,293 | 204 | — | **0.908** |
| nishas | 1,087 | 181 | — | **0.897** |
| salman | 1,197 | 169 | — | **0.976** |
| sreerag | 1,360 | 162 | — | **0.922** |
| **Overall** | 4,937 | 716 | **0.67** | **0.924** |

At the new threshold (0.32), out-of-fold: **sensitivity 77.1%, specificity 94.3%** (window level) — vs. the old model's 7.5% / 100% at its deployed threshold. That's not a tradeoff, it's a real improvement: the old model wasn't "conservative and safe," it was simply not separating falls from ADLs at all on your hardware (AUC 0.67 ≈ barely better than a coin flip on the finer cases).

### Full-pipeline simulation (ML threshold + your existing 2-consecutive-window rule + your existing stillness gate)

| | Old model | New model |
|---|---|---|
| Fall sessions with ≥1 confirmed alarm | 2 / 5 | **5 / 5** |
| False triggers across pure-ADL sessions (~24 min) | 0 (but see above — it wasn't detecting anything) | 34 → **5** after the stillness gate |

The 5 remaining false triggers are concentrated in session 1 — the same session with the flipped gravity vector, which is itself a good validation that the orientation-relative features are doing their job (the *one* session with a real mounting inconsistency is exactly where residual false alarms remain).

## What I changed in your code

- **`detection.py`**: `extract_features()` now takes a `gravity_vec` argument and computes vertical/horizontal/tilt relative to it, plus 4 new features (`free_fall_dip`, `time_dip_to_peak`, `post_tail_std`, `horizontal_std`). `check_stillness`, `check_altitude_drop`, `fuse_decision`, `validate_window` are untouched.
- **`main.py`**: loads `fall_rf_v2.joblib`, calibrates a gravity vector at startup (`calibrate_orientation()`, same pattern as your existing `calibrate_barometer()`) during the same still-standing period, passes it into `extract_features()`, and uses the new `FALL_PROB_THRESHOLD = 0.32`.
- **`sensors.py`, `alerts.py`, `calibrate_and_collect.py`**: unchanged.
- **`fall_rf_v2.joblib`**: new RandomForest, same `{"model", "feature_cols"}` load contract your code already expects.

## What I did NOT touch, and why

- **`FUSION_THRESHOLD` / `FUSION_WEIGHTS` in `detection.py`** (the ML+stillness+altitude fusion stage) — left as-is. The altitude leg depends on your barometer, and the pressure readings in your recorded data (~550–570 hPa, physically implausible at ground level — flagged in the earlier report) mean I can't honestly validate that leg. I only simulated ML + stillness above, deliberately leaving altitude out rather than fine-tune against data I don't trust.
- **Did not merge in SisFall.** I don't have network access to fetch it and it wasn't uploaded — this model is trained purely on your 10 sessions. It generalizes well *across your 4 subjects* (that's what LOSO tests), but a 5th, very different person is still an open question given the small subject count.

## Recommended next steps, in order of impact

1. **Fix the barometer calibration/units bug** — then I can properly validate and re-tune the altitude leg of the fusion, which should shrink that remaining session-1 false-alarm cluster further.
2. **Record 2–3 more subjects**, ideally with deliberately varied mounting/wear to stress-test the orientation-invariance fix.
3. If you get SisFall's raw files to me (upload, or a GitHub-mirror link — my sandbox can't reach arbitrary dataset hosts), I can fold it in as additional training volume on top of this gravity-relative feature set, likely improving robustness to subjects outside your current 4 without giving up the hardware-specific gains made here.
4. Swap `fall_rf_v2.joblib` in as your default (`MODEL_PATH`) once you're comfortable — I kept the filename distinct from the original so nothing is silently overwritten.
