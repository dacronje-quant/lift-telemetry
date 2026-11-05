# lift-telemetry

Turn motion-sensor recordings and any gym workout log into per-workout reports on reps, bar
speed and fatigue, improving with every session.

* **Sensors**: Apple Watch or phone via Sensor Logger, or any IMU CSV (accelerometer, ideally with
  gyroscope), worn on the wrist or clipped to the bar. Units and conventions are detected; gravity is
  estimated when the device doesn't provide it (`vbt/sensors.py`).
* **Workout logs**: Hevy share text, Strong CSV, any CSV, JSON or typed notes (`vbt/logs.py`).

A clean-room build from published velocity-based-training (VBT) methods.

## Recording
Anything that logs a 3-axis accelerometer at ≥ 25 Hz works; a gyroscope makes velocities much
more accurate. Supported setups (Apple Watch is validated on a real session; the generic IMU path
on that watch's raw accelerometer and on synthetic data; the phone path is not yet tested on a real
recording):
* **Apple Watch + Sensor Logger** (free): enable *Wrist Motion*, 100 Hz, export CSV/zip.
* **Phone + Sensor Logger** (iOS/Android): Accelerometer, Gravity, Gyroscope. Strap it to the
  forearm or clip it to the bar.
* **Any IMU / other app**: a CSV with a time column and acc x/y/z (+ gyro x/y/z). Pass
  `--accel-convention gravity` if the device reads −1 g on the axis pointing up at rest (Apple),
  otherwise the default `specific_force` (+1 g up; Android and most IMUs) applies.

Wear it snug and in the same place every session. Wrist ≈ bar path for squat / bench / OHP /
deadlift; `--placement bar` for a bar-mounted sensor.

## Single set (quick look)

```bash
pip install -r requirements.txt
python -m vbt WristMotion.csv --plot set.png
```
Output: rep count; mean concentric, peak velocity and ROM for each rep; velocity loss across the set.

1RM from several sets:
```python
from vbt import estimate_1rm
estimate_1rm([60, 80, 100], [0.95, 0.72, 0.50], lift="squat")
```

## How it works
1. Vertical accel = user acceleration projected onto −gravity (CoreMotion gives both).
2. Band-passed double integration → drift-free position → peaks/valleys = reps.
3. For each rep: integrate accel top→top with zero-velocity resets at both ends
   (stationary) + linear de-drift; the concentric phase is the positive-velocity lobe.
4. Load–velocity linear fit → e1RM at minimum velocity threshold.

## Whole-workout recording (no per-set exports)
Start one Sensor Logger recording at the start of the session and stop it at the end.
```bash
python -m vbt WristMotion.csv --session
```
Sets are split on rest periods of 20 s or more (`split_sets` in `vbt/session.py`; tune `min_rest_s`,
`active_thresh`). Walking between stations can show up as a false "set". Most of
those have no reps and get dropped.

## One workout → one report
```bash
python -m vbt.pipeline new --sensor recording.zip --log workout.txt     # ingest + detect + analyse + HTML
python -m vbt.pipeline review 2026-11-01 review.png                    # rep-marker grid for a quick check
python -m vbt.pipeline run --all                                       # re-run everything after a code change
python -m vbt.evaluate --record                                        # benchmark vs human-checked sessions
```
* `data/sessions/<id>/`: `log.json` + `log_source.*` (normalised and original log), `sensor_25hz.csv.gz`
  (canonical 25 Hz motion table),
  `meta.json`, `sets.json` (set windows: `auto` or human `reviewed`/`manual`), `notes.json`
  (optional commentary), `analysis.json` (compact results that feed the cross-session library).
* `reports/<id>.html`: the interactive report.
* `vbt/align.py`: finds the logged sets in the recording, guided by the workout log (rep counts, rhythm,
  wrist orientation and ROM consistency, rest gaps, plus an exercise model trained on earlier sessions).
  No hand marking needed.
* `vbt/workout.py`: per-exercise rep segmentation, per-rep velocity/ROM/tempo, plausibility flags,
  features, classifier. `vbt/library.py`: cross-session data. `vbt/evaluate.py`: scoreboard.
* `vbt/personal.py`: learns how *you* do each exercise from checked workouts (`python -m vbt.personal`).
  `vbt/infer.py`: sensor-only mode, run on every workout as a shadow test with the log hidden
  (`python -m vbt.infer <id>`); its "readiness" score tracks progress toward log-free use.
* Workflow for new workouts: `.claude/skills/workout-report/SKILL.md`. Improvement loop: `docs/IMPROVING.md`.
