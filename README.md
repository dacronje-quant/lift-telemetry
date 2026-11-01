# lift-telemetry: Apple Watch VBT & rep counter

Velocity-based training metrics and rep counting from Apple Watch motion data.
This is a clean-room build from published VBT methods, not copied from Motra.

## Record a set
1. Install **Sensor Logger** (free) on iPhone + Apple Watch.
2. On the watch enable *Wrist Motion* (CoreMotion deviceMotion), 100 Hz.
3. Start recording, do the set, stop. Export as CSV → `WristMotion.csv`.

Wear the watch snug. Wrist ≈ bar path for squat / bench / OHP / deadlift.

## Analyze
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
python -m vbt.pipeline new --sensor Hevy_Workout.zip --hevy hevy.txt   # ingest + detect + analyse + HTML
python -m vbt.pipeline review 2026-11-01 review.png                    # rep-marker grid for a quick check
python -m vbt.pipeline run --all                                       # re-run everything after a code change
python -m vbt.evaluate --record                                        # benchmark vs human-checked sessions
```
* `data/sessions/<id>/`: `hevy.txt`, `sensor_25hz.csv.gz` (anti-aliased 25 Hz copy of WristMotion),
  `meta.json`, `sets.json` (set windows: `auto` or human `reviewed`/`manual`), `notes.json`
  (optional commentary), `analysis.json` (compact results that feed the cross-session library).
* `reports/<id>.html`: the interactive report.
* `vbt/align.py`: finds the logged sets in the recording, guided by the Hevy log (rep counts, rhythm,
  wrist orientation and ROM consistency, rest gaps, plus an exercise model trained on earlier sessions).
  No hand marking needed.
* `vbt/workout.py`: per-exercise rep segmentation, per-rep velocity/ROM/tempo, plausibility flags,
  features, classifier. `vbt/library.py`: cross-session data. `vbt/evaluate.py`: scoreboard.
* Workflow for new workouts: `.claude/skills/workout-report/SKILL.md`. Improvement loop: `docs/IMPROVING.md`.
