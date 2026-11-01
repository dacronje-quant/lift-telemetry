# Apple Watch VBT & rep counter

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

## Real session: 2026-11-01 (Full Body C)
47-minute whole-workout recording, 22 sets / 7 exercises, aligned to the Hevy log.
```bash
python analysis/workout_2026_11_01.py path/to/WristMotion.csv   # -> analysis/out/workout_2026-11-01.json
python analysis/build_report.py                                  # -> report/full-body-c-2026-11-01.html
```
* `data/sets_2026-11-01.json`: annotated set windows (recording clock, Hevy order).
* `data/wristmotion_2026-11-01_10hz.csv.gz`: decimated derived signals (vertical accel, gravity, gyro). The raw 100 Hz zip is not committed.
* `vbt/workout.py`: per-exercise rep segmentation (`analyze_set`; vertical position for barbell / squat / pulldown / lateral raise, forearm rotation for curls and pushdowns), per-rep ZUPT velocity, ROM, tempo, plausibility flags, motion-signature features (`set_features`) and a leave-one-set-out classifier (`loso_classify`).
* `vbt/session.py::find_sets`: finds sets by rep rhythm. Use it when the wrist isn't still between sets (walking, loading plates), where `split_sets` fuses everything.
