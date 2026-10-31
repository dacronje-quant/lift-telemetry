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
