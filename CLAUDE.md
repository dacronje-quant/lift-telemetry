# lift-telemetry (Python package: vbt)

Strength-training telemetry: any motion-sensor recording (watch, phone, IMU; wrist or bar)
plus any workout log (Hevy, Strong, CSV, typed notes, JSON) becomes `data/sessions/<id>/`
and an interactive report on sets, reps, velocity and fatigue.
Inputs are normalised by `vbt/logs.py` (logs) and `vbt/sensors.py` (sensors); everything
downstream is source-agnostic.

- **When the user shares a workout, follow `.claude/skills/workout-report/SKILL.md`.**
- Algorithm changes: measure with `python -m vbt.evaluate` (see `docs/IMPROVING.md`) and
  refresh reports with `python -m vbt.pipeline run --all`.
- Tests: `python -m pytest -q tests` (fast, synthetic data only).
- Never commit raw sensor exports; the 25 Hz `sensor_25hz.csv.gz` is the stored copy.
- `sets.json` `source`: `auto` (detector), `reviewed`/`manual` (human-checked ground truth).
  Only mark windows reviewed after actually checking them.

## Working agreement (ongoing project)
This is a living project: the user sends a new workout (workout log + sensor recording)
after every session, and each one should leave the system better than before.
For every new workout:
1. Produce and publish its report (skill above) and commit the session.
2. Review its set windows and mark them `reviewed`, so it becomes ground truth.
3. Run `python -m vbt.evaluate --record` and compare with the previous line in
   `data/eval/history.jsonl`. Say in the reply whether detection got better or worse.
4. Fix at least one weakness the new data exposed (new exercise profile, a missed set,
   a miscounted rep pattern, a new log or sensor format, a template issue) or take one item
   from `docs/IMPROVING.md`,
   keeping it only if the scoreboard holds on all reviewed sessions. Then
   `python -m vbt.pipeline run --all` so older reports pick up the improvement.
5. Add a dated line to the log below: what was learned and what changed.

## Log
- 2026-11-01: first session (Full Body C, 22 sets / 7 exercises). Built the pipeline,
  log-guided set detection (22/22 on this session), cross-session library and eval harness.
  Weak spots: sets done but not logged (18/21 placed), invisible exercises placed on
  noise. Exercise model across sessions needs a second workout.
- 2026-11-01 (later): made inputs generic. vbt/logs.py (Hevy, Strong, CSV, notes, JSON; kg/lb)
  and vbt/sensors.py (Sensor Logger watch/phone, any IMU CSV; gravity estimated from raw
  accel). Validated on the Nov 1 watch's raw accelerometer: with gyro, gravity within 0.6°
  median and velocities within 0.004 m/s of Apple's; accelerometer-only, rep counts ±1,
  velocity ~0.07 m/s worse. Sensor placement "bar" counts every lift from height.
