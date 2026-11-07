# lift-telemetry (Python package: vbt)

Strength-training telemetry: any motion-sensor recording (watch, phone, IMU; wrist or bar)
plus any workout log (Hevy, Strong, CSV, typed notes, JSON) becomes `data/sessions/<id>/`
and an interactive report on sets, reps, velocity and fatigue.
Inputs are normalised by `vbt/logs.py` (logs) and `vbt/sensors.py` (sensors); everything
downstream is source-agnostic.

- **When the user shares a workout, follow `.claude/skills/workout-report/SKILL.md`.**
- Algorithm changes: measure with `python -m vbt.evaluate` (your workouts) **and**
  `python -m vbt.public --data <scratch dir>` (RecoFit public data: many people, true rep
  counts; downloads 1.56 GB outside the repo). Keep a change only if neither gets worse;
  record both after committing (`--record`). Refresh reports with
  `python -m vbt.pipeline run --all`.
- Tests: `python -m pytest -q tests` (fast, synthetic data only).
- Never commit raw sensor exports; the 25 Hz `sensor_25hz.csv.gz` is the stored copy.
- `sets.json` `source`: `auto` (detector), `reviewed`/`manual` (human-checked ground truth).
  Only mark windows reviewed after actually checking them.

## Goal
One day the user hands over **sensor data only** and gets the whole workout back (sets,
exercises, reps, velocity), tailored to how *they* lift. Every logged workout is training
data toward that. Two mechanisms carry it:
- `vbt/personal.py` learns per exercise, from checked sets: which rep signal and rest point
  count the user's reps best, whether setup/unrack cycles should count, a motion prototype,
  and personal tempo / range-of-motion / rotation norms. A learned setting replaces the
  generic one only with evidence from >= 2 workouts that also wins on held-out sets.
  `python -m vbt.personal` prints the current state.
- `vbt/infer.py` is the sensor-only mode. The pipeline and `vbt.evaluate` run it as a
  *shadow test*: log hidden, knowledge from other workouts only, scored against the log.
  "Readiness" = share of logged sets found, named and counted exactly. That number is the
  project's north star; report it after every workout.
- **Learning is kept out of the workout reports.** `reports/<id>.html` is about that
  workout only. Everything about the system learning (readiness trend, per-workout shadow
  tests, learned per-exercise settings, scoreboard history) lives on one page,
  `reports/learning.html` (`python -m vbt.learning`; the pipeline rebuilds it), published
  as a single Artifact that is updated in place after every workout:
  https://claude.ai/artifact/L6TKF62UyjYP97cihDmz83

## Working agreement (ongoing project)
This is a living project: the user sends a new workout (workout log + sensor recording)
after every session, and each one should leave the system better than before.
For every new workout:
1. Produce and publish its report (skill above) and commit the session.
2. Review its set windows and mark them `reviewed`, so it becomes ground truth.
3. Run `python -m vbt.evaluate --record` and compare with the previous line in
   `data/eval/history.jsonl`. Say in the reply whether log-guided detection and the
   sensor-only readiness got better or worse, and what `python -m vbt.personal` now
   adopts or still considers promising.
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
- 2026-11-05: learning loop. Attitude quaternion now stored. vbt/personal.py (evidence per
  checked session, held-out-gated adoption) and vbt/infer.py (sensor-only mode + shadow test).
  Nov 1 baseline, same-day only: 21/22 sets found, 16 named, 9 exact reps, 1 false set ->
  readiness 36%. One-day tuning of rep signals overfits (held-out error 0.64 vs generic 0.45),
  so nothing is adopted until a second checked workout. Candidates to confirm next time:
  pulldown and split squat counted from rotation y. Set-vs-noise thresholds in infer.py are
  provisional (tuned on Nov 1).
- 2026-11-06: rep-shape features (time-normalised height, rotation pattern, rotation rate per
  rep) added to exercise recognition. Nov 1, each set hidden: 20 -> 21/22 named; sensor-only:
  16 -> 17 named, readiness unchanged at 36% (8/22). Rhythm features (concentric share) hurt
  (19/22) and were dropped. Also fixed sensor-only choosing a hypothesis that then failed the
  set-vs-noise check (now only hypotheses that pass compete). Bottleneck is rep counting:
  edge cycles and -1 undercounts. Next: learned edge-cycle rejection (needs 2+ checked workouts).
- 2026-11-07: second workout (Full Body A, 26 sets). The watch paused 16x and stopped
  motion at 40 min: 61% recorded. Resampling had bridged gaps with invented motion -> gaps now
  marked (`valid`, `meta.coverage`), motion zeroed, report notice. Pushdown+curl superset
  (user) -> `--superset`, sets interleaved. Partial review (`unsure`, `not_recorded`): only 1
  set fully checked (pushdown 9), 2 curls identified but counts unconfirmed, 1 not recorded,
  22 unsure. Exercise names now grouped across apps/days (`same_exercise`): pushdown is the
  first exercise with 2 workouts of evidence; its learned signal lost on held-out sets, so
  generic stays. Detection prior learns from checked sessions only. First cross-day
  sensor-only test: 1/1 checkable set found and named, reps off by 1 (readiness 0/1).
  Also: "reps @ weight" logs, day-first dates, workout-name extraction.
  Later the same day the user confirmed the renamed exercises are the same
  (`data/aliases.json`), then chose to EXCLUDE Nov 7 from learning and scoring (unreliable
  recording): `meta.json` "exclude_from_learning" (vbt.io.is_excluded). Report and data kept;
  the code fixes stay. Ground truth is Nov 1 only again.
- 2026-11-06 (later): public benchmark added (vbt/public.py, RecoFit dataset, CDLA-P-2.0).
  It exposed a systematic -1 rep undercount: a first rep out of the unrack or a slower last
  rep broke the chain's regularity test and was dropped. Fix: extend the rep chain by one
  turnaround at each end with looser rules + reflect-pad edges (only when the window is known;
  off in sensor-only mode, where it picked up setup motion: readiness 36 -> 27%). Public: reps
  exact 13 -> 29%, within +-1 68 -> 76% (RecoFit published 93%), mean error -1.15 -> -0.40;
  recognition across people 72 -> 77% with rep shape. Nov 1 log-guided: consistent sets
  13 -> 16/22. Also: whole-word exercise matching ("row ... on bench" is a row), plural
  names, gravity check no longer rejects explosive moves. Tried and rejected: peeling
  "unusual" edge reps by shape/ROM/tempo (edge reps are real; their kinematics are just
  poorly estimated) - every variant cut exact counts.
- 2026-11-01 (later): made inputs generic. vbt/logs.py (Hevy, Strong, CSV, notes, JSON; kg/lb)
  and vbt/sensors.py (Sensor Logger watch/phone, any IMU CSV; gravity estimated from raw
  accel). Validated on the Nov 1 watch's raw accelerometer: with gyro, gravity within 0.6°
  median and velocities within 0.004 m/s of Apple's; accelerometer-only, rep counts ±1,
  velocity ~0.07 m/s worse. Sensor placement "bar" counts every lift from height.
