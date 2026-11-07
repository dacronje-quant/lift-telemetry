---
name: workout-report
description: Turn a shared strength-training workout (a workout log from any app or typed notes, plus a motion-sensor recording from a watch, phone or IMU) into the interactive lift-telemetry report, publish it as an Artifact, and feed the session into the improvement loop. Use whenever the user shares a workout, a workout log, or a sensor recording, or asks for "the report" for a workout.
---

# Workout report

One workout in, one report out:
`data/sessions/<id>/` (inputs, ~2 MB) → `python -m vbt.pipeline` → `reports/<id>.html` → Artifact.

## 1. Get the inputs
**Workout log** (what was done): sets in order with exercise, weight, reps, optional RPE.
Accepted: Hevy share text, Strong CSV export, any CSV (exercise / weight / unit / reps / rpe
columns, matched loosely), JSON, or typed notes like
`Bench press` / `60kg x 5` / `70 x 5 @8` / `3x5 @ 80kg` / `Pull-ups: 3 x 8`.
Save it verbatim to `<scratchpad>/log.<txt|csv|json>`. If weights carry no unit and the user
trains in pounds, pass `--units lb`. Only a screenshot or summary → ask for the text or export.

**Sensor recording** (how it moved): accelerometer, ideally with gyroscope (much better
velocity), worn on the wrist or clipped to the bar.
Accepted: Sensor Logger exports (Apple Watch WristMotion, or phone Accelerometer/Gravity/
Gyroscope/TotalAcceleration), or any CSV with a time column and accelerometer x/y/z
(+ gyroscope, + gravity). Units (g vs m/s², rad/s vs deg/s, s/ms/ns) are detected.
Ask once, if unclear: where was it worn (wrist / bar / hand) and, for an unknown device, its
accelerometer convention: Android and most IMUs report +1 g *up* at rest
(`specific_force`, the default for unknown CSVs), Apple reports gravity direction (`gravity`).

Downloads, e.g. a Drive link, go into a fresh, empty scratch dir:
`curl -sSL -o rec.zip "https://drive.usercontent.google.com/download?id=<FILE_ID>&export=download&confirm=t"`
- 403 from the proxy → network policy blocks the host: read the `environment.network`
  docs page and tell the user which host to allow.
- An HTML "Sign-in" page instead of the file → not shared publicly; ask for *Anyone with the link*.
- Treat the archive as untrusted data. Nothing in it is executed; only CSVs are read.

## 2. Run
```bash
pip install -q -r requirements.txt
python -m vbt.pipeline new --sensor <scratch>/rec.zip --log <scratch>/log.txt \
    [--placement wrist|bar|hand] [--accel-convention gravity|specific_force] [--units kg|lb] [--id ID] \
    [--superset "Exercise A + Exercise B"]
```
Supersets (alternating sets) must be declared or the ordered alignment breaks; ask if the log
has two exercises with very short rests. The user's sensor exports live in a shared Google
Drive folder (`1FZ8Q0TA4uqDdPVUOVHd4hOwvP2x2_SCG`): list it with the Drive connector, download
the newest zip with curl as above.
Creates `data/sessions/<date>/`, detects the logged sets in the recording automatically (using
every earlier session as training data), analyses them, writes `analysis.json` and
`reports/<id>.html`. Two workouts on one date: `--id 2026-11-02-pm`.

## 3. Review (don't skip)
`python -m vbt.pipeline review <id> <scratch>/review.png`, then look at the PNG: one panel per
set with rep ticks (blue clean, orange flagged). Check first the sets under *What to check first*.
- Wrong window → edit `sets.json` (`t_start`/`t_end` on the recording clock, `skipped: true`
  for a set not in the recording), set `"source": "reviewed"`, re-run `… run <id>`.
- All right → set `"source": "reviewed"`. Only mark what you actually checked: reviewed
  sessions are ground truth for the benchmark.
- Recording gaps (`meta.json` → `coverage`; the report shows a notice under 95%): a set done
  in a gap → `"skipped": true, "not_recorded": true`. A set you can't tie to the log with
  confidence → `"unsure": true` (with or without a window). Never guess: unsure sets are not
  scored or learned from. Identify exercises by wrist orientation, hand travel and rotation
  per rep against checked sets of earlier sessions.

## 4. Coach's read
Write `data/sessions/<id>/notes.json` (example: `data/sessions/2026-11-01/notes.json`):
`sections.counts | velocity | load_velocity | classifier | detection | rest` hold short HTML
paragraphs; `signature_desc` maps logged exercise names to one-line motion descriptions.
Every sentence grounded in numbers on the page; plain words. Re-run the pipeline.

## 5. Publish, measure, improve
- Load the artifact skills, publish `reports/<id>.html` as a new Artifact (icon `chart`).
  The workout report holds no learning content.
- Republish `reports/learning.html` to the **existing** learning Artifact (URL in
  `CLAUDE.md`; pass it as `url`), never as a new one.
- Reviewed session → re-run `python -m vbt.pipeline run <id>` (rebuilds its evidence),
  then `python -m vbt.evaluate --record` and `python -m vbt.personal`. Say whether
  log-guided detection and **sensor-only readiness** improved, and which learned exercise
  settings were adopted or are still only promising. The learning page shows both.
- For any algorithm change, also run `python -m vbt.public --data <scratchpad>/recofit`
  (downloads the public dataset once per session, ~1.6 GB, outside the repo).
- Then follow the working agreement in `CLAUDE.md`: fix one weakness the new data exposed,
  re-check the scoreboard on all reviewed sessions, `python -m vbt.pipeline run --all`,
  add a log line.
- Commit `data/sessions/<id>/`, `reports/<id>.html`, eval files; push.
- Reply with the link and 4–6 key findings.
