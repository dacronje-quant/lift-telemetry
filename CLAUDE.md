# lift-telemetry (Python package: vbt)

Strength-training telemetry: any motion-sensor recording (watch, phone, IMU; wrist or bar)
plus any workout log (Hevy, Strong, CSV, typed notes, JSON) becomes `data/sessions/<id>/`
and an interactive report on sets, reps, velocity and fatigue.
Inputs are normalised by `vbt/logs.py` (logs) and `vbt/sensors.py` (sensors); everything
downstream is source-agnostic.

- **When the user shares a workout, follow `.claude/skills/workout-report/SKILL.md`.**
- Algorithm changes: **the user's own workouts decide** (goal: accurate for this user first).
  Measure with `python -m vbt.evaluate`; keep a change only if no reviewed workout gets
  worse, and record it after committing (`--record`). Refresh reports with
  `python -m vbt.pipeline run --all`.
- Public check (RecoFit: many people, true rep counts) is an overfitting alarm, not a veto.
  Run `python -m vbt.public [--record]` (~20 s) only when the shared
  rep-counting code changes: `find_turnarounds`, `position` and the filters, or the generic
  path of `analyze_set` (ghost rules, edge extension, plausibility flags). Skip it for
  exercise-specific profiles, recognition, alignment, personal learning, scoring and
  reports (it cannot move; 5 of 5 recorded runs were identical). A RecoFit drop means look
  closer, not reject: keep a change that helps on >= 2 of the user's workouts even if
  RecoFit drops a little, and write the trade-off in the log. Data: the 114 sets it uses,
  committed as `data/public/recofit_subset.pkl.xz` (9.1 MB, CDLA-P-2.0, attribution in
  `data/public/README.md`); `--data <dir> --rebuild-cache` rebuilds it from the 1.56 GB file.
- Headline accuracy is the **first look**: each workout's sensor-only score the first time
  it is recorded, before any fix is tuned on it (`data/eval/first_look.json`, written by
  `evaluate --record`). So always record right after review, before fixing anything.
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
- 2026-11-14: third workout (Full Body A, 23 sets / 7 exercises, full 65 min recording, no
  gaps). First real cross-day test. Auto windows: 13/23 right; fixed by hand and marked
  reviewed: squats shifted by one set (warm-up motion taken as set 1), deadlift sets 4-5 and
  leg curl 1 on setup motion, leg curl 2 and pulldown 1 cut in half. Hammer curls left
  `unsure` (watch arm 10 and 6 curls vs 12 and 7 logged; same 10-for-12 as Nov 1: ask user).
  Cause found for squat double counts: the 0.2 Hz position band-pass splits a slow rep into
  the real cycle and a zero-travel ghost. Fix: ghost-cycle rejection in `analyze_set`
  (`GHOST_ROM` 6% of typical travel, rest-at-top lifts, reps >= 3 s apart). Same two workouts
  before -> after: log-guided recall 83.7 -> 95.4% (Nov 14: 66.7 -> 90.5%), IoU 0.63 -> 0.73,
  sensor-only found 81 -> 86%, reps exact 23 -> 26%, readiness 9.3% unchanged; Nov 1 and
  RecoFit unchanged (28.9% / 76.3%). Readiness is now cross-day only (Nov 1: 13.6%,
  Nov 14: 4.8%); the earlier 36% was same-day and optimistic. Velocity is now hidden when
  fewer than half of a set's cycles are plausible. Open: slow deadlifts, mid-set pauses,
  leg curl, pulldown naming (see docs/IMPROVING.md). Nothing adopted by vbt.personal:
  bench, deadlift and pushdown have 2 workouts but no learned signal beats generic held-out.
- 2026-11-22: fourth workout (Full Body A, 24 sets / 7 exercises, 61 min, no gaps), reviewed.
  Auto placement 21/24; fixed by hand: both 120 kg deadlifts (found by their floor-contact
  spikes, 5 per set) and two undeclared supersets (pulldown + pushdown, hammer curl + lateral
  raise; pulldown = arm overhead while upright). Fix: ghost-cycle rejection for deadlift at
  counting time (not in evidence): log-guided recall 94.0 -> 100% (all 4 heavy deadlifts of
  Nov 14 + 22), IoU 0.756 -> 0.800, readiness 25.4 -> 26.9% (18/67), RecoFit unchanged;
  cost: Nov 14 one deadlift 5 -> 4. Found and fixed a scoring bug: evidence.json was keyed on
  sets.json mtime, so evaluate mixed evidence from old and new code (same code scored 26.9
  vs 23.9%); now keyed on content + code hash and refreshed before run/evaluate. Filter cache:
  evaluate 6.1 -> 4.3 min. `pipeline review` now also writes a whole-session overview.
  Later: the user is very unsure of the logged reps in the supersets (log 15-24) -> new
  `reps_unsure` flag (window + exercise checked, count untrusted): such sets feed detection
  and exercise prototypes but never rep-count learning, the count verdict, or exact-reps /
  readiness (denominator = sets with a trusted count: 57 of 67). Same 57 sets before/after:
  17 fully right (29.8%); the recorded 26.9 -> 29.8% is the denominator, not a gain.
  Lat Pulldown's learned gy/top setting (briefly adopted) rested on those counts and is no
  longer adopted (count evidence from 1 workout). Nothing is adopted now. Open: aliases Triceps Pushdown / Triceps Rope Pushdown and Lat Pulldown /
  Pulldown (Wide Parallel Grip) need the user's confirmation. Full approach review with a
  ranked plan: docs/REVIEW-2026-11-22.md.
- 2026-11-22 (later): public check demoted to an overfitting alarm for shared rep-counting
  code only (user: accuracy for them first); RecoFit cached as a 9.1 MB subset of the 114
  sets it uses (identical results, ~20 s vs a 1.56 GB download; not yet stored anywhere
  persistent: Drive uploads from here are limited to ~3 MB). First look added
  (`data/eval/first_look.json`, learning page squares): Nov 14 4.8% (1/21), Nov 22 42.9%
  (6/14 trusted-count sets), backfilled from the shadow tests committed with each workout.
  Restored the f19e526 scoreboard line (lost when I reset the eval files after a trial run).
- 2026-11-22 (later): user chose to commit the RecoFit subset to the repo
  (`data/public/recofit_subset.pkl.xz`, 9.1 MB, checksum-verified, attribution and licence in
  `data/public/README.md`). `python -m vbt.public` needs no download now (~20 s); rebuilding
  from the full file reproduces it byte for byte.
