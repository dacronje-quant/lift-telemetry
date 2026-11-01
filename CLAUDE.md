# Statistics / vbt

Apple Watch velocity-based-training analysis. Workouts arrive as a Hevy share text plus a
Sensor Logger export; each becomes `data/sessions/<id>/` and an interactive report.

- **When the user shares a workout, follow `.claude/skills/workout-report/SKILL.md`.**
- Algorithm changes: measure with `python -m vbt.evaluate` (see `docs/IMPROVING.md`) and
  refresh reports with `python -m vbt.pipeline run --all`.
- Tests: `python -m pytest -q tests` (fast, synthetic data only).
- Never commit raw Sensor Logger zips; the 25 Hz `sensor_25hz.csv.gz` is the stored copy.
- `sets.json` `source`: `auto` (detector), `reviewed`/`manual` (human-checked ground truth).
  Only mark windows reviewed after actually checking them.

## Working agreement (ongoing project)
This is a living project: the user sends a new workout (Hevy text + Sensor Logger export)
after every session, and each one should leave the system better than before.
For every new workout:
1. Produce and publish its report (skill above) and commit the session.
2. Review its set windows and mark them `reviewed`, so it becomes ground truth.
3. Run `python -m vbt.evaluate --record` and compare with the previous line in
   `data/eval/history.jsonl`. Say in the reply whether detection got better or worse.
4. Fix at least one weakness the new data exposed (new exercise profile, a missed set,
   a miscounted rep pattern, a template issue) or take one item from `docs/IMPROVING.md`,
   keeping it only if the scoreboard holds on all reviewed sessions. Then
   `python -m vbt.pipeline run --all` so older reports pick up the improvement.
5. Add a dated line to the log below: what was learned and what changed.

## Log
- 2026-11-01: first session (Full Body C, 22 sets / 7 exercises). Built the pipeline,
  log-guided set detection (22/22 on this session), cross-session library and eval harness.
  Weak spots: sets done but not logged (18/21 placed), invisible exercises placed on
  noise. Exercise model across sessions needs a second workout.
