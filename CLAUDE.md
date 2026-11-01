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
