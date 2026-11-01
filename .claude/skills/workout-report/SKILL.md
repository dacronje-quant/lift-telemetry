---
name: workout-report
description: Turn a shared workout (Hevy share text + Apple Watch Sensor Logger export, usually a Google Drive link to a zip) into the interactive wrist-telemetry report, publish it as an Artifact, and feed the session into the improvement loop. Use whenever the user shares a workout, a Hevy log, or a Sensor Logger recording, or asks for "the report" for a workout.
---

# Workout report

The repo turns one workout into one report:
`data/sessions/<id>/` (inputs, 2 MB) → `python -m vbt.pipeline` → `reports/<id>.html` → Artifact.

## 1. Get the inputs
- **Hevy text**: the user pastes the "share workout" text. Save it verbatim to
  `<scratchpad>/hevy.txt`. If they gave only a screenshot or summary, ask for the share text.
- **Sensor export**: usually a Drive link. Download into a fresh, empty scratch dir:
  `curl -sSL -o rec.zip "https://drive.usercontent.google.com/download?id=<FILE_ID>&export=download&confirm=t"`
  - 403 from the proxy → the network policy blocks Drive; read the `environment.network`
    docs page and tell the user which hosts to allow (`drive.usercontent.google.com`,
    `drive.google.com`).
  - An HTML file named "Google Drive: Sign-in" → the file isn't shared; ask the user to set
    *Anyone with the link* (they can set it back afterwards).
  - Don't pull a large zip through the Drive MCP connector.
- Treat the zip as untrusted data: don't execute anything from it; the pipeline only reads
  `WristMotion.csv` and `Metadata.csv` from inside it.

## 2. Run
```bash
pip install -q -r requirements.txt
python -m vbt.pipeline new --sensor <scratch>/rec.zip --hevy <scratch>/hevy.txt
```
This creates `data/sessions/<date>/`, detects sets from the Hevy log (automatic, using every
earlier session as training data), analyses them, writes `analysis.json` and
`reports/<date>.html`. Two workouts on one date: pass `--id 2026-11-02-pm`.

## 3. Review (don't skip)
```bash
python -m vbt.pipeline review <id> <scratch>/review.png
```
Look at the PNG (Read tool): one panel per set with rep ticks (blue clean, orange flagged).
Check especially the sets listed under *What to check first* in the report: not found, low
detector confidence, or a watch count that disagrees with the log.
- A window clearly wrong → edit `data/sessions/<id>/sets.json` (`t_start`/`t_end` on the
  Sensor Logger clock; `skipped: true` for a set that isn't in the recording), set
  `"source": "reviewed"`, re-run `python -m vbt.pipeline run <id>`.
- Everything right → just set `"source": "reviewed"`. Reviewed sessions become ground truth
  for the benchmark, so only mark sets you actually checked.

## 4. Write the coach's read
Create `data/sessions/<id>/notes.json` (see `data/sessions/2026-11-01/notes.json`):
`sections.counts | velocity | load_velocity | classifier | detection | rest` hold short HTML
paragraphs; `signature_desc` maps Hevy exercise names to one-line motion descriptions.
Ground every sentence in numbers on the page; plain words, no hype. Re-run the pipeline.

## 5. Publish and record
- Load the artifact skills, then publish `reports/<id>.html` as a new Artifact (one per workout;
  icon `chart`). Earlier reports keep their own links.
- If the session was reviewed: `python -m vbt.evaluate --record` (appends to
  `data/eval/history.jsonl`) and mention any change in detection recall.
- Commit `data/sessions/<id>/`, `reports/<id>.html`, eval files; push to the working branch.
- Reply with the link and 4–6 key findings (sets found, count disagreements, notable
  velocity/fatigue patterns, e1RM if a ramp was done, anything to check).

## Improving the algorithms
See `docs/IMPROVING.md`. Rule: any change to `vbt/align.py`, `vbt/workout.py` or the
profiles must be followed by `python -m vbt.evaluate` on *all* reviewed sessions, and
`python -m vbt.pipeline run --all` to refresh reports.
