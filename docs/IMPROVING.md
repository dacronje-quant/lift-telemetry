# Improving detection and analysis

Every reviewed workout, from any sensor and any log, is a labelled example. The loop:

1. **Add a session** (`python -m vbt.pipeline new …`) and review it (`… review`), fixing
   `sets.json` where the detector was wrong, then mark it `"source": "reviewed"`.
2. **Measure**: `python -m vbt.evaluate --record`. It runs, for every reviewed session, the
   detector with an exercise model trained on the *other* sessions, and reports
   - detection recall and IoU per session,
   - robustness: a set done but not logged, and a logged set the watch can't see,
   - rep-count agreement between auto and checked windows,
   - leave-one-session-out exercise classification.
   `data/eval/history.jsonl` keeps one line per run with the commit hash.
3. **Change one thing**, re-run step 2, keep it only if it doesn't hurt any session.
4. `python -m vbt.pipeline run --all` to refresh every report.

## Where the algorithms live
| Stage | Code | Main knobs |
|---|---|---|
| Ingest / 25 Hz store | `vbt/ingest.py` | anti-alias cut-off |
| Rep detection per exercise | `vbt/workout.py` (`PROFILES`, `find_turnarounds`, `analyze_set`) | channel, rest point, bouts, `rel_prom`, plausibility flags |
| Set candidates | `vbt/align.py::candidate_runs` | prominence thresholds, run regularity |
| Log-guided alignment | `vbt/align.py::align_sets` | count / regularity / clean-fraction / ROM weights, rest limits, orientation & ROM consistency, skip costs |
| Exercise model | `vbt/library.py`, `vbt/align.py::CentroidModel` | features (`FEATURES`), temperature |
| Report | `vbt/pipeline.py`, `vbt/templates/report.html` | — |

## Baseline (2026-11-01, one reviewed session)
22/22 sets found, mean IoU 0.76; 18/21 when one set is missing from the log; inserting an
invisible machine set leaves the real sets intact but the fake set is placed on a fidget
window instead of being skipped. With one session the weights are tuned on that session:
treat the numbers as optimistic until more reviewed sessions exist.

## Known weak spots → ideas, roughly by value
1. **Sets done but not logged** shift neighbours. Idea: allow "unlogged set" as an explicit
   DP state with its own reward when a strong rhythm run sits between two logged sets.
2. **Invisible exercises** (machines, pull-ups) get placed on noise. Idea: a learned
   "lift-likeness" score (ROM, smoothness, rep regularity) from reviewed sets; skip below it.
3. **Edge cycles** (unrack, setup pull, re-rack) inflate counts. Idea: per-profile templates
   of a clean rep from the library; score each cycle by template correlation.
4. **Exercise model** is a nearest-centroid on 16 set features. With 5+ sessions: per-rep
   features, a small gradient-boosted model, and calibrated probabilities in the DP.
5. **Velocity accuracy**: compare with a known reference once (e.g. a phone-filmed bar path or
   a VBT device) to calibrate the per-exercise bias.
6. **Per-exercise MVTs** and e1RM trends once a lift has been ramped in several sessions.

7. **New sources**: the first real recording from a phone, a bar-mounted sensor or another
   IMU will need a check of sign convention, axis units and placement handling
   (`vbt/sensors.py`); add it as a test fixture once verified.
8. **New log formats**: add a parser to `vbt/logs.py` with a test whenever a new app's export
   shows up.

## Data that would help most
- Keep the watch on the same wrist and fit every session.
- For alternating dumbbell work, note in Hevy whether reps are per arm.
- Once in a while, ramp a main lift to a heavy single or double pulled at max intent: that
  anchors the load-velocity profile.
- Sensor Logger can record markers; a tap at the start of each set would give exact ground
  truth for free.
