"""Sensor data only: find the sets, name the exercises, count the reps.

This is the "one day just give it sensor data" mode. It runs on every workout
as a *shadow test*: with the log hidden, and with personal knowledge learned
only from *other* workouts, how close does it get to what was logged? The
score is tracked by vbt.evaluate and shown in each report, so progress toward
log-free use is measured, not guessed.

Steps:
1. candidate segments: regular-rhythm runs on every rep signal (vbt.align.candidate_runs),
   overlapping runs merged;
2. recognise: nearest personal prototype (vbt.personal.Prototypes) on the segment's motion
   features, then again on the refined rep span;
3. count: that exercise's profile (learned personal profile when one is adopted, else
   generic) re-detects the reps and the set's true edges;
4. filter: keep segments that look like lifting (enough regular, clean reps, range of
   motion plausible for that exercise as you do it); unrecognised but clean, regular
   movement is kept as "unknown exercise";
5. join: consecutive segments of the same exercise a few seconds apart are one set
   (split-squat legs, a re-grip).
"""
import json
from pathlib import Path

import numpy as np

from .align import candidate_runs
from .workout import PROFILES, FEATURES, analyze_set, set_features, split_bouts, rep_shape
from . import personal as P

MIN_REPS = 3
UNKNOWN_FACTOR = 2.5      # distance beyond this x the exercises' own spread -> "unknown"
# Set-vs-noise rule. PROVISIONAL: tuned on 2026-11-01 only (keeps 21/21 real sets, drops
# 13/14 noise segments there). Re-check with every new workout; replace with a learned
# filter once checked sessions provide enough real and noise examples.
MIN_ROM_M = 0.15          # reps smaller than this are fidgeting / walking arm swing
NEAR_DIST = 1.2           # close to a known exercise's prototype
LIFT_CV, LIFT_ROM, LIFT_CLEAN = 0.25, 0.25, 0.75   # or unmistakably lift-like on its own


def is_set(dist, cv, rom, clean_frac):
    """Does a candidate look like a real set rather than walking, loading plates or fidgeting?"""
    if rom < MIN_ROM_M:
        return False
    return dist <= NEAR_DIST or (cv <= LIFT_CV and rom >= LIFT_ROM and clean_frac >= LIFT_CLEAN)


def _blocks(data, fs=25.0, gap_s=1.5):
    runs = candidate_runs(data, fs, pos_prom_m=0.08, grav_prom=0.12, min_len=4)
    bl = []
    for r in sorted(runs, key=lambda r: r.t_start):
        if bl and r.t_start <= bl[-1][1] + gap_s:
            bl[-1][1] = max(bl[-1][1], r.t_end)
        else:
            bl.append([r.t_start, r.t_end])
    return [tuple(b) for b in bl]


def _count(data, a, b, label, key, prof_by_label, placement, fs):
    prof = (prof_by_label.get(label) or {}).get("profile")
    res = analyze_set(data, a, b, key, fs=fs, placement=placement, profile=prof)
    return res


def _quality(reps):
    if len(reps) < 2:
        return 1.0, 0.0, 0.0
    rs = np.array([r["rep_s"] for r in reps])
    cv = float(rs.std() / (rs.mean() + 1e-9))
    clean = float(np.mean([r["ok"] for r in reps]))
    rom = float(np.median([r["rom"] for r in reps if r["ok"]] or [r["rom"] for r in reps]))
    return cv, clean, rom


def infer_sets(data, protos, prof_by_label=None, placement="wrist", fs=50.0, distances=None):
    """Sensor-only set list. protos: vbt.personal.Prototypes (required).

    Every known exercise is tried as a hypothesis for each candidate segment: its
    own rep signal finds the reps and the set's edges, motion features are measured
    over those reps only, and the distance to that exercise's prototype says how
    well it fits. The best lift-like hypothesis wins. (Measuring features over the
    whole segment fails: segments often include setup fidgeting.)

    distances: optional callable(features, (t0, t1), shape) -> {label: distance}, used by
    the shadow test to hide the set being judged from the prototypes.
    """
    prof_by_label = prof_by_label or {}
    distances = distances or (lambda f, span, shape=None: protos.distances(f, shape))
    out = []
    for a, b in _blocks(data):
        best = None
        for label in protos.labels:
            key = protos.keys.get(label, "auto")
            res = _count(data, a - 2, b + 2, label, key, prof_by_label, placement, fs)
            reps = res["reps"]
            if len(reps) < MIN_REPS:
                continue
            cv, clean, rom = _quality(reps)
            norm = (prof_by_label.get(label) or {}).get("rom_m")
            if not (cv <= 0.45 and clean >= 0.5 and rom >= (0.5 * norm if norm else 0.10)):
                continue
            span = (reps[0]["t_start"], reps[-1]["t_end"])
            f = set_features(data, *span)
            # rep shape helps choose WHICH exercise; the set-vs-noise gate (is_set) stays on
            # the summary-feature distance it was calibrated with
            full = distances(f, span, rep_shape(data, reps)).get(label, np.inf)
            dist = distances(f, span, None).get(label, np.inf)
            score = full + 0.5 * cv
            # only hypotheses under which the segment looks like a real set compete
            if not is_set(dist, cv, rom, clean):
                continue
            if best is None or score < best[0]:
                best = (score, label, key, res, span, dist, cv, rom)
        if best is None:
            continue
        _, label, key, res, span, dist, cv, rom = best
        reps = res["reps"]
        known = dist <= UNKNOWN_FACTOR * protos.spread
        mode = (prof_by_label.get(label) or {}).get("count_mode", "all")
        bouts_n = PROFILES.get(key, PROFILES["auto"])["bouts"]
        units = split_bouts(reps, bouts_n)
        counted = [sum(r["ok"] for r in u) if mode == "clean" else len(u) for u in units]
        out.append({"t_start": round(span[0] - 1.0, 2), "t_end": round(span[1] + 1.0, 2),
                    "label": label if known else None, "guess": label, "key": key,
                    "distance": round(dist, 3),
                    "reps": sum(counted), "cycles": len(reps), "clean_reps": sum(r["ok"] for r in reps),
                    "bouts": counted, "count_mode": mode,
                    "rep_cv": round(cv, 3), "rom_m": round(rom, 3),
                    "mean_velocity": res.get("mean_velocity")})
    # join consecutive segments of the same exercise (per-side bouts, re-grips)
    joined = []
    for s in sorted(out, key=lambda s: s["t_start"]):
        if joined and s["label"] and s["label"] == joined[-1]["label"]:
            prev = joined[-1]
            bouts = PROFILES.get(prev["key"], PROFILES["auto"])["bouts"]
            if s["t_start"] - prev["t_end"] <= (15.0 if bouts > 1 else 4.0):
                res = _count(data, prev["t_start"], s["t_end"], s["label"], s["key"], prof_by_label, placement, fs)
                reps = res["reps"]
                prev.update(t_end=s["t_end"], reps=len(reps), clean_reps=sum(r["ok"] for r in reps),
                            bouts=[len(u) for u in split_bouts(reps, bouts)])
                continue
        if joined and s["t_start"] < joined[-1]["t_end"]:
            # overlapping segments: keep the one with more clean reps
            if s["clean_reps"] > joined[-1]["clean_reps"]:
                joined[-1] = s
            continue
        joined.append(s)
    return joined


# ------------------------------------------------------------------ shadow test
def _match(inferred, truth):
    """Greedy one-to-one matching by overlap. Returns {truth_index: inferred_index}."""
    pairs = []
    for i, g in enumerate(truth):
        if g.get("t_start") is None:
            continue
        for j, s in enumerate(inferred):
            ov = min(g["t_end"], s["t_end"]) - max(g["t_start"], s["t_start"])
            if ov > 0:
                pairs.append((ov, i, j))
    used_i, used_j, m = set(), set(), {}
    for ov, i, j in sorted(pairs, reverse=True):
        if i not in used_i and j not in used_j:
            m[i] = j
            used_i.add(i)
            used_j.add(j)
    return m


def shadow_test(d, data, log_sets, truth, root, placement="wrist"):
    """Run sensor-only inference with the log hidden and score it against the log.

    Personal knowledge comes from *other* checked sessions. With none yet, it falls
    back to this session's other sets (leave-one-set-out, flagged 'same day':
    optimistic, since grip and fit don't change within a day).
    """
    sid = Path(d).name
    others = P.load_evidence(root, exclude=sid)
    prof_by_label = P.profiles(root, exclude=sid)
    mode = "other workouts"
    protos = P.Prototypes.from_root(root, exclude=sid)
    distances = None
    if protos is None:
        mine = P.load_evidence(root)
        mine = [e for e in mine if e["session"] == sid]
        if not mine:
            return {"mode": "unavailable", "note": "no checked sets yet to learn from"}
        rows = mine[0]["sets"]
        protos = P.Prototypes(rows)
        mode = "same day (leave-one-set-out)"
        tw = {r["index"]: (truth[r["index"] - 1]["t_start"], truth[r["index"] - 1]["t_end"]) for r in rows
              if truth[r["index"] - 1].get("t_start") is not None}

        cache = {}

        def distances(f, span, shape=None):
            hide = tuple(sorted(r["index"] for r in rows if r["index"] in tw
                                and min(tw[r["index"]][1], span[1]) - max(tw[r["index"]][0], span[0]) > 0))
            if hide not in cache:
                keep = [r for r in rows if r["index"] not in hide]
                cache[hide] = P.Prototypes(keep) if len({r["label"] for r in keep}) >= 2 else protos
            return cache[hide].distances(f, shape)
    inferred = infer_sets(data, protos, prof_by_label, placement, distances=distances)
    m = _match(inferred, truth)
    rows = []
    for i, (s, g) in enumerate(zip(log_sets, truth)):
        j = m.get(i)
        row = {"set": i + 1, "label": s["exercise"], "logged": s["reps"], "found": j is not None}
        if j is not None:
            x = inferred[j]
            per_side = PROFILES.get(x["key"], PROFILES["auto"])["bouts"] > 1
            units = x["bouts"] if per_side else [x["reps"]]
            err = max(abs(u - s["reps"]) for u in units)
            row.update(guess=x["guess"], known=x["label"] is not None, named=x["label"] == s["exercise"],
                       sensed=units, rep_err=err,
                       t_start=x["t_start"], t_end=x["t_end"])
        rows.append(row)
    found = [r for r in rows if r["found"]]
    n = len(rows)
    res = {"mode": mode, "n_sets": n, "trained_on_sets": sum(len(e["sets"]) for e in others),
           "found": len(found), "false_sets": len(inferred) - len(found),
           "named": sum(r["named"] for r in found),
           "reps_exact": sum(r["rep_err"] == 0 for r in found),
           "reps_within1": sum(r["rep_err"] <= 1 for r in found),
           "fully_right": sum(r["named"] and r["rep_err"] == 0 for r in found),
           "rows": rows,
           "unmatched": [{k: v for k, v in x.items() if k in ("t_start", "t_end", "guess", "reps")}
                         for j, x in enumerate(inferred) if j not in m.values()]}
    res["readiness_pct"] = round(100 * res["fully_right"] / n, 1) if n else None
    return res


def main(argv=None):
    """python -m vbt.infer <session_id> : sensor-only shadow test for one session."""
    import argparse
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    from .pipeline import load_session, logged_sets, SESSIONS
    ap = argparse.ArgumentParser(prog="python -m vbt.infer")
    ap.add_argument("sid")
    a = ap.parse_args(argv)
    d, meta, log, data = load_session(a.sid)
    truth = json.loads((d / "sets.json").read_text())["sets"]
    r = shadow_test(d, data, logged_sets(log), truth, SESSIONS,
                    (meta.get("sensor_info") or {}).get("placement", "wrist"))
    print(json.dumps({k: v for k, v in r.items() if k != "rows"}, indent=1))
    for row in r.get("rows", []):
        print(row)


if __name__ == "__main__":
    main()
