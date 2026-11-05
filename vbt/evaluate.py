"""Benchmark the algorithms against every human-checked session.

    python -m vbt.evaluate            # print the scoreboard
    python -m vbt.evaluate --record   # also append it to data/eval/history.jsonl

Ground truth = sessions whose sets.json source is "manual" or "reviewed".
Metrics (each session evaluated with models trained on the *other* sessions):

* detection: share of logged sets placed in the right window, mean IoU;
* robustness: the same with (a) one logged set removed (a set done but not
  logged) and (b) a fake machine set inserted (logged but invisible);
* rep counts: |auto-window count - truth-window count|, and the share of sets
  whose verdict vs. the logged count is match/ambiguous;
* sensor only (the end goal): log hidden, personal knowledge from the other
  sessions only -> sets found, named, reps exact, "fully right" share. With a
  single checked session it falls back to same-day leave-one-set-out (flagged);
* classifier: leave-one-session-out accuracy over logged exercise names
  (needs >= 2 sessions sharing exercises).

Record a line after every algorithm change; the history shows whether a
change helped on *all* sessions, not just the one being looked at.
"""
import argparse
import json
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .align import detect_sets, evaluate_alignment
from .workout import analyze_set, count_verdict, PROFILES, FEATURES
from . import library
from .pipeline import SESSIONS, load_session, logged_sets, TRUSTED, ROOT
from .infer import shadow_test

EVAL = ROOT / "data/eval"


def trusted_sessions(root=SESSIONS):
    out = []
    for p in sorted(Path(root).glob("*/sets.json")):
        if json.loads(p.read_text()).get("source") in TRUSTED:
            out.append(p.parent.name)
    return out


def eval_session(sid, root=SESSIONS, stress=True):
    d, meta, log, data = load_session(sid, root)
    logged = logged_sets(log)
    truth = json.loads((d / "sets.json").read_text())["sets"]
    prior = library.exercise_model(library.load_rows(root, exclude=sid))
    found, _ = detect_sets(data, logged, prior_model=prior)
    ev = evaluate_alignment(found, truth)
    count_diff, verdict_ok = [], []
    for h, f, g in zip(logged, found, truth):
        if f["skipped"]:
            continue
        a = analyze_set(data, g["t_start"], g["t_end"], h["exercise_key"])
        b = analyze_set(data, f["t_start"], f["t_end"], h["exercise_key"])
        count_diff.append(abs(a["rep_count"] - b["rep_count"]))
        prof = PROFILES.get(h["exercise_key"], PROFILES["auto"])
        v = count_verdict(a["reps"], h["reps"], prof["bouts"])[0] if a["reps"] else "watch fewer"
        verdict_ok.append(v in ("match", "ambiguous"))
    res = {"session": sid, "n_sets": len(logged), "prior_model": prior is not None,
           "recall": round(ev["recall"], 3), "mean_iou": round(ev["mean_iou"], 3),
           "missed": [r["set"] for r in ev["rows"] if not r["found"]],
           "rep_count_mae": round(float(np.mean(count_diff)), 3) if count_diff else None,
           "verdict_consistent": round(float(np.mean(verdict_ok)), 3) if verdict_ok else None}
    if stress and len(logged) > 3:
        k = len(logged) // 3            # a set done but not logged
        lg = [s for i, s in enumerate(logged) if i != k]
        tr = [s for i, s in enumerate(truth) if i != k]
        f2, _ = detect_sets(data, lg, prior_model=prior)
        res["recall_unlogged_set"] = round(evaluate_alignment(f2, tr)["recall"], 3)
        j = len(logged) // 2            # a logged set the watch can't see
        fake = {"exercise": "Leg Extension (Machine)", "label": "Leg Extension (Machine)",
                "exercise_key": "machine", "reps": 12, "kg": 40.0, "rpe": None, "warmup": False}
        lg = logged[:j] + [fake] + logged[j:]
        f3, _ = detect_sets(data, lg, prior_model=prior)
        res["recall_invisible_set"] = round(evaluate_alignment(f3[:j] + f3[j + 1:], truth)["recall"], 3)
        res["invisible_set_skipped"] = bool(f3[j]["skipped"])
    sh = shadow_test(d, data, logged, truth, root, (meta.get("sensor_info") or {}).get("placement", "wrist"))
    res["sensor_only"] = {k: v for k, v in sh.items() if k not in ("rows", "unmatched")}
    return res


def eval_classifier(root=SESSIONS):
    rows = library.load_rows(root)
    sessions = sorted({r["session"] for r in rows})
    if len(sessions) < 2:
        return {"note": "needs >= 2 analysed sessions", "n_sessions": len(sessions)}
    hits, n = 0, 0
    per = {}
    for sid in sessions:
        model = library.exercise_model([r for r in rows if r["session"] != sid])
        if model is None:
            continue
        test = [r for r in rows if r["session"] == sid and r["label"] in model.labels
                and None not in r["features"]]
        ok = 0
        for r in test:
            lp = model.log_prob(r["features"])
            ok += max(lp, key=lp.get) == r["label"]
        per[sid] = {"n": len(test), "acc": round(ok / len(test), 3) if test else None}
        hits += ok
        n += len(test)
    return {"n_sessions": len(sessions), "n_test_sets": n,
            "acc": round(hits / n, 3) if n else None, "per_session": per}


def scoreboard(root=SESSIONS, stress=True):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ses = [eval_session(s, root, stress) for s in trusted_sessions(root)]
    tot = sum(s["n_sets"] for s in ses) or 1
    agg = {"recall": round(sum(s["recall"] * s["n_sets"] for s in ses) / tot, 3) if ses else None,
           "mean_iou": round(sum(s["mean_iou"] * s["n_sets"] for s in ses) / tot, 3) if ses else None}
    so = [s["sensor_only"] for s in ses if s.get("sensor_only", {}).get("n_sets")]
    n = sum(x["n_sets"] for x in so) or 1
    sensor_only = {"fully_right_pct": round(100 * sum(x["fully_right"] for x in so) / n, 1) if so else None,
                   "found_pct": round(100 * sum(x["found"] for x in so) / n, 1) if so else None,
                   "named_pct": round(100 * sum(x["named"] for x in so) / n, 1) if so else None,
                   "reps_exact_pct": round(100 * sum(x["reps_exact"] for x in so) / n, 1) if so else None,
                   "false_sets": sum(x["false_sets"] for x in so),
                   "modes": sorted({x["mode"] for x in so})}
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = None
    return {"when": datetime.now(timezone.utc).isoformat(timespec="seconds"), "commit": commit,
            "n_trusted_sessions": len(ses), "detection": agg, "sensor_only": sensor_only, "sessions": ses,
            "classifier_loso": eval_classifier(root)}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m vbt.evaluate")
    ap.add_argument("--record", action="store_true", help="append to data/eval/history.jsonl")
    ap.add_argument("--no-stress", action="store_true")
    a = ap.parse_args(argv)
    sb = scoreboard(stress=not a.no_stress)
    print(json.dumps(sb, indent=1))
    if a.record:
        EVAL.mkdir(parents=True, exist_ok=True)
        with open(EVAL / "history.jsonl", "a") as f:
            f.write(json.dumps(sb) + "\n")
        (EVAL / "latest.json").write_text(json.dumps(sb, indent=1))


if __name__ == "__main__":
    main()
