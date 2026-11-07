"""Cross-session memory: every analysed session contributes labelled sets.

Each session folder holds analysis.json (compact: per-set features, loads,
velocities, window source). The library reads them all so that

* the set aligner gets an exercise classifier trained on *other* sessions,
* load-velocity profiles and e1RM can be tracked over time,
* the evaluation harness can do leave-one-session-out tests.
"""
import json
from pathlib import Path

import numpy as np

from .io import is_excluded
from .workout import FEATURES, canonical_name
from .align import CentroidModel

TRUSTED = ("manual", "reviewed")


def load_rows(root="data/sessions", exclude=None, min_source=None):
    """All analysed sets as flat dicts. exclude: session id to leave out.

    min_source="trusted" keeps only sessions whose windows were checked by a human.
    """
    rows = []
    for p in sorted(Path(root).glob("*/analysis.json")):
        sid = p.parent.name
        if sid == exclude or is_excluded(p.parent):
            continue
        a = json.loads(p.read_text())
        if min_source == "trusted" and a.get("windows_source") not in TRUSTED:
            continue
        for s in a["sets"]:
            if s.get("skipped") or not s.get("features") or (min_source == "trusted" and s.get("unsure")):
                continue
            rows.append({"session": sid, "date": a.get("date"), "label": canonical_name(s["label"]),
                         "key": s["key"], "kg": s["kg"], "reps": s["logged_reps"],
                         "rpe": s.get("rpe"), "warmup": s.get("warmup"),
                         "best_velocity": s.get("best_velocity"),
                         "mean_velocity": s.get("mean_velocity"),
                         "features": [s["features"].get(k) for k in FEATURES],
                         "source": a.get("windows_source")})
    return rows


def exercise_model(rows, min_sets_per_label=2):
    """Centroid model over logged exercise names, or None if too little data."""
    from collections import Counter
    cnt = Counter(r["label"] for r in rows)
    rows = [r for r in rows if cnt[r["label"]] >= min_sets_per_label and None not in r["features"]]
    labels = {r["label"] for r in rows}
    if len(labels) < 2:
        return None
    X = np.array([r["features"] for r in rows], float)
    return CentroidModel(X, [r["label"] for r in rows])


def history(rows):
    """Per exercise, per session: top load, best velocity at that load, sets, volume."""
    out = {}
    for r in rows:
        e = out.setdefault(r["label"], {})
        s = e.setdefault(r["session"], {"date": r["date"], "sets": 0, "volume": 0.0,
                                        "top_kg": 0.0, "v_at_top": None, "points": []})
        s["sets"] += 1
        s["volume"] += (r["kg"] or 0) * (r["reps"] or 0)
        if r["best_velocity"]:
            s["points"].append([r["kg"], r["best_velocity"]])
        if (r["kg"] or 0) > s["top_kg"]:
            s["top_kg"], s["v_at_top"] = r["kg"], r["best_velocity"]
        elif r["kg"] == s["top_kg"] and r["best_velocity"]:
            s["v_at_top"] = max(s["v_at_top"] or 0, r["best_velocity"])
    return out
