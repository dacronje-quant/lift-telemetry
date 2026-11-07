"""Learn how *you* do each exercise, from every human-checked workout.

Generic exercise profiles (vbt.workout.PROFILES) guess which signal shows the
reps best and where a rep starts and ends. People differ: grip, tempo, how far
the forearm rotates, whether dumbbell work alternates. Every checked set is a
labelled example, so for each logged exercise name we keep evidence:

* rep-signal trial: for every candidate signal (wrist height, gravity on each
  sensor axis) and rest point (top/bottom), how far its rep count is from the
  logged count;
* motion prototype: the set's 16 motion features (orientation, rotation,
  translation), for recognising the exercise without a log;
* personal norms: rep duration, range of motion, forearm rotation, velocity.

Evidence lives in data/sessions/<id>/evidence.json (one file per checked
session, rebuilt when its sets.json changes). `profiles()` aggregates it, with
`exclude=<id>` for honest tests: a session is never tailored with its own sets.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .workout import PROFILES, FEATURES, analyze_set, set_features, split_bouts, exercise_key, rep_shape, same_exercise

TRUSTED = ("manual", "reviewed")
CANDIDATES = [(c, r) for c in ("pos", "gx", "gy", "gz") for r in ("top", "bottom")]
MIN_SETS = 2            # an exercise needs this many checked sets before its profile is trusted
MIN_GAIN = 0.2          # ...and must beat the generic profile by this many reps per set, held out
SHAPE_WEIGHT = 0.5      # rep-shape distance, in units of its typical within-exercise spread.
                        # 2026-11-01: +1 set named (sensor-only 16->17, hidden-set test 20->21/22),
                        # same at 0.25-2; readiness unchanged (rep counting is the bottleneck)
MIN_SESSIONS = 2        # ...using evidence from at least this many workouts (one day overfits:
                        #    on 2026-11-01 alone, held-out learned signals were worse, 0.64 vs 0.45)


def _count_error(reps, logged, bouts):
    """Reps the logged count lies outside [clean cycles, all cycles], worst unit."""
    units = split_bouts(reps, bouts) if reps else [[]]
    worst = 0
    for u in units:
        hi, lo = len(u), sum(r["ok"] for r in u)
        if not (lo <= logged <= hi):
            worst = max(worst, min(abs(logged - lo), abs(logged - hi)))
    return worst


def build_evidence(d, data, log_sets, windows, placement="wrist", fs=50.0):
    """Evidence for one checked session. d: session folder; windows: sets.json 'sets'."""
    rows = []
    for i, (s, w) in enumerate(zip(log_sets, windows)):
        if w.get("skipped") or w.get("unsure") or w.get("t_start") is None:
            continue
        key = s.get("exercise_key") or exercise_key(s["exercise"])
        base = PROFILES.get(key, PROFILES["auto"])
        trials = {}
        for ch, rest in CANDIDATES:
            prof = {**base, "channel": ch, "rest": rest}
            if placement == "bar" and ch != "pos":
                continue
            reps = analyze_set(data, w["t_start"], w["t_end"], key, fs=fs, placement=placement,
                               profile=prof)["reps"]
            trials[f"{ch}/{rest}"] = {"err": _count_error(reps, s["reps"], base["bouts"]), "n": len(reps)}
        gen = analyze_set(data, w["t_start"], w["t_end"], key, fs=fs, placement=placement)
        reps = gen["reps"]
        span = (reps[0]["t_start"], reps[-1]["t_end"]) if reps else (w["t_start"], w["t_end"])
        f = set_features(data, *span)
        ok = [r for r in reps if r["ok"]] or reps
        med = lambda k: float(np.median([r[k] for r in ok])) if ok else None  # noqa: E731
        rows.append({"index": i + 1, "label": s["exercise"], "key": key, "kg": s.get("kg"),
                     "reps": s["reps"], "rpe": s.get("rpe"), "bouts": base["bouts"],
                     "generic": f"{base['channel']}/{base['rest']}" if base["channel"] != "auto" else None,
                     "trials": trials, "features": [None if not np.isfinite(f[k]) else round(float(f[k]), 5)
                                                    for k in FEATURES],
                     "shape": rep_shape(data, reps) if reps else None,
                     "n_all": len(reps), "n_clean": sum(r["ok"] for r in reps),
                     "rep_s": med("rep_s"), "rom_m": med("rom"), "rotation_deg": med("rotation_deg"),
                     "mean_velocity": gen.get("mean_velocity"), "velocity_loss_pct": gen.get("velocity_loss_pct")})
    ev = {"session": d.name, "placement": placement, "sets_mtime": (d / "sets.json").stat().st_mtime,
          "features": FEATURES, "sets": rows}
    (d / "evidence.json").write_text(json.dumps(ev, indent=0))
    return ev


def load_evidence(root="data/sessions", exclude=None):
    out = []
    for p in sorted(Path(root).glob("*/evidence.json")):
        if p.parent.name == exclude:
            continue
        sj = p.parent / "sets.json"
        if not sj.exists() or json.loads(sj.read_text()).get("source") not in TRUSTED:
            continue
        out.append(json.loads(p.read_text()))
    # one exercise, many names (apps and days differ): group under the first name seen
    names = []
    for ev in out:
        for r in ev["sets"]:
            canon = next((c for c in names if same_exercise(c, r["label"])), None)
            if canon is None:
                names.append(r["label"])
            elif canon != r["label"]:
                r["name"], r["label"] = r["label"], canon
    return out


def lookup(learned, label):
    """learned[label], also when the exercise was learned under another name."""
    if label in learned:
        return learned[label]
    return next((v for k, v in learned.items() if same_exercise(k, label)), None)


def evidence_is_stale(d):
    e, s = d / "evidence.json", d / "sets.json"
    if not e.exists():
        return True
    try:
        return json.loads(e.read_text()).get("sets_mtime") != s.stat().st_mtime
    except Exception:
        return True


def profiles(root="data/sessions", exclude=None):
    """Per logged exercise name: learned profile, prototype and norms.

    Returns {label: {...}}. 'profile' is set only when the evidence clearly
    beats the generic profile (>= MIN_SETS sets, >= MIN_GAIN reps better).
    """
    by = defaultdict(list)
    sessions = defaultdict(set)
    for ev in load_evidence(root, exclude):
        for r in ev["sets"]:
            by[r["label"]].append(r)
            sessions[r["label"]].add(ev["session"])
    out = {}
    for label, rows in by.items():
        key = rows[0]["key"]
        base = PROFILES.get(key, PROFILES["auto"])
        combos = sorted({c for r in rows for c in r["trials"]})
        mean_err = {c: float(np.mean([r["trials"][c]["err"] for r in rows if c in r["trials"]])) for c in combos}
        exact = {c: float(np.mean([r["trials"][c]["err"] == 0 for r in rows if c in r["trials"]])) for c in combos}
        generic = rows[0]["generic"]
        best = min(combos, key=lambda c: (mean_err[c], -exact[c], c != generic))
        gen_err = mean_err.get(generic) if generic else None
        held_gen, held_learned = _held_out(rows, generic)
        # Adopt only what generalises: evidence from >= 2 workouts, and choosing the
        # signal on the other sets must beat the generic one on held-out sets.
        adopt = (len(rows) >= MIN_SETS and len(sessions[label]) >= MIN_SESSIONS and best != generic
                 and held_learned is not None
                 and (held_gen is None or held_learned <= held_gen - MIN_GAIN))
        # Without a log, should edge cycles (setup pull, unrack) count as reps? Learn it from
        # how your logged counts compare with all vs clean cycles, once >= 2 workouts agree.
        cm = [r for r in rows if r.get("n_all") is not None]
        per = rows[0]["bouts"] if rows[0].get("bouts") else 1
        err_all = float(np.mean([abs(r["n_all"] / per - r["reps"]) for r in cm])) if cm else None
        err_clean = float(np.mean([abs(r["n_clean"] / per - r["reps"]) for r in cm])) if cm else None
        count_mode = "all"
        if cm and len(sessions[label]) >= MIN_SESSIONS and err_clean < err_all - MIN_GAIN:
            count_mode = "clean"
        X = np.array([[np.nan if v is None else v for v in r["features"]] for r in rows], float)
        info = {"key": key, "n_sets": len(rows), "n_sessions": len(sessions[label]),
                "generic_signal": generic, "generic_err": gen_err,
                "best_signal": best, "best_err": mean_err[best], "best_exact": exact[best],
                "held_out_err_generic": held_gen, "held_out_err_learned": held_learned,
                "learned": bool(adopt),
                "count_mode": count_mode, "count_err_all": err_all, "count_err_clean": err_clean,
                "prototype": {"mean": np.nanmean(X, 0).round(5).tolist(), "n": len(rows)},
                "rep_s": _med(rows, "rep_s"), "rom_m": _med(rows, "rom_m"),
                "rotation_deg": _med(rows, "rotation_deg")}
        if adopt:
            ch, rest = best.split("/")
            info["profile"] = {**base, "channel": ch, "rest": rest}
        out[label] = info
    return out


def _held_out(rows, generic):
    """Leave-one-set-out: pick the best signal on the other sets, score it on the held-out one.

    Returns (mean held-out error of the generic signal, of the learn-then-apply rule).
    Sets are grouped by session so a held-out set's own session never chooses for it
    when other sessions exist.
    """
    if len(rows) < 2:
        return None, None
    eg, el = [], []
    for i, r in enumerate(rows):
        rest = [x for j, x in enumerate(rows) if j != i]
        combos = [c for c in r["trials"] if all(c in x["trials"] for x in rest)]
        if not combos:
            continue
        m = {c: np.mean([x["trials"][c]["err"] for x in rest]) for c in combos}
        pick = min(combos, key=lambda c: (m[c], c != generic))
        if generic in r["trials"]:
            eg.append(r["trials"][generic]["err"])
        el.append(r["trials"][pick]["err"])
    return (float(np.mean(eg)) if eg else None), (float(np.mean(el)) if el else None)


def _med(rows, k):
    v = [r[k] for r in rows if r.get(k) is not None]
    return round(float(np.median(v)), 3) if v else None


class Prototypes:
    """Nearest-prototype exercise recogniser over personal evidence, with 'unknown' rejection.

    Standardisation uses the spread of all evidence sets; a set is 'unknown'
    when its distance to the nearest prototype is far beyond how far that
    exercise's own sets sit from it.
    """

    def __init__(self, rows):
        X = np.array([[np.nan if v is None else v for v in r["features"]] for r in rows], float)
        X = np.where(np.isnan(X), np.nanmean(X, 0), X)
        # spread floor: with few sets, a feature that barely varies would otherwise turn
        # tiny differences into huge distances and drown out what matters (orientation)
        self.mu, self.sd = X.mean(0), np.maximum(X.std(0), 1e-3 + 0.1 * np.abs(X.mean(0)))
        Z = (X - self.mu) / self.sd
        labels = [r["label"] for r in rows]
        self.labels = sorted(set(labels))
        self.keys = {r["label"]: r["key"] for r in rows}
        self.cent = np.array([Z[[j for j, l in enumerate(labels) if l == c]].mean(0) for c in self.labels])
        # rep-shape prototypes (rows from evidence built before shapes existed have none)
        self.shape = {}
        for c in self.labels:
            sh = [r["shape"] for r in rows if r["label"] == c and r.get("shape")]
            if sh:
                self.shape[c] = np.mean(sh, 0)
        sh_err = [np.sqrt(((np.array(r["shape"]) - self.shape[r["label"]]) ** 2).mean())
                  for r in rows if r.get("shape") and r["label"] in self.shape]
        self.shape_scale = max(float(np.median(sh_err)), 1e-3) if sh_err else 1.0
        own = [np.sqrt(((Z[j] - self.cent[self.labels.index(labels[j])]) ** 2).mean())
               + self._shape_term(rows[j].get("shape"), labels[j]) for j in range(len(labels))]
        self.spread = float(np.percentile(own, 95)) if own else 1.0

    def _shape_term(self, shape, label):
        if shape is None or label not in self.shape:
            return 0.0
        return SHAPE_WEIGHT * float(np.sqrt(((np.array(shape) - self.shape[label]) ** 2).mean())) / self.shape_scale

    def distances(self, feats, shape=None):
        """feats: dict of FEATURES (+ optional rep shape) -> {label: distance to its prototype}."""
        z = (np.nan_to_num(np.array([feats[k] for k in FEATURES], float)) - self.mu) / self.sd
        d = np.sqrt(((self.cent - z) ** 2).mean(1))
        return {c: float(d[i]) + self._shape_term(shape, c) for i, c in enumerate(self.labels)}

    def classify(self, feats):
        """feats: dict of FEATURES -> (label or None, distance, margin to runner-up)."""
        z = (np.nan_to_num(np.array([feats[k] for k in FEATURES], float)) - self.mu) / self.sd
        d = np.sqrt(((self.cent - z) ** 2).mean(1))
        o = np.argsort(d)
        margin = float(d[o[1]] - d[o[0]]) if len(o) > 1 else float("inf")
        return self.labels[o[0]], float(d[o[0]]), margin

    @classmethod
    def from_root(cls, root="data/sessions", exclude=None, min_sets=1):
        rows = [r for ev in load_evidence(root, exclude) for r in ev["sets"]]
        if len({r["label"] for r in rows}) < 2:
            return None
        return cls(rows)


def _f(x):
    return "–" if x is None else f"{x:.2f}"


def main(argv=None):
    """python -m vbt.personal : print what has been learned so far."""
    import argparse
    ap = argparse.ArgumentParser(prog="python -m vbt.personal")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1] / "data/sessions"))
    a = ap.parse_args(argv)
    P = profiles(a.root)
    if not P:
        print("no checked sessions with evidence yet")
    for label, p in sorted(P.items()):
        print(f"{label:38s} {p['n_sets']:3d} sets/{p['n_sessions']} sess  generic {p['generic_signal']}"
              f" err {_f(p['generic_err'])}  best {p['best_signal']} err {p['best_err']:.2f}"
              f" exact {p['best_exact']:.0%} | held-out generic {_f(p['held_out_err_generic'])} learned"
              f" {_f(p['held_out_err_learned'])}  {'LEARNED' if p['learned'] else ''}")


if __name__ == "__main__":
    main()
