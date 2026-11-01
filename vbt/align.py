"""Log-guided set detection: find where each logged set happened in the recording.

Gym rest is not quiet (walking, loading plates, phone), so "still wrist = rest"
fails. Instead:

1. **Candidates.** Runs of regular, similar-sized turnarounds in any of four
   rep signals (vertical position, gravity on each watch axis), both
   polarities, kept separately (they may overlap). Candidate windows are single
   runs or pairs of nearby runs (a split squat's two legs, a re-grip).
   Many candidates are walking, plate loading or fidgeting.
2. **Alignment.** Ordered dynamic programming assigns the logged sets, in
   order, to non-overlapping windows. Unused windows are noise. A logged set
   may be skipped (an exercise the watch can't see) at a cost.
3. **Score** of "logged set k happened in window W":
   * the rep count the exercise's own profile finds in W vs. the logged count,
   * how regular the rhythm is,
   * optionally log P(exercise | motion features) from a classifier trained
     on past sessions or on a first pass of this one (self-training).

Everything is deterministic and runs in well under a minute for a 1 h session.
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.signal import find_peaks

from .core import vertical_acceleration
from .workout import (PROFILES, resample, position, _bp, analyze_set, set_features,
                      FEATURES, split_bouts, profile_for)


@dataclass
class Run:
    t_start: float
    t_end: float
    n_turns: int
    channel: str


def _runs(sig, t, fs, min_prom, min_len=3):
    pk, pr = find_peaks(sig, prominence=min_prom, distance=int(0.8 * fs))
    if len(pk) < min_len:
        return []
    am = pr["prominences"]
    out, cur = [], [0]
    for i in range(1, len(pk)):
        dt = (pk[i] - pk[cur[-1]]) / fs
        ok = 0.9 <= dt <= 7.0 and 0.4 < am[i] / am[cur[-1]] < 2.5
        if ok and len(cur) >= 2:
            ok = 0.5 < dt / ((pk[cur[-1]] - pk[cur[-2]]) / fs) < 2.0
        if ok:
            cur.append(i)
        else:
            out.append(cur)
            cur = [i]
    out.append(cur)
    res = []
    for c in out:
        if len(c) < min_len:
            continue
        per = float(np.median(np.diff(t[pk[c]])))
        res.append((float(t[pk[c[0]]] - 0.6 * per), float(t[pk[c[-1]]] + 0.6 * per), len(c)))
    return res


def candidate_runs(data, fs=25.0, pos_prom_m=0.08, grav_prom=0.12, min_len=3, dedupe_iou=0.6):
    """Regular-rhythm runs on every rep signal and polarity (overlaps allowed).

    Near-duplicates (IoU > dedupe_iou) keep the run with more turnarounds.
    """
    d = resample(data, data["t"][0], data["t"][-1], fs, pad=0)
    t = d["t"]
    g_hat = d["grav"] / np.linalg.norm(d["grav"], axis=1, keepdims=True)
    sigs = {"pos": (position(vertical_acceleration(d["acc"], d["grav"]), fs), pos_prom_m)}
    for i, ax in enumerate("xyz"):
        sigs["g" + ax] = (_bp(g_hat[:, i], fs, 0.1, 2.0), grav_prom)
    runs = []
    for name, (s, prom) in sigs.items():
        for sign in (1, -1):
            runs += [Run(a, b, n, name) for a, b, n in _runs(sign * s, t, fs, prom, min_len)]
    runs.sort(key=lambda r: -r.n_turns)
    kept = []
    for r in runs:
        if all(_iou(r, k) <= dedupe_iou for k in kept):
            kept.append(r)
    return sorted(kept, key=lambda r: r.t_start)


def _iou(a, b):
    inter = max(0.0, min(a.t_end, b.t_end) - max(a.t_start, b.t_start))
    union = max(a.t_end, b.t_end) - min(a.t_start, b.t_start)
    return inter / union if union > 0 else 0.0


def candidate_windows(runs, max_pair_gap_s=12.0, max_set_s=150.0):
    """Single runs plus pairs of non-overlapping runs close in time (two legs, a re-grip)."""
    wins = [(r.t_start, r.t_end, (i,)) for i, r in enumerate(runs)]
    for i, a in enumerate(runs):
        for j in range(i + 1, len(runs)):
            b = runs[j]
            if b.t_start - a.t_end > max_pair_gap_s:
                break
            if b.t_start >= a.t_end - 1.0 and b.t_end - a.t_start <= max_set_s:
                wins.append((a.t_start, b.t_end, (i, j)))
    return wins


class CentroidModel:
    """Nearest-centroid exercise model giving soft scores (log-softmax of -distance²/2)."""

    def __init__(self, X, y, temp=1.0):
        X = np.nan_to_num(np.asarray(X, float))
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd
        self.labels = sorted(set(y))
        self.cent = np.array([Z[np.array(y) == c].mean(0) for c in self.labels])
        self.temp = temp

    def log_prob(self, x):
        z = (np.nan_to_num(np.asarray(x, float)) - self.mu) / self.sd
        d2 = ((self.cent - z) ** 2).sum(1) / len(z)
        s = -d2 / self.temp
        s = s - s.max()
        lp = s - np.log(np.exp(s).sum())
        return dict(zip(self.labels, lp))


def _window_score(data, t0, t1, logged, ex, cache, fs, placement="wrist"):
    """How well 'logged set of ex happened inside [t0, t1]' fits the signal.

    The exercise's own rep detector runs on the window once per profile
    (cached). Then the best run of consecutive reps whose length is closest
    to the logged count is picked (windows often hold a set plus setup or
    fidget cycles), and scored on count error, rhythm regularity (CV of
    rep durations), fraction of plausible reps and median range of motion.
    """
    prof = profile_for(ex, placement)
    sig = (prof["channel"], prof["rest"], prof["bouts"])
    key = (round(t0, 2), round(t1, 2), sig)
    if key not in cache:
        cache[key] = analyze_set(data, t0, t1, ex, fs=fs, placement=placement)["reps"]
    reps = cache[key]
    target = max(1, logged * prof["bouts"])
    n = len(reps)
    out = {"n": n, "bouts": prof["bouts"], "count_err": 1.0, "cv": 1.0, "clean": 0.0,
           "rom": 0.0, "extra": 0, "span": (t0, t1), "fill": 0.0}
    if n == 0:
        return out
    best = None
    for m in sorted({min(n, max(1, x)) for x in (target - 1, target, target + 1)}):
        for i in range(0, n - m + 1):
            sub = reps[i:i + m]
            rs = np.array([r["rep_s"] for r in sub])
            cv = float(np.std(rs) / (np.mean(rs) + 1e-9)) if m > 1 else 1.0
            clean = sum(r["ok"] for r in sub) / m
            err = abs(m - target) / max(target, 3)
            cost = err + cv + (1 - clean)
            if best is None or cost < best[0]:
                best = (cost, i, m, cv, clean, err)
    _, i, m, cv, clean, err = best
    sub = reps[i:i + m]
    roms = [r["rom"] for r in sub if r["ok"]] or [r["rom"] for r in sub]
    med = float(np.median(roms))
    # strong cycles left outside the chosen run: the window is a set plus something else
    extra = sum(1 for r in reps[:i] + reps[i + m:] if r["ok"] and r["rom"] > 0.6 * med)
    out.update(count_err=err, cv=cv, clean=clean, rom=med, extra=extra,
               span=(sub[0]["t_start"], sub[-1]["t_end"]),
               fill=(sub[-1]["t_end"] - sub[0]["t_start"]) / max(t1 - t0, 1e-6))
    return out


def _window_gravity(data, wins, fs=25.0):
    """Mean unit gravity vector of each window (cumulative sums: O(N + W))."""
    t = data["t"]
    g = data["grav"] / np.linalg.norm(data["grav"], axis=1, keepdims=True)
    cs = np.vstack([np.zeros(3), np.cumsum(g, 0)])
    out = []
    for a, b, _ in wins:
        i, j = np.searchsorted(t, a), np.searchsorted(t, b)
        m = (cs[j] - cs[i]) / max(j - i, 1)
        out.append(m / (np.linalg.norm(m) + 1e-9))
    return np.array(out)


def align_sets(data, logged_sets, runs=None, model=None, fs=25.0, min_gap_s=5.0,
               w_count=4.0, w_cv=2.0, w_clean=2.0, w_extra=0.3, w_cls=1.0, reward=3.0,
               skip_cost=2.5, skip_cost_invisible=0.5, pad_s=2.0,
               w_ori=1.5, ori_scale_deg=25.0, w_gap=0.5, max_rest_same_s=240.0,
               max_rest_change_s=480.0, max_lead_s=360.0, w_rom=1.5, min_rom_m=0.15,
               w_romc=1.0, max_ori_cost=1.5, max_romc_cost=1.5, cache=None, placement="wrist"):
    """Assign each logged set (dicts with exercise_key, reps) to a time window, in order.

    Ordered DP over candidate windows. Per-window score: rep-count agreement,
    rhythm regularity, how much of the window the reps fill, and (optional)
    classifier log-probability. Transition costs between consecutive
    assigned sets:
      * rest longer than plausible (max_rest_same_s within an exercise,
        max_rest_change_s across) costs w_gap per extra minute;
      * consecutive sets of the same exercise whose mean wrist orientation
        differs cost w_ori per ori_scale_deg (the watch hangs the same way
        for every set of an exercise);
      * consecutive sets of the same exercise whose rep range of motion
        differs cost w_romc per doubling;
      * idle time before the first and after the last set beyond max_lead_s.
    Reps smaller than min_rom_m (arm swing while walking is ~6 cm) cost w_rom
    per halving.
    A set may be skipped (cheap for exercises the watch can't see).

    Returns list (same order as logged_sets) of dicts
    {t_start, t_end, score, n_sensed, skipped, window}.
    """
    if runs is None:
        runs = candidate_runs(data, fs)
    wins = candidate_windows(runs)
    wins.sort(key=lambda w: w[1])
    W, K = len(wins), len(logged_sets)
    starts = np.array([w[0] for w in wins])
    ends = np.array([w[1] for w in wins])
    gvec = _window_gravity(data, wins, fs)
    T0, T1 = float(data["t"][0]), float(data["t"][-1])
    cache = {} if cache is None else cache
    feats = {}
    NEG = -1e18

    def score(k, w):
        s = logged_sets[k]
        a, b, _ = wins[w]
        r = _window_score(data, a, b, s["reps"], s["exercise_key"], cache, fs, placement)
        if r["n"] == 0:
            return -np.inf, r
        sc = (reward - w_count * r["count_err"] - w_cv * min(r["cv"], 1.0)
              - w_clean * (1 - r["clean"]) - w_extra * min(r["extra"], 5))
        sc -= w_rom * max(0.0, np.log2(min_rom_m / max(r["rom"], 1e-3)))
        if model is not None:
            fk = (round(r["span"][0], 2), round(r["span"][1], 2))
            if fk not in feats:
                f = set_features(data, *r["span"]) if r["span"][1] - r["span"][0] > 2 else None
                feats[fk] = [f[k2] for k2 in FEATURES] if f else None
            if feats[fk] is not None:
                lp = model.log_prob(feats[fk]).get(s.get("label", s["exercise_key"]))
                if lp is not None:
                    sc += w_cls * max(lp, -6.0)
        return sc, r

    # A[k, w]: best total with set k (1-based) assigned to window w.
    # S[k, w]: best total with set k skipped and the last assigned set in window w
    #          (w = W means nothing assigned yet).
    A = np.full((K + 1, W + 1), NEG)
    S = np.full((K + 1, W + 1), NEG)
    S[0, W] = 0.0
    backA, backS = {}, {}
    ends_ext = np.append(ends, T0)          # "nothing assigned" ends at recording start
    rom_prev = np.full(W + 1, np.nan)
    for k in range(1, K + 1):
        rom_cur = np.full(W + 1, np.nan)
        s = logged_sets[k - 1]
        ex = s["exercise_key"]
        prev_ex = logged_sets[k - 2]["exercise_key"] if k > 1 else None
        vis = PROFILES.get(ex, PROFILES["auto"]).get("visible", True)
        sk = skip_cost if vis else skip_cost_invisible
        # skip set k
        carry = np.maximum(A[k - 1], S[k - 1])
        S[k] = carry - sk
        for c in range(W + 1):
            backS[(k, c)] = "A" if A[k - 1, c] >= S[k - 1, c] else "S"
        for w in range(W):
            sc, r = score(k - 1, w)
            rom_cur[w] = r["rom"]
            if not np.isfinite(sc):
                continue
            ok = ends_ext <= starts[w] - min_gap_s
            ok[W] = True                      # the first set may start right at the beginning
            gap = starts[w] - ends_ext
            # from "set k-1 assigned at w'": rest plausibility + orientation consistency
            same = prev_ex == ex
            lim = max_rest_same_s if same else max_rest_change_s
            cost_gap = w_gap * np.maximum(0.0, gap - lim) / 60.0
            fromA = A[k - 1] - cost_gap
            if same:
                ang = np.degrees(np.arccos(np.clip(gvec @ gvec[w], -1, 1)))
                # capped: a grip or watch-fit change must never make skipping a real set cheaper
                fromA[:W] -= np.minimum(w_ori * ang / ori_scale_deg, max_ori_cost)
                lr = np.abs(np.log2(np.maximum(rom_prev, 1e-3) / max(r["rom"], 1e-3)))
                fromA -= np.minimum(w_romc * np.nan_to_num(lr, nan=0.0), max_romc_cost)
            # from "set k-1 skipped": only the generic rest limit
            fromS = S[k - 1] - w_gap * np.maximum(0.0, gap - max_rest_change_s) / 60.0
            fromS[W] = S[k - 1, W] - w_gap * max(0.0, starts[w] - T0 - max_lead_s) / 60.0
            fromA[W] = NEG
            cand = np.where(ok, np.maximum(fromA, fromS), NEG)
            q = int(np.argmax(cand))
            if cand[q] <= NEG / 2:
                continue
            A[k, w] = cand[q] + sc
            backA[(k, w)] = (q, "A" if fromA[q] >= fromS[q] else "S")
        rom_prev = rom_cur
    # trailing idle time
    tail = np.append(w_gap * np.maximum(0.0, T1 - ends - max_lead_s) / 60.0, 0.0)
    fa, fs_ = A[K] - tail, S[K] - tail
    if fa.max() >= fs_.max():
        state, c = "A", int(np.argmax(fa))
    else:
        state, c = "S", int(np.argmax(fs_))
    out = [None] * K
    for k in range(K, 0, -1):
        if state == "S":
            out[k - 1] = {"skipped": True, "t_start": None, "t_end": None, "score": None,
                          "n_sensed": 0, "window": None}
            state = backS[(k, c)]
        else:
            sc, r = score(k - 1, c)
            a, b = r["span"]
            out[k - 1] = {"skipped": False, "t_start": round(a - pad_s, 2), "t_end": round(b + pad_s, 2),
                          "score": round(float(sc), 3), "n_sensed": r["n"],
                          "window": [round(wins[c][0], 2), round(wins[c][1], 2)]}
            c, state = backA[(k, c)]
    return out


def detect_sets(data, logged_sets, prior_model=None, fs=25.0, self_train=True, **kw):
    """Full detection: candidates -> alignment (-> self-trained second pass).

    prior_model: CentroidModel from past sessions (or None).
    self_train:  if no prior model, fit one on the first pass's confident
                 assignments and align again. Uses no ground truth.
    """
    runs = candidate_runs(data, fs)
    cache = {}
    first = align_sets(data, logged_sets, runs=runs, model=prior_model, fs=fs, cache=cache, **kw)
    if prior_model is not None or not self_train:
        return first, runs
    X, y = [], []
    for s, a in zip(logged_sets, first):
        if a["skipped"] or a["score"] is None or a["score"] < 1.0:
            continue
        f = set_features(data, a["t_start"], a["t_end"])
        X.append([f[k] for k in FEATURES])
        y.append(s.get("label", s["exercise_key"]))
    if len(set(y)) < 2:
        return first, runs
    model = CentroidModel(X, y)
    return align_sets(data, logged_sets, runs=runs, model=model, fs=fs, cache=cache, **kw), runs


def evaluate_alignment(found, truth):
    """Compare detected windows with ground-truth windows (same order).

    A set counts as found when the detected window overlaps its true window and
    overlaps no other true window more. Returns dict with per-set IoU and totals.
    """
    rows = []
    for k, (a, g) in enumerate(zip(found, truth)):
        if a is None or a["skipped"]:
            rows.append({"set": k + 1, "found": False, "iou": 0.0})
            continue
        inter = max(0.0, min(a["t_end"], g["t_end"]) - max(a["t_start"], g["t_start"]))
        union = max(a["t_end"], g["t_end"]) - min(a["t_start"], g["t_start"])
        best_other = max((min(a["t_end"], h["t_end"]) - max(a["t_start"], h["t_start"])
                          for q, h in enumerate(truth) if q != k), default=0.0)
        rows.append({"set": k + 1, "found": inter > 0 and inter >= best_other,
                     "iou": round(inter / union, 3) if union > 0 else 0.0})
    n = len(rows)
    return {"recall": sum(r["found"] for r in rows) / n if n else 0.0,
            "mean_iou": float(np.mean([r["iou"] for r in rows])) if n else 0.0, "rows": rows}
