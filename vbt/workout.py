"""Per-set rep segmentation, velocity and motion-signature features for a
whole-workout recording once set windows and exercises are known.

Exercise profiles encode two facts the generic pipeline can't guess:

* which signal counts reps best. Vertical wrist position works for bar
  lifts, but for elbow-hinge lifts (curl, pushdown) the forearm *rotation*
  (gravity seen on the watch X axis) is far cleaner;
* where the rest point of a rep is (top or bottom) and which way the
  concentric goes. Pulldown and pushdown are concentric *downwards*.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.integrate import cumulative_trapezoid

from .core import vertical_acceleration

G = 9.80665

# channel: rep-counting signal: "pos" vertical position, "gx"/"gy"/"gz" gravity on a
#          watch axis (forearm rotation), "auto" = pick the most regular one
# rest:    "top" / "bottom" = where the rep starts and ends (ZUPT boundary)
# conc:    +1 concentric moves the wrist up, -1 down
# bouts:   independent groups of reps in one set (split squat: left + right leg)
# mvt:     minimum velocity threshold for e1RM (m/s); None = no 1RM model
# visible: False when the watch wrist barely moves (machines, bodyweight on the
#          hands); the aligner may skip such sets cheaply
PROFILES = {
    "bench":    dict(channel="pos", rest="top", conc=+1, bouts=1, mvt=0.17),
    "deadlift": dict(channel="pos", rest="bottom", conc=+1, bouts=1, mvt=0.15),
    "rdl":      dict(channel="pos", rest="top", conc=+1, bouts=1, mvt=None),
    "squat":    dict(channel="pos", rest="top", conc=+1, bouts=1, mvt=0.30),
    "ohp":      dict(channel="pos", rest="bottom", conc=+1, bouts=1, mvt=0.19),
    "bss":      dict(channel="pos", rest="top", conc=+1, bouts=2, mvt=None),
    "lunge":    dict(channel="pos", rest="top", conc=+1, bouts=2, mvt=None),
    "hipthrust": dict(channel="pos", rest="bottom", conc=+1, bouts=1, mvt=None),
    "pulldown": dict(channel="pos", rest="top", conc=-1, bouts=1, mvt=None),
    "row":      dict(channel="auto", rest="bottom", conc=+1, bouts=1, mvt=None),
    "pushdown": dict(channel="gx", rest="top", conc=-1, bouts=1, mvt=None),
    "curl":     dict(channel="gx", rest="bottom", conc=+1, bouts=1, mvt=None),
    "lateral":  dict(channel="pos", rest="bottom", conc=+1, bouts=1, mvt=None),
    "extension": dict(channel="auto", rest="bottom", conc=+1, bouts=1, mvt=None),
    "fly":      dict(channel="auto", rest="bottom", conc=+1, bouts=1, mvt=None),
    "machine":  dict(channel="auto", rest="bottom", conc=+1, bouts=1, mvt=None, visible=False),
    "bodyweight": dict(channel="auto", rest="bottom", conc=+1, bouts=1, mvt=None, visible=False),
    "auto":     dict(channel="auto", rest="bottom", conc=+1, bouts=1, mvt=None),
}

# first match wins, so specific phrases go before generic ones
_KEYWORDS = [("romanian", "rdl"), ("rdl", "rdl"), ("stiff leg", "rdl"),
             ("split squat", "bss"), ("lunge", "lunge"), ("step up", "lunge"),
             ("leg press", "machine"), ("leg extension", "machine"), ("leg curl", "machine"),
             ("calf", "machine"), ("hip abduction", "machine"), ("hip adduction", "machine"),
             ("pull up", "bodyweight"), ("pull-up", "bodyweight"), ("chin up", "bodyweight"),
             ("chin-up", "bodyweight"), ("dip", "bodyweight"), ("push up", "bodyweight"),
             ("push-up", "bodyweight"), ("plank", "bodyweight"), ("crunch", "bodyweight"),
             ("hip thrust", "hipthrust"), ("glute bridge", "hipthrust"),
             ("deadlift", "deadlift"), ("bench", "bench"), ("chest press", "bench"),
             ("squat", "squat"), ("overhead press", "ohp"), ("shoulder press", "ohp"),
             ("military press", "ohp"), ("pulldown", "pulldown"), ("pull down", "pulldown"),
             ("pushdown", "pushdown"), ("push down", "pushdown"), ("row", "row"),
             ("curl", "curl"), ("lateral raise", "lateral"), ("front raise", "lateral"),
             ("raise", "lateral"), ("skull", "extension"), ("triceps extension", "extension"),
             ("overhead extension", "extension"), ("fly", "fly"), ("flye", "fly")]


def exercise_key(name):
    """Map a logged exercise name (any app) to a PROFILES key ("auto" if unknown)."""
    low = (name or "").lower().replace("(", " ").replace(")", " ")
    for kw, key in _KEYWORDS:
        if kw in low:
            return key
    return "auto"


def split_bouts(reps, n_bouts):
    """Split a rep list at gaps > 2x the median turnaround spacing (unilateral work)."""
    if n_bouts < 2 or len(reps) < 3:
        return [list(reps)]
    tt = np.array([r["t_turn"] for r in reps])
    gaps = np.diff(tt)
    cut = np.flatnonzero(gaps > 2.0 * np.median(gaps))
    edges = [0, *(cut + 1), len(reps)]
    return [list(reps[edges[j]:edges[j + 1]]) for j in range(len(edges) - 1)]


def count_verdict(reps, logged, n_bouts=1):
    """Compare a logged rep count with what the watch saw.

    Each unit (the set, or each bout of a unilateral set) has an ambiguity
    interval [clean cycles, all cycles]: flagged edge cycles (unrack, setup
    pull, re-rack) may or may not be reps. Returns (verdict, delta, bouts, bouts_clean)
    with verdict in match / ambiguous / watch fewer / watch more.
    """
    units = split_bouts(reps, n_bouts)
    bouts = [len(u) for u in units]
    clean = [sum(r["ok"] for r in u) for u in units]
    verdicts = []
    for lo_n, hi_n in zip(clean, bouts):
        if hi_n == logged:
            verdicts.append(("match", 0))
        elif lo_n <= logged <= hi_n:
            verdicts.append(("ambiguous", 0))
        elif logged > hi_n:
            verdicts.append(("watch fewer", logged - hi_n))
        else:
            verdicts.append(("watch more", lo_n - logged))
    rank = {"match": 0, "ambiguous": 1, "watch fewer": 2, "watch more": 2}
    v, delta = max(verdicts, key=lambda x: (rank[x[0]], x[1]))
    return v, delta, bouts, clean


def _bp(x, fs, lo=0.2, hi=3.0):
    b, a = butter(2, [lo / (fs / 2), hi / (fs / 2)], "band")
    return filtfilt(b, a, x, axis=0)


def _lp(x, fs, fc=6.0):
    b, a = butter(4, fc / (fs / 2), "low")
    return filtfilt(b, a, x, axis=0)


def resample(data, t0, t1, fs=100.0, pad=4.0):
    """Cut [t0-pad, t1+pad] and resample every channel present onto a uniform grid."""
    t = data["t"]
    m = (t >= t0 - pad) & (t <= t1 + pad)
    tu = np.arange(t[m][0], t[m][-1], 1.0 / fs)
    out = {"t": tu}
    for k, v in data.items():
        if k == "t" or v is None:
            continue
        v = np.asarray(v)[m]
        out[k] = np.column_stack([np.interp(tu, t[m], v[:, i]) for i in range(v.shape[1])]) \
            if v.ndim == 2 else np.interp(tu, t[m], v)
    return out


def position(a_up, fs):
    """Band-passed double integral of vertical accel: drift-free relative height (m)."""
    v = _bp(cumulative_trapezoid(_bp(a_up, fs), dx=1 / fs, initial=0), fs)
    return _bp(cumulative_trapezoid(v, dx=1 / fs, initial=0), fs)


def _chains(idx, prom, fs, min_len=3):
    """Group extrema into runs with regular spacing and similar amplitude."""
    if not len(idx):
        return []
    out, cur = [], [0]
    for i in range(1, len(idx)):
        dt = (idx[i] - idx[cur[-1]]) / fs
        ok = 0.8 <= dt <= 7.0 and 0.4 < prom[i] / prom[cur[-1]] < 2.5
        if ok and len(cur) >= 2:
            ok = 0.5 < dt / ((idx[cur[-1]] - idx[cur[-2]]) / fs) < 2.0
        if ok:
            cur.append(i)
        else:
            out.append(cur)
            cur = [i]
    out.append(cur)
    return [c for c in out if len(c) >= min_len]


def find_turnarounds(sig, fs, bouts=1, rel_prom=0.35, t0=None, t1=None, t=None):
    """Turnaround extrema (one per rep) of a rep signal whose turnaround is a *peak*.

    Keeps the longest regular chain(s): `bouts` chains for unilateral work,
    plus any chain within 4 s of a kept one (a re-grip mid-set).
    t0/t1 restrict to turnarounds inside the set window.
    """
    pk, pr = find_peaks(sig, prominence=1e-6, distance=int(0.7 * fs))
    if not len(pk):
        return np.array([], int)
    if t is not None and t0 is not None:
        keep = (t[pk] >= t0) & (t[pk] <= t1)
        pk, prom = pk[keep], pr["prominences"][keep]
    else:
        prom = pr["prominences"]
    if not len(pk):
        return np.array([], int)
    big = prom >= rel_prom * np.percentile(prom, 85)
    pk, prom = pk[big], prom[big]
    ch = _chains(pk, prom, fs)
    if not ch:
        return np.array([], int)
    ch.sort(key=len, reverse=True)
    kept = ch[:bouts]
    for c in ch[bouts:]:
        if any(min(abs(pk[c[0]] - pk[k[-1]]), abs(pk[k[0]] - pk[c[-1]])) / fs < 4.0 for k in kept):
            kept.append(c)
    sel = sorted(i for c in kept for i in c)
    return pk[sel]


@dataclass
class RepMetrics:
    n: int
    t_turn: float          # time of the turnaround (s, recording clock)
    t_start: float         # rest point before
    t_end: float           # rest point after
    mean_velocity: float   # mean concentric |v| (m/s), vertical
    peak_velocity: float   # peak concentric |v| (m/s)
    rom: float             # concentric vertical displacement (m)
    concentric_s: float
    eccentric_s: float
    rep_s: float           # rest->rest duration
    rotation_deg: float    # peak-to-peak forearm rotation (gravity-vector swing)
    ok: bool = True        # False = kinematics implausible vs. the rest of the set


def _rep_velocity(a_up, fs, conc):
    """ZUPT both ends + linear de-drift; return v, concentric lobe slice, eccentric length."""
    v = cumulative_trapezoid(a_up, dx=1 / fs, initial=0)
    v = v - np.linspace(0, v[-1], len(v))
    s = conc * v
    k = int(np.argmax(s))
    thr = 0.05 * max(s[k], 1e-6)
    lo = k
    while lo > 0 and s[lo - 1] > thr:
        lo -= 1
    hi = k
    while hi < len(s) - 1 and s[hi + 1] > thr:
        hi += 1
    ecc = int(np.sum(-s > thr))
    return v, slice(lo, hi + 1), ecc


def profile_for(exercise, placement="wrist"):
    """Exercise profile adjusted for where the sensor sits.

    Profiles are written for a wrist-worn sensor. On the bar (or a bar clip) the
    sensor moves with the load and doesn't rotate with the forearm, so every lift
    is counted from vertical position.
    """
    prof = dict(PROFILES.get(exercise, PROFILES["auto"]))
    if placement == "bar" and prof["channel"] in ("gx", "gy", "gz"):
        prof["channel"] = "pos"
    return prof


def analyze_set(data, t0, t1, exercise, fs=100.0, placement="wrist"):
    """Count reps and compute per-rep kinematics for one set window.

    data: dict with t, acc, grav (m/s^2, N x 3) and optional gyro (rad/s, N x 3).
    exercise: PROFILES key.
    """
    prof = profile_for(exercise, placement)
    d = resample(data, t0, t1, fs)
    t, acc, grav = d["t"], d["acc"], d["grav"]
    g_hat = grav / np.linalg.norm(grav, axis=1, keepdims=True)
    a_up = vertical_acceleration(acc, grav)
    pos = position(a_up, fs)
    signals = {"pos": pos}
    # rotation-counted reps follow the fast orientation when the source has one
    # (accelerometer-only sensors), otherwise gravity
    o_hat = d["orient"] / np.linalg.norm(d["orient"], axis=1, keepdims=True) if "orient" in d else g_hat
    for i, ax in enumerate("xyz"):
        signals["g" + ax] = _bp(o_hat[:, i], fs, 0.1, 2.0)
    if prof["channel"] == "auto":
        # unknown exercise: the channel + rest polarity with the longest regular run wins
        best = None
        for name, sig in signals.items():
            for rest in ("top", "bottom"):
                ts = -1 if rest == "top" else 1
                tr = find_turnarounds(ts * sig / (np.std(sig) + 1e-12), fs, bouts=prof["bouts"],
                                      t0=t0, t1=t1, t=t)
                if best is None or len(tr) > len(best[2]):
                    best = (name, rest, tr)
        channel, rest, turns = best
        turn_sign = -1 if rest == "top" else 1
        chan = signals[channel]
    else:
        channel, rest = prof["channel"], prof["rest"]
        chan = signals[channel]
        # turnaround = the extreme opposite the rest point
        turn_sign = -1 if rest == "top" else 1
        turns = find_turnarounds(turn_sign * chan / (np.std(chan) + 1e-12), fs,
                                 bouts=prof["bouts"], t0=t0, t1=t1, t=t)
    a_f = _lp(a_up, fs, 6.0)
    reps = []
    if len(turns):
        per = np.median(np.diff(turns)) if len(turns) > 1 else int(2.5 * fs)
        rest_sig = -turn_sign * chan
        bounds = []
        for i, k in enumerate(turns):
            lo = turns[i - 1] if i > 0 and turns[i] - turns[i - 1] < 1.8 * per else max(0, k - int(per))
            hi = turns[i + 1] if i + 1 < len(turns) and turns[i + 1] - k < 1.8 * per \
                else min(len(t) - 1, k + int(per))
            s = lo + int(np.argmax(rest_sig[lo:k + 1]))
            e = k + int(np.argmax(rest_sig[k:hi + 1]))
            bounds.append((s, k, e))
        for n, (s, k, e) in enumerate(bounds, 1):
            if e - s < int(0.4 * fs):
                continue
            v, cl, ecc = _rep_velocity(a_f[s:e + 1], fs, prof["conc"])
            vc = np.abs(v[cl])
            ang = np.degrees(np.arccos(np.clip(g_hat[s:e + 1] @ g_hat[s:e + 1].mean(0)
                                               / np.linalg.norm(g_hat[s:e + 1].mean(0)), -1, 1)))
            reps.append(RepMetrics(
                n=n, t_turn=float(t[k]), t_start=float(t[s]), t_end=float(t[e]),
                mean_velocity=float(vc.mean()), peak_velocity=float(vc.max()),
                rom=float(np.trapezoid(vc, dx=1 / fs)),
                concentric_s=len(vc) / fs, eccentric_s=ecc / fs,
                rep_s=(e - s) / fs, rotation_deg=float(np.ptp(ang))))
    if len(reps) >= 3:
        rom = np.median([r.rom for r in reps])
        mvm = np.median([r.mean_velocity for r in reps])
        for r in reps:
            r.ok = bool(0.5 * rom <= r.rom <= 1.8 * rom and r.mean_velocity >= 0.5 * mvm)
    out = {"exercise": exercise, "channel": channel, "rest": rest, "t": t, "pos": pos,
           "chan": chan, "a_up": a_f,
           "turns": turns, "reps": [asdict(r) for r in reps], "rep_count": len(reps)}
    good = [r for r in reps if r.ok]
    if good:
        mv = np.array([r.mean_velocity for r in good])
        if prof["bouts"] > 1:
            loss = None   # unilateral bouts alternate legs; loss is per-leg, not per-set
        else:
            # fastest rep vs mean of the last 2 (standard VBT velocity loss,
            # averaged at the end to damp one noisy rep)
            best, last = mv.max(), mv[-2:].mean()
            loss = float(100 * (best - last) / best)
        out.update(best_mean_velocity=float(mv.max()), velocity_loss_pct=loss,
                   mean_velocity=float(mv.mean()),
                   tut_s=float(sum(r.rep_s for r in reps)),
                   active_span_s=float(reps[-1].t_end - reps[0].t_start))
    return out


FEATURES = ["g_x", "g_y", "g_z", "g_std_x", "g_std_y", "g_std_z", "rot_range_deg",
            "gyro_rms_x", "gyro_rms_y", "gyro_rms_z", "gyro_dom_frac",
            "a_vert_rms", "a_horiz_rms", "vert_ratio", "rom_cm", "period_s"]
FEATURE_GROUPS = {"orientation": FEATURES[:3],
                  "rotation": FEATURES[3:11],
                  "translation": FEATURES[11:]}


def set_features(data, t0, t1, fs=50.0):
    """Exercise-agnostic motion signature of [t0, t1] (watch frame)."""
    d = resample(data, t0, t1, fs, pad=0.0)
    acc, grav = d["acc"], d["grav"]
    g_hat = grav / np.linalg.norm(grav, axis=1, keepdims=True)
    gm = g_hat.mean(0)
    gm_u = gm / np.linalg.norm(gm)
    ang = np.degrees(np.arccos(np.clip(g_hat @ gm_u, -1, 1)))
    a_up = vertical_acceleration(acc, grav)
    a_h = acc + a_up[:, None] * g_hat          # remove vertical component
    gyro = d.get("gyro", np.zeros_like(acc))
    grms = np.sqrt((gyro ** 2).mean(0))
    pos = position(a_up, fs)
    # dominant period from autocorrelation of the position
    x = pos - pos.mean()
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    lo, hi = int(0.8 * fs), min(len(ac) - 1, int(7 * fs))
    period = (lo + int(np.argmax(ac[lo:hi]))) / fs if hi > lo else np.nan
    vr, hr = np.sqrt((a_up ** 2).mean()), np.sqrt((np.linalg.norm(a_h, axis=1) ** 2).mean())
    return {"g_x": gm[0], "g_y": gm[1], "g_z": gm[2],
            "g_std_x": g_hat[:, 0].std(), "g_std_y": g_hat[:, 1].std(), "g_std_z": g_hat[:, 2].std(),
            "rot_range_deg": float(np.percentile(ang, 95) - np.percentile(ang, 5)),
            "gyro_rms_x": grms[0], "gyro_rms_y": grms[1], "gyro_rms_z": grms[2],
            "gyro_dom_frac": float(grms.max() / (grms.sum() + 1e-9)),
            "a_vert_rms": vr, "a_horiz_rms": hr, "vert_ratio": vr / (vr + hr + 1e-9),
            "rom_cm": float(100 * (np.percentile(pos, 95) - np.percentile(pos, 5))),
            "period_s": period}


def loso_classify(X, y, method="centroid", k=1):
    """Leave-one-set-out predictions with z-scored features (numpy only).

    method: "centroid" (nearest class mean) or "knn".
    Standardisation is fit on the training fold only.
    """
    X, y = np.asarray(X, float), np.asarray(y)
    pred = []
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Z, z = (X[tr] - mu) / sd, (X[i] - mu) / sd
        if method == "centroid":
            labs = np.unique(y[tr])
            cents = np.array([Z[y[tr] == c].mean(0) for c in labs])
            pred.append(labs[np.argmin(((cents - z) ** 2).sum(1))])
        else:
            dist = ((Z - z) ** 2).sum(1)
            nn = y[tr][np.argsort(dist)[:k]]
            vals, cnt = np.unique(nn, return_counts=True)
            pred.append(vals[np.argmax(cnt)])
    pred = np.array(pred)
    return pred, float((pred == y).mean())
