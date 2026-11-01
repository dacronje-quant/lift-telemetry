"""Split a whole-workout recording into sets using rest periods."""
import numpy as np
from .core import vertical_acceleration, analyze


def split_sets(data, fs=100.0, win_s=1.0, active_thresh=0.6, min_rest_s=20.0, min_set_s=4.0):
    """Return list of (t_start, t_end) for active periods separated by >= min_rest_s of rest.

    Activity = rolling RMS of vertical acceleration (m/s^2) above active_thresh.
    """
    t = data["t"]
    tu = np.arange(t[0], t[-1], 1 / fs)
    a = np.interp(tu, t, vertical_acceleration(data["acc"], data["grav"]))
    w = max(1, int(win_s * fs))
    rms = np.sqrt(np.convolve(a ** 2, np.ones(w) / w, mode="same"))
    active = rms > active_thresh
    idx = np.flatnonzero(active)
    if not len(idx):
        return []
    sets, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if (i - prev) / fs >= min_rest_s:
            sets.append((start, prev)); start = i
        prev = i
    sets.append((start, prev))
    pad = int(1.5 * fs)
    return [(tu[max(0, s - pad)], tu[min(len(tu) - 1, e + pad)])
            for s, e in sets if (e - s) / fs >= min_set_s]


def analyze_session(data, **kw):
    """Split into sets, then run the per-set analysis on each one."""
    out = []
    for n, (t0, t1) in enumerate(split_sets(data), 1):
        m = (data["t"] >= t0) & (data["t"] <= t1)
        sub = {"t": data["t"][m], "acc": data["acc"][m], "grav": data["grav"][m]}
        res = analyze(sub, **kw)
        res.update(set=n, t_start=float(t0), t_end=float(t1))
        if res["rep_count"]:
            out.append(res)
    return out


def find_sets(data, fs=50.0, min_prom_m=0.25, min_peaks=4, merge_gap_s=7.0):
    """Rep-periodicity set finder for noisy whole-workout recordings.

    `split_sets` assumes the wrist is quiet between sets. In a real gym it
    isn't (walking, loading plates, phone). Instead look for *runs of regular
    reps*: peaks of the drift-free vertical position with spacing 1-6.5 s,
    consecutive intervals within x0.55-1.8 of each other and similar height.
    Runs closer than merge_gap_s merge (e.g. left/right leg of a split squat).

    Returns list of dicts {t_start, t_end, n_peaks, period_s, prom_m}.
    """
    from scipy.signal import find_peaks
    from .workout import position
    t = data["t"]
    tu = np.arange(t[0], t[-1], 1 / fs)
    a = np.interp(tu, t, vertical_acceleration(data["acc"], data["grav"]))
    p = position(a, fs)
    pk, pr = find_peaks(p, prominence=0.06, distance=int(fs))
    tp, am = tu[pk], pr["prominences"]
    runs, cur = [], [0]
    for i in range(1, len(pk)):
        dt = tp[i] - tp[cur[-1]]
        ok = 1.0 <= dt <= 6.5
        if ok and len(cur) >= 2:
            ok = 0.55 < dt / (tp[cur[-1]] - tp[cur[-2]]) < 1.8 and 0.4 < am[i] / am[cur[-1]] < 2.5
        if ok:
            cur.append(i)
        else:
            runs.append(cur)
            cur = [i]
    runs.append(cur)
    runs = [c for c in runs if len(c) >= min_peaks and np.median(am[c]) >= min_prom_m]
    out = []
    for c in runs:
        d = {"t_start": float(tp[c[0]]), "t_end": float(tp[c[-1]]), "n_peaks": len(c),
             "period_s": float(np.diff(tp[c]).mean()), "prom_m": float(np.median(am[c]))}
        if out and d["t_start"] - out[-1]["t_end"] < merge_gap_s:
            o = out[-1]
            o.update(t_end=d["t_end"], n_peaks=o["n_peaks"] + d["n_peaks"])
        else:
            out.append(d)
    return out
