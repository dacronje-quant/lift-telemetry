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
