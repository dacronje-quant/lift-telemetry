"""Signal pipeline: vertical accel -> rep detection -> per-rep velocity (ZUPT)."""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.integrate import cumulative_trapezoid


def _resample(t, x, fs):
    tu = np.arange(t[0], t[-1], 1.0 / fs)
    return tu, np.interp(tu, t, x)


def _lp(x, fs, fc):
    b, a = butter(4, fc / (fs / 2), "low")
    return filtfilt(b, a, x)


def _bp(x, fs, lo, hi):
    b, a = butter(2, [lo / (fs / 2), hi / (fs / 2)], "band")
    return filtfilt(b, a, x)


def vertical_acceleration(acc, grav):
    """Project user acceleration onto 'up' (opposite of gravity). No quaternions needed."""
    g_hat = grav / np.linalg.norm(grav, axis=1, keepdims=True)
    return -np.einsum("ij,ij->i", acc, g_hat)


def detect_reps(t, a_up, fs=100.0, min_rep_s=0.8, min_rom_m=0.10):
    """Find reps from a drift-robust position estimate.

    Returns list of (start_top, bottom, end_top) sample indices on the resampled grid.
    """
    a = _bp(a_up, fs, 0.15, 4.0)
    v = _bp(cumulative_trapezoid(a, dx=1 / fs, initial=0), fs, 0.15, 4.0)
    p = _bp(cumulative_trapezoid(v, dx=1 / fs, initial=0), fs, 0.15, 4.0)
    dist = int(min_rep_s * fs)
    prom = min_rom_m * 0.4  # band-pass shrinks amplitude; be lenient
    tops, _ = find_peaks(p, distance=dist, prominence=prom)
    bots, _ = find_peaks(-p, distance=dist, prominence=prom)
    reps = []
    # a rep = bottom followed by the next top (concentric); eccentric starts at previous top
    for b in bots:
        nxt = tops[tops > b]
        if not len(nxt):
            continue
        prev = tops[tops < b]
        start = prev[-1] if len(prev) else max(0, b - dist)
        reps.append((int(start), int(b), int(nxt[0])))
    return reps, p


@dataclass
class Rep:
    n: int
    t_bottom: float
    mean_velocity: float   # mean concentric velocity (m/s)
    peak_velocity: float   # m/s
    rom: float             # concentric displacement (m)
    concentric_s: float
    eccentric_s: float


def _zupt_velocity(a, fs):
    """Integrate with zero velocity at both ends; remove linear drift."""
    v = cumulative_trapezoid(a, dx=1 / fs, initial=0)
    return v - np.linspace(0, v[-1], len(v))


def analyze(data, fs=100.0, lowpass_hz=6.0, **detect_kw):
    """Full pipeline on output of load_watch_csv (or synth_set). Returns dict."""
    a_up = vertical_acceleration(data["acc"], data["grav"])
    t, a_up = _resample(data["t"], a_up, fs)
    a_f = _lp(a_up, fs, lowpass_hz)
    reps_idx, pos = detect_reps(t, a_up, fs, **detect_kw)
    reps = []
    for i, (s, b, e) in enumerate(reps_idx, 1):
        # integrate the whole rep top->top (both ends ~stationary), then
        # take the concentric phase as the positive-velocity lobe around peak
        v = _zupt_velocity(a_f[s:e + 1], fs)
        k = int(np.argmax(v))
        thr = 0.05 * v[k]
        lo = k
        while lo > 0 and v[lo - 1] > thr:
            lo -= 1
        hi = k
        while hi < len(v) - 1 and v[hi + 1] > thr:
            hi += 1
        vc = v[lo:hi + 1]
        reps.append(Rep(
            n=i, t_bottom=float(t[s + lo]),
            mean_velocity=float(vc.mean()),
            peak_velocity=float(v[k]),
            rom=float(np.trapezoid(vc, dx=1 / fs)),
            concentric_s=len(vc) / fs, eccentric_s=lo / fs))
    out = {"reps": [asdict(r) for r in reps], "rep_count": len(reps),
           "t": t, "a_up": a_f, "pos": pos, "reps_idx": reps_idx}
    if reps:
        mv = [r.mean_velocity for r in reps]
        best = max(mv)
        out["best_mean_velocity"] = best
        out["velocity_loss_pct"] = 100 * (best - mv[-1]) / best
    return out
