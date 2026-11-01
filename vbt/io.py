"""Load Apple Watch motion exports.

Works with the free *Sensor Logger* app (Apple Watch -> "Wrist Motion",
file WristMotion.csv) and any CSV with CoreMotion deviceMotion columns:
time, userAcceleration x/y/z and gravity x/y/z. CoreMotion reports both in g.
"""
import numpy as np
import pandas as pd

G = 9.80665

_ALIASES = {
    "t": ["seconds_elapsed", "timestamp", "time_s", "t"],
    "ax": ["accelerationX", "userAccelerationX", "acc_x", "ax"],
    "ay": ["accelerationY", "userAccelerationY", "acc_y", "ay"],
    "az": ["accelerationZ", "userAccelerationZ", "acc_z", "az"],
    "gx": ["gravityX", "grav_x", "gx"],
    "gy": ["gravityY", "grav_y", "gy"],
    "gz": ["gravityZ", "grav_z", "gz"],
    "rx": ["rotationRateX", "gyro_x", "rx"],
    "ry": ["rotationRateY", "gyro_y", "ry"],
    "rz": ["rotationRateZ", "gyro_z", "rz"],
}


def _pick(df, key):
    for name in _ALIASES[key]:
        if name in df.columns:
            return df[name].to_numpy(float)
    raise KeyError(f"No column for {key}; tried {_ALIASES[key]}")


def load_watch_csv(path, units="g", keep_clock=False):
    """Return dict with t (s), acc (N,3 m/s^2, gravity removed), grav (N,3 m/s^2),
    and gyro (N,3 rad/s) when rotation-rate columns exist.

    units: "g" (CoreMotion default) or "ms2" if the export is already m/s^2.
    keep_clock: keep Sensor Logger's seconds_elapsed as-is instead of starting at 0
        (set windows and annotations are stored on that clock).
    """
    df = pd.read_csv(path)
    t = _pick(df, "t")
    if "seconds_elapsed" not in df.columns and t.max() > 1e12:  # ns epoch
        t = t / 1e9
    if not keep_clock:
        t = t - t[0]
    acc = np.column_stack([_pick(df, k) for k in ("ax", "ay", "az")])
    grav = np.column_stack([_pick(df, k) for k in ("gx", "gy", "gz")])
    scale = G if units == "g" else 1.0
    order = np.argsort(t)
    out = {"t": t[order], "acc": acc[order] * scale, "grav": grav[order] * scale}
    try:
        out["gyro"] = np.column_stack([_pick(df, k) for k in ("rx", "ry", "rz")])[order]
    except KeyError:
        pass
    return out
