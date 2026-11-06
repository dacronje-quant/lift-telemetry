"""Motion sensors from any source, normalised to one canonical table.

Canonical frame (what the rest of vbt expects), per sample:

    seconds_elapsed            s, monotonic
    accelerationX/Y/Z          user acceleration (gravity removed), in g
    gravityX/Y/Z               unit vector pointing *down* in the sensor frame
    rotationRateX/Y/Z          rad/s (zeros if the source has no gyroscope)
    quaternionW/X/Y/Z (opt.)   device attitude, when the source provides it (full earth frame:
                               lets later code look at horizontal bar path, not just vertical)
    orientX/Y/Z  (optional)    fast orientation unit vector, only without a gyroscope

i.e. Apple CoreMotion "device motion": raw accelerometer = gravity + user.

Supported sources (``fmt="auto"`` sniffs a zip, folder or file):

* ``sensorlogger-watch``: Sensor Logger Apple Watch export (WristMotion.csv), already canonical.
* ``sensorlogger-phone``: Sensor Logger phone export (Accelerometer/Gravity/Gyroscope.csv,
  or TotalAcceleration.csv). Platform from Metadata.csv decides the sign convention.
* ``imu-csv``: any CSV with a time column and accelerometer x/y/z, optionally gyroscope
  and gravity. Units are inferred (g vs m/s^2, rad/s vs deg/s, s/ms/ns time).
  When gravity isn't given it's estimated from the raw accelerometer with a
  gyro-aided complementary filter (low-pass only if there's no gyro).

Accelerometer sign conventions differ: Apple reports gravity direction at rest
("gravity", reading -1 g on the axis pointing up), Android and most IMUs report
specific force ("specific_force", +1 g up). Pass ``accel_convention`` when the
source isn't one of the known apps; the default for unknown CSVs is
"specific_force".
"""
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

G = 9.80665
QUAT = ["quaternionW", "quaternionX", "quaternionY", "quaternionZ"]
COLS = ["rotationRateX", "rotationRateY", "rotationRateZ", "gravityX", "gravityY", "gravityZ",
        "accelerationX", "accelerationY", "accelerationZ"]


# ------------------------------------------------------------------ file access
def _members(src):
    """name -> bytes loader for a zip, a folder or a single file."""
    src = Path(src)
    if src.is_file() and zipfile.is_zipfile(src):
        z = zipfile.ZipFile(src)
        return {Path(n).name: (lambda n=n: z.read(n)) for n in z.namelist() if not n.endswith("/")}
    if src.is_dir():
        return {p.name: (lambda p=p: p.read_bytes()) for p in src.rglob("*") if p.is_file()}
    return {src.name: (lambda: src.read_bytes())}


def _csv(loader):
    return pd.read_csv(io.BytesIO(loader()))


def _find(members, *names):
    low = {k.lower(): k for k in members}
    for n in names:
        if n.lower() in low:
            return members[low[n.lower()]]
    return None


def read_sensorlogger_meta(members):
    m = _find(members, "Metadata.csv")
    if m is None:
        return {}
    row = _csv(m).iloc[0].to_dict()
    out = {"device": str(row.get("device name", "") or ""), "timezone": str(row.get("recording timezone", "") or ""),
           "platform": str(row.get("platform", "") or "").lower(), "app": "Sensor Logger",
           "app_version": str(row.get("appVersion", "") or "")}
    if "recording epoch time" in row and pd.notna(row["recording epoch time"]):
        out["start_epoch_ms"] = float(row["recording epoch time"])
    return out


# ------------------------------------------------------------------ helpers
def _time_seconds(t):
    t = np.asarray(t, float)
    dt = np.median(np.diff(t))
    if dt > 1e5:            # ns
        t = t / 1e9
    elif dt > 0.5:          # ms
        t = t / 1e3
    return t


def _pick(df, *cands):
    low = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    for c in cands:
        k = c.lower().replace(" ", "").replace("_", "")
        if k in low:
            return low[k]
    return None


def _triplet(df, prefixes):
    """Find x/y/z columns for any of the prefixes (e.g. 'acc', 'accelerometer', '')."""
    for p in prefixes:
        cols = [_pick(df, f"{p}{a}", f"{p}_{a}", f"{p}{a.upper()}", f"{p} {a}") for a in "xyz"]
        if all(cols):
            return cols
    return None


def separate_gravity(t, acc_g, gyro=None, tau_s=1.5, slow_hz=0.2):
    """Estimate the down vector from raw accel (canonical sign, in g) [+ gyro, rad/s].

    Gyro-aided complementary filter: propagate the gravity vector with the
    rotation rate (dg/dt = -w x g), pull it towards the measured accel with
    time constant tau_s, trusting the accel less when |a| is far from 1 g.
    Without a gyro: zero-phase 0.2 Hz low-pass for "up" (tuned on real data: rep counts
    within ±1, velocity ~0.07 m/s worse than with a gyro).
    Returns (gravity unit vectors, user acceleration in g, orientation or None).
    Without a gyro, gravity (slow, for "up") and orientation (fast, for counting
    rotation-based reps) have to come from different filters; with a gyro they
    are the same vector and orientation is None.
    """
    acc_g = np.asarray(acc_g, float)
    n = len(acc_g)
    if gyro is None:
        # No gyro: two filters. Orientation (what rotation-counted lifts like curls
        # need) follows the accel direction low-passed at 2 Hz; the user acceleration
        # removes only the slow 0.3 Hz part so the lifting motion itself survives.
        fs = 1.0 / np.median(np.diff(t))
        b, a = butter(2, slow_hz / (fs / 2), "low")
        slow = filtfilt(b, a, acc_g, axis=0)
        b, a = butter(2, min(2.0, 0.4 * fs) / (fs / 2), "low")
        fast = filtfilt(b, a, acc_g, axis=0)
        g = slow / np.linalg.norm(slow, axis=1, keepdims=True)
        orient = fast / np.linalg.norm(fast, axis=1, keepdims=True)
        return g, acc_g - slow, orient
    else:
        g = np.empty_like(acc_g)
        gv = acc_g[0] / np.linalg.norm(acc_g[0])
        dts = np.diff(t, prepend=t[0])
        for i in range(n):
            dt = dts[i]
            gv = gv - dt * np.cross(gyro[i], gv)
            an = np.linalg.norm(acc_g[i])
            trust = np.exp(-((an - 1.0) / 0.15) ** 2)
            k = trust * dt / (tau_s + dt)
            gv = (1 - k) * gv + k * acc_g[i] / max(an, 1e-6)
            gv /= np.linalg.norm(gv)
            g[i] = gv
        # run backwards too and average: removes the filter's lag
        gb = np.empty_like(acc_g)
        gv = acc_g[-1] / np.linalg.norm(acc_g[-1])
        for i in range(n - 1, -1, -1):
            dt = dts[i + 1] if i + 1 < n else 0.0
            gv = gv + dt * np.cross(gyro[i], gv)
            an = np.linalg.norm(acc_g[i])
            trust = np.exp(-((an - 1.0) / 0.15) ** 2)
            k = trust * dt / (tau_s + dt)
            gv = (1 - k) * gv + k * acc_g[i] / max(an, 1e-6)
            gv /= np.linalg.norm(gv)
            gb[i] = gv
        g = g + gb
        g /= np.linalg.norm(g, axis=1, keepdims=True)
    # user acceleration = raw - gravity (canonical: raw = gravity + user, in g)
    return g, acc_g - g, None


def _canonical(t, user_g, grav_unit, gyro, orient=None):
    out = {"seconds_elapsed": t}
    for i, a in enumerate("XYZ"):
        out[f"rotationRate{a}"] = gyro[:, i] if gyro is not None else np.zeros(len(t))
        out[f"gravity{a}"] = grav_unit[:, i]
        out[f"acceleration{a}"] = user_g[:, i]
    if orient is not None:
        for i, a in enumerate("XYZ"):
            out[f"orient{a}"] = orient[:, i]
    return pd.DataFrame(out)


def _to_g(x):
    """Scale an accel array to g, judging units by the median magnitude (raw) or spread (user)."""
    mag = np.median(np.linalg.norm(x, axis=1))
    return (x / G, "m/s2") if mag > 3.0 else (x, "g")


def _gyro_rad(x):
    if x is None:
        return None, None
    if np.percentile(np.abs(x), 99) > 25:      # > 25 rad/s is implausible for a wrist -> deg/s
        return np.radians(x), "deg/s"
    return x, "rad/s"


# ------------------------------------------------------------------ readers
def read_sensorlogger_watch(members, info):
    df = _csv(_find(members, "WristMotion.csv"))
    missing = [c for c in COLS + ["seconds_elapsed"] if c not in df.columns]
    if missing:
        raise ValueError(f"WristMotion.csv lacks {missing}")
    has_q = all(c in df.columns for c in QUAT)
    info.update(format="sensorlogger-watch", placement=info.get("placement") or "wrist",
                has_gyro=True, gravity="device", has_attitude=has_q)
    return df[["seconds_elapsed"] + COLS + (QUAT if has_q else [])], info


def read_sensorlogger_phone(members, info, accel_convention=None):
    sign = -1.0 if (accel_convention or ("specific_force" if info.get("platform") == "android" else "gravity")) \
        == "specific_force" else 1.0
    gyr = _find(members, "Gyroscope.csv")
    gyro_df = _csv(gyr) if gyr else None
    acc_l, grav_l = _find(members, "Accelerometer.csv"), _find(members, "Gravity.csv")
    if acc_l and grav_l:
        a, gr = _csv(acc_l), _csv(grav_l)
        t = _time_seconds(a["seconds_elapsed"] if "seconds_elapsed" in a else a.iloc[:, 0])
        tg = _time_seconds(gr["seconds_elapsed"] if "seconds_elapsed" in gr else gr.iloc[:, 0])
        A = sign * a[["x", "y", "z"]].to_numpy(float)
        Gv = sign * np.column_stack([np.interp(t, tg, gr[c]) for c in "xyz"])
        Gv, unit = _to_g(Gv)
        A = A / G if unit == "m/s2" else A
        gu = Gv / np.linalg.norm(Gv, axis=1, keepdims=True)
        info.update(gravity="device")
    else:
        tot = _find(members, "TotalAcceleration.csv", "AccelerometerUncalibrated.csv")
        if tot is None:
            raise ValueError("phone export needs Accelerometer+Gravity or TotalAcceleration")
        a = _csv(tot)
        t = _time_seconds(a["seconds_elapsed"] if "seconds_elapsed" in a else a.iloc[:, 0])
        raw, _ = _to_g(sign * a[["x", "y", "z"]].to_numpy(float))
        gy = None
        if gyro_df is not None:
            tg = _time_seconds(gyro_df["seconds_elapsed"] if "seconds_elapsed" in gyro_df else gyro_df.iloc[:, 0])
            gy = np.column_stack([np.interp(t, tg, gyro_df[c]) for c in "xyz"])
        gu, A, orient = separate_gravity(t, raw, gy)
        info.update(gravity="estimated")
    gyro = None
    if gyro_df is not None:
        tg = _time_seconds(gyro_df["seconds_elapsed"] if "seconds_elapsed" in gyro_df else gyro_df.iloc[:, 0])
        gyro = np.column_stack([np.interp(t, tg, gyro_df[c]) for c in "xyz"])
        gyro, _ = _gyro_rad(gyro)
    info.update(format="sensorlogger-phone", placement=info.get("placement") or "unknown",
                has_gyro=gyro is not None)
    return _canonical(t, A, gu, gyro, locals().get("orient")), info


def read_imu_csv(df, info, accel_convention=None):
    tcol = _pick(df, "seconds_elapsed", "time_s", "t", "time", "timestamp", "seconds", "ts", "epoch")
    if tcol is None:
        tcol = df.columns[0]
    t = _time_seconds(df[tcol].to_numpy(float))
    acc_c = _triplet(df, ["acc", "accel", "accelerometer", "acceleration", "a", "linacc", ""])
    if acc_c is None:
        raise ValueError(f"no accelerometer x/y/z columns in {list(df.columns)}")
    gyro_c = _triplet(df, ["gyro", "gyroscope", "rotationrate", "rot", "w", "g"])
    grav_c = _triplet(df, ["gravity", "grav"])
    if gyro_c and grav_c and gyro_c == grav_c:
        gyro_c = None
    A = df[acc_c].to_numpy(float)
    gyro, gunit = _gyro_rad(df[gyro_c].to_numpy(float) if gyro_c else None)
    conv = accel_convention or "specific_force"
    sign = -1.0 if conv == "specific_force" else 1.0
    if grav_c:
        Gv, unit = _to_g(sign * df[grav_c].to_numpy(float))
        A = sign * (A / G if unit == "m/s2" else A)
        gu = Gv / np.linalg.norm(Gv, axis=1, keepdims=True)
        info["gravity"] = "device"
    else:
        raw, unit = _to_g(sign * A)
        # no gravity in the signal reads ~0 g; explosive work (wall balls, punches) can push
        # the median well above 1 g, which is fine
        if np.median(np.linalg.norm(raw, axis=1)) < 0.5:
            raise ValueError("accelerometer has no gravity in it and no gravity columns: can't find 'up'")
        gu, A, orient = separate_gravity(t, raw, gyro)
        info["gravity"] = "estimated" + ("" if gyro is not None else " (no gyro: low-pass)")
    orient = locals().get("orient")
    order = np.argsort(t)
    info.update(format="imu-csv", accel_convention=conv, gyro_unit=gunit,
                placement=info.get("placement") or "unknown", has_gyro=gyro is not None)
    return _canonical(t[order], A[order], gu[order], gyro[order] if gyro is not None else None,
                      orient[order] if orient is not None else None), info


def detect(members):
    names = {k.lower() for k in members}
    if "wristmotion.csv" in names:
        return "sensorlogger-watch"
    if ("accelerometer.csv" in names and "gravity.csv" in names) or "totalacceleration.csv" in names:
        return "sensorlogger-phone"
    return "imu-csv"


def load_sensor(src, fmt="auto", accel_convention=None, placement=None):
    """Read any supported source -> (canonical DataFrame at native rate, info dict)."""
    members = _members(src)
    info = read_sensorlogger_meta(members)
    if placement:
        info["placement"] = placement
    fmt = detect(members) if fmt == "auto" else fmt
    if fmt == "sensorlogger-watch":
        df, info = read_sensorlogger_watch(members, info)
    elif fmt == "sensorlogger-phone":
        df, info = read_sensorlogger_phone(members, info, accel_convention)
    elif fmt == "imu-csv":
        csvs = [k for k in members if k.lower().endswith((".csv", ".txt"))
                and k.lower() not in ("metadata.csv", "manifest.csv", "tags.csv")]
        if not csvs:
            raise ValueError(f"no CSV in {src}")
        best = max(csvs, key=lambda k: len(members[k]()))   # the big file is the IMU stream
        df, info = read_imu_csv(_csv(members[best]), info, accel_convention)
        info["file"] = best
    else:
        raise ValueError(f"unknown sensor format {fmt}")
    hr = _find(members, "HeartRate.csv")
    if hr is not None:
        h = _csv(hr)
        if "bpm" in h.columns:
            info["heart_rate_samples"] = int(len(h))
    info["n_samples"] = int(len(df))
    return df, info


def describe(info):
    """Short human description for reports: 'Apple Watch (Sensor Logger), wrist'."""
    dev = info.get("device") or ""
    fmt = info.get("format", "")
    if fmt == "sensorlogger-watch":
        name = "Apple Watch" + (f" {dev}" if dev and "watch" not in dev.lower() else "")
    elif fmt == "sensorlogger-phone":
        name = f"{info.get('platform', 'phone').capitalize()} phone"
    else:
        name = dev or "IMU"
    via = f" via {info['app']}" if info.get("app") else ""
    return f"{name}{via}, {info.get('placement', 'unknown')} placement"
