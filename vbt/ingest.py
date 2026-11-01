"""Turn a Sensor Logger export plus a Hevy text into a session folder.

    data/sessions/<id>/
        hevy.txt            Hevy "share workout" text
        sensor_25hz.csv.gz  WristMotion columns (g, rad/s), anti-aliased, 25 Hz
        meta.json           recording start (UTC + local tz), device, duration, source
        sets.json           set windows; "source": "auto" | "reviewed" | "manual"
        notes.json          optional per-session commentary for the report

25 Hz keeps everything the analysis uses (all lifting motion is < 6 Hz) at
~1/10 of the raw size, so every session can live in git and be re-analysed
whenever the algorithms improve.
"""
import io
import json
import re
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

STORE_HZ = 25.0
COLS = ["rotationRateX", "rotationRateY", "rotationRateZ", "gravityX", "gravityY", "gravityZ",
        "accelerationX", "accelerationY", "accelerationZ"]


def _read_member(src, name):
    """Read a file from a zip, a folder, or return None."""
    src = Path(src)
    if src.is_file() and zipfile.is_zipfile(src):
        with zipfile.ZipFile(src) as z:
            hits = [n for n in z.namelist() if Path(n).name.lower() == name.lower()]
            return z.read(hits[0]) if hits else None
    if src.is_dir():
        p = next((q for q in src.rglob("*") if q.name.lower() == name.lower()), None)
        return p.read_bytes() if p else None
    if src.is_file() and src.name.lower() == name.lower():
        return src.read_bytes()
    return None


def read_metadata(src):
    raw = _read_member(src, "Metadata.csv")
    if raw is None:
        return {}
    row = pd.read_csv(io.BytesIO(raw)).iloc[0].to_dict()
    out = {"device": str(row.get("device name", "")), "timezone": str(row.get("recording timezone", "")),
           "app_version": str(row.get("appVersion", ""))}
    if "recording epoch time" in row:
        start = datetime.fromtimestamp(float(row["recording epoch time"]) / 1000, tz=timezone.utc)
        out["start_utc"] = start.isoformat()
    return out


def downsample(df, fs=STORE_HZ):
    """Anti-alias (8 Hz low-pass) and resample WristMotion columns onto a uniform grid."""
    t = df["seconds_elapsed"].to_numpy(float)
    order = np.argsort(t)
    t = t[order]
    fs_in = 1.0 / np.median(np.diff(t))
    tu = np.arange(t[0], t[-1], 1.0 / fs)
    b, a = butter(4, min(8.0, 0.4 * fs) / (fs_in / 2), "low")
    out = {"seconds_elapsed": tu}
    for c in COLS:
        x = df[c].to_numpy(float)[order]
        out[c] = np.interp(tu, t, filtfilt(b, a, x))
    g = np.column_stack([out["gravityX"], out["gravityY"], out["gravityZ"]])
    g /= np.linalg.norm(g, axis=1, keepdims=True)          # keep gravity a unit vector
    out["gravityX"], out["gravityY"], out["gravityZ"] = g.T
    return pd.DataFrame(out), fs_in


def hevy_date(text):
    """'Sunday, Nov 01, 2026 at 10:37am' -> datetime (naive local)."""
    m = re.search(r"([A-Z][a-z]{2}) (\d{1,2}), (\d{4}) at (\d{1,2}):(\d{2})\s*([ap]m)", text)
    if not m:
        return None
    return datetime.strptime(" ".join(m.groups()[:5]) + m[6], "%b %d %Y %I %M%p")


def ingest(sensor_src, hevy_text, root="data/sessions", session_id=None, overwrite=False):
    """Create/refresh a session folder. Returns its Path."""
    raw = _read_member(sensor_src, "WristMotion.csv")
    if raw is None:
        raise FileNotFoundError(f"No WristMotion.csv in {sensor_src}")
    df = pd.read_csv(io.BytesIO(raw))
    missing = [c for c in COLS + ["seconds_elapsed"] if c not in df.columns]
    if missing:
        raise ValueError(f"WristMotion.csv lacks columns {missing}")
    meta = read_metadata(sensor_src)
    when = hevy_date(hevy_text)
    sid = session_id or (when.strftime("%Y-%m-%d") if when else meta.get("start_utc", "session")[:10])
    d = Path(root) / sid
    if d.exists() and not overwrite and (d / "sensor_25hz.csv.gz").exists():
        raise FileExistsError(f"{d} exists (pass overwrite=True or another session_id)")
    d.mkdir(parents=True, exist_ok=True)
    small, fs_in = downsample(df)
    small.to_csv(d / "sensor_25hz.csv.gz", index=False, float_format="%.5f", compression="gzip")
    (d / "hevy.txt").write_text(hevy_text)
    title = hevy_text.strip().splitlines()[0] if hevy_text.strip() else ""
    meta.update(session_id=sid, title=title, hevy_start_local=when.isoformat() if when else None,
                raw_rate_hz=round(fs_in, 2), stored_rate_hz=STORE_HZ,
                duration_s=round(float(small["seconds_elapsed"].iloc[-1]), 1),
                first_sample_s=round(float(small["seconds_elapsed"].iloc[0]), 3),
                n_raw_samples=int(len(df)), source=Path(str(sensor_src)).name)
    if meta.get("start_utc") and when:
        # Hevy start (local wall clock) vs recording start in the recording's tz offset
        try:
            from zoneinfo import ZoneInfo
            start = datetime.fromisoformat(meta["start_utc"]).astimezone(ZoneInfo(meta["timezone"]))
            meta["start_local"] = start.isoformat()
            meta["hevy_minus_recording_s"] = round((when - start.replace(tzinfo=None)) / timedelta(seconds=1))
        except Exception:
            pass
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    return d
