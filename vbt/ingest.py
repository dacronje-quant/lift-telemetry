"""Turn a sensor recording plus a workout log into a session folder.

    data/sessions/<id>/
        log.json            normalised workout log (vbt.logs): title, start time, sets
        log_source.<ext>    the log exactly as it was shared (app export, CSV, typed notes, ...)
        sensor_25hz.csv.gz  canonical motion table (vbt.sensors), anti-aliased, 25 Hz
        meta.json           recording start (UTC + local tz), sensor description, source formats
        sets.json           set windows; "source": "auto" | "reviewed" | "manual"
        notes.json          optional per-session commentary for the report

Any sensor vbt.sensors can read and any log vbt.logs can parse works. 25 Hz
keeps everything the analysis uses (lifting motion is < 6 Hz) at ~1/10 of
the raw size, so every session can live in git and be re-analysed whenever
the algorithms improve.
"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from .sensors import COLS, load_sensor, describe
from .logs import parse_log

STORE_HZ = 25.0
ORIENT = ["orientX", "orientY", "orientZ"]


def downsample(df, fs=STORE_HZ):
    """Anti-alias (8 Hz low-pass) and resample canonical columns onto a uniform grid."""
    t = df["seconds_elapsed"].to_numpy(float)
    order = np.argsort(t)
    t = t[order]
    fs_in = 1.0 / np.median(np.diff(t))
    tu = np.arange(t[0], t[-1], 1.0 / fs)
    cols = COLS + [c for c in ORIENT if c in df.columns]
    out = {"seconds_elapsed": tu}
    if fs_in > 2.5 * fs:
        b, a = butter(4, min(8.0, 0.4 * fs) / (fs_in / 2), "low")
        filt = lambda x: filtfilt(b, a, x)  # noqa: E731
    else:
        filt = lambda x: x  # noqa: E731   source already slow: just interpolate
    for c in cols:
        out[c] = np.interp(tu, t, filt(df[c].to_numpy(float)[order]))
    for trip in (["gravityX", "gravityY", "gravityZ"], ORIENT):
        if trip[0] in out:
            g = np.column_stack([out[c] for c in trip])
            g /= np.linalg.norm(g, axis=1, keepdims=True)      # keep unit vectors unit
            for i, c in enumerate(trip):
                out[c] = g[:, i]
    return pd.DataFrame(out), fs_in


def ingest(sensor_src, log_src, root="data/sessions", session_id=None, overwrite=False,
           log_format="auto", sensor_format="auto", accel_convention=None, placement=None,
           units=None):
    """Create/refresh a session folder. Returns its Path.

    log_src: path to the log file, or the log text itself.
    """
    p = Path(str(log_src)) if len(str(log_src)) < 4096 and "\n" not in str(log_src) else None
    if p is not None and p.exists():
        log_text, log_ext = p.read_text(), p.suffix or ".txt"
    else:
        log_text, log_ext = str(log_src), ".txt"
    sets, info = parse_log(log_text, log_format, units)
    df, sinfo = load_sensor(sensor_src, sensor_format, accel_convention, placement)
    start_utc = None
    if sinfo.get("start_epoch_ms"):
        start_utc = datetime.fromtimestamp(sinfo["start_epoch_ms"] / 1000, tz=timezone.utc)
    when = datetime.fromisoformat(info["start_local"]) if info.get("start_local") else None
    sid = session_id or (when.strftime("%Y-%m-%d") if when else
                         start_utc.strftime("%Y-%m-%d") if start_utc else datetime.now().strftime("%Y-%m-%d"))
    d = Path(root) / sid
    if d.exists() and not overwrite and (d / "sensor_25hz.csv.gz").exists():
        raise FileExistsError(f"{d} exists (pass overwrite=True or another session_id)")
    d.mkdir(parents=True, exist_ok=True)
    small, fs_in = downsample(df)
    small.to_csv(d / "sensor_25hz.csv.gz", index=False, float_format="%.5f", compression="gzip")
    (d / f"log_source{log_ext}").write_text(log_text)
    (d / "log.json").write_text(json.dumps({**info, "sets": sets}, indent=1, ensure_ascii=False))
    meta = {"session_id": sid, "title": info.get("title"), "log_source": info["source"],
            "log_start_local": info.get("start_local"),
            "sensor": describe(sinfo), "sensor_info": {k: v for k, v in sinfo.items() if k != "start_epoch_ms"},
            "device": sinfo.get("device"), "timezone": sinfo.get("timezone") or None,
            "start_utc": start_utc.isoformat() if start_utc else None,
            "raw_rate_hz": round(fs_in, 2), "stored_rate_hz": STORE_HZ,
            "duration_s": round(float(small["seconds_elapsed"].iloc[-1] - small["seconds_elapsed"].iloc[0]), 1),
            "first_sample_s": round(float(small["seconds_elapsed"].iloc[0]), 3),
            "n_raw_samples": int(len(df)), "source": Path(str(sensor_src)).name}
    if start_utc and when and meta["timezone"]:
        try:
            from zoneinfo import ZoneInfo
            start = start_utc.astimezone(ZoneInfo(meta["timezone"]))
            meta["start_local"] = start.isoformat()
            meta["log_minus_recording_s"] = round((when - start.replace(tzinfo=None)) / timedelta(seconds=1))
        except Exception:
            pass
    (d / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    return d
