"""Benchmark on public data: the RecoFit dataset (Microsoft, CDLA-Permissive-2.0).

    python -m vbt.public --data /path/outside/repo            # download (1.56 GB) + benchmark
    python -m vbt.public --data /path/outside/repo --record   # also append to data/eval/public_history.jsonl

Morris et al., "RecoFit: using a wearable sensor to find, recognize, and count repetitive
exercises", CHI 2014. Data: https://github.com/microsoft/Exercise-Recognition-from-Wearable-Sensors
(forearm-worn accelerometer + gyroscope, 50 Hz, 94 recorded people). Raw accelerometer only,
so gravity is estimated by vbt.sensors - the same path any third-party IMU takes.

Why: the user's own workouts test tailoring; this tests whether the algorithms hold across
many people and exercises. Published reference: RecoFit counted within +-1 rep 93% of the time.

Two tests:
* rep counting: sets with a true rep count, strength lifts only (cardio, slams, punches and
  floor work excluded); exact / within +-1 / mean error.
* exercise recognition across people: leave-one-person-out, nearest prototype on the 16
  summary features, with and without rep shape.
The raw data never enters the repo.
"""
import argparse
import hashlib
import json
import pickle
import subprocess
import sys
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .sensors import read_imu_csv
from .ingest import downsample
from .workout import analyze_set, exercise_key, set_features, rep_shape, FEATURES

URL = ("https://media.githubusercontent.com/media/microsoft/Exercise-Recognition-from-Wearable-Sensors/"
       "main/exercise_data.50.0000_singleonly.mat")
SHA256 = "70eea039362555daccb00065e3775ce37fff421f18d533c60933a9d859f33f8a"
SIZE = 1562300284
ROOT = Path(__file__).resolve().parents[1]
G = 9.80665
NOT_STRENGTH = ("Punches", "Burpee", "Butterfly", "Kettlebell", "Russian", "Slam", "Wall Ball",
                "Lawnmower", "V-up", "Crunch", "Jump", "Running", "Walk", "machine", "Plank", "Stretch",
                "Non-Exercise", "Tap ", "Device", "Band Adjustment", "Note", "Unlisted", "Invalid",
                "Initial", "Rest", "Wall Squat", "Arm straight", "Dip", "Sit-up", "Side Plank", "pose")


def fetch(data_dir):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    f = data_dir / "recofit_singleonly.mat"
    for _ in range(30):
        if f.exists() and f.stat().st_size >= SIZE:
            break
        subprocess.run(["curl", "-sSL", "-C", "-", "--max-time", "600", "-o", str(f), URL], check=False)
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    if h.hexdigest() != SHA256:
        raise RuntimeError(f"checksum mismatch for {f}; delete it and retry")
    return f


def convert(mat, out):
    """Flatten the Matlab cell array into one record per recorded segment."""
    import scipy.io as sio
    m = sio.loadmat(mat, squeeze_me=True, struct_as_record=False)
    sd = m["subject_data"]
    recs = []
    for i in range(sd.shape[0]):
        for j in range(sd.shape[1]):
            c = sd[i, j]
            if c is None or not getattr(c, "size", 0):
                continue
            items = c if isinstance(c, np.ndarray) and c.dtype == object else [c]
            for x in np.atleast_1d(items):
                if not hasattr(x, "data"):
                    continue
                reps = None if np.size(x.activityReps) == 0 else float(np.atleast_1d(x.activityReps)[0])
                recs.append({"subject": int(x.subjectID), "activity": str(x.activityName), "reps": reps,
                             "accel": np.asarray(x.data.accelDataMatrix, float),
                             "gyro": np.asarray(getattr(x.data, "gyroDataMatrix", np.zeros((0, 4))), float)})
    pickle.dump(recs, open(out, "wb"))
    return recs


def canonical(rec):
    a, g = rec["accel"], rec["gyro"]
    n = min(len(a), len(g)) if len(g) else len(a)
    df = pd.DataFrame({"t": a[:n, 0], "acc_x": a[:n, 1], "acc_y": a[:n, 2], "acc_z": a[:n, 3]})
    if len(g):
        df["gyro_x"], df["gyro_y"], df["gyro_z"] = g[:n, 1], g[:n, 2], g[:n, 3]
    can, _ = read_imu_csv(df, {}, "specific_force")
    small, _ = downsample(can)
    return {"t": small.seconds_elapsed.to_numpy(),
            "acc": small[["accelerationX", "accelerationY", "accelerationZ"]].to_numpy() * G,
            "grav": small[["gravityX", "gravityY", "gravityZ"]].to_numpy() * G,
            "gyro": small[["rotationRateX", "rotationRateY", "rotationRateZ"]].to_numpy()}


def strength(rec):
    return not any(k.lower() in rec["activity"].lower() for k in NOT_STRENGTH)


def bench_reps(recs):
    rows = [r for r in recs if strength(r) and r["reps"] is not None and np.isfinite(r["reps"]) and r["reps"] > 0]
    err, per = [], defaultdict(list)
    for r in rows:
        d = canonical(r)
        n = analyze_set(d, d["t"][0] + 0.5, d["t"][-1] - 0.5, exercise_key(r["activity"]), fs=50.0)["rep_count"]
        err.append(n - r["reps"])
        per[r["activity"]].append(n - r["reps"])
    e = np.array(err)
    return {"n_sets": int(len(e)), "n_people": len({r["subject"] for r in rows}),
            "exact_pct": round(100 * float(np.mean(e == 0)), 1),
            "within1_pct": round(100 * float(np.mean(np.abs(e) <= 1)), 1),
            "mean_error": round(float(e.mean()), 2),
            "per_exercise": {a: {"n": len(v), "exact_pct": round(100 * float(np.mean(np.array(v) == 0)), 1),
                                 "within1_pct": round(100 * float(np.mean(np.abs(np.array(v)) <= 1)), 1),
                                 "mean_error": round(float(np.mean(v)), 2)} for a, v in per.items()}}


def bench_recognition(recs, min_people=4, shape_weight=0.5):
    """Leave-one-person-out nearest prototype, with and without rep shape."""
    rows = [r for r in recs if strength(r) and len(r["accel"]) > 50 * 8]
    people = defaultdict(set)
    for r in rows:
        people[r["activity"]].add(r["subject"])
    rows = [r for r in rows if len(people[r["activity"]]) >= min_people]
    X, SH, y, subj = [], [], [], []
    for r in rows:
        d = canonical(r)
        res = analyze_set(d, d["t"][0] + 0.5, d["t"][-1] - 0.5, exercise_key(r["activity"]), fs=50.0)
        reps = res["reps"]
        span = (reps[0]["t_start"], reps[-1]["t_end"]) if len(reps) >= 3 else (d["t"][0], d["t"][-1])
        f = set_features(d, *span)
        X.append(np.nan_to_num([f[k] for k in FEATURES]))
        SH.append(np.array(rep_shape(d, reps)) if len(reps) >= 3 else None)
        y.append(r["activity"])
        subj.append(r["subject"])
    X, y, subj = np.array(X), np.array(y), np.array(subj)
    out = {"n_sets": int(len(y)), "n_exercises": int(len(set(y))), "n_people": int(len(set(subj)))}
    for name, w in (("summary_only", 0.0), ("with_rep_shape", shape_weight)):
        hit = 0
        for p in sorted(set(subj)):
            tr, te = subj != p, subj == p
            mu = X[tr].mean(0)
            sd = np.maximum(X[tr].std(0), 1e-3 + 0.1 * np.abs(mu))
            labs = sorted(set(y[tr]))
            C = {c: ((X[tr][y[tr] == c] - mu) / sd).mean(0) for c in labs}
            S = {}
            for c in labs:
                sh = [SH[i] for i in np.flatnonzero(tr & (y == c)) if SH[i] is not None]
                if sh:
                    S[c] = np.mean(sh, 0)
            scale = np.median([np.sqrt(((SH[i] - S[y[i]]) ** 2).mean()) for i in np.flatnonzero(tr)
                               if SH[i] is not None and y[i] in S]) or 1.0
            for i in np.flatnonzero(te):
                z = (X[i] - mu) / sd
                best = min(labs, key=lambda c: np.sqrt(((z - C[c]) ** 2).mean())
                           + (w * np.sqrt(((SH[i] - S[c]) ** 2).mean()) / scale
                              if w and SH[i] is not None and c in S else 0.0))
                hit += best == y[i]
        out[name + "_pct"] = round(100 * hit / len(y), 1)
    return out


def main(argv=None):
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(prog="python -m vbt.public")
    ap.add_argument("--data", required=True, help="folder OUTSIDE the repo for the 1.56 GB download")
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args(argv)
    d = Path(a.data)
    pk = d / "recofit_sets.pkl"
    if pk.exists():
        recs = pickle.load(open(pk, "rb"))
    else:
        recs = convert(fetch(d), pk)
    res = {"reps": bench_reps(recs), "recognition": bench_recognition(recs)}
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True).stdout.strip()
        if subprocess.run(["git", "status", "--porcelain", "--", "vbt"], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip():
            commit += "+uncommitted"
    except Exception:
        commit = None
    res = {"when": datetime.now(timezone.utc).isoformat(timespec="seconds"), "commit": commit,
           "dataset": "RecoFit (Morris et al. 2014), CDLA-Permissive-2.0", **res}
    print(json.dumps({k: v for k, v in res.items()} | {"reps": {k: v for k, v in res["reps"].items()
                                                                if k != "per_exercise"}}, indent=1))
    if a.record:
        p = ROOT / "data/eval/public_history.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    sys.exit(main())
