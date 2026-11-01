"""Full analysis of the 2026-11-01 whole-workout recording.

python analysis/workout_2026_11_01.py path/to/WristMotion.csv [out.json]

Inputs : WristMotion.csv (Sensor Logger, 100 Hz), data/hevy_2026-11-01.txt,
         data/sets_2026-11-01.json (annotated set windows, Hevy order).
Outputs: analysis/out/workout_2026-11-01.json  (everything the report page plots)
         data/wristmotion_2026-11-01_10hz.csv.gz (decimated derived signals)
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vbt.hevy import parse_hevy_text as parse_hevy  # noqa: E402
from vbt import (load_watch_csv, analyze_set, set_features, loso_classify,  # noqa: E402
                 exercise_key, PROFILES, find_sets, split_sets, estimate_1rm)
from vbt.core import vertical_acceleration  # noqa: E402
from vbt.workout import FEATURES, FEATURE_GROUPS, resample, find_turnarounds  # noqa: E402
import vbt.workout as W  # noqa: E402

NAMES = {"bench": "Bench press", "deadlift": "Deadlift", "bss": "Bulgarian split squat",
         "pulldown": "Wide-grip pulldown", "pushdown": "Rope pushdown",
         "curl": "Hammer curl", "lateral": "Lateral raise"}
R = lambda x, n=3: None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), n)  # noqa: E731


def lp(x, fs, fc):
    b, a = butter(4, fc / (fs / 2), "low")
    return filtfilt(b, a, x, axis=0)


def main(csv, out_json):
    data = load_watch_csv(csv)
    # load_watch_csv zeroes the clock; annotations use Sensor Logger's seconds_elapsed
    with open(csv) as f:
        hdr = f.readline().strip().split(",")
        data["t"] = data["t"] + float(f.readline().split(",")[hdr.index("seconds_elapsed")])
    hevy = parse_hevy((ROOT / "data/hevy_2026-11-01.txt").read_text())
    windows = json.loads((ROOT / "data/sets_2026-11-01.json").read_text())["sets"]
    assert len(hevy) == len(windows) == 22
    t = data["t"]
    T_END = float(t[-1])

    # ---------------- timeline (decimated) ----------------
    fs = 100.0
    tu = np.arange(0, T_END, 1 / fs)
    a_up = np.interp(tu, t, vertical_acceleration(data["acc"], data["grav"]))
    g_hat = data["grav"] / np.linalg.norm(data["grav"], axis=1, keepdims=True)
    g_u = np.column_stack([np.interp(tu, t, g_hat[:, i]) for i in range(3)])
    gyro_u = np.column_stack([np.interp(tu, t, data["gyro"][:, i]) for i in range(3)])
    den = lp(a_up, fs, 2.0)
    B = 50                                     # 0.5 s bins
    nb = len(tu) // B
    env_lo = a_up[:nb * B].reshape(nb, B).min(1)
    env_hi = a_up[:nb * B].reshape(nb, B).max(1)
    rms = np.sqrt((a_up[:nb * B].reshape(nb, B) ** 2).mean(1))
    den_d = den[:nb * B].reshape(nb, B).mean(1)
    gb = g_u[:nb * B].reshape(nb, B, 3).mean(1)
    timeline = {"dt": 0.5, "lo": [R(x, 2) for x in env_lo], "hi": [R(x, 2) for x in env_hi],
                "den": [R(x, 3) for x in den_d], "rms": [R(x, 3) for x in rms],
                "gx": [R(x, 2) for x in gb[:, 0]], "gy": [R(x, 2) for x in gb[:, 1]],
                "gz": [R(x, 2) for x in gb[:, 2]]}

    # derived 10 Hz file for the repo
    d10 = slice(None, None, 10)
    deriv = np.column_stack([tu[d10], lp(a_up, fs, 4.0)[d10], g_u[d10], lp(gyro_u, fs, 4.0)[d10]])
    import gzip
    with gzip.open(ROOT / "data/wristmotion_2026-11-01_10hz.csv.gz", "wt") as f:
        f.write("t_s,a_vert_ms2,g_x,g_y,g_z,gyro_x,gyro_y,gyro_z\n")
        np.savetxt(f, deriv, fmt="%.3f", delimiter=",")

    # ---------------- per-set analysis ----------------
    sets, rep_rows = [], []
    for i, (w, h) in enumerate(zip(windows, hevy)):
        ex = exercise_key(h["exercise"])
        res = analyze_set(data, w["t_start"], w["t_end"], ex)
        # threshold sensitivity of the rep count
        counts = []
        orig = W.find_turnarounds
        for rp in (0.25, 0.35, 0.45):
            W.find_turnarounds = lambda *a, _rp=rp, **k: orig(*a, **{**k, "rel_prom": _rp})
            counts.append(analyze_set(data, w["t_start"], w["t_end"], ex)["rep_count"])
        W.find_turnarounds = orig
        reps = res["reps"]
        # bouts (split squat legs) = gaps > 2x median rep spacing
        tt = np.array([r["t_turn"] for r in reps])
        edges = [0, len(reps)]
        if PROFILES[ex]["bouts"] > 1 and len(tt) > 2:
            cut = np.flatnonzero(np.diff(tt) > 2.0 * np.median(np.diff(tt)))
            edges = [0, *(cut + 1), len(reps)]
        units = [reps[edges[j]:edges[j + 1]] for j in range(len(edges) - 1)]
        bouts = [len(u) for u in units]
        bouts_clean = [sum(r["ok"] for r in u) for u in units]
        clean = [r for r in reps if r["ok"]]
        span = (reps[0]["t_start"], reps[-1]["t_end"]) if reps else (w["t_start"], w["t_end"])
        feats = set_features(data, *span)
        # drill-down traces at 25 Hz
        d = resample(data, w["t_start"] - 4, w["t_end"] + 4, 25.0, pad=0)
        gh = d["grav"] / np.linalg.norm(d["grav"], axis=1, keepdims=True)
        pos = np.interp(d["t"], res["t"], res["pos"])
        chan = np.interp(d["t"], res["t"], res["chan"])
        a_f = np.interp(d["t"], res["t"], res["a_up"])
        logged = h["reps"]
        sensor = res["rep_count"]
        # Hevy logs split squats per leg; the watch sees the two legs as two bouts.
        # Each unit (set or bout) has an ambiguity interval [clean cycles, all cycles]:
        # flagged edge cycles (unrack, setup pull, re-rack) may or may not be reps.
        verdicts = []
        for lo_n, hi_n in zip(bouts_clean, bouts):
            if hi_n == logged:
                verdicts.append(("match", 0))
            elif lo_n <= logged <= hi_n:
                verdicts.append(("ambiguous", 0))
            elif logged > hi_n:
                verdicts.append(("watch fewer", logged - hi_n))
            else:
                verdicts.append(("watch more", lo_n - logged))
        rank = {"match": 0, "ambiguous": 1, "watch fewer": 2, "watch more": 2}
        flag, delta = max(verdicts, key=lambda v: (rank[v[0]], v[1]))
        s = {"i": i + 1, "exercise": ex, "name": NAMES[ex], "hevy_name": h["exercise"],
             "kg": h["kg"], "logged_reps": logged, "rpe": h["rpe"], "warmup": h["warmup"],
             "window": [w["t_start"], w["t_end"]], "span": [R(span[0], 2), R(span[1], 2)],
             "sensor_reps": sensor, "sensor_range": [min(counts), max(counts)], "bouts": bouts,
             "clean_reps": len(clean), "bouts_clean": bouts_clean,
             "per_leg": ex == "bss", "flag": flag, "delta": delta,
             "mean_velocity": R(res.get("mean_velocity")), "best_velocity": R(res.get("best_mean_velocity")),
             "velocity_loss_pct": R(res.get("velocity_loss_pct"), 1),
             "tut_s": R(res.get("tut_s"), 1), "active_s": R(res.get("active_span_s"), 1),
             "con_s": R(sum(r["concentric_s"] for r in reps), 1),
             "ecc_s": R(sum(r["eccentric_s"] for r in reps), 1),
             "features": {k: R(v, 4) for k, v in feats.items()},
             "reps": [{k: (R(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in reps],
             "trace": {"t0": R(d["t"][0], 2), "dt": 0.04,
                       "pos": [R(x * 100, 1) for x in pos], "a": [R(x, 2) for x in a_f],
                       "chan": [R(x, 3) for x in chan],
                       "gx": [R(x, 3) for x in gh[:, 0]], "gy": [R(x, 3) for x in gh[:, 1]],
                       "gz": [R(x, 3) for x in gh[:, 2]],
                       "wx": [R(x, 2) for x in d["gyro"][:, 0]], "wy": [R(x, 2) for x in d["gyro"][:, 1]],
                       "wz": [R(x, 2) for x in d["gyro"][:, 2]]}}
        sets.append(s)
        for r in reps:
            rf = set_features(data, r["t_start"], r["t_end"])
            rep_rows.append((i, ex, [rf[k] for k in FEATURES]))
        print(f"{i+1:2d} {ex:8s} {h['kg']:5.1f} logged {logged:2d} sensor {sensor:2d} clean {len(clean)} {bouts} {bouts_clean} {counts} {flag}"
              f"  MV {s['mean_velocity']}  loss {s['velocity_loss_pct']}")

    # ---------------- rest / density ----------------
    for a, b in zip(sets, sets[1:]):
        a["rest_after_s"] = R(b["span"][0] - a["span"][1], 1)
    sets[-1]["rest_after_s"] = None
    first, last = sets[0]["span"][0], sets[-1]["span"][1]
    summary = {
        "recording_s": R(T_END, 1), "fs_hz": 100, "first_rep_s": first, "last_rep_s": last,
        "work_span_min": R((last - first) / 60, 1),
        "logged_reps": int(sum(s["logged_reps"] for s in sets)),
        "sensor_reps": int(sum(s["sensor_reps"] for s in sets)),
        "sensor_reps_cmp": R(sum(s["sensor_reps"] / (2 if s["per_leg"] else 1) for s in sets), 1),
        "clean_reps_cmp": R(sum(s["clean_reps"] / (2 if s["per_leg"] else 1) for s in sets), 1),
        "tut_s": R(sum(s["tut_s"] or 0 for s in sets), 0),
        "volume_kg": R(sum(s["kg"] * s["logged_reps"] * (2 if s["per_leg"] else 1) for s in sets), 0),
        "median_rest_s": R(np.median([s["rest_after_s"] for s in sets[:-1]]), 0),
        "n_match": sum(s["flag"] == "match" for s in sets),
        "n_ambiguous": sum(s["flag"] == "ambiguous" for s in sets),
        "n_fewer": sum(s["flag"] == "watch fewer" for s in sets),
        "n_more": sum(s["flag"] == "watch more" for s in sets),
    }
    summary["work_density_pct"] = R(100 * summary["tut_s"] / (last - first), 1)

    # ---------------- motion signatures (time-normalised reps) ----------------
    N = 60
    sig = {}
    for ex in NAMES:
        chans = {k: [] for k in ("pos", "gx", "gy", "gz", "wx", "wy", "wz", "a")}
        for s in sets:
            if s["exercise"] != ex:
                continue
            tr = s["trace"]
            tt = tr["t0"] + np.arange(len(tr["pos"])) * tr["dt"]
            for r in s["reps"]:
                if not r["ok"]:
                    continue
                q = np.linspace(r["t_start"], r["t_end"], N)
                for k in chans:
                    y = np.interp(q, tt, np.array(tr[k], float))
                    if k == "pos":
                        y = y - y[0]
                    chans[k].append(y)
        sig[ex] = {k: {"mean": [R(x, 3) for x in np.mean(v, 0)], "sd": [R(x, 3) for x in np.std(v, 0)]}
                   for k, v in chans.items() if v}
        sig[ex]["n_reps"] = len(chans["pos"])
        sig[ex]["rep_s"] = R(np.median([r["rep_s"] for s in sets if s["exercise"] == ex
                                        for r in s["reps"] if r["ok"]]), 2)

    # ---------------- classifier ----------------
    y = np.array([s["exercise"] for s in sets])
    X = np.array([[s["features"][k] for k in FEATURES] for s in sets], float)
    clf = {"features": FEATURES, "groups": FEATURE_GROUPS, "set_level": {}, "rep_level": {}}
    for gname, cols in [("all", FEATURES), *FEATURE_GROUPS.items()]:
        idx = [FEATURES.index(c) for c in cols]
        for m in ("centroid", "knn"):
            pred, acc = loso_classify(X[:, idx], y, method=m)
            clf["set_level"][f"{gname}|{m}"] = {"acc": R(acc), "pred": pred.tolist()}
    # rep-level, grouped leave-one-SET-out
    Xr = np.array([r[2] for r in rep_rows], float)
    yr = np.array([r[1] for r in rep_rows])
    gr = np.array([r[0] for r in rep_rows])
    for gname, cols in [("all", FEATURES), *FEATURE_GROUPS.items()]:
        idx = [FEATURES.index(c) for c in cols]
        Xi = np.nan_to_num(Xr[:, idx])
        pred = np.empty(len(yr), dtype=object)
        for g in np.unique(gr):
            tr, te = gr != g, gr == g
            mu, sd = Xi[tr].mean(0), Xi[tr].std(0) + 1e-9
            Z, z = (Xi[tr] - mu) / sd, (Xi[te] - mu) / sd
            labs = np.unique(yr[tr])
            cents = np.array([Z[yr[tr] == c].mean(0) for c in labs])
            pred[te] = labs[np.argmin(((z[:, None, :] - cents[None]) ** 2).sum(2), 1)]
        acc = float((pred == yr).mean())
        set_vote = [max(set(pred[gr == g]), key=list(pred[gr == g]).count) for g in np.unique(gr)]
        clf["rep_level"][gname] = {"acc": R(acc), "n": int(len(yr)),
                                   "set_vote_acc": R(np.mean(np.array(set_vote) == y))}
    try:
        from sklearn.ensemble import RandomForestClassifier
        pred = []
        for i in range(len(y)):
            tr = np.arange(len(y)) != i
            m = RandomForestClassifier(300, random_state=0).fit(X[tr], y[tr])
            pred.append(m.predict(X[i:i + 1])[0])
        clf["set_level"]["all|rf"] = {"acc": R(np.mean(np.array(pred) == y)), "pred": pred}
        imp = RandomForestClassifier(500, random_state=0).fit(X, y).feature_importances_
        clf["rf_importance"] = {k: R(v, 4) for k, v in zip(FEATURES, imp)}
    except ImportError:
        pass
    # PCA of standardised set features
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    pc = Z @ Vt[:2].T
    clf["pca"] = {"xy": [[R(a, 3), R(b, 3)] for a, b in pc],
                  "var": [R(v, 3) for v in (S ** 2 / (S ** 2).sum())[:2]],
                  "loadings": {k: [R(Vt[0, j], 3), R(Vt[1, j], 3)] for j, k in enumerate(FEATURES)}}
    clf["labels"] = y.tolist()
    print("classifier", {k: v["acc"] for k, v in clf["set_level"].items()}, clf["rep_level"])

    # ---------------- load-velocity ----------------
    lv = {}
    for ex in ("bench", "deadlift"):
        rows = [s for s in sets if s["exercise"] == ex and s["best_velocity"]]
        loads = [s["kg"] for s in rows]
        best = [s["best_velocity"] for s in rows]
        mean = [s["mean_velocity"] for s in rows]
        fit = estimate_1rm(loads, best, lift=ex)
        fit_m = estimate_1rm(loads, mean, lift=ex)
        lv[ex] = {"loads": loads, "best": best, "mean": mean,
                  "fit_best": {k: R(v, 4) for k, v in fit.items()},
                  "fit_mean": {k: R(v, 4) for k, v in fit_m.items()}}
        print(ex, "e1RM best-rep", R(fit["e1rm"], 1), "r2", R(fit["r2"]), "| mean", R(fit_m["e1rm"], 1))

    # ---------------- set detection benchmark ----------------
    auto = find_sets(data)
    old = split_sets(data)
    def overlaps(c, w):
        return min(w[1], c[1]) - max(w[0], c[0]) > 0
    tp_sets = [j for j, s in enumerate(sets) if any(overlaps((a["t_start"], a["t_end"]), s["window"]) for a in auto)]
    fp = [a for a in auto if not any(overlaps((a["t_start"], a["t_end"]), s["window"]) for s in sets)]
    detect = {"auto": [{k: R(v, 2) for k, v in a.items()} for a in auto],
              "recall": len(tp_sets), "n_auto": len(auto), "false_pos": len(fp),
              "missed": [j + 1 for j in range(len(sets)) if j not in tp_sets],
              "split_sets_n": len(old),
              "split_sets_longest_min": R(max((b - a) for a, b in old) / 60, 1) if old else None}
    print("detect", {k: v for k, v in detect.items() if k != "auto"})

    out = {"summary": summary, "timeline": timeline, "sets": sets, "signatures": sig,
           "classifier": clf, "load_velocity": lv, "detection": detect,
           "names": NAMES}
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(out, separators=(",", ":")))
    print("wrote", out_json, Path(out_json).stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ROOT / "analysis/out/workout_2026-11-01.json")
