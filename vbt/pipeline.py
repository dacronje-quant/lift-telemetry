"""One workout in, one report out.

    python -m vbt.pipeline new --sensor recording.zip --log workout.txt
    python -m vbt.pipeline run 2026-11-01            # re-analyse an existing session
    python -m vbt.pipeline run --all                 # after an algorithm change
    python -m vbt.pipeline review 2026-11-01 out.png # rep-marker grid for a human check

Steps for a session folder (see vbt.ingest):
1. set windows: sets.json if a human checked them ("manual"/"reviewed"),
   otherwise log-guided detection (vbt.align) with an exercise model
   trained on all *other* sessions, saved as source "auto";
2. per-set reps, velocity, tempo, verdict vs. the logged count;
3. session-level summary, motion signatures, classifier checks,
   load-velocity, history from the library, auto-generated findings;
4. analysis.json (compact, feeds the library) + reports/<id>.html.
"""
import argparse
import json
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

from .io import load_watch_csv
from .logs import parse_log
from .core import vertical_acceleration
from .workout import (PROFILES, FEATURES, FEATURE_GROUPS, analyze_set, set_features,
                      loso_classify, exercise_key, count_verdict, resample, _lp)
from . import workout as W
from .align import detect_sets, evaluate_alignment
from .loadvelocity import estimate_1rm
from . import library
from . import personal
from .infer import shadow_test

ROOT = Path(__file__).resolve().parents[1]
SESSIONS = ROOT / "data/sessions"
REPORTS = ROOT / "reports"
TEMPLATE = Path(__file__).resolve().parent / "templates/report.html"
TRUSTED = ("manual", "reviewed")
LOG_NAMES = {"hevy": "Hevy log", "strong": "Strong log", "csv": "workout log", "text": "workout notes",
             "json": "workout log"}


def kg(x):
    return f"{x:g}"


def R(x, n=3):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return x
    return round(x, n) if np.isfinite(x) else None


KEY_CODES = {"bench": "BP", "deadlift": "DL", "rdl": "RDL", "squat": "SQ", "ohp": "OHP", "bss": "BSS",
             "lunge": "LUN", "hipthrust": "HT", "pulldown": "PD", "row": "ROW", "pushdown": "PUSH",
             "curl": "CURL", "lateral": "LAT", "extension": "EXT", "fly": "FLY"}


def short_code(name, taken, key=None):
    """'Bench Press (Barbell)' -> 'BP'; unique within the session."""
    if key in KEY_CODES and KEY_CODES[key] not in taken:
        return KEY_CODES[key]
    base = re.sub(r"\(.*?\)", "", name)
    words = [w for w in re.findall(r"[A-Za-z]+", base) if w.lower() not in ("the", "with", "and")]
    code = "".join(w[0] for w in words[:3]).upper() or "EX"
    if len(code) == 1:
        code = words[0][:3].upper()
    c, n = code, 2
    while c in taken:
        c, n = f"{code}{n}", n + 1
    return c


def report_title(title_line):
    """'Sunday — Full Body C | Enhanced Comeback W1–2' -> 'Full Body C'."""
    t = title_line.split(" — ")[-1] if " — " in title_line else title_line
    return t.split(" | ")[0].strip()[:48]


def logged_sets(log):
    """log: dict from log.json (or raw log text for older sessions)."""
    sets = [dict(s) for s in log["sets"]] if isinstance(log, dict) else parse_log(log)[0]
    for s in sets:
        s["exercise_key"] = exercise_key(s["exercise"])
        s["label"] = s["exercise"]
    return sets


def load_session(sid, root=SESSIONS):
    d = Path(root) / sid
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    if (d / "log.json").exists():
        log = json.loads((d / "log.json").read_text())
    else:                                   # sessions ingested before vbt.logs existed
        text = next(d.glob("hevy.txt"), None) or next(d.glob("log_source.*"))
        sets, info = parse_log(text.read_text())
        log = {**info, "sets": sets}
    data = load_watch_csv(d / "sensor_25hz.csv.gz", keep_clock=True)
    return d, meta, log, data


def windows_for(d, data, logged, sid, redetect=False, root=SESSIONS, benchmark=True, placement="wrist"):
    """(windows used for analysis, automatic detection result or None).

    Human-checked windows ("manual"/"reviewed") always win; with benchmark=True
    the detector still runs so the report can show how close it got.
    """
    f = d / "sets.json"
    existing = json.loads(f.read_text()) if f.exists() else None
    if existing and existing.get("source") in TRUSTED and not redetect and not benchmark:
        return existing, None
    prior = library.exercise_model(library.load_rows(root, exclude=sid))
    found, _ = detect_sets(data, logged, prior_model=prior, placement=placement)
    rec = {"source": "auto", "clock": "Sensor Logger seconds_elapsed",
           "detector": "vbt.align.detect_sets", "prior_model": prior is not None,
           "sets": [{"log_index": i + 1, **{k: v for k, v in a.items() if k != "window"}}
                    for i, a in enumerate(found)]}
    if not existing or existing.get("source") not in TRUSTED:
        f.write_text(json.dumps(rec, indent=1))
    return (existing if existing and existing.get("source") in TRUSTED else rec), rec


def _count_range(data, t0, t1, key, placement="wrist", profile=None):
    counts = []
    orig = W.find_turnarounds
    try:
        for rp in (0.25, 0.35, 0.45):
            W.find_turnarounds = lambda *a, _rp=rp, **k: orig(*a, **{**k, "rel_prom": _rp})
            counts.append(analyze_set(data, t0, t1, key, placement=placement, profile=profile)["rep_count"])
    finally:
        W.find_turnarounds = orig
    return [min(counts), max(counts)]


def analyze_sets(data, logged, wins, groups, placement="wrist"):
    sets, rep_rows = [], []
    for i, (h, w) in enumerate(zip(logged, wins["sets"])):
        g = groups[h["label"]]
        prof = PROFILES.get(g["key"], PROFILES["auto"])
        base = {"i": i + 1, "ex": g["id"], "label": h["label"], "key": g["key"],
                "kg": h["kg"], "logged_reps": h["reps"], "rpe": h["rpe"], "warmup": h["warmup"],
                "per_unit": prof["bouts"] > 1}
        if w.get("skipped") or w.get("t_start") is None:
            sets.append({**base, "skipped": True, "reps": [], "features": None,
                         "flag": "not seen", "delta": None, "sensor_reps": 0})
            continue
        t0, t1 = w["t_start"], w["t_end"]
        res = analyze_set(data, t0, t1, g["key"], placement=placement, profile=h.get("profile"))
        reps = res["reps"]
        flag, delta, bouts, bouts_clean = count_verdict(reps, h["reps"], prof["bouts"]) if reps \
            else ("watch fewer", h["reps"], [0], [0])
        span = (reps[0]["t_start"], reps[-1]["t_end"]) if reps else (t0, t1)
        feats = set_features(data, *span)
        d = resample(data, t0 - 4, t1 + 4, 25.0, pad=0)
        gh = d["grav"] / np.linalg.norm(d["grav"], axis=1, keepdims=True)
        gy = d.get("gyro", np.zeros_like(gh))
        pos = np.interp(d["t"], res["t"], res["pos"])
        a_f = np.interp(d["t"], res["t"], res["a_up"])
        s = {**base, "skipped": False, "window": [t0, t1], "span": [R(span[0], 2), R(span[1], 2)],
             "auto_score": w.get("score"), "channel": res.get("channel"),
             "sensor_reps": len(reps), "clean_reps": sum(r["ok"] for r in reps),
             "sensor_range": _count_range(data, t0, t1, g["key"], placement, h.get("profile")),
             "profile_source": "personal" if h.get("profile") else "generic",
             "bouts": bouts, "bouts_clean": bouts_clean, "flag": flag, "delta": delta,
             "mean_velocity": R(res.get("mean_velocity")), "best_velocity": R(res.get("best_mean_velocity")),
             "velocity_loss_pct": R(res.get("velocity_loss_pct"), 1),
             "tut_s": R(res.get("tut_s"), 1),
             "con_s": R(sum(r["concentric_s"] for r in reps), 1),
             "ecc_s": R(sum(r["eccentric_s"] for r in reps), 1),
             "features": {k: R(v, 4) for k, v in feats.items()},
             "reps": [{k: (R(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in reps],
             "trace": {"t0": R(d["t"][0], 2), "dt": 0.04,
                       "pos": [R(x * 100, 1) for x in pos], "a": [R(x, 2) for x in a_f],
                       "gx": [R(x, 3) for x in gh[:, 0]], "gy": [R(x, 3) for x in gh[:, 1]],
                       "gz": [R(x, 3) for x in gh[:, 2]],
                       "wx": [R(x, 2) for x in gy[:, 0]], "wy": [R(x, 2) for x in gy[:, 1]],
                       "wz": [R(x, 2) for x in gy[:, 2]]}}
        sets.append(s)
        for r in reps:
            if r["t_end"] - r["t_start"] > 0.8:
                rf = set_features(data, r["t_start"], r["t_end"])
                rep_rows.append((i, h["label"], [rf[k] for k in FEATURES]))
    return sets, rep_rows


def timeline(data, bin_s=0.48):
    t = data["t"]
    fs = 25.0
    d = resample(data, t[0], t[-1], fs, pad=0)
    a = vertical_acceleration(d["acc"], d["grav"])
    den = _lp(a, fs, 2.0)
    gh = d["grav"] / np.linalg.norm(d["grav"], axis=1, keepdims=True)
    B = int(round(bin_s * fs))
    nb = len(a) // B
    rs = lambda x: x[:nb * B].reshape(nb, B, *x.shape[1:])  # noqa: E731
    gb = rs(gh).mean(1)
    return {"t0": R(d["t"][0], 2), "dt": B / fs,
            "lo": [R(x, 2) for x in rs(a).min(1)], "hi": [R(x, 2) for x in rs(a).max(1)],
            "den": [R(x, 3) for x in rs(den).mean(1)],
            "gx": [R(x, 2) for x in gb[:, 0]], "gy": [R(x, 2) for x in gb[:, 1]],
            "gz": [R(x, 2) for x in gb[:, 2]]}


def signatures(sets, groups, N=60):
    sig = {}
    for g in groups.values():
        chans = {k: [] for k in ("pos", "gx", "gy", "gz", "wx", "wy", "wz")}
        rep_s = []
        for s in sets:
            if s["ex"] != g["id"] or s["skipped"]:
                continue
            tr = s["trace"]
            tt = tr["t0"] + np.arange(len(tr["pos"])) * tr["dt"]
            for r in s["reps"]:
                if not r["ok"]:
                    continue
                q = np.linspace(r["t_start"], r["t_end"], N)
                rep_s.append(r["rep_s"])
                for k in chans:
                    y = np.interp(q, tt, np.array(tr[k], float))
                    chans[k].append(y - y[0] if k == "pos" else y)
        if not chans["pos"]:
            continue
        sig[g["id"]] = {k: {"mean": [R(x, 3) for x in np.mean(v, 0)], "sd": [R(x, 3) for x in np.std(v, 0)]}
                        for k, v in chans.items()}
        sig[g["id"]].update(n_reps=len(chans["pos"]), rep_s=R(np.median(rep_s), 2))
    return sig


def classifier_block(sets, rep_rows, sid, root):
    seen = [s for s in sets if not s["skipped"]]
    y = np.array([s["ex"] for s in seen])
    X = np.nan_to_num(np.array([[s["features"][k] for k in FEATURES] for s in seen], float))
    out = {"features": FEATURES, "groups": FEATURE_GROUPS, "set_level": {}, "rep_level": {},
           "set_index": [s["i"] for s in seen], "labels": y.tolist()}
    from collections import Counter
    cnt = Counter(y)
    usable = [c for c, n in cnt.items() if n >= 2]
    out["within_ok"] = len(usable) >= 2
    if out["within_ok"]:
        for gname, cols in [("all", FEATURES), *FEATURE_GROUPS.items()]:
            idx = [FEATURES.index(c) for c in cols]
            for m in ("centroid", "knn"):
                pred, acc = loso_classify(X[:, idx], y, method=m)
                out["set_level"][f"{gname}|{m}"] = {"acc": R(acc), "pred": pred.tolist()}
        try:
            from sklearn.ensemble import RandomForestClassifier
            pred = []
            for i in range(len(y)):
                tr = np.arange(len(y)) != i
                pred.append(RandomForestClassifier(300, random_state=0).fit(X[tr], y[tr]).predict(X[i:i + 1])[0])
            out["set_level"]["all|rf"] = {"acc": R(np.mean(np.array(pred) == y)), "pred": list(pred)}
            imp = RandomForestClassifier(500, random_state=0).fit(X, y).feature_importances_
            out["rf_importance"] = {k: R(v, 4) for k, v in zip(FEATURES, imp)}
        except ImportError:
            pass
        if rep_rows:
            lab2id = {s["label"]: s["ex"] for s in seen}
            Xr = np.nan_to_num(np.array([r[2] for r in rep_rows], float))
            yr = np.array([lab2id[r[1]] for r in rep_rows])
            gr = np.array([r[0] for r in rep_rows])
            pred = np.empty(len(yr), dtype=object)
            for gi in np.unique(gr):
                tr, te = gr != gi, gr == gi
                mu, sd = Xr[tr].mean(0), Xr[tr].std(0) + 1e-9
                Z, z = (Xr[tr] - mu) / sd, (Xr[te] - mu) / sd
                labs = np.unique(yr[tr])
                cents = np.array([Z[yr[tr] == c].mean(0) for c in labs])
                pred[te] = labs[np.argmin(((z[:, None, :] - cents[None]) ** 2).sum(2), 1)]
            votes = {gi: Counter(pred[gr == gi]).most_common(1)[0][0] for gi in np.unique(gr)}
            truth = {i: s["ex"] for i, s in enumerate(sets)}
            out["rep_level"] = {"acc": R((pred == yr).mean()), "n": int(len(yr)),
                                "set_vote_acc": R(np.mean([votes[gi] == truth[gi] for gi in votes]))}
    if len(X) >= 3:
        Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
        _, S, Vt = np.linalg.svd(Z, full_matrices=False)
        pc = Z @ Vt[:2].T
        out["pca"] = {"xy": [[R(a, 3), R(b, 3)] for a, b in pc],
                      "var": [R(v, 3) for v in (S ** 2 / (S ** 2).sum())[:2]]}
    # cross-session: model trained only on other sessions
    rows = library.load_rows(root, exclude=sid)
    model = library.exercise_model(rows)
    if model is not None:
        known = [s for s in seen if s["label"] in model.labels]
        preds = []
        for s in known:
            lp = model.log_prob([s["features"][k] for k in FEATURES])
            preds.append(max(lp, key=lp.get))
        out["cross_session"] = {
            "n_sessions": len({r["session"] for r in rows}), "n_train_sets": len(rows),
            "n_test": len(known), "acc": R(np.mean([p == s["label"] for p, s in zip(preds, known)])) if known else None,
            "unknown_labels": sorted({s["label"] for s in seen} - set(model.labels))}
    return out


def load_velocity(sets, groups, hist):
    lv = {}
    for g in groups.values():
        prof = PROFILES.get(g["key"], PROFILES["auto"])
        rows = [s for s in sets if s["ex"] == g["id"] and not s["skipped"] and s.get("best_velocity")]
        if prof.get("mvt") is None or len({s["kg"] for s in rows}) < 3:
            continue
        loads = [s["kg"] for s in rows]
        best = [s["best_velocity"] for s in rows]
        mean = [s["mean_velocity"] for s in rows]
        fb = estimate_1rm(loads, best, mvt=prof["mvt"])
        fm = estimate_1rm(loads, mean, mvt=prof["mvt"])
        usable = fb["slope"] < 0 and fb["r2"] > 0.8 and fb["e1rm"] < 3 * max(loads)
        top = max(rows, key=lambda s: (s["kg"], -(s.get("rpe") or 10)))
        epley = None
        if top.get("rpe"):
            rir = 10 - top["rpe"]
            epley = top["kg"] * (1 + (top["logged_reps"] + rir) / 30)
        lv[g["id"]] = {"loads": loads, "best": best, "mean": mean, "usable": bool(usable),
                       "fit_best": {k: R(v, 4) for k, v in fb.items()},
                       "fit_mean": {k: R(v, 4) for k, v in fm.items()},
                       "epley_rpe": R(epley, 1), "epley_from": [top["kg"], top["logged_reps"], top.get("rpe")]}
    return lv


def summarize(sets, data):
    seen = [s for s in sets if not s["skipped"]]
    for a, b in zip(seen, seen[1:]):
        a["rest_after_s"] = R(b["span"][0] - a["span"][1], 1)
    if seen:
        seen[-1]["rest_after_s"] = None
    first, last = (seen[0]["span"][0], seen[-1]["span"][1]) if seen else (0, 0)
    per = lambda s: (s["bouts"] and len(s["bouts"])) if s["per_unit"] else 1  # noqa: E731
    rests = [s["rest_after_s"] for s in seen[:-1] if s.get("rest_after_s") is not None]
    tut = sum(s["tut_s"] or 0 for s in seen)
    flags = [s["flag"] for s in sets]
    return {"recording_s": R(data["t"][-1] - data["t"][0], 1), "n_sets": len(sets), "n_seen": len(seen),
            "n_exercises": len({s["ex"] for s in sets}),
            "first_rep_s": first, "last_rep_s": last, "work_span_min": R((last - first) / 60, 1),
            "logged_reps": int(sum(s["logged_reps"] for s in sets)),
            "sensor_reps": int(sum(s["sensor_reps"] for s in seen)),
            "sensor_reps_cmp": R(sum(s["sensor_reps"] / max(per(s), 1) for s in seen), 1),
            "tut_s": R(tut, 0), "work_density_pct": R(100 * tut / max(last - first, 1), 1),
            "volume_kg": R(sum((s["kg"] or 0) * s["logged_reps"] * (2 if s["per_unit"] else 1) for s in sets), 0),
            "median_rest_s": R(np.median(rests), 0) if rests else None,
            "n_match": flags.count("match"), "n_ambiguous": flags.count("ambiguous"),
            "n_fewer": flags.count("watch fewer"), "n_more": flags.count("watch more"),
            "n_not_seen": flags.count("not seen")}


def findings(out):
    """Plain-language observations computed from the numbers (no hand-written text)."""
    S, G, sm = out["sets"], out["exercises"], out["summary"]
    seen = [s for s in S if not s["skipped"]]
    f = {}
    c = []
    c.append(f"{sm['n_match']} of {sm['n_sets']} sets match the logged count exactly, and "
             f"{sm['n_ambiguous']} more fall between the clean and all-cycle counts.")
    fewer = [s for s in S if s["flag"] == "watch fewer"]
    more = [s for s in S if s["flag"] == "watch more"]
    if fewer:
        c.append(f"The sensor saw fewer reps than logged on {len(fewer)} set(s) "
                 f"({', '.join(f'#{s['i']} −{s['delta']}' for s in fewer)}). Usually a first rep blended "
                 "into setup or two fast reps merged; for one-arm work a wrist sensor only sees its own side.")
    if more:
        c.append(f"The sensor saw more on {len(more)} set(s) "
                 f"({', '.join(f'#{s['i']} +{s['delta']}' for s in more)}): worth a second look at the log.")
    if sm["n_not_seen"]:
        c.append(f"{sm['n_not_seen']} logged set(s) weren't found in the recording "
                 "(exercises where the sensor barely moves, or a set that was logged but not done).")
    f["counts"] = c
    v = []
    for g in G.values():
        ss = [s for s in seen if s["ex"] == g["id"] and s.get("mean_velocity")]
        if len(ss) < 2:
            continue
        a, b = ss[0], ss[-1]
        ch = 100 * (b["mean_velocity"] - a["mean_velocity"]) / a["mean_velocity"]
        load = "same load" if a["kg"] == b["kg"] else f"{kg(a['kg'])}→{kg(b['kg'])} kg"
        rpe = (f", RPE {kg(a['rpe']) if a['rpe'] else '–'}→{kg(b['rpe']) if b['rpe'] else '–'}"
               if (a["rpe"] or b["rpe"]) else "")
        if abs(ch) >= 8:
            v.append(f"{g['name']}: mean velocity {a['mean_velocity']:.2f}→{b['mean_velocity']:.2f} m/s "
                     f"from first to last set ({ch:+.0f}%, {load}{rpe}).")
    f["velocity"] = v or ["Velocity held steady across sets of each exercise (changes under 8%)."]
    lvf = []
    for gid, L in out["load_velocity"].items():
        nm = G[gid]["name"]
        fb = L["fit_best"]
        if L["usable"]:
            t = f"{nm}: e1RM ≈ {fb['e1rm']:.0f} kg from {len(L['loads'])} loads (r² {fb['r2']:.2f})."
            if L["epley_rpe"]:
                t += f" The RPE route (Epley, {kg(L['epley_from'][0])} kg × {L['epley_from'][1]} @ RPE {kg(L['epley_from'][2])}) gives ≈{L['epley_rpe']:.0f} kg."
            lvf.append(t)
        else:
            lvf.append(f"{nm}: velocity didn't fall cleanly with load (r² {fb['r2']:.2f}), so no e1RM. "
                       "Add a heavier set pulled with maximal intent.")
    f["load_velocity"] = lvf
    return f


def build_html(out, path):
    tpl = TEMPLATE.read_text()
    data = json.dumps(out, separators=(",", ":")).replace("</", "<\\/")
    title = out["meta"].get("report_title") or "Workout Telemetry"
    html = tpl.replace("/*DATA*/", data).replace("{{TITLE}}", title)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)
    return path


def run(sid, root=SESSIONS, redetect=False, html=True, quiet=False):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    d, meta, log, data = load_session(sid, root)
    logged = logged_sets(log)
    placement = (meta.get("sensor_info") or {}).get("placement") or "wrist"
    # personal profiles learned from OTHER checked workouts (never from this one)
    learned = personal.profiles(root, exclude=sid)
    for s in logged:
        if (learned.get(s["label"]) or {}).get("profile"):
            s["profile"] = learned[s["label"]]["profile"]
    wins, auto = windows_for(d, data, logged, sid, redetect, root, placement=placement)
    if wins.get("source") in TRUSTED and personal.evidence_is_stale(d):
        personal.build_evidence(d, data, logged, wins["sets"], placement)
    if len(wins["sets"]) != len(logged):
        raise ValueError(f"{sid}: sets.json has {len(wins['sets'])} sets, the workout log has {len(logged)}")
    # exercise groups in order of first appearance -> colour slot, short code
    groups, taken = {}, set()
    for h in logged:
        if h["label"] not in groups:
            code = short_code(h["label"], taken, h["exercise_key"])
            taken.add(code)
            groups[h["label"]] = {"id": code, "name": re.sub(r"\s*\(.*?\)", "", h["label"]).strip() or h["label"],
                                  "label": h["label"], "key": h["exercise_key"], "slot": len(groups) + 1,
                                  "visible": PROFILES.get(h["exercise_key"], PROFILES["auto"]).get("visible", True)}
    sets, rep_rows = analyze_sets(data, logged, wins, groups, placement)
    exercises = {g["id"]: g for g in groups.values()}
    summary = summarize(sets, data)
    when = None
    try:
        from zoneinfo import ZoneInfo
        when = datetime.fromisoformat(meta["start_utc"]).astimezone(ZoneInfo(meta.get("timezone") or "UTC"))
    except Exception:
        pass
    title_line = log.get("title") or meta.get("title") or sid
    m = {"session_id": sid, "title": title_line,
         "report_title": report_title(title_line) or sid,
         "date": sid[:10], "device": meta.get("device"), "timezone": meta.get("timezone"),
         "start_local": when.isoformat() if when else None,
         "start_local_s": (when.hour * 3600 + when.minute * 60 + when.second + when.microsecond / 1e6)
                          if when else None,
         "clock0_s": R(meta.get("first_sample_s", data["t"][0]), 3),
         "raw_rate_hz": meta.get("raw_rate_hz"), "n_raw_samples": meta.get("n_raw_samples"),
         "log_minus_recording_s": meta.get("log_minus_recording_s", meta.get("hevy_minus_recording_s")),
         "log_source": LOG_NAMES.get(log.get("source") or meta.get("log_source"), "workout log"),
         "sensor": meta.get("sensor") or "Apple Watch via Sensor Logger, wrist placement",
         "placement": (meta.get("sensor_info") or {}).get("placement", "wrist"),
         "gravity": (meta.get("sensor_info") or {}).get("gravity", "device"),
         "windows_source": wins.get("source")}
    out = {"meta": m, "summary": summary, "exercises": exercises, "sets": sets,
           "timeline": timeline(data), "signatures": signatures(sets, groups),
           "classifier": classifier_block(sets, rep_rows, sid, root)}
    rows = library.load_rows(root, exclude=sid)
    out["load_velocity"] = load_velocity(sets, groups, rows)
    hist = library.history(rows + [{"session": sid, "date": m["date"], "label": s["label"], "kg": s["kg"],
                                    "reps": s["logged_reps"], "best_velocity": s.get("best_velocity")}
                                   for s in sets if not s["skipped"]])
    out["history"] = {lab: v for lab, v in hist.items() if len(v) >= 2}
    det = {"source": wins.get("source")}
    if auto is not None and wins.get("source") in TRUSTED:
        ev = evaluate_alignment(auto["sets"], wins["sets"])
        det.update(auto=auto["sets"], recall=R(ev["recall"]), mean_iou=R(ev["mean_iou"]),
                   rows=ev["rows"])
    elif wins.get("source") == "auto":
        det.update(auto=wins["sets"])
    out["detection"] = det
    # sensor-only shadow test: log hidden, knowledge from other workouts only
    sh = shadow_test(d, data, logged, wins["sets"], root, placement)
    sh["reference"] = "checked windows" if wins.get("source") in TRUSTED else "automatic windows (unchecked)"
    # learning lives on its own page (vbt.learning), not in the workout report
    (d / "shadow.json").write_text(json.dumps(sh, indent=0))
    notes = d / "notes.json"
    out["notes"] = json.loads(notes.read_text()) if notes.exists() else {}
    out["findings"] = findings(out)
    # compact copy for the library (no traces)
    compact = {"session_id": sid, "date": m["date"], "windows_source": wins.get("source"),
               "summary": summary,
               "shadow": {k: v for k, v in sh.items() if k not in ("rows", "unmatched")},
               "sets": [{k: v for k, v in s.items() if k not in ("trace", "reps")} for s in sets]}
    (d / "analysis.json").write_text(json.dumps(compact, indent=0))
    if html:
        p = build_html(out, REPORTS / f"{sid}.html")
        if not quiet:
            print("report:", p, p.stat().st_size // 1024, "KB")
        if Path(root).resolve() == SESSIONS.resolve():
            from .learning import build as build_learning
            lp = build_learning(root)
            if not quiet:
                print("learning page:", lp)
    if not quiet:
        print(f"{sid}: {summary['n_seen']}/{summary['n_sets']} sets seen, windows={wins.get('source')}, "
              f"match {summary['n_match']} ambiguous {summary['n_ambiguous']} "
              f"fewer {summary['n_fewer']} more {summary['n_more']}")
    return out


def review_png(sid, path, root=SESSIONS):
    """Grid of every set with rep markers, for a quick human check of windows and counts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a = json.loads((Path(root) / sid / "analysis.json").read_text())
    out = run(sid, root, html=False, quiet=True)
    S = out["sets"]
    n = len(S)
    cols = 2
    rows = (n + 1) // cols
    fig, ax = plt.subplots(rows, cols, figsize=(24, 2.6 * rows), squeeze=False)
    ax = ax.T.ravel()
    for k, s in enumerate(S):
        x = ax[k]
        x.set_title(f"#{s['i']} {s['label']} {s['kg']}kg x{s['logged_reps']} | sensed {s['sensor_reps']} "
                    f"clean {s.get('clean_reps')} {s['flag']}", loc="left", fontsize=9)
        if s["skipped"]:
            continue
        tr = s["trace"]
        tt = tr["t0"] + np.arange(len(tr["pos"])) * tr["dt"]
        x.plot(tt, tr["pos"], "k", lw=1)
        x.plot(tt, np.array(tr["gx"]) * 20, "r", lw=.6)
        for r in s["reps"]:
            x.axvline(r["t_turn"], c="b" if r["ok"] else "orange")
            x.axvspan(r["t_start"], r["t_end"], color=f"C{r['n'] % 2}", alpha=.12)
        x.set_ylim(-45, 45)
    fig.tight_layout()
    fig.savefig(path, dpi=45)
    return path, a.get("windows_source")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m vbt.pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new", help="ingest a sensor recording + a workout log, then run")
    n.add_argument("--sensor", required=True,
                   help="zip, folder or CSV: Sensor Logger (watch/phone) or any IMU CSV")
    n.add_argument("--log", "--hevy", dest="log", required=True,
                   help="workout log file: Hevy share text, Strong CSV, any CSV, typed notes or JSON")
    n.add_argument("--log-format", default="auto", choices=["auto", "hevy", "strong", "csv", "text", "json"])
    n.add_argument("--units", choices=["kg", "lb"], help="unit for weights written without one")
    n.add_argument("--sensor-format", default="auto",
                   choices=["auto", "sensorlogger-watch", "sensorlogger-phone", "imu-csv"])
    n.add_argument("--accel-convention", choices=["gravity", "specific_force"],
                   help="raw accel sign: 'gravity' (Apple: -1 g up at rest) or 'specific_force' "
                        "(Android/most IMUs: +1 g up). Default from the source, else specific_force")
    n.add_argument("--placement", choices=["wrist", "bar", "hand", "other"],
                   help="where the sensor was worn (default: wrist for watches)")
    n.add_argument("--id", help="session id (default: the log's date)")
    n.add_argument("--overwrite", action="store_true")
    r = sub.add_parser("run", help="(re)analyse session(s)")
    r.add_argument("sid", nargs="?")
    r.add_argument("--all", action="store_true")
    r.add_argument("--redetect", action="store_true", help="ignore auto windows and detect again")
    v = sub.add_parser("review", help="PNG with rep markers per set")
    v.add_argument("sid")
    v.add_argument("png")
    a = ap.parse_args(argv)
    if a.cmd == "new":
        from .ingest import ingest
        d = ingest(a.sensor, a.log, SESSIONS, a.id, a.overwrite, a.log_format, a.sensor_format,
                   a.accel_convention, a.placement, a.units)
        print("session:", d)
        run(d.name)
    elif a.cmd == "run":
        ids = sorted(p.name for p in SESSIONS.iterdir() if (p / "sensor_25hz.csv.gz").exists()) \
            if a.all else [a.sid]
        for sid in ids:
            run(sid, redetect=a.redetect)
    elif a.cmd == "review":
        p, src = review_png(a.sid, a.png)
        print("review grid:", p, "windows:", src)


if __name__ == "__main__":
    sys.exit(main())
