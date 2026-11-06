import json
import numpy as np
from vbt import synth_set, analyze, estimate_1rm


def test_rep_count_and_velocity():
    errs = []
    for seed in range(10):
        data, true_mv = synth_set(n_reps=6, seed=seed)
        res = analyze(data)
        assert res["rep_count"] == 6
        mv = np.array([r["mean_velocity"] for r in res["reps"]])
        errs.append(np.abs(mv - true_mv))
        assert res["velocity_loss_pct"] > 10
    errs = np.concatenate(errs)
    assert errs.mean() < 0.05 and errs.max() < 0.15


def test_e1rm():
    loads = [60, 80, 100, 120]
    v = [1.0 - 0.006 * l for l in loads]
    out = estimate_1rm(loads, v, mvt=0.3)
    assert abs(out["e1rm"] - 116.67) < 0.1


def test_session_split():
    from vbt import analyze_session
    parts, t0, counts = [], 0.0, [5, 3, 6]
    rng = np.random.default_rng(0)
    for i, n in enumerate(counts):
        d, _ = synth_set(n_reps=n, seed=i)
        rest = 60.0
        rt = np.arange(0, rest, 0.01)
        quiet = {"t": rt, "acc": rng.normal(0, 0.02 * 9.8, (len(rt), 3)),
                 "grav": np.tile(d["grav"][0], (len(rt), 1))}
        for seg in (d, quiet):
            parts.append({k: (seg[k] + t0 if k == "t" else seg[k]) for k in seg})
            t0 = parts[-1]["t"][-1] + 0.01
    data = {k: np.concatenate([p[k] for p in parts]) for k in ("t", "acc", "grav")}
    sets = analyze_session(data)
    assert [s["rep_count"] for s in sets] == counts


def _noisy_session(counts, rest_s=60.0, seed=0):
    """Sets separated by *non-quiet* rests (random wrist motion, like walking about)."""
    from vbt import synth_set
    rng = np.random.default_rng(seed)
    parts, t0, wins = [], 0.0, []
    for i, n in enumerate(counts):
        d, _ = synth_set(n_reps=n, seed=i)
        rt = np.arange(0, rest_s, 0.01)
        # aperiodic fidgeting: smoothed noise, ~1 m/s^2
        jitter = np.convolve(rng.normal(0, 3.0, len(rt)), np.ones(20) / 20, "same")
        rest = {"t": rt, "acc": np.c_[jitter, jitter * 0.5, -jitter] * 0.3,
                "grav": np.tile(d["grav"][0], (len(rt), 1))}
        for j, seg in enumerate((d, rest)):
            if j == 0:
                wins.append((t0, t0 + seg["t"][-1]))
            parts.append({k: (seg[k] + t0 if k == "t" else seg[k]) for k in seg})
            t0 = parts[-1]["t"][-1] + 0.01
    data = {k: np.concatenate([p[k] for p in parts]) for k in ("t", "acc", "grav")}
    return data, wins


def test_find_sets_ignores_fidgeting_rest():
    from vbt import find_sets
    data, wins = _noisy_session([6, 5, 8])
    found = find_sets(data, min_prom_m=0.1)
    assert len(found) == 3
    for f, (a, b) in zip(found, wins):
        assert a - 1 <= f["t_start"] and f["t_end"] <= b + 1


def test_analyze_set_counts_and_velocity():
    from vbt import synth_set, analyze_set
    for seed in range(5):
        data, true_mv = synth_set(n_reps=7, seed=seed)
        res = analyze_set(data, data["t"][0] + 0.5, data["t"][-1] - 0.5, "bench")
        assert res["rep_count"] == 7
        mv = np.array([r["mean_velocity"] for r in res["reps"]])
        assert np.abs(mv - true_mv).mean() < 0.08
        assert res["velocity_loss_pct"] > 10          # synth set slows down


def test_down_concentric_profile():
    """Pulldown-style: fast pull down from a rest at the top, slow return.

    Time-reversing the synthetic bench set gives exactly that (accel is
    invariant under t -> -t), with the reps in reverse order.
    """
    from vbt import synth_set, analyze_set
    data, true_mv = synth_set(n_reps=6, seed=3)
    flipped = {**data, "acc": data["acc"][::-1]}
    true_mv = true_mv[::-1]
    res = analyze_set(flipped, data["t"][0] + 0.5, data["t"][-1] - 0.5, "pulldown")
    assert res["rep_count"] == 6
    mv = np.array([r["mean_velocity"] for r in res["reps"]])
    assert np.abs(mv - true_mv).mean() < 0.1


def test_exercise_key_and_hevy_names():
    from vbt import exercise_key
    from vbt.hevy import parse_hevy_text
    from pathlib import Path
    sets = parse_hevy_text((Path(__file__).parents[1] / "data/sessions/2026-11-01/log_source.txt").read_text())
    keys = [exercise_key(s["exercise"]) for s in sets]
    assert None not in keys and len(sets) == 22
    assert keys.count("bench") == 4 and keys.count("lateral") == 2


def test_loso_classifier_separates_orientations():
    from vbt import loso_classify
    rng = np.random.default_rng(1)
    X = np.r_[rng.normal(0, .1, (6, 3)) + [1, 0, 0], rng.normal(0, .1, (6, 3)) + [0, 1, 0],
              rng.normal(0, .1, (6, 3)) + [0, 0, 1]]
    y = np.repeat(["a", "b", "c"], 6)
    for m in ("centroid", "knn"):
        pred, acc = loso_classify(X, y, method=m)
        assert acc == 1.0


def test_set_features_orientation():
    from vbt import synth_set, set_features
    data, _ = synth_set(n_reps=5, seed=2)
    f = set_features(data, data["t"][0], data["t"][-1])
    g = data["grav"][0] / np.linalg.norm(data["grav"][0])
    assert np.allclose([f["g_x"], f["g_y"], f["g_z"]], g, atol=1e-6)
    assert 30 < f["rom_cm"] < 70


# ---------------------------------------------------------------- pipeline pieces

def test_align_finds_sets_in_noisy_session():
    """Log-guided detection on synthetic sets separated by fidgeting rests."""
    from vbt.align import detect_sets, evaluate_alignment
    counts = [6, 5, 8, 6]
    data, wins = _noisy_session(counts, rest_s=70.0, seed=3)
    logged = [{"exercise": "Bench Press (Barbell)", "label": "Bench Press (Barbell)",
               "exercise_key": "bench", "reps": n, "kg": 60.0} for n in counts]
    found, _ = detect_sets(data, logged)
    truth = [{"t_start": a, "t_end": b} for a, b in wins]
    ev = evaluate_alignment(found, truth)
    assert ev["recall"] == 1.0
    assert all(f["n_sensed"] >= n - 1 for f, n in zip(found, counts))


def test_evaluate_alignment_scores():
    from vbt.align import evaluate_alignment
    truth = [{"t_start": 0, "t_end": 10}, {"t_start": 50, "t_end": 60}]
    good = [{"skipped": False, "t_start": 1, "t_end": 9}, {"skipped": False, "t_start": 50, "t_end": 62}]
    swapped = [{"skipped": False, "t_start": 50, "t_end": 60}, {"skipped": True, "t_start": None, "t_end": None}]
    assert evaluate_alignment(good, truth)["recall"] == 1.0
    assert evaluate_alignment(swapped, truth)["recall"] == 0.0


def test_count_verdict_interval():
    from vbt import count_verdict
    reps = [{"t_turn": float(i), "ok": i not in (0, 6)} for i in range(7)]  # 7 cycles, 5 clean
    assert count_verdict(reps, 7)[0] == "match"
    assert count_verdict(reps, 5)[0] == "ambiguous"
    assert count_verdict(reps, 9)[:2] == ("watch fewer", 2)
    assert count_verdict(reps, 3)[:2] == ("watch more", 2)
    legs = [{"t_turn": t, "ok": True} for t in (0, 2, 4, 6, 20, 22, 24, 26)]
    v, d, bouts, clean = count_verdict(legs, 4, n_bouts=2)
    assert v == "match" and bouts == [4, 4]


def test_exercise_keys_cover_common_lifts():
    from vbt import exercise_key
    assert exercise_key("Squat (Barbell)") == "squat"
    assert exercise_key("Romanian Deadlift (Dumbbell)") == "rdl"
    assert exercise_key("Overhead Press (Barbell)") == "ohp"
    assert exercise_key("Leg Press (Machine)") == "machine"
    assert exercise_key("Bicep Curl (Cable)") == "curl"
    assert exercise_key("Something New") == "auto"


def test_report_title_and_codes():
    from vbt.pipeline import report_title, short_code
    assert report_title("Sunday — Full Body C | Enhanced Comeback W1–2") == "Full Body C"
    assert report_title("Push Day") == "Push Day"
    taken = set()
    a = short_code("Bench Press (Barbell)", taken, "bench"); taken.add(a)
    b = short_code("Incline Bench Press (Dumbbell)", taken, "bench")
    assert a == "BP" and b != a


def test_ingest_roundtrip(tmp_path):
    """Zip with a Sensor Logger-style WristMotion.csv -> 25 Hz session folder."""
    import zipfile
    import pandas as pd
    from vbt import synth_set, load_watch_csv
    from vbt.ingest import ingest, COLS
    data, _ = synth_set(n_reps=5, seed=1)
    G = 9.80665
    t = data["t"] + 1.5
    df = pd.DataFrame({"time": (1.79e18 + t * 1e9).astype("int64"), "seconds_elapsed": t})
    for i, ax in enumerate("XYZ"):
        df[f"rotationRate{ax}"] = 0.0
        df[f"gravity{ax}"] = data["grav"][:, i] / G
        df[f"acceleration{ax}"] = data["acc"][:, i] / G
    z = tmp_path / "rec.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("WristMotion.csv", df.to_csv(index=False))
        f.writestr("Metadata.csv", "version,device name,recording epoch time,recording time,"
                   "recording timezone\n4,Watch,1793525926050,2026-11-01_09-38-46,Europe/Stockholm\n")
    hevy = "Test Day\nSunday, Nov 01, 2026 at 10:37am\n\nBench Press (Barbell)\nSet 1: 60 kg x 5\n"
    d = ingest(z, hevy, root=tmp_path / "sessions")
    assert d.name == "2026-11-01"
    small = load_watch_csv(d / "sensor_25hz.csv.gz", keep_clock=True)
    assert abs(small["t"][0] - 1.5) < 0.05
    assert abs(np.median(np.diff(small["t"])) - 0.04) < 1e-6
    meta = json.loads((d / "meta.json").read_text())
    assert meta["timezone"] == "Europe/Stockholm" and meta["log_minus_recording_s"] == -106
    assert set(COLS) <= set(pd.read_csv(d / "sensor_25hz.csv.gz").columns)


def test_pipeline_end_to_end(tmp_path):
    """Synthetic noisy session -> ingest -> run (auto windows) -> analysis.json."""
    import zipfile
    import pandas as pd
    from vbt.ingest import ingest
    from vbt.pipeline import run
    counts = [6, 5, 7]
    data, _ = _noisy_session(counts, rest_s=60.0, seed=5)
    G = 9.80665
    t = data["t"] + 2.0
    df = pd.DataFrame({"time": (1.79e18 + t * 1e9).astype("int64"), "seconds_elapsed": t})
    for i, ax in enumerate("XYZ"):
        df[f"rotationRate{ax}"] = 0.0
        df[f"gravity{ax}"] = data["grav"][:, i] / G
        df[f"acceleration{ax}"] = data["acc"][:, i] / G
    z = tmp_path / "rec.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("WristMotion.csv", df.to_csv(index=False))
    hevy = ("Push A\nMonday, Nov 02, 2026 at 6:05pm\n\nBench Press (Barbell)\n"
            + "".join(f"Set {i + 1}: 60 kg x {n}\n" for i, n in enumerate(counts)))
    root = tmp_path / "sessions"
    d = ingest(z, hevy, root=root)
    out = run(d.name, root=root, html=False, quiet=True)
    assert out["summary"]["n_seen"] == 3
    assert [s["sensor_reps"] >= n - 1 for s, n in zip(out["sets"], counts)] == [True] * 3
    assert json.loads((d / "sets.json").read_text())["source"] == "auto"
    assert (d / "analysis.json").exists()


# ---------------------------------------------------------------- any log, any sensor

def test_log_formats():
    from vbt.logs import parse_log
    s, i = parse_log("Push day\nNov 3, 2026 at 6:15pm\n\nBench press\n60kg x 5\n70 x 5 @8\n"
                     "3x5 @ 80kg\nPull-ups: 3 x 8\nLateral raise: 8kg x 12, 8kg x 10\n")
    assert i["source"] == "text" and i["start_local"] == "2026-11-03T18:15:00"
    assert [(x["exercise"], x["kg"], x["reps"]) for x in s[:5]] == [
        ("Bench press", 60, 5), ("Bench press", 70, 5), ("Bench press", 80, 5),
        ("Bench press", 80, 5), ("Bench press", 80, 5)]
    assert s[1]["rpe"] == 8 and sum(x["exercise"] == "Pull-ups" for x in s) == 3
    strong = ("Date,Workout Name,Exercise Name,Set Order,Weight,Reps,RPE\n"
              "2026-10-30 18:00:00,Legs,Squat (Barbell),1,100,5,7\n"
              "2026-11-02 07:30:00,Upper,Bench Press (Barbell),1,80,6,\n"
              "2026-11-02 07:30:00,Upper,Plank,1,0,0,\n")
    s, i = parse_log(strong)
    assert i["source"] == "strong" and len(s) == 1 and s[0]["kg"] == 80
    s, _ = parse_log("exercise;weight;unit;reps\nDeadlift;225;lb;5\n")
    assert abs(s[0]["kg"] - 102.06) < 0.01
    s, _ = parse_log("Leg day\nSquat: 225x5\n", default_unit="lb")
    assert abs(s[0]["kg"] - 102.06) < 0.01
    s, _ = parse_log('[{"exercise": "Row", "weight": 60, "reps": 10}]')
    assert s[0]["kg"] == 60 and s[0]["set"] == 1


def _raw_imu_frame(data, gyro_deg=True):
    """Synthetic set as an Android-style IMU CSV: specific force in m/s^2, gyro, ms clock."""
    import pandas as pd
    G = 9.80665
    raw = -(data["acc"] + data["grav"])            # canonical raw = gravity + user; flip sign
    return pd.DataFrame({"timestamp_ms": data["t"] * 1000 + 5000,
                         "acc_x": raw[:, 0], "acc_y": raw[:, 1], "acc_z": raw[:, 2],
                         "gyro_x": 0.0, "gyro_y": 0.0, "gyro_z": 0.0})


def test_generic_imu_csv_matches_device_motion(tmp_path):
    from vbt import synth_set, analyze_set
    from vbt.sensors import load_sensor
    data, true_mv = synth_set(n_reps=6, seed=4)
    p = tmp_path / "imu.csv"
    _raw_imu_frame(data).to_csv(p, index=False)
    for with_gyro in (True, False):
        df = __import__("pandas").read_csv(p)
        if not with_gyro:
            df = df[["timestamp_ms", "acc_x", "acc_y", "acc_z"]]
        df.to_csv(p, index=False)
        can, info = load_sensor(p)
        assert info["format"] == "imu-csv" and info["gravity"].startswith("estimated")
        G = 9.80665
        g = can[["gravityX", "gravityY", "gravityZ"]].to_numpy()
        g_true = data["grav"][0] / G
        assert np.degrees(np.arccos(np.clip(np.median(g @ g_true), -1, 1))) < 3
        d = {"t": can.seconds_elapsed.to_numpy(), "grav": g * G,
             "acc": can[["accelerationX", "accelerationY", "accelerationZ"]].to_numpy() * G}
        res = analyze_set(d, d["t"][0] + 0.5, d["t"][-1] - 0.5, "bench")
        assert res["rep_count"] == 6
        mv = np.array([r["mean_velocity"] for r in res["reps"]])
        # accelerometer-only is a documented compromise: velocity is less accurate
        assert np.abs(mv - true_mv).mean() < (0.08 if with_gyro else 0.15)


def test_bar_placement_counts_curls_from_height():
    from vbt.workout import profile_for
    assert profile_for("curl", "wrist")["channel"] == "gx"
    assert profile_for("curl", "bar")["channel"] == "pos"


def test_pipeline_generic_inputs(tmp_path):
    """Generic IMU CSV + typed notes -> report data, no Apple or Hevy involved."""
    from vbt.ingest import ingest
    from vbt.pipeline import run
    counts = [6, 5, 7]
    data, _ = _noisy_session(counts, rest_s=60.0, seed=7)
    p = tmp_path / "imu.csv"
    _raw_imu_frame(data).to_csv(p, index=False)
    notes = "Upper A\nNov 4, 2026 at 7:00am\n\nBench press\n" + "".join(f"60kg x {n}\n" for n in counts)
    d = ingest(p, notes, root=tmp_path / "sessions")
    assert json.loads((d / "log.json").read_text())["source"] == "text"
    out = run(d.name, root=tmp_path / "sessions", html=False, quiet=True)
    assert out["summary"]["n_seen"] == 3
    assert out["meta"]["log_source"] == "workout notes"
    assert all(abs(s["sensor_reps"] - n) <= 1 for s, n in zip(out["sets"], counts))


# ---------------------------------------------------------------- learning your lifting

def _fake_evidence(root, sid, n_sets, generic_err, alt_err, label="Bench Press (Barbell)", key="bench"):
    d = root / sid
    d.mkdir(parents=True)
    (d / "sets.json").write_text(json.dumps({"source": "reviewed", "sets": []}))
    from vbt.workout import FEATURES
    rows = [{"index": i + 1, "label": label, "key": key, "reps": 6, "bouts": 1, "generic": "pos/top",
             "trials": {"pos/top": {"err": generic_err, "n": 6 + generic_err},
                        "gy/top": {"err": alt_err, "n": 6 + alt_err}},
             "features": [0.1 * i] * len(FEATURES), "n_all": 7, "n_clean": 6,
             "rep_s": 2.5, "rom_m": 0.4, "rotation_deg": 10} for i in range(n_sets)]
    (d / "evidence.json").write_text(json.dumps({"session": sid, "sets_mtime": 0, "sets": rows}))


def test_personal_profile_needs_two_workouts(tmp_path):
    from vbt.personal import profiles
    _fake_evidence(tmp_path, "2026-11-01", 4, generic_err=1, alt_err=0)
    p = profiles(tmp_path)["Bench Press (Barbell)"]
    assert p["best_signal"] == "gy/top" and not p["learned"]          # one workout is not enough
    _fake_evidence(tmp_path, "2026-11-08", 4, generic_err=1, alt_err=0)
    p = profiles(tmp_path)["Bench Press (Barbell)"]
    assert p["learned"] and p["profile"]["channel"] == "gy" and p["count_mode"] == "clean"
    # honest tests never use a session's own evidence
    assert profiles(tmp_path, exclude="2026-11-08")["Bench Press (Barbell)"]["n_sessions"] == 1


def test_personal_profile_not_adopted_when_generic_is_as_good(tmp_path):
    from vbt.personal import profiles
    for sid in ("2026-11-01", "2026-11-08"):
        _fake_evidence(tmp_path, sid, 3, generic_err=0, alt_err=0)
    assert not profiles(tmp_path)["Bench Press (Barbell)"]["learned"]


def test_sensor_only_finds_and_names_sets():
    """Two synthetic 'exercises' (different wrist tilts); prototypes from labelled sets."""
    from vbt import synth_set, set_features
    from vbt.personal import Prototypes
    from vbt.infer import infer_sets, _match
    from vbt.workout import FEATURES
    rng = np.random.default_rng(0)
    plan = [("A", 0, 6), ("A", 0, 5), ("B", 1, 8), ("B", 1, 7)]
    parts, t0, wins = [], 0.0, []
    for lab, seed, n in plan:
        d, _ = synth_set(n_reps=n, seed=seed)                 # same seed -> same wrist tilt
        rt = np.arange(0, 60.0, 0.01)
        jit = np.convolve(rng.normal(0, 3.0, len(rt)), np.ones(20) / 20, "same")
        rest = {"t": rt, "acc": np.c_[jit, jit * 0.5, -jit] * 0.3, "grav": np.tile(d["grav"][0], (len(rt), 1))}
        wins.append((lab, n, t0, t0 + d["t"][-1]))
        for seg in (d, rest):
            parts.append({k: (seg[k] + t0 if k == "t" else seg[k]) for k in seg})
            t0 = parts[-1]["t"][-1] + 0.01
    data = {k: np.concatenate([p[k] for p in parts]) for k in ("t", "acc", "grav")}
    rows = []
    for i, (lab, n, a, b) in enumerate(wins):
        f = set_features(data, a + 0.5, b - 0.5)
        rows.append({"index": i + 1, "label": lab, "key": "bench", "features": [f[k] for k in FEATURES]})
    protos = Prototypes(rows)
    inferred = infer_sets(data, protos)
    truth = [{"t_start": a, "t_end": b} for _, _, a, b in wins]
    m = _match(inferred, truth)
    assert len(m) == 4 and len(inferred) == 4
    for i, j in m.items():
        assert inferred[j]["guess"] == wins[i][0]
        assert abs(inferred[j]["reps"] - wins[i][1]) <= 1


def test_learning_page_is_separate(tmp_path):
    """Workout reports carry no learning sections; the learning page builds from sessions."""
    from pathlib import Path
    from vbt.learning import build, collect
    tpl = (Path(__file__).parents[1] / "vbt/templates/report.html").read_text()
    assert "sec-shadow" not in tpl and "personalTable" not in tpl
    root = tmp_path / "sessions"
    d = root / "2026-11-01"
    d.mkdir(parents=True)
    (d / "analysis.json").write_text(json.dumps({"date": "2026-11-01", "windows_source": "reviewed",
                                                 "summary": {"n_sets": 2}, "sets": []}))
    (d / "shadow.json").write_text(json.dumps({"mode": "other workouts", "n_sets": 2, "found": 2,
                                               "named": 1, "reps_exact": 1, "reps_within1": 2,
                                               "false_sets": 0, "fully_right": 1, "readiness_pct": 50.0,
                                               "rows": []}))
    c = collect(root, eval_dir=tmp_path)
    assert c["sessions"][0]["shadow"]["readiness_pct"] == 50.0
    out = build(root, tmp_path / "learning.html")
    html = out.read_text()
    assert "/*DATA*/" not in html and "readyChart" in html


def test_rep_shape_separates_movements():
    """Same rhythm, different wrist rotation -> different rep shapes; same movement -> similar."""
    from vbt import synth_set, analyze_set
    from vbt.workout import rep_shape
    a, _ = synth_set(n_reps=6, seed=2)
    b, _ = synth_set(n_reps=6, seed=2, noise=0.02)
    ra = analyze_set(a, a["t"][0] + 0.5, a["t"][-1] - 0.5, "bench")["reps"]
    rb = analyze_set(b, b["t"][0] + 0.5, b["t"][-1] - 0.5, "bench")["reps"]
    sa, sb = np.array(rep_shape(a, ra)), np.array(rep_shape(b, rb))
    assert sa.shape == (7 * 12,)
    # a rotating forearm: tilt gravity back and forth with each rep
    c = {k: v.copy() for k, v in a.items()}
    ang = 0.6 * np.sin(2 * np.pi * c["t"] / 2.6)
    g = c["grav"][0] / np.linalg.norm(c["grav"][0])
    perp = np.cross(g, [1.0, 0, 0]); perp /= np.linalg.norm(perp)
    c["grav"] = 9.80665 * (np.cos(ang)[:, None] * g + np.sin(ang)[:, None] * perp)
    sc = np.array(rep_shape(c, ra))
    assert np.abs(sa - sb).mean() < np.abs(sa - sc).mean()


def _set_with_roms(roms, fs=100.0, seed=0):
    """Wrist moving down/up per rep with the given ranges of motion (m), constant tilt."""
    rng = np.random.default_rng(seed)
    pos = [np.zeros(int(fs))]
    for rom in roms:
        te = np.linspace(0, 1, int(1.2 * fs))
        pos.append(-rom * (1 - np.cos(np.pi * te)) / 2)
        pos.append(np.full(int(0.15 * fs), -rom))
        tc = np.linspace(0, 1, int(0.9 * fs))
        pos.append(-rom + rom * (1 - np.cos(np.pi * tc)) / 2)
        pos.append(np.zeros(int(0.6 * fs)))
    pos.append(np.zeros(int(fs)))
    p = np.concatenate(pos)
    t = np.arange(len(p)) / fs
    a = np.gradient(np.gradient(p, t), t)
    g = np.array([0.2, -0.3, -0.93]); g /= np.linalg.norm(g)
    G = 9.80665
    acc = (-a)[:, None] * g[None, :] + rng.normal(0, 0.03 * G, (len(t), 3))
    return {"t": t, "acc": acc, "grav": np.tile(g * G, (len(t), 1))}


def test_first_rep_out_of_the_unrack_is_kept():
    """A shallow first rep breaks the chain's similar-size rule; the end extension keeps it."""
    from vbt import analyze_set
    data = _set_with_roms([0.15, 0.45, 0.45, 0.45, 0.45, 0.45])
    w = (data["t"][0] + 0.5, data["t"][-1] - 0.5)
    assert analyze_set(data, *w, "bench", extend=0)["rep_count"] == 5      # the old -1
    assert analyze_set(data, *w, "bench")["rep_count"] == 6


def test_exercise_key_whole_words():
    from vbt import exercise_key
    assert exercise_key("Dumbbell Row (knee on bench) (left arm)") == "row"
    assert exercise_key("Triceps Kickback (knee on bench)") == "extension"
    assert exercise_key("Narrow Grip Bench Press") == "bench"
    assert exercise_key("Squat Rack Shoulder Press") == "ohp"
    assert exercise_key("Pull-ups") == "bodyweight"
