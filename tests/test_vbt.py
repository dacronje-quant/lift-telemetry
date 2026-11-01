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
    sets = parse_hevy_text((Path(__file__).parents[1] / "data/hevy_2026-11-01.txt").read_text())
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
