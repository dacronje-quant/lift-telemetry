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
