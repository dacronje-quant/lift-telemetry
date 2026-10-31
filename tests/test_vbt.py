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
