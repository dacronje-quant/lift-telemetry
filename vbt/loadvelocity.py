"""Load-velocity profile and 1RM estimation."""
import numpy as np

# Typical minimum velocity thresholds (m/s) from VBT literature
MVT = {"squat": 0.30, "bench": 0.17, "deadlift": 0.15, "ohp": 0.19, "default": 0.25}


def fit_load_velocity(loads, velocities):
    """Linear fit v = slope*load + intercept. Returns (slope, intercept, r2)."""
    x, y = np.asarray(loads, float), np.asarray(velocities, float)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return slope, intercept, r2


def estimate_1rm(loads, velocities, lift="default", mvt=None):
    slope, intercept, r2 = fit_load_velocity(loads, velocities)
    v_min = mvt if mvt is not None else MVT.get(lift, MVT["default"])
    return {"e1rm": (v_min - intercept) / slope, "slope": slope,
            "intercept": intercept, "r2": r2, "mvt": v_min}
