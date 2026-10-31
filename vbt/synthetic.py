"""Synthetic Apple Watch data for testing (wrist moving vertically, random tilt)."""
import numpy as np

G = 9.80665


def synth_set(n_reps=5, rom=0.5, conc_s=(0.7, 1.1), ecc_s=1.2, pause_s=0.6,
              fs=100.0, noise=0.05, seed=0):
    """Return (data, true_mean_velocities). Concentric times grow linearly (fatigue)."""
    rng = np.random.default_rng(seed)
    conc = np.linspace(conc_s[0], conc_s[1], n_reps)
    pos = [np.zeros(int(1.0 * fs))]
    true_mv = []
    for c in conc:
        te = np.linspace(0, 1, int(ecc_s * fs))
        pos.append(-rom * (1 - np.cos(np.pi * te)) / 2)           # down
        pos.append(np.full(int(0.15 * fs), -rom))                 # bottom
        tc = np.linspace(0, 1, int(c * fs))
        pos.append(-rom + rom * (1 - np.cos(np.pi * tc)) / 2)     # up
        pos.append(np.zeros(int(pause_s * fs)))
        true_mv.append(rom / c)
    pos.append(np.zeros(int(1.0 * fs)))
    p = np.concatenate(pos)
    t = np.arange(len(p)) / fs
    a_up = np.gradient(np.gradient(p, t), t)
    # constant random wrist tilt: gravity direction in device frame
    g_dir = rng.normal(size=3); g_dir /= np.linalg.norm(g_dir)
    grav = np.tile(g_dir * G, (len(t), 1))
    acc = (-a_up)[:, None] * g_dir[None, :] + rng.normal(0, noise * G, (len(t), 3))
    return {"t": t, "acc": acc, "grav": grav}, np.array(true_mv)
