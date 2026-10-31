"""CLI: python -m vbt WristMotion.csv [--units g|ms2] [--plot out.png]"""
import argparse
from .io import load_watch_csv
from .core import analyze

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--units", default="g", choices=["g", "ms2"])
ap.add_argument("--min-rom", type=float, default=0.10)
ap.add_argument("--plot")
args = ap.parse_args()
res = analyze(load_watch_csv(args.csv, args.units), min_rom_m=args.min_rom)
print(f"Reps: {res['rep_count']}")
for r in res["reps"]:
    print(f"  #{r['n']}: MV {r['mean_velocity']:.2f} m/s  PV {r['peak_velocity']:.2f}  "
          f"ROM {r['rom']*100:.0f} cm  conc {r['concentric_s']:.2f}s")
if res["reps"]:
    print(f"Velocity loss: {res['velocity_loss_pct']:.0f}%")
if args.plot:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 5))
    ax[0].plot(res["t"], res["a_up"]); ax[0].set_ylabel("vertical accel (m/s²)")
    ax[1].plot(res["t"], res["pos"]); ax[1].set_ylabel("rel. position")
    for s, b, e in res["reps_idx"]:
        ax[1].axvspan(res["t"][b], res["t"][e], alpha=0.2)
    ax[1].set_xlabel("s"); fig.tight_layout(); fig.savefig(args.plot)
