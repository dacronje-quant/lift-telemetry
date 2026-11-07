"""The learning page: how the system is improving, across all workouts.

Kept separate from the per-workout reports, which stay about the workout.

    python -m vbt.learning            # -> reports/learning.html

Sources (all already in the repo, nothing recomputed):
* data/sessions/*/shadow.json      sensor-only shadow test per workout
* data/sessions/*/evidence.json    per-exercise evidence from checked sets (vbt.personal)
* data/sessions/*/analysis.json    session date, title, windows source
* data/eval/history.jsonl          scoreboard after each recorded algorithm change
* data/eval/public_history.jsonl   public-data benchmark (vbt.public: many people, true rep counts)
"""
import json
import sys
from pathlib import Path

from . import personal
from .io import is_excluded

ROOT = Path(__file__).resolve().parents[1]
SESSIONS = ROOT / "data/sessions"
TEMPLATE = Path(__file__).resolve().parent / "templates/learning.html"
OUT = ROOT / "reports/learning.html"


def _r(x, n=3):
    return round(float(x), n) if isinstance(x, (int, float)) and x == x else x


def collect(root=SESSIONS, eval_dir=ROOT / "data/eval"):
    root = Path(root)
    sessions = []
    for d in sorted(p for p in root.iterdir() if (p / "analysis.json").exists() and not is_excluded(p)):
        a = json.loads((d / "analysis.json").read_text())
        log = json.loads((d / "log.json").read_text()) if (d / "log.json").exists() else {}
        sh = json.loads((d / "shadow.json").read_text()) if (d / "shadow.json").exists() else {}
        sessions.append({"id": d.name, "date": a.get("date"), "title": log.get("title"),
                         "windows_source": a.get("windows_source"),
                         "n_sets": a.get("summary", {}).get("n_sets"),
                         "shadow": sh})
    P = personal.profiles(root)
    exercises = {}
    for label, p in sorted(P.items()):
        e = {k: _r(v) for k, v in p.items() if k not in ("prototype", "profile")}
        e["signal_in_use"] = (f"{p['profile']['channel']}/{p['profile']['rest']}" if p.get("profile")
                              else p.get("generic_signal"))
        exercises[label] = e
    history = []
    h = Path(eval_dir) / "history.jsonl"
    if h.exists():
        for line in h.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                history.append({"when": r.get("when"), "commit": r.get("commit"),
                                "n_trusted_sessions": r.get("n_trusted_sessions"),
                                "detection": r.get("detection"), "sensor_only": r.get("sensor_only")})
    public = []
    pp = Path(eval_dir) / "public_history.jsonl"
    if pp.exists():
        for line in pp.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                public.append({"when": r.get("when"), "commit": r.get("commit"),
                               "reps": {k: v for k, v in (r.get("reps") or {}).items() if k != "per_exercise"},
                               "recognition": r.get("recognition")})
    return {"sessions": sessions, "exercises": exercises, "eval_history": history, "public": public,
            "rules": {"min_sessions": personal.MIN_SESSIONS, "min_gain": personal.MIN_GAIN}}


def build(root=SESSIONS, out=OUT):
    data = json.dumps(collect(root), separators=(",", ":")).replace("</", "<\\/")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(TEMPLATE.read_text().replace("/*DATA*/", data))
    return out


def main(argv=None):
    p = build()
    print("learning page:", p, p.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    sys.exit(main())
