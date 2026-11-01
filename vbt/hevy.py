"""Parse a Hevy shared-workout text into a flat list of sets."""
import re

_SET = re.compile(r"Set\s+(\d+):\s*([\d.]+)\s*kg\s*x\s*(\d+)(?:\s*@\s*([\d.]+)\s*rpe)?(\s*\[Warmup\])?", re.I)


def parse_hevy_text(text):
    """Exercise headers are the non-set lines that follow a blank line (title/date/footer skipped)."""
    sets, exercise, prev_blank = [], None, False
    for raw in text.splitlines():
        line = raw.strip()
        m = _SET.match(line)
        if m:
            sets.append({"exercise": exercise, "set": int(m[1]), "kg": float(m[2]),
                         "reps": int(m[3]), "rpe": float(m[4]) if m[4] else None,
                         "warmup": bool(m[5])})
        elif line and prev_blank and not line.startswith(("@", "http")):
            exercise = line
        prev_blank = not line
    return [s for s in sets if s["exercise"]]
