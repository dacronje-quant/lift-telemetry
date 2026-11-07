"""Workout logs from any app, normalised to one list of sets.

Every parser returns ``(sets, info)``:

    sets: [{"exercise": str, "kg": float, "reps": int, "rpe": float|None,
            "warmup": bool, "set": int}, ...]   in the order performed
    info: {"source": "hevy"|"strong"|"csv"|"text"|"json", "title": str|None,
           "start_local": "YYYY-MM-DDTHH:MM:SS"|None}

Supported inputs (``fmt="auto"`` sniffs which one it is):

* **hevy**: Hevy's "share workout" text (``Set 1: 60 kg x 6 @ 7 rpe``).
* **strong**: Strong app CSV export (``Exercise Name``, ``Set Order``, ``Weight``, ``Reps``, ...).
* **csv**: any table with an exercise column and reps, plus weight in kg or lb.
  Column names are matched loosely (``exercise/name/movement``, ``weight/kg/lbs/load``,
  ``reps/repetitions``, ``rpe``, ``warmup/type``, ``unit``). One row per set, or a
  ``sets`` column to repeat a row.
* **text**: hand-typed notes, one exercise per header line and sets below it::

      Bench press
      60kg x 5
      70 x 5 @8
      3x5 @ 80kg            (3 sets of 5 at 80 kg)
      Pull-ups: 3 x 8       (bodyweight, weight 0)

  ``lb``/``lbs`` anywhere converts to kg.
* **json**: a list of set dicts with the fields above (``weight`` and ``unit`` accepted too).
"""
import csv
import io
import json
import re
from datetime import datetime

LB = 0.45359237


def _num(x):
    try:
        return float(str(x).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None


def _set(exercise, kg, reps, rpe=None, warmup=False, n=None):
    return {"exercise": exercise.strip(), "kg": round(float(kg or 0.0), 3), "reps": int(reps),
            "rpe": _num(rpe) if rpe not in (None, "") else None, "warmup": bool(warmup), "set": n}


def _number_sets(sets):
    count = {}
    for s in sets:
        count[s["exercise"]] = count.get(s["exercise"], 0) + 1
        if not s.get("set"):
            s["set"] = count[s["exercise"]]
    return sets


# ------------------------------------------------------------------ hevy
_HEVY_SET = re.compile(r"Set\s+(\d+):\s*([\d.]+)\s*(kg|lbs?)\s*x\s*(\d+)(?:\s*@\s*([\d.]+)\s*rpe)?(\s*\[Warm[- ]?up\])?",
                       re.I)
_HEVY_BW = re.compile(r"Set\s+(\d+):\s*(\d+)\s*reps?(?:\s*@\s*([\d.]+)\s*rpe)?(\s*\[Warm[- ]?up\])?", re.I)


def parse_date_line(text):
    """'Sunday, Nov 01, 2026 at 10:37am' -> ISO string (naive local) or None."""
    m = re.search(r"([A-Z][a-z]{2})[a-z]* (\d{1,2}),? (\d{4})(?: at (\d{1,2}):(\d{2})\s*([ap]m)?)?", text)
    if not m:                                                   # 'Sat 7 Nov 2026' (day first)
        m = re.search(r"\b(\d{1,2}) ([A-Z][a-z]{2})[a-z]*,? (\d{4})(?: at (\d{1,2}):(\d{2})\s*([ap]m)?)?", text)
        if not m:
            return None
        day, mon, year, hh, mm, ap = m.groups()
    else:
        mon, day, year, hh, mm, ap = m.groups()
    try:
        if hh:
            fmt = "%b %d %Y %I %M%p" if ap else "%b %d %Y %H %M"
            return datetime.strptime(f"{mon} {day} {year} {hh} {mm}{ap or ''}", fmt).isoformat()
        return datetime.strptime(f"{mon} {day} {year}", "%b %d %Y").isoformat()
    except ValueError:
        return None


def parse_hevy(text):
    sets, exercise, prev_blank = [], None, False
    for raw in text.splitlines():
        line = raw.strip()
        m = _HEVY_SET.match(line)
        b = None if m else _HEVY_BW.match(line)
        if m:
            kg = float(m[2]) * (LB if m[3].lower().startswith("lb") else 1.0)
            sets.append(_set(exercise, kg, m[4], m[5], bool(m[6]), int(m[1])))
        elif b:
            sets.append(_set(exercise, 0.0, b[2], b[3], bool(b[4]), int(b[1])))
        elif line and prev_blank and not line.startswith(("@", "http")):
            exercise = line
        prev_blank = not line
    sets = [s for s in sets if s["exercise"]]
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return sets, {"source": "hevy", "title": lines[0] if lines else None,
                  "start_local": parse_date_line(text)}


# ------------------------------------------------------------------ tables
_COLS = {
    "date": ["date", "start time", "start_time", "datetime", "timestamp"],
    "title": ["workout name", "workout", "title", "session"],
    "exercise": ["exercise name", "exercise", "exercise_title", "movement", "lift", "name"],
    "set": ["set order", "set", "set_index", "set number", "set_no"],
    "kg": ["weight_kg", "kg", "weight (kg)", "load_kg"],
    "lb": ["weight_lbs", "weight_lb", "lbs", "lb", "weight (lbs)", "weight (lb)"],
    "weight": ["weight", "load"],
    "unit": ["unit", "weight unit", "units"],
    "reps": ["reps", "repetitions", "rep"],
    "rpe": ["rpe"],
    "warmup": ["warmup", "warm up", "warm-up", "set_type", "set type", "type"],
    "nsets": ["sets", "n_sets"],
}


def _colmap(header):
    low = {h.strip().lower(): h for h in header}
    out = {}
    for key, names in _COLS.items():
        for n in names:
            if n in low:
                out[key] = low[n]
                break
    return out


def parse_table(rows, header, source="csv", default_unit=None):
    c = _colmap(header)
    if "exercise" not in c or "reps" not in c:
        raise ValueError(f"table needs exercise and reps columns, got {header}")
    sets, title, date = [], None, None
    for r in rows:
        reps = _num(r.get(c["reps"]))
        if not reps:
            continue                                   # cardio rows, notes, empty
        if "kg" in c and _num(r.get(c["kg"])) is not None:
            kg = _num(r[c["kg"]])
        elif "lb" in c and _num(r.get(c["lb"])) is not None:
            kg = _num(r[c["lb"]]) * LB
        elif "weight" in c:
            w = _num(r.get(c["weight"])) or 0.0
            unit = str(r.get(c.get("unit", ""), "") or default_unit or "").lower()
            kg = w * LB if unit.startswith("lb") else w
        else:
            kg = 0.0
        wu = str(r.get(c.get("warmup", ""), "") or "").strip().lower()
        warm = wu in ("1", "true", "yes", "w", "warmup", "warm up", "warm-up", "warm_up")
        n_rep = int(_num(r.get(c.get("nsets", ""), 1)) or 1)
        for _ in range(max(1, n_rep)):
            sets.append(_set(r[c["exercise"]], kg, reps, r.get(c.get("rpe", "")), warm,
                             int(_num(r.get(c.get("set", ""))) or 0) or None))
        title = title or (r.get(c["title"]) if "title" in c else None)
        date = date or (r.get(c["date"]) if "date" in c else None)
    start = None
    if date:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                start = datetime.strptime(date.strip()[:19], fmt).isoformat()
                break
            except ValueError:
                continue
    return _number_sets(sets), {"source": source, "title": title, "start_local": start}


def parse_csv(text, default_unit=None):
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    rd = csv.DictReader(io.StringIO(text), dialect=dialect)
    rows = list(rd)
    src = "strong" if "Exercise Name" in (rd.fieldnames or []) and "Set Order" in (rd.fieldnames or []) else "csv"
    if src == "strong":
        # one export holds many workouts: keep the most recent one
        dates = sorted({r.get("Date") for r in rows if r.get("Date")})
        if dates:
            rows = [r for r in rows if r.get("Date") == dates[-1]]
    return parse_table(rows, rd.fieldnames or [], src, default_unit)


# ------------------------------------------------------------------ free text
_SXR_AT = re.compile(r"(\d+)\s*[x×]\s*(\d+)\s*(?:@|at)\s*([\d.]+)\s*(kg|lbs?)?", re.I)      # 3x5 @ 80kg
_W_X_R = re.compile(r"([\d.]+)\s*(kg|lbs?)?\s*[x×]\s*(\d+)(?:\s*[x×]\s*(\d+))?", re.I)    # 80kg x 5 (x 3)
_R_AT_W = re.compile(r"^(\d+)\s*(?:reps?\s*)?(?:@|at)\s*([\d.]+)\s*(kg|lbs?)\b", re.I)   # 8 @ 60 kg
_RPE = re.compile(r"@\s*(?:rpe\s*)?([\d.]+)(?!\s*(?:kg|lb))", re.I)


def parse_text(text, default_unit=None):
    """Hand-typed notes. Units per set; a set with no unit uses the document's unit
    (lb only if the notes mention lb and never kg)."""
    default_lb = (default_unit or "").lower().startswith("lb") or (
        default_unit is None and bool(re.search(r"\d\s*lbs?\b", text, re.I))
        and not re.search(r"\d\s*kg\b", text, re.I))

    def to_kg(v, unit):
        lb = unit.lower().startswith("lb") if unit else default_lb
        return float(v) * (LB if lb else 1.0)

    sets, exercise = [], None
    lines = [ln.strip() for ln in text.splitlines()]
    title = next((ln for ln in lines if ln), None)
    date = parse_date_line(text)
    for line in lines:
        if not line or line == title or (date and parse_date_line(line)):
            continue
        head, sep, rest = line.partition(":")
        if sep and not re.search(r"\d", head):
            exercise, body = head.strip(), rest
        else:
            body = line
        found = False
        for chunk in re.split(r"[,;]", body):
            chunk = chunk.strip()
            warm = bool(re.search(r"warm", chunk, re.I))
            m = _SXR_AT.search(chunk)
            # "3x5 @ 80kg" = 3 sets of 5; "70 x 5 @8" = 70 kg x 5 at RPE 8
            if m and exercise and int(m[1]) <= 10 and (m[4] or float(m[3]) > 12):
                for _ in range(int(m[1])):
                    sets.append(_set(exercise, to_kg(m[3], m[4]), m[2], None, warm))
                found = True
                continue
            m = _R_AT_W.search(chunk)
            if m and exercise:                       # reps @ weight (needs a unit: "8 @ 9" is RPE)
                rpe = _RPE.search(chunk[m.end():])
                sets.append(_set(exercise, to_kg(m[2], m[3]), int(m[1]), rpe[1] if rpe else None, warm))
                found = True
                continue
            m = _W_X_R.search(chunk)
            if m and exercise:
                a, unit, b2, c = m.groups()
                rpe = _RPE.search(chunk[m.end():])
                if unit or "." in a or float(a) > 10 or c:
                    kg, reps, n = to_kg(a, unit), int(b2), int(c or 1)     # weight x reps (x sets)
                else:
                    kg, reps, n = 0.0, int(b2), int(a)                      # bodyweight: sets x reps
                for _ in range(n):
                    sets.append(_set(exercise, kg, reps, rpe[1] if rpe else None, warm))
                found = True
        if not found and not re.search(r"\d+\s*[x×]\s*\d+", line):
            exercise = re.sub(r"^\d+\s*[.)]\s*", "", line.rstrip(":").strip())   # "1. Bench Press"
    return _number_sets([s for s in sets if s["exercise"]]), {
        "source": "text", "title": title, "start_local": date}


def parse_json(text):
    data = json.loads(text)
    items = data.get("sets", data) if isinstance(data, dict) else data
    sets = []
    for r in items:
        kg = r.get("kg")
        if kg is None:
            w = r.get("weight", 0) or 0
            kg = w * LB if str(r.get("unit", "kg")).lower().startswith("lb") else w
        sets.append(_set(r["exercise"], kg, r["reps"], r.get("rpe"), r.get("warmup", False), r.get("set")))
    info = data if isinstance(data, dict) else {}
    return _number_sets(sets), {"source": "json", "title": info.get("title"),
                                "start_local": info.get("start_local")}


def detect_format(text):
    t = text.lstrip()
    if t.startswith(("[", "{")):
        return "json"
    if re.search(r"^\s*Set\s+\d+:\s*[\d.]+\s*(kg|lbs?)\s*x", text, re.I | re.M):
        return "hevy"
    first = t.splitlines()[0] if t else ""
    if first.count(",") >= 2 or first.count(";") >= 2 or first.count("\t") >= 2:
        # a header row: some cell *is* a reps column ("Reps"), not "26 sets / 169 reps"
        cells = re.split(r"[,;\t]", first)
        if any(re.fullmatch(r"\s*\"?(reps?|repetitions)\b[^\d]{0,12}", c, re.I) for c in cells):
            return "csv"
    return "text"


PARSERS = {"hevy": parse_hevy, "csv": parse_csv, "strong": parse_csv, "text": parse_text, "json": parse_json}


def parse_log(text, fmt="auto", default_unit=None):
    """Parse any supported workout log. Returns (sets, info).

    default_unit: "kg" or "lb" for weights written without a unit (text and CSV logs).
    """
    fmt = detect_format(text) if fmt == "auto" else fmt
    if fmt in ("text", "csv", "strong"):
        sets, info = PARSERS[fmt](text, default_unit)
    else:
        sets, info = PARSERS[fmt](text)
    if not sets:
        raise ValueError(f"no sets found in the workout log (format: {fmt})")
    return sets, info
