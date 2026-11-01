"""Inject analysis/out/workout_2026-11-01.json into report/template.html."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
tpl = (ROOT / "report/template.html").read_text()
data = (ROOT / "analysis/out/workout_2026-11-01.json").read_text().replace("</", "<\\/")
out = ROOT / "report/full-body-c-2026-11-01.html"
out.write_text(tpl.replace("/*DATA*/", data))
print(out, out.stat().st_size // 1024, "KB")
