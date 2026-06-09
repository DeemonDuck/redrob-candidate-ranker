from datetime import date, datetime
from src.utils.constants import MUST_HAVE_SKILLS, PURE_SERVICES_COMPANIES
 
# ── helpers ───────────────────────────────────────────────────────────────────
 
def _norm(text: str) -> str:
    return text.lower().strip()
 
def _career_text(candidate: dict) -> str:
    parts = []
    for job in candidate.get("career_history", []):
        parts.append(_norm(job.get("title", "")))
        parts.append(_norm(job.get("description", "")))
    return " ".join(parts)
 
def _months_since(date_str: str) -> int:
    if not date_str:
        return 999
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        today = date.today()
        return (today.year - d.year) * 12 + (today.month - d.month)
    except ValueError:
        return 999