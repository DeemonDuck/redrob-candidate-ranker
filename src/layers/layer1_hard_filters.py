from datetime import date, datetime
from src.utils.constants import (
    PURE_SERVICES_COMPANIES,
    WRONG_DOMAIN_KEYWORDS,
    MUST_HAVE_SKILLS,
    LLM_WRAPPER_ONLY_SKILLS,
    NLP_IR_KEYWORDS,
    MIN_YEARS_EXPERIENCE,
    MAX_INACTIVE_MONTHS,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    return text.lower().strip()

def _skill_names(candidate: dict) -> set[str]:
    return {_normalise(s["name"]) for s in candidate.get("skills", [])}

def _career_history(candidate: dict) -> list[dict]:
    return candidate.get("career_history", [])

def _all_career_text(candidate: dict) -> str:
    """Concatenate all titles + descriptions for keyword search."""
    parts = []
    for job in _career_history(candidate):
        parts.append(_normalise(job.get("title", "")))
        parts.append(_normalise(job.get("description", "")))
    return " ".join(parts)

def _months_since(date_str: str) -> int:
    """Return how many months ago a date string (YYYY-MM-DD) was."""
    if not date_str:
        return 9999
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        today = date.today()
        return (today.year - d.year) * 12 + (today.month - d.month)
    except ValueError:
        return 9999



# ── Rule 1: Experience too low ────────────────────────────────────────────────

"""JD: 5-9 year band; under 3 is a definitive no."""

def check_experience_too_low(candidate: dict) -> tuple[bool, str]:
    
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)
    if yoe < MIN_YEARS_EXPERIENCE:
        return True, f"years_of_experience={yoe} < {MIN_YEARS_EXPERIENCE} (hard floor)"
    return False, ""


# ── Rule 2: Pure services company entire career ───────────────────────────────

"""JD: entire career at TCS/Wipro/Infosys etc. → not a fit."""

def check_pure_services_career(candidate: dict) -> tuple[bool, str]:
    
    history = _career_history(candidate)
    if not history:
        return False, ""

    services_count = sum(
        1 for job in history
        if _normalise(job.get("company", "")) in PURE_SERVICES_COMPANIES
    )
    if services_count == len(history):
        return True, "Entire career at pure-services companies (no product company experience)"
    return False, ""
