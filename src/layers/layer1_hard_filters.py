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

# ── Rule 3: Wrong domain — CV/speech/robotics, no NLP/IR ────────────────────

"""
    JD: PRIMARY expertise is CV/speech/robotics without NLP/IR → not a fit.
    'Primary' = majority of career months, not just any mention.
    A candidate who did CV for 2 years then search for 5 years is fine.
    """

def check_wrong_domain(candidate: dict) -> tuple[bool, str]:
    
    history = _career_history(candidate)
    if not history:
        return False, ""
 
    wrong_domain_months = 0
    nlp_ir_months = 0
    total_months = 0
 
    for job in history:
        duration = job.get("duration_months", 0)
        total_months += duration
        text = _normalise(job.get("title", "")) + " " + _normalise(job.get("description", ""))
 
        if any(kw in text for kw in WRONG_DOMAIN_KEYWORDS):
            wrong_domain_months += duration
        if any(kw in text for kw in NLP_IR_KEYWORDS):
            nlp_ir_months += duration
 
    # Also check skills for NLP/IR signals (Tier 5 candidates)
    skills = _skill_names(candidate)
    has_nlp_ir_skills = any(kw in skills for kw in NLP_IR_KEYWORDS)
 
    if total_months == 0:
        return False, ""
 
    wrong_ratio = wrong_domain_months / total_months
    # Eliminate only if: >50% of career is wrong domain AND no meaningful NLP/IR anywhere
    if wrong_ratio > 0.5 and nlp_ir_months == 0 and not has_nlp_ir_skills:
        return True, f"Primary domain is CV/speech/robotics ({wrong_ratio:.0%} of career) with no NLP/IR exposure"
    return False, ""
 

# ── Rule 4: Zero must-have skills ────────────────────────────────────────────

"""Candidate has no overlap with JD's core skill list at all."""

def check_zero_must_have_skills(candidate: dict) -> tuple[bool, str]:
    
    skills = _skill_names(candidate)
    career_text = _all_career_text(candidate)

    # Check skills list AND career text (Tier 5 candidates may not list skills formally)
    matched = skills & MUST_HAVE_SKILLS
    career_match = any(kw in career_text for kw in MUST_HAVE_SKILLS)

    if not matched and not career_match:
        return True, "Zero must-have skills matched in skills list or career history"
    return False, ""


# ── Rule 5: Pure research, no production deployment ──────────────────────────

"""JD: pure research/academic, no production deployment → explicit disqualifier."""

def check_pure_research_no_production(candidate: dict) -> tuple[bool, str]:
    
    history = _career_history(candidate)
    if not history:
        return False, ""

    research_titles = {"researcher", "research scientist", "research engineer",
                       "phd student", "postdoc", "postdoctoral", "intern"}
    production_signals = {
        "deployed", "production", "shipped", "served", "api",
        "service", "platform", "users", "scale", "product"
    }

    all_research = all(
        any(rt in _normalise(job.get("title", "")) for rt in research_titles)
        for job in history
    )
    career_text = _all_career_text(candidate)
    has_production = any(sig in career_text for sig in production_signals)

    if all_research and not has_production:
        return True, "Pure research/academic background with no production deployment signals"
    return False, ""


# ── Rule 6: Recent LLM wrapper only, no pre-LLM ML background ────────────────

"""
JD: <12 months AI experience = only LangChain/OpenAI wrappers,
no pre-LLM production ML → disqualifier.
"""

def check_llm_wrapper_only(candidate: dict) -> tuple[bool, str]:

    skills = _skill_names(candidate)
    career_text = _all_career_text(candidate)

    has_llm_wrapper = bool(skills & LLM_WRAPPER_ONLY_SKILLS)
    has_pre_llm_signal = any(kw in career_text for kw in {
        "search", "retrieval", "ranking", "recommendation",
        "embedding", "bm25", "index", "classification",
        "regression", "tensorflow", "pytorch", "keras",
        "sklearn", "scikit", "xgboost", "lightgbm",
    })

    # Only a disqualifier when LLM wrappers are the ONLY signal AND no pre-LLM work
    if has_llm_wrapper and not has_pre_llm_signal:
        return True, "AI skills appear limited to LLM wrappers (LangChain/OpenAI) with no pre-LLM ML background"
    return False, ""

# ── Master function ───────────────────────────────────────────────────────────

LAYER1_CHECKS = [
    check_experience_too_low,
    check_pure_services_career,
    check_wrong_domain,
    check_zero_must_have_skills,
    check_pure_research_no_production,
    check_llm_wrapper_only,
]

def apply_layer1(candidate: dict) -> tuple[bool, str]:
    """
    Run all Layer 1 checks on a candidate.
    Returns (True, reason) on first disqualification hit.
    Returns (False, "") if candidate passes all checks.
    """
    for check in LAYER1_CHECKS:
        disqualified, reason = check(candidate)
        if disqualified:
            return True, reason
    return False, ""
