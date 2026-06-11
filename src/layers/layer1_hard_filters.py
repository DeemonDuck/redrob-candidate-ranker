"""
layer1_hard_filters.py

STRICT eliminations only — absolute disqualifiers from the JD.
Only 3 rules remain. Rules 4, 5, 6 moved to Layer 2 as weighted signals.

Rule 1: years_of_experience < 3
Rule 2: Entire career at pure services companies
Rule 3: Primary domain CV/speech/robotics with zero NLP/IR exposure
"""

from datetime import date, datetime
from src.utils.constants import (
    PURE_SERVICES_COMPANIES,
    WRONG_DOMAIN_KEYWORDS,
    NLP_IR_KEYWORDS,
    MIN_YEARS_EXPERIENCE,
)


def _normalise(text: str) -> str:
    return text.lower().strip()

def _career_history(candidate: dict) -> list[dict]:
    return candidate.get("career_history", [])

def _skill_names(candidate: dict) -> set[str]:
    return {_normalise(s["name"]) for s in candidate.get("skills", [])}


# ── Rule 1: Experience too low ────────────────────────────────────────────────

def check_experience_too_low(candidate: dict) -> tuple[bool, str]:
    """Hard floor — under 3 years is definitive no per JD."""
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)
    if yoe < MIN_YEARS_EXPERIENCE:
        return True, f"years_of_experience={yoe} < {MIN_YEARS_EXPERIENCE} (hard floor)"
    return False, ""


# ── Rule 2: Pure services company entire career ───────────────────────────────

def check_pure_services_career(candidate: dict) -> tuple[bool, str]:
    """JD: entire career at named services companies → not a fit."""
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

def check_wrong_domain(candidate: dict) -> tuple[bool, str]:
    """
    PRIMARY expertise is CV/speech/robotics without ANY NLP/IR exposure.
    Uses duration ratio — must be >50% wrong domain AND zero NLP/IR anywhere.
    """
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

    skills = _skill_names(candidate)
    has_nlp_ir_skills = any(kw in skills for kw in NLP_IR_KEYWORDS)

    if total_months == 0:
        return False, ""

    wrong_ratio = wrong_domain_months / total_months
    if wrong_ratio > 0.5 and nlp_ir_months == 0 and not has_nlp_ir_skills:
        return True, f"Primary domain is CV/speech/robotics ({wrong_ratio:.0%} of career) with no NLP/IR exposure"
    return False, ""


# ── Master function ───────────────────────────────────────────────────────────

LAYER1_CHECKS = [
    check_experience_too_low,
    check_pure_services_career,
    check_wrong_domain,
]

def apply_layer1(candidate: dict) -> tuple[bool, str]:
    for check in LAYER1_CHECKS:
        disqualified, reason = check(candidate)
        if disqualified:
            return True, reason
    return False, ""