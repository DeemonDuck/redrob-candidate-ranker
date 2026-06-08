from datetime import date, datetime


# ── helpers ──────────────────────────────────────────────────────────────────

def _parse_year(date_str: str) -> int | None:
    """Extract year from a YYYY-MM-DD date string."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").year
    except (ValueError, TypeError):
        return None

def _current_year() -> int:
    return date.today().year


# ── Check 1: Impossible tenure ────────────────────────────────────────────────

def check_impossible_tenure(candidate: dict) -> tuple[bool, str]:
    """
    Pattern: candidate claims N years at a company, but the company's
    start_date implies the company is younger than their tenure there.

    E.g. worked at a company for 8 years but start_date of that job
    means the company would have been founded only 3 years ago.
    """
    history = candidate.get("career_history", [])

    for job in history:
        duration_months = job.get("duration_months", 0)
        start_date_str = job.get("start_date", "")
        start_year = _parse_year(start_date_str)

        if not start_year:
            continue

        company_age_months = (_current_year() - start_year) * 12

        # If the job duration is longer than the company could possibly exist
        # (i.e. the job started before year 0 relative to now) — flag it
        # More concretely: duration > total months since start_date means
        # they would have started before the company existed
        if duration_months > company_age_months + 12:  # +12 for date rounding tolerance
            return (
                True,
                f"Impossible tenure: {duration_months} months at '{job.get('company')}' "
                f"but role started {start_year} ({company_age_months} months ago)"
            )

    return False, ""


# ── Check 2: Expert skills with zero usage ────────────────────────────────────

def check_expert_skills_zero_duration(candidate: dict) -> tuple[bool, str]:
    """
    Pattern: multiple 'expert' proficiency skills with duration_months = 0.
    One might be a data entry error. Many is a honeypot signal.
    Threshold: >= 3 expert skills with duration_months = 0.
    """
    skills = candidate.get("skills", [])

    fake_expert_skills = [
        s["name"] for s in skills
        if s.get("proficiency") == "expert"
        and s.get("duration_months", 0) == 0
        and s.get("endorsements", 0) == 0
    ]

    if len(fake_expert_skills) >= 3:
        return (
            True,
            f"Honeypot signal: {len(fake_expert_skills)} 'expert' skills "
            f"with 0 months usage and 0 endorsements: {fake_expert_skills[:5]}"
        )

    return False, ""


# ── Check 3: Years of experience vs career history mismatch ──────────────────

def check_experience_vs_history_mismatch(candidate: dict) -> tuple[bool, str]:
    """
    Sanity check: profile.years_of_experience vs sum of career_history durations.
    If the mismatch is extreme (>5 years), it's a fabrication signal.
    """
    stated_years = candidate.get("profile", {}).get("years_of_experience", 0)
    history = candidate.get("career_history", [])

    total_months = sum(job.get("duration_months", 0) for job in history)
    history_years = total_months / 12

    # Allow generous tolerance for overlapping roles, gaps, freelance etc.
    if stated_years > history_years + 6:
        return (
            True,
            f"Experience mismatch: claims {stated_years:.1f} years but "
            f"career history only accounts for {history_years:.1f} years"
        )

    return False, ""


# ── Master function ───────────────────────────────────────────────────────────

HONEYPOT_CHECKS = [
    check_impossible_tenure,
    check_expert_skills_zero_duration,
    check_experience_vs_history_mismatch,
]

def apply_honeypot_check(candidate: dict) -> tuple[bool, str]:
    """
    Run all honeypot checks.
    Returns (True, reason) on first hit.
    Returns (False, "") if profile appears legitimate.
    """
    for check in HONEYPOT_CHECKS:
        is_honeypot, reason = check(candidate)
        if is_honeypot:
            return True, reason
    return False, ""