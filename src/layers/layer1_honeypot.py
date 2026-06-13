"""
layer1_honeypot.py

Detects impossible/fake profiles (honeypots) before any scoring.
~80 honeypots exist in the 100K pool. >10% in top-100 = disqualification.

Patterns detected:
  1. Impossible tenure — job duration > company age
  2. Expert skills with zero usage
  3. Experience vs history mismatch (both directions)
  4. Duplicate job descriptions across different companies
"""

from datetime import date, datetime


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_year(date_str: str) -> int | None:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").year
    except (ValueError, TypeError):
        return None

def _current_year() -> int:
    return date.today().year


# ── Check 1: Impossible tenure ────────────────────────────────────────────────

def check_impossible_tenure(candidate: dict) -> tuple[bool, str]:
    history = candidate.get("career_history", [])
    for job in history:
        duration_months = job.get("duration_months", 0)
        start_year = _parse_year(job.get("start_date", ""))
        if not start_year:
            continue
        company_age_months = (_current_year() - start_year) * 12
        if duration_months > company_age_months + 12:
            return (
                True,
                f"Impossible tenure: {duration_months} months at '{job.get('company')}' "
                f"but role started {start_year} ({company_age_months} months ago)"
            )
    return False, ""


# ── Check 2: Expert skills with zero usage ────────────────────────────────────

def check_expert_skills_zero_duration(candidate: dict) -> tuple[bool, str]:
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


# ── Check 3: Experience vs history mismatch (BOTH directions) ─────────────────

def check_experience_vs_history_mismatch(candidate: dict) -> tuple[bool, str]:
    """
    Original: catches claimed > history by >6 years (fabricated experience).
    NEW: also catches history > claimed by >2 years (Finding #6 — profile=3.0, history=4.9).
    Both directions are suspicious — real candidates have roughly consistent numbers.
    """
    stated_years = candidate.get("profile", {}).get("years_of_experience", 0)
    history = candidate.get("career_history", [])
    total_months = sum(job.get("duration_months", 0) for job in history)
    history_years = total_months / 12

    # Original check — claimed far more than history
    if stated_years > history_years + 6:
        return (
            True,
            f"Experience mismatch: claims {stated_years:.1f} years but "
            f"career history only accounts for {history_years:.1f} years"
        )

    # History far more than claimed (honeypot pattern Finding #6)
    if history_years > stated_years + 2:
        return (
            True,
            f"Experience mismatch: career history totals {history_years:.1f} years "
            f"but profile claims only {stated_years:.1f} years"
        )

    return False, ""


# ── Check 4 Duplicate job descriptions across companies ────────────────

def check_duplicate_job_descriptions(candidate: dict) -> tuple[bool, str]:
    """
    Finding #2 — same paragraph copy-pasted across different companies.
    Real candidates write different descriptions for different jobs.
    Honeypot generator reuses description templates.

    Method: compare normalised description fingerprints.
    If 2+ jobs share >85% of their content → honeypot.
    """
    history = candidate.get("career_history", [])
    if len(history) < 2:
        return False, ""

    def _fingerprint(text: str) -> set:
        # Use word trigrams as fingerprint — robust to minor edits
        words = text.lower().split()
        if len(words) < 3:
            return set(words)
        return {" ".join(words[i:i+3]) for i in range(len(words) - 2)}

    descriptions = []
    for job in history:
        desc = job.get("description", "").strip()
        if desc:
            descriptions.append((job.get("company", "?"), _fingerprint(desc)))

    # Compare every pair
    for i in range(len(descriptions)):
        for j in range(i + 1, len(descriptions)):
            co_a, fp_a = descriptions[i]
            co_b, fp_b = descriptions[j]
            if not fp_a or not fp_b:
                continue
            # Jaccard similarity
            intersection = len(fp_a & fp_b)
            union = len(fp_a | fp_b)
            similarity = intersection / union if union > 0 else 0

            if similarity > 0.99:
                return (
                    True,
                    f"Duplicate job descriptions: '{co_a}' and '{co_b}' "
                    f"share {similarity:.0%} content — likely synthetic profile"
                )

    return False, ""


# ── Master function ───────────────────────────────────────────────────────────

HONEYPOT_CHECKS = [
    check_impossible_tenure,
    check_expert_skills_zero_duration,
    check_experience_vs_history_mismatch
    #check_duplicate_job_descriptions,
]

def apply_honeypot_check(candidate: dict) -> tuple[bool, str]:
    for check in HONEYPOT_CHECKS:
        is_honeypot, reason = check(candidate)
        if is_honeypot:
            return True, reason
    return False, ""