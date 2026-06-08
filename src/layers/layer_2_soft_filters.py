from src.utils.constants import MUST_HAVE_SKILLS, PURE_SERVICES_COMPANIES
 
 
# ── helpers ───────────────────────────────────────────────────────────────────
 
def _normalise(text: str) -> str:
    return text.lower().strip()

"""Return normalised text for each job separately."""

def _career_text_per_job(candidate: dict) -> list[str]:
    
    texts = []
    for job in candidate.get("career_history", []):
        t = _normalise(job.get("title", ""))
        d = _normalise(job.get("description", ""))
        texts.append(t + " " + d)
    return texts

# ── Check 1: Career-skills alignment ─────────────────────────────────────────

    """
    Skills claimed must show up in at least one job description
    OR have a platform assessment score (meaning Redrob verified it).
    
    Alignment ratio < 0.25 = too many ghost skills = eliminated.
    
    Ghost skill = listed in skills but:
      - never mentioned in any job description AND
      - no assessment score from redrob_signals
    """

def check_career_skills_alignment(candidate: dict) -> tuple[bool, str]:

    skills = candidate.get("skills", [])
    if not skills:
        return False, ""
 
    assessment_scores = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
    career_texts = _career_text_per_job(candidate)
    all_career_text = " ".join(career_texts)
 
    ghost_count = 0
    total = len(skills)
 
    for skill in skills:
        name = _normalise(skill["name"])
        in_career = any(name in text for text in career_texts)
        # partial match — e.g. "elasticsearch" matches "elastic"
        partial_in_career = any(
            word in all_career_text
            for word in name.split()
            if len(word) > 4  # skip short words like "or", "and"
        )
        has_assessment = skill["name"] in assessment_scores
 
        if not in_career and not partial_in_career and not has_assessment:
            ghost_count += 1
 
    alignment_ratio = 1 - (ghost_count / total)
 
    if alignment_ratio < 0.25:
        return (
            True,
            f"Career-skills misalignment: only {alignment_ratio:.0%} of skills "
            f"appear in career history or have assessment scores "
            f"({ghost_count}/{total} ghost skills)"
        )
    return False, ""



# ── Check 2: Trusted must-have skill count ────────────────────────────────────

    """
    Layer 1 only required >=1 must-have skill to exist anywhere.
    Layer 2 requires >=2 must-have skills with real proof:
      proof = endorsements > 0 OR duration_months > 6 OR has assessment score
    
    Catches candidates who list one legit keyword + many ghost skills.
    """

def check_trusted_must_have_count(candidate: dict) -> tuple[bool, str]:

    skills = candidate.get("skills", [])
    assessment_scores = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
    career_texts = " ".join(_career_text_per_job(candidate))
 
    trusted_must_haves = []
 
    for skill in skills:
        name = _normalise(skill["name"])
        if name not in MUST_HAVE_SKILLS:
            # also check partial — e.g. "elasticsearch" not exact but valid
            if not any(mh in name or name in mh for mh in MUST_HAVE_SKILLS):
                continue
 
        has_proof = (
            skill.get("endorsements", 0) > 0
            or skill.get("duration_months", 0) > 6
            or skill["name"] in assessment_scores
            or name in career_texts  # mentioned in actual work
        )
 
        if has_proof:
            trusted_must_haves.append(skill["name"])
 
    if len(trusted_must_haves) < 2:
        return (
            True,
            f"Fewer than 2 trusted must-have skills "
            f"(found: {trusted_must_haves if trusted_must_haves else 'none'})"
        )
    return False, ""


# ── Check 3: Implicit services-only pattern ───────────────────────────────────

    """
    Catches services companies NOT in our hardcoded list.
    Pattern: ALL jobs are at (IT Services industry + large company size)
    with no product company signal anywhere.
 
    This is what catches Mindtree, Tech Mahindra, Mphasis etc.
    without needing to hardcode every name.
    """

def check_implicit_services_pattern(candidate: dict) -> tuple[bool, str]:

    history = candidate.get("career_history", [])
    if not history:
        return False, ""
 
    SERVICES_INDUSTRIES = {"it services", "information technology", "consulting", "outsourcing"}
    LARGE_COMPANY_SIZES = {"1001-5000", "5001-10000", "10001+"}
    PRODUCT_SIGNALS = {
        "product", "saas", "platform", "startup", "fintech",
        "edtech", "healthtech", "ecommerce", "marketplace"
    }
 
    services_months = 0
    product_months = 0
    total_months = 0
 
    for job in history:
        duration = job.get("duration_months", 0)
        total_months += duration
        industry = _normalise(job.get("industry", ""))
        company_size = job.get("company_size", "")
        company_name = _normalise(job.get("company", ""))
        description = _normalise(job.get("description", ""))
 
        # Already caught by Layer 1 hardcoded list
        if company_name in PURE_SERVICES_COMPANIES:
            services_months += duration
            continue
 
        is_services_industry = any(s in industry for s in SERVICES_INDUSTRIES)
        is_large = company_size in LARGE_COMPANY_SIZES
        has_product_signal = any(p in description or p in company_name for p in PRODUCT_SIGNALS)
 
        if is_services_industry and is_large and not has_product_signal:
            services_months += duration
        else:
            product_months += duration
 
    if total_months == 0:
        return False, ""
 
    services_ratio = services_months / total_months
 
    if services_ratio >= 1.0 and product_months == 0:
        return (
            True,
            f"Implicit services-only pattern: 100% of career at large IT services "
            f"companies with no product company signal"
        )
    return False, ""
 
 
# ── Master function ───────────────────────────────────────────────────────────
 
LAYER2_CHECKS = [
    check_career_skills_alignment,
    check_trusted_must_have_count,
    check_implicit_services_pattern,
]
 
def apply_layer2(candidate: dict) -> tuple[bool, str]:
    """
    Run all Layer 2 soft filters.
    Returns (True, reason) on first elimination.
    Returns (False, "") if candidate passes.
    """
    for check in LAYER2_CHECKS:
        eliminated, reason = check(candidate)
        if eliminated:
            return True, reason
    return False, ""
 