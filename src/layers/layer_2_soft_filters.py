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