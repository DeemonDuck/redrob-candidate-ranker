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
 