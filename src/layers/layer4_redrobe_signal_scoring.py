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
    

# PART A — PROFILE SCORING --------------------------------------------

# ── A1: Skills score ──────────────────────────────────────────────────

NICE_TO_HAVE_SKILLS = {
    "lora", "qlora", "peft", "fine-tuning", "finetuning",
    "learning to rank", "ltr", "xgboost", "lightgbm",
    "hr tech", "recruiting", "marketplace",
    "distributed systems", "inference optimization",
    "open source", "pytorch", "tensorflow",
}
 
def score_skills(candidate: dict) -> float:
    """
    Weighted skill match.
    Must-have skills (trusted) = 70% weight
    Nice-to-have skills        = 30% weight
 
    Trust = endorsements > 0 OR duration_months > 6 OR in career text OR has assessment
    """
    skills = candidate.get("skills", [])
    assessment_scores = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
    career_text = _career_text(candidate)
 
    must_have_hits = 0
    must_have_total = len(MUST_HAVE_SKILLS)
 
    nice_have_hits = 0
    nice_have_total = len(NICE_TO_HAVE_SKILLS)
 
    for skill in skills:
        name = _norm(skill["name"])
        trusted = (
            skill.get("endorsements", 0) > 0
            or skill.get("duration_months", 0) > 6
            or skill["name"] in assessment_scores
            or name in career_text
        )
 
        if not trusted:
            continue
 
        # Check must-have match (exact or partial)
        if name in MUST_HAVE_SKILLS or any(mh in name or name in mh for mh in MUST_HAVE_SKILLS):
            # Bonus: assessment score available → scale by quality
            if skill["name"] in assessment_scores:
                must_have_hits += assessment_scores[skill["name"]] / 100
            else:
                must_have_hits += 1.0
 
        # Check nice-to-have
        if name in NICE_TO_HAVE_SKILLS or any(nh in name or name in nh for nh in NICE_TO_HAVE_SKILLS):
            nice_have_hits += 1.0
 
    must_score = min(must_have_hits / max(5, 1), 1.0)   # cap at 5 must-haves = full score
    nice_score = min(nice_have_hits / max(3, 1), 1.0)   # cap at 3 nice-to-haves = full score
 
    return round(0.70 * must_score + 0.30 * nice_score, 4)