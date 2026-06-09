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




# ── A2: Career score ──────────────────────────────────────────────────
 
PRODUCT_INDUSTRIES = {
    "food delivery", "fintech", "edtech", "healthtech", "ecommerce",
    "saas", "ai/ml", "transportation", "marketplace", "gaming",
    "media", "travel", "proptech", "insurtech",
}
SERVICES_INDUSTRIES = {"it services", "information technology", "consulting", "outsourcing"}
LARGE_SIZES = {"1001-5000", "5001-10000", "10001+"}
 
def score_career(candidate: dict) -> float:
    """
    Three components:
      1. product_company_ratio  (0.5 weight) — JD strongly prefers product cos
      2. ai_role_ratio          (0.3 weight) — time spent in actual AI/ML roles
      3. title_relevance        (0.2 weight) — current title alignment with JD
    """
    history = candidate.get("career_history", [])
    if not history:
        return 0.0
 
    total_months = sum(j.get("duration_months", 0) for j in history)
    if total_months == 0:
        return 0.0
 
    product_months = 0
    ai_months = 0
 
    AI_TITLE_KEYWORDS = {
        "ml", "machine learning", "ai", "nlp", "data science",
        "recommendation", "search", "ranking", "retrieval",
        "applied scientist", "research engineer"
    }
 
    for job in history:
        duration = job.get("duration_months", 0)
        industry = _norm(job.get("industry", ""))
        company_name = _norm(job.get("company", ""))
        title = _norm(job.get("title", ""))
        description = _norm(job.get("description", ""))
        company_size = job.get("company_size", "")
 
        # Product company detection
        is_services = (
            company_name in PURE_SERVICES_COMPANIES
            or (any(s in industry for s in SERVICES_INDUSTRIES) and company_size in LARGE_SIZES)
        )
        if not is_services:
            product_months += duration
 
        # AI role detection
        if any(kw in title or kw in description for kw in AI_TITLE_KEYWORDS):
            ai_months += duration
 
    product_ratio = product_months / total_months
    ai_ratio = min(ai_months / total_months, 1.0)
 
    # Title relevance — current title
    current_title = _norm(candidate.get("profile", {}).get("current_title", ""))
    title_score = 1.0 if any(kw in current_title for kw in AI_TITLE_KEYWORDS) else 0.3
 
    return round(0.5 * product_ratio + 0.3 * ai_ratio + 0.2 * title_score, 4)



# ── A3: Experience score ──────────────────────────────────────────────
 
def score_experience(candidate: dict) -> float:
    """
    JD sweet spot: 5–9 years. Soft penalties outside that range.
    Not a hard cutoff — "some people hit senior judgment at 4 years."
    """
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)
 
    if 5 <= yoe <= 9:
        return 1.0
    elif 4 <= yoe < 5:
        return 0.8
    elif 9 < yoe <= 12:
        return 0.75
    elif 3 <= yoe < 4:
        return 0.5   # Layer 1 floor is 3; 3-4 is weak but alive
    elif yoe > 12:
        return 0.5   # Over-experienced, possible mismatch
    else:
        return 0.1




# ── A4: Education score ───────────────────────────────────────────────
 
RELEVANT_FIELDS = {
    "computer science", "computer engineering", "software engineering",
    "artificial intelligence", "machine learning", "data science",
    "statistics", "mathematics", "information technology",
    "electronics", "ece", "electrical engineering",
}
 
TIER_MAP = {"tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.5, "tier_4": 0.25, "unknown": 0.4}
 
def score_education(candidate: dict) -> float:
    """
    Low weight feature — JD never mentions education as a criterion.
    Relevant degree (0.6 weight) + institution tier (0.4 weight).
    """
    education = candidate.get("education", [])
    if not education:
        return 0.3  # no info, don't penalise heavily
 
    # Take best education entry
    best_tier = 0.0
    relevant_degree = False
 
    for edu in education:
        field = _norm(edu.get("field_of_study", ""))
        tier = TIER_MAP.get(edu.get("tier", "unknown"), 0.4)
        best_tier = max(best_tier, tier)
 
        if any(f in field for f in RELEVANT_FIELDS):
            relevant_degree = True
 
    degree_score = 1.0 if relevant_degree else 0.4
    return round(0.6 * degree_score + 0.4 * best_tier, 4)




# ── Composite profile score ───────────────────────────────────────────
 
PROFILE_WEIGHTS = {
    "skills":     0.40,   # most important — JD is skills-heavy
    "career":     0.35,   # product co + AI role history
    "experience": 0.15,   # years matter but not decisive
    "education":  0.10,   # low weight — JD never mandates it
}
 
def compute_profile_score(candidate: dict) -> dict:
    skills_s     = score_skills(candidate)
    career_s     = score_career(candidate)
    experience_s = score_experience(candidate)
    education_s  = score_education(candidate)
 
    profile_score = (
        PROFILE_WEIGHTS["skills"]     * skills_s +
        PROFILE_WEIGHTS["career"]     * career_s +
        PROFILE_WEIGHTS["experience"] * experience_s +
        PROFILE_WEIGHTS["education"]  * education_s
    )
 
    return {
        "skills_score":     skills_s,
        "career_score":     career_s,
        "experience_score": experience_s,
        "education_score":  education_s,
        "profile_score":    round(profile_score, 4),
    }
 

