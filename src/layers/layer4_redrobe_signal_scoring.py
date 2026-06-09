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
 

# ═══════════════════════════════════════════════════════════════════════
# PART B — BEHAVIORAL MULTIPLIER (redrob signals)
# ═══════════════════════════════════════════════════════════════════════
 
def compute_behavioral_multiplier(candidate: dict) -> dict:
    """
    Uses remaining redrob signals not consumed by Layer 3.
    Returns a multiplier in [0.2, 1.0] — never zeros out a good profile,
    but can heavily down-weight an unreachable/unverified candidate.
 
    Signals used:
      - github_activity_score       (credibility)
      - interview_completion_rate   (reliability)
      - offer_acceptance_rate       (reliability)
      - profile_completeness_score  (seriousness)
      - verified_email + phone      (identity trust)
      - linkedin_connected          (identity trust)
      - applications_submitted_30d  (active job seeking)
      - saved_by_recruiters_30d     (market validation)
    """
    s = candidate.get("redrob_signals", {})
 
    # 1. Credibility — github activity (0.25 weight)
    github = s.get("github_activity_score", -1)
    if github == -1:
        github_score = 0.5   # no github — neutral, not penalised heavily
    else:
        github_score = github / 100
 
    # 2. Reliability — interview completion (0.20 weight)
    icr = s.get("interview_completion_rate", 0.5)
    interview_score = icr  # already 0–1
 
    # 3. Reliability — offer acceptance (0.15 weight)
    oar = s.get("offer_acceptance_rate", -1)
    if oar == -1:
        offer_score = 0.5   # no history — neutral
    else:
        offer_score = max(oar, 0)  # -1 already handled above
 
    # 4. Profile seriousness (0.15 weight)
    completeness = s.get("profile_completeness_score", 50) / 100
    verified = (
        int(s.get("verified_email", False)) +
        int(s.get("verified_phone", False)) +
        int(s.get("linkedin_connected", False))
    ) / 3
    seriousness_score = 0.5 * completeness + 0.5 * verified
 
    # 5. Active job seeking (0.15 weight)
    apps = min(s.get("applications_submitted_30d", 0), 10) / 10
    saved = min(s.get("saved_by_recruiters_30d", 0), 10) / 10
    active_score = 0.5 * apps + 0.5 * saved
 
    # 6. Market validation — saved by recruiters (0.10 weight)
    # Already folded into active_score above
 
    behavioral_raw = (
        0.25 * github_score +
        0.20 * interview_score +
        0.15 * offer_score +
        0.15 * seriousness_score +
        0.15 * active_score +
        0.10 * verified   # double-weighting identity trust slightly
    )
 
    # Floor at 0.2 — never completely zero out a good profile
    behavioral_multiplier = max(round(behavioral_raw, 4), 0.2)
 
    return {
        "github_score":       round(github_score, 4),
        "interview_score":    round(interview_score, 4),
        "offer_score":        round(offer_score, 4),
        "seriousness_score":  round(seriousness_score, 4),
        "active_score":       round(active_score, 4),
        "behavioral_multiplier": behavioral_multiplier,
    }


# ═══════════════════════════════════════════════════════════════════════
# MASTER FUNCTION
# ═══════════════════════════════════════════════════════════════════════
 
def apply_layer4(candidate: dict, location_score: float, availability_score: float) -> dict:
    """
    Combines profile score + location + availability + behavioral multiplier
    into a single final_score.
 
    final_score = (profile_score * 0.70 + location_score * 0.15 + availability_score * 0.15)
                  * behavioral_multiplier
 
    Returns full feature dict for Layer 5 (LightGBM re-ranker).
    """
    profile   = compute_profile_score(candidate)
    behavioral = compute_behavioral_multiplier(candidate)
 
    base_score = (
        0.70 * profile["profile_score"] +
        0.15 * location_score +
        0.15 * availability_score
    )
 
    final_score = round(base_score * behavioral["behavioral_multiplier"], 4)
 
    return {
        "candidate_id": candidate["candidate_id"],
        # Component scores
        **profile,
        **behavioral,
        "location_score":      location_score,
        "availability_score":  availability_score,
        # Final
        "base_score":          round(base_score, 4),
        "final_score":         final_score,
    }