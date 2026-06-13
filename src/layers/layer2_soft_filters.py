"""
layer2_soft_filters.py

Converted from elimination → weighted scoring.
Returns a penalty_score in [0.0, 1.0] where:
  1.0 = no issues found
  0.0 = severe issues across all dimensions

Only hard elimination remaining: implicit services-only pattern (100% services
career with zero product signal — still a definitive no).

"""

from src.utils.constants import MUST_HAVE_SKILLS, PURE_SERVICES_COMPANIES, LLM_WRAPPER_ONLY_SKILLS

MUST_HAVE_SORTED   = sorted(MUST_HAVE_SKILLS)
LLM_WRAPPER_SORTED = sorted(LLM_WRAPPER_ONLY_SKILLS)


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _normalise(text: str) -> str:
    return text.lower().strip()

def _career_text_per_job(candidate: dict) -> list[str]:
    texts = []
    for job in candidate.get("career_history", []):
        t = _normalise(job.get("title", ""))
        d = _normalise(job.get("description", ""))
        texts.append(t + " " + d)
    return texts

def _all_career_text(candidate: dict) -> str:
    return " ".join(_career_text_per_job(candidate))


# ═══════════════════════════════════════════════════════════════════════
# SIGNAL 1: Career-skills alignment score
# ═══════════════════════════════════════════════════════════════════════

def score_career_skills_alignment(candidate: dict) -> float:
    """
    Returns 0.0–1.0.
    Ghost skills (listed but never appear in career or assessments) reduce score.
    Was: eliminate if <0.25. Now: return alignment ratio as a score.
    """
    skills = candidate.get("skills", [])
    if not skills:
        return 0.5  # no skills listed — neutral

    assessment_scores = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
    career_texts      = _career_text_per_job(candidate)
    all_career_text   = " ".join(career_texts)

    ghost_count = 0
    for skill in skills:
        name = _normalise(skill["name"])
        in_career         = any(name in text for text in career_texts)
        partial_in_career = any(
            word in all_career_text
            for word in name.split()
            if len(word) > 4
        )
        has_assessment = skill["name"] in assessment_scores
        if not in_career and not partial_in_career and not has_assessment:
            ghost_count += 1

    return round(1 - (ghost_count / len(skills)), 4)


# ═══════════════════════════════════════════════════════════════════════
# SIGNAL 2: Trusted must-have skill score
# ═══════════════════════════════════════════════════════════════════════

def score_trusted_must_haves(candidate: dict) -> float:
    """
    Returns 0.0–1.0 based on count of trusted must-have skills.
    Was: eliminate if <2. Now: graduated score.
    0 trusted = 0.0, 1 = 0.4, 2 = 0.7, 3+ = 1.0
    """
    skills            = candidate.get("skills", [])
    assessment_scores = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
    career_text       = _all_career_text(candidate)

    trusted_count = 0
    for skill in skills:
        name = _normalise(skill["name"])

        # FIX: iterate MUST_HAVE_SORTED (list) not MUST_HAVE_SKILLS (set)
        # so substring matching order is identical every run
        is_must_have = (
            name in MUST_HAVE_SKILLS
            or any(mh in name or name in mh for mh in MUST_HAVE_SORTED)
        )
        if not is_must_have:
            continue

        has_proof = (
            skill.get("endorsements", 0) > 0
            or skill.get("duration_months", 0) > 6
            or skill["name"] in assessment_scores
            or name in career_text
        )
        if has_proof:
            trusted_count += 1

    if trusted_count == 0:   return 0.0
    elif trusted_count == 1: return 0.4
    elif trusted_count == 2: return 0.7
    else:                    return 1.0


# ═══════════════════════════════════════════════════════════════════════
# SIGNAL 3: Must-have skills in career history (ex-Rule 4)
# ═══════════════════════════════════════════════════════════════════════

def score_must_have_in_career(candidate: dict) -> float:
    """
    Zero must-have skills in skills list OR career text.
    Now graduated — partial career evidence gets partial credit.
    """
    skills      = {_normalise(s["name"]) for s in candidate.get("skills", [])}
    career_text = _all_career_text(candidate)

    # Set intersection is deterministic — no fix needed here
    skill_matches  = len(skills & MUST_HAVE_SKILLS)

    # FIX: iterate MUST_HAVE_SORTED (list) not MUST_HAVE_SKILLS (set)
    career_matches = sum(1 for kw in MUST_HAVE_SORTED if kw in career_text)
    total_signal   = skill_matches + career_matches

    if total_signal == 0:   return 0.0
    elif total_signal <= 2: return 0.4
    elif total_signal <= 4: return 0.7
    else:                   return 1.0


# ═══════════════════════════════════════════════════════════════════════
# SIGNAL 4: Production evidence score (ex-Rule 5)
# ═══════════════════════════════════════════════════════════════════════

def score_production_evidence(candidate: dict) -> float:
    """
    Pure research with no production signals.
    Now: graduated score based on production signal density in career.

    production_signals is a local set used only for membership checks
    (kw in career_text) — not substring iteration — so no fix needed here.
    """
    career_text = _all_career_text(candidate)
    production_signals = {
        "deployed", "production", "shipped", "served", "api",
        "service", "platform", "users", "scale", "product",
        "live", "launch", "rollout", "serving", "real-time"
    }
    matches = sum(1 for sig in production_signals if sig in career_text)

    if matches == 0:   return 0.1
    elif matches <= 2: return 0.5
    elif matches <= 5: return 0.8
    else:              return 1.0


# ═══════════════════════════════════════════════════════════════════════
# SIGNAL 5: LLM wrapper penalty (ex-Rule 6)
# ═══════════════════════════════════════════════════════════════════════

def score_pre_llm_background(candidate: dict) -> float:
    """
    LLM wrapper only with no pre-LLM background.
    Now: graduated — LangChain + no history = 0.2, LangChain + some history = 0.7+

    has_llm_wrapper uses set intersection (&) which is deterministic — no fix needed.
    pre_llm_signals is a local set used only for (kw in career_text) — also fine.
    """
    skills      = {_normalise(s["name"]) for s in candidate.get("skills", [])}
    career_text = _all_career_text(candidate)

    # Set intersection — deterministic, no fix needed
    has_llm_wrapper = bool(skills & LLM_WRAPPER_ONLY_SKILLS)

    pre_llm_signals = {
        "search", "retrieval", "ranking", "recommendation",
        "embedding", "bm25", "index", "classification",
        "tensorflow", "pytorch", "keras", "sklearn",
        "scikit", "xgboost", "lightgbm", "regression",
    }
    pre_llm_count = sum(1 for kw in pre_llm_signals if kw in career_text)

    if not has_llm_wrapper:
        return 1.0   # no LLM wrapper dependency at all — full score
    if pre_llm_count == 0:
        return 0.2   # LLM wrappers only, no pre-LLM evidence
    elif pre_llm_count <= 2:
        return 0.6
    else:
        return 0.9   # LLM wrapper + solid pre-LLM background = fine


# ═══════════════════════════════════════════════════════════════════════
# SIGNAL 6: Implicit services pattern (hard elimination stays)
# ═══════════════════════════════════════════════════════════════════════

def check_implicit_services_hard(candidate: dict) -> tuple[bool, str]:
    """
    Only hard elimination in Layer 2.
    100% of career at large IT services companies with zero product signal.
    """
    history = candidate.get("career_history", [])
    if not history:
        return False, ""

    SERVICES_INDUSTRIES = {"it services", "information technology", "consulting", "outsourcing"}
    LARGE_SIZES         = {"1001-5000", "5001-10000", "10001+"}
    PRODUCT_SIGNALS     = {"product", "saas", "platform", "startup", "fintech",
                           "edtech", "healthtech", "ecommerce", "marketplace"}

    services_months = 0
    product_months  = 0
    total_months    = 0

    for job in history:
        duration     = job.get("duration_months", 0)
        total_months += duration
        industry     = _normalise(job.get("industry", ""))
        company_size = job.get("company_size", "")
        company_name = _normalise(job.get("company", ""))
        description  = _normalise(job.get("description", ""))

        if company_name in PURE_SERVICES_COMPANIES:
            services_months += duration
            continue

        is_services = any(s in industry for s in SERVICES_INDUSTRIES) and company_size in LARGE_SIZES
        has_product = any(p in description or p in company_name for p in PRODUCT_SIGNALS)

        if is_services and not has_product:
            services_months += duration
        else:
            product_months += duration

    if total_months > 0 and services_months / total_months >= 1.0 and product_months == 0:
        return True, "Implicit services-only pattern: 100% of career at large IT services companies"
    return False, ""


# ═══════════════════════════════════════════════════════════════════════
# MASTER FUNCTION
# ═══════════════════════════════════════════════════════════════════════

# Weights for each signal — sum to 1.0
SIGNAL_WEIGHTS = {
    "alignment":   0.20,   # career-skills alignment
    "trusted_mh":  0.20,   # trusted must-have skills
    "mh_career":   0.15,   # must-have evidence in career
    "production":  0.30,   # production deployment evidence
    "pre_llm":     0.15,   # pre-LLM background
}

def apply_layer2(candidate: dict) -> tuple[bool, str, float]:
    """
    Returns (is_eliminated, reason, layer2_score).

    is_eliminated : True only for implicit services-only pattern
    layer2_score  : 0.0–1.0, feeds directly into Layer 4 final_score
    """
    # Hard elimination check first
    eliminated, reason = check_implicit_services_hard(candidate)
    if eliminated:
        return True, reason, 0.0

    # Weighted scoring
    alignment  = score_career_skills_alignment(candidate)
    trusted_mh = score_trusted_must_haves(candidate)
    mh_career  = score_must_have_in_career(candidate)
    production = score_production_evidence(candidate)
    pre_llm    = score_pre_llm_background(candidate)

    layer2_score = (
        SIGNAL_WEIGHTS["alignment"]  * alignment +
        SIGNAL_WEIGHTS["trusted_mh"] * trusted_mh +
        SIGNAL_WEIGHTS["mh_career"]  * mh_career +
        SIGNAL_WEIGHTS["production"] * production +
        SIGNAL_WEIGHTS["pre_llm"]    * pre_llm
    )

    return False, "", round(layer2_score, 4)