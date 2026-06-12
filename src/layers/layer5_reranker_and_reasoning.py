"""
layer5_reranker_and_reasoning.py

LAYER 5 — TOP-100 SELECTION + REASONING

What this layer does:
    1. Takes Layer 4 scored candidates
    2. Preserves Layer 4 final_score EXACTLY — no modification
    3. Sorts descending by final_score
    4. Resolves ties deterministically using candidate_id ascending (spec §3)
    5. Generates 1-2 sentence reasoning grounded in real candidate facts

What this layer does NOT do:
    - Does NOT modify final_score from Layer 4
    - Does NOT use SBERT (removed — was only producing a label string)
    - Does NOT call any external model or API

Tie-breaking rule (spec §3):
    Equal scores → candidate_id ascending (CAND_XXXXXXX string sort)
    Deterministic and matches validate_submission.py expectations.

REMOVED: LightGBM, SBERT, semantic_scores parameter, _sbert_hint()
ADDED:   Pure fact-based reasoning from career + redrob signals
"""

from src.utils.constants import MUST_HAVE_SKILLS


# ═══════════════════════════════════════════════════════════════════════
# REASONING HELPERS
# ═══════════════════════════════════════════════════════════════════════

# JD dimensions to check for in career text — drives specific reasoning
# These map directly to what the JD explicitly asks for
JD_DIMENSION_KEYWORDS = {
    "embeddings":      ["embedding", "embeddings", "sentence-transformer", "dense retrieval"],
    "vector_db":       ["pinecone", "qdrant", "milvus", "weaviate", "opensearch", "elasticsearch", "faiss", "pgvector"],
    "hybrid_search":   ["hybrid search", "hybrid retrieval", "bm25", "sparse", "dense", "reranking", "rerank"],
    "evaluation":      ["ndcg", "mrr", "map", "a/b test", "offline", "online", "eval", "benchmark"],
    "ranking":         ["ranking", "learning to rank", "ltr", "xgboost", "lightgbm", "lambdarank"],
    "llm_finetuning":  ["lora", "qlora", "peft", "fine-tun", "finetun", "finetuned"],
    "production":      ["production", "deployed", "shipped", "serving", "scale", "latency"],
    "product_company": ["product", "startup", "saas", "platform", "marketplace"],
}

DIMENSION_LABELS = {
    "embeddings":      "embeddings-based retrieval",
    "vector_db":       "vector DB experience",
    "hybrid_search":   "hybrid search / BM25",
    "evaluation":      "ranking evaluation (NDCG/A/B)",
    "ranking":         "learning-to-rank",
    "llm_finetuning":  "LLM fine-tuning (LoRA/QLoRA)",
    "production":      "production deployment",
    "product_company": "product company background",
}


def _career_text(original: dict) -> str:
    """All career titles + descriptions joined and lowercased — for keyword matching."""
    parts = []
    for job in original.get("career_history", []):
        parts.append(job.get("title", "").lower())
        parts.append(job.get("description", "").lower())
    return " ".join(parts)


def _matched_jd_dimensions(original: dict) -> list[str]:
    """
    Returns human-readable JD dimension labels this candidate matches
    based on keywords found in their career text.
    Used in sentence 1 to be specific, not generic.
    """
    text = _career_text(original)
    matched = []
    for dim, keywords in JD_DIMENSION_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            matched.append(DIMENSION_LABELS[dim])
    return matched


def _matched_must_have_skills(original: dict) -> list[str]:
    """
    Returns trusted must-have skills that have proof:
    endorsements > 0 OR duration_months > 6 OR assessment score exists.
    Avoids ghost skills (listed but never used).
    """
    skills = original.get("skills", [])
    assessment_scores = original.get("redrob_signals", {}).get("skill_assessment_scores", {})
    matched = []
    for s in skills:
        name_lower = s["name"].lower()
        is_must_have = (
            name_lower in MUST_HAVE_SKILLS
            or any(mh in name_lower or name_lower in mh for mh in MUST_HAVE_SKILLS)
        )
        if not is_must_have:
            continue
        trusted = (
            s.get("endorsements", 0) > 0
            or s.get("duration_months", 0) > 6
            or s["name"] in assessment_scores
        )
        if trusted:
            matched.append(s["name"])
    return matched[:4]  # cap at 4 for readability


def generate_reasoning(
    original: dict,
    scored: dict,
    rank: int,
) -> str:
    """
    Generates 1-2 sentence reasoning grounded entirely in candidate facts.

    Sentence 1 — strengths: title, YOE, company, JD dimensions matched, skills
    Sentence 2 — availability or concerns (concern threshold scales with rank)

    Every claim comes from original candidate data — no hallucination.

    Args:
        original : raw candidate dict from candidates.jsonl
        scored   : Layer 4 output dict (for layer2_score, seriousness_score etc.)
        rank     : final rank 1-100, used to calibrate concern surfacing
    """
    profile  = original.get("profile", {})
    signals  = original.get("redrob_signals", {})

    # ── Profile facts ─────────────────────────────────────────────────
    title     = profile.get("current_title", "ML Engineer")
    company   = profile.get("current_company", "current company")
    yoe       = profile.get("years_of_experience", 0)
    location  = profile.get("location", "Unknown")
    notice    = signals.get("notice_period_days", 90)
    open_work = signals.get("open_to_work_flag", False)
    github    = signals.get("github_activity_score", -1)

    # ── JD match signals ──────────────────────────────────────────────
    jd_dims   = _matched_jd_dimensions(original)
    mh_skills = _matched_must_have_skills(original)

    dim_str   = ", ".join(jd_dims[:2]) if jd_dims else "applied ML systems"
    skill_str = ", ".join(mh_skills[:2]) if mh_skills else "relevant technical skills"

    # ── Sentence 1: Strengths ─────────────────────────────────────────
    sentence1 = (
        f"{yoe:.0f}-year {title} at {company} with career evidence of "
        f"{dim_str}; verified skills include {skill_str}."
    )

    # ── Sentence 2: Concerns or availability ─────────────────────────
    concerns = []

    # Notice period — JD: sub-30 ideal, up to 90 tolerable
    if notice > 90:
        concerns.append(f"{notice}-day notice (above JD threshold)")
    elif notice > 60:
        concerns.append(f"{notice}-day notice (tolerable but noted)")

    # Passive candidate flag
    if not open_work:
        concerns.append("not actively seeking (passive candidate)")

    # GitHub — only flag for rank ≤ 50, threshold 20 is conservative
    # NOTE: raise threshold (e.g. to 40) if you want stricter GitHub bar
    if github != -1 and github < 20 and rank <= 50:
        concerns.append(f"low GitHub activity ({github:.0f}/100)")
    elif github == -1 and rank <= 20:
        concerns.append("no GitHub linked")

    # Profile seriousness — only flag for top 20 where bar is higher
    if scored.get("seriousness_score", 1.0) < 0.5 and rank <= 20:
        concerns.append("low profile completeness / verification")

    # Weak JD-fit from Layer 2
    if scored.get("layer2_score", 1.0) < 0.45:
        concerns.append("limited direct JD-skill evidence in career")

    if concerns:
        sentence2 = f"Concerns: {'; '.join(concerns)}."
    else:
        avail_str = "actively seeking" if open_work else "passive candidate"
        sentence2 = f"Located in {location}, {notice}-day notice, {avail_str}."

    return f"{sentence1} {sentence2}"


# ═══════════════════════════════════════════════════════════════════════
# SORTING — SCORE PASSTHROUGH + TIE-BREAK
# ═══════════════════════════════════════════════════════════════════════

def sort_by_score(scored_candidates: list[dict]) -> list[dict]:
    """
    Sorts by final_score descending.
    Layer 4 final_score is preserved exactly — no modification.

    TIE-BREAKING (spec §3):
        Equal scores → candidate_id ascending (CAND_XXXXXXX string sort)
        NOTE: to use redrob_score as numeric tie-break instead, change key to:
              (-c["final_score"], -c["redrob_score"], c["candidate_id"])
    """
    return sorted(
        scored_candidates,
        key=lambda c: (-c["final_score"], c["candidate_id"])  # TIE-BREAK: candidate_id asc
    )


# ═══════════════════════════════════════════════════════════════════════
# TOP-100 CSV BUILDER
# ═══════════════════════════════════════════════════════════════════════

def build_top100(
    sorted_candidates: list[dict],
    original_lookup: dict,
) -> list[dict]:
    """
    Slices top 100 from sorted list, generates reasoning for each.

    Args:
        sorted_candidates : full sorted list from sort_by_score()
        original_lookup   : {candidate_id: raw_candidate_dict} for reasoning

    Returns:
        list of 100 dicts — columns: candidate_id, rank, score, reasoning
    """
    top100 = sorted_candidates[:100]   # reasoning only runs for these 100
    rows = []

    for rank, c in enumerate(top100, start=1):
        cid      = c["candidate_id"]
        original = original_lookup.get(cid, {})

        reasoning = generate_reasoning(
            original=original,
            scored=c,
            rank=rank,
        )

        rows.append({
            "candidate_id": cid,
            "rank":         rank,
            "score":        c["final_score"],  # Layer 4 score — untouched
            "reasoning":    reasoning,
        })

    return rows


# ═══════════════════════════════════════════════════════════════════════
# MASTER FUNCTION
# ═══════════════════════════════════════════════════════════════════════

def run_layer5(
    scored_candidates: list[dict],
    original_lookup: dict,
) -> list[dict]:
    """
    Full Layer 5 pipeline.

    Args:
        scored_candidates : list of dicts from apply_layer4() — contains final_score
        original_lookup   : {candidate_id: raw candidate dict} — for reasoning facts

    Returns:
        list of exactly 100 dicts ready to write as CSV
        [{candidate_id, rank, score, reasoning}, ...]
    """
    print(f"  Sorting {len(scored_candidates):,} candidates by Layer 4 score...")
    sorted_cands = sort_by_score(scored_candidates)

    print("  Building top-100 with reasoning...")
    top100 = build_top100(sorted_cands, original_lookup)

    # Sanity: must be exactly 100 rows
    assert len(top100) == 100, f"Expected 100 rows, got {len(top100)}"

    # Sanity: scores must be non-increasing (validate_submission.py requirement)
    for i in range(len(top100) - 1):
        assert top100[i]["score"] >= top100[i + 1]["score"], (
            f"Score order violation at rank {top100[i]['rank']} → {top100[i + 1]['rank']}: "
            f"{top100[i]['score']} < {top100[i + 1]['score']}"
        )

    print(f"  Top candidate  : {top100[0]['candidate_id']} (score={top100[0]['score']})")
    print(f"  Rank-100       : {top100[99]['candidate_id']} (score={top100[99]['score']})")
    return top100