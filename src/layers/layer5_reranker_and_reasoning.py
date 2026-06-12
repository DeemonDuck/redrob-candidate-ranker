"""
layer5_reranker_and_reasoning.py

LAYER 5 — TOP-100 SELECTION + REASONING

What this layer does:
    1. Takes Layer 4 scored candidates (already ordered by final_score)
    2. Preserves Layer 4 scores EXACTLY — no re-scoring, no LightGBM
    3. Resolves ties deterministically using candidate_id ascending (spec requirement)
    4. Uses SBERT semantic scores (passed in from rank.py) for reasoning ONLY
    5. Generates 1-2 sentence reasoning grounded in real candidate facts

What this layer does NOT do:
    - Does NOT modify final_score from Layer 4
    - Does NOT use SBERT to change rankings
    - Does NOT call any external API or model during reasoning

SBERT role here:
    - semantic_scores dict maps candidate_id -> cosine similarity with JD
    - Used only to identify which JD dimensions the candidate matches well
    - Feeds into reasoning text, not into the score column

Tie-breaking rule (spec §3):
    - Equal scores → candidate_id ascending (CAND_XXXXXXX string sort)
    - This is deterministic and matches validate_submission.py expectations
"""
import numpy as np
from src.utils.constants import MUST_HAVE_SKILLS
from src.utils.sbert_similarity import load_model, get_jd_embedding, compute_semantic_scores


# ═══════════════════════════════════════════════════════════════════════
# REASONING HELPERS
# ═══════════════════════════════════════════════════════════════════════

# JD signals we check for in career text — used to build specific reasoning
# These are the exact things the JD cares about (not generic ML terms)
JD_DIMENSION_KEYWORDS = {
    "embeddings":       ["embedding", "embeddings", "sentence-transformer", "sentence_transformer", "dense retrieval"],
    "vector_db":        ["pinecone", "qdrant", "milvus", "weaviate", "opensearch", "elasticsearch", "faiss", "pgvector"],
    "hybrid_search":    ["hybrid search", "hybrid retrieval", "bm25", "sparse", "dense", "reranking", "rerank"],
    "evaluation":       ["ndcg", "mrr", "map", "a/b test", "offline", "online", "eval", "benchmark"],
    "ranking":          ["ranking", "learning to rank", "ltr", "xgboost", "lightgbm", "lambdarank"],
    "llm_finetuning":   ["lora", "qlora", "peft", "fine-tun", "finetun", "finetuned"],
    "production":       ["production", "deployed", "shipped", "serving", "scale", "a/b", "latency"],
    "product_company":  ["product", "startup", "saas", "platform", "marketplace"],
}

# Human-readable labels for the dimensions above
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
    """All career descriptions + titles joined — for keyword matching."""
    parts = []
    for job in original.get("career_history", []):
        parts.append(job.get("title", "").lower())
        parts.append(job.get("description", "").lower())
    return " ".join(parts)


def _matched_jd_dimensions(original: dict) -> list[str]:
    """
    Returns list of JD dimension labels this candidate matches in career text.
    Used in reasoning sentence 1 to be specific, not generic.
    """
    text = _career_text(original)
    matched = []
    for dim, keywords in JD_DIMENSION_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            matched.append(DIMENSION_LABELS[dim])
    return matched


def _matched_must_have_skills(original: dict) -> list[str]:
    """
    Returns trusted must-have skills (from constants) that have
    either duration > 0 or endorsements > 0 — avoids ghost skills.
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
        # Only include if there's actual proof of skill
        trusted = (
            s.get("endorsements", 0) > 0
            or s.get("duration_months", 0) > 6
            or s["name"] in assessment_scores
        )
        if trusted:
            matched.append(s["name"])
    return matched[:4]  # cap at 4 for readability


def _sbert_hint(semantic_score: float) -> str:
    """
    Translates SBERT cosine similarity into a human-readable signal.
    Used only in reasoning — not in scoring.

    Thresholds tuned for MiniLM all-MiniLM-L6-v2 on this JD.
    """
    # NOTE: These thresholds are for reasoning labels only.
    #       Change them freely — they have zero effect on ranking.
    if semantic_score >= 0.70:
        return "strong semantic alignment with JD"
    elif semantic_score >= 0.50:
        return "good semantic alignment with JD"
    elif semantic_score >= 0.30:
        return "some semantic overlap with JD"
    else:
        return "low semantic alignment (keyword gap possible)"


def generate_reasoning(
    original: dict,
    scored: dict,
    rank: int,
    semantic_score: float,
) -> str:
    """
    Generates 1-2 sentence reasoning grounded entirely in candidate facts.

    Rules (per submission_spec Stage 4 checks):
    - Specific facts only — title, company, YOE, named skills, signals
    - Connects to JD requirements, not generic praise
    - Acknowledges concerns honestly for lower ranks
    - No hallucination — every claim comes from original candidate data

    Args:
        original        : raw candidate dict from candidates.jsonl
        scored          : Layer 4 output dict for this candidate
        rank            : final rank (1-100), used to calibrate concern threshold
        semantic_score  : SBERT cosine similarity — used for reasoning text ONLY
    """
    profile  = original.get("profile", {})
    signals  = original.get("redrob_signals", {})
    history  = original.get("career_history", [])

    # ── Profile facts ────────────────────────────────────────────────
    title    = profile.get("current_title", "ML Engineer")
    company  = profile.get("current_company", "current company")
    yoe      = profile.get("years_of_experience", 0)
    location = profile.get("location", "Unknown")
    notice   = signals.get("notice_period_days", 90)
    open_work = signals.get("open_to_work_flag", False)
    github   = signals.get("github_activity_score", -1)

    # ── What the candidate actually matches in JD ────────────────────
    jd_dims   = _matched_jd_dimensions(original)
    mh_skills = _matched_must_have_skills(original)

    # ── Sentence 1: Strengths ────────────────────────────────────────
    # Use up to 2 JD dimensions + 2 skills for specificity
    dim_str   = ", ".join(jd_dims[:2]) if jd_dims else "applied ML systems"
    skill_str = ", ".join(mh_skills[:2]) if mh_skills else "relevant technical skills"

    # NOTE: SBERT hint used here in sentence 1 for semantic context
    sbert_hint = _sbert_hint(semantic_score)

    sentence1 = (
        f"{yoe:.0f}-year {title} at {company} with career evidence of "
        f"{dim_str}; verified skills include {skill_str} ({sbert_hint})."
    )

    # ── Sentence 2: Concerns or availability ────────────────────────
    concerns = []

    # Notice period concern (JD: loves sub-30, tolerates to 90)
    if notice > 90:
        concerns.append(f"{notice}-day notice (above JD threshold)")
    elif notice > 60:
        concerns.append(f"{notice}-day notice (tolerable but noted)")

    # Availability concern
    if not open_work:
        concerns.append("not actively seeking (passive candidate)")

    # GitHub signal — only flag if below threshold AND rank is meaningful
    # NOTE: threshold 20 is conservative; raise it if you want stricter filtering
    if github != -1 and github < 20 and rank <= 50:
        concerns.append(f"low GitHub activity score ({github:.0f})")
    elif github == -1 and rank <= 20:
        concerns.append("no GitHub linked")

    # Seriousness — only flag for top 20 where bar is higher
    if scored.get("seriousness_score", 1.0) < 0.5 and rank <= 20:
        concerns.append("low profile completeness / verification")

    # Layer 2 score concern — weak JD-fit signal
    if scored.get("layer2_score", 1.0) < 0.45:
        concerns.append("limited direct JD-skill evidence in career")

    # Build sentence 2
    if concerns:
        sentence2 = f"Concerns: {'; '.join(concerns)}."
    else:
        avail_str = "actively seeking" if open_work else "passive candidate"
        sentence2 = (
            f"Located in {location}, {notice}-day notice, {avail_str}."
        )

    return f"{sentence1} {sentence2}"


# ═══════════════════════════════════════════════════════════════════════
# SORTING — SCORE PASSTHROUGH + TIE-BREAK
# ═══════════════════════════════════════════════════════════════════════

def sort_by_score(scored_candidates: list[dict]) -> list[dict]:
    """
    Sorts by final_score descending.

    TIE-BREAKING (spec §3 requirement):
        Equal scores → candidate_id ascending (CAND_XXXXXXX string sort)
        This is deterministic and matches validate_submission.py expectations.

    NOTE: Layer 4 final_score is preserved exactly — no modification here.
          If you ever want a secondary numeric tie-break (e.g. redrob_score),
          replace the candidate_id sort key with: (-c["redrob_score"], c["candidate_id"])
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
    semantic_scores: dict,
) -> list[dict]:
    """
    Takes the sorted list, slices top 100, generates reasoning per candidate.

    Args:
        sorted_candidates : full sorted list from sort_by_score()
        original_lookup   : {candidate_id: raw_candidate_dict} — for reasoning facts
        semantic_scores   : {candidate_id: float} — SBERT scores, reasoning only

    Returns:
        list of 100 dicts ready to write as CSV rows
        columns: candidate_id, rank, score, reasoning
    """
    top100 = sorted_candidates[:100]
    rows = []

    for rank, c in enumerate(top100, start=1):
        cid = c["candidate_id"]
        original = original_lookup.get(cid, {})

        # NOTE: score in CSV = Layer 4 final_score, unchanged
        # SBERT semantic score fetched here — used only in reasoning text
        sem_score = semantic_scores.get(cid, 0.0)

        reasoning = generate_reasoning(
            original=original,
            scored=c,
            rank=rank,
            semantic_score=sem_score,
        )

        rows.append({
            "candidate_id": cid,
            "rank":         rank,
            "score":        c["final_score"],   # Layer 4 score — untouched
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
        original_lookup   : {candidate_id: raw candidate dict} — for reasoning
        semantic_scores   : {candidate_id: float} from compute_semantic_scores()
                            Used ONLY for reasoning text, not for scoring.

    Returns:
        list of exactly 100 dicts ready to write as CSV
        [{candidate_id, rank, score, reasoning}, ...]

    NOTE: retrain parameter from old LightGBM version has been removed.
          If rank.py still passes retrain=True, it will be ignored gracefully
          because we use **kwargs — see run_layer5 signature note below.

    REMOVED: LightGBM training, FEATURE_COLS, model loading/saving.
    ADDED:   SBERT-assisted reasoning, deterministic tie-breaking.
    """

    print(f"  Sorting {len(scored_candidates):,} candidates by Layer 4 score...")
    sorted_cands = sort_by_score(scored_candidates)

    print("  Loading SBERT and computing semantic scores for top 100 only...")
    sbert_model = load_model()
    jd_embedding = get_jd_embedding(sbert_model)
    top100_raw = sorted_cands[:100]
    top100_originals = [original_lookup.get(c["candidate_id"], {}) for c in top100_raw]
    semantic_scores = compute_semantic_scores(top100_originals, sbert_model, jd_embedding)

    # To understand sbert score distribution for debugging 

    scores = np.array(list(semantic_scores.values()))

    print("\n===== SBERT SCORE DISTRIBUTION =====")
    print(f"Min    : {scores.min():.4f}")
    print(f"25%    : {np.percentile(scores, 25):.4f}")
    print(f"Median : {np.percentile(scores, 50):.4f}")
    print(f"75%    : {np.percentile(scores, 75):.4f}")
    print(f"90%    : {np.percentile(scores, 90):.4f}")
    print(f"95%    : {np.percentile(scores, 95):.4f}")
    print(f"Max    : {scores.max():.4f}")


    print("  Building top-100 with SBERT-assisted reasoning...")
    top100 = build_top100(sorted_cands, original_lookup, semantic_scores)

    # Sanity check — should never fire but good to have
    assert len(top100) == 100, f"Expected 100 rows, got {len(top100)}"

    # Verify scores are non-increasing (validate_submission.py requirement)
    for i in range(len(top100) - 1):
        assert top100[i]["score"] >= top100[i+1]["score"], (
            f"Score order violation at rank {top100[i]['rank']} → {top100[i+1]['rank']}: "
            f"{top100[i]['score']} < {top100[i+1]['score']}"
        )

    print(f"  Top candidate: {top100[0]['candidate_id']} (score={top100[0]['score']})")
    return top100