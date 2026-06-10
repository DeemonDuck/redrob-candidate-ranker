import json
import numpy as np
import lightgbm as lgb
from pathlib import Path


# ── Feature columns fed into LightGBM ────────────────────────────────────────

FEATURE_COLS = [
    "skills_score",
    "career_score",
    "experience_score",
    "education_score",
    "semantic_score",        # SBERT cosine similarity with JD
    "location_score",
    "availability_score",
    "github_score",
    "interview_score",
    "response_speed_score",
    "offer_score",
    "seriousness_score",
    "active_score",
    "behavioral_multiplier",
]

MODEL_PATH = Path("models/lgbm_ranker.txt")


# ── Training ──────────────────────────────────────────────────────────────────

def train(scored_candidates: list[dict], save: bool = True) -> lgb.Booster:
    """
    Train LightGBM LambdaRank on pseudo-labels from Layer 4.

    scored_candidates: list of dicts from apply_layer4()
    Pseudo-label = quantile bucket of final_score (0–9)
    so LightGBM learns relative ordering, not absolute scores.
    """
    scores = np.array([c["final_score"] for c in scored_candidates])
    features = np.array([[c[f] for f in FEATURE_COLS] for c in scored_candidates])

    # Convert continuous scores to relevance buckets 0-9
    # LightGBM ranker needs integer relevance labels
    percentiles = np.percentile(scores, np.linspace(0, 100, 11))
    labels = np.digitize(scores, percentiles[1:-1])  # 0–9

    # Single query group — all candidates compete against each other
    group = [len(scored_candidates)]

    train_data = lgb.Dataset(
        features,
        label=labels,
        group=group,
        feature_name=FEATURE_COLS,
    )

    params = {
        "objective":       "lambdarank",
        "metric":          "ndcg",
        "ndcg_eval_at":    [10, 50],     # optimise exactly what we're scored on
        "learning_rate":   0.05,
        "num_leaves":      31,
        "min_data_in_leaf": 5,
        "num_iterations":  300,
        "verbose":         -1,
    }

    model = lgb.train(params, train_data)

    if save:
        MODEL_PATH.parent.mkdir(exist_ok=True)
        model.save_model(str(MODEL_PATH))
        print(f"Model saved → {MODEL_PATH}")

    return model


def load_model() -> lgb.Booster:
    return lgb.Booster(model_file=str(MODEL_PATH))


# ── Re-ranking ────────────────────────────────────────────────────────────────

def rerank(scored_candidates: list[dict], model: lgb.Booster) -> list[dict]:
    """
    Re-score candidates using LightGBM, return sorted list.
    """
    features = np.array([[c[f] for f in FEATURE_COLS] for c in scored_candidates])
    lgbm_scores = model.predict(features)

    for i, c in enumerate(scored_candidates):
        c["lgbm_score"] = round(float(lgbm_scores[i]), 6)

    return sorted(scored_candidates, key=lambda x: x["lgbm_score"], reverse=True)


# ── Reasoning generation ──────────────────────────────────────────────────────

def generate_reasoning(candidate: dict, scored: dict, rank: int, original_data: dict) -> str:
    """
    1-2 sentence reasoning grounded in specific candidate facts.
    No hallucination — only uses fields that exist in the data.
    """
    profile = original_data.get("profile", {})
    history = original_data.get("career_history", [])
    signals = original_data.get("redrob_signals", {})

    title       = profile.get("current_title", "Unknown")
    company     = profile.get("current_company", "Unknown")
    yoe         = profile.get("years_of_experience", 0)
    location    = profile.get("location", "Unknown")
    notice      = signals.get("notice_period_days", 90)
    open_work   = signals.get("open_to_work_flag", False)
    github      = signals.get("github_activity_score", -1)

    # Top 2 product companies from career
    product_cos = [
        j["company"] for j in history
        if j.get("company_size", "") not in {"10001+", "5001-10000"}
        or j.get("industry", "").lower() not in {"it services", "consulting"}
    ][:2]

    # Top matched skills
    skills = original_data.get("skills", [])
    from src.utils.constants import MUST_HAVE_SKILLS
    matched = [
        s["name"] for s in skills
        if s["name"].lower() in MUST_HAVE_SKILLS
        and (s.get("duration_months", 0) > 0 or s.get("endorsements", 0) > 0)
    ][:3]

    # Build sentence 1 — strengths
    cos_str = " and ".join(product_cos) if product_cos else company
    skills_str = ", ".join(matched) if matched else "relevant AI/ML skills"
    sentence1 = (
        f"{yoe:.0f}-year {title} with production experience at {cos_str}; "
        f"strong signal on {skills_str}."
    )

    # Build sentence 2 — concerns or availability
    concerns = []
    if not open_work:
        concerns.append("not actively looking")
    if notice > 60:
        concerns.append(f"{notice}-day notice period")
    if github == -1 or github < 20:
        concerns.append("low GitHub activity")
    if scored.get("career_score", 0) < 0.4:
        concerns.append("limited product company exposure")

    if concerns and rank > 10:
        sentence2 = f"Notable concerns: {'; '.join(concerns)}."
    elif not concerns:
        sentence2 = f"Located in {location}, {notice}-day notice, {'actively seeking' if open_work else 'passive candidate'}."
    else:
        sentence2 = f"Located in {location}; {'; '.join(concerns)}."

    return f"{sentence1} {sentence2}"


# ── Top-100 CSV builder ───────────────────────────────────────────────────────

def build_top100(
    reranked: list[dict],
    original_lookup: dict[str, dict],
) -> list[dict]:
    """
    Takes reranked list, returns top 100 rows ready for CSV output.
    Scores are normalised to [0, 1] and guaranteed non-increasing.
    """
    top100 = reranked[:100]

    # Normalise lgbm_scores to 0-1 range
    raw_scores = [c["lgbm_score"] for c in top100]
    min_s, max_s = min(raw_scores), max(raw_scores)
    score_range = max_s - min_s if max_s != min_s else 1.0

    rows = []
    for rank, c in enumerate(top100, start=1):
        cid = c["candidate_id"]
        normalised = round((c["lgbm_score"] - min_s) / score_range, 6)
        original = original_lookup.get(cid, {})
        reasoning = generate_reasoning(c, c, rank, original)

        rows.append({
            "candidate_id": cid,
            "rank":         rank,
            "score":        normalised,
            "reasoning":    reasoning,
        })

    return rows


# ── Master pipeline ───────────────────────────────────────────────────────────

def run_layer5(
    scored_candidates: list[dict],
    original_lookup: dict[str, dict],
    retrain: bool = True,
) -> list[dict]:
    """
    Full Layer 5 pipeline:
      1. Train (or load) LightGBM ranker
      2. Re-rank scored candidates
      3. Build top-100 with reasoning

    Args:
        scored_candidates : list of dicts from apply_layer4()
        original_lookup   : {candidate_id: raw_candidate_dict} for reasoning
        retrain           : if False, loads saved model instead of retraining

    Returns:
        list of 100 dicts ready to write as CSV
    """
    if retrain or not MODEL_PATH.exists():
        print(f"Training LightGBM on {len(scored_candidates)} candidates...")
        model = train(scored_candidates)
    else:
        print("Loading saved model...")
        model = load_model()

    print("Re-ranking...")
    reranked = rerank(scored_candidates, model)

    print("Building top-100 with reasoning...")
    top100 = build_top100(reranked, original_lookup)

    return top100