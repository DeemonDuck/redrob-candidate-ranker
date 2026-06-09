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
