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