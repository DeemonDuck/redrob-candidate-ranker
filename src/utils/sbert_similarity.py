"""
sbert_similarity.py

Semantic similarity between JD and candidate profiles using SBERT.
Catches Tier 5 candidates — built real systems but never used buzzwords.

Pre-computation: run download_sbert.py once before submission.
At ranking time: loads from disk, zero network calls.
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

MODEL_PATH = Path("models/all-MiniLM-L6-v2")
JD_EMBEDDING_PATH = Path("models/jd_embedding.npy")

# ── JD text — the query we're matching against ───────────────────────────────
# Condensed to the most signal-rich parts of the JD
JD_TEXT = """
Senior AI Engineer role at Series A AI-native talent intelligence platform.
Production experience with embeddings-based retrieval systems using sentence-transformers,
BGE, E5 or similar. Vector databases: Pinecone, Weaviate, Qdrant, Milvus, FAISS,
Elasticsearch, OpenSearch. Strong Python. Evaluation frameworks for ranking systems:
NDCG, MRR, MAP, offline to online correlation, A/B testing.
Shipped end-to-end ranking, search, or recommendation system to real users at scale.
Strong opinions about hybrid vs dense retrieval, evaluation, LLM integration.
Product company experience preferred over services or consulting.
5 to 9 years experience in applied ML at product companies.
Not pure research. Writes production code. Scrappy product engineering attitude.
Learning to rank models, XGBoost, LightGBM ranker. Fine-tuning with LoRA QLoRA PEFT.
Located in Pune Noida Hyderabad Mumbai Delhi NCR India.
"""


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model() -> SentenceTransformer:
    """Load from local disk — no network call."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"SBERT model not found at {MODEL_PATH}. "
            "Run: python scripts/download_sbert.py"
        )
    return SentenceTransformer(str(MODEL_PATH))


# ── JD embedding ──────────────────────────────────────────────────────────────

def get_jd_embedding(model: SentenceTransformer) -> np.ndarray:
    """Load cached JD embedding or compute and cache it."""
    if JD_EMBEDDING_PATH.exists():
        return np.load(str(JD_EMBEDDING_PATH))
    jd_emb = model.encode(JD_TEXT, normalize_embeddings=True)
    np.save(str(JD_EMBEDDING_PATH), jd_emb)
    return jd_emb


# ── Candidate text builder ────────────────────────────────────────────────────

def build_candidate_text(candidate: dict) -> str:
    """
    Combines profile summary + career descriptions into one text.
    Prioritises recent roles (first 2 jobs = most recent).
    Keeps it short — MiniLM has 256 token limit.
    """
    parts = []

    profile = candidate.get("profile", {})
    summary = profile.get("summary", "")
    if summary:
        parts.append(summary[:300])

    history = candidate.get("career_history", [])
    for job in history[:2]:   # top 2 most recent jobs
        title = job.get("title", "")
        desc = job.get("description", "")
        if title or desc:
            parts.append(f"{title}. {desc[:200]}")

    return " ".join(parts)


# ── Batch scoring ─────────────────────────────────────────────────────────────

def compute_semantic_scores(
    candidates: list[dict],
    model: SentenceTransformer,
    jd_embedding: np.ndarray,
) -> dict[str, float]:
    """
    Batch encode all candidate texts, compute cosine similarity with JD.
    Returns {candidate_id: similarity_score} in [0.0, 1.0].

    ~1200 candidates with MiniLM ≈ 8-12 seconds on CPU.
    """
    cids = [c["candidate_id"] for c in candidates]
    texts = [build_candidate_text(c) for c in candidates]

    # Batch encode — MiniLM handles this efficiently
    embeddings = model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # Cosine similarity — since embeddings are normalised, dot product = cosine
    similarities = embeddings @ jd_embedding

    # Clip to [0, 1] — negative cosine similarity means actively irrelevant
    similarities = np.clip(similarities, 0, 1)

    return {cid: round(float(sim), 6) for cid, sim in zip(cids, similarities)}