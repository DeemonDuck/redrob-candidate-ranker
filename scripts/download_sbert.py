"""
download_sbert.py
Run ONCE before submission to cache model weights to disk.
After this, ranking runs fully offline.

Usage: python scripts/download_sbert.py
"""

from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import sys
sys.path.insert(0, '.')
from src.utils.sbert_similarity import JD_TEXT

MODEL_NAME = "all-MiniLM-L6-v2"
SAVE_PATH  = Path("models/all-MiniLM-L6-v2")
JD_EMB_PATH = Path("models/jd_embedding.npy")

print(f"Downloading {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)

print(f"Saving to {SAVE_PATH}...")
SAVE_PATH.parent.mkdir(exist_ok=True)
model.save(str(SAVE_PATH))

print("Pre-computing JD embedding...")
jd_emb = model.encode(JD_TEXT, normalize_embeddings=True)
np.save(str(JD_EMB_PATH), jd_emb)

print("Done. Model size:", sum(f.stat().st_size for f in SAVE_PATH.rglob('*')) // (1024*1024), "MB")
print("Ready for offline ranking.")