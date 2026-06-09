"""
rank.py — Main entry point
Usage: python rank.py --candidates ./data/candidates.jsonl.gz --out ./submission.csv
Runs end-to-end in <5 min, CPU only, no network.
"""

import argparse
import csv
import gzip
import json
import sys
from tqdm import tqdm

from src.layers.layer1_honeypot      import apply_honeypot_check
from src.layers.layer1_hard_filters  import apply_layer1
from src.layers.layer2_soft_filters  import apply_layer2
from src.layers.layer3_location_availability import apply_layer3
from src.layers.layer4_redrobe_signal_scoring       import apply_layer4
from src.layers.layer5_reranker_and_reasoning      import run_layer5
from src.utils.sbert_similarity       import load_model as load_sbert, get_jd_embedding, compute_semantic_scores


def load_candidates(path: str) -> list[dict]:
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_pipeline(candidates_path: str, out_path: str):
    print(f"Loading candidates from {candidates_path}...")
    candidates = load_candidates(candidates_path)
    print(f"Loaded {len(candidates):,} candidates")

    original_lookup = {c["candidate_id"]: c for c in candidates}

    # ── Layer 1 + Honeypot ────────────────────────────────────────────
    print("\nLayer 1 — Hard filters + honeypot detection...")
    l1_passed = []
    l1_stats  = {"honeypot": 0, "eliminated": 0}

    for c in tqdm(candidates):
        is_hp, _ = apply_honeypot_check(c)
        if is_hp:
            l1_stats["honeypot"] += 1
            continue
        disq, _ = apply_layer1(c)
        if disq:
            l1_stats["eliminated"] += 1
            continue
        l1_passed.append(c)

    print(f"  Honeypots : {l1_stats['honeypot']}")
    print(f"  Eliminated: {l1_stats['eliminated']}")
    print(f"  Passed    : {len(l1_passed):,}")

    # ── Layer 2 ───────────────────────────────────────────────────────
    print("\nLayer 2 — Soft filters...")
    l2_passed = []
    for c in tqdm(l1_passed):
        disq, _ = apply_layer2(c)
        if not disq:
            l2_passed.append(c)
    print(f"  Passed: {len(l2_passed):,}")

    # ── SBERT — load once, score after Layer 2 ───────────────────────
    print("\nLoading SBERT model...")
    sbert_model = load_sbert()
    jd_embedding = get_jd_embedding(sbert_model)
    print("Computing semantic similarity scores...")
    semantic_scores = compute_semantic_scores(l2_passed, sbert_model, jd_embedding)
    print(f"  Scored {len(semantic_scores)} candidates semantically")

    # ── Layer 3 ───────────────────────────────────────────────────────
    print("\nLayer 3 — Location + availability scoring...")
    l3_results = []
    for c in tqdm(l2_passed):
        l3 = apply_layer3(c)
        l3_results.append((c, l3))
    print(f"  Scored: {len(l3_results):,}")

    # ── Layer 4 ───────────────────────────────────────────────────────
    print("\nLayer 4 — Feature scoring + behavioral multiplier...")
    scored = []
    for c, l3 in tqdm(l3_results):
        sem_score = semantic_scores.get(c["candidate_id"], 0.0)
        l4 = apply_layer4(c, l3["location_score"], l3["availability_score"], sem_score)
        scored.append(l4)
    print(f"  Scored: {len(scored):,}")

    # ── Layer 5 ───────────────────────────────────────────────────────
    print("\nLayer 5 — LightGBM re-rank + top-100 selection...")
    top100 = run_layer5(scored, original_lookup, retrain=True)

    # ── Write CSV ─────────────────────────────────────────────────────
    print(f"\nWriting submission to {out_path}...")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(top100)

    print(f"\nDone. Top candidate: {top100[0]['candidate_id']} (score={top100[0]['score']})")
    print("Run validate_submission.py to verify format before submitting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="data/candidates.jsonl.gz")
    parser.add_argument("--out",        default="submission.csv")
    args = parser.parse_args()
    run_pipeline(args.candidates, args.out)