"""
rank.py — Main entry point
Usage: python rank.py --candidates ./data/candidates.jsonl.gz --out ./submission.csv
Runs end-to-end in <5 min, CPU only, no network.

CHANGES FROM PREVIOUS VERSION:
    - Layer 4 now receives layer2_score (from apply_layer2) as its JD-fit input,
      NOT the SBERT semantic score. SBERT is a separate component.
    - semantic_scores dict is now passed into run_layer5() for reasoning only.
    - Removed retrain=True from run_layer5() call (LightGBM removed from Layer 5).
    - Layer 2 now returns (eliminated, reason, layer2_score) — score captured here.
"""

import argparse
import csv
import gzip
import json
import sys
from tqdm import tqdm

from src.layers.layer1_honeypot               import apply_honeypot_check
from src.layers.layer1_hard_filters           import apply_layer1
from src.layers.layer2_soft_filters           import apply_layer2
from src.layers.layer3_location_availability  import apply_layer3
from src.layers.layer4_redrobe_signal_scoring import apply_layer4
from src.layers.layer5_reranker_and_reasoning import run_layer5
from src.utils.sbert_similarity               import (
    load_model as load_sbert,
    get_jd_embedding,
    compute_semantic_scores,
)


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

    # Keep a lookup of raw candidate data — needed by Layer 5 for reasoning
    original_lookup = {c["candidate_id"]: c for c in candidates}

    # ── Layer 1: Honeypot + Hard filters ─────────────────────────────
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

    # ── Layer 2: Soft filters + JD-fit scoring ────────────────────────
    # IMPORTANT: apply_layer2 returns (eliminated, reason, layer2_score)
    # layer2_score is the weighted JD-fit signal — this feeds into Layer 4.
    # It is NOT the SBERT score. SBERT is computed separately below.
    print("\nLayer 2 — Soft filters + JD-fit scoring...")
    l2_passed  = []
    l2_scores  = {}  # {candidate_id: layer2_score} — passed into Layer 4

    for c in tqdm(l1_passed):
        eliminated, _, layer2_score = apply_layer2(c)
        if not eliminated:
            l2_passed.append(c)
            l2_scores[c["candidate_id"]] = layer2_score

    print(f"  Passed: {len(l2_passed):,}")

    # ── SBERT: Semantic similarity — loaded once, scored after Layer 2 ──
    # NOTE: SBERT scores are passed into Layer 5 for REASONING ONLY.
    #       They do NOT affect Layer 4 final_score.
    print("\nLoading SBERT model...")
    sbert_model  = load_sbert()
    jd_embedding = get_jd_embedding(sbert_model)

    print("Computing semantic similarity scores (reasoning use only)...")
    semantic_scores = compute_semantic_scores(l2_passed, sbert_model, jd_embedding)
    print(f"  Scored {len(semantic_scores):,} candidates semantically")

    # ── Layer 3: Location + availability scoring ──────────────────────
    print("\nLayer 3 — Location + availability scoring...")
    l3_results = []
    for c in tqdm(l2_passed):
        l3 = apply_layer3(c)
        l3_results.append((c, l3))
    print(f"  Scored: {len(l3_results):,}")

    # ── Layer 4: Redrob signal scoring + final_score ──────────────────
    # Inputs:
    #   layer2_score    — JD-fit score from Layer 2 (NOT SBERT)
    #   location_score  — from Layer 3
    #   availability_score — from Layer 3
    # Output:
    #   final_score = 0.65*layer2_score + 0.10*location + 0.10*availability + 0.15*redrob
    print("\nLayer 4 — Redrob signal scoring...")
    scored = []
    for c, l3 in tqdm(l3_results):
        cid = c["candidate_id"]
        l2_score = l2_scores.get(cid, 0.0)   # JD-fit score from Layer 2
        l4 = apply_layer4(
            c,
            location_score=l3["location_score"],
            availability_score=l3["availability_score"],
            layer2_score=l2_score,            # Layer 2 JD-fit, not SBERT
        )
        scored.append(l4)
    print(f"  Scored: {len(scored):,}")

    # ── Layer 5: Sort + top-100 + reasoning ──────────────────────────
    # semantic_scores passed in for reasoning text generation only.
    # Layer 4 final_score is preserved exactly in the CSV score column.
    print("\nLayer 5 — Top-100 selection + reasoning...")
    top100 = run_layer5(
        scored_candidates=scored,
        original_lookup=original_lookup,
        semantic_scores=semantic_scores,   # NEW: passed for reasoning only
    )

    # ── Write CSV ─────────────────────────────────────────────────────
    print(f"\nWriting submission to {out_path}...")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["candidate_id", "rank", "score", "reasoning"]
        )
        writer.writeheader()
        writer.writerows(top100)

    print(f"\nDone.")
    print(f"  Top candidate : {top100[0]['candidate_id']} (score={top100[0]['score']})")
    print(f"  Bottom (rank 100): {top100[99]['candidate_id']} (score={top100[99]['score']})")
    print("  Run validate_submission.py to verify format before submitting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="data/candidates.jsonl.gz")
    parser.add_argument("--out",        default="submission.csv")
    args = parser.parse_args()
    run_pipeline(args.candidates, args.out)