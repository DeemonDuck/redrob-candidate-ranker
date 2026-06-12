"""
rank.py — Main entry point
Usage: python rank.py --candidates ./data/candidates.jsonl.gz --out ./submission.csv
Runs end-to-end in <5 min, CPU only, no network.

CHANGES FROM PREVIOUS VERSION:
    - SBERT fully removed — no model loading, no semantic scoring
    - Layer 4 receives layer2_score from apply_layer2 (JD-fit weighted score)
    - run_layer5 signature simplified: no semantic_scores argument
    - Layer 2 score captured per-candidate in l2_scores dict, passed into Layer 4
"""

import argparse
import csv
import gzip
import json
from tqdm import tqdm

from src.layers.layer1_honeypot               import apply_honeypot_check
from src.layers.layer1_hard_filters           import apply_layer1
from src.layers.layer2_soft_filters           import apply_layer2
from src.layers.layer3_location_availability  import apply_layer3
from src.layers.layer4_redrobe_signal_scoring import apply_layer4
from src.layers.layer5_reranker_and_reasoning import run_layer5


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

    # Raw candidate data — needed by Layer 5 for reasoning text
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
    # apply_layer2 returns (eliminated, reason, layer2_score)
    # layer2_score is the weighted JD-fit signal fed into Layer 4
    print("\nLayer 2 — Soft filters + JD-fit scoring...")
    l2_passed = []
    l2_scores = {}  # {candidate_id: layer2_score}

    for c in tqdm(l1_passed):
        eliminated, _, layer2_score = apply_layer2(c)
        if not eliminated:
            l2_passed.append(c)
            l2_scores[c["candidate_id"]] = layer2_score

    print(f"  Passed: {len(l2_passed):,}")

    # ── Layer 3: Location + availability scoring ──────────────────────
    print("\nLayer 3 — Location + availability scoring...")
    l3_results = []
    for c in tqdm(l2_passed):
        l3 = apply_layer3(c)
        l3_results.append((c, l3))
    print(f"  Scored: {len(l3_results):,}")

    # ── Layer 4: Redrob signal scoring + final_score ──────────────────
    # final_score = 0.65*layer2_score + 0.10*location + 0.10*availability + 0.15*redrob
    print("\nLayer 4 — Redrob signal scoring...")
    scored = []
    for c, l3 in tqdm(l3_results):
        l4 = apply_layer4(
            c,
            location_score=l3["location_score"],
            availability_score=l3["availability_score"],
            layer2_score=l2_scores.get(c["candidate_id"], 0.0),
        )
        scored.append(l4)
    print(f"  Scored: {len(scored):,}")

    # ── Layer 5: Sort + top-100 + reasoning ──────────────────────────
    # Layer 4 final_score preserved exactly in CSV score column
    print("\nLayer 5 — Top-100 selection + reasoning...")
    top100 = run_layer5(
        scored_candidates=scored,
        original_lookup=original_lookup,
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
    print(f"  Top candidate  : {top100[0]['candidate_id']} (score={top100[0]['score']})")
    print(f"  Rank-100       : {top100[99]['candidate_id']} (score={top100[99]['score']})")
    print("  Run validate_submission.py to verify format before submitting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="data/candidates.jsonl.gz")
    parser.add_argument("--out",        default="submission.csv")
    args = parser.parse_args()
    run_pipeline(args.candidates, args.out)