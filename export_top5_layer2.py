import json
from pathlib import Path

from src.layers.layer2_soft_filters import apply_layer2

OUTPUT_DIR = Path("top5_layer2")
OUTPUT_DIR.mkdir(exist_ok=True)

scored = []

print("Scoring candidates through Layer 2...")

with open("data/candidates.jsonl", "r", encoding="utf-8") as f:

    for line in f:

        candidate = json.loads(line)

        eliminated, reason, score = apply_layer2(candidate)

        if eliminated:
            continue

        scored.append((score, candidate))

print(f"Candidates surviving Layer 2: {len(scored):,}")

# Highest scores first
scored.sort(key=lambda x: x[0], reverse=True)

top5 = scored[:5]

for rank, (score, candidate) in enumerate(top5, start=1):

    cid = candidate["candidate_id"]

    with open(
        OUTPUT_DIR / f"rank_{rank:03d}_{cid}.txt",
        "w",
        encoding="utf-8"
    ) as out:

        out.write("=" * 100 + "\n")
        out.write(f"RANK: {rank}\n")
        out.write(f"CANDIDATE ID: {cid}\n")
        out.write(f"LAYER2 SCORE: {score:.4f}\n")
        out.write("=" * 100 + "\n\n")

        out.write(json.dumps(candidate, indent=2))

print("\nTop 5 Layer 2 profiles exported.")