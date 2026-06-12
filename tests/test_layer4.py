import json
import sys
from statistics import mean
from tqdm import tqdm

sys.path.insert(0, ".")

from src.layers.layer1_hard_filters import apply_layer1
from src.layers.layer1_honeypot import apply_honeypot_check
from src.layers.layer2_soft_filters import apply_layer2
from src.layers.layer3_location_availability import apply_layer3
from src.layers.layer4_redrobe_signal_scoring import apply_layer4

results = []

with open("data/candidates.jsonl", "r", encoding="utf-8") as f:
    for line in tqdm(f, total=100000):
        candidate = json.loads(line)

        # Honeypot
        is_hp, _ = apply_honeypot_check(candidate)
        if is_hp:
            continue

        # Layer 1
        disq1, _ = apply_layer1(candidate)
        if disq1:
            continue

        # Layer 2
        disq2, _ = apply_layer2(candidate)
        if disq2:
            continue

        # Layer 3
        layer3 = apply_layer3(candidate)

        # Layer 4
        layer4 = apply_layer4(
            candidate,
            location_score=layer3["location_score"],
            availability_score=layer3["availability_score"]
        )

        results.append({
            "candidate_id": candidate["candidate_id"],
            "title": candidate["profile"]["current_title"],
            **layer4
        })

print("\n===== LAYER 4 SUMMARY =====")
print(f"Candidates scored: {len(results)}")

scores = [r["final_score"] for r in results]

print(f"Average score : {mean(scores):.4f}")
print(f"Max score     : {max(scores):.4f}")
print(f"Min score     : {min(scores):.4f}")

# Top 20
top20 = sorted(results, key=lambda x: x["final_score"], reverse=True)[:20]

print("\n===== TOP 20 CANDIDATES =====")

for i, c in enumerate(top20, start=1):
    print(
        f"{i:>2}. "
        f"{c['candidate_id']} | "
        f"{c['title']} | "
        f"score={c['final_score']:.4f}"
    )

# Bottom 20
bottom20 = sorted(results, key=lambda x: x["final_score"])[:20]

print("\n===== BOTTOM 20 CANDIDATES =====")

for i, c in enumerate(bottom20, start=1):
    print(
        f"{i:>2}. "
        f"{c['candidate_id']} | "
        f"{c['title']} | "
        f"score={c['final_score']:.4f}"
    )