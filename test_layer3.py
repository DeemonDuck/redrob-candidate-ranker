# test_layer3.py

import json
import sys
from statistics import mean
from tqdm import tqdm

sys.path.insert(0, ".")

from src.layers.layer1_hard_filters import apply_layer1
from src.layers.layer1_honeypot import apply_honeypot_check
from src.layers.layer2_soft_filters import apply_layer2
from src.layers.layer3_location_availability import apply_layer3

layer3_candidates = []

with open("data/candidates.jsonl", "r", encoding="utf-8") as f:
    for line in tqdm(f, total=100000):
        candidate = json.loads(line)

        is_hp, _ = apply_honeypot_check(candidate)
        if is_hp:
            continue

        disq1, _ = apply_layer1(candidate)
        if disq1:
            continue

        disq2, _ = apply_layer2(candidate)
        if disq2:
            continue

        scores = apply_layer3(candidate)

        layer3_candidates.append({
            "candidate_id": candidate["candidate_id"],
            "title": candidate["profile"]["current_title"],
            "location_score": scores["location_score"],
            "availability_score": scores["availability_score"],
            "combined_score": (
                scores["location_score"]
                + scores["availability_score"]
            ) / 2
        })

print("\n===== LAYER 3 SUMMARY =====")
print(f"Candidates reaching Layer 3: {len(layer3_candidates)}")

if layer3_candidates:
    print(
        f"Average location score: "
        f"{mean(c['location_score'] for c in layer3_candidates):.3f}"
    )

    print(
        f"Average availability score: "
        f"{mean(c['availability_score'] for c in layer3_candidates):.3f}"
    )

print("\n===== TOP 20 LAYER 3 SCORES =====")

top = sorted(
    layer3_candidates,
    key=lambda x: x["combined_score"],
    reverse=True
)[:20]

for c in top:
    print(
        f"{c['candidate_id']} | "
        f"{c['title']} | "
        f"loc={c['location_score']:.2f} | "
        f"avail={c['availability_score']:.2f} | "
        f"combined={c['combined_score']:.2f}"
    )