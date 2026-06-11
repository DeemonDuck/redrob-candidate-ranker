import json
from collections import Counter
from src.layers.layer2_soft_filters import apply_layer2

INPUT_FILE = "data/candidates.jsonl"

scores = []
eliminated = 0
reasons = Counter()

top_candidates = []
bottom_candidates = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        candidate = json.loads(line)

        is_eliminated, reason, score = apply_layer2(candidate)

        if is_eliminated:
            eliminated += 1
            reasons[reason] += 1

        scores.append(
            (
                candidate["candidate_id"],
                candidate.get("profile", {}).get("current_title", ""),
                score,
            )
        )

scores.sort(key=lambda x: x[2], reverse=True)

print("\n===== LAYER 2 SUMMARY =====")
print(f"Candidates processed : {len(scores):,}")
print(f"Eliminated           : {eliminated:,}")
print(f"Average score        : {sum(s[2] for s in scores)/len(scores):.4f}")
print(f"Max score            : {scores[0][2]:.4f}")
print(f"Min score            : {scores[-1][2]:.4f}")

print("\n===== TOP 20 SCORES =====")
for cid, title, score in scores[:20]:
    print(f"{cid} | {title} | score={score:.4f}")

print("\n===== BOTTOM 20 SCORES =====")
for cid, title, score in scores[-20:]:
    print(f"{cid} | {title} | score={score:.4f}")

if reasons:
    print("\n===== ELIMINATION REASONS =====")
    for reason, count in reasons.most_common():
        print(f"{count:5d} | {reason}")