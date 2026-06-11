# test_honeypot_v2.py

import json
from collections import Counter
from tqdm import tqdm

from src.layers.layer1_honeypot2 import apply_honeypot_check


DATA_PATH = "data/candidates.jsonl"   # change if needed


total_candidates = 0
honeypots = 0

reason_counter = Counter()

examples = {}

with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in tqdm(f):
        candidate = json.loads(line)

        total_candidates += 1

        is_hp, reason = apply_honeypot_check(candidate)

        if is_hp:
            honeypots += 1

            category = reason.split(":")[0]
            reason_counter[category] += 1

            if category not in examples:
                examples[category] = (
                    candidate["candidate_id"],
                    reason
                )

print("\n" + "=" * 60)
print("HONEYPOT TEST SUMMARY")
print("=" * 60)

print(f"Total candidates: {total_candidates:,}")
print(f"Honeypots found : {honeypots:,}")
print(f"Percentage      : {(honeypots/total_candidates)*100:.2f}%")

print("\nBREAKDOWN")
print("-" * 60)

for category, count in reason_counter.most_common():
    print(f"{category:<35} {count:>6}")

print("\nEXAMPLES")
print("-" * 60)

for category, (cid, reason) in examples.items():
    print(f"\n[{category}]")
    print(f"Candidate: {cid}")
    print(reason)