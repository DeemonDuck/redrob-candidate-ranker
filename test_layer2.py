# test_layer2.py

import json
import sys
from collections import Counter
from tqdm import tqdm

sys.path.insert(0, '.')

from src.layers.layer1_hard_filters import apply_layer1
from src.layers.layer1_honeypot import apply_honeypot_check
from layers.layer2_soft_filters import apply_layer2

honeypots = []
layer1_eliminated = []
layer2_eliminated = []
survivors = []

with open("data/candidates.jsonl", "r", encoding="utf-8") as f:
    for line in tqdm(f, total=100000):
        candidate = json.loads(line)

        # Honeypot check
        is_hp, hp_reason = apply_honeypot_check(candidate)
        if is_hp:
            honeypots.append((candidate["candidate_id"], hp_reason))
            continue

        # Layer 1
        disq1, reason1 = apply_layer1(candidate)
        if disq1:
            layer1_eliminated.append((candidate["candidate_id"], reason1))
            continue

        # Layer 2
        disq2, reason2 = apply_layer2(candidate)
        if disq2:
            layer2_eliminated.append((candidate["candidate_id"], reason2))
            continue

        survivors.append(candidate["candidate_id"])

print("\n===== PIPELINE SUMMARY =====")
print(f"Honeypots         : {len(honeypots)}")
print(f"Layer 1 Removed   : {len(layer1_eliminated)}")
print(f"Layer 2 Removed   : {len(layer2_eliminated)}")
print(f"Remaining         : {len(survivors)}")

print("\n===== LAYER 2 ELIMINATION REASONS =====")

reason_counts = Counter(reason for _, reason in layer2_eliminated)

for reason, count in reason_counts.most_common():
    print(f"{count:>6}  {reason}")

print("\n===== SAMPLE SURVIVORS =====")

for cid in survivors[:20]:
    print(cid)