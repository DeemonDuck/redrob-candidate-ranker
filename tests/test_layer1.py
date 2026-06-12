# test_layer1.py  ← put this in project root
import gzip, json, sys
from tqdm import tqdm
sys.path.insert(0, '.')
from src.layers.layer1_hard_filters import apply_layer1
from src.layers.layer1_honeypot import apply_honeypot_check

passed, eliminated, honeypots = [], [], []

with open("data/candidates.jsonl", encoding="utf-8") as f:
    for line in tqdm(f, total=100000):
        c = json.loads(line)
        
        # Honeypot check first
        is_hp, hp_reason = apply_honeypot_check(c)
        if is_hp:
            honeypots.append((c['candidate_id'], hp_reason))
            continue
        
        # Then hard filters
        disq, reason = apply_layer1(c)
        if disq:
            eliminated.append((c['candidate_id'], reason))
        else:
            passed.append(c['candidate_id'])

print(f"Passed    : {len(passed)}")
print(f"Eliminated: {len(eliminated)}")
print(f"Honeypots : {len(honeypots)}")

# Breakdown of elimination reasons
from collections import Counter
reasons = Counter(r for _, r in eliminated)
print("\nTop elimination reasons:")
for reason, count in reasons.most_common():
    print(f"  {count:>5}  {reason}")