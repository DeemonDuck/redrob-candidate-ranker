import os
import json

from src.layers.layer1_honeypot import apply_honeypot_check
from src.layers.layer1_hard_filters import apply_layer1
from src.layers.layer2_soft_filters import apply_layer2
from src.layers.layer3_location_availability import apply_layer3


results = []

total = 0
honeypots = 0
layer1_rejects = 0
layer2_rejects = 0


with open("data/candidates.jsonl", "r", encoding="utf-8") as f:

    for line in f:

        total += 1
        candidate = json.loads(line)

        # ==================================================
        # Honeypot
        # ==================================================
        is_honeypot, _ = apply_honeypot_check(candidate)

        if is_honeypot:
            honeypots += 1
            continue

        # ==================================================
        # Layer 1
        # ==================================================
        eliminated, _ = apply_layer1(candidate)

        if eliminated:
            layer1_rejects += 1
            continue

        # ==================================================
        # Layer 2
        # ==================================================
        eliminated, _, layer2_score = apply_layer2(candidate)

        if eliminated:
            layer2_rejects += 1
            continue

        # ==================================================
        # Layer 3
        # ==================================================
        layer3 = apply_layer3(candidate)

        location_score = layer3["location_score"]
        availability_score = layer3["availability_score"]

        # ==================================================
        # Temporary Layer 4 ranking formula
        # ==================================================
        final_score = (
            0.75 * layer2_score +
            0.15 * availability_score +
            0.10 * location_score
        )

        results.append(
            (
                final_score,
                layer2_score,
                location_score,
                availability_score,
                candidate,
            )
        )


# ==========================================================
# Sort best → worst
# ==========================================================
results.sort(reverse=True, key=lambda x: x[0])


# ==========================================================
# Summary
# ==========================================================
print("\n" + "=" * 60)
print("PIPELINE SUMMARY")
print("=" * 60)

print(f"Total candidates : {total:,}")
print(f"Honeypots        : {honeypots:,}")
print(f"Layer1 rejects   : {layer1_rejects:,}")
print(f"Layer2 rejects   : {layer2_rejects:,}")
print(f"Passed Layer 3   : {len(results):,}")


# ==========================================================
# Output folder
# ==========================================================
os.makedirs("outputs/layer3", exist_ok=True)


# ==========================================================
# Export Top 5
# ==========================================================
output_file = "outputs/layer3/top5_layer3.txt"

with open(output_file, "w", encoding="utf-8") as out:

    for rank, item in enumerate(results[:5], start=1):

        (
            final_score,
            layer2_score,
            location_score,
            availability_score,
            candidate,
        ) = item

        out.write("=" * 100 + "\n")
        out.write(f"RANK: {rank}\n")
        out.write(f"CANDIDATE ID: {candidate['candidate_id']}\n")
        out.write(f"FINAL SCORE: {final_score:.4f}\n")
        out.write(f"LAYER2 SCORE: {layer2_score:.4f}\n")
        out.write(f"LOCATION SCORE: {location_score:.4f}\n")
        out.write(f"AVAILABILITY SCORE: {availability_score:.4f}\n")
        out.write("=" * 100 + "\n\n")

        out.write(
            json.dumps(
                candidate,
                indent=2,
                ensure_ascii=False
            )
        )

        out.write("\n\n")


print(f"\nSaved Top 5 candidates to:\n{output_file}")