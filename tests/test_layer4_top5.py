import os
import json

from src.layers.layer1_honeypot import apply_honeypot_check
from src.layers.layer1_hard_filters import apply_layer1
from src.layers.layer2_soft_filters import apply_layer2
from src.layers.layer3_location_availability import apply_layer3
from src.layers.layer4_redrobe_signal_scoring import apply_layer4


results = []

total = 0
honeypots = 0
layer1_rejects = 0
layer2_rejects = 0


with open("data/candidates.jsonl", "r", encoding="utf-8") as f:

    for line in f:

        total += 1

        candidate = json.loads(line)

        # --------------------------------------------------
        # Honeypot
        # --------------------------------------------------

        is_honeypot, _ = apply_honeypot_check(candidate)

        if is_honeypot:
            honeypots += 1
            continue

        # --------------------------------------------------
        # Layer 1
        # --------------------------------------------------

        eliminated, _ = apply_layer1(candidate)

        if eliminated:
            layer1_rejects += 1
            continue

        # --------------------------------------------------
        # Layer 2
        # --------------------------------------------------

        eliminated, _, layer2_score = apply_layer2(candidate)

        if eliminated:
            layer2_rejects += 1
            continue

        # --------------------------------------------------
        # Layer 3
        # --------------------------------------------------

        layer3 = apply_layer3(candidate)

        location_score = layer3["location_score"]
        availability_score = layer3["availability_score"]

        # --------------------------------------------------
        # Layer 4
        # --------------------------------------------------

        layer4 = apply_layer4(
            candidate=candidate,
            location_score=location_score,
            availability_score=availability_score,
            layer2_score=layer2_score,
        )

        results.append(
            (
                layer4["final_score"],
                layer4,
                candidate,
            )
        )


results.sort(
    key=lambda x: x[0],
    reverse=True
)

# --------------------------------------------------
# Output folder
# --------------------------------------------------

os.makedirs(
    "outputs/layer4",
    exist_ok=True
)

output_file = "outputs/layer4/top5_layer4.txt"

with open(
    output_file,
    "w",
    encoding="utf-8"
) as out:

    for rank, item in enumerate(results[:5], start=1):

        final_score, layer4, candidate = item

        out.write("=" * 100 + "\n")
        out.write(f"RANK: {rank}\n")
        out.write(f"CANDIDATE ID: {candidate['candidate_id']}\n")
        out.write(f"FINAL SCORE: {final_score:.4f}\n")
        out.write("\n")

        out.write("LAYER 4 BREAKDOWN\n")
        out.write("-" * 50 + "\n")

        out.write(
            f"layer2_score        : {layer4['layer2_score']:.4f}\n"
        )

        out.write(
            f"location_score      : {layer4['location_score']:.4f}\n"
        )

        out.write(
            f"availability_score  : {layer4['availability_score']:.4f}\n"
        )

        out.write(
            f"github_score        : {layer4['github_score']:.4f}\n"
        )

        out.write(
            f"interview_score     : {layer4['interview_score']:.4f}\n"
        )

        out.write(
            f"response_speed      : {layer4['response_speed_score']:.4f}\n"
        )

        out.write(
            f"offer_score         : {layer4['offer_score']:.4f}\n"
        )

        out.write(
            f"seriousness_score   : {layer4['seriousness_score']:.4f}\n"
        )

        out.write(
            f"saved_score         : {layer4['saved_score']:.4f}\n"
        )

        out.write(
            f"applications_score  : {layer4['applications_score']:.4f}\n"
        )

        out.write(
            f"redrob_score        : {layer4['redrob_score']:.4f}\n"
        )

        out.write("\n")
        out.write("=" * 100 + "\n\n")

        out.write(
            json.dumps(
                candidate,
                indent=2,
                ensure_ascii=False
            )
        )

        out.write("\n\n")


print("\nSUMMARY")
print("-" * 50)
print(f"Total candidates : {total:,}")
print(f"Honeypots        : {honeypots:,}")
print(f"Layer1 rejects   : {layer1_rejects:,}")
print(f"Layer2 rejects   : {layer2_rejects:,}")
print(f"Passed Layer 4   : {len(results):,}")

print(f"\nSaved Top 5 candidates to:\n{output_file}")