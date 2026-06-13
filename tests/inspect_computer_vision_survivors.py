import gzip
import json
from pathlib import Path

from src.layers.layer2_soft_filters import (
    apply_layer2,
    score_career_skills_alignment,
    score_trusted_must_haves,
    score_must_have_in_career,
    score_production_evidence,
    score_pre_llm_background,
)

OUTPUT_DIR = Path("cv_survivors")
OUTPUT_DIR.mkdir(exist_ok=True)

count = 0

with gzip.open("data/candidates.jsonl.gz", "rt", encoding="utf-8") as f:

    for line in f:

        candidate = json.loads(line)

        title = candidate.get("profile", {}).get("current_title", "").lower()

        if "computer vision" not in title:
            continue

        eliminated, reason, layer2_score = apply_layer2(candidate)

        if eliminated:
            continue

        count += 1

        cid = candidate["candidate_id"]

        alignment = score_career_skills_alignment(candidate)
        trusted_mh = score_trusted_must_haves(candidate)
        mh_career = score_must_have_in_career(candidate)
        production = score_production_evidence(candidate)
        pre_llm = score_pre_llm_background(candidate)

        with open(
            OUTPUT_DIR / f"{cid}.txt",
            "w",
            encoding="utf-8"
        ) as out:

            out.write(f"CANDIDATE: {cid}\n")
            out.write(f"TITLE: {candidate['profile'].get('current_title')}\n")
            out.write(f"YOE: {candidate['profile'].get('years_of_experience')}\n\n")

            out.write("LAYER 2 BREAKDOWN\n")
            out.write("-" * 50 + "\n")
            out.write(f"alignment  : {alignment:.4f}\n")
            out.write(f"trusted_mh : {trusted_mh:.4f}\n")
            out.write(f"mh_career  : {mh_career:.4f}\n")
            out.write(f"production : {production:.4f}\n")
            out.write(f"pre_llm    : {pre_llm:.4f}\n")
            out.write(f"\nFINAL LAYER2 SCORE : {layer2_score:.4f}\n\n")

            out.write("=" * 80 + "\n")
            out.write(json.dumps(candidate, indent=2))

print(f"\nCreated {count} CV survivor reports.")