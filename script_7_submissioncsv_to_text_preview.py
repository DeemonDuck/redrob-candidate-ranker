import csv
import json
import os

# ------------------------------------------------------------------
# Files
# ------------------------------------------------------------------

SUBMISSION_FILE = "submission.csv"
CANDIDATES_FILE = "data/candidates.jsonl"

OUTPUT_DIR = "outputs/layer5"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "inspect_candidates.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Which ranks do we want?
# ------------------------------------------------------------------

wanted_ranks = set(range(1, 21))
wanted_ranks.update([25, 50, 100])

# ------------------------------------------------------------------
# Read submission.csv
# ------------------------------------------------------------------

selected = {}  # candidate_id -> (rank, score, reasoning)

with open(SUBMISSION_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        rank = int(row["rank"])

        if rank in wanted_ranks:
            selected[row["candidate_id"]] = {
                "rank": rank,
                "score": row["score"],
                "reasoning": row["reasoning"],
            }

# ------------------------------------------------------------------
# Read candidates.jsonl and dump matching candidates
# ------------------------------------------------------------------

with open(CANDIDATES_FILE, "r", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

    for line in fin:

        candidate = json.loads(line)
        cid = candidate["candidate_id"]

        if cid not in selected:
            continue

        info = selected[cid]

        fout.write("=" * 100 + "\n")
        fout.write(f"RANK         : {info['rank']}\n")
        fout.write(f"CANDIDATE ID : {cid}\n")
        fout.write(f"SCORE        : {info['score']}\n")
        fout.write("\n")
        fout.write("REASONING\n")
        fout.write("-" * 100 + "\n")
        fout.write(info["reasoning"] + "\n\n")

        fout.write("FULL CANDIDATE PROFILE\n")
        fout.write("-" * 100 + "\n")
        fout.write(
            json.dumps(
                candidate,
                indent=2,
                ensure_ascii=False,
            )
        )
        fout.write("\n\n")

print(f"\nSaved inspection file to:\n{OUTPUT_FILE}")