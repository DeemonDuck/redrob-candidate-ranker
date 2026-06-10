import csv
import json

submission_ids = {}

with open("submission.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        submission_ids[row["candidate_id"]] = row["rank"]

cv_candidates = []

with open("data/candidates.jsonl", "r", encoding="utf-8") as f:

    for line in f:

        candidate = json.loads(line)

        cid = candidate["candidate_id"]

        if cid not in submission_ids:
            continue

        title = candidate["profile"].get("current_title", "").lower()

        if "computer vision" in title:
            cv_candidates.append(
                (
                    int(submission_ids[cid]),
                    cid,
                    candidate["profile"]["current_title"]
                )
            )

cv_candidates.sort()

print("\n=== COMPUTER VISION CANDIDATES IN TOP 100 ===\n")

for rank, cid, title in cv_candidates:
    print(f"Rank {rank:3d} | {cid} | {title}")