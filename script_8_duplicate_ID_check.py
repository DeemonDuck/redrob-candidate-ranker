#!/usr/bin/env python3

import csv
import sys
from collections import Counter

def check_duplicates(csv_file):
    with open(csv_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if "candidate_id" not in reader.fieldnames:
            print("Error: 'candidate_id' column not found.")
            return

        candidate_ids = [row["candidate_id"].strip() for row in reader]

    counts = Counter(candidate_ids)
    duplicates = {cid: count for cid, count in counts.items() if count > 1}

    if not duplicates:
        print("✅ No duplicate candidate_id values found.")
    else:
        print("❌ Duplicate candidate_id values detected:\n")
        for cid, count in sorted(duplicates.items()):
            print(f"{cid}: appears {count} times")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_duplicates.py <submission.csv>")
        sys.exit(1)

    check_duplicates(sys.argv[1])