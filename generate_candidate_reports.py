import json
import csv
import textwrap
from pathlib import Path

CANDIDATES_FILE = "data/candidates.jsonl"
SUBMISSION_FILE = "submission.csv"

OUTPUT_DIR = Path("candidate_reports")
OUTPUT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# Load submission rankings
# --------------------------------------------------

submission_rows = []

with open(SUBMISSION_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    submission_rows = list(reader)

candidate_lookup = {
    row["candidate_id"]: row
    for row in submission_rows
}

# --------------------------------------------------
# Targets:
# Top 10
# Top 25
# Rank 50
# Rank 100
# All Computer Vision profiles in Top 100
# --------------------------------------------------

targets = set()

for row in submission_rows[:25]:
    targets.add(row["candidate_id"])

if len(submission_rows) >= 50:
    targets.add(submission_rows[49]["candidate_id"])

if len(submission_rows) >= 100:
    targets.add(submission_rows[99]["candidate_id"])

# --------------------------------------------------
# Load candidate dataset
# --------------------------------------------------

with open(CANDIDATES_FILE, "r", encoding="utf-8") as f:

    for line in f:
        candidate = json.loads(line)

        cid = candidate["candidate_id"]

        if cid not in targets:
            continue

        submission_data = candidate_lookup[cid]

        rank = int(submission_data["rank"])

        profile = candidate["profile"]
        signals = candidate["redrob_signals"]

        report_file = OUTPUT_DIR / f"rank_{rank:03d}_{cid}.txt"

        with open(report_file, "w", encoding="utf-8") as out:

            out.write("=" * 100 + "\n")
            out.write(f"RANK: {rank}\n")
            out.write(f"CANDIDATE ID: {cid}\n")
            out.write(f"SCORE: {submission_data['score']}\n")
            out.write("=" * 100 + "\n\n")

            out.write("GENERATED REASONING\n")
            out.write("-" * 100 + "\n")
            out.write(submission_data["reasoning"] + "\n\n")

            out.write("PROFILE\n")
            out.write("-" * 100 + "\n")
            out.write(f"Title: {profile.get('current_title')}\n")
            out.write(f"Company: {profile.get('current_company')}\n")
            out.write(f"Experience: {profile.get('years_of_experience')} years\n")
            out.write(f"Location: {profile.get('location')}, {profile.get('country')}\n\n")

            out.write("SUMMARY\n")
            out.write("-" * 100 + "\n")
            out.write(
                textwrap.fill(
                    profile.get("summary", ""),
                    width=110
                )
            )
            out.write("\n\n")

            out.write("SKILLS\n")
            out.write("-" * 100 + "\n")

            for skill in candidate.get("skills", []):
                out.write(
                    f"{skill['name']} | "
                    f"{skill.get('proficiency')} | "
                    f"{skill.get('duration_months',0)} months | "
                    f"{skill.get('endorsements',0)} endorsements\n"
                )

            out.write("\nCAREER HISTORY\n")
            out.write("-" * 100 + "\n")

            for idx, job in enumerate(candidate.get("career_history", []), start=1):

                out.write(f"\n[{idx}] {job.get('title')} @ {job.get('company')}\n")
                out.write(
                    f"Industry: {job.get('industry')} | "
                    f"Duration: {job.get('duration_months')} months\n"
                )

                out.write(
                    textwrap.fill(
                        job.get("description", ""),
                        width=110
                    )
                )
                out.write("\n")

            out.write("\nEDUCATION\n")
            out.write("-" * 100 + "\n")

            for edu in candidate.get("education", []):

                out.write(
                    f"{edu.get('degree')} | "
                    f"{edu.get('field_of_study')} | "
                    f"{edu.get('institution')} | "
                    f"{edu.get('tier')}\n"
                )

            out.write("\nREDROB SIGNALS\n")
            out.write("-" * 100 + "\n")

            out.write(f"Open to work: {signals.get('open_to_work_flag')}\n")
            out.write(f"Notice period: {signals.get('notice_period_days')}\n")
            out.write(f"Github score: {signals.get('github_activity_score')}\n")
            out.write(f"Response rate: {signals.get('recruiter_response_rate')}\n")
            out.write(f"Interview completion: {signals.get('interview_completion_rate')}\n")
            out.write(f"Saved by recruiters: {signals.get('saved_by_recruiters_30d')}\n")

print(f"Generated reports in: {OUTPUT_DIR}")