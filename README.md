# Redrob Ranker

> A five-layer candidate ranking pipeline that filters 100,000 profiles down to the top 100 for a Senior ML Engineer role — with a one-sentence reasoning per candidate. Runs in under 5 minutes, CPU only, zero network calls.

**[🚀 Live Demo → redrobe-candidate-ranker.streamlit.app](https://redrobe-candidate-ranker.streamlit.app/)**

---

## What this actually does

You give it a `.jsonl.gz` file with 100K candidates. It gives you back a `submission.csv` with the top 100, ranked and explained.

Under the hood, five layers of logic work sequentially — each one passing only the strongest candidates to the next:

```
100,000 candidates
      │
      ▼  Layer 1a — kick out fake profiles (honeypots)
      ▼  Layer 1b — hard disqualifiers from the JD
      ▼  Layer 2  — score JD-fit (skills, career, production evidence)
      ▼  Layer 3  — score location + availability
      ▼  Layer 4  — score platform behaviour (GitHub, interview rate, saves)
      │
      ▼
   Top 100 with reasoning  →  submission.csv
```

No LLM. No external API. Pure Python logic on CPU.

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python rank.py --candidates ./data/candidates.jsonl.gz --out ./submission.csv

# Validate the output
python tests/validate_submission.py submission.csv
```

**Default paths** (if no flags are passed):
- Candidates: `data/candidates.jsonl.gz`
- Output: `submission.csv`

---

## Output Format

`submission.csv` — exactly 100 rows, UTF-8 encoded.

| Column | Type | Description |
|---|---|---|
| `candidate_id` | string | Format: `CAND_XXXXXXX` (7 digits) |
| `rank` | integer | 1 to 100 |
| `score` | float | Final score from Layer 4 (non-increasing by rank) |
| `reasoning` | string | 1–2 sentences grounded in candidate facts |

Tie-breaking rule: equal scores → `candidate_id` ascending (string sort).

---

## Project Structure

```
Redrobe_Ranker/
├── rank.py                         # Main entry point — orchestrates all 5 layers
├── requirements.txt
├── submission.csv                  # Generated output
├── src/
│   ├── layers/
│   │   ├── layer1_honeypot.py
│   │   ├── layer1_hard_filters.py
│   │   ├── layer2_soft_filters.py
│   │   ├── layer3_location_availability.py
│   │   ├── layer4_redrobe_signal_scoring.py
│   │   └── layer5_reranker_and_reasoning.py
│   └── utils/
│       └── constants.py            # Shared keyword sets and thresholds
├── tests/
│   ├── validate_submission.py      # Official submission validator
│   ├── test_layer1.py
│   ├── test_layer2.py
│   ├── test_layer3.py
│   ├── test_layer3_top5.py
│   ├── test_layer4.py
│   ├── test_layer4_top5.py
│   ├── test_honeypot_v2.py
│   ├── script_7_submissioncsv_to_text_preview.py
│   ├── script_8_duplicate_ID_check.py
│   └── inspect_computer_vision_survivors.py
└── data/
    └── candidates.jsonl.gz         # ~100K candidate profiles (not committed)
```
---

## Want to customise the ranking for a different role?

The pipeline is intentionally modular. Here's what to change and where:

| What you want to change | Where to change it |
|---|---|
| Target skills (e.g. swap NLP → Computer Vision) | `src/utils/constants.py` → `MUST_HAVE_SKILLS` |
| Location preferences | `src/layers/layer3_location_availability.py` → `TIER_1_CITIES`, `TIER_2_CITIES` |
| Experience floor (currently 3 years) | `src/utils/constants.py` → `MIN_YEARS_EXPERIENCE` |
| Weight of JD-fit vs platform signals | `src/layers/layer4_redrobe_signal_scoring.py` → final `apply_layer4` formula |
| Which companies count as "pure services" | `src/utils/constants.py` → `PURE_SERVICES_COMPANIES` |
| How much each Layer 2 signal matters | `src/layers/layer2_soft_filters.py` → `SIGNAL_WEIGHTS` |

Example — to target a Computer Vision role instead:
1. Replace `MUST_HAVE_SKILLS` with CV-relevant terms (`opencv`, `yolo`, `torchvision` etc.)
2. Remove `WRONG_DOMAIN_KEYWORDS` check from Layer 1 (currently it eliminates CV profiles)
3. Update `JD_DIMENSION_KEYWORDS` in Layer 5 for accurate reasoning

---

## Pipeline Overview

The pipeline runs five sequential layers. Each layer either eliminates candidates or scores them. Only Layer 5 produces the final output.

```
candidates.jsonl.gz
       │
       ▼
 ┌─────────────┐
 │  Layer 1a   │  Honeypot detection  ──► removed (~80 fake profiles)
 └──────┬──────┘
        ▼
 ┌─────────────┐
 │  Layer 1b   │  Hard filters        ──► eliminated (definitive disqualifiers)
 └──────┬──────┘
        ▼
 ┌─────────────┐
 │   Layer 2   │  Soft filters + JD-fit score (0.0–1.0)
 └──────┬──────┘
        ▼
 ┌─────────────┐
 │   Layer 3   │  Location + availability scores (0.0–1.0 each)
 └──────┬──────┘
        ▼
 ┌─────────────┐
 │   Layer 4   │  Redrob platform signals → final_score
 └──────┬──────┘
        ▼
 ┌─────────────┐
 │   Layer 5   │  Sort → top 100 → reasoning → submission.csv
 └─────────────┘
```

---

## Layer 1 — Honeypot Detection + Hard Filters

Layer 1 is split into two parts run back-to-back. Candidates removed here are never scored.

### Layer 1a — Honeypot Detection (`layer1_honeypot.py`)

The dataset contains ~80 synthetically generated fake profiles ("honeypots"). Having more than 10% honeypots in the top 100 is a disqualification condition in the hackathon rules. Three checks are applied:

| Check | Logic |
|---|---|
| **Impossible tenure** | `duration_months` at a job exceeds the company's possible age (role start year → today) by more than 12 months |
| **Expert skills, zero usage** | 3 or more skills marked `"expert"` proficiency with `0` months of usage and `0` endorsements |
| **Experience vs history mismatch** | Claimed `years_of_experience` is 6+ years more than career history totals (fabricated XP), OR career history totals 2+ years more than claimed (synthetic over-generation) |

A fourth check (duplicate job descriptions via Jaccard similarity of word trigrams) is implemented but currently disabled — it was found to be too aggressive on real candidates.

### Layer 1b — Hard Filters (`layer1_hard_filters.py`)

Three absolute disqualifiers derived directly from the job description:

| Rule | Logic |
|---|---|
| **Experience too low** | `years_of_experience < 3` (JD floor — under 3 is a definitive no) |
| **Pure services career** | Every job in career history is at a named pure-IT-services company (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, HCL) |
| **Wrong domain, no NLP/IR** | >50% of career months are in computer vision / speech / robotics / autonomous driving, AND zero NLP or information retrieval exposure anywhere in career or skills |

Shared keyword constants used by Layer 1 live in `src/utils/constants.py`:
- `PURE_SERVICES_COMPANIES` — named services companies
- `WRONG_DOMAIN_KEYWORDS` — CV/speech/robotics terms
- `NLP_IR_KEYWORDS` — NLP/IR terms that override the wrong-domain flag
- `MUST_HAVE_SKILLS` — embeddings, vector DBs, ranking/retrieval, Python
- `MIN_YEARS_EXPERIENCE = 3`

---

## Layer 2 — Soft Filters + JD-Fit Scoring (`layer2_soft_filters.py`)

Layer 2 converts what were previously hard eliminations into a **weighted score** (0.0–1.0). This means candidates with partial signal survive instead of being dropped — a candidate who has some but not all JD requirements still reaches the top-100 consideration pool.

**One hard elimination remains**: a candidate whose entire career is at large IT services companies with zero product company signal is still eliminated outright.

### Scoring Signals

| Signal | Weight | What it measures |
|---|---|---|
| `alignment` | 0.20 | Career-skills alignment — ghost skills (listed but absent from career history and assessments) reduce this score |
| `trusted_mh` | 0.20 | Trusted must-have skills — must-have skills (embeddings, vector DBs, ranking, Python) that have proof: endorsements > 0, or duration_months > 6, or an assessment score on the platform |
| `mh_career` | 0.15 | Must-have skill evidence in career text — keyword matches of MUST_HAVE_SKILLS in job titles and descriptions |
| `production` | 0.30 | Production deployment evidence — presence of terms like "deployed", "shipped", "serving", "scale", "real-time" in career history (highest weight — the JD explicitly asks for production engineers) |
| `pre_llm` | 0.15 | Pre-LLM background — candidates who only list LangChain/GPT-4 wrappers with no search/retrieval/ML background score low here |

**Formula:**
```
layer2_score = 0.20×alignment + 0.20×trusted_mh + 0.15×mh_career + 0.30×production + 0.15×pre_llm
```

The `layer2_score` is passed directly into Layer 4 as the dominant JD-fit component.

---

## Layer 3 — Location & Availability Scoring (`layer3_location_availability.py`)

Layer 3 produces two independent scores (no eliminations). Both feed into the final score in Layer 4.

### Location Score

Based on the job description's preferred office locations:

| Score | Tier | Cities / Condition |
|---|---|---|
| 1.0 | Tier 1 | Pune, Noida (explicitly preferred in JD) |
| 0.9 | Tier 2 | Hyderabad, Mumbai, Delhi, Delhi NCR, Gurgaon, Bengaluru, Bangalore, Faridabad, Ghaziabad |
| 0.8 | Tier 3 | Anywhere else in India |
| 0.65 | Outside India | `willing_to_relocate = true` |
| 0.4 | Outside India | `willing_to_relocate = false` |

### Availability Score

Weighted composite of four platform signals:

| Component | Weight | Logic |
|---|---|---|
| `open_to_work_flag` | 0.35 | `true` → 1.0; `false` → 0.4 (passive candidates are still hireable) |
| `last_active_date` recency | 0.35 | ≤1 month → 1.0 · ≤3 months → 0.8 · ≤6 months → 0.6 · ≤12 months → 0.3 · >12 months → 0.05 |
| `notice_period_days` | 0.15 | ≤30d → 1.0 · ≤60d → 0.8 · ≤90d → 0.65 · >90d → 0.3 |
| `recruiter_response_rate` | 0.15 | Used directly as a 0.0–1.0 score |

---

## Layer 4 — Redrob Signal Scoring + Final Score (`layer4_redrobe_signal_scoring.py`)

Layer 4 scores each candidate using Redrob platform behavioural signals and combines all previous layer scores into a single `final_score`.

### Redrob Score Components

| Component | Weight | Source field | Logic |
|---|---|---|---|
| GitHub activity | 0.12 | `github_activity_score` | Score/100; -1 (no GitHub) treated as neutral 0.5 |
| Interview completion | 0.25 | `interview_completion_rate` | Used directly (0.0–1.0) |
| Response speed | 0.20 | `avg_response_time_hours` | ≤4h → 1.0 · ≤24h → 0.8 · ≤72h → 0.5 · >72h → 0.2 |
| Offer acceptance | 0.05 | `offer_acceptance_rate` | Used directly; -1 (no history) → neutral 0.5 |
| Profile seriousness | 0.15 | `profile_completeness_score`, `verified_email`, `verified_phone`, `linkedin_connected`, `endorsements_received` | `0.4×completeness + 0.4×verification_ratio + 0.2×endorsements` |
| Saved by recruiters | 0.18 | `saved_by_recruiters_30d` | Capped at 20, then /20 |
| Applications submitted | 0.05 | `applications_submitted_30d` | Capped at 20, then /20 |

```
redrob_score = 0.12×github + 0.25×interview + 0.20×response + 0.05×offer
             + 0.15×seriousness + 0.18×saved + 0.05×applications
```

### Final Score Formula

```
final_score = 0.65 × layer2_score
            + 0.10 × location_score
            + 0.10 × availability_score
            + 0.15 × redrob_score
```

The JD-fit score from Layer 2 is the dominant signal (65%). Platform behaviour (15%) and logistics — location + availability (10% each) — make up the rest.

---

## Layer 5 — Re-ranker & Reasoning (`layer5_reranker_and_reasoning.py`)

Layer 5 takes all Layer 4 scored candidates, selects the top 100, and generates human-readable reasoning for each. It does **not** modify `final_score` in any way.

### Sorting & Tie-breaking

- Sort by `final_score` descending.
- Equal scores: `candidate_id` ascending (string sort — deterministic, matches validator expectations).
- Slice top 100.

### Reasoning Generation

Each row gets a 1–2 sentence reasoning string built entirely from real candidate facts — no LLM, no hallucination.

**Sentence 1 — strengths:** Current title, years of experience, company, up to 2 matched JD dimensions (embeddings, vector DB, hybrid search, evaluation metrics, LTR, LLM fine-tuning, production deployment, product company background), and up to 2 verified must-have skills.

**Sentence 2 — availability or concerns:** If any concern exists (notice > 90 days, passive candidate, low GitHub for top-50, low profile seriousness for top-20, weak Layer 2 score), it is listed. Otherwise, location, notice period, and work status are reported positively.


## Testing & Validation

All test scripts are in `tests/` and read from `data/candidates.jsonl` (uncompressed).

| Script | Purpose |
|---|---|
| `test_layer1.py` | Runs Layer 1a + 1b over the full dataset, prints pass/eliminate/honeypot counts and top elimination reasons |
| `test_honeypot_v2.py` | Honeypot-only breakdown with category counts and example candidates |
| `test_layer2.py` | Scores all Layer 1 survivors with Layer 2, prints top/bottom 20 and score distribution |
| `test_layer3.py` | Runs full pipeline through Layer 3, prints top 20 by combined location+availability |
| `test_layer4.py` | Runs full pipeline through Layer 4, prints top/bottom 20 by final_score |
| `test_layer3_top5.py` | Saves detailed JSON + scores for top 5 Layer 3 candidates to `outputs/layer3/` |
| `test_layer4_top5.py` | Saves detailed JSON + full score breakdown for top 5 Layer 4 candidates to `outputs/layer4/` |
| `validate_submission.py` | **Official validator** — checks header, 100 rows, ID format, non-increasing scores, tie-break order |
| `script_7_submissioncsv_to_text_preview.py` | Readable text preview of submission.csv output |
| `script_8_duplicate_ID_check.py` | Flags any duplicate `candidate_id` values in the CSV |
| `inspect_computer_vision_survivors.py` | Debug script — finds CV-title candidates that survive Layer 2 and saves their score breakdowns |

Run the validator before submitting:
```bash
python tests/validate_submission.py submission.csv
```

---

## Dependencies

```
pandas
tqdm
numpy
sentence-transformers
pyyaml
streamlit
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Hackathon Context

- **Challenge:** Redrob India Runs Data & AI Challenge — Intelligent Candidate Discovery & Ranking
- **Task:** Rank ~100,000 candidates for a Senior ML Engineer (NLP/Search/Retrieval) role and return the top 100 with reasoning
- **Constraints:** Must complete in <5 minutes, CPU only, no network calls during ranking
- **Submission format:** `submission.csv` — 4 columns, 100 rows, UTF-8
