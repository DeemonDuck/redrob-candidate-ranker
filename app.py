"""
app.py — Redrob Hackathon Sandbox (HuggingFace Spaces)

What this does:
    - Accepts ≤100 candidates via file upload OR uses pre-loaded sample
    - Runs the full pipeline (Layer 1 → 2 → 3 → 4 → 5) on that sample
    - Shows ranked output as a table
    - Provides a CSV download button

Run locally:
    streamlit run app.py
"""

import io
import json
import csv

import streamlit as st

from src.layers.layer1_honeypot               import apply_honeypot_check
from src.layers.layer1_hard_filters           import apply_layer1
from src.layers.layer2_soft_filters           import apply_layer2
from src.layers.layer3_location_availability  import apply_layer3
from src.layers.layer4_redrobe_signal_scoring import apply_layer4
from src.layers.layer5_reranker_and_reasoning import run_layer5, sort_by_score, generate_reasoning


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(candidates: list[dict]) -> tuple[list[dict], dict]:
    """
    Runs all 5 layers on a list of candidate dicts.
    Returns (top100_rows, stats_dict).
    """
    stats = {}
    original_lookup = {c["candidate_id"]: c for c in candidates}

    # Layer 1
    l1_passed = []
    honeypot_count = 0
    eliminated_count = 0

    for c in candidates:
        is_hp, _ = apply_honeypot_check(c)
        if is_hp:
            honeypot_count += 1
            continue
        disq, _ = apply_layer1(c)
        if disq:
            eliminated_count += 1
            continue
        l1_passed.append(c)

    stats["layer1_in"]     = len(candidates)
    stats["honeypots"]     = honeypot_count
    stats["l1_eliminated"] = eliminated_count
    stats["layer1_passed"] = len(l1_passed)

    # Layer 2
    l2_passed = []
    l2_scores = {}

    for c in l1_passed:
        eliminated, _, layer2_score = apply_layer2(c)
        if not eliminated:
            l2_passed.append(c)
            l2_scores[c["candidate_id"]] = layer2_score

    stats["layer2_passed"] = len(l2_passed)

    # Layer 3
    l3_results = [(c, apply_layer3(c)) for c in l2_passed]
    stats["layer3_scored"] = len(l3_results)

    # Layer 4
    scored = [
        apply_layer4(
            c,
            location_score=l3["location_score"],
            availability_score=l3["availability_score"],
            layer2_score=l2_scores.get(c["candidate_id"], 0.0),
        )
        for c, l3 in l3_results
    ]
    stats["layer4_scored"] = len(scored)

    # Layer 5
    if not scored:
        return [], stats

    top_n = run_layer5_sample(scored, original_lookup)
    stats["final_ranked"] = len(top_n)

    return top_n, stats


def run_layer5_sample(scored: list[dict], original_lookup: dict) -> list[dict]:
    """
    Layer 5 wrapper that handles samples smaller than 100 candidates.
    For the real 100K run, run_layer5() always produces exactly 100 rows.
    """
    sorted_cands = sort_by_score(scored)
    effective_top = sorted_cands[:100] if len(sorted_cands) >= 100 else sorted_cands

    return [
        {
            "candidate_id": c["candidate_id"],
            "rank":         rank,
            "score":        c["final_score"],
            "reasoning":    generate_reasoning(
                original=original_lookup.get(c["candidate_id"], {}),
                scored=c,
                rank=rank,
            ),
        }
        for rank, c in enumerate(effective_top, start=1)
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def load_sample_candidates() -> list[dict]:
    try:
        with open("sample_candidates.json") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    except FileNotFoundError:
        return []


def parse_upload(uploaded_file) -> tuple[list[dict], str | None]:
    try:
        raw = uploaded_file.read().decode("utf-8")
    except Exception as e:
        return [], f"Could not read file: {e}"

    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    try:
        parsed = [json.loads(l) for l in lines]
        candidates = parsed if all(isinstance(p, dict) for p in parsed) else json.loads(raw)
    except json.JSONDecodeError:
        try:
            candidates = json.loads(raw)
        except json.JSONDecodeError as e:
            return [], f"JSON parse error: {e}"

    if isinstance(candidates, dict):
        candidates = [candidates]

    if not isinstance(candidates, list):
        return [], "File must contain a JSON array or JSONL of candidate objects."

    if len(candidates) > 100:
        return [], f"Sandbox accepts ≤100 candidates. Your file has {len(candidates)}."

    return candidates, None


def rows_to_csv_string(rows: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["candidate_id", "rank", "score", "reasoning"])
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Redrob Candidate Ranker",
        page_icon="🏆",
        layout="wide",
    )

    st.title("🏆 Redrob Candidate Ranker")
    st.caption(
        "Hackathon sandbox — ranks candidates against the Senior AI Engineer JD. "
        "Accepts ≤100 candidates. Full 100K run happens via `rank.py` locally."
    )
    st.divider()

    with st.sidebar:
        st.header("Pipeline")
        st.markdown("""
        **Layer 1** — Hard filters + honeypot detection  
        **Layer 2** — JD-fit weighted scoring  
        **Layer 3** — Location + availability scoring  
        **Layer 4** — Redrob behavioural signals  
        **Layer 5** — Sort + top-100 + reasoning  
        """)
        st.divider()
        st.markdown("""
        **Score formula (Layer 4)**
        ```
        final = 0.65 × jd_fit
              + 0.10 × location
              + 0.10 × availability
              + 0.15 × redrob_signals
        ```
        """)
        st.divider()
        st.caption("No GPU · No network during ranking · CPU only")

    # Input
    st.subheader("1. Load Candidates")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**Option A — Upload your own file**")
        uploaded = st.file_uploader(
            "Upload JSON or JSONL (≤100 candidates)",
            type=["json", "jsonl"],
            help="Must match the candidate schema from the hackathon bundle.",
        )

    with col2:
        st.markdown("**Option B — Use pre-loaded sample**")
        use_sample = st.button("Load sample_candidates.json (50 candidates)", type="secondary")
        st.caption("Pre-bundled 50-candidate sample from the hackathon bundle.")

    candidates = None

    if uploaded is not None:
        candidates, load_error = parse_upload(uploaded)
        if load_error:
            st.error(f"Upload error: {load_error}")
            candidates = None
        else:
            st.success(f"Loaded {len(candidates)} candidates from upload.")
    elif use_sample:
        candidates = load_sample_candidates()
        if not candidates:
            st.error("sample_candidates.json not found. Make sure it's in the repo root.")
        else:
            st.success(f"Loaded {len(candidates)} candidates from sample file.")

    # Run
    st.divider()
    st.subheader("2. Run Pipeline")

    if candidates is None:
        st.info("Load candidates above to enable the run button.")
        st.stop()

    with st.expander(f"Preview loaded candidates ({len(candidates)} total)", expanded=False):
        preview = [
            {
                "candidate_id": c.get("candidate_id"),
                "name":         c.get("profile", {}).get("anonymized_name"),
                "title":        c.get("profile", {}).get("current_title"),
                "yoe":          c.get("profile", {}).get("years_of_experience"),
                "location":     c.get("profile", {}).get("location"),
            }
            for c in candidates[:10]
        ]
        st.dataframe(preview, use_container_width=True)
        if len(candidates) > 10:
            st.caption(f"Showing first 10 of {len(candidates)} candidates.")

    if not st.button("▶ Run Ranking Pipeline", type="primary"):
        st.stop()

    # Results
    st.divider()
    st.subheader("3. Results")

    with st.spinner("Running pipeline..."):
        try:
            top_rows, stats = run_pipeline(candidates)
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.exception(e)
            st.stop()

    st.markdown("**Pipeline stats**")
    stat_cols = st.columns(5)
    stat_cols[0].metric("Input",        stats.get("layer1_in", 0))
    stat_cols[1].metric("After L1",     stats.get("layer1_passed", 0),
                        delta=f"-{stats.get('honeypots', 0) + stats.get('l1_eliminated', 0)} eliminated")
    stat_cols[2].metric("After L2",     stats.get("layer2_passed", 0))
    stat_cols[3].metric("After L3+L4",  stats.get("layer4_scored", 0))
    stat_cols[4].metric("Final ranked", stats.get("final_ranked", 0))

    st.divider()

    if not top_rows:
        st.warning("No candidates survived the pipeline filters. Try a different sample.")
        st.stop()

    st.markdown(f"**Top {len(top_rows)} candidates**")
    st.dataframe(
        [{"Rank": r["rank"], "Candidate ID": r["candidate_id"],
          "Score": round(r["score"], 4), "Reasoning": r["reasoning"]}
         for r in top_rows],
        use_container_width=True,
        column_config={
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=1, format="%.4f"),
            "Reasoning": st.column_config.TextColumn("Reasoning", width="large"),
        },
    )

    # Download
    st.divider()
    st.subheader("4. Download")
    st.download_button(
        label="⬇ Download submission.csv",
        data=rows_to_csv_string(top_rows),
        file_name="submission.csv",
        mime="text/csv",
        type="primary",
    )
    st.caption(
        "This CSV matches the format required by validate_submission.py. "
        "For your actual submission, run `python rank.py` on the full 100K pool."
    )


if __name__ == "__main__":
    main()
