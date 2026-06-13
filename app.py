"""
app.py — Redrob Hackathon Sandbox (HuggingFace Spaces)

What this does:
    - Accepts ≤100 candidates via file upload OR uses pre-loaded sample
    - Runs the full pipeline (Layer 1 → 2 → 3 → 4 → 5) on that sample
    - Shows ranked output as a table
    - Provides a CSV download button

This is the sandbox required by submission_spec.md §10.5.
It does NOT handle the full 100K pool — small-sample reproducibility only.

Run locally:
    streamlit run app.py
"""

import io
import json
import csv
import streamlit as st

# ── Pipeline imports ─────────
from src.layers.layer1_honeypot               import apply_honeypot_check
from src.layers.layer1_hard_filters           import apply_layer1
from src.layers.layer2_soft_filters           import apply_layer2
from src.layers.layer3_location_availability  import apply_layer3
from src.layers.layer4_redrobe_signal_scoring import apply_layer4
from src.layers.layer5_reranker_and_reasoning import sort_by_score, generate_reasoning


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE (same logic as rank.py — no file I/O, works on a list of dicts)
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(candidates: list[dict]) -> tuple[list[dict], dict]:
    """
    Runs all 5 layers on a list of candidate dicts.
    Returns (top_rows, stats_dict) where top_rows is ready for CSV output.
    """
    stats = {}
    original_lookup = {c["candidate_id"]: c for c in candidates}

    # ── Layer 1: Hard filters + honeypot detection ────────────────────
    l1_passed      = []
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

    # ── Layer 2: JD-fit weighted scoring ─────────────────────────────
    # apply_layer2 returns (eliminated, reason, layer2_score)
    # layer2_score is the JD-fit signal fed into Layer 4
    l2_passed = []
    l2_scores = {}   # {candidate_id: layer2_score}

    for c in l1_passed:
        eliminated, _, layer2_score = apply_layer2(c)
        if not eliminated:
            l2_passed.append(c)
            l2_scores[c["candidate_id"]] = layer2_score

    stats["layer2_passed"] = len(l2_passed)

    # ── Layer 3: Location + availability scoring ──────────────────────
    l3_results = []
    for c in l2_passed:
        l3 = apply_layer3(c)
        l3_results.append((c, l3))

    stats["layer3_scored"] = len(l3_results)

    # ── Layer 4: Redrob signal scoring + final_score ──────────────────
    # final_score = 0.65*layer2_score + 0.10*location + 0.10*availability + 0.15*redrob
    scored = []
    for c, l3 in l3_results:
        l4 = apply_layer4(
            c,
            location_score=l3["location_score"],
            availability_score=l3["availability_score"],
            layer2_score=l2_scores.get(c["candidate_id"], 0.0),
        )
        scored.append(l4)

    stats["layer4_scored"] = len(scored)

    # ── Layer 5: Sort + top-N selection + reasoning ───────────────────
    # Sandbox note: real run always has 100+ survivors from 100K pool.
    # For small samples (≤100 candidates), we rank whatever survived — no crash.
    if len(scored) == 0:
        return [], stats

    top_rows = run_layer5_sample(scored, original_lookup)
    stats["final_ranked"] = len(top_rows)

    return top_rows, stats


def run_layer5_sample(scored: list[dict], original_lookup: dict) -> list[dict]:
    """
    Layer 5 wrapper for sandbox use — handles samples smaller than 100.

    For the real 100K run, run_layer5() in layer5 file enforces exactly 100 rows.
    Here we rank however many survived, so organizers don't hit an assertion error
    when testing with the 50-candidate sample file.
    """
    sorted_cands = sort_by_score(scored)

    # Cap at 100 if we have more; otherwise rank all survivors
    effective_top = sorted_cands[:100] if len(sorted_cands) >= 100 else sorted_cands

    rows = []
    for rank, c in enumerate(effective_top, start=1):
        cid      = c["candidate_id"]
        original = original_lookup.get(cid, {})

        reasoning = generate_reasoning(original=original, scored=c, rank=rank)

        rows.append({
            "candidate_id": cid,
            "rank":         rank,
            "score":        c["final_score"],  # Layer 4 score — untouched
            "reasoning":    reasoning,
        })

    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def load_sample_candidates() -> list[dict]:
    """Load the pre-bundled 50-candidate sample from the repo root."""
    try:
        with open("sample_candidates.json") as f:
            data = json.load(f)
        # Handle both a JSON list and a single JSON object
        return data if isinstance(data, list) else [data]
    except FileNotFoundError:
        return []


def parse_upload(uploaded_file) -> tuple[list[dict], str | None]:
    """
    Parse an uploaded file into a list of candidate dicts.
    Supports:
      - .json  → JSON array of dicts OR a single dict
      - .jsonl → one JSON object per line (standard hackathon format)
    Returns (candidates, error_message). error_message is None on success.
    """
    try:
        raw = uploaded_file.read().decode("utf-8")
    except Exception as e:
        return [], f"Could not read file: {e}"

    candidates = []
    lines = [line.strip() for line in raw.splitlines() if line.strip()]

    # Try parsing as JSONL first (each line is a separate JSON object)
    try:
        parsed = [json.loads(line) for line in lines]
        if all(isinstance(p, dict) for p in parsed):
            candidates = parsed   # valid JSONL
        else:
            candidates = json.loads(raw)   # fallback: JSON array on one line
    except json.JSONDecodeError:
        try:
            candidates = json.loads(raw)
        except json.JSONDecodeError as e:
            return [], f"JSON parse error: {e}"

    # Wrap single object into a list
    if isinstance(candidates, dict):
        candidates = [candidates]

    if not isinstance(candidates, list):
        return [], "File must contain a JSON array or JSONL of candidate objects."

    if len(candidates) > 100:
        return [], f"Sandbox accepts ≤100 candidates. Your file has {len(candidates)}."

    return candidates, None


def rows_to_csv_string(rows: list[dict]) -> str:
    """Convert ranked rows to a CSV string ready for download."""
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=["candidate_id", "rank", "score", "reasoning"]
    )
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

    # ── Header ────────────────────────────────────────────────────────
    st.title("🏆 Redrob Candidate Ranker")
    st.caption(
        "Hackathon sandbox — ranks candidates against the Senior AI Engineer JD. "
        "Accepts ≤100 candidates. Full 100K run happens via `rank.py` locally."
    )
    st.divider()

    # ── Sidebar: pipeline overview ────────────────────────────────────
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

    # ── Session state init ────────────────────────────────────────────
    # WHY: Streamlit reruns the entire script on every button click.
    # Without session_state, candidates loaded via Option B (a button)
    # are lost the moment "Run Pipeline" triggers another rerun.
    # session_state persists values across reruns within the same session.
    if "candidates" not in st.session_state:
        st.session_state.candidates  = None
    if "load_source" not in st.session_state:
        st.session_state.load_source = None

    # ── Step 1: Load candidates ───────────────────────────────────────
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
        use_sample = st.button(
            "Load sample_candidates.json (50 candidates)",
            type="secondary",
        )
        st.caption("Pre-bundled 50-candidate sample from the hackathon bundle.")

    # ── Candidate loading logic ────────────────────────────────────────
    # Priority order: uploaded file > sample button > existing session_state
    # Uploading a new file always overrides whatever was loaded before.

    if uploaded is not None:
        # st.file_uploader keeps the file in memory across reruns by itself,
        # so we parse it fresh every time and update session_state.
        candidates, load_error = parse_upload(uploaded)
        if load_error:
            st.error(f"Upload error: {load_error}")
            st.session_state.candidates  = None
            st.session_state.load_source = None
        else:
            st.session_state.candidates  = candidates
            st.session_state.load_source = "upload"

    elif use_sample:
        # A plain button only fires on the single rerun immediately after the click.
        # We save into session_state here so it survives the NEXT rerun
        # (i.e. when the user clicks "Run Pipeline").
        loaded = load_sample_candidates()
        if not loaded:
            st.error("sample_candidates.json not found. Make sure it's in the repo root.")
            st.session_state.candidates  = None
            st.session_state.load_source = None
        else:
            st.session_state.candidates  = loaded
            st.session_state.load_source = "sample"

    # Always read candidates from session_state — works on any rerun
    candidates = st.session_state.candidates

    if candidates is not None:
        source_label = "upload" if st.session_state.load_source == "upload" else "sample file"
        st.success(f"✅ {len(candidates)} candidates loaded from {source_label}.")

    # ── Step 2: Run pipeline ──────────────────────────────────────────
    st.divider()
    st.subheader("2. Run Pipeline")

    if candidates is None:
        st.info("Load candidates above to enable the run button.")
        st.stop()

    # Quick preview before running — collapsed by default to keep UI clean
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

    run_clicked = st.button("▶ Run Ranking Pipeline", type="primary")

    if not run_clicked:
        st.stop()

    # ── Step 3: Results ───────────────────────────────────────────────
    st.divider()
    st.subheader("3. Results")

    with st.spinner("Running pipeline... (usually <10 seconds for ≤100 candidates)"):
        try:
            top_rows, stats = run_pipeline(candidates)
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.exception(e)
            st.stop()

    # Pipeline stats — shown as metric cards
    st.markdown("**Pipeline stats**")
    stat_cols = st.columns(5)
    stat_cols[0].metric("Input",        stats.get("layer1_in", 0))
    stat_cols[1].metric(
        "After L1",
        stats.get("layer1_passed", 0),
        delta=f"-{stats.get('honeypots', 0) + stats.get('l1_eliminated', 0)} eliminated",
    )
    stat_cols[2].metric("After L2",    stats.get("layer2_passed", 0))
    stat_cols[3].metric("After L3+L4", stats.get("layer4_scored", 0))
    stat_cols[4].metric("Final ranked", stats.get("final_ranked", 0))

    st.divider()

    if not top_rows:
        st.warning("No candidates survived the pipeline filters. Try a different sample.")
        st.stop()

    # Ranked results table
    st.markdown(f"**Top {len(top_rows)} candidates**")

    display_rows = [
        {
            "Rank":         r["rank"],
            "Candidate ID": r["candidate_id"],
            "Score":        round(r["score"], 4),
            "Reasoning":    r["reasoning"],
        }
        for r in top_rows
    ]

    st.dataframe(
        display_rows,
        use_container_width=True,
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Score",
                min_value=0,
                max_value=1,
                format="%.4f",
            ),
            "Reasoning": st.column_config.TextColumn(
                "Reasoning",
                width="large",
            ),
        },
    )

    # ── Step 4: Download ──────────────────────────────────────────────
    st.divider()
    st.subheader("4. Download")

    csv_string = rows_to_csv_string(top_rows)

    st.download_button(
        label="⬇ Download submission.csv",
        data=csv_string,
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