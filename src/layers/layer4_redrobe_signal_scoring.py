"""
layer4_scoring.py

LAYER 4 — REDROB PLATFORM SIGNALS

Purpose:
    Score candidate quality using Redrob behavioural signals only.

Inputs:
    - layer2_score
    - location_score
    - availability_score

Outputs:
    - redrob_score
    - final_score

No JD-fit scoring here.
No skills scoring.
No career scoring.

Those already exist in Layer 2.
"""

from datetime import date, datetime


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _months_since(date_str: str) -> int:
    if not date_str:
        return 999

    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        today = date.today()

        return (
            (today.year - d.year) * 12
            + (today.month - d.month)
        )

    except ValueError:
        return 999


# ═══════════════════════════════════════════════════════════════════════
# REDROB SCORE
# ═══════════════════════════════════════════════════════════════════════

REDROB_WEIGHTS = {
    "github":       0.12,
    "interview":    0.25,
    "response":     0.20,
    "offer":        0.05,
    "seriousness":  0.15,
    "saved":        0.18,
    "applications": 0.05,
}


def compute_redrob_score(candidate: dict) -> dict:
    """
    Returns Redrob behavioural score.

    Components:

    - github activity
    - interview completion
    - response speed
    - offer acceptance
    - profile seriousness
    - recruiter saves
    - applications submitted
    """

    s = candidate.get("redrob_signals", {})

    # --------------------------------------------------
    # GitHub activity
    # --------------------------------------------------

    github = s.get("github_activity_score", -1)

    github_score = (
        0.5
        if github == -1
        else github / 100
    )

    # --------------------------------------------------
    # Interview completion
    # --------------------------------------------------

    interview_score = s.get(
        "interview_completion_rate",
        0.5
    )

    # --------------------------------------------------
    # Response speed
    # --------------------------------------------------

    avg_rt = s.get(
        "avg_response_time_hours",
        48
    )

    if avg_rt <= 4:
        response_speed_score = 1.0

    elif avg_rt <= 24:
        response_speed_score = 0.8

    elif avg_rt <= 72:
        response_speed_score = 0.5

    else:
        response_speed_score = 0.2

    # --------------------------------------------------
    # Offer acceptance
    # --------------------------------------------------

    oar = s.get(
        "offer_acceptance_rate",
        -1
    )

    offer_score = (
        0.5
        if oar == -1
        else max(oar, 0)
    )

    # --------------------------------------------------
    # Seriousness
    # --------------------------------------------------

    completeness = (
        s.get("profile_completeness_score", 50)
        / 100
    )

    verified = (
        int(s.get("verified_email", False))
        + int(s.get("verified_phone", False))
        + int(s.get("linkedin_connected", False))
    ) / 3

    endorsements = min(
        s.get("endorsements_received", 0),
        100
    ) / 100

    seriousness_score = (
        0.4 * completeness +
        0.4 * verified +
        0.2 * endorsements
    )

    # --------------------------------------------------
    # Recruiter saves
    # --------------------------------------------------

    saved_score = min(
        s.get("saved_by_recruiters_30d", 0),
        20
    ) / 20

    # --------------------------------------------------
    # Applications
    # --------------------------------------------------

    applications_score = min(
        s.get("applications_submitted_30d", 0),
        20
    ) / 20

    # --------------------------------------------------
    # Final Redrob score
    # --------------------------------------------------

    redrob_score = (
        REDROB_WEIGHTS["github"]       * github_score +
        REDROB_WEIGHTS["interview"]    * interview_score +
        REDROB_WEIGHTS["response"]     * response_speed_score +
        REDROB_WEIGHTS["offer"]        * offer_score +
        REDROB_WEIGHTS["seriousness"]  * seriousness_score +
        REDROB_WEIGHTS["saved"]        * saved_score +
        REDROB_WEIGHTS["applications"] * applications_score
    )

    return {
        "github_score": round(github_score, 4),
        "interview_score": round(interview_score, 4),
        "response_speed_score": round(response_speed_score, 4),
        "offer_score": round(offer_score, 4),
        "seriousness_score": round(seriousness_score, 4),
        "saved_score": round(saved_score, 4),
        "applications_score": round(applications_score, 4),
        "redrob_score": round(redrob_score, 4),
    }


# ═══════════════════════════════════════════════════════════════════════
# MASTER FUNCTION
# ═══════════════════════════════════════════════════════════════════════

def apply_layer4(
    candidate: dict,
    location_score: float,
    availability_score: float,
    layer2_score: float,
) -> dict:
    """
    Layer 4 final score.

    Layer 2 = JD fit
    Layer 3 = location + availability
    Layer 4 = Redrob behavioural signals
    """

    redrob = compute_redrob_score(candidate)

    final_score = (
        0.65 * layer2_score +
        0.10 * location_score +
        0.10 * availability_score +
        0.15 * redrob["redrob_score"]
    )

    return {
        "candidate_id": candidate["candidate_id"],

        "layer2_score": round(layer2_score, 4),
        "location_score": round(location_score, 4),
        "availability_score": round(availability_score, 4),

        **redrob,

        "final_score": round(final_score, 4),
    }