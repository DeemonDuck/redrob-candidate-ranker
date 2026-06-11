from datetime import date, datetime
  
# ── Location tiers (from JD) ──────────────────────────────────────────────────
 
# Score 1.0 — explicitly preferred
TIER_1_CITIES = {
    "pune", "noida"
}
 
# Score 0.9 — explicitly welcomed
TIER_2_CITIES = {
    "hyderabad", "mumbai", "delhi", "delhi ncr",
    "gurgaon", "gurugram", "bengaluru", "bangalore",
    "new delhi", "faridabad", "ghaziabad"
}
 
# Score 0.8 — India but not preferred cities
TIER_3_COUNTRY = "india"
 
# Score 0.65 — outside India but willing to relocate
# Score 0.4 — outside India, not willing to relocate
 
 
def get_location_score(candidate: dict) -> float:
    """
    Returns 0.0–1.0 based on location fit.
    Uses city first, then country, then relocation flag.
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
 
    location = profile.get("location", "").lower().strip()
    country = profile.get("country", "").lower().strip()
    willing_to_relocate = signals.get("willing_to_relocate", False)
 
    # Check tier 1 cities
    if any(city in location for city in TIER_1_CITIES):
        return 1.0
 
    # Check tier 2 cities
    if any(city in location for city in TIER_2_CITIES):
        return 0.9
 
    # Other Indian cities
    if country == TIER_3_COUNTRY or "india" in location:
        return 0.8
 
    # Outside India
    if willing_to_relocate:
        return 0.65
 
    return 0.4


# ── Availability scoring ──────────────────────────────────────────────────────
 
def _months_since(date_str: str) -> int:
    if not date_str:
        return 999
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        today = date.today()
        return (today.year - d.year) * 12 + (today.month - d.month)
    except ValueError:
        return 999
    
def get_availability_score(candidate: dict) -> float:
    """
    Returns 0.0-1.0 based on how reachable this candidate actually is.
 
    Components:
      - open_to_work_flag        (hard signal)
      - last_active_date         (recency)
      - notice_period_days       (JD: loves sub-30, tolerates up to 90)
      - recruiter_response_rate  (will they reply?)
    
    All components averaged with weights.
    """
    signals = candidate.get("redrob_signals", {})
 
    # 1. Open to work (weight: 0.35)
    open_to_work = signals.get("open_to_work_flag", False)
    open_score = 1.0 if open_to_work else 0.4
    # 0.4 not 0 — passive candidates are still hirable

     # 2. Recency — last active date (weight: 0.30)
    months_inactive = _months_since(signals.get("last_active_date", ""))
    if months_inactive <= 1:
        recency_score = 1.0
    elif months_inactive <= 3:
        recency_score = 0.8
    elif months_inactive <= 6:
        recency_score = 0.6
    elif months_inactive <= 12:
        recency_score = 0.3
    else:
        recency_score = 0.05  # over a year inactive — basically unreachable

    
    # 3. Notice period (weight: 0.20)
    notice_days = signals.get("notice_period_days", 90)
    if notice_days <= 30:
        notice_score = 1.0   # JD: "love sub-30, can buy out up to 30"
    elif notice_days <= 60:
        notice_score = 0.8
    elif notice_days <= 90:
        notice_score = 0.65   # JD: "still in scope but bar gets higher"
    else:
        notice_score = 0.3   # 90+ days is a real logistical problem

    
     # 4. Recruiter response rate (weight: 0.15)
    response_rate = signals.get("recruiter_response_rate", 0.5)
    response_score = response_rate  # already 0.0–1.0
 
    availability_score = (
        0.35 * open_score +
        0.30 * recency_score +
        0.20 * notice_score +
        0.15 * response_score
    )
 
    return round(availability_score, 4)


# ── Master function ───────────────────────────────────────────────────────────

    """
    Returns a dict with location and availability scores.
    No elimination — just scoring for Layer 4.
 
    {
        "location_score": 0.0–1.0,
        "availability_score": 0.0–1.0,
        "location_debug": "Hyderabad → tier 2 city",
        "availability_debug": "open=True, inactive=2mo, notice=30d, response=0.8"
    }
    """

def apply_layer3(candidate: dict) -> dict:

    signals = candidate.get("redrob_signals", {})
    profile = candidate.get("profile", {})
 
    loc_score = get_location_score(candidate)
    avail_score = get_availability_score(candidate)
 
    return {
        "location_score": loc_score,
        "availability_score": avail_score,
        "location_debug": f"{profile.get('location')} | country={profile.get('country')} | relocate={signals.get('willing_to_relocate')}",
        "availability_debug": (
            f"open={signals.get('open_to_work_flag')} | "
            f"inactive={_months_since(signals.get('last_active_date', ''))}mo | "
            f"notice={signals.get('notice_period_days')}d | "
            f"response={signals.get('recruiter_response_rate')}"
        )
    }