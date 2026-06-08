from datetime import date, datetime
  
# ── Location tiers (from JD) ──────────────────────────────────────────────────
 
# Score 1.0 — explicitly preferred
TIER_1_CITIES = {
    "pune", "noida"
}
 
# Score 0.8 — explicitly welcomed
TIER_2_CITIES = {
    "hyderabad", "mumbai", "delhi", "delhi ncr",
    "gurgaon", "gurugram", "bengaluru", "bangalore",
    "new delhi", "faridabad", "ghaziabad"
}
 
# Score 0.5 — India but not preferred cities
TIER_3_COUNTRY = "india"
 
# Score 0.2 — outside India but willing to relocate
# Score 0.05 — outside India, not willing to relocate
 
 
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
        return 0.8
 
    # Other Indian cities
    if country == TIER_3_COUNTRY or "india" in location:
        return 0.5
 
    # Outside India
    if willing_to_relocate:
        return 0.2
 
    return 0.05


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
    
    