from src.utils.constants import MUST_HAVE_SKILLS, PURE_SERVICES_COMPANIES
 
 
# ── helpers ───────────────────────────────────────────────────────────────────
 
def _normalise(text: str) -> str:
    return text.lower().strip()

"""Return normalised text for each job separately."""

def _career_text_per_job(candidate: dict) -> list[str]:
    
    texts = []
    for job in candidate.get("career_history", []):
        t = _normalise(job.get("title", ""))
        d = _normalise(job.get("description", ""))
        texts.append(t + " " + d)
    return texts

