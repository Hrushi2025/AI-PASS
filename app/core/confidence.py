from typing import List, Dict

def confidence_from_hits(hits: List[Dict], default_no_evidence: float = 0.40) -> float:
    """
    Convert retrieval hit scores into a confidence value for decision-making.
    Assumes hits contain 'score' floats where higher is better.
    """
    if not hits:
        return default_no_evidence

    top = float(hits[0].get("score", 0.0))

    # Threshold buckets (tune later)
    if top < 0.25:
        return 0.45
    if top < 0.40:
        return 0.65
    return 0.85
