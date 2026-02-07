from typing import List, Dict


def confidence_from_hits(hits: List[Dict], default_no_evidence: float = 0.40) -> float:
    """
    Evidence-based confidence:
    - No evidence => default low confidence
    - More evidence + higher top score => higher confidence
    Assumes 'hits' are sorted by score desc (top hit first).
    """
    if not hits:
        return default_no_evidence

    top = float(hits[0].get("score", 0.0))
    count = min(len(hits), 5)

    # Base 0.50, boosted by similarity score and evidence count
    conf = 0.50 + 0.40 * (top - 0.50) + 0.02 * count

    # Clamp to [0.0, 0.95]
    return max(0.0, min(0.95, conf))
