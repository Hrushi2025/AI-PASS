# app/core/config.py

# Minimum similarity score to consider a retrieved chunk as valid evidence
EVIDENCE_MIN_SCORE = 0.55

# Minimum number of valid evidence hits required to count as "evidence=True"
EVIDENCE_MIN_HITS = 1

# Maximum evidence hits to return (stability)
EVIDENCE_MAX_HITS = 5

# Decision confidence threshold for PASS in agent draft
CONFIDENCE_PASS_THRESHOLD = 0.70
