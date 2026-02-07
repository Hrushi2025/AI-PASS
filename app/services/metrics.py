import json
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = PROJECT_ROOT / "data" / "audit_logs"

def compute_metrics() -> Dict[str, Any]:
    files = sorted(AUDIT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    total = 0
    counts = {"PASS": 0, "NEEDS_INFO": 0, "FAIL": 0}
    confidences = []
    suspicious_pass = 0  # PASS but no evidence

    recent = []

    for f in files:
        try:
            event = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue

        total += 1
        dec = event.get("decision", "NEEDS_INFO")
        counts[dec] = counts.get(dec, 0) + 1

        conf = event.get("confidence")
        if isinstance(conf, (int, float)):
            confidences.append(float(conf))

        ev = event.get("evidence", [])
        if dec == "PASS" and (not ev):
            suspicious_pass += 1

        if len(recent) < 10:
            recent.append({
                "audit_id": event.get("audit_id"),
                "task_type": event.get("task_type"),
                "decision": dec,
                "confidence": conf,
                "top_retrieval_score": event.get("top_retrieval_score"),
                "evidence_count": event.get("evidence_count"),
            })

    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    return {
        "total": total,
        "counts": counts,
        "avg_confidence": round(avg_conf, 4),
        "suspicious_pass": suspicious_pass,
        "recent": recent,
    }
