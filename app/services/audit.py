import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = PROJECT_ROOT / "data" / "audit_logs"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

def write_audit_log(event: Dict[str, Any]) -> str:
    ts = datetime.utcnow().isoformat().replace(":", "-")
    audit_id = event.get("audit_id", ts)

    path = AUDIT_DIR / f"{audit_id}.json"
    path.write_text(json.dumps(event, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)
