import json
from fastapi import Depends
from sqlmodel import Session, select

from app.db import init_db, get_session
from app.models import OfferRow, EventRow, EvaluationRow, ChatSessionRow

import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.services.rag import RAGService
from app.services.agent import AgentEngine
from app.services.metrics import compute_metrics

from app.core.schemas import (
    TaskInput,
    DecisionOutput,
    SourcingEvent,
    Offer,
    OfferParseResult,
)

app = FastAPI(title="AI-Pass Core MVP")

@app.on_event("startup")
def _startup():
    init_db()
@app.get("/health")
def health():
    return {
        "api": "ok",
        "rag_ready": bool(getattr(rag, "enabled", False)),
        "rag_error": getattr(rag, "init_error", None),
    }


# --------------------------
# Safe startup: RAGService itself is now safe (no-crash if Qdrant down)
# --------------------------
rag = RAGService()
agent = AgentEngine()

# ==========================
# In-memory stores (MVP only)
# ==========================
EVENTS: Dict[str, SourcingEvent] = {}
OFFERS: Dict[str, Offer] = {}

UPLOAD_DIR = Path(__file__).resolve().parents[1] / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ==========================
# Phase C — In-memory state (MVP only)
# ==========================
EVALUATIONS: Dict[str, dict] = {}   # key: event_id -> evaluation payload
CHAT_SESSIONS: Dict[str, dict] = {} # key: session_id -> {event_id, created_at}


class RetrieveRequest(BaseModel):
    query: str
    doc_ids: list[str] = []
    top_k: int = 5


# ==========================
# Phase C — Request Schemas
# ==========================
class ChatCreateRequest(BaseModel):
    event_id: str


class ChatAskRequest(BaseModel):
    session_id: str
    question: str
    policy_doc_ids: list[str] = []


# ==========================
# Phase B2 — Parsing Helpers
# ==========================
CRITICAL_FIELDS = [
    "unit_price",
    "total_price",
    "currency",
    "incoterms",
    "lead_time_days",
    "payment_terms",
]


def parse_offer_pdf_to_fields(pdf_path: str, doc_id: str):
    """
    Lightweight MVP parser:
    - Extract key fields using regex from PDF text
    - Attach confidence and provenance (doc_id + page + snippet)
    """
    doc = fitz.open(pdf_path)
    extracted = []

    def find_on_pages(pattern: str, field_name: str, postprocess=lambda x: x):
        regex = re.compile(pattern, re.IGNORECASE)
        for page_idx in range(len(doc)):
            text = doc[page_idx].get_text("text")
            m = regex.search(text)
            if m:
                raw = m.group(1).strip()
                val = postprocess(raw)
                snippet = text[max(0, m.start() - 60): m.end() + 60].replace("\n", " ")

                # Simple confidence calibration (quick win)
                conf = 0.85
                if field_name == "payment_terms":
                    conf = 0.75

                extracted.append({
                    "name": field_name,
                    "value": str(val),
                    "confidence": conf,
                    "provenance": {
                        "doc_id": doc_id,
                        "page": page_idx + 1,
                        "snippet": snippet
                    }
                })
                return True
        return False

    # Extraction patterns
    find_on_pages(r"Unit\s*Price:\s*€?\s*([0-9]+(?:\.[0-9]+)?)", "unit_price")
    find_on_pages(r"Total\s*Price:\s*€?\s*([0-9,]+(?:\.[0-9]+)?)", "total_price", lambda x: x.replace(",", ""))

    # Currency: explicit line
    has_currency = find_on_pages(r"Currency:\s*([A-Z]{3})", "currency", lambda x: x.upper())

    # Currency: inline fallback
    if not has_currency:
        regex_inline = re.compile(r"\b(EUR|USD|INR|GBP)\b", re.IGNORECASE)
        for page_idx in range(len(doc)):
            text = doc[page_idx].get_text("text")
            m = regex_inline.search(text)
            if m:
                val = m.group(1).upper()
                snippet = text[max(0, m.start() - 60): m.end() + 60].replace("\n", " ")
                extracted.append({
                    "name": "currency",
                    "value": val,
                    "confidence": 0.70,
                    "provenance": {
                        "doc_id": doc_id,
                        "page": page_idx + 1,
                        "snippet": snippet
                    }
                })
                break

    find_on_pages(r"Incoterms:\s*([A-Z]{3})", "incoterms", lambda x: x.upper())
    find_on_pages(r"Lead\s*Time:\s*([0-9]+)\s*days", "lead_time_days")
    find_on_pages(r"Payment\s*Terms:\s*(.+)", "payment_terms")

    # Contradiction detection
    contradictions = []
    seen = {}
    for f in extracted:
        k = f["name"]
        v = f["value"]
        if k in seen and seen[k] != v:
            contradictions.append(f"Contradiction: {k} has values '{seen[k]}' and '{v}'")
        else:
            seen[k] = v

    present = {f["name"] for f in extracted}
    missing = [c for c in CRITICAL_FIELDS if c not in present]

    return extracted, missing, contradictions


# ==========================
# Phase D — Scoring helpers (weighted, normalized, policy-aware gates)
# ==========================
def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _incoterms_score(incoterms: Optional[str]) -> float:
    if not incoterms:
        return 0.0
    inc = incoterms.upper().strip()
    prefs = {
        "DAP": 100.0,
        "DDP": 90.0,
        "CIF": 70.0,
        "FOB": 60.0,
        "FCA": 55.0,
        "EXW": 20.0,
    }
    return prefs.get(inc, 40.0)


def _payment_terms_score(payment_terms: Optional[str]) -> float:
    if not payment_terms:
        return 0.0
    t = payment_terms.lower()
    if "net 45" in t:
        return 85.0
    if "net 30" in t:
        return 75.0
    if "net 60" in t:
        return 65.0
    return 55.0


def compute_offer_score(offer: Offer, event: Optional[SourcingEvent] = None) -> dict:
    """
    Weighted scoring (0–100):
    - Price 0.4
    - Lead time 0.3
    - Incoterms 0.15
    - Payment terms 0.15

    Policy-aware hard gates (if event.requirements exists):
    - max_lead_time_days exceeded => FAIL
    - budget_cap exceeded beyond tolerance => FAIL
    """
    missing_fields = []
    reasons = []

    # ---- Extract values ----
    price = None
    if offer.total_price and offer.total_price.get("amount") is not None:
        price = float(offer.total_price["amount"])
    else:
        missing_fields.append("total_price")

    lead = None
    if offer.lead_time_days is not None:
        lead = float(offer.lead_time_days)
    else:
        missing_fields.append("lead_time_days")

    inc = offer.incoterms if offer.incoterms else None
    if not inc:
        missing_fields.append("incoterms")

    pay = offer.payment_terms if offer.payment_terms else None
    if not pay:
        missing_fields.append("payment_terms")

    if missing_fields:
        return {
            "status": "NEEDS_INFO",
            "score": 0.0,
            "reasons": [f"Missing required fields: {', '.join(missing_fields)}"],
            "missing_fields": missing_fields,
            "component_scores": {},
        }

    # ---- Policy-aware hard gates ----
    if event and getattr(event, "requirements", None):
        req = event.requirements

        # Lead time gate
        max_lead = getattr(req, "max_lead_time_days", None)
        if max_lead is not None and lead is not None and lead > float(max_lead):
            return {
                "status": "FAIL",
                "score": 0.0,
                "reasons": [f"Policy violation: lead_time_days {int(lead)} > max_lead_time_days {int(max_lead)}"],
                "missing_fields": [],
                "component_scores": {"lead_time": 0.0},
            }

        # Budget gate (with tolerance)
        cap = getattr(req, "budget_cap", None)
        tol = getattr(req, "budget_tolerance_percent", 0) or 0
        if cap and getattr(cap, "amount", None) is not None:
            cap_amt = float(cap.amount)
            allowed = cap_amt * (1.0 + float(tol) / 100.0)
            if price is not None and price > allowed:
                return {
                    "status": "FAIL",
                    "score": 0.0,
                    "reasons": [f"Policy violation: total_price {price:.2f} > budget cap {cap_amt:.2f} (+{tol}% tolerance)"],
                    "missing_fields": [],
                    "component_scores": {"price": 0.0},
                }

    # ---- Normalized component scores (0–100) ----
    # Price normalization window
    if event and getattr(event, "requirements", None) and getattr(event.requirements, "budget_cap", None):
        cap_amt = float(event.requirements.budget_cap.amount)
        worst = cap_amt * 1.2
        best = cap_amt * 0.6
    else:
        best, worst = 3000.0, 6000.0

    price_score = _clamp((worst - price) / (worst - best) * 100.0)

    # Lead time normalization window
    if event and getattr(event, "requirements", None) and getattr(event.requirements, "max_lead_time_days", None):
        max_lead = float(event.requirements.max_lead_time_days)
        best_lead = max(1.0, max_lead * 0.5)
        worst_lead = max_lead * 1.5
    else:
        best_lead, worst_lead = 7.0, 21.0

    lead_score = _clamp((worst_lead - lead) / (worst_lead - best_lead) * 100.0)

    inc_score = _incoterms_score(inc)
    pay_score = _payment_terms_score(pay)

    # ---- Weighted final ----
    w_price, w_lead, w_inc, w_pay = 0.4, 0.3, 0.15, 0.15
    final = (w_price * price_score) + (w_lead * lead_score) + (w_inc * inc_score) + (w_pay * pay_score)
    final = round(_clamp(final), 2)

    status = "PASS" if final >= 70 else "FAIL"

    reasons.append(f"Price: {price:.2f} -> {price_score:.1f}/100 (weight {w_price})")
    reasons.append(f"Lead time: {int(lead)} days -> {lead_score:.1f}/100 (weight {w_lead})")
    reasons.append(f"Incoterms: {inc} -> {inc_score:.1f}/100 (weight {w_inc})")
    reasons.append(f"Payment terms: {pay} -> {pay_score:.1f}/100 (weight {w_pay})")

    return {
        "status": status,
        "score": final,
        "reasons": reasons,
        "missing_fields": [],
        "component_scores": {
            "price": round(price_score, 2),
            "lead_time": round(lead_score, 2),
            "incoterms": round(inc_score, 2),
            "payment_terms": round(pay_score, 2),
        },
    }


def offer_evidence_stub(offer_id: str):
    """
    Return real evidence using parse provenance:
    - page
    - snippet
    - doc_id
    """
    if offer_id not in OFFERS:
        return []

    offer = OFFERS[offer_id]
    if not offer.attachments:
        return []

    att0 = offer.attachments[0]
    storage_key = att0.get("storage_key") if isinstance(att0, dict) else getattr(att0, "storage_key", None)
    doc_id = att0.get("doc_id") if isinstance(att0, dict) else getattr(att0, "doc_id", offer_id)

    if not storage_key:
        return []

    extracted_fields, _, _ = parse_offer_pdf_to_fields(storage_key, doc_id)

    wanted = {"total_price", "lead_time_days", "incoterms", "payment_terms", "unit_price", "currency"}
    ev = []
    for f in extracted_fields:
        if f.get("name") in wanted:
            prov = f.get("provenance", {}) or {}
            ev.append({
                "doc_id": prov.get("doc_id", doc_id),
                "page": prov.get("page"),
                "field": f.get("name"),
                "snippet": (prov.get("snippet") or "")[:200],
            })
    return ev[:8]


# ==========================
# Core endpoints
# ==========================
@app.get("/")
def root():
    return {
        "status": "AI-Pass running",
        "docs": "/docs",
        "rag_ready": bool(getattr(rag, "enabled", False)),
        "rag_error": getattr(rag, "init_error", None),
    }


@app.get("/debug/paths")
def debug_paths():
    project_root = Path(__file__).resolve().parents[1]
    audit_dir = project_root / "data" / "audit_logs"
    return {
        "main_py": str(Path(__file__).resolve()),
        "project_root": str(project_root),
        "audit_dir": str(audit_dir),
        "audit_dir_exists": audit_dir.exists(),
        "audit_files": [p.name for p in audit_dir.glob("*.json")] if audit_dir.exists() else [],
        "rag_ready": bool(getattr(rag, "enabled", False)),
        "rag_init_error": getattr(rag, "init_error", None),
    }


@app.get("/metrics")
def metrics():
    return compute_metrics()


@app.post("/ingest")
async def ingest(file: UploadFile = File(...), doc_id: str = "doc_1"):
    data = await file.read()
    return rag.ingest_pdf(data, doc_id=doc_id)


@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    return {"results": rag.retrieve(req.query, req.doc_ids, top_k=req.top_k)}


@app.post("/agent/run", response_model=DecisionOutput)
def run_agent(task: TaskInput):
    return agent.execute(task)


# ==========================
# Phase A — Event / Offer foundation
# ==========================
@app.post("/sourcing-events/create", response_model=SourcingEvent)
def create_sourcing_event(event: SourcingEvent, session: Session = Depends(get_session)):
    row = EventRow(event_id=event.event_id, payload_json=json.dumps(event.model_dump()))
    session.merge(row)  # upsert
    session.commit()
    return event



@app.get("/sourcing-events/{event_id}", response_model=SourcingEvent)
def get_sourcing_event(event_id: str, session: Session = Depends(get_session)):
    row = session.get(EventRow, event_id)
    if not row:
        raise HTTPException(status_code=404, detail="SourcingEvent not found")
    return SourcingEvent(**json.loads(row.payload_json))



@app.post("/offers/create", response_model=Offer)
def create_offer(offer: Offer, session: Session = Depends(get_session)):
    row = OfferRow(
        offer_id=offer.offer_id,
        supplier_id=offer.supplier_id,
        payload_json=json.dumps(offer.model_dump()),
    )
    session.merge(row)
    session.commit()
    return offer



@app.get("/offers/{offer_id}", response_model=Offer)
def get_offer(offer_id: str, session: Session = Depends(get_session)):
    row = session.get(OfferRow, offer_id)
    if not row:
        raise HTTPException(status_code=404, detail="Offer not found")
    return Offer(**json.loads(row.payload_json))



@app.post("/sourcing-events/{event_id}/attach-offer/{offer_id}", response_model=SourcingEvent)
def attach_offer_to_event(event_id: str, offer_id: str, session: Session = Depends(get_session)):
    erow = session.get(EventRow, event_id)
    if not erow:
        raise HTTPException(status_code=404, detail="SourcingEvent not found")

    orow = session.get(OfferRow, offer_id)
    if not orow:
        raise HTTPException(status_code=404, detail="Offer not found")

    event = SourcingEvent(**json.loads(erow.payload_json))
    if offer_id not in event.offer_ids:
        event.offer_ids.append(offer_id)

    erow.payload_json = json.dumps(event.model_dump())
    session.add(erow)
    session.commit()
    return event


# ==========================
# Phase B — Upload / Parse / Normalize / Full
# ==========================
@app.post("/offers/upload")
async def upload_offer_file(file: UploadFile = File(...), supplier_id: str = "SUP-UNKNOWN"):
    file_ext = Path(file.filename).suffix
    stored_name = f"{uuid.uuid4()}{file_ext}"
    file_path = UPLOAD_DIR / stored_name

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    offer_id = f"OFF-{uuid.uuid4().hex[:8]}"
    offer = Offer(
        offer_id=offer_id,
        supplier_id=supplier_id,
        attachments=[{
            "file_name": file.filename,
            "file_type": file_ext.replace(".", ""),
            "storage_key": str(file_path),
            "doc_id": offer_id
        }]
    )
    OFFERS[offer_id] = offer

    return {
        "offer_id": offer_id,
        "stored_file": str(file_path),
        "message": "File uploaded. Extraction pipeline pending (Phase B2 parse available)."
    }


@app.post("/offers/{offer_id}/parse", response_model=OfferParseResult)
def parse_offer(offer_id: str):
    if offer_id not in OFFERS:
        raise HTTPException(status_code=404, detail="Offer not found")

    offer = OFFERS[offer_id]
    att0 = offer.attachments[0] if offer.attachments else None
    storage_key = att0.get("storage_key") if isinstance(att0, dict) else getattr(att0, "storage_key", None)
    doc_id = att0.get("doc_id") if isinstance(att0, dict) else getattr(att0, "doc_id", offer_id)

    if not storage_key:
        raise HTTPException(status_code=400, detail="No uploaded file found for this offer")

    extracted_fields, missing, contradictions = parse_offer_pdf_to_fields(storage_key, doc_id)

    return OfferParseResult(
        offer_id=offer_id,
        doc_id=doc_id,
        extracted_fields=extracted_fields,
        missing_critical_fields=missing,
        contradictions=contradictions
    )


@app.post("/offers/{offer_id}/normalize", response_model=Offer)
def normalize_offer(offer_id: str):
    if offer_id not in OFFERS:
        raise HTTPException(status_code=404, detail="Offer not found")

    offer = OFFERS[offer_id]
    att0 = offer.attachments[0] if offer.attachments else None
    storage_key = att0.get("storage_key") if isinstance(att0, dict) else getattr(att0, "storage_key", None)
    doc_id = att0.get("doc_id") if isinstance(att0, dict) else getattr(att0, "doc_id", offer_id)

    if not storage_key:
        raise HTTPException(status_code=400, detail="No uploaded file found for this offer")

    extracted_fields, missing, contradictions = parse_offer_pdf_to_fields(storage_key, doc_id)

    if contradictions:
        raise HTTPException(
            status_code=409,
            detail={"message": "Contradictions found; cannot normalize safely", "contradictions": contradictions}
        )

    fdict = {f["name"]: f["value"] for f in extracted_fields}
    currency = fdict.get("currency")

    if "unit_price" in fdict and currency:
        offer.unit_price = {"amount": float(fdict["unit_price"]), "currency": currency}
    if "total_price" in fdict and currency:
        offer.total_price = {"amount": float(fdict["total_price"]), "currency": currency}
    if "incoterms" in fdict:
        offer.incoterms = fdict["incoterms"]
    if "lead_time_days" in fdict:
        offer.lead_time_days = int(fdict["lead_time_days"])
    if "payment_terms" in fdict:
        offer.payment_terms = fdict["payment_terms"]

    OFFERS[offer_id] = offer
    return offer


@app.get("/offers/{offer_id}/full")
def get_offer_full(offer_id: str):
    if offer_id not in OFFERS:
        raise HTTPException(status_code=404, detail="Offer not found")

    offer = OFFERS[offer_id]
    att0 = offer.attachments[0] if offer.attachments else None
    storage_key = att0.get("storage_key") if isinstance(att0, dict) else getattr(att0, "storage_key", None)
    doc_id = att0.get("doc_id") if isinstance(att0, dict) else getattr(att0, "doc_id", offer_id)

    if storage_key:
        extracted_fields, missing, contradictions = parse_offer_pdf_to_fields(storage_key, doc_id)
        return {
            "offer": offer,
            "parse": {
                "offer_id": offer_id,
                "doc_id": doc_id,
                "extracted_fields": extracted_fields,
                "missing_critical_fields": missing,
                "contradictions": contradictions
            }
        }

    return {"offer": offer, "parse": None}


# ==========================
# Phase C — Evaluate + Chat
# ==========================
@app.post("/events/{event_id}/evaluate")
def evaluate_event(event_id: str, session: Session = Depends(get_session)):
    # 1) Load event from DB
    erow = session.get(EventRow, event_id)
    if not erow:
        raise HTTPException(status_code=404, detail="SourcingEvent not found")

    event = SourcingEvent(**json.loads(erow.payload_json))

    if not event.offer_ids:
        raise HTTPException(status_code=400, detail="No offers attached to this event")

    # 2) Evaluate offers (load each from DB)
    per_offer = []
    for oid in event.offer_ids:
        orow = session.get(OfferRow, oid)
        if not orow:
            continue

        offer = Offer(**json.loads(orow.payload_json))
        result = compute_offer_score(offer, event)

        per_offer.append({
            "offer_id": oid,
            "supplier_id": offer.supplier_id,
            "status": result["status"],
            "score": result["score"],
            "reasons": result["reasons"],
            "missing_fields": result["missing_fields"],
            "component_scores": result.get("component_scores", {}),
            "evidence_refs": offer_evidence_stub(oid),
        })

    # 3) Rank & select winner
    ranked = sorted(
        per_offer,
        key=lambda x: (0 if x["status"] == "PASS" else 1, -float(x["score"]))
    )
    winner = next((x for x in ranked if x["status"] == "PASS"), None)

    # 4) Persist evaluation into DB
    evaluation = {
        "event_id": event_id,
        "created_at": datetime.utcnow().isoformat(),
        "offers": ranked,
        "winner": winner,
    }

    session.merge(EvaluationRow(event_id=event_id, payload_json=json.dumps(evaluation)))
    session.commit()

    return evaluation


@app.post("/chat/session/create")
def create_chat_session(req: ChatCreateRequest, session: Session = Depends(get_session)):
    erow = session.get(EventRow, req.event_id)
    if not erow:
        raise HTTPException(status_code=404, detail="SourcingEvent not found")

    session_id = f"CHAT-{uuid.uuid4().hex[:10]}"
    session.add(ChatSessionRow(session_id=session_id, event_id=req.event_id))
    session.commit()
    return {"session_id": session_id, "event_id": req.event_id}



def _require_eval(event_id: str, session: Session) -> dict:
    row = session.get(EvaluationRow, event_id)
    if not row:
        raise HTTPException(status_code=400, detail="No evaluation found. Run POST /events/{event_id}/evaluate first.")
    return json.loads(row.payload_json)



@app.post("/chat/ask")
def chat_ask(req: ChatAskRequest):
    if req.session_id not in CHAT_SESSIONS:
        raise HTTPException(status_code=404, detail="Chat session not found")

    event_id = CHAT_SESSIONS[req.session_id]["event_id"]
    evaluation = _require_eval(event_id)

    policy_hits = []
    if req.policy_doc_ids:
        if not getattr(rag, "enabled", False):
            # safe: chat still works without policy grounding
            policy_hits = []
        else:
            policy_hits = rag.retrieve(req.question, req.policy_doc_ids, top_k=5)

    q = req.question.strip().lower()

    # Why A over B
    if "why" in q and ("over" in q or "above" in q):
        tokens = req.question.replace(",", " ").split()
        ids = [t for t in tokens if t.startswith("OFF-")]
        if len(ids) < 2:
            return {"answer": "NEEDS_INFO: Please specify two offer IDs like: 'Why OFF-123 over OFF-456?'", "citations": []}

        a, b = ids[0], ids[1]
        offers_map = {o["offer_id"]: o for o in evaluation["offers"]}
        if a not in offers_map or b not in offers_map:
            return {"answer": "NEEDS_INFO: One of the offer IDs is not part of this event evaluation.", "citations": []}

        oa, ob = offers_map[a], offers_map[b]

        oa_obj = OFFERS.get(a)
        ob_obj = OFFERS.get(b)

        delta_parts = []
        try:
            if oa_obj and ob_obj and oa_obj.total_price and ob_obj.total_price:
                pa = float(oa_obj.total_price["amount"])
                pb = float(ob_obj.total_price["amount"])
                diff = pb - pa
                if diff > 0:
                    delta_parts.append(f"it is €{diff:.0f} cheaper")
                elif diff < 0:
                    delta_parts.append(f"it is €{abs(diff):.0f} more expensive")
        except Exception:
            pass

        try:
            if oa_obj and ob_obj and oa_obj.lead_time_days is not None and ob_obj.lead_time_days is not None:
                la = int(oa_obj.lead_time_days)
                lb = int(ob_obj.lead_time_days)
                d = lb - la
                if d > 0:
                    delta_parts.append(f"delivers {d} days faster")
                elif d < 0:
                    delta_parts.append(f"delivers {abs(d)} days slower")
        except Exception:
            pass

        if delta_parts:
            justification = f"{a} ranked higher because " + ", ".join(delta_parts) + "."
        else:
            justification = f"{a} ranked higher based on weighted scoring across price, lead time, incoterms, and payment terms."

        answer = (
            f"{a} ranked above {b}.\n\n"
            f"{justification}\n\n"
            f"Summary:\n"
            f"- {a}: status={oa['status']} score={oa['score']}\n"
            f"- {b}: status={ob['status']} score={ob['score']}\n\n"
            f"Key reasons for {a}:\n- " + "\n- ".join(oa["reasons"][:4]) + "\n\n"
            f"Key reasons for {b}:\n- " + "\n- ".join(ob["reasons"][:4])
        )

        citations = []
        citations += oa.get("evidence_refs", []) + ob.get("evidence_refs", [])
        for h in policy_hits:
            citations.append({
                "doc_id": h.get("doc_id"),
                "chunk_id": h.get("chunk_id"),
                "page": h.get("page"),
                "snippet": (h.get("text") or "")[:200],
            })

        return {"answer": answer, "citations": citations}

    # Missing info
    if "missing" in q or "needs info" in q:
        needs = [o for o in evaluation["offers"] if o["status"] == "NEEDS_INFO"]
        if not needs:
            return {"answer": "All offers are PASS/FAIL. No missing info detected.", "citations": []}
        lines = [f"{o['offer_id']} missing: {', '.join(o['missing_fields'])}" for o in needs]
        citations = []
        for o in needs:
            citations += o.get("evidence_refs", [])
        return {"answer": "\n".join(lines), "citations": citations}

    # Draft email
    if "draft" in q and ("email" in q or "mail" in q):
        tokens = req.question.replace(",", " ").split()
        ids = [t for t in tokens if t.startswith("OFF-")]
        if not ids:
            return {"answer": "NEEDS_INFO: Please include offer id like 'Draft email for OFF-123'.", "citations": []}

        oid = ids[0]
        offers_map = {o["offer_id"]: o for o in evaluation["offers"]}
        if oid not in offers_map:
            return {"answer": "NEEDS_INFO: Offer id not found in this event evaluation.", "citations": []}

        o = offers_map[oid]
        missing = o.get("missing_fields", [])
        if not missing:
            return {"answer": "No missing fields detected for this offer; no clarification email needed.", "citations": []}

        email_text = (
            "Subject: Clarification Request — Missing Information for Your Offer\n\n"
            f"Hello,\n\nThank you for submitting your offer ({oid}). "
            "To proceed with evaluation, we need clarification / missing details on the following items:\n"
            + "\n".join([f"- {m}" for m in missing]) +
            "\n\nPlease share the requested information (and supporting documents if applicable) at your earliest convenience.\n\n"
            "Best regards,\nProcurement Team"
        )
        return {"answer": email_text, "citations": o.get("evidence_refs", [])}

    winner = evaluation.get("winner")
    base = "No PASS winner found yet. Some offers may be NEEDS_INFO." if not winner else \
        f"Current top-ranked offer is {winner['offer_id']} with score {winner['score']}."
    citations = [] if not winner else winner.get("evidence_refs", [])
    return {"answer": base, "citations": citations}
