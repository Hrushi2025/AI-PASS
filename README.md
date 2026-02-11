AI-PASS — Core MVP (Reliability & Decision Intelligence Layer)
RAG + Deterministic Agent + Governance + Explainability + Evaluation + Early Productization
Overview

AI-PASS is a governance-aware reliability layer for enterprise AI workflows, designed to ensure that every AI-driven decision is:

Evidence-grounded (via Retrieval-Augmented Generation)

Explainable and auditable

Policy-aware and compliant

Deterministic and reproducible

Measurable via clear KPIs

Moving toward production readiness (Phase E started)

This system was developed as the backend intelligence core for AI-PASS in the Procurement / Invoice domain, focusing on offer evaluation, automated decisioning, and explainability.

Scope of This MVP (What AI-PASS Solves)

AI-PASS acts as a decision reliability layer, not just an invoice parser. It provides:

Structured document understanding

Evidence-backed AI reasoning

Transparent scoring instead of black-box decisions

Governance controls for enterprise compliance

Traceable audit records

Interactive justification via chat

Early engineering hardening (Phase E started)

What Has Been Implemented (Phases A–E)
Phase A — Document Intelligence (Extraction & Structuring) — COMPLETED

Key file:
app/main.py → parse_offer_pdf_to_fields()

AI-PASS now includes a reliable, explainable extraction pipeline that:

Accepts PDF offer uploads via /offers/upload
Extracts structured fields from multi-page PDFs using rule-based parsing:

Extracted fields include:

unit_price

total_price

currency

incoterms

lead_time_days

payment_terms

Each extracted field includes:

Confidence score

Provenance metadata:

Document ID

Page number

Text snippet (evidence)

The system also detects:

Missing critical fields

Contradictions across pages

 Why this matters:
This makes AI-PASS transparent, auditable, and research-aligned rather than a black-box extractor.

 Phase B — RAG Evidence Engine — COMPLETED

 Key file:
app/services/rag.py

AI-PASS includes a full Retrieval-Augmented Generation (RAG) pipeline using Qdrant, supporting:

PDF ingestion → chunking → embedding → storage
Semantic retrieval with:

Score thresholding

Minimum evidence hit requirement

Document-level filtering

Each retrieved evidence item contains:

Page number

Chunk snippet

Document reference

Why this matters:
All AI decisions are now grounded in real document evidence, which is essential for governance-aware AI.

Phase C — Explainable Chat + Decision Transparency — COMPLETED

Key file:
app/main.py → /chat/ask

The interactive chat layer supports:

“Why OFF-A over OFF-B?”

Price comparison

Lead time comparison

Weighted scoring explanation

Evidence citations (page + snippet)

Queries like:

“Which offers are missing information?”

“Draft an email to supplier for missing fields”

Graceful fallback if RAG is unavailable

Example response:

“OFF-123 ranked higher because it is €300 cheaper and delivers 4 days faster than OFF-456.”

Why this matters:
AI-PASS is now an explainable decision-support system, not just an automation script.

Phase D — AI/ML Evaluation Upgrade — COMPLETED (Major Milestone)

Key file:
app/main.py → compute_offer_score()

We replaced simple rule-based scoring with a principled, weighted, normalized evaluation model (0–100 scale):

New Weighted Scoring Model
Factor	Weight
Price	0.40
Lead Time	0.30
Incoterms	0.15
Payment Terms	0.15
Policy-Aware Hard Gates

If an event defines requirements:

If lead_time_days exceeds max_lead_time_days → FAIL

If total_price exceeds budget cap (with tolerance) → FAIL

Output includes:

Final score

Component-level breakdown

Clear human-readable reasons

Why this matters:
This aligns with your research theme:

Data + Policy + AI = Reliable Decision

Phase E — Engineering Hardening (STARTED)

Key file updated:
app/main.py → /events/{event_id}/evaluate

We moved from pure in-memory evaluation to persistent storage, meaning:

Evaluations are now saved in the database (EvaluationRow)
Stored data includes:

Ranked offers

Scores

Reasons

Evidence references

Winner selection

Why this matters:
This is the first step toward a production-ready, auditable system, rather than a throwaway prototype.

Core Capabilities (Technical Summary)
RAG Evidence Engine

PDF ingest → chunk → embed → store in Qdrant

Query → retrieve top-k evidence with citations (doc_id, page, snippet)

Structured Decision Schema

Every decision returns:

decision: PASS / FAIL / NEEDS_INFO

confidence: float

evidence: citations (doc_id, page, snippet)

policy_violations: governance flags

Deterministic Agent Loop (Auditable)

Agent executes in a fixed pipeline:

Intake → Plan → Execute → Evaluate → Deliver


(Not a chat-based LLM loop — fully deterministic and reproducible.)

Governance Rules

No evidence → cannot PASS

Low confidence → NEEDS_INFO

Invoice > €5000 → human approval

Fraud / duplicate flags → FAIL

Audit Logging

Audit logs are stored in:

data/audit_logs/*.json


Each log includes:

audit_id

task input

decision

confidence

evidence

policy violations

Evaluation Suite

Run evaluation locally:

python -m app.eval.run_eval_v2

Key Files in the Repository
File	Purpose
app/main.py	Core API, extraction, scoring, chat, evaluation
app/services/rag.py	RAG engine (Qdrant integration)
app/services/agent.py	Deterministic decision agent
app/core/schemas.py	Data models
app/db.py	Database setup
app/models.py	SQLModel tables
app/services/evaluator.py	Evaluation logic
docker-compose.yml	Qdrant + backend stack
app/test_smoke_flow.py	Basic end-to-end test
