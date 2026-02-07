# AI-PASS — Core MVP (Reliability Layer)
**RAG + Deterministic Agent + Governance + Audit + KPIs**

AI-PASS is a reliability layer for enterprise AI workflows that enforces **evidence-first decisions** (RAG), **deterministic agent execution**, and **governance controls** with full **auditability**.

This MVP focuses on the “reliability core”:
- Retrieval-Augmented Generation (RAG) evidence engine
- Structured decision output (PASS / FAIL / NEEDS_INFO)
- Governance rules (no evidence → cannot PASS, low confidence → NEEDS_INFO)
- Invoice policy gate (amount > €5000 → route to human approval)
- Audit logging + evaluation suite + reliability metrics endpoint

---

## ✅ What’s implemented (MVP Scope)

### 1) RAG Evidence Engine
- PDF ingest → chunk → embed → store in **Qdrant**
- Query → retrieve **top-k** evidence with citations (doc_id / chunk_id / page)

### 2) Decision Schema (Structured Output)
Every decision returns:
- `decision`: PASS / FAIL / NEEDS_INFO  
- `confidence`: float  
- `evidence`: citations (doc_id, chunk_id, page, snippet)  
- `policy_violations`: governance flags  
- `audit_id`: trace id for audit logs  

### 3) Deterministic Agent Loop (Auditable)
Agent executes in a fixed pipeline (not “chat”):
**Intake → Plan → Execute → Evaluate → Deliver**

### 4) Governance Rules
- No evidence → cannot PASS  
- Low confidence → NEEDS_INFO  
- Invoice policy gate: `invoice_amount > 5000` → human approval route  
- Fraud / duplicate flags override → FAIL  

### 5) Audit Logging
- Audit logs are saved as JSON:
  `data/audit_logs/*.json`
- Includes: audit_id, task, decision, confidence, evidence, policy violations

### 6) Evaluation Suite
Run eval:
```bash
python -m app.eval.run_eval_v2
