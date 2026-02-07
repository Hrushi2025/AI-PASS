from typing import List

from app.core.schemas import TaskInput, DecisionOutput, EvidenceItem
from app.core.confidence import confidence_from_hits
from app.services.rag import RAGService
from app.services.evaluator import Evaluator
from app.services.audit import write_audit_log


class AgentEngine:
    def __init__(self):
        self.rag = RAGService()
        self.eval = Evaluator(confidence_threshold=0.70)

    def plan(self, task: TaskInput) -> List[str]:
        if task.task_type == "invoice_approval":
            return ["retrieve_evidence", "draft_decision", "evaluate_rules", "deliver"]
        return ["retrieve_evidence", "draft_decision", "evaluate_rules", "deliver"]

    def execute(self, task: TaskInput) -> DecisionOutput:
        # 1) Retrieve evidence
        evidence_hits = self.rag.retrieve(task.query, task.doc_ids, top_k=5)

        # ✅ Day 6: Deduplicate evidence by chunk_id (keep best-ranked hits)
        seen = set()
        dedup_hits = []
        for h in evidence_hits:
            cid = h.get("chunk_id")
            if cid and cid not in seen:
                seen.add(cid)
                dedup_hits.append(h)
        evidence_hits = dedup_hits

        # ✅ Day 6: Filter weak/irrelevant evidence (prevents unrelated citations)
        MIN_EVIDENCE_SCORE = 0.30  # tune later
        evidence_hits = [
            h for h in evidence_hits
            if float(h.get("score", 0.0)) >= MIN_EVIDENCE_SCORE
        ]

        # 2) Build evidence items AFTER filtering
        evidence_items = [
            EvidenceItem(
                doc_id=h["doc_id"],
                chunk_id=h["chunk_id"],
                page=h.get("page"),
                text_snippet=h.get("text", "")
            )
            for h in evidence_hits
        ]

        # ✅ Day 6: Recompute confidence AFTER filtering
        conf = confidence_from_hits(evidence_hits)

        # Draft decision (PASS only if strong evidence + high confidence)
        draft = DecisionOutput(
            decision="PASS" if (evidence_items and conf >= 0.70) else "NEEDS_INFO",
            confidence=conf,
            evidence=evidence_items
        )

        # 3) Evaluate rules + policy enforcement (final gatekeeper)
        # ✅ FIX: pass task.task_type so invoice rules apply ONLY to invoice_approval
        draft, _ = self.eval.apply_rules(draft, evidence_hits, task.policy, task.task_type)

        # 4) Audit log (add evidence quality metrics)
        top_score = float(evidence_hits[0]["score"]) if evidence_hits else None

        audit_event = {
            "audit_id": draft.audit_id,
            "task_type": task.task_type,
            "query": task.query,
            "doc_ids": task.doc_ids,
            "decision": draft.decision,
            "confidence": draft.confidence,
            "top_retrieval_score": top_score,
            "evidence_count": len(evidence_hits),
            "min_evidence_score": MIN_EVIDENCE_SCORE,
            "policy": task.policy,
            "policy_violations": draft.policy_violations,
            "missing_info": draft.missing_info,
            "evidence": [e.model_dump() for e in draft.evidence],
        }

        saved_path = write_audit_log(audit_event)
        print("AUDIT SAVED TO:", saved_path)

        return draft
