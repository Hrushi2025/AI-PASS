from typing import List

from app.core.schemas import TaskInput, DecisionOutput, EvidenceItem
from app.core.confidence import confidence_from_hits
from app.core.config import EVIDENCE_MIN_SCORE, EVIDENCE_MIN_HITS, CONFIDENCE_PASS_THRESHOLD
from app.services.rag import RAGService
from app.services.evaluator import Evaluator
from app.services.audit import write_audit_log


class AgentEngine:
    def __init__(self):
        self.rag = RAGService()
        self.eval = Evaluator(confidence_threshold=CONFIDENCE_PASS_THRESHOLD)

    def plan(self, task: TaskInput) -> List[str]:
        # Keeping structure consistent for auditability
        return ["retrieve_evidence", "draft_decision", "evaluate_rules", "deliver"]

    def execute(self, task: TaskInput) -> DecisionOutput:
        # 1) Retrieve evidence (already filtered by min score + min hits inside RAG)
        evidence_hits = self.rag.retrieve(task.query, task.doc_ids, top_k=5)

        # 2) Deduplicate evidence by chunk_id (keep best-ranked hits)
        seen = set()
        dedup_hits = []
        for h in evidence_hits:
            cid = h.get("chunk_id")
            if cid and cid not in seen:
                seen.add(cid)
                dedup_hits.append(h)
        evidence_hits = dedup_hits

        # 3) Enforce min hits requirement (extra safety, in case future RAG changes)
        if len(evidence_hits) < EVIDENCE_MIN_HITS:
            evidence_hits = []

        # 4) Build evidence items AFTER filtering
        evidence_items = [
            EvidenceItem(
                doc_id=h.get("doc_id"),
                chunk_id=h.get("chunk_id"),
                page=h.get("page"),
                text_snippet=h.get("text", "")
            )
            for h in evidence_hits
        ]

        # 5) Compute confidence from evidence quality
        conf = confidence_from_hits(evidence_hits)

        # Draft decision: PASS only if we have evidence AND confidence passes threshold
        draft = DecisionOutput(
            decision="PASS" if (len(evidence_items) > 0 and conf >= CONFIDENCE_PASS_THRESHOLD) else "NEEDS_INFO",
            confidence=conf,
            evidence=evidence_items
        )

        # 6) Apply governance + policy enforcement
        draft, _ = self.eval.apply_rules(draft, evidence_hits, task.policy, task.task_type)

        # 7) Audit log
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
            "evidence_min_score": EVIDENCE_MIN_SCORE,
            "evidence_min_hits": EVIDENCE_MIN_HITS,
            "policy": task.policy,
            "policy_violations": draft.policy_violations,
            "missing_info": draft.missing_info,
            "evidence": [e.model_dump() for e in draft.evidence],
        }

        saved_path = write_audit_log(audit_event)
        print("AUDIT SAVED TO:", saved_path)

        return draft
