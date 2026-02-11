from typing import List, Dict, Any, Tuple
from app.core.schemas import DecisionOutput


class Evaluator:
    def __init__(self, confidence_threshold: float = 0.70):
        self.conf_threshold = confidence_threshold

    def apply_rules(
        self,
        draft: DecisionOutput,
        evidence: List[Dict[str, Any]],
        policy: dict,
        task_type: str = ""
    ) -> Tuple[DecisionOutput, List[str]]:

        violations = []

        # ============================
        # RULE 0 (SAFETY): If NO evidence at all, system must not PASS
        # ============================
        if not evidence or len(evidence) == 0:
            if draft.decision == "PASS":
                violations.append("NO_EVIDENCE_FOR_PASS")
                draft.decision = "NEEDS_INFO"
            # Even if draft was not PASS, keep decision as-is.
            # Add missing info only when decision becomes NEEDS_INFO.
            if draft.decision == "NEEDS_INFO":
                if "Provide supporting evidence/citations." not in draft.missing_info:
                    draft.missing_info.append("Provide supporting evidence/citations.")

        # ============================
        # RULE 1: No evidence → cannot PASS (kept for clarity)
        # ============================
        if draft.decision == "PASS" and len(evidence) == 0:
            violations.append("NO_EVIDENCE_FOR_PASS")
            draft.decision = "NEEDS_INFO"
            draft.missing_info.append("Provide supporting evidence/citations.")

        # ============================
        # RULE 2: Low confidence → NEEDS_INFO
        # ============================
        if draft.confidence < self.conf_threshold and draft.decision == "PASS":
            violations.append("LOW_CONFIDENCE")
            draft.decision = "NEEDS_INFO"
            draft.missing_info.append("Low confidence. Human review needed.")

        # ==================================================
        # INVOICE POLICY PACK (ONLY FOR invoice_approval)
        # ==================================================
        if task_type == "invoice_approval":
            invoice_amount = policy.get("invoice_amount")
            threshold = policy.get("human_approval_threshold", 5000)

            required_fields = policy.get(
                "required_fields",
                ["vendor", "invoice_number", "invoice_date", "total_amount"]
            )
            provided_fields = policy.get("provided_fields", [])

            fraud_flag = policy.get("fraud_flag", False)
            duplicate_flag = policy.get("duplicate_flag", False)

            # -------- FAIL RULES (Hard stop) --------
            if fraud_flag:
                violations.append("FRAUD_SUSPECTED")
                draft.decision = "FAIL"
                draft.confidence = min(draft.confidence, 0.60)

            if duplicate_flag:
                violations.append("DUPLICATE_INVOICE")
                draft.decision = "FAIL"
                draft.confidence = min(draft.confidence, 0.60)

            # -------- NEEDS_INFO RULES --------
            missing = [f for f in required_fields if f not in provided_fields]

            if missing and draft.decision != "FAIL":
                violations.append("MISSING_REQUIRED_FIELDS")
                draft.decision = "NEEDS_INFO"
                draft.confidence = min(draft.confidence, 0.55)
                draft.missing_info.append(
                    f"Missing required fields: {', '.join(missing)}"
                )

            if (
                invoice_amount is not None
                and invoice_amount > threshold
                and draft.decision != "FAIL"
            ):
                violations.append("HUMAN_APPROVAL_REQUIRED")
                draft.decision = "NEEDS_INFO"
                draft.confidence = min(draft.confidence, 0.55)
                draft.missing_info.append(
                    f"Invoice amount {invoice_amount} > {threshold}. Require human approval."
                )

        # Attach all violations to the draft object
        draft.policy_violations.extend(violations)

        return draft, violations
