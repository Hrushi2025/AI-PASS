import json
from pathlib import Path

from app.services.agent import AgentEngine
from app.core.schemas import TaskInput


def load_cases() -> list[dict]:
    """
    Loads evaluation cases from:
      D:\\AI intern\\AI-PASS\\app\\eval\\eval_cases_v2.json

    Use relative path so it works across machines as well.
    """
    path = Path("app/eval/eval_cases_v2.json")

    if not path.exists():
        raise FileNotFoundError(
            f"Eval cases file not found: {path.resolve()}\n"
            f"Expected it at: D:\\AI intern\\AI-PASS\\app\\eval\\eval_cases_v2.json"
        )

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise ValueError("Eval cases JSON must be a list of cases (top-level array).")

    return data


def parse_expected(case: dict) -> tuple[str, bool]:
    """
    Supports BOTH formats:

    New format:
      "expected": { "decision": "PASS", "evidence": true }

    Old format (backward compatible):
      "expected_decision": "PASS",
      "expect_evidence": true
    """
    if "expected" in case and isinstance(case["expected"], dict):
        exp_dec = case["expected"].get("decision")
        exp_ev = case["expected"].get("evidence")
    else:
        exp_dec = case.get("expected_decision")
        exp_ev = case.get("expect_evidence")

    if exp_dec is None or exp_ev is None:
        raise ValueError(
            f"Case '{case.get('name', case.get('id', '<unknown>'))}' is missing expected fields.\n"
            f"Provide either:\n"
            f"  expected: {{ decision: ..., evidence: ... }}\n"
            f"OR the legacy:\n"
            f"  expected_decision + expect_evidence"
        )

    if not isinstance(exp_dec, str):
        raise ValueError("expected decision must be a string (PASS/FAIL/NEEDS_INFO).")

    if not isinstance(exp_ev, bool):
        raise ValueError("expected evidence must be boolean (true/false).")

    return exp_dec, exp_ev


def case_name(case: dict) -> str:
    return case.get("name") or case.get("id") or "<unnamed case>"


def main():
    agent = AgentEngine()
    cases = load_cases()

    total = 0
    correct_decision = 0
    false_pass = 0
    evidence_correct = 0

    print("\n===== RUNNING EVAL =====\n")
    print(f"Loaded cases from: {Path('app/eval/eval_cases_v2.json').resolve()}\n")

    for c in cases:
        total += 1

        if "task" not in c or not isinstance(c["task"], dict):
            raise ValueError(f"Case '{case_name(c)}' must contain a 'task' object.")

        task = TaskInput(**c["task"])
        out = agent.execute(task)

        exp_dec, exp_ev = parse_expected(c)

        got_dec = out.decision
        got_ev = len(out.evidence) > 0

        # Metrics
        if got_dec == exp_dec:
            correct_decision += 1

        if got_dec == "PASS" and exp_dec != "PASS":
            false_pass += 1

        # Evidence correctness rule:
        # - if PASS => must have evidence
        # - if expected says no evidence => should have none
        ev_ok = True
        if got_dec == "PASS" and not got_ev:
            ev_ok = False
        if exp_ev is False and got_ev is True:
            ev_ok = False

        if ev_ok:
            evidence_correct += 1

        print(f"CASE: {case_name(c)}")
        print(f"EXPECTED: decision={exp_dec}, evidence={exp_ev}")
        print(f"GOT     : decision={got_dec}, evidence={got_ev}, confidence={out.confidence}")
        print("-" * 60)

    acc = correct_decision / total if total else 0.0
    fp = false_pass / total if total else 0.0
    ev_acc = evidence_correct / total if total else 0.0

    print("\n===== SUMMARY =====")
    print(f"Total cases: {total}")
    print(f"Decision accuracy: {acc:.2%} ({correct_decision}/{total})")
    print(f"False-PASS rate  : {fp:.2%} ({false_pass}/{total})")
    print(f"Evidence correct : {ev_acc:.2%} ({evidence_correct}/{total})")


if __name__ == "__main__":
    main()
