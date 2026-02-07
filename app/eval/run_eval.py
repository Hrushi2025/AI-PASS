from app.services.agent import AgentEngine
from app.core.schemas import TaskInput

def run():
    agent = AgentEngine()

    tests = [
        {
            "name": "No evidence -> should not PASS",
            "task": TaskInput(task_type="invoice_approval", query="What is the total amount?", doc_ids=["unknown_doc"]),
            "expect_not_pass": True
        },
        {
            "name": "Policy: amount>5000 -> NEEDS_INFO",
            "task": TaskInput(
                task_type="invoice_approval",
                query="Approve this invoice",
                doc_ids=[],
                policy={"invoice_amount": 6200, "human_approval_threshold": 5000}
            ),
            "expect_decision": "NEEDS_INFO"
        }
    ]

    total = len(tests)
    pass_count = 0

    for t in tests:
        out = agent.execute(t["task"])
        ok = True

        if t.get("expect_not_pass") and out.decision == "PASS":
            ok = False
        if "expect_decision" in t and out.decision != t["expect_decision"]:
            ok = False

        print("\nTEST:", t["name"])
        print("OUTPUT:", out.model_dump())

        if ok:
            pass_count += 1

    print("\n====== SUMMARY ======")
    print(f"Passed {pass_count}/{total}")

if __name__ == "__main__":
    run()
