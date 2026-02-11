from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    r = client.get("/")
    assert r.status_code == 200

def test_end_to_end_minimal():
    # Create event
    evt = {
        "version": 1,
        "event_id": "EVT-TEST",
        "title": "Test Event",
        "category": "X",
        "material_or_service_type": "Y",
        "buyer_org": "HOPn",
        "department": "Supply Chain",
        "deadline": "2026-02-15T12:00:00",
        "currency": "EUR",
        "target_delivery_window": "Within 14 days",
        "mandatory_policy_set_id": "POLICY-SET-DEFAULT",
        "required_documents_checklist": [],
        "requirements": {
            "version": 1,
            "budget_cap": {"amount": 5000, "currency": "EUR"},
            "budget_tolerance_percent": 5,
            "max_lead_time_days": 14,
            "required_delivery_date": None,
            "required_certifications": [],
            "payment_terms_constraints": None,
            "quality_thresholds": {},
            "esg_min_score": None,
            "esg_required_certificates": [],
            "location_restrictions": [],
            "nl_requirements_text": "",
            "nl_parsed_constraints": {},
            "nl_parse_confirmed": False
        },
        "offer_ids": [],
        "created_at": "2026-02-08T00:00:00"
    }
    assert client.post("/sourcing-events/create", json=evt).status_code == 200
