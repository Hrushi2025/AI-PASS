# ==========================
# Procurement / Supply Chain AI (Phase 1 Foundation)
# ==========================

from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import date, datetime
from pydantic import BaseModel, Field, validator

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    doc_id: str
    chunk_id: str
    page: int
    snippet: str
    score: float


class TaskInput(BaseModel):
    task_type: str = Field(..., description="e.g., doc_qa, invoice_policy, procurement_offer_eval")
    query: str
    doc_ids: List[str] = Field(default_factory=list)
    policy: Dict[str, Any] = Field(default_factory=dict)
    top_k: int = 5


class DecisionOutput(BaseModel):
    decision: str = Field(..., description="PASS|FAIL|NEEDS_INFO")
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: List[EvidenceItem] = Field(default_factory=list)
    missing_info: List[str] = Field(default_factory=list)
    policy_violations: List[str] = Field(default_factory=list)
    audit_id: Optional[str] = None


class VersionedModel(BaseModel):
    """
    Simple versioning base for configs that must be reproducible/auditable.
    """
    version: int = Field(1, ge=1, description="Monotonic version number")


class OfferStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NEEDS_INFO = "NEEDS_INFO"


class AttachmentRef(BaseModel):
    """
    Reference to a stored file (offer pdf, certificate, etc.).
    In your system later this can map to object storage keys.
    """
    file_name: str = Field(..., min_length=1)
    file_type: str = Field(..., min_length=1, description="pdf|docx|xlsx|csv|png etc.")
    storage_key: Optional[str] = Field(None, description="Object storage key/path if used")
    doc_id: Optional[str] = Field(None, description="If indexed in RAG, doc_id used for citations")


class Money(BaseModel):
    amount: float = Field(..., ge=0)
    currency: str = Field(..., min_length=3, max_length=3, description="ISO 4217 like EUR, USD, INR")

    @validator("currency")
    def currency_upper(cls, v: str) -> str:
        return v.upper()


class Incoterm(str, Enum):
    EXW = "EXW"
    FCA = "FCA"
    CPT = "CPT"
    CIP = "CIP"
    DAP = "DAP"
    DPU = "DPU"
    DDP = "DDP"
    FAS = "FAS"
    FOB = "FOB"
    CFR = "CFR"
    CIF = "CIF"


class Certification(BaseModel):
    name: str = Field(..., min_length=1, description="e.g., ISO 9001, ISO 27001, RoHS")
    issuer: Optional[str] = None
    valid_until: Optional[date] = None
    attachment: Optional[AttachmentRef] = None


class ESGClaim(BaseModel):
    """
    Keep it flexible. Real ESG scoring comes later.
    """
    claim: str = Field(..., min_length=1, description="ESG statement from supplier")
    evidence: Optional[AttachmentRef] = None


class OfferItem(BaseModel):
    """
    Optional (multi-lot / multi-item). For MVP you can keep one item.
    """
    item_id: str = Field(..., min_length=1)
    description: Optional[str] = None
    quantity: float = Field(1, gt=0)
    unit_price: Optional[Money] = None
    total_price: Optional[Money] = None
    moq: Optional[float] = Field(None, gt=0, description="Minimum order quantity")
    capacity: Optional[float] = Field(None, gt=0, description="Supplier capacity for this item")


class Offer(BaseModel):
    """
    Canonical Offer Schema (normalized).
    Later, extracted fields with confidence/provenance will be stored separately.
    """
    offer_id: str = Field(..., min_length=1)
    supplier_id: str = Field(..., min_length=1)
    supplier_name: Optional[str] = None

    # Pricing
    unit_price: Optional[Money] = None
    total_price: Optional[Money] = None

    # Logistics / commercial
    incoterms: Optional[Incoterm] = None
    shipping_terms: Optional[str] = None
    lead_time_days: Optional[int] = Field(None, ge=0)
    promised_delivery_date: Optional[date] = None
    location_region: Optional[str] = Field(None, description="Country/region; used for restrictions/sanctions")

    # Commercial constraints
    payment_terms: Optional[str] = None
    validity_until: Optional[date] = None
    warranty_months: Optional[int] = Field(None, ge=0)
    sla_terms: Optional[str] = None
    penalties: Optional[str] = None
    service_clauses: Optional[str] = None

    # Supply capability
    moq: Optional[float] = Field(None, gt=0)
    capacity: Optional[float] = Field(None, gt=0)

    # Compliance / ESG
    certifications: List[Certification] = Field(default_factory=list)
    esg_claims: List[ESGClaim] = Field(default_factory=list)

    # Attachments (offer PDF, spreadsheets, certificates, etc.)
    attachments: List[AttachmentRef] = Field(default_factory=list)

    # Optional multi-item support
    items: List[OfferItem] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=datetime.utcnow)

    @validator("supplier_id", "offer_id")
    def strip_ids(cls, v: str) -> str:
        return v.strip()


class RequirementsProfile(VersionedModel):
    """
    Requirements/constraints for a sourcing event.
    Includes both structured requirements and the NL requirements + parsed output.
    """
    # Structured constraints (mandatory in roadmap)
    budget_cap: Optional[Money] = None
    budget_tolerance_percent: Optional[float] = Field(None, ge=0, le=100)

    max_lead_time_days: Optional[int] = Field(None, ge=0)
    required_delivery_date: Optional[date] = None

    required_certifications: List[str] = Field(default_factory=list)
    payment_terms_constraints: Optional[str] = None

    quality_thresholds: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="e.g., {'max_defect_rate': 0.02, 'min_warranty_months': 12}"
    )

    esg_min_score: Optional[float] = Field(None, ge=0, le=100)
    esg_required_certificates: List[str] = Field(default_factory=list)

    location_restrictions: List[str] = Field(
        default_factory=list,
        description="Restricted countries/regions; can be used in rules engine"
    )

    # Natural language requirements (mandatory)
    nl_requirements_text: str = Field("", description="Plain language requirements written by the user")
    nl_parsed_constraints: Dict[str, Any] = Field(
        default_factory=dict,
        description="System-parsed constraints/weights from NL requirements"
    )
    nl_parse_confirmed: bool = Field(
        False,
        description="UI must require user confirmation; backend stores confirmation flag"
    )


class SourcingEvent(VersionedModel):
    """
    Sourcing event metadata + linkage to requirements/policies.
    """
    event_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    category: Optional[str] = None
    material_or_service_type: Optional[str] = None
    buyer_org: Optional[str] = None
    department: Optional[str] = None

    deadline: Optional[datetime] = None
    currency: str = Field("EUR", min_length=3, max_length=3)
    target_delivery_window: Optional[str] = None

    mandatory_policy_set_id: Optional[str] = Field(None, description="Active policy set ID for this org")
    required_documents_checklist: List[str] = Field(default_factory=list)

    requirements: RequirementsProfile = Field(default_factory=RequirementsProfile)

    offer_ids: List[str] = Field(default_factory=list, description="Offers associated with this event")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @validator("currency")
    def event_currency_upper(cls, v: str) -> str:
        return v.upper()


# ==========================
# Phase B2 — Extraction Output Models
# ==========================

class FieldProvenance(BaseModel):
    doc_id: str
    page: int
    snippet: str


class ExtractedField(BaseModel):
    name: str
    value: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    provenance: FieldProvenance


class OfferParseResult(BaseModel):
    offer_id: str
    doc_id: str
    extracted_fields: List[ExtractedField] = Field(default_factory=list)
    missing_critical_fields: List[str] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list)
