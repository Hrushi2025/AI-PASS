from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from uuid6 import uuid7

DecisionType = Literal["PASS", "FAIL", "NEEDS_INFO"]

class EvidenceItem(BaseModel):
    doc_id: str
    chunk_id: str
    page: Optional[int] = None
    text_snippet: str

class TaskInput(BaseModel):
    task_type: str = Field(..., examples=["invoice_approval", "doc_qa"])
    query: str
    doc_ids: List[str] = Field(default_factory=list)
    policy: dict = Field(default_factory=dict)

class DecisionOutput(BaseModel):
    decision: DecisionType
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: List[EvidenceItem] = Field(default_factory=list)
    missing_info: List[str] = Field(default_factory=list)
    policy_violations: List[str] = Field(default_factory=list)
    audit_id: str = Field(default_factory=lambda: str(uuid7()))
