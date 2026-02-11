from typing import Optional
from datetime import datetime
from sqlmodel import SQLModel, Field

class OfferRow(SQLModel, table=True):
    offer_id: str = Field(primary_key=True)
    supplier_id: str
    offer_json: str  # store full Offer as JSON string
    created_at: datetime = Field(default_factory=datetime.utcnow)

class EventRow(SQLModel, table=True):
    event_id: str = Field(primary_key=True)
    event_json: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class EvaluationRow(SQLModel, table=True):
    event_id: str = Field(primary_key=True)
    evaluation_json: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ChatSessionRow(SQLModel, table=True):
    session_id: str = Field(primary_key=True)
    event_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
