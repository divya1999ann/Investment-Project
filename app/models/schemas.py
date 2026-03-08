import uuid
from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

class DocumentCreate(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10, example="AAPL")
    raw_text: str = Field(..., min_length=50, description="Financial document text")
    title: Optional[str] = Field(None, example="AAPL Q4 FY2024 Earnings")


class DocumentOut(BaseModel):
    id: uuid.UUID
    ticker: str
    title: Optional[str]
    word_count: int
    chunk_count: int = 0
    created_at: datetime

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

class AnalysisRequest(BaseModel):
    document_id: uuid.UUID = Field(..., description="ID of a previously ingested document")


class AgentAnalysisOut(BaseModel):
    agent_id: str
    agent_name: str
    verdict: Optional[str]
    conviction: Optional[int]
    result: Optional[dict[str, Any]]


class AnalysisSessionOut(BaseModel):
    id: uuid.UUID
    ticker: str
    status: str
    committee_verdict: Optional[str]
    consensus: Optional[dict[str, Any]]
    agent_analyses: list[AgentAnalysisOut] = []
    created_at: datetime
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthOut(BaseModel):
    status: str
    version: str
    environment: str
