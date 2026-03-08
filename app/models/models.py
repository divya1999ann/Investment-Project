import uuid
from datetime import datetime
from sqlalchemy import String, Text, DateTime, ForeignKey, Integer, Float, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector

from app.core.database import Base
from app.core.config import settings


class Document(Base):
    """A financial document submitted for analysis (earnings report, 10-K, etc.)"""
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=True)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    word_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    chunks: Mapped[list["DocumentChunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")
    sessions: Mapped[list["AnalysisSession"]] = relationship(back_populates="document")


class DocumentChunk(Base):
    """A chunked + embedded slice of a document — the RAG layer"""
    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)

    # pgvector column — stores the embedding for semantic search
    embedding: Mapped[list[float]] = mapped_column(
        Vector(settings.VECTOR_DIMENSIONS), nullable=True
    )

    # Relationships
    document: Mapped["Document"] = relationship(back_populates="chunks")


class AnalysisSession(Base):
    """One full committee analysis run for a document"""
    __tablename__ = "analysis_sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id"), index=True)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending | running | complete | failed
    committee_verdict: Mapped[str] = mapped_column(String(10), nullable=True)
    consensus: Mapped[dict] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)

    # Relationships
    document: Mapped["Document"] = relationship(back_populates="sessions")
    agent_analyses: Mapped[list["AgentAnalysis"]] = relationship(back_populates="session", cascade="all, delete-orphan")


class AgentAnalysis(Base):
    """Individual analysis produced by one AI agent for a session"""
    __tablename__ = "agent_analyses"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("analysis_sessions.id", ondelete="CASCADE"), index=True)
    agent_id: Mapped[str] = mapped_column(String(50), nullable=False)   # graham | wood | risk
    agent_name: Mapped[str] = mapped_column(String(100), nullable=False)
    verdict: Mapped[str] = mapped_column(String(10), nullable=True)     # BUY | HOLD | SELL
    conviction: Mapped[int] = mapped_column(Integer, nullable=True)     # 1–10
    result: Mapped[dict] = mapped_column(JSON, nullable=True)           # full structured output
    chunks_used: Mapped[list] = mapped_column(JSON, nullable=True)      # chunk IDs used for this analysis
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    session: Mapped["AnalysisSession"] = relationship(back_populates="agent_analyses")
