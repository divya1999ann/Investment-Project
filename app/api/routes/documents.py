from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.database import get_db
from app.models.models import Document, DocumentChunk
from app.models.schemas import DocumentCreate, DocumentOut
from app.services.rag_service import ingest_document

router = APIRouter()


@router.post("/", response_model=DocumentOut, status_code=201)
async def upload_document(
    payload: DocumentCreate,
    db: AsyncSession = Depends(get_db),
):
    """
    Ingest a financial document:
    - Persists raw text to PostgreSQL
    - Chunks and embeds the text (RAG layer)
    - Stores embedding vectors via pgvector
    """
    doc = await ingest_document(
        db=db,
        ticker=payload.ticker,
        raw_text=payload.raw_text,
        title=payload.title,
    )

    chunk_count = await db.scalar(
        select(func.count()).where(DocumentChunk.document_id == doc.id)
    )

    return DocumentOut(
        id=doc.id,
        ticker=doc.ticker,
        title=doc.title,
        word_count=doc.word_count,
        chunk_count=chunk_count or 0,
        created_at=doc.created_at,
    )


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    chunk_count = await db.scalar(
        select(func.count()).where(DocumentChunk.document_id == doc.id)
    )
    return DocumentOut(
        id=doc.id,
        ticker=doc.ticker,
        title=doc.title,
        word_count=doc.word_count,
        chunk_count=chunk_count or 0,
        created_at=doc.created_at,
    )


@router.get("/", response_model=list[DocumentOut])
async def list_documents(
    ticker: str = None,
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
):
    query = select(Document).order_by(Document.created_at.desc()).limit(limit)
    if ticker:
        query = query.where(Document.ticker == ticker.upper())
    result = await db.execute(query)
    docs = result.scalars().all()

    out = []
    for doc in docs:
        chunk_count = await db.scalar(
            select(func.count()).where(DocumentChunk.document_id == doc.id)
        )
        out.append(DocumentOut(
            id=doc.id,
            ticker=doc.ticker,
            title=doc.title,
            word_count=doc.word_count,
            chunk_count=chunk_count or 0,
            created_at=doc.created_at,
        ))
    return out
