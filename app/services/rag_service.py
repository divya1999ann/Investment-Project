"""
RAG Service — Document chunking, embedding generation, and vector retrieval.

Flow:
  1. ingest_document()  → chunk raw text → generate embeddings → store in pgvector
  2. retrieve_chunks()  → embed a query → cosine similarity search → return top-K chunks
"""

import uuid
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text

from app.core.config import settings
from app.models.models import Document, DocumentChunk


def chunk_text(raw: str, chunk_size: int = settings.CHUNK_SIZE, overlap: int = settings.CHUNK_OVERLAP) -> list[str]:
    """
    Simple sliding-window character chunker.
    In production, swap for a sentence-aware splitter (e.g. LangChain RecursiveCharacterTextSplitter).
    """
    chunks = []
    start = 0
    while start < len(raw):
        end = start + chunk_size
        chunks.append(raw[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]


async def get_embedding(text_input: str) -> Optional[list[float]]:
    """
    Generate an embedding vector for a string.
    Uses OpenAI text-embedding-3-small (1536 dims) if key is set,
    otherwise falls back to a zero vector so the app still runs without an OpenAI key.
    """
    if not settings.OPENAI_API_KEY:
        # Fallback: zero vector (semantic search won't work but app won't crash)
        return [0.0] * settings.VECTOR_DIMENSIONS

    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input=text_input,
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"[RAG] Embedding error: {e}")
        return [0.0] * settings.VECTOR_DIMENSIONS


async def ingest_document(
    db: AsyncSession,
    ticker: str,
    raw_text: str,
    title: str = None,
) -> Document:
    """
    Persist a document and its embedded chunks to PostgreSQL.
    Returns the saved Document ORM object.
    """
    # 1. Save the document
    doc = Document(
        ticker=ticker.upper(),
        title=title or f"{ticker.upper()} Financial Document",
        raw_text=raw_text,
        word_count=len(raw_text.split()),
    )
    db.add(doc)
    await db.flush()  # get doc.id before inserting chunks

    # 2. Chunk the text
    chunks = chunk_text(raw_text)

    # 3. Embed each chunk and save
    for i, chunk_text_str in enumerate(chunks):
        embedding = await get_embedding(chunk_text_str)
        chunk = DocumentChunk(
            document_id=doc.id,
            chunk_index=i,
            text=chunk_text_str,
            embedding=embedding,
        )
        db.add(chunk)

    await db.commit()
    await db.refresh(doc)
    return doc


async def retrieve_chunks(
    db: AsyncSession,
    document_id: uuid.UUID,
    query: str,
    top_k: int = settings.TOP_K_CHUNKS,
) -> list[DocumentChunk]:
    """
    Retrieve the most semantically relevant chunks for a given query using
    pgvector cosine similarity search.
    """
    query_embedding = await get_embedding(query)

    # pgvector cosine distance operator: <=>
    # We order by distance ASC (smallest distance = most similar)
    result = await db.execute(
        text("""
            SELECT id, chunk_index, text,
                   embedding <=> CAST(:embedding AS vector) AS distance
            FROM document_chunks
            WHERE document_id = :doc_id
            ORDER BY distance ASC
            LIMIT :top_k
        """),
        {
            "embedding": str(query_embedding),
            "doc_id": str(document_id),
            "top_k": top_k,
        },
    )
    rows = result.fetchall()

    # Re-fetch as ORM objects
    chunk_ids = [row.id for row in rows]
    chunks_result = await db.execute(
        select(DocumentChunk).where(DocumentChunk.id.in_(chunk_ids))
    )
    return list(chunks_result.scalars().all())
