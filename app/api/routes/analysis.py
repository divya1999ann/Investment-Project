from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.models.models import Document, AnalysisSession, AgentAnalysis
from app.models.schemas import AnalysisRequest, AnalysisSessionOut, AgentAnalysisOut
from app.services.rag_service import retrieve_chunks
from app.services.agent_service import run_committee, AGENTS

router = APIRouter()


@router.post("/", response_model=AnalysisSessionOut, status_code=201)
async def run_analysis(
    payload: AnalysisRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Trigger a full committee analysis for a previously ingested document.

    Pipeline:
      1. Load document from PostgreSQL
      2. Retrieve top-K semantically relevant chunks per agent (RAG)
      3. Run all three agents concurrently via Claude API
      4. Generate consensus verdict
      5. Persist results
    """
    # 1. Fetch document
    result = await db.execute(select(Document).where(Document.id == payload.document_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found. Ingest it first via POST /documents")

    # 2. Create session
    session = AnalysisSession(
        document_id=doc.id,
        ticker=doc.ticker,
        status="running",
    )
    db.add(session)
    await db.flush()

    try:
        # 3. Retrieve relevant chunks (RAG)
        #    Query with a general finance prompt so we get broadly relevant chunks
        rag_query = f"investment analysis {doc.ticker} revenue earnings risk valuation"
        chunks = await retrieve_chunks(db, doc.id, rag_query)
        chunk_texts = [c.text for c in chunks] if chunks else [doc.raw_text[:3000]]

        # 4. Run the multi-agent committee
        committee_result = await run_committee(
            ticker=doc.ticker,
            context_chunks=chunk_texts,
        )

        # 5. Persist agent analyses
        agent_data = committee_result.get("agents", {})
        for agent_def in AGENTS:
            r = agent_data.get(agent_def["id"], {})
            agent_analysis = AgentAnalysis(
                session_id=session.id,
                agent_id=agent_def["id"],
                agent_name=agent_def["name"],
                verdict=r.get("verdict"),
                conviction=r.get("conviction"),
                result=r,
                chunks_used=[str(c.id) for c in chunks],
            )
            db.add(agent_analysis)

        # 6. Update session with consensus
        consensus = committee_result.get("consensus", {})
        session.status = "complete"
        session.committee_verdict = consensus.get("committee_verdict")
        session.consensus = consensus
        session.completed_at = datetime.utcnow()

    except Exception as e:
        session.status = "failed"
        await db.commit()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    await db.commit()
    await db.refresh(session)

    # Build response
    agent_analyses_out = [
        AgentAnalysisOut(
            agent_id=agent_def["id"],
            agent_name=agent_def["name"],
            verdict=agent_data.get(agent_def["id"], {}).get("verdict"),
            conviction=agent_data.get(agent_def["id"], {}).get("conviction"),
            result=agent_data.get(agent_def["id"]),
        )
        for agent_def in AGENTS
    ]

    return AnalysisSessionOut(
        id=session.id,
        ticker=session.ticker,
        status=session.status,
        committee_verdict=session.committee_verdict,
        consensus=session.consensus,
        agent_analyses=agent_analyses_out,
        created_at=session.created_at,
        completed_at=session.completed_at,
    )


@router.get("/{session_id}", response_model=AnalysisSessionOut)
async def get_analysis(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(AnalysisSession).where(AnalysisSession.id == session_id)
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Analysis session not found")

    agent_rows = await db.execute(
        select(AgentAnalysis).where(AgentAnalysis.session_id == session.id)
    )
    agent_analyses = agent_rows.scalars().all()

    return AnalysisSessionOut(
        id=session.id,
        ticker=session.ticker,
        status=session.status,
        committee_verdict=session.committee_verdict,
        consensus=session.consensus,
        agent_analyses=[
            AgentAnalysisOut(
                agent_id=a.agent_id,
                agent_name=a.agent_name,
                verdict=a.verdict,
                conviction=a.conviction,
                result=a.result,
            )
            for a in agent_analyses
        ],
        created_at=session.created_at,
        completed_at=session.completed_at,
    )
