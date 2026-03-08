"""
Integration tests for the Investment Committee API.
Run with: pytest tests/ -v
"""
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch

from app.main import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


SAMPLE_DOC = """
Apple Inc. Q4 FY2024 Earnings.
Revenue: $94.9B (+6% YoY). Services revenue: $24.2B (+12% YoY).
Gross margin: 46.2%. Operating cash flow: $26.8B.
China revenue declined 2% to $15.0B. P/E ~31x.
"""


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch("app.services.rag_service.get_embedding", new_callable=AsyncMock)
async def test_upload_document(mock_embed, client):
    mock_embed.return_value = [0.0] * 1536

    response = await client.post("/api/v1/documents/", json={
        "ticker": "AAPL",
        "raw_text": SAMPLE_DOC,
        "title": "AAPL Q4 Test",
    })
    assert response.status_code == 201
    data = response.json()
    assert data["ticker"] == "AAPL"
    assert data["word_count"] > 0
    return data["id"]


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch("app.services.rag_service.get_embedding", new_callable=AsyncMock)
@patch("app.services.agent_service.run_committee", new_callable=AsyncMock)
async def test_run_analysis(mock_committee, mock_embed, client):
    mock_embed.return_value = [0.0] * 1536
    mock_committee.return_value = {
        "agents": {
            "graham": {"verdict": "HOLD", "conviction": 6, "reasoning": "Fairly valued.", "bull_case": "Strong cash.", "bear_case": "Premium valuation.", "key_quote": "P/E ~31x", "price_target_commentary": "Near fair value."},
            "wood": {"verdict": "BUY", "conviction": 8, "reasoning": "AI catalyst.", "bull_case": "Services growth.", "bear_case": "China risk.", "key_quote": "Services +12% YoY", "price_target_commentary": "Upside in AI."},
            "risk": {"verdict": "HOLD", "conviction": 5, "reasoning": "Watch China.", "bull_case": "Cash flow strong.", "bear_case": "China declining.", "key_quote": "China -2%", "price_target_commentary": "Neutral."},
        },
        "consensus": {
            "committee_verdict": "HOLD",
            "confidence": "MEDIUM",
            "agreement_level": "MAJORITY",
            "summary": "Committee sees a fairly valued stock with upside optionality.",
            "key_risk": "China revenue decline",
            "key_opportunity": "Services and AI integration",
        }
    }

    # Upload doc first
    doc_resp = await client.post("/api/v1/documents/", json={
        "ticker": "AAPL",
        "raw_text": SAMPLE_DOC,
    })
    doc_id = doc_resp.json()["id"]

    # Run analysis
    response = await client.post("/api/v1/analysis/", json={"document_id": doc_id})
    assert response.status_code == 201
    data = response.json()
    assert data["committee_verdict"] == "HOLD"
    assert data["status"] == "complete"
    assert len(data["agent_analyses"]) == 3
