# Investment Committee — Backend API

Multi-agent AI financial analysis grounded in documents (RAG).  
Built with **FastAPI · PostgreSQL · pgvector · OpenAI API**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (React)                      │
│              POST /documents  →  POST /analysis              │
└────────────────────────────┬────────────────────────────────┘
                             │ HTTP
┌────────────────────────────▼────────────────────────────────┐
│                     FastAPI (Python)                         │
│                                                              │
│  POST /api/v1/documents/          POST /api/v1/analysis/     │
│  ┌──────────────────────┐         ┌──────────────────────┐  │
│  │   RAG Service        │         │   Agent Service       │  │
│  │  1. Chunk text       │         │  1. Retrieve chunks   │  │
│  │  2. Embed chunks     │         │  2. Run agents (×3)   │  │
│  │  3. Store vectors    │         │     in parallel       │  │
│  └──────────┬───────────┘         │  3. Generate consensus│  │
│             │                     └──────────┬────────────┘  │
└─────────────┼──────────────────────────────-─┼───────────────┘
              │                                │
┌─────────────▼────────────────────────────────▼───────────────┐
│               PostgreSQL + pgvector                           │
│                                                               │
│  documents          document_chunks        analysis_sessions  │
│  ┌──────────┐       ┌──────────────┐       ┌──────────────┐  │
│  │id        │──1:N─▶│id            │       │id            │  │
│  │ticker    │       │document_id   │       │document_id   │  │
│  │raw_text  │       │text          │       │consensus     │  │
│  │word_count│       │embedding     │       │verdict       │  │
│  └──────────┘       │  (vector)    │       └──────┬───────┘  │
│                     └──────────────┘              │1:N        │
│                                               ┌───▼────────┐  │
│                         pgvector cosine       │agent_      │  │
│                         similarity search  ──▶│analyses    │  │
│                                               └────────────┘  │
└───────────────────────────────────────────────────────────────┘
                             │
              ┌──────────────▼──────────────┐
              │       OpenAI API             │
              │  gpt-4o-mini                 │
              │  · Graham agent              │
              │  · Wood agent                │
              │  · Risk agent                │
              │  · Consensus synthesis       │
              └─────────────────────────────┘
```

---

## Quick Start

```bash
# 1. Clone and enter project
git clone <repo>
cd investment-committee

# 2. Set up environment
cp .env.example .env
# Edit .env — add your OPENAI_API_KEY

# 3. Start PostgreSQL + API
docker-compose up --build

# API is live at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

---

## API Usage

### 1. Ingest a document
```bash
curl -X POST http://localhost:8000/api/v1/documents/ \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "title": "AAPL Q4 FY2024 Earnings",
    "raw_text": "Revenue: $94.9B (+6% YoY)..."
  }'
# Returns: { "id": "uuid", "chunk_count": 4, ... }
```

### 2. Run the committee
```bash
curl -X POST http://localhost:8000/api/v1/analysis/ \
  -H "Content-Type: application/json" \
  -d '{ "document_id": "<uuid from step 1>" }'
# Returns: committee verdict + 3 agent analyses
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.12, FastAPI, Uvicorn |
| ORM | SQLAlchemy 2.0 (async) |
| Database | PostgreSQL 16 |
| Vector Search | pgvector (cosine similarity) |
| Embeddings | OpenAI text-embedding-3-small |
| LLM | OpenAI GPT-4o-mini |
| Containers | Docker, Docker Compose |
| Testing | pytest, pytest-asyncio |

---

## Frontend Connection

The React frontend (`ai-investment-committee.jsx`) calls this backend via a typed `api` client at the top of the file:

```
POST /api/v1/documents/  ← Step 1: ingest + chunk + embed the document
POST /api/v1/analysis/   ← Step 2: run 3 agents + consensus
GET  /health             ← Connection status indicator in header
```

To connect:
1. Start the backend: `docker-compose up --build`
2. Open the React app (Claude artifact or local Vite dev server)
3. The header will show **BACKEND CONNECTED** in green when live
4. The **CONVENE COMMITTEE** button is disabled while backend is offline

To change the backend URL (e.g. for a deployed instance):
```js
// ai-investment-committee.jsx, line 7
const BASE_URL = "https://your-deployed-api.com";
```

---

## Running Tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

---

## Project Structure

```
investment-committee/
├── app/
│   ├── main.py                 # FastAPI app + lifespan
│   ├── core/
│   │   ├── config.py           # Settings (pydantic-settings)
│   │   └── database.py         # Async SQLAlchemy engine
│   ├── models/
│   │   ├── models.py           # ORM models (Document, Chunk, Session, Agent)
│   │   └── schemas.py          # Pydantic request/response schemas
│   ├── services/
│   │   ├── rag_service.py      # Chunking, embedding, vector retrieval
│   │   └── agent_service.py    # Multi-agent orchestration
│   └── api/routes/
│       ├── documents.py        # /documents endpoints
│       ├── analysis.py         # /analysis endpoints
│       └── health.py           # /health endpoint
├── tests/
│   └── test_analysis.py
├── scripts/
│   └── init_db.sql             # Enables pgvector extension
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```
