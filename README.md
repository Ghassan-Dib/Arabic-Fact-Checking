# Arabic Fact-Checking

A REST API for verifying Arabic claims using retrieval-augmented generation. The system retrieves evidence from the web, generates QA pairs with an LLM, predicts a verdict, and evaluates against gold-standard datasets.

---

## Tech Stack

| Layer | Technology |
|---|---|
| API framework | FastAPI + Uvicorn |
| LLM | Anthropic Claude (via `agent/` abstraction) |
| Package manager | [uv](https://docs.astral.sh/uv/) |
| Validation | Pydantic v2 |
| Type checking | mypy (strict) |
| Linting / formatting | Ruff |
| Testing | pytest + pytest-asyncio |

---

## Setup

**Requirements:** Python 3.12+, [uv](https://docs.astral.sh/uv/)

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies (including dev tools)
uv sync --group dev

# 3. Configure environment
cp .env.example .env
# Fill in the required values in .env
```

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key |
| `API_KEY` | Yes | Google Fact Check Tools API key |
| `FACT_CHECK_TOOLS_URL` | Yes | Google Fact Check Tools endpoint |
| `CLAUDE_MODEL` | No | Model name (default: `claude-sonnet-4-20250514`) |
| `LOG_LEVEL` | No | Logging level (default: `INFO`) |
| `DATA_DIR` | No | Data directory for evaluation files (default: `data/`) |

---

## Running the Server

```bash
PYTHONPATH=src uv run uvicorn api.app:app --reload
```

Interactive API docs are available at `http://localhost:8000/docs`.

---

## API Endpoints

### Claims

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/claims/search` | Search for claims via Google Fact Check Tools |

**Request body:**
```json
{ "query": "ادعاء للبحث", "language": "ar" }
```

---

### Evidence

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/evidence/retrieve` | Retrieve web evidence for a claim using DuckDuckGo search |
| `POST` | `/api/v1/evidence/gold` | Scrape cited sources from a fact-check article URL |

**Retrieve evidence request:**
```json
{ "claim_text": "نص الادعاء", "claim_date": "2024-01-01" }
```

**Gold evidence request:**
```json
{ "source_url": "https://example.com/fact-check-article" }
```

---

### Verification

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/verify` | Generate QA pairs from evidence and predict a verdict |

**Request body:**
```json
{
  "claim_id": "claim-123",
  "claim_text": "نص الادعاء",
  "evidence_text": "نص الأدلة المسترجعة"
}
```

**Response:**
```json
{
  "claim_id": "claim-123",
  "qa_pairs": [
    { "question": "سؤال؟", "answer": "إجابة" }
  ],
  "predicted_label": "supported"
}
```

Possible labels: `supported`, `refuted`, `Not Enough Evidence`, `Conflicting Evidence/Cherrypicking`

---

### Pipeline

Runs the full fact-checking pipeline as a background job.

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/pipeline/run` | Start a pipeline job |
| `GET` | `/api/v1/pipeline/{job_id}/status` | Poll job status |
| `GET` | `/api/v1/pipeline/{job_id}/result` | Retrieve job results |

**Run request body:**
```json
{
  "collect_claims": true,
  "max_claims": 100,
  "batch_size": 10
}
```

---

### Evaluation

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/evaluate` | Score predictions against gold labels using AVeriTeC/METEOR metrics |

**Request body:**
```json
{
  "predicted_path": "results/predictions.json",
  "gold_path": "data/gold.json"
}
```

Paths are resolved relative to `DATA_DIR` and validated against path traversal.

---

### Health

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/health/ready` | Readiness check (verifies settings are loaded) |

---

## Project Structure

```
src/
├── agent/                  # LLM client abstraction
│   ├── agent.py            # Agent class — wraps a client, exposes run()
│   ├── factory.py          # create_client(config) — selects provider
│   ├── types.py            # Provider, Message, LLMResponse, AgentConfig
│   └── clients/
│       ├── base.py         # Abstract BaseClient
│       ├── anthropic.py    # Anthropic SDK client
│       ├── openai.py       # OpenAI client (optional dep)
│       └── genai.py        # Google GenAI client (optional dep)
│
├── api/
│   ├── app.py              # FastAPI app factory
│   ├── deps.py             # Dependency injection (cached singletons)
│   └── routes/
│       ├── claims.py       # POST /api/v1/claims/search
│       ├── evidence.py     # POST /api/v1/evidence/{retrieve,gold}
│       ├── verify.py       # POST /api/v1/verify
│       ├── pipeline.py     # POST/GET /api/v1/pipeline/...
│       ├── evaluate.py     # POST /api/v1/evaluate
│       └── health.py       # GET /health, /health/ready
│
├── core/
│   ├── config.py           # Settings (pydantic-settings, reads .env)
│   ├── exceptions.py       # Domain exception hierarchy
│   └── logging.py          # Logging configuration
│
├── models/                 # Pydantic schemas (request/response bodies)
├── prompts/                # Arabic LLM prompt templates
│   ├── qa.py               # EVIDENCE_QA_PROMPT, GOLD_QA_PROMPT
│   └── label.py            # LABEL_PROMPT
│
├── pipeline/
│   ├── runner.py           # FactCheckingPipeline orchestrator
│   └── job_store.py        # In-memory job state
│
├── retrieval/
│   ├── claim_retriever.py  # Google Fact Check Tools API
│   ├── evidence_retriever.py # DuckDuckGo search
│   └── gold_retriever.py   # Web scraping for fact-check sources
│
├── verification/
│   ├── qa_generator.py     # Generates QA pairs from evidence
│   └── label_predictor.py  # Predicts claim verdict
│
├── evaluation/             # METEOR / AVeriTeC scoring
└── utils/                  # Date parsing, text processing, web scraping
```

---

## Development

### Quality checks

```bash
# Lint
uv run ruff check src/ tests/

# Format
uv run ruff format src/ tests/

# Type check
PYTHONPATH=src uv run mypy src/

# Tests with coverage
PYTHONPATH=src uv run pytest
```

Pre-commit hooks (ruff + mypy) run automatically on every commit after:

```bash
uv run pre-commit install
```

### Adding a new LLM provider

1. Add a value to `Provider` in [src/agent/types.py](src/agent/types.py)
2. Implement `BaseClient` in a new file under [src/agent/clients/](src/agent/clients/)
3. Add a branch to `create_client()` in [src/agent/factory.py](src/agent/factory.py)

The rest of the system (routes, verification, pipeline) is unaffected.

---

## Architecture Notes

**Layered architecture** — routes translate HTTP ↔ domain, retrieval/verification contain business logic, utils are pure functions. Layers only call downward; no layer skipping.

**Agent abstraction** — `QAGenerator` and `LabelPredictor` depend on `Agent`, not on the Anthropic SDK directly. Swapping providers requires only a config change.

**Dependency injection** — all stateful objects (`ClaimRetriever`, `QAGenerator`, `FactCheckingPipeline`, etc.) are constructed once and cached in `deps.py`, then injected via `Depends()`. No global mutable state.

**Background jobs** — `POST /api/v1/pipeline/run` returns a job ID immediately; the pipeline runs via FastAPI `BackgroundTasks`. Poll `/status` to track progress.
