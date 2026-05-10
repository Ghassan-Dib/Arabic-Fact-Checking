# CLAUDE.md

This file provides coding standards and guidelines for this project.
It is intended for Claude Code (claude.ai/code), PR reviewers, and AI agents working with this codebase.

**Run Ruff locally:**

```bash
uv run ruff check src/ tests/        # Check for issues
uv run ruff check --fix src/ tests/  # Auto-fix issues
uv run ruff format src/ tests/       # Format code
```

See `pyproject.toml` for complete Ruff configuration.

---

## Testing Standards

> **Critical:** 100% test coverage is required for all new code.

```bash
PYTHONPATH=src uv run pytest  # Coverage configured in pyproject.toml
```

### Test Types

| Type            | Purpose                                | What to Mock                             |
| --------------- | -------------------------------------- | ---------------------------------------- |
| **Unit**        | Test individual functions in isolation | All external dependencies                |
| **Integration** | Test component interactions            | Only external boundaries (APIs, network) |
| **E2E**         | Test complete request/response cycles  | External APIs via FastAPI test client    |

### Arrange/Act/Assert Structure (Mandatory)

All tests must use a clear three-section structure with single-line docstrings:

```python
class TestClaimLabel:
    def test_all_values_defined(self) -> None:
        """ClaimLabel enum must cover all four AVeriTeC verdict categories."""
        # Arrange — nothing needed, testing enum constants

        # Act — access enum values
        supported = ClaimLabel.SUPPORTED
        refuted = ClaimLabel.REFUTED

        # Assert
        assert supported == "supported"
        assert refuted == "refuted"
```

**Why:** Enforces single-behavior testing, improves readability.

### Async Test Patterns & Patching

Use `pytest-asyncio` with `@patch` decorators (not context managers):

```python
from unittest.mock import MagicMock, patch
from anthropic.types import TextBlock

class TestQAGenerator:
    @patch("anthropic.Anthropic")
    def test_generate_from_evidence_returns_pairs(self, MockAnthropic: MagicMock) -> None:
        """QAGenerator should parse QA pairs from a valid LLM JSON response."""
        # Arrange
        mock_instance = MagicMock()
        MockAnthropic.return_value = mock_instance
        mock_instance.messages.create.return_value = MagicMock(
            content=[TextBlock(type="text", text='{"qa_pairs": [{"question": "ما هو؟", "answer": "هذا"}]}')]
        )
        generator = QAGenerator(api_key="test", model="claude-test")

        # Act
        pairs = generator.generate_from_evidence("ادعاء", "دليل نصي")

        # Assert
        assert len(pairs) == 1
        assert pairs[0].question == "ما هو؟"
```

**Why decorators:** Cleaner, explicit dependencies in signature, easier to stack.
**When context managers:** Dynamic patching or scope limited to part of a test.

**Configuration:** `asyncio_mode = "auto"` in pyproject.toml

### Mock Naming & Boundaries

- **`mock_`** prefix: Isolating system under test (boundary replacement)
- **`spy_`** prefix: Verifying calls (interaction testing)

**Critical principle: Mock external boundaries, NOT internal logic.**

| Mock These ✅                   | Don't Mock These ❌                  |
| ------------------------------- | ------------------------------------ |
| `requests.get` / Anthropic API  | Pure functions (`parse_arabic_date`) |
| Selenium / DuckDuckGo search    | Pydantic model validation            |
| `find_published_date` (network) | `_extract_json` (internal logic)     |
| `scrape_html` (browser)         | `ClaimLabel` enum                    |

**Example:**

```python
# ✅ Good: Mock the external HTTP boundary only
@patch("requests.get")
def test_retrieve_by_query(mock_get: MagicMock) -> None:
    mock_get.return_value = MagicMock(
        json=lambda: {"claims": [{"text": "ادعاء 1"}]},
        raise_for_status=lambda: None,
    )
    retriever = ClaimRetriever(api_url="https://api.example.com", api_key="key")
    claims = retriever.retrieve_by_query("فلسطين")
    # Tests REAL retry logic, REAL response parsing
    assert len(claims) == 1
```

**Why:** Tests that over-mock become brittle. Mock only system boundaries.

### Fixture Organization & Error Testing

Place shared fixtures in `tests/conftest.py`. Keep test-specific fixtures in test files.

**Always test error paths:** validation errors, not-found cases, upstream failures.

```python
def test_status_404_for_unknown_job(client: TestClient) -> None:
    # Act
    response = client.get("/api/v1/pipeline/nonexistent-id/status")

    # Assert
    assert response.status_code == 404
```

---

## Code Standards

### Exception-Based Error Handling

> **Critical:** Never return `None` to indicate errors. Always raise exceptions.

**Prefer custom exceptions for domain-specific errors:**

```python
# src/core/exceptions.py
class FactCheckerError(Exception):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}

class RetrievalError(FactCheckerError):
    pass

class LLMClientError(FactCheckerError):
    pass
```

**When to use:**

| Custom Exceptions                         | Built-in Exceptions                           |
| ----------------------------------------- | --------------------------------------------- |
| Domain-specific errors (`RetrievalError`) | Pure validation (`ValueError`)                |
| Need to carry context/metadata            | Programming errors (`TypeError`)              |
| Multiple handling strategies              | Standard library errors (`FileNotFoundError`) |

**Why:** Type safety, carries context, explicit error contract.

### Avoiding Assert in Source Code

> **🤖 Ruff S101:** Assert statements are for tests only.

**Why:** Python's `-O` flag disables all assertions, removing validation in production.

```python
# ✅ Correct
if not api_key:
    raise ConfigurationError("ANTHROPIC_API_KEY must be set")

# ❌ Wrong - disabled with -O flag
assert api_key, "ANTHROPIC_API_KEY must be set"
```

### Keyword-Only Arguments

**Use `*` separator for functions with 2+ parameters. Use keywords at call sites with 2+ arguments.**

```python
# ✅ Definition
def __init__(
    self,
    api_url: str,
    api_key: str,
    max_retries: int = 5,
    initial_retry_delay: float = 1.0,
) -> None: ...

# ✅ Call
retriever = ClaimRetriever(
    api_url=settings.fact_check_tools_url,
    api_key=settings.api_key,
)

# ❌ Wrong
retriever = ClaimRetriever("https://...", "key123")  # Which is url vs key?
```

**Why:** Prevents argument order bugs, self-documenting, refactoring-safe.

### Type Hints

All functions must have complete type hints using Python 3.12+ syntax (`str | None` not `Optional[str]`).

**Why:** Static type checking, IDE autocomplete, documents expected types.

### Resource Cleanup Patterns

Always use context managers for resources that need cleanup:

```python
# ✅ Selenium driver with guaranteed cleanup
options = Options()
options.add_argument("--headless")
driver = webdriver.Chrome(options=options)
try:
    driver.get(url)
    html = driver.page_source
finally:
    driver.quit()
```

**Why:** Prevents resource leaks, ensures cleanup on errors.

### Client Configuration Standards

> **Critical:** All external API clients must have explicit configuration.

**Required:**

- ✅ **Timeout**: Always set explicit timeout (prevents hangs)
- ✅ **User-Agent**: Identify your service
- ✅ **API key**: Passed explicitly, never hardcoded

```python
# ✅ Good
resp = requests.get(
    self.api_url,
    params={"query": query, "key": self.api_key},
    timeout=30,
)

# ❌ Bad - no timeout, hangs indefinitely
resp = requests.get(self.api_url, params={"query": query})
```

### Retry Logic for External APIs

Use manual exponential backoff (the project's existing pattern):

```python
delay = self.initial_retry_delay
for attempt in range(self.max_retries):
    try:
        resp = requests.get(self.api_url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json().get("claims", [])
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code < 500:
            break  # Don't retry 4xx
    if attempt < self.max_retries - 1:
        time.sleep(delay)
        delay *= 2
raise RetrievalError(f"Failed after {self.max_retries} attempts")
```

**Retry:** Network timeouts, 5xx errors, 503 service unavailable.
**Don't retry:** 4xx client errors (except 408/429).

### Early Returns & Pydantic Validation

> **🤖 Ruff RET:** Use guard clauses instead of deep nesting.

```python
# ✅ Good: Guard clauses
def retrieve(self, source_url: str) -> GoldEvidence | None:
    soup, _ = scrape_html(source_url)
    if not soup:
        logger.warning("Failed to scrape %s", source_url)
        return None
    # Happy path continues unindented...
```

**Pydantic validation:**

```python
class PipelineConfig(BaseModel):
    batch_size: int = Field(default=10, ge=1, le=100)
    max_claims: int | None = None
    collect_claims: bool = True
```

---

## Documentation Standards

### Google-Style Docstrings

Public functions must have Google-style docstrings:

```python
def retrieve(self, source_url: str) -> GoldEvidence | None:
    """Scrape a fact-check article page and extract its cited source URLs.

    Args:
        source_url: URL of the fact-check article to scrape.

    Returns:
        GoldEvidence with extracted sources, or None if scraping fails.

    Raises:
        WebScrapingError: If the page cannot be loaded after retries.
    """
```

### Code Comments & TODO Policy

Write self-documenting code. Add comments only when the "why" isn't obvious.

> **🤖 Ruff FIX001:** Never commit TODO or FIXME comments. Create GitHub issues instead.

---

## Code Organization

### Module Structure: Layered Architecture

> **Rule:** Routes → Pipeline/Verification → Retrieval → Utils (no layer skipping)

| Layer            | Location            | Responsibility                                              |
| ---------------- | ------------------- | ----------------------------------------------------------- |
| **Routes**       | `src/api/routes/`   | HTTP layer, translate domain exceptions to `HTTPException`  |
| **Pipeline**     | `src/pipeline/`     | Orchestration, job state management                         |
| **Verification** | `src/verification/` | LLM calls: QA generation, label prediction                  |
| **Retrieval**    | `src/retrieval/`    | External data: claims API, web search, scraping             |
| **Utils**        | `src/utils/`        | Pure functions: text processing, date parsing, web scraping |
| **Models**       | `src/models/`       | Pydantic schemas for validation and serialization           |

**Example:**

```python
# Route — translates errors to HTTP
@router.post("/gold", response_model=GoldEvidence)
async def get_gold_evidence(
    body: GoldEvidenceRequest,
    retriever: Annotated[GoldEvidenceRetriever, Depends(get_gold_retriever)],
) -> GoldEvidence:
    result = retriever.retrieve(str(body.source_url))
    if result is None:
        raise HTTPException(status_code=404, detail="Could not extract evidence from URL")
    return result

# Retrieval — domain logic, raises domain exceptions
def retrieve(self, source_url: str) -> GoldEvidence | None:
    soup, _ = scrape_html(source_url)
    if not soup:
        return None
    raw = extract_sources(soup)
    return GoldEvidence(sources=[Evidence(title=s["name"], url=s["url"]) for s in raw])

# Utils — pure function, no HTTP concerns
def extract_sources(soup: BeautifulSoup) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    h = soup.find(lambda tag: tag.get_text(strip=True) == "المصدر")
    if isinstance(h, Tag):
        for a in h.find_next_sibling("div").find_all("a", href=True):
            sources.append({"name": a.get_text(strip=True), "url": str(a["href"])})
    return sources
```

**Why:** Clear separation of concerns, testability, reusability.

### Error Translation at Layer Boundaries

> **Critical:** Each layer translates errors to its abstraction level.

**Rules:**

- **Routes**: Catch domain exceptions → `HTTPException` with status codes
- **Pipeline/Verification/Retrieval**: Raise domain-specific custom exceptions (never `HTTPException`)
- **Utils**: Raise built-in or domain exceptions
- **Always use exception chaining**: `raise ... from exc`

```python
# ✅ Route translates domain errors to HTTP
@router.post("/search")
async def search_claims(
    body: ClaimSearchRequest,
    retriever: Annotated[ClaimRetriever, Depends(get_claim_retriever)],
) -> list[dict[str, Any]]:
    try:
        return cast(list[dict[str, Any]], retriever.retrieve_by_query(body.query))
    except RetrievalError as exc:
        raise HTTPException(status_code=502, detail="Claim retrieval service unavailable") from exc
```

**Why:** Separates concerns, prevents HTTP leaking into business logic, preserves error context.

### Pipeline Orchestration

The pipeline orchestrates steps and delegates detail to retrieval/verification layers:

```python
def run(self, job_id: str, config: PipelineConfig) -> None:
    job_store.update_job(job_id, status=JobStatus.RUNNING)
    try:
        results: dict[str, Any] = {}
        if config.collect_claims:
            job_store.update_job(job_id, current_step="collect_claims")
            claims = []
            for query in QUERIES:
                claims.extend(self.claim_retriever.retrieve_by_query(query))
            results["claims_count"] = len(claims)
            logger.info("Collected %d claims", len(claims))
        job_store.complete_job(job_id, results)
    except Exception as exc:
        logger.exception("Pipeline job %s failed", job_id)
        job_store.fail_job(job_id, str(exc))
```

**Pattern:** Public methods coordinate workflow, private helpers (`_` prefix) organize sub-steps.

### Boolean Naming

Use standard prefixes (`is_`, `has_`, `can_`, `should_`, `needs_`, `will_`)

```python
def is_error_page(soup: BeautifulSoup) -> bool:
    text = soup.get_text().lower()
    return (
        "404" in text
        or "page not found" in text
        or "الصفحة غير موجودة" in text
    )
```

### When to Extract Abstractions

> **Rule of Three:** Don't abstract until you have 3 implementations.

**Evolution:** Inline → Copy-paste-modify → Extract on third occurrence

**Exceptions:** Obviously reusable utility, complex algorithm, external library wrapper.

**Why:** Premature abstraction makes code harder to change. Three implementations reveal the true pattern.

### Constants & File Naming

Extract values used 2+ times into module-level constants:

```python
# src/retrieval/evidence_retriever.py
_SLEEP_BETWEEN_CLAIMS = 2.0
_MAX_RETRIES = 3
_BACKOFF_FACTOR = 2
```

**File naming:**

- **Modules**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`

---

## Quality Assurance

### Required Tools

```bash
uv run ruff check src/ tests/    # Linting
uv run ruff format src/ tests/   # Formatting
PYTHONPATH=src uv run mypy src/  # Type checking
PYTHONPATH=src uv run pytest     # Tests with coverage
```

### Pre-Commit Hooks & CI/CD

Pre-commit hooks (ruff + mypy) run automatically before every commit:

```bash
uv sync --group dev   # installs pre-commit
uv run pre-commit install
```

**CI/CD requirements:** All tests pass, coverage ≥ 50%, Ruff clean, mypy clean.

### Security

Never commit secrets. All credentials live in `.env` and are read via `Settings`:

```python
class Settings(BaseSettings):
    anthropic_api_key: str   # ANTHROPIC_API_KEY in .env
    api_key: str             # API_KEY (Google Fact Check Tools)
    fact_check_tools_url: str
    model_config = SettingsConfigDict(env_file=".env")
```

---

## API Patterns (FastAPI-Specific)

### Router Organization & Models

```python
router = APIRouter(prefix="/api/v1/evidence", tags=["evidence"])

@router.post("/gold", response_model=GoldEvidence, responses={
    404: {"description": "Could not extract evidence from the given URL"},
})
async def get_gold_evidence(
    body: GoldEvidenceRequest,
    retriever: Annotated[GoldEvidenceRetriever, Depends(get_gold_retriever)],
) -> GoldEvidence:
```

**Why:** Organizes endpoints logically, automatic OpenAPI schema generation at `/docs`.

### Error Handling & Dependency Injection

> **Rule:** Never use global state. Use FastAPI's `Depends()`.

```python
# src/api/deps.py
def get_claim_retriever(settings: SettingsDep) -> ClaimRetriever:
    return ClaimRetriever(
        api_url=settings.fact_check_tools_url,
        api_key=settings.api_key,
    )

# Route
async def search_claims(
    body: ClaimSearchRequest,
    retriever: Annotated[ClaimRetriever, Depends(get_claim_retriever)],
) -> list[dict[str, Any]]:
```

**Benefits:** Testable, proper lifecycle, no shared state.

### Structured Logging

> **🤖 Ruff G004:** Use `%` formatting, not f-strings in logging calls.

```python
# ✅ Correct
logger.info("Collected %d claims for query: %s", len(claims), query)
logger.warning("Service unavailable, retry in %.1fs", delay)

# ❌ Wrong
logger.info(f"Collected {len(claims)} claims for query: {query}")
```

**Why:** `%` formatting delays interpolation until the log record is actually emitted.

### Logging Strategy

**Log at the right layer with the right level:**

| Layer               | What to Log                                   |
| ------------------- | --------------------------------------------- |
| **Pipeline**        | Workflow steps, step completion, claim counts |
| **Routes**          | Errors only (with `logger.exception()`)       |
| **Retrieval/Utils** | Warnings for recoverable failures             |

**Example:**

```python
def run(self, job_id: str, config: PipelineConfig) -> None:
    job_store.update_job(job_id, status=JobStatus.RUNNING)

    if config.collect_claims:
        job_store.update_job(job_id, current_step="collect_claims")
        # ... collect ...
        logger.info("Collected %d claims", len(claims))

    job_store.complete_job(job_id, results)
```

**Why:** Pipeline layer provides workflow visibility. Routes log HTTP errors. Retrieval logs warnings.

---

## Development Workflow

### Setup

1. Install Python 3.12+
2. Install UV: `curl -LsSf https://astral.sh/uv/install.sh | sh`
3. `uv sync --group dev` to install all dependencies
4. Copy `.env.example` to `.env` and fill in keys
5. `PYTHONPATH=src uv run uvicorn api.app:app --reload` to start server

### Change Process

1. Create branch from `main`
2. Write tests first (TDD)
3. Implement feature/fix
4. `PYTHONPATH=src uv run pytest`
5. `uv run ruff check src/ tests/ && uv run ruff format src/ tests/`
6. `PYTHONPATH=src uv run mypy src/`
7. Commit (pre-commit hooks run automatically)
8. Push and create PR

### Conventional Commits

```
<type>(<scope>): <description>

[optional body]
```

**Types:** `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`

**Examples:**

```
feat(pipeline): add evidence retrieval step to background job
fix(retrieval): handle 503 from Google Fact Check Tools API
feat(verification): use TextBlock isinstance check for Anthropic response
```

### PR Size Guidelines

> **Critical:** Keep PRs small and focused for effective review.

**Thresholds:** 400 lines, 10 files, 1 feature per PR

**Split when:**

- Multiple features → Separate PRs
- Refactor + feature → Refactor first, feature second
- Infrastructure + feature → Infrastructure first

**Why:** Faster review, easier to spot bugs, safer to revert.

---

## Additional Guidelines

### Configuration Management

**Pydantic Settings** (already in use — follow this pattern):

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    anthropic_api_key: str
    api_key: str
    fact_check_tools_url: str
    claude_model: str = "claude-sonnet-4-20250514"
    default_language: str = "ar"
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

@lru_cache
def get_settings() -> Settings:
    return Settings()
```

**Never** add new bare `os.getenv()` calls — always extend `Settings`.

### Error Messages

Error messages should be user-friendly and not leak internal details:

```python
# ✅ Good
detail="Could not extract evidence from URL"

# ❌ Bad - leaks internal library names
detail=f"BeautifulSoup failed to parse: {exc}"
```

---

## Manual Review Required

**Testing:**

- Arrange/Act/Assert with single-line docstrings
- Mock external boundaries (`requests.get`, Anthropic API, Selenium)
- Use real `TextBlock(type="text", text=...)` in Anthropic response mocks

**Architecture:**

- Layered: Routes → Pipeline/Verification/Retrieval → Utils
- Error translation at boundaries (domain exceptions → `HTTPException` in routes)
- Dependency injection via `Depends()`, no global state

**Code Design:**

- Custom exceptions (`RetrievalError`, `LLMClientError`, `VerificationError`)
- Keyword arguments at call sites with 2+ args
- `isinstance(tag, Tag)` guards before calling BeautifulSoup methods

**External Integration:**

- Explicit timeouts on all `requests.get` calls
- Exponential backoff for retrieval retries
- `isinstance(b, TextBlock)` before accessing `.text` on Anthropic response blocks

**Documentation:**

- Google-style docstrings on public methods
- Complete type hints (Python 3.12+ syntax: `str | None`)
- Pipeline layer logs workflow steps, routes log errors only
