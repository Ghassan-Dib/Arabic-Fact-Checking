FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir uv

FROM base AS deps
WORKDIR /app
COPY pyproject.toml ./
RUN uv sync --no-dev --no-install-project

FROM deps AS prod
COPY src/ ./src/
ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
