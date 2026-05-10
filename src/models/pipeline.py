from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineConfig(BaseModel):
    max_claims: int | None = None
    batch_size: int = Field(default=10, ge=1, le=100)
    collect_claims: bool = True
    extract_gold_evidence: bool = True
    retrieve_evidence: bool = True
    predict_labels: bool = True
    evaluate_results: bool = True


class PipelineJobState(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.QUEUED
    current_step: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
