import uuid
from datetime import UTC, datetime

from models.pipeline import JobStatus, PipelineJobState

# In-memory store: job state is lost on restart and not shared across multiple
# uvicorn workers (--workers N). Replace with Redis or SQLite before running
# in a multi-worker or production environment.
_jobs: dict[str, PipelineJobState] = {}


def create_job() -> PipelineJobState:
    job = PipelineJobState(job_id=str(uuid.uuid4()), created_at=datetime.now(UTC))
    _jobs[job.job_id] = job
    return job


def get_job(job_id: str) -> PipelineJobState | None:
    return _jobs.get(job_id)


def update_job(job_id: str, **kwargs: object) -> None:
    job = _jobs.get(job_id)
    if job is None:
        return
    updated = job.model_copy(update=kwargs)
    _jobs[job_id] = updated


def complete_job(job_id: str, result: dict[str, object]) -> None:
    update_job(
        job_id,
        status=JobStatus.COMPLETED,
        completed_at=datetime.now(UTC),
        current_step=None,
        result=result,
    )


def fail_job(job_id: str, error: str) -> None:
    update_job(
        job_id,
        status=JobStatus.FAILED,
        completed_at=datetime.now(UTC),
        current_step=None,
        error=error,
    )
