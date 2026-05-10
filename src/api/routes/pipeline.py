from typing import Annotated, Any, cast

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from api.deps import get_pipeline
from models.pipeline import PipelineConfig, PipelineJobState
from pipeline import job_store
from pipeline.runner import FactCheckingPipeline

router = APIRouter(prefix="/api/v1/pipeline", tags=["pipeline"])


@router.post("/run", response_model=PipelineJobState, status_code=202)
async def run_pipeline(
    config: PipelineConfig,
    background_tasks: BackgroundTasks,
    pipeline: Annotated[FactCheckingPipeline, Depends(get_pipeline)],
) -> PipelineJobState:
    job = job_store.create_job()
    background_tasks.add_task(pipeline.run, job.job_id, config)
    return job


@router.get("/{job_id}/status", response_model=PipelineJobState)
async def get_pipeline_status(job_id: str) -> PipelineJobState:
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/{job_id}/result")
async def get_pipeline_result(job_id: str) -> dict[str, Any]:
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.result is None:
        raise HTTPException(status_code=404, detail="Result not available yet")
    return cast(dict[str, Any], job.result)
