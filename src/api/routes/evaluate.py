from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from evaluation.evaluator import evaluate_from_files
from models.evaluation import EvaluationResult

router = APIRouter(prefix="/api/v1", tags=["evaluation"])


class EvaluateRequest(BaseModel):
    predicted_path: str
    gold_path: str


@router.post("/evaluate", response_model=EvaluationResult)
async def evaluate(body: EvaluateRequest) -> EvaluationResult:
    p, g = Path(body.predicted_path), Path(body.gold_path)
    if not p.exists():
        raise HTTPException(status_code=400, detail=f"predicted_path not found: {p}")
    if not g.exists():
        raise HTTPException(status_code=400, detail=f"gold_path not found: {g}")
    return evaluate_from_files(p, g)
