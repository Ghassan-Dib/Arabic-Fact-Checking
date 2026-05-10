from pathlib import Path

from fastapi import APIRouter, HTTPException

from api.deps import SettingsDep
from evaluation.evaluator import evaluate_from_files
from models.evaluation import EvaluationResult
from models.requests import EvaluateRequest

router = APIRouter(prefix="/api/v1", tags=["evaluation"])


def _safe_path(raw: str, allowed_dir: Path) -> Path:
    resolved = (allowed_dir / raw).resolve()
    if not resolved.is_relative_to(allowed_dir.resolve()):
        raise HTTPException(status_code=400, detail="Path is outside the allowed data directory")
    return resolved


@router.post("/evaluate", response_model=EvaluationResult)
async def evaluate(body: EvaluateRequest, settings: SettingsDep) -> EvaluationResult:
    allowed = settings.data_dir
    p = _safe_path(body.predicted_path, allowed)
    g = _safe_path(body.gold_path, allowed)
    if not p.exists():
        raise HTTPException(status_code=400, detail=f"predicted_path not found: {p.name}")
    if not g.exists():
        raise HTTPException(status_code=400, detail=f"gold_path not found: {g.name}")
    return evaluate_from_files(p, g)
