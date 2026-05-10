from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from api.deps import get_label_predictor, get_qa_generator
from core.exceptions import LLMClientError
from models.requests import VerifyRequest
from models.verification import QAPair, VerificationResult
from verification.label_predictor import LabelPredictor
from verification.qa_generator import QAGenerator

router = APIRouter(prefix="/api/v1", tags=["verification"])


@router.post("/verify", response_model=VerificationResult)
async def verify_claim(
    body: VerifyRequest,
    qa_gen: Annotated[QAGenerator, Depends(get_qa_generator)],
    predictor: Annotated[LabelPredictor, Depends(get_label_predictor)],
) -> VerificationResult:
    try:
        qa_pairs: list[QAPair] = qa_gen.generate_from_evidence(
            claim=body.claim_text,
            evidence_text=body.evidence_text,
        )
        evidence_for_label = (
            "\n".join(f"Q: {qa.question}\nA: {qa.answer}" for qa in qa_pairs) or body.evidence_text
        )
        label = predictor.predict(body.claim_text, evidence_for_label)
    except LLMClientError as exc:
        raise HTTPException(
            status_code=HTTPStatus.BAD_GATEWAY,
            detail="LLM service unavailable",
        ) from exc

    return VerificationResult(
        claim_id=body.claim_id,
        qa_pairs=qa_pairs,
        predicted_label=label,
    )
