from http import HTTPStatus
from typing import Annotated, cast

from fastapi import APIRouter, Depends, HTTPException

from api.deps import get_evidence_retriever, get_gold_retriever
from core.exceptions import RetrievalError, WebScrapingError
from models.evidence import Evidence, GoldEvidence
from models.requests import GoldEvidenceRequest, RetrieveEvidenceRequest
from retrieval.evidence_retriever import EvidenceRetriever
from retrieval.gold_retriever import GoldEvidenceRetriever

router = APIRouter(prefix="/api/v1/evidence", tags=["evidence"])


@router.post("/gold", response_model=GoldEvidence)
async def get_gold_evidence(
    body: GoldEvidenceRequest,
    retriever: Annotated[GoldEvidenceRetriever, Depends(get_gold_retriever)],
) -> GoldEvidence:
    try:
        return retriever.retrieve(str(body.source_url))
    except WebScrapingError as exc:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND, detail="Could not load the source URL"
        ) from exc
    except RetrievalError as exc:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND, detail="No evidence sources found at URL"
        ) from exc


@router.post("/retrieve", response_model=list[Evidence])
async def retrieve_evidence(
    body: RetrieveEvidenceRequest,
    retriever: Annotated[EvidenceRetriever, Depends(get_evidence_retriever)],
) -> list[Evidence]:
    return cast(list[Evidence], retriever.retrieve(body.claim_text, body.claim_date))
