from datetime import datetime
from typing import Annotated, cast

from fastapi import APIRouter, Depends, HTTPException
from pydantic import AnyHttpUrl, BaseModel

from api.deps import get_evidence_retriever, get_gold_retriever
from models.evidence import Evidence, GoldEvidence
from retrieval.evidence_retriever import EvidenceRetriever
from retrieval.gold_retriever import GoldEvidenceRetriever

router = APIRouter(prefix="/api/v1/evidence", tags=["evidence"])


class GoldEvidenceRequest(BaseModel):
    source_url: AnyHttpUrl


class RetrieveEvidenceRequest(BaseModel):
    claim_text: str
    claim_date: datetime | None = None


@router.post("/gold", response_model=GoldEvidence)
async def get_gold_evidence(
    body: GoldEvidenceRequest,
    retriever: Annotated[GoldEvidenceRetriever, Depends(get_gold_retriever)],
) -> GoldEvidence:
    result = retriever.retrieve(str(body.source_url))
    if result is None:
        raise HTTPException(status_code=404, detail="Could not extract evidence from URL")
    return result


@router.post("/retrieve", response_model=list[Evidence])
async def retrieve_evidence(
    body: RetrieveEvidenceRequest,
    retriever: Annotated[EvidenceRetriever, Depends(get_evidence_retriever)],
) -> list[Evidence]:
    return cast(list[Evidence], retriever.retrieve(body.claim_text, body.claim_date))
