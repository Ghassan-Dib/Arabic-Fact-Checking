from http import HTTPStatus
from typing import Annotated, Any, cast

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.deps import get_claim_retriever
from core.exceptions import RetrievalError
from retrieval.claim_retriever import ClaimRetriever

router = APIRouter(prefix="/api/v1/claims", tags=["claims"])


class ClaimSearchRequest(BaseModel):
    query: str
    language: str = "ar"
    max_results: int = 20


@router.post("/search")
async def search_claims(
    body: ClaimSearchRequest,
    retriever: Annotated[ClaimRetriever, Depends(get_claim_retriever)],
) -> list[dict[str, Any]]:
    try:
        return cast(
            list[dict[str, Any]],
            retriever.retrieve_by_query(body.query, languageCode=body.language),
        )
    except RetrievalError as exc:
        raise HTTPException(
            status_code=HTTPStatus.BAD_GATEWAY,
            detail="Claim retrieval service unavailable",
        ) from exc
