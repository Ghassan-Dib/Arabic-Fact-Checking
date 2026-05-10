from http import HTTPStatus
from typing import Annotated, Any, cast

from fastapi import APIRouter, Depends, HTTPException

from api.deps import get_claim_retriever
from core.exceptions import RetrievalError
from models.requests import ClaimSearchRequest
from retrieval.claim_retriever import ClaimRetriever

router = APIRouter(prefix="/api/v1/claims", tags=["claims"])


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
