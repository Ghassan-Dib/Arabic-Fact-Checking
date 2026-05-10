from typing import Annotated, Any, cast

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.deps import get_claim_retriever
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
    return cast(
        list[dict[str, Any]], retriever.retrieve_by_query(body.query, languageCode=body.language)
    )
