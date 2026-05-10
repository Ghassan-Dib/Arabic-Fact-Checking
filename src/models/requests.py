from datetime import datetime

from pydantic import AnyHttpUrl, BaseModel


class ClaimSearchRequest(BaseModel):
    query: str
    language: str = "ar"
    max_results: int = 20


class GoldEvidenceRequest(BaseModel):
    source_url: AnyHttpUrl


class RetrieveEvidenceRequest(BaseModel):
    claim_text: str
    claim_date: datetime | None = None


class VerifyRequest(BaseModel):
    claim_id: str
    claim_text: str
    evidence_text: str


class EvaluateRequest(BaseModel):
    predicted_path: str
    gold_path: str
