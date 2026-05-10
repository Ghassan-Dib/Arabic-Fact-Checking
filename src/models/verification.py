from pydantic import BaseModel

from models.claim import ClaimLabel


class QAPair(BaseModel):
    question: str
    answer: str


class VerificationResult(BaseModel):
    claim_id: str
    qa_pairs: list[QAPair]
    predicted_label: ClaimLabel
