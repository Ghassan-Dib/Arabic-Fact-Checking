from datetime import datetime
from enum import StrEnum

from pydantic import AnyHttpUrl, BaseModel


class ClaimLabel(StrEnum):
    SUPPORTED = "supported"
    REFUTED = "refuted"
    NOT_ENOUGH_EVIDENCE = "Not Enough Evidence"
    CONFLICTING_EVIDENCE = "Conflicting Evidence/Cherrypicking"


class Claim(BaseModel):
    id: str
    text: str
    date: datetime | None = None
    source_url: AnyHttpUrl | None = None
    source_label: str | None = None
    normalized_label: ClaimLabel | None = None
