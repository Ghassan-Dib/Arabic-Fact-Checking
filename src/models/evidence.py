from datetime import datetime

from pydantic import AnyHttpUrl, BaseModel


class Evidence(BaseModel):
    title: str
    url: AnyHttpUrl
    snippet: str | None = None
    published_date: datetime | None = None
    text: str | None = None


class GoldEvidence(BaseModel):
    sources: list[Evidence]
