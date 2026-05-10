import logging
import time
from datetime import datetime

from models.evidence import Evidence
from utils.date_utils import find_published_date

logger = logging.getLogger(__name__)

_SLEEP_BETWEEN_CLAIMS = 2.0
_MAX_RETRIES = 3
_BACKOFF_FACTOR = 2


class EvidenceRetriever:
    def __init__(self, max_results: int = 10, sleep: float = _SLEEP_BETWEEN_CLAIMS) -> None:
        self.max_results = max_results
        self.sleep = sleep

    def retrieve(self, claim_text: str, claim_date: datetime | None = None) -> list[Evidence]:
        """Search DuckDuckGo and return only results published before claim_date."""
        try:
            from duckduckgo_search import DDGS
        except ImportError as exc:
            raise ImportError("duckduckgo-search is required for evidence retrieval") from exc

        results: list[Evidence] = []
        retries = 0
        search = DDGS()

        while retries < _MAX_RETRIES:
            try:
                hits: list[dict[str, str]] = list(
                    search.text(claim_text, max_results=self.max_results)
                )
                for hit in hits:
                    url: str = hit.get("href", "")
                    if not url:
                        continue

                    pub_date = find_published_date(url)
                    if claim_date and pub_date and pub_date >= claim_date:
                        continue

                    results.append(
                        Evidence(
                            title=hit.get("title", ""),
                            url=url,
                            snippet=hit.get("body"),
                            published_date=pub_date,
                        )
                    )
                break
            except Exception as exc:
                retries += 1
                wait = _BACKOFF_FACTOR**retries
                logger.warning("DuckDuckGo error (%s), retry %d in %ds", exc, retries, wait)
                time.sleep(wait)

        time.sleep(self.sleep)
        return results
