import logging
import time
from typing import Any

import requests

from core.exceptions import RetrievalError

logger = logging.getLogger(__name__)


class ClaimRetriever:
    def __init__(
        self,
        api_url: str,
        api_key: str,
        max_retries: int = 5,
        initial_retry_delay: float = 1.0,
    ) -> None:
        self.api_url = api_url
        self.api_key = api_key
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay

    def _is_service_unavailable(self, data: dict[str, Any]) -> bool:
        return (
            "error" in data and isinstance(data["error"], dict) and data["error"].get("code") == 503
        )

    def query_api(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        delay = self.initial_retry_delay
        last_exc: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                resp = requests.get(self.api_url, params=params, timeout=30)
                resp.raise_for_status()
                data: dict[str, Any] = resp.json()

                if self._is_service_unavailable(data):
                    if attempt < self.max_retries - 1:
                        logger.warning("Service unavailable, retry in %.1fs", delay)
                        time.sleep(delay)
                        delay *= 2
                        continue
                    raise RetrievalError("Service unavailable after all retries")

                raw = data.get("claims", [])
                claims: list[dict[str, Any]] = raw if isinstance(raw, list) else []
                logger.info("Retrieved %d claims", len(claims))
                return claims

            except requests.HTTPError as exc:
                last_exc = exc
                if exc.response is not None and exc.response.status_code < 500:
                    break
            except requests.RequestException as exc:
                last_exc = exc

            if attempt < self.max_retries - 1:
                time.sleep(delay)
                delay *= 2

        raise RetrievalError(
            f"Failed after {self.max_retries} attempts",
            details={"last_error": str(last_exc)},
        ) from last_exc

    def retrieve_by_query(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        return self.query_api({"query": query, "key": self.api_key, **kwargs})

    def retrieve_recent(self, limit: int = 100, **kwargs: Any) -> list[dict[str, Any]]:
        return self.query_api({"limit": limit, "sort": "recent", **kwargs})
