# src/fact_checker/retrieval/claim_retriever.py
"""Claim retrieval functionality for fact checking."""

import time
import logging
import requests
from typing import List, Dict, Any, Optional

from src.config.settings import FACT_CHECK_TOOLS_URL
from src.core.exceptions import RetrievalError

logger = logging.getLogger(__name__)


class ClaimRetriever:
    """Retrieves claims from fact-checking APIs."""

    def __init__(
        self,
        api_url: Optional[str] = None,
        max_retries: int = 5,
        initial_retry_delay: float = 1.0,
    ):
        self.api_url = api_url or FACT_CHECK_TOOLS_URL
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay

        if not self.api_url:
            raise RetrievalError("FACT_CHECK_TOOLS_URL must be configured")

        logger.info(f"Initialized ClaimRetriever with URL: {self.api_url}")

    def query_api(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        retry_delay = self.initial_retry_delay
        last_exception = None

        logger.info(f"Querying API with params: {params}")

        for attempt in range(self.max_retries):
            try:
                response = self._make_request(params)
                data = response.json()

                # Handle service unavailable with exponential backoff
                if self._is_service_unavailable(data):
                    if attempt < self.max_retries - 1:  # Don't sleep on last attempt
                        logger.warning(
                            f"Service unavailable. Retrying in {retry_delay:.1f} seconds..."
                        )
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        raise RetrievalError(
                            "Service unavailable after all retry attempts"
                        )

                claims = data.get("claims", [])
                logger.info(f"Successfully retrieved {len(claims)} claims")
                return claims

            except requests.exceptions.HTTPError as e:
                last_exception = e
                logger.error(f"HTTP error on attempt {attempt + 1}: {e}")
                if e.response.status_code < 500:  # Don't retry client errors
                    break

            except requests.exceptions.RequestException as e:
                last_exception = e
                logger.error(f"Request error on attempt {attempt + 1}: {e}")

            except Exception as e:
                last_exception = e
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")

            # Add delay before retry (except on last attempt)
            if attempt < self.max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2

        # All retries failed
        error_msg = f"Failed to retrieve claims after {self.max_retries} attempts"
        if last_exception:
            error_msg += f". Last error: {last_exception}"

        logger.error(error_msg)
        raise RetrievalError(error_msg) from last_exception

    def _make_request(self, params: Dict[str, Any]) -> requests.Response:
        response = requests.get(self.api_url, params=params, timeout=30)
        response.raise_for_status()
        return response

    def _is_service_unavailable(self, data: Dict[str, Any]) -> bool:
        return (
            "error" in data
            and isinstance(data["error"], dict)
            and data["error"].get("code") == 503
        )

    def retrieve_claims_by_query(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        params = {"query": query, **kwargs}
        return self.query_api(params)

    def retrieve_claims_by_id(self, claim_ids: List[str]) -> List[Dict[str, Any]]:
        params = {"ids": ",".join(claim_ids)}
        return self.query_api(params)

    def retrieve_recent_claims(
        self, limit: int = 100, **kwargs
    ) -> List[Dict[str, Any]]:
        params = {"limit": limit, "sort": "recent", **kwargs}
        return self.query_api(params)


# Usage
# from fact_checker.retrieval import ClaimRetriever

# retriever = ClaimRetriever()
# claims = retriever.query_api(params)
