from unittest.mock import MagicMock, patch

import pytest
import requests

from core.exceptions import RetrievalError
from retrieval.claim_retriever import ClaimRetriever


@pytest.fixture
def retriever() -> ClaimRetriever:
    return ClaimRetriever(api_url="https://api.example.com", api_key="test-key", max_retries=3)


class TestQueryApi:
    def test_returns_claims_on_success(self, retriever: ClaimRetriever) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"claims": [{"text": "ادعاء 1"}]}
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_resp):
            claims = retriever.query_api({"query": "test"})

        assert len(claims) == 1
        assert claims[0]["text"] == "ادعاء 1"

    def test_empty_response_returns_empty_list(self, retriever: ClaimRetriever) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_resp):
            claims = retriever.query_api({"query": "test"})

        assert claims == []

    def test_raises_on_all_retries_exhausted(self, retriever: ClaimRetriever) -> None:
        with (
            patch("requests.get", side_effect=requests.ConnectionError("down")),
            pytest.raises(RetrievalError),
        ):
            retriever.query_api({"query": "test"})

    def test_no_retry_on_4xx(self, retriever: ClaimRetriever) -> None:
        mock_resp = MagicMock()
        http_err = requests.HTTPError(response=MagicMock(status_code=400))
        mock_resp.raise_for_status.side_effect = http_err

        call_count = 0

        def mock_get(*args: object, **kwargs: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            return mock_resp

        with (
            patch("requests.get", side_effect=mock_get),
            pytest.raises(RetrievalError),
        ):
            retriever.query_api({"query": "test"})

        assert call_count == 1

    def test_503_triggers_backoff(self, retriever: ClaimRetriever) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"error": {"code": 503}}
        mock_resp.raise_for_status = MagicMock()

        with (
            patch("requests.get", return_value=mock_resp),
            patch("time.sleep"),
            pytest.raises(RetrievalError),
        ):
            retriever.query_api({"query": "test"})
