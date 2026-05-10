from unittest.mock import MagicMock, patch

import pytest
import requests

from core.exceptions import RetrievalError
from retrieval.claim_retriever import ClaimRetriever


@pytest.fixture
def retriever() -> ClaimRetriever:
    return ClaimRetriever(api_url="https://api.example.com", api_key="test-key", max_retries=3)


class TestQueryApi:
    @patch("requests.get")
    def test_returns_claims_on_success(
        self, mock_get: MagicMock, retriever: ClaimRetriever
    ) -> None:
        mock_get.return_value = MagicMock(
            json=lambda: {"claims": [{"text": "ادعاء 1"}]},
            raise_for_status=lambda: None,
        )

        claims = retriever.query_api({"query": "test"})

        assert len(claims) == 1
        assert claims[0]["text"] == "ادعاء 1"

    @patch("requests.get")
    def test_empty_response_returns_empty_list(
        self, mock_get: MagicMock, retriever: ClaimRetriever
    ) -> None:
        mock_get.return_value = MagicMock(
            json=lambda: {},
            raise_for_status=lambda: None,
        )

        claims = retriever.query_api({"query": "test"})

        assert claims == []

    @patch("requests.get", side_effect=requests.ConnectionError("down"))
    def test_raises_on_all_retries_exhausted(self, _: MagicMock, retriever: ClaimRetriever) -> None:
        with pytest.raises(RetrievalError):
            retriever.query_api({"query": "test"})

    @patch("requests.get")
    def test_no_retry_on_4xx(self, mock_get: MagicMock, retriever: ClaimRetriever) -> None:
        call_count = 0

        def side_effect(*args: object, **kwargs: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            mock = MagicMock()
            mock.raise_for_status.side_effect = requests.HTTPError(
                response=MagicMock(status_code=400)
            )
            return mock

        mock_get.side_effect = side_effect

        with pytest.raises(RetrievalError):
            retriever.query_api({"query": "test"})

        assert call_count == 1

    @patch("time.sleep")
    @patch("requests.get")
    def test_503_triggers_backoff(
        self, mock_get: MagicMock, _sleep: MagicMock, retriever: ClaimRetriever
    ) -> None:
        mock_get.return_value = MagicMock(
            json=lambda: {"error": {"code": 503}},
            raise_for_status=lambda: None,
        )

        with pytest.raises(RetrievalError):
            retriever.query_api({"query": "test"})
