from unittest.mock import MagicMock, patch

import pytest
from anthropic.types import TextBlock

from models.claim import ClaimLabel
from verification.label_predictor import LabelPredictor


@pytest.fixture
def predictor() -> LabelPredictor:
    return LabelPredictor(api_key="test", model="claude-test")


class TestPredict:
    def _mock_response(self, text: str) -> MagicMock:
        mock = MagicMock()
        mock.content = [TextBlock(type="text", text=text)]
        return mock

    def test_supported_label(self, predictor: LabelPredictor) -> None:
        with patch.object(
            predictor.client.messages,
            "create",
            return_value=self._mock_response('{"predicted_label": "SUPPORTED"}'),
        ):
            label = predictor.predict("ادعاء", "دليل")
        assert label == ClaimLabel.SUPPORTED

    def test_refuted_label(self, predictor: LabelPredictor) -> None:
        with patch.object(
            predictor.client.messages,
            "create",
            return_value=self._mock_response('{"predicted_label": "REFUTED"}'),
        ):
            label = predictor.predict("ادعاء", "دليل")
        assert label == ClaimLabel.REFUTED

    def test_unknown_label_defaults_to_nei(self, predictor: LabelPredictor) -> None:
        with patch.object(
            predictor.client.messages,
            "create",
            return_value=self._mock_response('{"predicted_label": "UNKNOWN"}'),
        ):
            label = predictor.predict("ادعاء", "دليل")
        assert label == ClaimLabel.NOT_ENOUGH_EVIDENCE

    def test_invalid_json_defaults_to_nei(self, predictor: LabelPredictor) -> None:
        with patch.object(
            predictor.client.messages,
            "create",
            return_value=self._mock_response("not json"),
        ):
            label = predictor.predict("ادعاء", "دليل")
        assert label == ClaimLabel.NOT_ENOUGH_EVIDENCE
