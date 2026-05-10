from unittest.mock import MagicMock, patch

import pytest
from anthropic.types import TextBlock

from models.claim import ClaimLabel
from verification.label_predictor import LabelPredictor

MODULE = "verification.label_predictor"


@pytest.fixture
def predictor() -> LabelPredictor:
    return LabelPredictor(api_key="test", model="claude-test")


def _text_block(text: str) -> MagicMock:
    mock = MagicMock()
    mock.content = [TextBlock(type="text", text=text)]
    return mock


class TestPredict:
    @patch(f"{MODULE}.anthropic.Anthropic")
    def test_supported_label(self, MockAnthropic: MagicMock) -> None:
        """SUPPORTED JSON response maps to ClaimLabel.SUPPORTED."""
        # Arrange
        mock_client = MagicMock()
        MockAnthropic.return_value = mock_client
        mock_client.messages.create.return_value = _text_block('{"predicted_label": "SUPPORTED"}')
        predictor = LabelPredictor(api_key="test", model="claude-test")

        # Act
        label = predictor.predict("ادعاء", "دليل")

        # Assert
        assert label == ClaimLabel.SUPPORTED

    @patch(f"{MODULE}.anthropic.Anthropic")
    def test_refuted_label(self, MockAnthropic: MagicMock) -> None:
        """REFUTED JSON response maps to ClaimLabel.REFUTED."""
        # Arrange
        mock_client = MagicMock()
        MockAnthropic.return_value = mock_client
        mock_client.messages.create.return_value = _text_block('{"predicted_label": "REFUTED"}')
        predictor = LabelPredictor(api_key="test", model="claude-test")

        # Act
        label = predictor.predict("ادعاء", "دليل")

        # Assert
        assert label == ClaimLabel.REFUTED

    @patch(f"{MODULE}.anthropic.Anthropic")
    def test_unknown_label_defaults_to_nei(self, MockAnthropic: MagicMock) -> None:
        """Unrecognised label string falls back to NOT_ENOUGH_EVIDENCE."""
        # Arrange
        mock_client = MagicMock()
        MockAnthropic.return_value = mock_client
        mock_client.messages.create.return_value = _text_block('{"predicted_label": "UNKNOWN"}')
        predictor = LabelPredictor(api_key="test", model="claude-test")

        # Act
        label = predictor.predict("ادعاء", "دليل")

        # Assert
        assert label == ClaimLabel.NOT_ENOUGH_EVIDENCE

    @patch(f"{MODULE}.anthropic.Anthropic")
    def test_invalid_json_defaults_to_nei(self, MockAnthropic: MagicMock) -> None:
        """Non-JSON response falls back to NOT_ENOUGH_EVIDENCE."""
        # Arrange
        mock_client = MagicMock()
        MockAnthropic.return_value = mock_client
        mock_client.messages.create.return_value = _text_block("not json")
        predictor = LabelPredictor(api_key="test", model="claude-test")

        # Act
        label = predictor.predict("ادعاء", "دليل")

        # Assert
        assert label == ClaimLabel.NOT_ENOUGH_EVIDENCE
