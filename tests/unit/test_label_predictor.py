from unittest.mock import MagicMock

import pytest

from models.claim import ClaimLabel
from verification.label_predictor import LabelPredictor


@pytest.fixture
def mock_agent() -> MagicMock:
    return MagicMock()


class TestPredict:
    def test_supported_label(self, mock_agent: MagicMock) -> None:
        """SUPPORTED JSON response maps to ClaimLabel.SUPPORTED."""
        # Arrange
        mock_agent.run.return_value = '{"predicted_label": "SUPPORTED"}'
        predictor = LabelPredictor(agent=mock_agent)

        # Act
        label = predictor.predict("ادعاء", "دليل")

        # Assert
        assert label == ClaimLabel.SUPPORTED

    def test_refuted_label(self, mock_agent: MagicMock) -> None:
        """REFUTED JSON response maps to ClaimLabel.REFUTED."""
        # Arrange
        mock_agent.run.return_value = '{"predicted_label": "REFUTED"}'
        predictor = LabelPredictor(agent=mock_agent)

        # Act
        label = predictor.predict("ادعاء", "دليل")

        # Assert
        assert label == ClaimLabel.REFUTED

    def test_not_enough_evidence_label(self, mock_agent: MagicMock) -> None:
        """NOT_ENOUGH_EVIDENCE JSON response maps to ClaimLabel.NOT_ENOUGH_EVIDENCE."""
        # Arrange
        mock_agent.run.return_value = '{"predicted_label": "NOT_ENOUGH_EVIDENCE"}'
        predictor = LabelPredictor(agent=mock_agent)

        # Act
        label = predictor.predict("ادعاء", "دليل")

        # Assert
        assert label == ClaimLabel.NOT_ENOUGH_EVIDENCE

    def test_conflicting_evidence_label(self, mock_agent: MagicMock) -> None:
        """CONFLICTING_EVIDENCE JSON response maps to ClaimLabel.CONFLICTING_EVIDENCE."""
        # Arrange
        mock_agent.run.return_value = '{"predicted_label": "CONFLICTING_EVIDENCE"}'
        predictor = LabelPredictor(agent=mock_agent)

        # Act
        label = predictor.predict("ادعاء", "دليل")

        # Assert
        assert label == ClaimLabel.CONFLICTING_EVIDENCE

    def test_unknown_label_defaults_to_nei(self, mock_agent: MagicMock) -> None:
        """Unrecognised label string falls back to NOT_ENOUGH_EVIDENCE."""
        # Arrange
        mock_agent.run.return_value = '{"predicted_label": "UNKNOWN"}'
        predictor = LabelPredictor(agent=mock_agent)

        # Act
        label = predictor.predict("ادعاء", "دليل")

        # Assert
        assert label == ClaimLabel.NOT_ENOUGH_EVIDENCE

    def test_invalid_json_defaults_to_nei(self, mock_agent: MagicMock) -> None:
        """Non-JSON response falls back to NOT_ENOUGH_EVIDENCE."""
        # Arrange
        mock_agent.run.return_value = "not json"
        predictor = LabelPredictor(agent=mock_agent)

        # Act
        label = predictor.predict("ادعاء", "دليل")

        # Assert
        assert label == ClaimLabel.NOT_ENOUGH_EVIDENCE
