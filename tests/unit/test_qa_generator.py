from unittest.mock import MagicMock

import pytest

from core.exceptions import JSONParsingError
from verification.qa_generator import QAGenerator, _extract_json


class TestExtractJson:
    def test_plain_json(self) -> None:
        """Plain JSON object is parsed without modification."""
        # Arrange
        raw = '{"key": "value"}'

        # Act
        result = _extract_json(raw)

        # Assert
        assert result == {"key": "value"}

    def test_strips_json_fence(self) -> None:
        """```json ... ``` fences are stripped before parsing."""
        # Arrange
        raw = '```json\n{"key": "value"}\n```'

        # Act
        result = _extract_json(raw)

        # Assert
        assert result == {"key": "value"}

    def test_strips_plain_fence(self) -> None:
        """Plain ``` ... ``` fences are stripped before parsing."""
        # Arrange
        raw = '```\n{"key": "val"}\n```'

        # Act
        result = _extract_json(raw)

        # Assert
        assert result == {"key": "val"}

    def test_raises_on_no_json(self) -> None:
        """Text with no JSON object raises JSONParsingError."""
        # Arrange / Act / Assert
        with pytest.raises(JSONParsingError):
            _extract_json("no json here")

    def test_raises_on_invalid_json(self) -> None:
        """Malformed JSON object raises JSONParsingError."""
        # Arrange / Act / Assert
        with pytest.raises(JSONParsingError):
            _extract_json("{bad json}")


class TestQAGenerator:
    @pytest.fixture
    def mock_agent(self) -> MagicMock:
        return MagicMock()

    def test_generate_from_evidence_returns_pairs(self, mock_agent: MagicMock) -> None:
        """Valid LLM JSON response is parsed into QAPair objects."""
        # Arrange
        mock_agent.run.return_value = '{"qa_pairs": [{"question": "سؤال؟", "answer": "جواب"}]}'
        generator = QAGenerator(agent=mock_agent)

        # Act
        pairs = generator.generate_from_evidence("ادعاء ما", "دليل نصي")

        # Assert
        assert len(pairs) == 1
        assert pairs[0].question == "سؤال؟"
        assert pairs[0].answer == "جواب"

    def test_empty_qa_pairs_returns_empty_list(self, mock_agent: MagicMock) -> None:
        """LLM response with empty qa_pairs array returns an empty list."""
        # Arrange
        mock_agent.run.return_value = '{"qa_pairs": []}'
        generator = QAGenerator(agent=mock_agent)

        # Act
        pairs = generator.generate_from_evidence("ادعاء", "دليل")

        # Assert
        assert pairs == []
