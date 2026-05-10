from unittest.mock import MagicMock, patch

import pytest
from anthropic.types import TextBlock

from core.exceptions import JSONParsingError
from verification.qa_generator import QAGenerator, _extract_json


class TestExtractJson:
    def test_plain_json(self) -> None:
        result = _extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_strips_json_fence(self) -> None:
        result = _extract_json('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_strips_plain_fence(self) -> None:
        result = _extract_json('```\n{"key": "val"}\n```')
        assert result == {"key": "val"}

    def test_raises_on_no_json(self) -> None:
        with pytest.raises(JSONParsingError):
            _extract_json("no json here")

    def test_raises_on_invalid_json(self) -> None:
        with pytest.raises(JSONParsingError):
            _extract_json("{bad json}")


class TestQAGenerator:
    @pytest.fixture
    def generator(self) -> QAGenerator:
        return QAGenerator(api_key="test", model="claude-test")

    def test_generate_from_evidence_returns_pairs(self, generator: QAGenerator) -> None:
        mock_response = MagicMock()
        mock_response.content = [
            TextBlock(type="text", text='{"qa_pairs": [{"question": "سؤال؟", "answer": "جواب"}]}')
        ]

        with patch.object(generator.client.messages, "create", return_value=mock_response):
            pairs = generator.generate_from_evidence("ادعاء ما", "دليل نصي")

        assert len(pairs) == 1
        assert pairs[0].question == "سؤال؟"
        assert pairs[0].answer == "جواب"

    def test_empty_qa_pairs_returns_empty(self, generator: QAGenerator) -> None:
        mock_response = MagicMock()
        mock_response.content = [TextBlock(type="text", text='{"qa_pairs": []}')]

        with patch.object(generator.client.messages, "create", return_value=mock_response):
            pairs = generator.generate_from_evidence("ادعاء", "دليل")

        assert pairs == []
