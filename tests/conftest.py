import os
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest
from anthropic.types import TextBlock
from fastapi.testclient import TestClient

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("API_KEY", "test-google-key")
os.environ.setdefault("FACT_CHECK_TOOLS_URL", "https://factcheck.example.com")


@pytest.fixture(scope="session")
def client() -> Generator[TestClient, None, None]:
    from api.app import create_app
    from core.config import get_settings

    get_settings.cache_clear()
    app = create_app()
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_anthropic() -> Generator[MagicMock, None, None]:
    with patch("anthropic.Anthropic") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        mock_msg = MagicMock()
        mock_msg.content = [
            TextBlock(type="text", text='{"qa_pairs": [{"question": "سؤال؟", "answer": "جواب"}]}')
        ]
        mock_instance.messages.create.return_value = mock_msg
        yield mock_instance
