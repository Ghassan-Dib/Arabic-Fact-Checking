import logging
import sys
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest
from anthropic.types import TextBlock

from agent.agent import Agent
from agent.clients.anthropic import AnthropicClient
from agent.clients.genai import GenAIClient
from agent.clients.openai import OpenAIClient
from agent.factory import create_client
from agent.types import AgentConfig, LLMResponse, Message, Provider, TokenUsage
from core.exceptions import ConfigurationError, LLMClientError

# ---------------------------------------------------------------------------
# types.py
# ---------------------------------------------------------------------------


class TestAgentTypes:
    def test_provider_enum_values(self) -> None:
        """Provider enum must expose anthropic, openai, and genai values."""
        # Arrange / Act / Assert
        assert Provider.ANTHROPIC == "anthropic"
        assert Provider.OPENAI == "openai"
        assert Provider.GENAI == "genai"

    def test_message_model(self) -> None:
        """Message accepts role and content fields."""
        # Arrange / Act
        msg = Message(role="user", content="hello")

        # Assert
        assert msg.role == "user"
        assert msg.content == "hello"

    def test_token_usage_model(self) -> None:
        """TokenUsage stores input and output token counts."""
        # Arrange / Act
        usage = TokenUsage(input_tokens=10, output_tokens=5)

        # Assert
        assert usage.input_tokens == 10
        assert usage.output_tokens == 5

    def test_llm_response_without_usage(self) -> None:
        """LLMResponse can be created without usage metadata."""
        # Arrange / Act
        resp = LLMResponse(text="answer", model="claude-test")

        # Assert
        assert resp.text == "answer"
        assert resp.model == "claude-test"
        assert resp.usage is None

    def test_llm_response_with_usage(self) -> None:
        """LLMResponse stores TokenUsage when provided."""
        # Arrange
        usage = TokenUsage(input_tokens=20, output_tokens=10)

        # Act
        resp = LLMResponse(text="answer", model="claude-test", usage=usage)

        # Assert
        assert resp.usage is not None
        assert resp.usage.input_tokens == 20

    def test_agent_config_defaults(self) -> None:
        """AgentConfig applies sensible defaults for max_tokens and temperature."""
        # Arrange / Act
        config = AgentConfig(provider=Provider.ANTHROPIC, model="m", api_key="k")

        # Assert
        assert config.max_tokens == 1024
        assert config.temperature == 0.0

    def test_agent_config_custom_values(self) -> None:
        """AgentConfig accepts custom max_tokens and temperature."""
        # Arrange / Act
        config = AgentConfig(
            provider=Provider.OPENAI,
            model="gpt-4",
            api_key="key",
            max_tokens=500,
            temperature=0.7,
        )

        # Assert
        assert config.max_tokens == 500
        assert config.temperature == 0.7


# ---------------------------------------------------------------------------
# agent.py — Agent class
# ---------------------------------------------------------------------------


class TestAgent:
    @pytest.fixture
    def config(self) -> AgentConfig:
        return AgentConfig(
            provider=Provider.ANTHROPIC,
            model="claude-test",
            api_key="test-key",
            max_tokens=256,
            temperature=0.5,
        )

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        client = MagicMock()
        client.complete.return_value = LLMResponse(text="response", model="claude-test")
        return client

    def test_run_uses_config_defaults(self, mock_client: MagicMock, config: AgentConfig) -> None:
        """Agent.run() passes config max_tokens and temperature when not overridden."""
        # Arrange
        agent = Agent(client=mock_client, config=config)

        # Act
        result = agent.run("hello")

        # Assert
        assert result == "response"
        mock_client.complete.assert_called_once()
        _, kwargs = mock_client.complete.call_args
        assert kwargs["max_tokens"] == 256
        assert kwargs["temperature"] == 0.5

    def test_run_overrides_max_tokens(self, mock_client: MagicMock, config: AgentConfig) -> None:
        """Agent.run() uses the caller-supplied max_tokens over the config default."""
        # Arrange
        agent = Agent(client=mock_client, config=config)

        # Act
        agent.run("hello", max_tokens=100)

        # Assert
        _, kwargs = mock_client.complete.call_args
        assert kwargs["max_tokens"] == 100

    def test_run_overrides_temperature(self, mock_client: MagicMock, config: AgentConfig) -> None:
        """Agent.run() uses the caller-supplied temperature over the config default."""
        # Arrange
        agent = Agent(client=mock_client, config=config)

        # Act
        agent.run("hello", temperature=0.0)

        # Assert
        _, kwargs = mock_client.complete.call_args
        assert kwargs["temperature"] == 0.0

    def test_run_wraps_prompt_in_user_message(
        self, mock_client: MagicMock, config: AgentConfig
    ) -> None:
        """Agent.run() wraps the prompt as a user-role Message."""
        # Arrange
        agent = Agent(client=mock_client, config=config)

        # Act
        agent.run("my prompt")

        # Assert
        args, _ = mock_client.complete.call_args
        messages: list[Message] = args[0]
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert messages[0].content == "my prompt"


# ---------------------------------------------------------------------------
# clients/anthropic.py
# ---------------------------------------------------------------------------


class TestAnthropicClient:
    @pytest.fixture
    def mock_anthropic_cls(self) -> Generator[MagicMock, None, None]:
        with patch("agent.clients.anthropic.anthropic.Anthropic") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance
            yield mock_instance

    def _make_resp(self, text: str) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.content = [TextBlock(type="text", text=text)]
        mock_resp.model = "claude-test"
        mock_resp.usage = MagicMock(input_tokens=10, output_tokens=5)
        return mock_resp

    def test_complete_returns_llm_response(self, mock_anthropic_cls: MagicMock) -> None:
        """AnthropicClient.complete() returns an LLMResponse with text and usage."""
        # Arrange
        mock_anthropic_cls.messages.create.return_value = self._make_resp("hello")
        client = AnthropicClient(api_key="key", model="claude-test")
        messages = [Message(role="user", content="hi")]

        # Act
        resp = client.complete(messages, max_tokens=100, temperature=0.0)

        # Assert
        assert resp.text == "hello"
        assert resp.model == "claude-test"
        assert resp.usage is not None
        assert resp.usage.input_tokens == 10

    def test_complete_raises_on_no_text_block(self, mock_anthropic_cls: MagicMock) -> None:
        """AnthropicClient.complete() raises LLMClientError when response has no TextBlock."""
        # Arrange
        mock_resp = MagicMock()
        mock_resp.content = []
        mock_anthropic_cls.messages.create.return_value = mock_resp
        client = AnthropicClient(api_key="key", model="claude-test")
        messages = [Message(role="user", content="hi")]

        # Act / Assert
        with pytest.raises(LLMClientError, match="No text block"):
            client.complete(messages, max_tokens=100, temperature=0.0)

    def test_complete_wraps_unexpected_exceptions(self, mock_anthropic_cls: MagicMock) -> None:
        """AnthropicClient.complete() wraps unexpected errors as LLMClientError."""
        # Arrange
        mock_anthropic_cls.messages.create.side_effect = RuntimeError("network failure")
        client = AnthropicClient(api_key="key", model="claude-test")
        messages = [Message(role="user", content="hi")]

        # Act / Assert
        with pytest.raises(LLMClientError, match="Anthropic API call failed"):
            client.complete(messages, max_tokens=100, temperature=0.0)


# ---------------------------------------------------------------------------
# clients/openai.py
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_openai_module() -> Generator[MagicMock, None, None]:
    mock_module = MagicMock()
    sys.modules["openai"] = mock_module
    yield mock_module
    sys.modules.pop("openai", None)


class TestOpenAIClient:
    def test_init_raises_when_package_missing(self) -> None:
        """OpenAIClient raises LLMClientError when openai is not installed."""
        # Arrange — ensure openai is absent from sys.modules
        sys.modules.pop("openai", None)

        # Act / Assert
        with pytest.raises(LLMClientError, match="openai package not installed"):
            OpenAIClient(api_key="key", model="gpt-4")

    def test_complete_returns_llm_response(self, mock_openai_module: MagicMock) -> None:
        """OpenAIClient.complete() returns an LLMResponse when openai is available."""
        # Arrange
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "openai answer"
        mock_resp.model = "gpt-4"
        mock_resp.usage = MagicMock(prompt_tokens=8, completion_tokens=4)
        mock_openai_module.OpenAI.return_value.chat.completions.create.return_value = mock_resp

        client = OpenAIClient(api_key="key", model="gpt-4")
        messages = [Message(role="user", content="hello")]

        # Act
        resp = client.complete(messages, max_tokens=50, temperature=0.0)

        # Assert
        assert resp.text == "openai answer"
        assert resp.model == "gpt-4"
        assert resp.usage is not None
        assert resp.usage.input_tokens == 8

    def test_complete_handles_null_usage(self, mock_openai_module: MagicMock) -> None:
        """OpenAIClient.complete() returns None usage when response has no usage field."""
        # Arrange
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "ok"
        mock_resp.model = "gpt-4"
        mock_resp.usage = None
        mock_openai_module.OpenAI.return_value.chat.completions.create.return_value = mock_resp

        client = OpenAIClient(api_key="key", model="gpt-4")
        messages = [Message(role="user", content="hello")]

        # Act
        resp = client.complete(messages, max_tokens=50, temperature=0.0)

        # Assert
        assert resp.usage is None

    def test_complete_wraps_unexpected_exceptions(self, mock_openai_module: MagicMock) -> None:
        """OpenAIClient.complete() wraps unexpected errors as LLMClientError."""
        # Arrange
        mock_openai_module.OpenAI.return_value.chat.completions.create.side_effect = RuntimeError(
            "timeout"
        )
        client = OpenAIClient(api_key="key", model="gpt-4")
        messages = [Message(role="user", content="hello")]

        # Act / Assert
        with pytest.raises(LLMClientError, match="OpenAI API call failed"):
            client.complete(messages, max_tokens=50, temperature=0.0)


# ---------------------------------------------------------------------------
# clients/genai.py
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_genai_module() -> Generator[MagicMock, None, None]:
    mock_genai = MagicMock()
    mock_google = MagicMock()
    mock_google.generativeai = mock_genai
    sys.modules["google"] = mock_google
    sys.modules["google.generativeai"] = mock_genai
    yield mock_genai
    sys.modules.pop("google", None)
    sys.modules.pop("google.generativeai", None)


class TestGenAIClient:
    def test_init_raises_when_package_missing(self) -> None:
        """GenAIClient raises LLMClientError when google-generativeai is not installed."""
        # Arrange — ensure package is absent
        sys.modules.pop("google", None)
        sys.modules.pop("google.generativeai", None)

        # Act / Assert
        with pytest.raises(LLMClientError, match="google-generativeai package not installed"):
            GenAIClient(api_key="key", model="gemini-pro")

    def test_complete_returns_llm_response(self, mock_genai_module: MagicMock) -> None:
        """GenAIClient.complete() returns an LLMResponse when genai is available."""
        # Arrange
        mock_response = MagicMock()
        mock_response.text = "gemini answer"
        mock_genai_module.GenerativeModel.return_value.generate_content.return_value = mock_response

        client = GenAIClient(api_key="key", model="gemini-pro")
        messages = [Message(role="user", content="hello")]

        # Act
        resp = client.complete(messages, max_tokens=200, temperature=0.3)

        # Assert
        assert resp.text == "gemini answer"
        assert resp.model == "gemini-pro"

    def test_complete_warns_on_non_user_messages(
        self, mock_genai_module: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """GenAIClient.complete() logs a warning when non-user messages are dropped."""
        # Arrange
        mock_response = MagicMock()
        mock_response.text = "ok"
        mock_genai_module.GenerativeModel.return_value.generate_content.return_value = mock_response
        client = GenAIClient(api_key="key", model="gemini-pro")
        messages = [
            Message(role="system", content="be helpful"),
            Message(role="user", content="hello"),
        ]

        # Act
        with caplog.at_level(logging.WARNING, logger="agent.clients.genai"):
            client.complete(messages, max_tokens=100, temperature=0.0)

        # Assert
        assert "non-user" in caplog.text

    def test_complete_wraps_unexpected_exceptions(self, mock_genai_module: MagicMock) -> None:
        """GenAIClient.complete() wraps unexpected errors as LLMClientError."""
        # Arrange
        mock_genai_module.GenerativeModel.return_value.generate_content.side_effect = RuntimeError(
            "quota exceeded"
        )
        client = GenAIClient(api_key="key", model="gemini-pro")
        messages = [Message(role="user", content="hello")]

        # Act / Assert
        with pytest.raises(LLMClientError, match="GenAI API call failed"):
            client.complete(messages, max_tokens=200, temperature=0.0)


# ---------------------------------------------------------------------------
# factory.py
# ---------------------------------------------------------------------------


class TestCreateClient:
    @patch("agent.clients.anthropic.anthropic.Anthropic")
    def test_creates_anthropic_client(self, _: MagicMock) -> None:
        """create_client returns an AnthropicClient for Provider.ANTHROPIC."""
        # Arrange
        config = AgentConfig(provider=Provider.ANTHROPIC, model="claude-test", api_key="k")

        # Act
        client = create_client(config)

        # Assert
        assert isinstance(client, AnthropicClient)

    def test_creates_openai_client(self, mock_openai_module: MagicMock) -> None:
        """create_client returns an OpenAIClient for Provider.OPENAI."""
        # Arrange
        config = AgentConfig(provider=Provider.OPENAI, model="gpt-4", api_key="k")

        # Act
        client = create_client(config)

        # Assert
        assert isinstance(client, OpenAIClient)

    def test_creates_genai_client(self, mock_genai_module: MagicMock) -> None:
        """create_client returns a GenAIClient for Provider.GENAI."""
        # Arrange
        config = AgentConfig(provider=Provider.GENAI, model="gemini-pro", api_key="k")

        # Act
        client = create_client(config)

        # Assert
        assert isinstance(client, GenAIClient)

    def test_raises_for_unknown_provider(self) -> None:
        """create_client raises ConfigurationError for an unrecognised provider string."""
        # Arrange — forge a config with an invalid provider value
        config = AgentConfig(provider=Provider.ANTHROPIC, model="m", api_key="k")
        object.__setattr__(config, "provider", "unknown")  # type: ignore[arg-type]

        # Act / Assert
        with pytest.raises(ConfigurationError):
            create_client(config)
