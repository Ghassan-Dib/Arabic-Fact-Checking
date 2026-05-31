from agent.clients.base import BaseClient
from agent.types import AgentConfig, LLMResponse, Message


class Agent:
    """Single-provider LLM agent backed by a BaseClient."""

    def __init__(self, *, client: BaseClient, config: AgentConfig) -> None:
        self._client: BaseClient = client
        self._config: AgentConfig = config

    def run(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Send a single user prompt and return the response text."""
        messages = [Message(role="user", content=prompt)]
        response: LLMResponse = self._client.complete(
            messages,
            max_tokens=max_tokens if max_tokens is not None else self._config.max_tokens,
            temperature=temperature if temperature is not None else self._config.temperature,
        )
        text: str = response.text
        return text
