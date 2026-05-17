from agent.clients.anthropic import AnthropicClient
from agent.clients.base import BaseClient
from agent.clients.genai import GenAIClient
from agent.clients.openai import OpenAIClient
from agent.types import AgentConfig, Provider
from core.exceptions import ConfigurationError


def create_client(config: AgentConfig) -> BaseClient:
    """Instantiate the appropriate LLM client for the given provider config."""
    if config.provider == Provider.ANTHROPIC:
        return AnthropicClient(api_key=config.api_key, model=config.model)
    if config.provider == Provider.OPENAI:
        return OpenAIClient(api_key=config.api_key, model=config.model)
    if config.provider == Provider.GENAI:
        return GenAIClient(api_key=config.api_key, model=config.model)
    raise ConfigurationError(f"Unknown provider: {config.provider}")
