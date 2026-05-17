from enum import StrEnum

from pydantic import BaseModel


class Provider(StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GENAI = "genai"


class Message(BaseModel):
    role: str
    content: str


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int


class LLMResponse(BaseModel):
    text: str
    model: str
    usage: TokenUsage | None = None


class AgentConfig(BaseModel):
    provider: Provider
    model: str
    api_key: str
    max_tokens: int = 1024
    temperature: float = 0.0
