import anthropic
from anthropic.types import TextBlock

from agent.clients.base import BaseClient
from agent.types import LLMResponse, Message, TokenUsage
from core.exceptions import LLMClientError


class AnthropicClient(BaseClient):
    def __init__(self, *, api_key: str, model: str) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        try:
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": m.role, "content": m.content} for m in messages],
            )
            block = next((b for b in resp.content if isinstance(b, TextBlock)), None)
            if block is None:
                raise LLMClientError("No text block in LLM response")
            return LLMResponse(
                text=block.text,
                model=resp.model,
                usage=TokenUsage(
                    input_tokens=resp.usage.input_tokens,
                    output_tokens=resp.usage.output_tokens,
                ),
            )
        except LLMClientError:
            raise
        except Exception as exc:
            raise LLMClientError(f"Anthropic API call failed: {exc}") from exc
