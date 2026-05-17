from agent.clients.base import BaseClient
from agent.types import LLMResponse, Message, TokenUsage
from core.exceptions import LLMClientError


class OpenAIClient(BaseClient):
    def __init__(self, *, api_key: str, model: str) -> None:
        try:
            import openai  # pyright: ignore[reportMissingImports]

            self._client = openai.OpenAI(api_key=api_key)
        except ImportError as exc:
            raise LLMClientError("openai package not installed. Run: uv add openai") from exc
        self._model = model

    def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": m.role, "content": m.content} for m in messages],
            )
            text: str = resp.choices[0].message.content or ""
            usage = (
                TokenUsage(
                    input_tokens=resp.usage.prompt_tokens,
                    output_tokens=resp.usage.completion_tokens,
                )
                if resp.usage
                else None
            )
            return LLMResponse(text=text, model=resp.model, usage=usage)
        except LLMClientError:
            raise
        except Exception as exc:
            raise LLMClientError(f"OpenAI API call failed: {exc}") from exc
