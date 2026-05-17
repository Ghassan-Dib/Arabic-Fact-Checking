from agent.clients.base import BaseClient
from agent.types import LLMResponse, Message
from core.exceptions import LLMClientError


class GenAIClient(BaseClient):
    def __init__(self, *, api_key: str, model: str) -> None:
        try:
            import google.generativeai as genai  # pyright: ignore[reportMissingImports]

            genai.configure(api_key=api_key)
            self._genai_model = genai.GenerativeModel(model)
        except ImportError as exc:
            raise LLMClientError(
                "google-generativeai package not installed. Run: uv add google-generativeai"
            ) from exc
        self._model_name = model

    def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        try:
            import google.generativeai as genai  # pyright: ignore[reportMissingImports]

            prompt = "\n".join(m.content for m in messages if m.role == "user")
            response = self._genai_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                ),
            )
            return LLMResponse(text=response.text, model=self._model_name)
        except LLMClientError:
            raise
        except Exception as exc:
            raise LLMClientError(f"GenAI API call failed: {exc}") from exc
