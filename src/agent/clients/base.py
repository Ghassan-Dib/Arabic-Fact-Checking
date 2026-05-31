from abc import ABC, abstractmethod

from agent.types import LLMResponse, Message


class BaseClient(ABC):
    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse: ...
