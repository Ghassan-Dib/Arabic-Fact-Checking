from typing import Any


class FactCheckerError(Exception):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ConfigurationError(FactCheckerError):
    pass


class RetrievalError(FactCheckerError):
    pass


class VerificationError(FactCheckerError):
    pass


class EvaluationError(FactCheckerError):
    pass


class WebScrapingError(FactCheckerError):
    pass


class JSONParsingError(FactCheckerError):
    def __init__(
        self,
        message: str,
        raw_response: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        d = dict(details or {})
        if raw_response:
            d["raw_response"] = raw_response[:500]
        super().__init__(message, d)
        self.raw_response = raw_response


class LLMClientError(FactCheckerError):
    pass
