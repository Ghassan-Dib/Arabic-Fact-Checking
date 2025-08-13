# src/fact_checker/core/exceptions.py
"""Custom exceptions for the fact checker system."""

from typing import Optional, Dict, Any, List


class FactCheckerError(Exception):
    """Base exception for all fact checker errors.

    This is the base class for all custom exceptions in the fact checker system.
    All other custom exceptions should inherit from this class.
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize the base exception.

        Args:
            message: Human-readable error message.
            details: Optional dictionary with additional error details.
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        """Return string representation of the exception."""
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ConfigurationError(FactCheckerError):
    """Exception raised for configuration-related errors.

    This includes missing environment variables, invalid configuration values,
    missing configuration files, etc.
    """

    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        """Initialize configuration error.

        Args:
            message: Error message.
            config_key: The configuration key that caused the error.
            **kwargs: Additional details passed to parent class.
        """
        details = kwargs.get("details", {})
        if config_key:
            details["config_key"] = config_key
        super().__init__(message, details)
        self.config_key = config_key


class LLMClientError(FactCheckerError):
    """Exception raised when LLM client operations fail.

    This includes API errors, authentication failures, rate limiting,
    and other LLM service-related issues.
    """

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        **kwargs,
    ):
        """Initialize LLM client error.

        Args:
            message: Error message.
            status_code: HTTP status code if applicable.
            response_body: Response body from the LLM service.
            **kwargs: Additional details passed to parent class.
        """
        details = kwargs.get("details", {})
        if status_code:
            details["status_code"] = status_code
        if response_body:
            details["response_body"] = response_body[:500]  # Truncate long responses

        super().__init__(message, details)
        self.status_code = status_code
        self.response_body = response_body


class JSONParsingError(FactCheckerError):
    """Exception raised when JSON parsing fails.

    This occurs when the LLM returns malformed JSON or when expected
    JSON structure is not found in the response.
    """

    def __init__(
        self,
        message: str,
        raw_response: Optional[str] = None,
        expected_fields: Optional[List[str]] = None,
        **kwargs,
    ):
        """Initialize JSON parsing error.

        Args:
            message: Error message.
            raw_response: Raw response that failed to parse.
            expected_fields: List of expected JSON fields that were missing.
            **kwargs: Additional details passed to parent class.
        """
        details = kwargs.get("details", {})
        if raw_response:
            # Store truncated response for debugging
            details["raw_response"] = (
                raw_response[:500] + "..." if len(raw_response) > 500 else raw_response
            )
        if expected_fields:
            details["expected_fields"] = expected_fields

        super().__init__(message, details)
        self.raw_response = raw_response
        self.expected_fields = expected_fields or []


class RetrievalError(FactCheckerError):
    """Exception raised when retrieval operations fail.

    This includes errors in claim retrieval, evidence retrieval,
    API failures, and data source unavailability.
    """

    def __init__(
        self,
        message: str,
        retrieval_type: Optional[str] = None,
        source_url: Optional[str] = None,
        retry_count: Optional[int] = None,
        **kwargs,
    ):
        """Initialize retrieval error.

        Args:
            message: Error message.
            retrieval_type: Type of retrieval that failed (e.g., 'claims', 'evidence').
            source_url: URL of the data source that failed.
            retry_count: Number of retries attempted.
            **kwargs: Additional details passed to parent class.
        """
        details = kwargs.get("details", {})
        if retrieval_type:
            details["retrieval_type"] = retrieval_type
        if source_url:
            details["source_url"] = source_url
        if retry_count is not None:
            details["retry_count"] = retry_count

        super().__init__(message, details)
        self.retrieval_type = retrieval_type
        self.source_url = source_url
        self.retry_count = retry_count


class VerificationError(FactCheckerError):
    """Exception raised when verification operations fail.

    This includes errors in fact checking, evaluation metrics calculation,
    QA generation, and label prediction.
    """

    def __init__(
        self,
        message: str,
        verification_step: Optional[str] = None,
        claim_id: Optional[str] = None,
        **kwargs,
    ):
        """Initialize verification error.

        Args:
            message: Error message.
            verification_step: The verification step that failed.
            claim_id: ID of the claim being verified when error occurred.
            **kwargs: Additional details passed to parent class.
        """
        details = kwargs.get("details", {})
        if verification_step:
            details["verification_step"] = verification_step
        if claim_id:
            details["claim_id"] = claim_id

        super().__init__(message, details)
        self.verification_step = verification_step
        self.claim_id = claim_id


class DataProcessingError(FactCheckerError):
    """Exception raised when data processing operations fail.

    This includes errors in data loading, preprocessing, transformation,
    and validation of input data.
    """

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        data_format: Optional[str] = None,
        line_number: Optional[int] = None,
        **kwargs,
    ):
        """Initialize data processing error.

        Args:
            message: Error message.
            file_path: Path to the file being processed.
            data_format: Format of the data (e.g., 'csv', 'json').
            line_number: Line number where error occurred (for file processing).
            **kwargs: Additional details passed to parent class.
        """
        details = kwargs.get("details", {})
        if file_path:
            details["file_path"] = file_path
        if data_format:
            details["data_format"] = data_format
        if line_number is not None:
            details["line_number"] = line_number

        super().__init__(message, details)
        self.file_path = file_path
        self.data_format = data_format
        self.line_number = line_number


class EvaluationError(FactCheckerError):
    """Exception raised when evaluation operations fail.

    This includes errors in metric calculation, evaluation pipeline execution,
    and result analysis.
    """

    def __init__(
        self,
        message: str,
        metric_name: Optional[str] = None,
        evaluation_type: Optional[str] = None,
        **kwargs,
    ):
        """Initialize evaluation error.

        Args:
            message: Error message.
            metric_name: Name of the metric that failed to calculate.
            evaluation_type: Type of evaluation (e.g., 'recall', 'precision').
            **kwargs: Additional details passed to parent class.
        """
        details = kwargs.get("details", {})
        if metric_name:
            details["metric_name"] = metric_name
        if evaluation_type:
            details["evaluation_type"] = evaluation_type

        super().__init__(message, details)
        self.metric_name = metric_name
        self.evaluation_type = evaluation_type


class WebScrapingError(FactCheckerError):
    """Exception raised when web scraping operations fail.

    This includes errors in HTML parsing, content extraction,
    and web page access issues.
    """

    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs,
    ):
        """Initialize web scraping error.

        Args:
            message: Error message.
            url: URL that failed to scrape.
            status_code: HTTP status code received.
            **kwargs: Additional details passed to parent class.
        """
        details = kwargs.get("details", {})
        if url:
            details["url"] = url
        if status_code:
            details["status_code"] = status_code

        super().__init__(message, details)
        self.url = url
        self.status_code = status_code


class ValidationError(FactCheckerError):
    """Exception raised when data validation fails.

    This includes errors in input validation, schema validation,
    and data integrity checks.
    """

    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        invalid_value: Optional[Any] = None,
        expected_type: Optional[str] = None,
        **kwargs,
    ):
        """Initialize validation error.

        Args:
            message: Error message.
            field_name: Name of the field that failed validation.
            invalid_value: The invalid value that caused the error.
            expected_type: Expected type or format.
            **kwargs: Additional details passed to parent class.
        """
        details = kwargs.get("details", {})
        if field_name:
            details["field_name"] = field_name
        if invalid_value is not None:
            details["invalid_value"] = str(invalid_value)
        if expected_type:
            details["expected_type"] = expected_type

        super().__init__(message, details)
        self.field_name = field_name
        self.invalid_value = invalid_value
        self.expected_type = expected_type


# Convenience functions for creating specific exceptions
def create_config_error(
    config_key: str, reason: str = "is missing or invalid"
) -> ConfigurationError:
    """Create a configuration error for a specific config key.

    Args:
        config_key: The configuration key that has an issue.
        reason: Reason for the configuration error.

    Returns:
        ConfigurationError instance.
    """
    return ConfigurationError(
        f"Configuration key '{config_key}' {reason}", config_key=config_key
    )


def create_retrieval_error(
    retrieval_type: str, source: str, error_details: str
) -> RetrievalError:
    """Create a retrieval error with context.

    Args:
        retrieval_type: Type of retrieval (e.g., 'claims', 'evidence').
        source: Source that failed (e.g., URL, API endpoint).
        error_details: Detailed error information.

    Returns:
        RetrievalError instance.
    """
    return RetrievalError(
        f"Failed to retrieve {retrieval_type} from {source}: {error_details}",
        retrieval_type=retrieval_type,
        source_url=source,
    )


def create_llm_error(operation: str, error_details: str, **kwargs) -> LLMClientError:
    """Create an LLM error with context.

    Args:
        operation: The operation that failed (e.g., 'generate_response', 'parse_json').
        error_details: Detailed error information.
        **kwargs: Additional error details.

    Returns:
        LLMClientError instance.
    """
    return LLMClientError(
        f"LLM operation '{operation}' failed: {error_details}", **kwargs
    )
