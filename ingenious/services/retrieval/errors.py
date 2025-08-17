"""Defines custom exception types for the retrieval service.

This module centralizes all custom errors related to the data retrieval and
answer generation service. By providing a hierarchy of specific exception types,
it allows upstream consumers to implement more granular and robust error handling.

The primary base exception is `RetrievalError`, with more specific errors like
`PreflightError` inheriting from it. This allows catching specific failures
(e.g., bad configuration) or all retrieval-related issues.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class RetrievalError(Exception):
    """Base exception for all retrieval-related errors."""

    pass


class PreflightError(RetrievalError):
    """Raised for failures during pre-flight checks before query execution."""

    def __init__(
        self,
        provider: str,
        reason: str,
        detail: str = "",
        snapshot: dict[str, Any] | None = None,
    ) -> None:
        """Initializes the PreflightError with context.

        This constructor captures essential details about a pre-flight failure,
        enabling structured error logging and handling.

        Args:
            provider: The name of the retrieval provider that failed.
            reason: A standardized reason for the failure.
            detail: A human-readable explanation of the error.
            snapshot: A dictionary of relevant state at the time of the error.
        """
        super().__init__(f"[{provider}] {reason}: {detail}")
        self.provider: str = provider
        self.reason: str = reason
        self.detail: str = detail
        self.snapshot: dict[str, Any] = snapshot or {}


class PreflightReason(str, Enum):
    """Provides standardized, machine-readable reasons for pre-flight failures."""

    GENERATION_DISABLED = "generation_disabled"
    # (optional) add more normalized reasons over time:
    # INVALID_CONFIG = "invalid_config"
    # MISSING_SEMANTIC_CONFIG = "missing_semantic_config"


class GenerationDisabledError(PreflightError):
    """
    Raised when a caller invokes `answer()` while `enable_answer_generation` is False.
    Subclasses PreflightError so existing `except PreflightError` code continues to work.
    """

    def __init__(
        self,
        detail: str = "",
        snapshot: dict[str, Any] | None = None,
        provider: str = "azure_search",
    ) -> None:
        """Initializes the error with a standardized 'generation_disabled' reason.

        This constructor provides a convenient way to raise a pre-flight error
        specifically for when answer generation is disabled in the configuration.

        Args:
            detail: Optional human-readable details about the disabled setting.
            snapshot: A dictionary of relevant state at the time of the error.
            provider: The name of the retrieval provider being used.
        """
        super().__init__(
            provider=provider,
            reason=PreflightReason.GENERATION_DISABLED,
            detail=detail,
            snapshot=snapshot,
        )
