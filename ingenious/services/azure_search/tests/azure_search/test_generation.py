"""Tests the AnswerGenerator for RAG response synthesis.

This module contains unit tests for the `AnswerGenerator` class, which is
responsible for generating a final answer from a user query and retrieved
context chunks. The tests verify the generator's initialization, context
formatting logic, and the end-to-end generation process. Key behaviors
tested include the success path, short-circuiting with no context,
exception handling during LLM calls, and proper resource cleanup.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest

from ingenious.services.azure_search.components.generation import (
    DEFAULT_RAG_PROMPT,
    AnswerGenerator,
)
from ingenious.services.azure_search.config import SearchConfig

if TYPE_CHECKING:
    from pytest import MonkeyPatch


@pytest.fixture
def generator(config: SearchConfig) -> AnswerGenerator:
    """Provides a default AnswerGenerator instance for tests.

    This fixture creates a reusable generator object, avoiding repetitive
    setup in each test function.

    Returns:
        A configured instance of AnswerGenerator.
    """
    return AnswerGenerator(config)


def test_generator_init_and_prompt(generator: AnswerGenerator) -> None:
    """Tests the AnswerGenerator's initialization and default prompt.

    This ensures the generator is created with the correct default RAG prompt
    template and that the internal LLM client is properly initialized.
    """
    assert generator.rag_prompt_template == DEFAULT_RAG_PROMPT
    assert hasattr(generator, "_llm_client")


def test_format_context(generator: AnswerGenerator, config: SearchConfig) -> None:
    """Tests the formatting of search result chunks into a context string.

    This verifies that context documents are correctly numbered, sourced,
    and concatenated into a single string suitable for an LLM prompt.
    """
    chunks: list[dict[str, str]] = [
        {"id": "1", config.content_field: "A."},
        {"id": "2", config.content_field: "B."},
    ]
    out: str = generator._format_context(chunks)
    assert "[Source 1]" in out and "A." in out
    assert "[Source 2]" in out and "B." in out
    assert "\n---\n" in out


def test_format_context_missing_content(generator: AnswerGenerator) -> None:
    """Tests context formatting when a chunk is missing the content field.

    This ensures the system handles malformed or incomplete search results
    gracefully by inserting a placeholder ('N/A') for the missing content.
    """
    out: str = generator._format_context([{"id": "1"}])
    assert "N/A" in out


@pytest.mark.asyncio
async def test_generate_success(
    generator: AnswerGenerator, config: SearchConfig, monkeypatch: MonkeyPatch
) -> None:
    """Tests the successful answer generation path.

    This verifies that given a question and context, the generator formats a
    prompt, calls the LLM client, and returns the expected generated answer.
    """
    client: Any = generator._llm_client
    monkeypatch.setattr(
        client.chat.completions,
        "create",
        AsyncMock(
            return_value=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="Answer [Source 1]")
                    )
                ]
            )
        ),
    )
    ans: str = await generator.generate("Q", [{"id": "1", config.content_field: "C"}])
    assert ans.startswith("Answer")
    client.chat.completions.create.assert_awaited()


@pytest.mark.asyncio
async def test_generate_empty_short_circuit(generator: AnswerGenerator) -> None:
    """Tests the short-circuit behavior when no context is provided.

    This ensures the generator immediately returns a fallback message without
    making an unnecessary LLM call if no relevant information is found.
    """
    ans: str = await generator.generate("Q", [])
    assert "could not find any relevant information" in ans.lower()
    generator._llm_client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_generate_exception(
    generator: AnswerGenerator, config: SearchConfig, monkeypatch: MonkeyPatch
) -> None:
    """Tests the error handling during answer generation.

    This verifies that if the LLM client raises an exception, the generator
    catches it and returns a user-friendly error message instead of crashing.
    """
    client: Any = generator._llm_client
    monkeypatch.setattr(
        client.chat.completions,
        "create",
        AsyncMock(side_effect=RuntimeError("oops")),
    )
    ans: str = await generator.generate("Q", [{"id": "1", config.content_field: "C"}])
    assert "error occurred" in ans.lower()


@pytest.mark.asyncio
async def test_generator_close(generator: AnswerGenerator) -> None:
    """Tests that the generator's close method works correctly.

    This ensures that the generator properly delegates the close operation to
    its underlying LLM client, allowing for graceful resource cleanup.
    """
    await generator.close()
    generator._llm_client.close.assert_awaited()
