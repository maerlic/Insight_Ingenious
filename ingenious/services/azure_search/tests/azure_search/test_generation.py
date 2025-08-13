# -- coding: utf-8 --

"""
FILE TEST PLAN

    AnswerGenerator:
        initialization and default prompt
        _format_context formatting (with/without content)
        generate(): success path calls, empty-context short-circuit, exception fallback
        close() closes client
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ingenious.services.azure_search.components.generation import (
    DEFAULT_RAG_PROMPT,
    AnswerGenerator,
)
from ingenious.services.azure_search.config import SearchConfig


@pytest.fixture
def generator(config: SearchConfig):
    return AnswerGenerator(config)


def test_generator_init_and_prompt(generator: AnswerGenerator):
    assert generator.rag_prompt_template == DEFAULT_RAG_PROMPT
    assert hasattr(generator, "_llm_client")


def test_format_context(generator: AnswerGenerator, config: SearchConfig):
    chunks = [
        {"id": "1", config.content_field: "A."},
        {"id": "2", config.content_field: "B."},
    ]
    out = generator._format_context(chunks)
    assert "[Source 1]" in out and "A." in out
    assert "[Source 2]" in out and "B." in out
    assert "\n---\n" in out


def test_format_context_missing_content(generator: AnswerGenerator):
    out = generator._format_context([{"id": "1"}])
    assert "N/A" in out


@pytest.mark.asyncio
async def test_generate_success(
    generator: AnswerGenerator, config: SearchConfig, monkeypatch
):
    client = generator._llm_client
    client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="Answer [Source 1]"))
            ]
        )
    )
    ans = await generator.generate("Q", [{"id": "1", config.content_field: "C"}])
    assert ans.startswith("Answer")
    client.chat.completions.create.assert_awaited()


@pytest.mark.asyncio
async def test_generate_empty_short_circuit(generator: AnswerGenerator):
    ans = await generator.generate("Q", [])
    assert "could not find any relevant information" in ans.lower()
    generator._llm_client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_generate_exception(
    generator: AnswerGenerator, config: SearchConfig, monkeypatch
):
    client = generator._llm_client
    client.chat.completions.create = AsyncMock(side_effect=RuntimeError("oops"))
    ans = await generator.generate("Q", [{"id": "1", config.content_field: "C"}])
    assert "error occurred" in ans.lower()


@pytest.mark.asyncio
async def test_generator_close(generator: AnswerGenerator):
    await generator.close()
    generator._llm_client.close.assert_awaited()
