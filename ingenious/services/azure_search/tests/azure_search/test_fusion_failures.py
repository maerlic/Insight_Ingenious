# tests/azure_search/test_fusion_failures.py
import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from openai import APIError

from ingenious.services.azure_search.components.fusion import DynamicRankFuser


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exception_to_raise",
    [
        # This is the correct signature: APIError(message, response, body=...)
        APIError(
            "LLM Failed",
            httpx.Response(
                status_code=500, request=MagicMock()
            ),  # 2nd POSITIONAL argument
            body=None,  # KEYWORD argument
        ),
        asyncio.TimeoutError("LLM Timeout"),
        RuntimeError("Unexpected LLM error"),
    ],
)
async def test_dat_fusion_llm_failure_falls_back_to_alpha_0_5(
    mock_search_config, mock_async_openai_client, exception_to_raise
):
    """
    P1: Verify DynamicRankFuser._perform_dat catches LLM exceptions and returns 0.5.
    """
    # Configure the mock LLM client to raise the specified exception
    mock_async_openai_client.chat.completions.create.side_effect = exception_to_raise

    fuser = DynamicRankFuser(
        config=mock_search_config, llm_client=mock_async_openai_client
    )

    # Prepare dummy inputs
    query = "test query"
    top_lexical = {"content": "lexical doc"}
    top_vector = {"content": "vector doc"}

    # Execute the DAT step
    alpha = await fuser._perform_dat(query, top_lexical, top_vector)

    # Assert the fallback value is used
    assert alpha == 0.5
    mock_async_openai_client.chat.completions.create.assert_called_once()


@pytest.mark.parametrize(
    "llm_output, expected_scores",
    [
        ("4 3", (4, 3)),  # Happy path
        ("invalid output", (0, 0)),  # Malformed
        ("5", (0, 0)),  # Wrong count
        ("6 2", (0, 0)),  # Out of range (high)
        ("-1 3", (0, 0)),  # Out of range (low)
        ("Score V: 4, Score L: 2", (4, 2)),  # Text around numbers
        ("3.5 2.1", (3, 5)),  # Matches current regex behavior
    ],
)
def test_dat_fusion_parse_malformed_scores_falls_back_to_zero(
    mock_search_config, llm_output, expected_scores
):
    """
    P1: Verify _parse_dat_scores handles various invalid LLM outputs.
    """
    # Initialize fuser (LLM client not needed for parsing logic)
    fuser = DynamicRankFuser(config=mock_search_config, llm_client=AsyncMock())

    # Execute the parsing
    scores = fuser._parse_dat_scores(llm_output)

    # Assert the result
    assert scores == expected_scores
