import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

# Import or define Azure exceptions
try:
    from azure.core.exceptions import HttpResponseError, ResourceNotFoundError
except ImportError:
    # Define dummy exceptions if azure-core is not installed
    class ResourceNotFoundError(Exception):
        pass

    class HttpResponseError(Exception):
        pass


# Mocking OpenAI exceptions
try:
    from openai import APIConnectionError, AuthenticationError, BadRequestError
except ImportError:
    # Define dummy exception classes if openai library is not installed
    class AuthenticationError(Exception):
        def __init__(self, message, *args, **kwargs):
            super().__init__(message)

    class APIConnectionError(Exception):
        def __init__(self, *args, **kwargs):
            super().__init__("API Connection Error")

    class BadRequestError(Exception):
        def __init__(self, message, *args, **kwargs):
            super().__init__(message)


from ingenious.services.azure_search.components.retrieval import AzureSearchRetriever
from ingenious.services.azure_search.config import SearchConfig


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exception_to_raise, expected_exception",
    [
        (ResourceNotFoundError("Index not found (404)"), ResourceNotFoundError),
        (HttpResponseError("Bad Request - Invalid syntax (400)"), HttpResponseError),
        (HttpResponseError("Service Unavailable (503)"), HttpResponseError),
        (ConnectionError("Network issue"), ConnectionError),
    ],
)
async def test_retrieval_handles_azure_search_api_errors(
    config: SearchConfig, exception_to_raise, expected_exception
):
    """
    Verify that AzureSearchRetriever propagates non-transient exceptions from the
    Azure Search client during both lexical and vector search paths.
    """
    # Setup: Mock clients
    mock_search_client = MagicMock()
    # Configure the search client to raise the specific exception
    mock_search_client.search = AsyncMock(side_effect=exception_to_raise)
    mock_embedding_client = MagicMock()

    retriever = AzureSearchRetriever(
        config, search_client=mock_search_client, embedding_client=mock_embedding_client
    )

    # Test Lexical Path
    with pytest.raises(expected_exception) as excinfo_lex:
        await retriever.search_lexical("test query")
    assert str(exception_to_raise) in str(excinfo_lex.value)

    # Test Vector Path
    # Ensure the embedding step succeeds so the error occurs during the search step
    mock_embedding_client.embeddings.create = AsyncMock(
        return_value=MagicMock(data=[MagicMock(embedding=[0.1, 0.2])])
    )

    with pytest.raises(expected_exception) as excinfo_vec:
        await retriever.search_vector("test query")
    assert str(exception_to_raise) in str(excinfo_vec.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exception_to_raise, expected_exception",
    [
        (
            AuthenticationError(
                "Invalid API Key (401)", response=MagicMock(status_code=401), body={}
            ),
            AuthenticationError,
        ),
        (asyncio.TimeoutError("Embedding request timed out"), asyncio.TimeoutError),
        (APIConnectionError(request=MagicMock()), APIConnectionError),
        # Simulating a token limit exceeded error (400 Bad Request)
        (
            BadRequestError(
                "Token limit exceeded", response=MagicMock(status_code=400), body={}
            ),
            BadRequestError,
        ),
    ],
)
async def test_retrieval_handles_openai_embedding_errors(
    config: SearchConfig, exception_to_raise, expected_exception
):
    """
    Verify that AzureSearchRetriever.search_vector propagates exceptions raised
    during the OpenAI embedding generation step (_generate_embedding).
    """
    # Setup: Mock clients
    mock_search_client = MagicMock()  # Should not be called if embedding fails
    mock_embedding_client = MagicMock()
    # Configure the embedding client's create method to raise the exception
    mock_embedding_client.embeddings.create = AsyncMock(side_effect=exception_to_raise)

    retriever = AzureSearchRetriever(
        config, search_client=mock_search_client, embedding_client=mock_embedding_client
    )

    # Execute Vector Path
    with pytest.raises(expected_exception) as excinfo:
        await retriever.search_vector("test query that causes error")

    # Assert the correct exception propagated and the search client was not called
    # For APIConnectionError, we can't check the message since it doesn't accept one
    if expected_exception != APIConnectionError:
        assert str(exception_to_raise) in str(excinfo.value)
    mock_search_client.search.assert_not_called()
