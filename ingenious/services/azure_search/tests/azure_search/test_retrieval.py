import pytest
from unittest.mock import AsyncMock, MagicMock
from types import SimpleNamespace

# Imports rely on the structure defined in conftest.py
try:
    from ingenious.services.azure_search.config import SearchConfig
    from ingenious.services.azure_search.components.retrieval import AzureSearchRetriever
    # Attempt to import Azure SDK models for type checking and parameter verification
    from azure.search.documents.models import QueryType, VectorizedQuery
except ImportError:
    # Mock the Azure SDK models if the SDK is not installed
    QueryType = MagicMock()
    QueryType.SIMPLE = "simple"
    VectorizedQuery = MagicMock()


@pytest.fixture
def mock_embedding_response():
    """Mocks the OpenAI embedding API response structure."""
    # response.data[0].embedding
    return SimpleNamespace(
        data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])]
    )

class AsyncIter:
    """Helper class to mock asynchronous iterators (like SearchClient results)."""
    def __init__(self, items):
        self.items = items
    async def __aiter__(self):
        for item in self.items:
            yield item

def test_retriever_initialization(config, mock_search_client, mock_openai_client, mock_azure_credential):
    """Test that the retriever initializes clients using the configuration."""
    # The retriever fixture already initializes it, we verify the state here.
    retriever = AzureSearchRetriever(config)

    # Verify clients are assigned (they will be the mocked instances from the fixtures)
    assert retriever._search_client == mock_search_client
    assert retriever._embedding_client == mock_openai_client
    
    # Note: We cannot easily assert the calls to the mocked classes (like SearchClient(...)) 
    # here because the patching strategy in conftest replaces the class with a MagicMock 
    # that returns the instance (mock_search_client).

# --- Test Lexical Search ---

@pytest.mark.asyncio
async def test_search_lexical_success(retriever: AzureSearchRetriever, config: SearchConfig, mock_search_client, fake_results):
    """Test successful execution of lexical search."""
    query = "test query"
    expected_results = fake_results(5, prefix="lexical", start_score=10.0)

    # Configure the mock search client to return an async iterator
    mock_search_client.search.return_value = AsyncIter(expected_results)

    results = await retriever.search_lexical(query)

    # Verify the call to the search client
    mock_search_client.search.assert_called_once_with(
        search_text=query,
        vector_queries=None,
        top=config.top_k_retrieval,
        query_type=QueryType.SIMPLE,
    )

    # Verify the results processing
    assert len(results) == 5
    assert results[0][config.id_field] == "lexical_1"
    # Check that pipeline metadata was added correctly
    assert results[0]["_retrieval_type"] == "lexical_bm25"
    assert results[0]["_retrieval_score"] == 10.0

@pytest.mark.asyncio
async def test_search_lexical_empty(retriever: AzureSearchRetriever, mock_search_client):
    """Test lexical search returning no results."""
    mock_search_client.search.return_value = AsyncIter([])
    results = await retriever.search_lexical("query")
    assert results == []

# --- Test Vector Search ---

@pytest.mark.asyncio
async def test_generate_embedding(retriever: AzureSearchRetriever, config: SearchConfig, mock_openai_client, mock_embedding_response):
    """Test the embedding generation helper method."""
    text = "embed this text"
    mock_openai_client.embeddings.create.return_value = mock_embedding_response

    embedding = await retriever._generate_embedding(text)

    # Verify the call to the OpenAI client
    mock_openai_client.embeddings.create.assert_called_once_with(
        input=[text],
        model=config.embedding_deployment_name
    )
    assert embedding == [0.1, 0.2, 0.3]

@pytest.mark.asyncio
async def test_search_vector_success(retriever: AzureSearchRetriever, config: SearchConfig, mock_search_client, mock_openai_client, mock_embedding_response, fake_results):
    """Test successful execution of vector search."""
    query = "vector query"
    expected_embedding = mock_embedding_response.data[0].embedding
    expected_results = fake_results(5, prefix="vector", start_score=0.9)

    # Configure mocks
    mock_openai_client.embeddings.create.return_value = mock_embedding_response
    mock_search_client.search.return_value = AsyncIter(expected_results)

    results = await retriever.search_vector(query)

    # Verify embedding generation occurred
    mock_openai_client.embeddings.create.assert_called_once()

    # Verify the call to the search client
    mock_search_client.search.assert_called_once()
    args, kwargs = mock_search_client.search.call_args

    assert kwargs['search_text'] is None
    assert kwargs['top'] == config.top_k_retrieval

    # Inspect the VectorizedQuery object construction
    vector_queries = kwargs['vector_queries']
    assert len(vector_queries) == 1
    vq = vector_queries[0]

    # Handle both real and mocked VectorizedQuery
    if not isinstance(VectorizedQuery, MagicMock):
         assert isinstance(vq, VectorizedQuery)

    assert vq.vector == expected_embedding
    assert vq.k_nearest_neighbors == config.top_k_retrieval
    assert vq.fields == config.vector_field
    assert vq.exhaustive is True

    # Verify the results processing
    assert len(results) == 5
    assert results[0][config.id_field] == "vector_1"
    # Check metadata
    assert results[0]["_retrieval_type"] == "vector_dense"
    assert results[0]["_retrieval_score"] == 0.9

# --- Test Close ---

@pytest.mark.asyncio
async def test_retriever_close(retriever: AzureSearchRetriever, mock_search_client, mock_openai_client):
    """Test that the close method closes both underlying clients."""
    await retriever.close()

    mock_search_client.close.assert_awaited_once()
    mock_openai_client.close.assert_awaited_once()