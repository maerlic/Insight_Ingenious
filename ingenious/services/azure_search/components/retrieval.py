"""Provide a retriever for Azure AI Search supporting lexical and vector queries.

This module defines the AzureSearchRetriever class, which acts as a client
for performing L1 retrieval from an Azure AI Search index. It is designed to
abstract the details of constructing and executing both keyword-based (BM25)
and vector-based (ANN) searches.

The primary entry point is the AzureSearchRetriever class. Instantiate it with
a configuration object and then use its `search_lexical` or `search_vector`
methods to retrieve documents. It manages its own search and embedding clients.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from azure.search.documents.models import (
    QueryType,
    VectorizedQuery,
)

try:
    from ingenious.services.azure_search.config import SearchConfig
except ImportError:
    from ..config import SearchConfig

if TYPE_CHECKING:
    from azure.search.documents.aio import SearchClient
    from openai import AsyncOpenAI


logger = logging.getLogger(__name__)


class AzureSearchRetriever:
    """Handles the L1 retrieval stage using Azure AI Search.

    Provides methods for executing pure lexical (BM25) and pure vector searches.
    """

    _search_client: "SearchClient"
    _embedding_client: "AsyncOpenAI"

    def __init__(
        self,
        config: SearchConfig,
        search_client: Optional["SearchClient"] = None,
        embedding_client: Optional["AsyncOpenAI"] = None,
    ) -> None:
        """Initialize the retriever with configuration and optional clients.

        This constructor sets up the Azure Search and OpenAI embedding clients.
        If clients are not provided, it creates them dynamically using the
        configuration. This allows for dependency injection during testing.
        """
        self._config = config
        if search_client is None or embedding_client is None:
            from ..client_init import make_async_openai_client, make_search_client

            self._search_client = search_client or make_search_client(config)
            self._embedding_client = embedding_client or make_async_openai_client(
                config
            )
        else:
            self._search_client = search_client
            self._embedding_client = embedding_client

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding vector for the input text using the OpenAI client.

        This is a helper method to encapsulate the call to the embedding service.
        """
        response = await self._embedding_client.embeddings.create(
            input=[text], model=self._config.embedding_deployment_name
        )
        return response.data[0].embedding

    async def search_lexical(self, query: str) -> List[Dict[str, Any]]:
        """Perform a pure BM25 keyword search (sparse retrieval).

        This method executes a search against Azure AI Search using only the
        text query, leveraging the BM25 ranking algorithm. It is used for
        retrieving documents based on keyword relevance.
        """
        logger.info(
            f"Executing lexical (BM25) search (Top K: {self._config.top_k_retrieval})"
        )

        # Execute the search with only the search_text parameter for BM25 ranking
        search_results = await self._search_client.search(
            search_text=query,
            vector_queries=None,
            top=self._config.top_k_retrieval,
            query_type=QueryType.SIMPLE,
        )

        results_list: list[dict[str, Any]] = []
        async for result in search_results:
            # Store the original score for later fusion
            raw = result.get("@search.score")
            result["_retrieval_score"] = raw
            result["_bm25_score"] = raw
            result["_retrieval_type"] = "lexical_bm25"
            results_list.append(result)

        logger.info(f"Lexical search returned {len(results_list)} results.")
        return results_list

    async def search_vector(self, query: str) -> List[Dict[str, Any]]:
        """Perform a pure vector similarity search (dense retrieval).

        This method first generates an embedding for the query text and then
        searches for the most similar document vectors in the index using
        Approximate Nearest Neighbor (ANN) search. It's used for semantic
        relevance.
        """
        # Short-circuit for empty query
        if not query or not query.strip():
            logger.info("Empty query provided, returning empty results.")
            return []

        logger.info(
            f"Executing vector (Dense) search (Top K: {self._config.top_k_retrieval})"
        )

        # Generate the query embedding
        query_embedding = await self._generate_embedding(query)

        # Define the vector query
        vector_query = VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=self._config.top_k_retrieval,
            fields=self._config.vector_field,
            exhaustive=True,  # Ensures accurate similarity scores across the index
        )

        # Execute the search with only the vector_queries parameter (search_text=None)
        search_results = await self._search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            top=self._config.top_k_retrieval,
        )

        results_list: list[dict[str, Any]] = []
        async for result in search_results:
            # Store the original score for later fusion
            raw = result.get("@search.score")
            result["_retrieval_score"] = raw
            result["_vector_score"] = raw
            result["_retrieval_type"] = "vector_dense"
            results_list.append(result)

        logger.info(f"Vector search returned {len(results_list)} results.")
        return results_list

    async def close(self) -> None:
        """Close the underlying asynchronous search and embedding clients.

        This should be called to gracefully release network resources.
        """
        await self._search_client.close()
        await self._embedding_client.close()
