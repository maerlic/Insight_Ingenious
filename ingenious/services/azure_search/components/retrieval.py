# Insight_Ingenious/ingenious/services/azure_search/components/retrieval.py

import logging
from typing import List, Dict, Any
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import (
    VectorizedQuery,
    QueryType,
)
from openai import AsyncAzureOpenAI

# Import the configuration model (assuming standard package structure)
try:
    from ingenious.services.azure_search.config import SearchConfig
except ImportError:
    from ..config import SearchConfig

logger = logging.getLogger(__name__)

class AzureSearchRetriever:
    """
    Handles the L1 retrieval stage using Azure AI Search.
    Provides methods for executing pure lexical (BM25) and pure vector searches.
    """

    def __init__(self, config: SearchConfig):
        self._config = config
        self._search_client = self._initialize_search_client()
        self._embedding_client = self._initialize_embedding_client()

    def _initialize_search_client(self) -> SearchClient:
        """Initializes the asynchronous Azure Search Client."""
        return SearchClient(
            endpoint=self._config.search_endpoint,
            index_name=self._config.search_index_name,
            credential=AzureKeyCredential(self._config.search_key.get_secret_value()),
        )

    def _initialize_embedding_client(self) -> AsyncAzureOpenAI:
        """Initializes the asynchronous Azure OpenAI client for embeddings."""
        return AsyncAzureOpenAI(
            azure_endpoint=self._config.openai_endpoint,
            api_key=self._config.openai_key.get_secret_value(),
            api_version=self._config.openai_version,
        )

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generates an embedding vector for the input text."""
        response = await self._embedding_client.embeddings.create(
            input=[text], model=self._config.embedding_deployment_name
        )
        return response.data[0].embedding

    async def search_lexical(self, query: str) -> List[Dict[str, Any]]:
        """
        Performs a pure BM25 keyword search. (Sparse Retrieval)
        """
        logger.info(f"Executing lexical (BM25) search (Top K: {self._config.top_k_retrieval})")
        
        # Execute the search with only the search_text parameter for BM25 ranking
        search_results = await self._search_client.search(
            search_text=query,
            vector_queries=None,
            top=self._config.top_k_retrieval,
            query_type=QueryType.SIMPLE,
        )

        results_list = []
        async for result in search_results:
            # Store the original score for later fusion
            result["_retrieval_score"] = result.get("@search.score")
            result["_retrieval_type"] = "lexical_bm25"
            results_list.append(result)
        
        logger.info(f"Lexical search returned {len(results_list)} results.")
        return results_list

    async def search_vector(self, query: str) -> List[Dict[str, Any]]:
        """
        Performs a pure vector similarity search (ANN). (Dense Retrieval)
        """
        logger.info(f"Executing vector (Dense) search (Top K: {self._config.top_k_retrieval})")

        # Generate the query embedding
        query_embedding = await self._generate_embedding(query)

        # Define the vector query
        vector_query = VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=self._config.top_k_retrieval,
            fields=self._config.vector_field,
            exhaustive=True # Ensures accurate similarity scores across the index
        )

        # Execute the search with only the vector_queries parameter (search_text=None)
        search_results = await self._search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            top=self._config.top_k_retrieval,
        )

        results_list = []
        async for result in search_results:
            # Store the original score for later fusion
            result["_retrieval_score"] = result.get("@search.score")
            result["_retrieval_type"] = "vector_dense"
            results_list.append(result)

        logger.info(f"Vector search returned {len(results_list)} results.")
        return results_list

    async def close(self):
        """Closes the underlying clients."""
        await self._search_client.close()
        await self._embedding_client.close()