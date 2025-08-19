"""Provides a unified search interface for Azure AI Search.

This module contains the AzureSearchProvider, a concrete implementation of a
search provider that abstracts the complexities of hybrid search (lexical + vector)
and semantic reranking. It exists to offer a simple, high-level API for
retrieving relevant documents or generating direct answers from an Azure index,
without exposing the underlying multi-step query process.

The main entry point is the AzureSearchProvider class. It depends on settings
from the `ingenious.config` module and connects to Azure AI Search.
"""

from __future__ import annotations

import asyncio
import inspect
import logging

# 1. Import TYPE_CHECKING from the typing module
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Protocol

from azure.search.documents.models import QueryType

from ingenious.config import IngeniousSettings
from ingenious.services.azure_search import SearchConfig, build_search_pipeline

if TYPE_CHECKING:
    from azure.search.documents.aio import SearchClient

    from ingenious.services.azure_search import AdvancedSearchPipeline

from ingenious.services.retrieval.errors import GenerationDisabledError

from .builders import build_search_config_from_settings
from .client_init import make_search_client

logger = logging.getLogger(__name__)


class SearchProvider(Protocol):
    """Defines the standard interface for a search provider.

    This protocol exists to create a stable contract for different search
    implementations (e.g., Azure AI Search, Elasticsearch), allowing callers
    to switch between providers without changing their code.
    """

    async def retrieve(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Retrieves a ranked list of source documents for a given query.

        This method is for fetching relevant "chunks" or documents that can be
        used for context in a RAG system or displayed as search results.
        """
        ...

    async def answer(self, query: str) -> dict[str, Any]:
        """Generates a direct answer to a query using the underlying data.

        This method encapsulates a full RAG (Retrieval-Augmented Generation)
        pipeline, returning a structured answer object.
        """
        ...

    async def close(self) -> None:
        """Closes any open connections or resources.

        This ensures graceful shutdown of network clients or other stateful
        resources used by the provider.
        """
        ...


class AzureSearchProvider:
    """Unified provider for document retrieval and RAG-based answers.

    This implementation orchestrates hybrid search and semantic reranking using the
    public Azure SDK, providing a simplified interface for complex search operations.
    It avoids private SDK methods to ensure stability and maintainability.
    """

    _cfg: SearchConfig
    _pipeline: "AdvancedSearchPipeline"
    _rerank_client: "SearchClient"

    def __init__(
        self, settings: IngeniousSettings, enable_answer_generation: bool | None = None
    ) -> None:
        """Initializes the Azure Search provider and its components.

        This constructor sets up the configuration, search pipeline, and a
        separate search client needed for the explicit semantic reranking step,
        all based on the provided application settings.

        Args:
            settings: The application settings object.
            enable_answer_generation: Overrides the answer generation setting.
        """
        self._cfg = build_search_config_from_settings(settings)
        # Allow the caller to override generation gating without changing settings schema.
        if enable_answer_generation is not None:
            self._cfg = self._cfg.copy(
                update={"enable_answer_generation": bool(enable_answer_generation)}
            )
        self._pipeline = build_search_pipeline(self._cfg)
        # Separate client for L2 (public call)
        self._rerank_client = make_search_client(self._cfg)

    async def retrieve(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Executes a hybrid search, fuses results, and applies semantic reranking.

        This is the main entry point for retrieving documents. It orchestrates
        parallel lexical and vector searches, fuses their results, and if configured,
        performs a final semantic reranking pass to improve relevance. It also handles
        cancellation of concurrent search tasks if one fails.

        Args:
            query: The user's search query.
            top_k: The final number of documents to return.

        Returns:
            A list of cleaned, ranked source documents.
        """
        k: int = top_k if top_k is not None else self._cfg.top_n_final

        # L1 - Create tasks explicitly for proper cancellation handling
        lex_task: asyncio.Task[list[dict[str, Any]]] = asyncio.create_task(
            self._pipeline.retriever.search_lexical(query)
        )
        vec_task: asyncio.Task[list[dict[str, Any]]] = asyncio.create_task(
            self._pipeline.retriever.search_vector(query)
        )

        try:
            # Use gather with return_exceptions=False (default) to fail fast
            lex: list[dict[str, Any]]
            vec: list[dict[str, Any]]
            lex, vec = await asyncio.gather(lex_task, vec_task)
        except Exception:
            # Cancel any pending task when one fails
            lex_task.cancel()
            vec_task.cancel()
            # Wait for cancellations to complete (suppress CancelledError)
            await asyncio.gather(lex_task, vec_task, return_exceptions=True)
            raise

        # DAT
        fused: list[dict[str, Any]] = await self._pipeline.fuser.fuse(query, lex, vec)

        ranked: list[dict[str, Any]]
        if self._cfg.use_semantic_ranking:
            ranked = await self._semantic_rerank(query, fused)
        else:
            for r in fused:
                r["_final_score"] = r.get("_fused_score", 0.0)
            ranked = fused

        return self._clean_sources(ranked[:k])

    async def answer(self, query: str) -> dict[str, Any]:
        """Generates a complete answer via the full RAG pipeline.

        This method executes the entire retrieval and generation pipeline to
        produce a direct answer to the user's query.

        Args:
            query: The user's question.

        Returns:
            A dictionary containing the generated answer and supporting data.

        Raises:
            GenerationDisabledError: If answer generation is not enabled.
        """
        # Short-circuit on empty/whitespace-only input
        if not query or not query.strip():
            logger.info("Blank query provided; skipping AzureSearchProvider.")
            return {
                "answer": "Please enter a question so I can search the knowledge base.",
                "source_chunks": [],
            }

        # Fail fast before doing retrieval/DAT to avoid unnecessary cost.
        if not getattr(self._cfg, "enable_answer_generation", False):
            raise GenerationDisabledError(
                detail=(
                    "answer() requires enable_answer_generation=True. "
                    "Construct AzureSearchProvider(..., enable_answer_generation=True) "
                    "or call retrieve() and run your own generator."
                ),
                snapshot={
                    "use_semantic_ranking": self._cfg.use_semantic_ranking,
                    "top_n_final": self._cfg.top_n_final,
                },
            )
        return await self._pipeline.get_answer(query)

    async def _semantic_rerank(
        self, query: str, fused_results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Performs a second-stage semantic reranking of fused search results.

        This method simulates the L2 reranker by taking the top N fused results,
        re-querying Azure AI Search with a filter for just those document IDs, and
        requesting a semantic-only search. This improves relevance by leveraging
        the semantic reranker on a smaller, high-quality set of candidates.

        Args:
            query: The user's search query.
            fused_results: The list of results from the hybrid fusion stage.

        Returns:
            A reranked list of documents, preserving the order of any documents
            not included in the reranking pass.
        """
        if not fused_results:
            return []
        MAX_RERANK: int = 50
        top: list[dict[str, Any]] = fused_results[:MAX_RERANK]
        remain: list[dict[str, Any]] = fused_results[MAX_RERANK:]

        id_field: str = self._cfg.id_field
        ids: list[str] = [str(r[id_field]) for r in top if id_field in r]
        if not ids:
            # fallback: promote fused scores
            for r in fused_results:
                r["_final_score"] = r.get("_fused_score", 0.0)
            return fused_results

        # Build OR filter with properly escaped single quotes
        # Each ID needs to be in the format: id eq 'value'
        # Single quotes in values must be doubled
        filter_clauses: list[str] = []
        for id_value in ids:
            # Escape single quotes by doubling them
            escaped_id: str = id_value.replace("'", "''")
            filter_clauses.append(f"{id_field} eq '{escaped_id}'")

        # Join with ' or ' to create the final filter
        filter_query: str = " or ".join(filter_clauses)

        results_iter: AsyncIterator[dict[str, Any]] = await self._rerank_client.search(
            search_text=query,
            filter=filter_query,
            query_type=QueryType.SEMANTIC,
            semantic_configuration_name=self._cfg.semantic_configuration_name,
            top=len(ids),
        )

        fused_map: dict[Any, dict[str, Any]] = {
            r[id_field]: r for r in top if id_field in r
        }
        reranked: list[dict[str, Any]] = []
        matched_ids: set[Any] = set()

        async for r in results_iter:
            rid: Any | None = r.get(id_field)
            if rid in fused_map:
                merged: dict[str, Any] = fused_map[rid].copy()
                merged.update(r)
                merged["_final_score"] = merged.get("@search.reranker_score")
                reranked.append(merged)
                matched_ids.add(rid)

        # Preserve any unmatched documents from top 50 with fallback score
        for doc in top:
            doc_id: Any | None = doc.get(id_field)
            if doc_id not in matched_ids:
                preserved: dict[str, Any] = doc.copy()
                preserved["_final_score"] = preserved.get("_fused_score", 0.0)
                reranked.append(preserved)

        return reranked + remain

    def _clean_sources(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Removes internal and temporary fields from result documents.

        This method exists to present a clean, stable contract to the API consumer,
        stripping away implementation details like intermediate scores, vector data,
        and Azure-specific metadata fields before returning the final results.

        Args:
            chunks: A list of result documents with internal fields.

        Returns:
            A cleaned list of result documents.
        """
        cleaned: list[dict[str, Any]] = []
        for c in chunks:
            d: dict[str, Any] = c.copy()
            # remove pipeline internals & azure metadata
            for k in (
                "_retrieval_score",
                "_normalized_score",
                "_fused_score",
                "@search.score",
                "@search.reranker_score",
                "@search.captions",
                "_final_score",
            ):
                d.pop(k, None)
            d.pop(self._cfg.vector_field, None)
            cleaned.append(d)
        return cleaned

    async def close(self) -> None:
        """Closes the underlying search pipeline and reranking client.

        This method ensures that all network connections are gracefully terminated.
        It is designed to be resilient, checking for the existence of `close`
        methods and handling both synchronous and asynchronous versions.
        """
        # Tolerate both async and sync close()
        maybe_close: Callable[[], Any] | None = getattr(self._pipeline, "close", None)
        if callable(maybe_close):
            res: Any = maybe_close()
            if inspect.isawaitable(res):
                await res
        maybe_close2: Callable[[], Any] | None = getattr(
            self._rerank_client, "close", None
        )
        if callable(maybe_close2):
            res2: Any = maybe_close2()
            if inspect.isawaitable(res2):
                await res2
