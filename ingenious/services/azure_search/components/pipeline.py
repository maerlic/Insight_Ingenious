"""Orchestrate a multi-stage search pipeline using Azure AI Search.

This module provides the `AdvancedSearchPipeline` class, which integrates several
components to execute a sophisticated search and retrieval-augmented generation (RAG)
workflow. The pipeline is designed to enhance search relevance by combining lexical
and vector search, fusing the results, applying an optional semantic re-ranking
step, and finally generating a concise answer from the top documents.

The primary entry point is the `build_search_pipeline` factory function, which
constructs and configures the pipeline based on a `SearchConfig` object.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import TYPE_CHECKING, Any, Coroutine

from azure.search.documents.models import QueryType

from ingenious.services.azure_search.components.fusion import DynamicRankFuser
from ingenious.services.azure_search.components.generation import AnswerGenerator
from ingenious.services.azure_search.components.retrieval import AzureSearchRetriever
from ingenious.services.azure_search.config import SearchConfig
from ingenious.services.retrieval.errors import GenerationDisabledError

if TYPE_CHECKING:
    from azure.search.documents.aio import SearchClient


logger = logging.getLogger(__name__)


class AdvancedSearchPipeline:
    """Orchestrates the multi-stage Advanced AI Search pipeline.

    Pipeline Flow: L1 Retrieval -> DAT Fusion -> L2 Semantic Ranking -> RAG Generation.
    """

    _config: SearchConfig
    retriever: AzureSearchRetriever
    fuser: DynamicRankFuser
    answer_generator: AnswerGenerator | None
    _rerank_client: SearchClient

    def __init__(
        self,
        config: SearchConfig,
        retriever: AzureSearchRetriever,
        fuser: DynamicRankFuser,
        answer_generator: AnswerGenerator | None,
        rerank_client: SearchClient | None = None,
    ) -> None:
        """Initializes the pipeline with its core components and configuration.

        This constructor sets up the retriever, fuser, and optional answer generator.
        It also ensures a dedicated `SearchClient` is available for the semantic
        re-ranking step, creating one if not provided.

        Args:
            config: The search configuration object.
            retriever: The component for L1 lexical and vector retrieval.
            fuser: The component for fusing retrieval results.
            answer_generator: The component for generating answers (RAG).
            rerank_client: An optional, pre-configured client for re-ranking.
        """
        self._config = config
        self.retriever = retriever
        self.fuser = fuser
        self.answer_generator = answer_generator

        # A dedicated SearchClient is needed for the L2 Semantic Ranking step
        if rerank_client is None:
            from ..client_init import make_search_client

            self._rerank_client = make_search_client(config)
        else:
            self._rerank_client = rerank_client

    async def _apply_semantic_ranking(
        self, query: str, fused_results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Applies Azure AI Search Semantic Ranking as an L2 re-ranker.

        This method re-ranks the top results from the fusion stage using Azure's
        semantic ranker. It employs a workaround by executing a new search query
        filtered to the document IDs from the fused results, thereby forcing the
        semantic ranker to score and re-order only that specific set.

        Args:
            query: The original user search query.
            fused_results: The list of documents after the fusion stage.

        Returns:
            A re-ranked list of documents, with the most semantically relevant
            results first, combined with any documents that were not re-ranked.
        """
        # Semantic Ranker is optimized for the top 50 results.
        MAX_RERANK_DOCS = 50
        docs_to_rerank: list[dict[str, Any]] = fused_results[:MAX_RERANK_DOCS]
        remaining_docs: list[dict[str, Any]] = fused_results[MAX_RERANK_DOCS:]

        if not docs_to_rerank:
            return fused_results  # Return original list if empty

        logger.info(
            f"Applying Semantic Ranking (L2) to the top {len(docs_to_rerank)} fused results."
        )

        # 1. Extract document IDs
        id_field: str = self._config.id_field
        doc_ids: list[str] = [
            str(result[id_field]) for result in docs_to_rerank if id_field in result
        ]

        if not doc_ids:
            logger.warning("Could not extract document IDs. Skipping Semantic Ranking.")
            return fused_results

        # 2. Construct the filter clause to restrict the search space
        # Using OR clauses with proper escaping to handle IDs with commas and quotes
        # Build filter as: id eq 'id1' or id eq 'id2' or id eq 'id3'
        or_clauses: list[str] = []
        for doc_id in doc_ids:
            # Escape single quotes by doubling them for OData
            escaped_id = doc_id.replace("'", "''")
            or_clauses.append(f"{id_field} eq '{escaped_id}'")

        filter_query: str = " or ".join(or_clauses)

        # 3. Execute the restricted semantic search
        try:
            search_results = await self._rerank_client.search(
                search_text=query,
                filter=filter_query,
                query_type=QueryType.SEMANTIC,
                semantic_configuration_name=self._config.semantic_configuration_name,
                top=len(doc_ids),  # Request all documents in the restricted set
            )

            # 4. Process results and map back to original data structure
            # The results are inherently sorted by @search.reranker_score
            reranked_results: list[dict[str, Any]] = []

            # We need to ensure we retain metadata from the fusion step (like _retrieval_type)
            id_field = self._config.id_field
            fused_data_map: dict[str, dict[str, Any]] = {
                str(r[id_field]): r
                for r in docs_to_rerank
                if id_field in r and r.get(id_field) is not None
            }
            matched_ids: set[str] = set()

            async for result in search_results:
                doc_id_value = result.get(id_field)
                if doc_id_value is not None:
                    doc_id = str(doc_id_value)  # Ensure the type is string
                    if doc_id in fused_data_map:
                        # Start with the original fused data
                        merged_result: dict[str, Any] = fused_data_map[doc_id].copy()
                        # Update with fields returned by the semantic search
                        merged_result.update(result)
                        # The Semantic Ranker score is the new primary score
                        merged_result["_final_score"] = merged_result.get(
                            "@search.reranker_score"
                        )
                        reranked_results.append(merged_result)
                        matched_ids.add(doc_id)

            logger.info("Semantic Ranking complete.")
            # Keep any of the top-50 that weren't returned by the reranker
            for r in docs_to_rerank:
                if str(r.get(id_field)) not in matched_ids:
                    # Preserve unmatched docs with their fused score as final score
                    preserved: dict[str, Any] = r.copy()
                    preserved["_final_score"] = preserved.get("_fused_score", 0.0)
                    reranked_results.append(preserved)

            # Append the documents beyond top-50 unchanged
            return reranked_results + remaining_docs

        except Exception as e:
            logger.error(
                f"Error during Semantic Ranking execution: {e}. Falling back to DAT fused results."
            )
            # If reranking fails, fall back to the DAT scores
            for result in fused_results:
                result["_final_score"] = result.get("_fused_score", 0.0)
            return fused_results

    async def get_answer(self, query: str) -> dict[str, Any]:
        """Executes the full Advanced Search pipeline for a given query.

        This is the main entry point for running a search. It orchestrates the
        full sequence of operations:
        1.  Parallel L1 retrieval (lexical and vector).
        2.  Result fusion (DAT).
        3.  Optional L2 semantic re-ranking.
        4.  Optional RAG-based answer generation.

        Args:
            query: The user's search query.

        Returns:
            A dictionary containing the generated answer and the list of source
            document chunks that support it.
        """
        # Short-circuit on empty/whitespace-only input
        if not query or not query.strip():
            logger.info("Blank query provided; skipping Advanced Search Pipeline.")
            return {
                "answer": "Please enter a question so I can search the knowledge base.",
                "source_chunks": [],
            }

        # Fail fast before doing retrieval/DAT to avoid unnecessary cost.
        if not getattr(self._config, "enable_answer_generation", False):
            raise GenerationDisabledError(
                detail=(
                    "get_answer() requires enable_answer_generation=True. "
                    "Construct SearchConfig(..., enable_answer_generation=True) and pass it to the pipeline."
                ),
                snapshot={
                    "use_semantic_ranking": self._config.use_semantic_ranking,
                    "top_n_final": self._config.top_n_final,
                },
            )

        logger.info(f"Starting Advanced Search Pipeline for query: '{query}'")

        # Step 1: L1 Retrieval (Parallel Lexical/BM25 and Vector/Dense)
        try:
            lexical_results: list[dict[str, Any]]
            vector_results: list[dict[str, Any]]
            lexical_results, vector_results = await asyncio.gather(
                self.retriever.search_lexical(query),
                self.retriever.search_vector(query),
            )
        except Exception as e:
            logger.error(f"Error during L1 retrieval phase: {e}")
            raise RuntimeError("L1 Retrieval failed.") from e

        # Step 2: Fusion (DAT)
        try:
            fused_or_coro: (
                list[dict[str, Any]] | Coroutine[Any, Any, list[dict[str, Any]]]
            ) = self.fuser.fuse(query, lexical_results, vector_results)
            fused_results: list[dict[str, Any]] = (
                await fused_or_coro
                if inspect.isawaitable(fused_or_coro)
                else fused_or_coro
            )
        except Exception as e:
            logger.error(f"Error during DAT fusion phase: {e}")
            raise RuntimeError("DAT Fusion failed.") from e

        # Step 3: L2 Re-ranking (Optional Semantic Ranking)
        final_ranked_results: list[dict[str, Any]]
        if self._config.use_semantic_ranking:
            # We pass the fused results (up to top 50) to the semantic ranker
            final_ranked_results = await self._apply_semantic_ranking(
                query, fused_results
            )
        else:
            logger.info(
                "Skipping Semantic Ranking (L2) as configured. Using DAT fused scores."
            )
            final_ranked_results = fused_results
            # Use the fused score as the final score
            for result in final_ranked_results:
                result["_final_score"] = result.get("_fused_score", 0.0)

        # Step 4: Select Top N results
        top_n_chunks: list[dict[str, Any]] = final_ranked_results[
            : self._config.top_n_final
        ]

        if not top_n_chunks:
            logger.info("No relevant context found after ranking.")
            return {
                "answer": "I could not find any relevant information in the knowledge base to answer your question.",
                "source_chunks": [],
            }

        # Handle the 'None' case.
        if self.answer_generator is None:
            logger.error(
                "Attempted to generate an answer, but the generator is not configured."
            )
            raise GenerationDisabledError("Answer generation is not enabled.")

        answer: str = ""
        try:
            answer = await self.answer_generator.generate(query, top_n_chunks)
        except Exception as e:
            logger.error(f"Error during generation phase: {e}")
            raise RuntimeError("Answer Generation failed.") from e

        logger.info("Advanced Search Pipeline complete.")
        return {"answer": answer, "source_chunks": self._clean_sources(top_n_chunks)}

    def _clean_sources(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Removes internal metadata and large fields from the final source documents.

        This helper function prepares the source chunks for the final output by
        stripping away pipeline-internal data (like intermediate scores) and
        heavyweight fields (like embedding vectors) that are not needed by the
        end-user or calling application.

        Args:
            chunks: The list of source document chunks to clean.

        Returns:
            A cleaned list of source document chunks.
        """
        cleaned_chunks: list[dict[str, Any]] = []
        for chunk in chunks:
            cleaned: dict[str, Any] = chunk.copy()
            # Remove only truly internal scores, keep everything needed for display
            cleaned.pop("_retrieval_score", None)  # Internal fusion input
            cleaned.pop("_normalized_score", None)  # Internal fusion normalized
            cleaned.pop(
                "_fused_score", None
            )  # Internal fusion output (replaced by _final_score)
            # Keep these for display/debugging:
            # - _final_score (primary ranking score)
            # - _retrieval_type (shows retrieval method)
            # - _bm25_score_raw (raw BM25 score)
            # - _vector_score_raw (raw vector score)
            # - @search.reranker_score (semantic ranker score if present)

            # Remove large fields
            cleaned.pop(self._config.vector_field, None)

            # Remove redundant Azure Search metadata
            cleaned.pop("@search.score", None)
            cleaned.pop(
                "@search.reranker_score", None
            )  # Also remove this as it's redundant with _final_score
            cleaned.pop("@search.captions", None)

            cleaned_chunks.append(cleaned)
        return cleaned_chunks

    async def close(self) -> None:
        """Closes all underlying asynchronous clients gracefully.

        This method should be called to ensure that all network connections
        held by the pipeline's components (retriever, fuser, generator, etc.)
        are properly terminated.
        """
        await self.retriever.close()
        await self.fuser.close()
        if self.answer_generator is not None:
            await self.answer_generator.close()
        await self._rerank_client.close()


def build_search_pipeline(config: SearchConfig) -> AdvancedSearchPipeline:
    """Constructs and configures the AdvancedSearchPipeline via a factory function.

    This function centralizes the instantiation and dependency injection for the
    entire search pipeline. It creates the necessary retriever, fuser, and
    answer generator components based on the provided configuration, then
    assembles them into a ready-to-use `AdvancedSearchPipeline` instance.

    Args:
        config: A `SearchConfig` object containing all necessary settings.

    Returns:
        A fully initialized `AdvancedSearchPipeline` instance.

    Raises:
        ValueError: If semantic ranking is enabled but no configuration name
            is provided.
    """
    logger.info("Building Advanced Search Pipeline via factory...")

    # Validation specific to pipeline construction
    if config.use_semantic_ranking and not config.semantic_configuration_name:
        raise ValueError(
            "Configuration Error: 'use_semantic_ranking' is True, but 'semantic_configuration_name' is not provided."
        )

    # Initialize components
    retriever = AzureSearchRetriever(config)
    fuser = DynamicRankFuser(config)
    answer_generator: AnswerGenerator | None = (
        AnswerGenerator(config)
        if getattr(config, "enable_answer_generation", False)
        else None
    )

    # Assemble the pipeline
    pipeline = AdvancedSearchPipeline(
        config=config,
        retriever=retriever,
        fuser=fuser,
        answer_generator=answer_generator,
    )

    logger.info("Advanced Search Pipeline built successfully.")
    return pipeline
