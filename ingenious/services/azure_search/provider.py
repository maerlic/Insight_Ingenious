import asyncio
import inspect
from typing import Any, Dict, List, Protocol

from azure.search.documents.models import QueryType

from ingenious.config import IngeniousSettings
from ingenious.services.azure_search import build_search_pipeline

from .builders import build_search_config_from_settings
from .client_init import make_search_client


class SearchProvider(Protocol):
    async def retrieve(self, query: str, top_k: int = 10) -> List[dict]: ...
    async def answer(self, query: str) -> dict: ...
    async def close(self) -> None: ...


class AzureSearchProvider:
    """
    Unified provider that can return ranked chunks (retrieve) or full RAG answer (answer).
    Implementation avoids private methods and replicates the L2 semantic re-rank using public SDK.
    """

    def __init__(self, settings: IngeniousSettings):
        self._cfg = build_search_config_from_settings(settings)
        self._pipeline = build_search_pipeline(self._cfg)
        # Separate client for L2 (public call)
        self._rerank_client = make_search_client(self._cfg)

    async def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        # L1 - Create tasks explicitly for proper cancellation handling
        lex_task = asyncio.create_task(self._pipeline.retriever.search_lexical(query))
        vec_task = asyncio.create_task(self._pipeline.retriever.search_vector(query))

        try:
            # Use gather with return_exceptions=False (default) to fail fast
            lex, vec = await asyncio.gather(lex_task, vec_task)
        except Exception:
            # Cancel any pending task when one fails
            lex_task.cancel()
            vec_task.cancel()
            # Wait for cancellations to complete (suppress CancelledError)
            await asyncio.gather(lex_task, vec_task, return_exceptions=True)
            raise

        # DAT
        fused = await self._pipeline.fuser.fuse(query, lex, vec)

        ranked: List[Dict[str, Any]]
        if self._cfg.use_semantic_ranking:
            ranked = await self._semantic_rerank(query, fused)
        else:
            for r in fused:
                r["_final_score"] = r.get("_fused_score", 0.0)
            ranked = fused

        return self._clean_sources(ranked[:top_k])

    async def answer(self, query: str) -> Dict[str, Any]:
        return await self._pipeline.get_answer(query)

    async def _semantic_rerank(
        self, query: str, fused_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not fused_results:
            return []
        MAX_RERANK = 50
        top = fused_results[:MAX_RERANK]
        remain = fused_results[MAX_RERANK:]

        id_field = self._cfg.id_field
        ids = [str(r[id_field]) for r in top if id_field in r]
        if not ids:
            # fallback: promote fused scores
            for r in fused_results:
                r["_final_score"] = r.get("_fused_score", 0.0)
            return fused_results

        # Build OR filter with properly escaped single quotes
        # Each ID needs to be in the format: id eq 'value'
        # Single quotes in values must be doubled
        filter_clauses = []
        for id_value in ids:
            # Escape single quotes by doubling them
            escaped_id = id_value.replace("'", "''")
            filter_clauses.append(f"{id_field} eq '{escaped_id}'")

        # Join with ' or ' to create the final filter
        filter_query = " or ".join(filter_clauses)

        results_iter = await self._rerank_client.search(
            search_text=query,
            filter=filter_query,
            query_type=QueryType.SEMANTIC,
            semantic_configuration_name=self._cfg.semantic_configuration_name,
            top=len(ids),
        )

        fused_map = {r[id_field]: r for r in top}
        reranked: List[Dict[str, Any]] = []
        matched_ids = set()

        async for r in results_iter:
            rid = r.get(id_field)
            if rid in fused_map:
                merged = fused_map[rid].copy()
                merged.update(r)
                merged["_final_score"] = merged.get("@search.reranker_score")
                reranked.append(merged)
                matched_ids.add(rid)

        # Preserve any unmatched documents from top 50 with fallback score
        for doc in top:
            doc_id = doc.get(id_field)
            if doc_id not in matched_ids:
                preserved = doc.copy()
                preserved["_final_score"] = preserved.get("_fused_score", 0.0)
                reranked.append(preserved)

        return reranked + remain

    def _clean_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cleaned = []
        for c in chunks:
            d = c.copy()
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
        # NEW: tolerate both async and sync close()
        maybe_close = getattr(self._pipeline, "close", None)
        if callable(maybe_close):
            res = maybe_close()
            if inspect.isawaitable(res):
                await res
        maybe_close2 = getattr(self._rerank_client, "close", None)
        if callable(maybe_close2):
            res2 = maybe_close2()
            if inspect.isawaitable(res2):
                await res2
