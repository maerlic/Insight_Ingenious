"""Fuse hybrid search results using an LLM-based dynamic weighting model.

This module provides the DynamicRankFuser, which implements a technique called
Dynamic Alpha Tuning (DAT). The goal is to intelligently combine ranked lists
from parallel lexical (BM25) and vector (dense) search systems. It uses a
Large Language Model (LLM) to assess the relevance of the top results from each
method for a given query, then calculates an optimal fusion weight (alpha) to
produce a single, superior re-ranked list.

The primary entry point is the `DynamicRankFuser.fuse()` method. This component
requires a connection to an LLM service (like Azure OpenAI) to function.

Usage:
    fuser = DynamicRankFuser(config, llm_client)
    fused_results = await fuser.fuse(query, lexical_docs, vector_docs)
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, cast

try:
    from ingenious.services.azure_search.config import SearchConfig
except ImportError:
    from ..config import SearchConfig

if TYPE_CHECKING:
    from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class DynamicRankFuser:
    """
    Implements Dynamic Alpha Tuning (DAT) to fuse results from lexical and vector searches.

    DAT uses an LLM to determine the optimal weighting (alpha) based on the specific query
    and the effectiveness of the top results from each retrieval method.
    Alpha (α) represents the weight assigned to Dense (Vector) retrieval. (1-α) is assigned to Sparse (BM25) retrieval.
    """

    def __init__(
        self, config: SearchConfig, llm_client: AsyncOpenAI | None = None
    ) -> None:
        """Initialize the fuser with configuration and an LLM client.

        This sets up the fuser with the necessary search configuration. If an
        LLM client isn't provided, it will create one on-demand using the
        settings from the config.

        Args:
            config: The search configuration object.
            llm_client: An optional pre-initialized asynchronous OpenAI client.
        """
        self._config = config
        self._llm_client: AsyncOpenAI
        self._alpha_cache: dict[str, float] = {}
        if llm_client is None:
            from ..client_init import make_async_openai_client

            self._llm_client = make_async_openai_client(config)
        else:
            self._llm_client = llm_client

    async def _perform_dat(
        self, query: str, top_lexical: dict[str, Any], top_vector: dict[str, Any]
    ) -> float:
        """
        Execute the Dynamic Alpha Tuning (DAT) step using the LLM.

        This method constructs a prompt with the query and the content of the top-1
        result from both lexical and vector searches. It then sends this prompt to
        the configured LLM to get relevance scores, which are used to calculate the
        fusion weight (alpha).

        Returns:
            The calculated alpha (α), the weight for Dense (Vector) retrieval.
        """
        logger.info("Starting Dynamic Alpha Tuning (DAT) weight calculation...")

        prompt = f"""
Question: {query}

--- Dense Retrieval Top-1 Result ---
{top_vector.get(self._config.content_field, "")[:1500]}

--- BM25 Retrieval Top-1 Result ---
{top_lexical.get(self._config.content_field, "")[:1500]}
"""
        try:
            response = await self._llm_client.chat.completions.create(
                model=self._config.generation_deployment_name,
                messages=[
                    {"role": "system", "content": self._config.dat_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=10,
            )

            llm_output = (response.choices[0].message.content or "").strip()
            score_vector, score_lexical = self._parse_dat_scores(llm_output)
            alpha = self._calculate_alpha(score_vector, score_lexical)
            logger.info(
                f"DAT Scores: Vector(Sv)={score_vector}, Lexical(Sb)={score_lexical}. Calculated Alpha (α)={alpha:.1f}"
            )
            return alpha

        except Exception as e:
            logger.error(
                f"Error during DAT execution (e.g., API error or parsing failure): {e}. Falling back to equal weight (0.5)."
            )
            return 0.5

    def _parse_dat_scores(self, llm_output: str) -> tuple[int, int]:
        """Parse the two relevance scores from the LLM's raw string output.

        This function is designed to robustly extract two integers from the LLM's
        response, which represent the relevance scores (0-5) for the vector and
        lexical results, respectively. It handles malformed or out-of-range outputs.

        Returns:
            A tuple containing the vector score and the lexical score.
        """
        nums = re.findall(r"-?\d+", llm_output or "")
        if len(nums) >= 2:
            try:
                score_v, score_l = int(nums[0]), int(nums[1])
                if 0 <= score_v <= 5 and 0 <= score_l <= 5:
                    return score_v, score_l
                else:
                    logger.warning(
                        f"DAT scores out of range (0-5): '{llm_output}'. Falling back to (0, 0)."
                    )
            except ValueError:
                pass
        logger.warning(
            f"Failed to parse DAT scores from LLM output: '{llm_output}'. Falling back to (0, 0)."
        )
        return 0, 0

    def _calculate_alpha(self, score_vector: int, score_lexical: int) -> float:
        """Calculate alpha (α) from relevance scores.

        This calculation follows the specific case-aware logic defined in the
        DAT paper (Eq. 6), handling edge cases where one or both methods are
        judged to be maximally or minimally relevant.

        Args:
            score_vector: The LLM-assigned relevance score for the vector result (0-5).
            score_lexical: The LLM-assigned relevance score for the lexical result (0-5).

        Returns:
            The final alpha weight, rounded to one decimal place.
        """
        if score_vector == 0 and score_lexical == 0:
            alpha = 0.5
        elif score_vector == 5 and score_lexical != 5:
            alpha = 1.0
        elif score_lexical == 5 and score_vector != 5:
            alpha = 0.0
        else:
            denom = score_vector + score_lexical
            # denom > 0 unless both are zero (handled above)
            alpha = (score_vector / denom) if denom > 0 else 0.5

        return round(alpha, 1)

    def _normalize_scores(self, results: list[dict[str, Any]]) -> None:
        """
        Perform in-place Min-Max normalization on retrieval scores for a set of results.

        This method normalizes the `_retrieval_score` for each document within a
        single result set (e.g., all lexical results) to a scale of [0, 1]. The
        normalized score is stored in the `_normalized_score` field.

        Why: Normalization is required before applying the fusion formula to ensure
        scores from different methods are on a comparable scale.
        """
        if not results:
            return

        raw_scores: list[float] = []
        for r in results:
            s = r.get("_retrieval_score")
            try:
                raw_scores.append(float(s) if s is not None else 0.0)
            except (ValueError, TypeError):
                raw_scores.append(0.0)

        if not raw_scores:
            return

        min_score = min(raw_scores)
        max_score = max(raw_scores)

        if max_score == min_score:
            # Spec-compliant degenerate handling: constant mid-scale value
            for r in results:
                r["_normalized_score"] = 0.5
            return

        span = max_score - min_score
        for i, r in enumerate(results):
            v = (raw_scores[i] - min_score) / span
            # Clamp to [0,1] for safety
            v = 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)
            r["_normalized_score"] = v

    async def fuse(
        self,
        query: str,
        lexical_results: list[dict[str, Any]],
        vector_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Fuse lexical and vector results using Dynamic Alpha Tuning (DAT).

        This is the main orchestration method. It calculates the dynamic alpha,
        normalizes scores for each result set, then computes a final fused score
        for each unique document based on the formula:
        `R(q, d) = α(q) · S_dense_norm + (1 − α(q)) · S_BM25_norm`

        Args:
            query: The user's search query.
            lexical_results: A list of documents from the lexical (BM25) search.
            vector_results: A list of documents from the vector search.

        Returns:
            A single list of documents, sorted by the new fused score.
        """
        # ─────────────────────────────
        # 0) Fast exits
        # ─────────────────────────────
        if not lexical_results and not vector_results:
            return []

        # ─────────────────────────────
        # 1) Compute α
        #    - Use DAT only if both sides have a Top-1
        #    - Otherwise use consistent defaults:
        #        * only vector → α = 1.0
        #        * only lexical → α = 0.0
        # ─────────────────────────────
        alpha: float
        if lexical_results and vector_results:
            qkey = (query or "").strip().lower()
            if qkey in self._alpha_cache:
                alpha = self._alpha_cache[qkey]
            else:
                alpha = await self._perform_dat(
                    query, lexical_results[0], vector_results[0]
                )
                self._alpha_cache[qkey] = alpha
        elif vector_results and not lexical_results:
            alpha = 1.0
        else:  # lexical_results and not vector_results
            alpha = 0.0

        one_minus_alpha = round(1.0 - alpha, 1)

        # ─────────────────────────────
        # 2) Per-method Min-Max normalization (no rank fallback)
        # ─────────────────────────────
        self._normalize_scores(lexical_results)
        self._normalize_scores(vector_results)

        id_field: str = self._config.id_field
        diag: bool = bool(getattr(self._config, "expose_retrieval_diagnostics", False))

        def _safe_float(x: Any) -> float:
            """Safely convert a value to a float, returning 0.0 on failure."""
            try:
                # The float() constructor can raise TypeError or ValueError
                return float(x)
            except (ValueError, TypeError):
                return 0.0

        # Build normalized lookups
        lex_norm_lookup: dict[str, float] = {
            cast(str, doc_id): _safe_float(r.get("_normalized_score"))
            for r in lexical_results
            if (doc_id := r.get(id_field)) is not None
        }
        vec_norm_lookup: dict[str, float] = {
            cast(str, doc_id): _safe_float(r.get("_normalized_score"))
            for r in vector_results
            if (doc_id := r.get(id_field)) is not None
        }

        # Raw lookups for diagnostics
        lex_raw_lookup: dict[str, Any | None] = {
            cast(str, doc_id): r.get("_retrieval_score")
            for r in lexical_results
            if (doc_id := r.get(id_field))
        }
        vec_raw_lookup: dict[str, Any | None] = {
            cast(str, doc_id): r.get("_retrieval_score")
            for r in vector_results
            if (doc_id := r.get(id_field))
        }

        # ─────────────────────────────
        # 3) Convex combination (core DAT)
        # ─────────────────────────────
        fused_results: dict[str, dict[str, Any]] = {}

        # Process lexical side first: (1 − α) · S_BM25_norm
        for result in lexical_results:
            doc_id_any = result.get(id_field)
            if not doc_id_any:
                continue
            doc_id = cast(str, doc_id_any)

            bm25_norm = lex_norm_lookup.get(doc_id, 0.0)
            vec_norm = vec_norm_lookup.get(doc_id, 0.0)

            bm25_component = one_minus_alpha * bm25_norm
            vector_component = alpha * vec_norm

            fused = bm25_component + vector_component
            result["_fused_score"] = fused

            # Preserve raw scores for display
            result["_bm25_score_raw"] = lex_raw_lookup.get(doc_id)
            result["_vector_score_raw"] = vec_raw_lookup.get(doc_id)  # may be None

            if diag:
                result["_dat_alpha"] = alpha
                result["_dat_weight_vector"] = alpha
                result["_dat_weight_bm25"] = one_minus_alpha
                result["_bm25_norm"] = bm25_norm
                result["_vector_norm"] = vec_norm
                result["_bm25_component"] = bm25_component
                result["_vector_component"] = vector_component

            fused_results[doc_id] = result

        # Process vector side: α · S_dense_norm (and add if missing or merge if overlap)
        for result in vector_results:
            doc_id_any = result.get(id_field)
            if not doc_id_any:
                continue
            doc_id = cast(str, doc_id_any)

            vec_norm = vec_norm_lookup.get(doc_id, 0.0)
            bm25_norm = lex_norm_lookup.get(doc_id, 0.0)

            bm25_component = one_minus_alpha * bm25_norm
            vector_component = alpha * vec_norm
            fused = bm25_component + vector_component

            if doc_id in fused_results:
                existing = fused_results[doc_id]
                existing["_fused_score"] = fused  # recompute with both components

                # update raw scores for overlap docs
                existing["_vector_score_raw"] = vec_raw_lookup.get(doc_id)

                if diag:
                    existing["_bm25_norm"] = bm25_norm
                    existing["_vector_norm"] = vec_norm
                    existing["_bm25_component"] = bm25_component
                    existing["_vector_component"] = vector_component

                existing["_retrieval_type"] = f"hybrid_dat_alpha_{alpha:.1f}"
            else:
                result["_fused_score"] = fused

                # Preserve raw scores for display
                result["_bm25_score_raw"] = lex_raw_lookup.get(doc_id)  # may be None
                result["_vector_score_raw"] = vec_raw_lookup.get(doc_id)

                if diag:
                    result["_dat_alpha"] = alpha
                    result["_dat_weight_vector"] = alpha
                    result["_dat_weight_bm25"] = one_minus_alpha
                    result["_bm25_norm"] = bm25_norm
                    result["_vector_norm"] = vec_norm
                    result["_bm25_component"] = bm25_component
                    result["_vector_component"] = vector_component

                fused_results[doc_id] = result

        # ─────────────────────────────
        # 4) Stable sort + tiebreakers
        # ─────────────────────────────
        overlap_ids = set(lex_norm_lookup.keys()) & set(vec_norm_lookup.keys())

        def _sort_key(x: dict[str, Any]) -> tuple[float, int, float, str]:
            """Define the sorting logic for the final ranked list."""
            doc_id: str = str(x.get(id_field) or "")
            fused = _safe_float(x.get("_fused_score"))
            overlap = 1 if doc_id in overlap_ids else 0
            max_single = max(
                lex_norm_lookup.get(doc_id, 0.0), vec_norm_lookup.get(doc_id, 0.0)
            )
            return (fused, overlap, max_single, doc_id)

        sorted_fused = sorted(fused_results.values(), key=_sort_key, reverse=True)

        # Set _final_score only if absent (do not trample later stages)
        for r in sorted_fused:
            if r.get("_final_score") is None:
                r["_final_score"] = r.get("_fused_score", 0.0)

        logger.info("DAT Fusion complete. docs=%d alpha=%.1f", len(sorted_fused), alpha)
        return sorted_fused

    async def close(self) -> None:
        """Close the underlying asynchronous LLM client.

        Why: This is important for graceful shutdown, ensuring that network
        connections are properly terminated.
        """
        await self._llm_client.close()
