# Insight_Ingenious/ingenious/services/azure_search/components/fusion.py

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from ingenious.services.azure_search.config import SearchConfig
except ImportError:
    from ..config import SearchConfig

logger = logging.getLogger(__name__)


class DynamicRankFuser:
    """
    Implements Dynamic Alpha Tuning (DAT) to fuse results from lexical and vector searches.
    DAT uses an LLM to determine the optimal weighting (alpha) based on the specific query
    and the effectiveness of the top results from each retrieval method.
    Alpha (α) represents the weight assigned to Dense (Vector) retrieval. (1-α) is assigned to Sparse (BM25) retrieval.
    """

    def __init__(self, config: SearchConfig, llm_client: Optional[Any] = None):
        self._config = config
        if llm_client is None:
            from ..client_init import make_async_openai_client

            self._llm_client = make_async_openai_client(config)
        else:
            self._llm_client = llm_client

    async def _perform_dat(
        self, query: str, top_lexical: Dict[str, Any], top_vector: Dict[str, Any]
    ) -> float:
        """
        Executes the Dynamic Alpha Tuning (DAT) step using the LLM.
        Returns the calculated alpha (α), the weight for Dense (Vector) retrieval.
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

    def _parse_dat_scores(self, llm_output: str) -> Tuple[int, int]:
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
        """Implements the specific case-aware logic for calculating alpha (α) as defined in the DAT paper (Eq. 6)."""
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

    def _normalize_scores(self, results: List[Dict[str, Any]]) -> None:
        """
        Per-method Min-Max normalization (Section 3).
        Degenerate case (max == min): assign constant 0.5 to all.
        """
        if not results:
            return

        raw_scores: List[float] = []
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
        lexical_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Fuses lexical and vector results using Dynamic Alpha Tuning (DAT).
        R(q, d) = α(q) · S_dense_norm + (1 − α(q)) · S_BM25_norm
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
        if lexical_results and vector_results:
            qkey = (query or "").strip().lower()
            if not hasattr(self, "_alpha_cache"):
                self._alpha_cache: Dict[str, float] = {}
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

        id_field = self._config.id_field
        diag = bool(getattr(self._config, "expose_retrieval_diagnostics", False))

        def _safe_float(x: Any) -> float:
            try:
                return float(x)
            except Exception:
                return 0.0

        # Build normalized lookups
        lex_norm_lookup: Dict[str, float] = {
            doc_id: _safe_float(r.get("_normalized_score"))
            for r in lexical_results
            if (doc_id := r.get(id_field)) is not None
        }
        vec_norm_lookup: Dict[str, float] = {
            doc_id: _safe_float(r.get("_normalized_score"))
            for r in vector_results
            if (doc_id := r.get(id_field)) is not None
        }

        # Raw lookups for diagnostics
        lex_raw_lookup = {
            r.get(id_field): r.get("_retrieval_score")
            for r in lexical_results
            if r.get(id_field)
        }
        vec_raw_lookup = {
            r.get(id_field): r.get("_retrieval_score")
            for r in vector_results
            if r.get(id_field)
        }

        # ─────────────────────────────
        # 3) Convex combination (core DAT)
        # ─────────────────────────────
        fused_results: Dict[str, Dict[str, Any]] = {}

        # Process lexical side first: (1 − α) · S_BM25_norm
        for result in lexical_results:
            doc_id = result.get(id_field)
            if not doc_id:
                continue

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
            doc_id = result.get(id_field)
            if not doc_id:
                continue

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

        def _sort_key(x: Dict[str, Any]) -> Tuple[float, int, float, str]:
            doc_id = x.get(id_field) or ""
            fused = _safe_float(x.get("_fused_score"))
            overlap = 1 if doc_id in overlap_ids else 0
            max_single = max(
                lex_norm_lookup.get(doc_id, 0.0), vec_norm_lookup.get(doc_id, 0.0)
            )
            return (fused, overlap, max_single, str(doc_id))

        sorted_fused = sorted(fused_results.values(), key=_sort_key, reverse=True)

        # Set _final_score only if absent (do not trample later stages)
        for r in sorted_fused:
            if r.get("_final_score") is None:
                r["_final_score"] = r.get("_fused_score", 0.0)

        logger.info("DAT Fusion complete. docs=%d alpha=%.1f", len(sorted_fused), alpha)
        return sorted_fused

    async def close(self) -> None:
        """Closes the underlying LLM client."""
        await self._llm_client.close()
