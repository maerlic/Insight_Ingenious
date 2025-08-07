# Insight_Ingenious/ingenious/services/azure_search/components/fusion.py

import logging
import re
from typing import List, Dict, Any, Tuple
from openai import AsyncAzureOpenAI

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

    def __init__(self, config: SearchConfig):
        self._config = config
        self._llm_client = self._initialize_llm_client()

    def _initialize_llm_client(self) -> AsyncAzureOpenAI:
        """Initializes the asynchronous Azure OpenAI client for generation (DAT)."""
        return AsyncAzureOpenAI(
            azure_endpoint=self._config.openai_endpoint,
            api_key=self._config.openai_key.get_secret_value(),
            api_version=self._config.openai_version,
        )

    async def _perform_dat(self, query: str, top_lexical: Dict[str, Any], top_vector: Dict[str, Any]) -> float:
        """
        Executes the Dynamic Alpha Tuning (DAT) step using the LLM.
        Returns the calculated alpha (α), the weight for Dense (Vector) retrieval.
        """
        logger.info("Starting Dynamic Alpha Tuning (DAT) weight calculation...")

        # Prepare the user prompt with the specific documents (truncated for efficiency)
        # Aligning with the input format specified in the DAT prompt
        prompt = f"""
Question: {query}

--- Dense Retrieval Top-1 Result ---
{top_vector.get(self._config.content_field, '')[:1500]}

--- BM25 Retrieval Top-1 Result ---
{top_lexical.get(self._config.content_field, '')[:1500]}
"""
        try:
            # Execute the LLM call, expecting plain text output (two space-separated integers)
            response = await self._llm_client.chat.completions.create(
                model=self._config.generation_deployment_name,
                messages=[
                    {"role": "system", "content": self._config.dat_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0, # Deterministic scoring
                max_tokens=10, # Only need 2 digits and a space
            )

            llm_output = response.choices[0].message.content.strip()
            
            # Parse the scores robustly
            score_vector, score_lexical = self._parse_dat_scores(llm_output)

            # Calculate Alpha (α) using the case-aware formulation (Hsu & Tzeng, 2025, Section 4.2)
            alpha = self._calculate_alpha(score_vector, score_lexical)
            
            logger.info(f"DAT Scores: Vector(Sv)={score_vector}, Lexical(Sb)={score_lexical}. Calculated Alpha (α)={alpha:.1f}")
            return alpha

        except Exception as e:
            logger.error(f"Error during DAT execution (e.g., API error or parsing failure): {e}. Falling back to equal weight (0.5).")
            return 0.5

    def _parse_dat_scores(self, llm_output: str) -> Tuple[int, int]:
        """Parses the LLM output string "ScoreV ScoreL" into integers robustly."""
        # Use regex to find the first two numbers in the output
        match = re.search(r'(\d+)\s+(\d+)', llm_output)
        if match:
            try:
                score_v = int(match.group(1))
                score_l = int(match.group(2))
                # Ensure scores are within the expected 0-5 range
                if 0 <= score_v <= 5 and 0 <= score_l <= 5:
                    return score_v, score_l
                else:
                    logger.warning(f"DAT scores out of range (0-5): '{llm_output}'. Falling back to (0, 0).")
            except ValueError:
                # Handle cases where match groups might not be valid integers (though regex \d+ should prevent this)
                pass
        
        logger.warning(f"Failed to parse DAT scores from LLM output: '{llm_output}'. Falling back to (0, 0).")
        return 0, 0

    def _calculate_alpha(self, score_vector: int, score_lexical: int) -> float:
        """Implements the specific case-aware logic for calculating alpha (α) as defined in the DAT paper (Eq. 6)."""
        
        # Case 1: Both methods fail (0, 0)
        if score_vector == 0 and score_lexical == 0:
            alpha = 0.5
        # Case 2: Vector is a direct hit (5) and Lexical is not
        elif score_vector == 5 and score_lexical != 5:
            alpha = 1.0
        # Case 3: Lexical is a direct hit (5) and Vector is not
        elif score_lexical == 5 and score_vector != 5:
            alpha = 0.0
        # Case 4: Proportional weighting
        else:
            # Since we handled the 0/0 case, the denominator here is guaranteed to be > 0
            alpha = score_vector / (score_vector + score_lexical)

        # The paper specifies rounding the final alpha to one decimal place for stability (Section 4.2).
        return round(alpha, 1)

    def _normalize_scores(self, results: List[Dict[str, Any]]) -> None:
        """
        Performs Min-Max normalization on the retrieval scores in place (Section 3).
        Normalization is essential because BM25 and Vector scores are on different scales.
        """
        if not results:
            return

        # Ensure scores are treated as floats and handle potential None values
        scores = []
        for r in results:
            score = r.get("_retrieval_score")
            if score is not None:
                try:
                    scores.append(float(score))
                except (ValueError, TypeError):
                    scores.append(0.0)
            else:
                scores.append(0.0)

        if not scores:
            return

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            # Handle division by zero if all scores are the same
            normalized_value = 1.0 if max_score > 0 else 0.0
            for result in results:
                result["_normalized_score"] = normalized_value
        else:
            for i, result in enumerate(results):
                normalized_score = (scores[i] - min_score) / (max_score - min_score)
                result["_normalized_score"] = normalized_score

    async def fuse(self, query: str, lexical_results: List[Dict[str, Any]], vector_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fuses the lexical and vector results using Dynamic Alpha Tuning.
        R(q, d) = α(q) · S_dense_norm + (1 - α(q)) · S_BM25_norm
        """
        
        # Handle edge cases where one or both lists might be empty
        if not lexical_results and not vector_results:
            return []
        # If one list is empty, DAT is unnecessary; return the non-empty list.
        # Note: We don't normalize here as it's unnecessary for ranking if only one list exists.
        if not lexical_results:
            logger.info("Lexical results empty. Proceeding with vector results only (α=1.0).")
            for r in vector_results: r["_fused_score"] = r.get("_retrieval_score", 0.0)
            return vector_results
        if not vector_results:
            logger.info("Vector results empty. Proceeding with lexical results only (α=0.0).")
            for r in lexical_results: r["_fused_score"] = r.get("_retrieval_score", 0.0)
            return lexical_results

        # 1. Determine weights using DAT on top results
        top_lexical = lexical_results[0]
        top_vector = vector_results[0]
        
        # Get the dynamic alpha (weight for vector/dense)
        alpha = await self._perform_dat(query, top_lexical, top_vector)
        one_minus_alpha = round(1.0 - alpha, 1)

        # 2. Normalize scores within each list
        self._normalize_scores(lexical_results)
        self._normalize_scores(vector_results)

        # 3. Compute combined score (Convex Combination) and merge results
        fused_results: Dict[str, Dict[str, Any]] = {}
        id_field = self._config.id_field

        # Process lexical results (Weight = 1 - α)
        for result in lexical_results:
            doc_id = result.get(id_field)
            if doc_id:
                # Fused Score component = (1 - α) * normalized_lexical_score
                fused_score_component = one_minus_alpha * result.get("_normalized_score", 0.0)
                result["_fused_score"] = fused_score_component
                fused_results[doc_id] = result

        # Process vector results (Weight = α)
        for result in vector_results:
            doc_id = result.get(id_field)
            if doc_id:
                # Fused Score component = α * normalized_vector_score
                vector_score_component = alpha * result.get("_normalized_score", 0.0)
                
                if doc_id in fused_results:
                    # If the document exists in both lists, combine the components
                    # Total Fused Score = (α * norm_vec) + ((1-α) * norm_lex)
                    existing_result = fused_results[doc_id]
                    existing_result["_fused_score"] += vector_score_component
                    # Update retrieval type metadata to reflect the dynamic alpha used
                    existing_result["_retrieval_type"] = f"hybrid_dat_alpha_{alpha:.1f}"
                else:
                    result["_fused_score"] = vector_score_component
                    fused_results[doc_id] = result

        # 4. Sort the final list based on the fused score
        sorted_fused_results = sorted(
            fused_results.values(),
            key=lambda x: x.get("_fused_score", 0.0),
            reverse=True
        )

        logger.info(f"DAT Fusion complete. Total unique documents: {len(sorted_fused_results)}")
        return sorted_fused_results

    async def close(self):
        """Closes the underlying LLM client."""
        await self._llm_client.close()