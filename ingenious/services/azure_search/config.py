"""Defines the configuration model for an advanced search pipeline.

This module provides a Pydantic-based configuration class, `SearchConfig`,
to centralize all settings for a Retrieval-Augmented Generation (RAG) system
that uses Azure AI Search and Azure OpenAI. It ensures that all necessary
parameters—such as API endpoints, keys, model names, and pipeline behavior
(e.g., Dynamic Alpha Tuning)—are provided and validated at startup.

The primary entry point is the `SearchConfig` class, which should be
instantiated to configure the search service.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, SecretStr

# Default prompt for Dynamic Alpha Tuning (DAT)
# Adapted from the methodology described in "DAT: Dynamic Alpha Tuning for Hybrid Retrieval in RAG" (Hsu & Tzeng, 2025), specifically Appendix A.
# This prompt asks the LLM to evaluate the retrieval effectiveness (likelihood of finding the answer nearby) of the top-1 results.
DEFAULT_DAT_PROMPT: str = """
System:
You are an evaluator assessing the retrieval effectiveness of Dense Retrieval (Vector/Semantic) and Sparse Retrieval (BM25/Keyword).

## Task:
Given a user question and the top-1 search result from each method, score each retrieval method from 0 to 5. This score reflects how likely the correct answer is to be found near this top result (e.g., in the top-2 or top-3 results of that method).

### Scoring Criteria:
1. **Direct Hit -> 5 points**
   - The retrieved document directly and completely answers the question.
2. **Good Wrong Result (High likelihood correct answer is nearby) -> 3-4 points**
   - The result is conceptually very close to the correct answer (mentions relevant entities, related events, or a partial answer).
   - The search method is clearly heading in the right direction.
   - Give 4 if very close, 3 if somewhat close.
3. **Bad Wrong Result (Low likelihood correct answer is nearby) -> 1-2 points**
   - The result is loosely related but misleading (e.g., shares keywords but the context is wrong).
   - Correct answers are unlikely to be in the immediate vicinity (top-2, top-3).
   - Give 2 if there's a small chance, 1 if very unlikely.
4. **Completely Off-Track -> 0 points**
   - The result is totally unrelated. The retrieval method is failing for this query.

### Input Format:
You will receive the Question, the Dense Retrieval Top-1 Result, and the BM25 Retrieval Top-1 Result.

### Output Format:
Respond ONLY with two integers separated by a single space. Do not include any other text or explanation.
- First number: Dense Retrieval score.
- Second number: BM25 Retrieval score.
- Example output: 3 4
"""


class SearchConfig(BaseModel):
    """Configuration model for the Advanced Azure AI Search service.

    This class uses Pydantic to define and validate all connection settings
    for Azure services and behavior parameters for the search pipeline. It
    serves as a single source of truth to ensure the system is correctly
    configured before use.
    """

    # Azure AI Search Configuration
    search_endpoint: str = Field(
        ..., description="The endpoint URL for the Azure AI Search service."
    )
    search_key: SecretStr = Field(
        ..., description="The API key for the Azure AI Search service."
    )
    search_index_name: str = Field(
        ..., description="The name of the target index in Azure AI Search."
    )
    semantic_configuration_name: Optional[str] = Field(
        None,
        description="The name of the semantic configuration required if use_semantic_ranking is True.",
    )

    # Azure OpenAI Configuration
    openai_endpoint: str = Field(
        ..., description="The endpoint URL for the Azure OpenAI service."
    )
    openai_key: SecretStr = Field(
        ..., description="The API key for the Azure OpenAI service."
    )
    openai_version: str = Field(
        "2024-02-01", description="The API version for the Azure OpenAI service."
    )
    embedding_deployment_name: str = Field(
        ...,
        description="The deployment name for the embeddings model (e.g., text-embedding-3-small).",
    )
    generation_deployment_name: str = Field(
        ...,
        description="The deployment name for the generation model (e.g., gpt-4o). Used for DAT and final answer.",
    )

    # Pipeline Behavior Configuration
    top_k_retrieval: int = Field(
        20,
        description="The number of initial results to fetch from both lexical and vector searches.",
    )
    use_semantic_ranking: bool = Field(
        True,
        description="Flag to enable or disable Azure's Semantic Ranking feature after DAT fusion.",
    )
    top_n_final: int = Field(
        5,
        description="The number of re-ranked chunks to feed to the final answer generation model.",
    )
    dat_prompt: str = Field(
        DEFAULT_DAT_PROMPT,
        description="The prompt template to use for the Dynamic Alpha Tuning (DAT) scoring step.",
    )

    # Index Schema Field Mappings (Defaults provided, adjust if necessary)
    id_field: str = Field(
        "id", description="The unique identifier field in the search index."
    )
    content_field: str = Field(
        "content", description="The primary text content field in the search index."
    )
    vector_field: str = Field(
        "vector", description="The vector embedding field in the search index."
    )
    # ── Generation toggle ─────────────────────────────────────────────────────
    enable_answer_generation: bool = Field(
        False,
        description="If True, the pipeline will call the LLM to synthesize a final answer.",
    )

    class Config:
        """Pydantic model configuration.

        This inner class defines metadata for the parent `SearchConfig` model.
        Its purpose is to enforce immutability (`frozen = True`) and to allow
        specialized types like `SecretStr` (`arbitrary_types_allowed = True`).
        """

        # Enforce immutability for configuration objects
        frozen: bool = True
        # Allow SecretStr type
        arbitrary_types_allowed: bool = True
