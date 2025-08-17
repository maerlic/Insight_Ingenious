"""Synthesizes answers using a Retrieval-Augmented Generation (RAG) model.

This module provides the AnswerGenerator class, which is responsible for the
final step in the RAG pipeline. It takes retrieved document chunks and a user
query, formats them into a structured prompt, and uses an Azure OpenAI
large language model (LLM) to generate a coherent, context-grounded answer.
The primary goal is to produce answers based solely on the provided information,
citing sources appropriately and avoiding hallucination.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from openai import AsyncOpenAI

try:
    from ingenious.services.azure_search.config import SearchConfig
except ImportError:
    from ..config import SearchConfig

logger = logging.getLogger(__name__)

# Default RAG prompt template
DEFAULT_RAG_PROMPT = """
System:
You are an intelligent assistant designed to answer user questions based strictly on the provided context.

Instructions:
1. Analyze the user's question.
2. Review the provided context (Source Chunks).
3. Synthesize a comprehensive answer using only information found in the context.
4. If the context does not contain the answer, state clearly that the information is not available in the provided sources.
5. Do not use any external knowledge or make assumptions beyond the given context.
6. Cite the sources used by referencing the source number (e.g., [Source 1], [Source 2]).

Context (Source Chunks):
{context}
"""


class AnswerGenerator:
    """Generates a final synthesized answer using a RAG approach with Azure OpenAI."""

    _llm_client: AsyncOpenAI

    def __init__(
        self, config: SearchConfig, llm_client: AsyncOpenAI | None = None
    ) -> None:
        """Initialize the AnswerGenerator with configuration and an LLM client.

        This constructor sets up the generator, primarily by establishing a connection
        to the Azure OpenAI service. If a client is not provided, it creates a new one
        based on the given configuration.

        Args:
            config: The search and generation configuration settings.
            llm_client: An optional pre-configured async OpenAI client.
        """
        self._config = config
        self.rag_prompt_template: str = DEFAULT_RAG_PROMPT
        if llm_client is None:
            from ..client_init import make_async_openai_client

            self._llm_client = make_async_openai_client(config)
        else:
            self._llm_client = llm_client

    def _format_context(self, context_chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into a string for the RAG prompt.

        This method compiles a list of document chunks into a single string,
        annotating each with a source number for citation purposes. This formatted
        string serves as the "context" for the language model.

        Args:
            context_chunks: A list of dictionaries, each representing a retrieved chunk.

        Returns:
            A single string containing all context chunks formatted for the prompt.
        """
        context_parts: List[str] = []
        for i, chunk in enumerate(context_chunks):
            content: Any = chunk.get(self._config.content_field, "N/A")
            # Use a simple numbering scheme for citation
            metadata: str = f"[Source {i + 1}]"

            context_parts.append(f"{metadata}\n{content}\n")

        return "\n---\n".join(context_parts)

    async def generate(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Generate an answer to the query based on the provided context chunks.

        This is the main entry point for the generation process. It constructs the full
        RAG prompt by combining the system instructions, the formatted context, and the
        user's query, then calls the Azure OpenAI API to get a synthesized answer.

        Args:
            query: The user's original question.
            context_chunks: The list of relevant document chunks retrieved from search.

        Returns:
            The generated answer string. Returns an error message if generation fails.
        """
        logger.info(f"Generating answer using {len(context_chunks)} context chunks.")

        if not context_chunks:
            return "I could not find any relevant information in the knowledge base to answer your question."

        # Format the context and prepare the prompt
        formatted_context: str = self._format_context(context_chunks)
        system_prompt: str = self.rag_prompt_template.format(context=formatted_context)

        try:
            # Call the Azure OpenAI Chat Completions API
            response = await self._llm_client.chat.completions.create(
                model=self._config.generation_deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Question: {query}"},
                ],
                temperature=0.1,  # Low temperature for factual adherence
                max_tokens=1500,
            )

            message_content: str | None = response.choices[0].message.content
            if message_content is None:
                logger.warning("Received None content from Azure OpenAI response")
                return "The model did not generate a response."

            answer: str = message_content.strip()
            logger.info("Answer generation complete.")
            return answer

        except Exception as e:
            logger.error(f"Error during answer generation with Azure OpenAI: {e}")
            return "An error occurred while generating the answer."

    async def close(self) -> None:
        """Close the underlying LLM client connection.

        This method is intended to be called during application shutdown to gracefully
        release network resources held by the HTTP client.
        """
        await self._llm_client.close()
