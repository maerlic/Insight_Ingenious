# Insight_Ingenious/ingenious/services/azure_search/components/generation.py

import logging
from typing import Any, Dict, List, Optional

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
    """
    Generates the final synthesized answer using a Retrieval-Augmented Generation (RAG) approach
    with Azure OpenAI Service.
    """

    def __init__(self, config: SearchConfig, llm_client: Optional[Any] = None) -> None:
        self._config = config
        self.rag_prompt_template = DEFAULT_RAG_PROMPT
        if llm_client is None:
            from ..client_init import make_async_openai_client

            self._llm_client = make_async_openai_client(config)
        else:
            self._llm_client = llm_client

    def _format_context(self, context_chunks: List[Dict[str, Any]]) -> str:
        """Formats the retrieved chunks into a string suitable for the RAG prompt."""
        context_parts = []
        for i, chunk in enumerate(context_chunks):
            content = chunk.get(self._config.content_field, "N/A")
            # Use a simple numbering scheme for citation
            metadata = f"[Source {i + 1}]"

            context_parts.append(f"{metadata}\n{content}\n")

        return "\n---\n".join(context_parts)

    async def generate(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Generates an answer to the query based on the provided context chunks.
        """
        logger.info(f"Generating answer using {len(context_chunks)} context chunks.")

        if not context_chunks:
            return "I could not find any relevant information in the knowledge base to answer your question."

        # Format the context and prepare the prompt
        formatted_context = self._format_context(context_chunks)
        system_prompt = self.rag_prompt_template.format(context=formatted_context)

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

            message_content = response.choices[0].message.content
            if message_content is None:
                logger.warning("Received None content from Azure OpenAI response")
                return "The model did not generate a response."

            answer = message_content.strip()
            logger.info("Answer generation complete.")
            return answer

        except Exception as e:
            logger.error(f"Error during answer generation with Azure OpenAI: {e}")
            return "An error occurred while generating the answer."

    async def close(self) -> None:
        """Closes the underlying LLM client."""
        await self._llm_client.close()
