# Insight_Ingenious/ingenious/services/chat_services/multi_agent/conversation_flows/knowledge_base_agent/knowledge_base_agent.py
import asyncio
import logging
import os
import time
import uuid
from typing import AsyncIterator, Dict, List, Optional, Tuple

# Re-exported for test monkey-patching compatibility
from autogen_agentchat.agents import AssistantAgent as _AssistantAgent
from autogen_core import (  # noqa: F401 (CancellationToken kept for API parity)
    EVENT_LOGGER_NAME,
    CancellationToken,
)
from autogen_core.tools import FunctionTool as _FunctionTool

from ingenious.common.utils import create_aoai_chat_completion_client_from_config
from ingenious.models.agent import (
    LLMUsageTracker as LLMUsageTracker,  # type: ignore  # harmonize name if needed
)
from ingenious.models.chat import ChatRequest, ChatResponse, ChatResponseChunk
from ingenious.services.chat_services.multi_agent.service import IConversationFlow

# Back-compat names so tests can patch: knowledge_base_agent.FunctionTool / AssistantAgent
FunctionTool = _FunctionTool
AssistantAgent = _AssistantAgent

__all__ = ["ConversationFlow", "FunctionTool", "AssistantAgent"]


class ConversationFlow(IConversationFlow):
    """
    Knowledge base conversation flow.

    - Non-streaming: direct KB search by default (deterministic "direct" mode).
      Optional "assist" mode uses AssistantAgent.on_messages for LLM summarization.
    - Streaming: uses AssistantAgent.run_stream and forwards content; robust error chunking,
      final flush to surface terminal results, and safe token-count fallback.
    - Careful resource/handler lifecycle and resilient Azure->Chroma fallback.
    """

    # -----------------------------
    # Public API
    # -----------------------------
    async def get_conversation_response(
        self, chat_request: ChatRequest
    ) -> ChatResponse:
        model_config = self._config.models[0]

        # Dedicated logger; don't clobber global handlers
        base_logger = logging.getLogger(f"{EVENT_LOGGER_NAME}.kb")
        base_logger.setLevel(logging.INFO)

        llm_logger = LLMUsageTracker(
            agents=[],
            config=self._config,
            chat_history_repository=self._chat_service.chat_history_repository
            if self._chat_service
            else None,
            revision_id=str(uuid.uuid4()),
            identifier=str(uuid.uuid4()),
            event_type="knowledge_base",
        )
        base_logger.addHandler(llm_logger)

        memory_context = await self._build_memory_context(chat_request)

        # Configurable mode: "direct" (deterministic) | "assist" (LLM-composed)
        mode = (
            (
                getattr(self._config, "knowledge_base_mode", None)
                or os.getenv("KB_MODE", "direct")
            )
            .strip()
            .lower()
        )
        if mode not in ("direct", "assist"):
            mode = "direct"

        # Create model client early (shared across modes)
        model_client = create_aoai_chat_completion_client_from_config(model_config)

        try:
            use_azure_search = self._should_use_azure_search()
            # --------- DIRECT MODE (default) ---------
            if mode == "direct":
                # Perform KB search directly (no AssistantAgent.on_messages)
                search_text = await self._search_knowledge_base(
                    search_query=chat_request.user_prompt,
                    use_azure_search=use_azure_search,
                    top_k=3,
                    logger=base_logger,
                )

                # Align header backend label to actual result (handles Azure->Chroma fallback)
                backend_from_result = (
                    "Azure AI Search"
                    if isinstance(search_text, str)
                    and search_text.startswith(
                        "Found relevant information from Azure AI Search"
                    )
                    else "local ChromaDB"
                    if isinstance(search_text, str)
                    and search_text.startswith(
                        "Found relevant information from ChromaDB"
                    )
                    else ("Azure AI Search" if use_azure_search else "local ChromaDB")
                )
                context = f"Knowledge base search assistant using {backend_from_result} for finding information."

                # Deterministic final message, with explicit "User question:" line
                header = f"Context: {context}\n\n"
                if memory_context:
                    header += memory_context
                header += f"User question: {chat_request.user_prompt}\n\n"
                final_message = header + (search_text or "No response generated")

                # Token accounting (non-fatal)
                total_tokens, completion_tokens = await self._safe_count_tokens(
                    system_message=self._static_system_message(memory_context),
                    user_message=chat_request.user_prompt,
                    assistant_message=final_message,
                    model=model_config.model,
                    logger=base_logger,
                )

                return ChatResponse(
                    thread_id=chat_request.thread_id or "",
                    message_id=str(uuid.uuid4()),
                    agent_response=final_message,
                    token_count=total_tokens,
                    max_token_count=completion_tokens,
                    memory_summary=final_message,
                )

            # --------- ASSIST MODE (optional) ---------
            # Use an agent to summarize/format based on tool results
            else:
                use_azure_search = self._should_use_azure_search()
                search_backend = (
                    "Azure AI Search" if use_azure_search else "local ChromaDB"
                )
                context = f"Knowledge base search assistant using {search_backend} for finding information."

                async def search_tool(search_query: str, topic: str = "general") -> str:
                    """Search for information using Azure AI Search or local ChromaDB."""
                    return await self._search_knowledge_base(
                        search_query=search_query,
                        use_azure_search=use_azure_search,
                        top_k=5,
                        logger=base_logger,
                    )

                search_function_tool = FunctionTool(
                    search_tool,
                    description=f"Search for information using {search_backend}. Use relevant keywords to find relevant information.",
                )

                system_message = self._assist_system_message(memory_context)
                search_assistant = AssistantAgent(
                    name="search_assistant",
                    system_message=system_message,
                    model_client=model_client,
                    tools=[search_function_tool],
                    reflect_on_tool_use=True,
                )

                from autogen_agentchat.messages import TextMessage

                user_msg = (
                    f"Context: {context}\n\nUser question: {chat_request.user_prompt}"
                    if context
                    else chat_request.user_prompt
                )

                cancellation_token = CancellationToken()
                response = await search_assistant.on_messages(
                    messages=[TextMessage(content=user_msg, source="user")],
                    cancellation_token=cancellation_token,
                )

                assistant_text = (
                    response.chat_message.content
                    if getattr(response, "chat_message", None)
                    else "No response generated"
                )

                # Optionally prepend header for parity
                final_message = assistant_text

                total_tokens, completion_tokens = await self._safe_count_tokens(
                    system_message=system_message,
                    user_message=user_msg,
                    assistant_message=final_message,
                    model=model_config.model,
                    logger=base_logger,
                )

                return ChatResponse(
                    thread_id=chat_request.thread_id or "",
                    message_id=str(uuid.uuid4()),
                    agent_response=final_message,
                    token_count=total_tokens,
                    max_token_count=completion_tokens,
                    memory_summary=final_message,
                )

        finally:
            try:
                await model_client.close()
            except Exception:
                pass
            try:
                base_logger.removeHandler(llm_logger)
            except Exception:
                pass

    async def get_streaming_conversation_response(
        self, chat_request: ChatRequest
    ) -> AsyncIterator[ChatResponseChunk]:
        """Streaming version of knowledge base agent conversation."""
        message_id = str(uuid.uuid4())
        thread_id = chat_request.thread_id or ""

        model_config = self._config.models[0]
        base_logger = logging.getLogger(f"{EVENT_LOGGER_NAME}.kb")
        base_logger.setLevel(logging.INFO)

        llm_logger = LLMUsageTracker(
            agents=[],
            config=self._config,
            chat_history_repository=self._chat_service.chat_history_repository
            if self._chat_service
            else None,
            revision_id=str(uuid.uuid4()),
            identifier=str(uuid.uuid4()),
            event_type="knowledge_base_streaming",
        )
        base_logger.addHandler(llm_logger)

        model_client = create_aoai_chat_completion_client_from_config(model_config)

        try:
            # Initial status
            yield ChatResponseChunk(
                thread_id=thread_id,
                message_id=message_id,
                chunk_type="status",
                content="Searching knowledge base...",
                is_final=False,
            )

            memory_context = await self._build_memory_context(chat_request)
            use_azure_search = self._should_use_azure_search()
            search_backend = "Azure AI Search" if use_azure_search else "local ChromaDB"

            # Define a tool that uses our unified KB search helper (agent may call this)
            async def search_tool(search_query: str, topic: str = "general") -> str:
                """Search for information using Azure AI Search or local ChromaDB."""
                return await self._search_knowledge_base(
                    search_query=search_query,
                    use_azure_search=use_azure_search,
                    top_k=5,
                    logger=base_logger,
                )

            search_function_tool = FunctionTool(
                search_tool,
                description=f"Search for information using {search_backend}. Use relevant keywords to find relevant information.",
            )

            system_message = self._streaming_system_message(memory_context)
            search_assistant = AssistantAgent(
                name="search_assistant",
                system_message=system_message,
                model_client=model_client,
                tools=[search_function_tool],
                reflect_on_tool_use=True,
            )

            user_msg = f"User query: {chat_request.user_prompt}"

            # Next status
            yield ChatResponseChunk(
                thread_id=thread_id,
                message_id=message_id,
                chunk_type="status",
                content="Generating response...",
                is_final=False,
            )

            accumulated_content = ""
            total_tokens = 0
            completion_tokens = 0

            cancellation_token = CancellationToken()

            try:
                # Let tests monkey-patch run_stream; we forward its messages.
                stream = search_assistant.run_stream(
                    task=user_msg, cancellation_token=cancellation_token
                )

                async for message in stream:
                    if hasattr(message, "content") and message.content:
                        accumulated_content += message.content
                        yield ChatResponseChunk(
                            thread_id=thread_id,
                            message_id=message_id,
                            chunk_type="content",
                            content=message.content,
                            is_final=False,
                        )
                    if hasattr(message, "usage"):
                        usage = message.usage
                        if hasattr(usage, "total_tokens"):
                            total_tokens = usage.total_tokens
                        if hasattr(usage, "completion_tokens"):
                            completion_tokens = usage.completion_tokens
                        yield ChatResponseChunk(
                            thread_id=thread_id,
                            message_id=message_id,
                            chunk_type="token_count",
                            token_count=total_tokens,
                            is_final=False,
                        )

                    # ---- Final flush restoration: surface terminal TaskResult content ----
                    if hasattr(message, "__class__") and "TaskResult" in str(
                        message.__class__
                    ):
                        try:
                            final_msgs = getattr(message, "messages", None)
                            if final_msgs:
                                final_msg = final_msgs[-1]
                                final_text = getattr(final_msg, "content", None)
                                if final_text and final_text not in accumulated_content:
                                    accumulated_content += final_text
                                    yield ChatResponseChunk(
                                        thread_id=thread_id,
                                        message_id=message_id,
                                        chunk_type="content",
                                        content=final_text,
                                        is_final=False,
                                    )
                        except Exception:
                            # Avoid breaking stream on flush issues
                            pass

            except Exception as e:
                # Surface an explicit content chunk with the error
                base_logger.error(f"Streaming error: {e}")
                error_text = f"[Error during streaming: {str(e)}]"
                accumulated_content += error_text
                yield ChatResponseChunk(
                    thread_id=thread_id,
                    message_id=message_id,
                    chunk_type="content",
                    content=error_text,
                    is_final=False,
                )

            # ---- Safe token-count fallback if usage not reported ----
            if total_tokens == 0:
                try:
                    total_tokens, completion_tokens = await self._safe_count_tokens(
                        system_message=system_message,
                        user_message=user_msg,
                        assistant_message=accumulated_content,
                        model=model_config.model,
                        logger=base_logger,
                    )
                except Exception:
                    total_tokens, completion_tokens = 0, 0

                # Rough estimate if still zero
                if total_tokens == 0:
                    total_tokens = (
                        len(system_message) + len(user_msg) + len(accumulated_content)
                    ) // 4
                    completion_tokens = len(accumulated_content) // 4

                # Emit a final token_count update before the final chunk
                yield ChatResponseChunk(
                    thread_id=thread_id,
                    message_id=message_id,
                    chunk_type="token_count",
                    token_count=total_tokens,
                    is_final=False,
                )

            # Finalize stream with deterministic memory summary
            yield ChatResponseChunk(
                thread_id=thread_id,
                message_id=message_id,
                chunk_type="final",
                token_count=total_tokens,
                max_token_count=completion_tokens,
                memory_summary=(accumulated_content[:200] + "...")
                if len(accumulated_content) > 200
                else accumulated_content,
                event_type="knowledge_base_streaming",
                is_final=True,
            )

        except Exception as outer:
            base_logger.error(f"Error in streaming knowledge base response: {outer}")
            yield ChatResponseChunk(
                thread_id=thread_id,
                message_id=message_id,
                chunk_type="error",
                content=f"An error occurred: {str(outer)}",
                is_final=True,
            )
        finally:
            try:
                await model_client.close()
            except Exception:
                pass
            try:
                base_logger.removeHandler(llm_logger)
            except Exception:
                pass

    # -----------------------------
    # Internal helpers
    # -----------------------------
    async def _build_memory_context(self, chat_request: ChatRequest) -> str:
        """Build a compact memory context from the last 10 thread messages (non-fatal)."""
        memory_context = ""
        if chat_request.thread_id and self._chat_service:
            try:
                thread_messages = await self._chat_service.chat_history_repository.get_thread_messages(
                    chat_request.thread_id
                )
                if thread_messages:
                    recent = (
                        thread_messages[-10:]
                        if len(thread_messages) > 10
                        else thread_messages
                    )
                    preview = [f"{m.role}: {m.content[:100]}..." for m in recent]
                    memory_context = (
                        "Previous conversation:\n" + "\n".join(preview) + "\n\n"
                    )
            except Exception as e:
                # ⚠️7: Throttled warn + debug to maintain observability without noise
                logger = logging.getLogger(f"{EVENT_LOGGER_NAME}.kb")
                now = time.monotonic()
                last = getattr(self, "_last_mem_warn_ts", 0.0)
                if (now - last) > 60.0:
                    logger.warning(f"Failed to retrieve thread memory: {e}")
                    self._last_mem_warn_ts = now
                else:
                    logger.debug(f"Failed to retrieve thread memory (suppressed): {e}")
        return memory_context

    def _is_azure_search_available(self) -> bool:
        """
        Best-effort check that the Azure Search provider/SDK is importable.
        Does not validate network/keys; runtime failures still fall back.
        """
        try:
            from ingenious.services.azure_search.provider import (
                AzureSearchProvider,  # type: ignore
            )

            _ = AzureSearchProvider  # silence linter
            return True
        except Exception:
            return False

    def _azure_service(self):
        """Return first azure_search_services entry or None."""
        cfg = getattr(self._config, "azure_search_services", None)
        if not cfg or len(cfg) == 0:
            return None
        return cfg[0]

    def _ensure_default_azure_index(
        self, logger: Optional[logging.Logger] = None
    ) -> None:
        """
        If service.index_name is missing, set from env AZURE_SEARCH_DEFAULT_INDEX or 'test-index',
        and emit a clear setup hint.
        """
        service = self._azure_service()
        if not service:
            return
        idx = getattr(service, "index_name", "")
        if not idx:
            default_idx = os.getenv("AZURE_SEARCH_DEFAULT_INDEX", "test-index")
            try:
                setattr(service, "index_name", default_idx)
            finally:
                if logger:
                    logger.warning(
                        f"Azure Search 'index_name' not configured; using default '{default_idx}'. "
                        "Set azure_search_services[0].index_name or AZURE_SEARCH_DEFAULT_INDEX to override."
                    )

    def _should_use_azure_search(self) -> bool:
        """
        Return True if Azure AI Search is configured (endpoint/key), not mocked, and SDK/provider is available.
        Missing index_name is tolerated by applying a default when needed.
        """
        service = self._azure_service()
        if not service:
            return False
        has_creds = bool(
            getattr(service, "endpoint", "")
            and getattr(service, "key", "")
            and getattr(service, "key") != "mock-search-key-12345"
        )
        if not has_creds:
            return False
        # Index name may be absent; we will fill it with a default when used.
        return self._is_azure_search_available()

    async def _search_knowledge_base(
        self,
        search_query: str,
        use_azure_search: bool,
        top_k: int,
        logger: Optional[logging.Logger] = None,
    ) -> str:
        """
        Unified knowledge-base search.
        Returns a formatted text block matching test expectations.
        Implements Azure->Chroma fallback if Azure is unavailable or fails.
        """
        # ---- Azure path with safe fallback ----
        if use_azure_search:
            provider = None
            try:
                # ⚠️8: Ensure a default index is present before constructing provider
                self._ensure_default_azure_index(logger)
                from ingenious.services.azure_search.provider import (
                    AzureSearchProvider,  # type: ignore
                )

                provider = AzureSearchProvider(self._config)
                chunks: List[Dict] = await provider.retrieve(search_query, top_k=top_k)

                if not chunks:
                    # No hits: keep Azure semantics ("No relevant information..."), do NOT fall back
                    return f"No relevant information found in Azure AI Search for query: {search_query}"

                parts: List[str] = []
                for i, c in enumerate(chunks, 1):
                    content = (c.get("content", "") or "")[:600]
                    score = c.get("_final_score", "")
                    title = c.get("title", c.get("id", f"Source {i}"))
                    parts.append(f"[{i}] {title} (score={score})\n{content}")

                return (
                    "Found relevant information from Azure AI Search:\n\n"
                    + "\n\n---\n\n".join(parts)
                )

            except ImportError:
                if logger:
                    logger.warning(
                        "Azure Search SDK/provider not available; falling back to ChromaDB."
                    )
                # fall through to Chroma
            except Exception as e:
                if logger:
                    logger.warning(
                        f"Azure Search provider failed: {e}; falling back to ChromaDB."
                    )
                # fall through to Chroma
            finally:
                if provider:
                    try:
                        await provider.close()
                    except Exception:
                        pass
            # If we reach here, we are intentionally falling back to Chroma
            # without surfacing an Azure error to the end user.

        # ---- Local ChromaDB fallback/path ----
        try:
            import chromadb  # type: ignore
        except ImportError:
            return "Error: ChromaDB not installed. Please install with: uv add chromadb"

        knowledge_base_path = os.path.join(self._memory_path, "knowledge_base")
        chroma_path = os.path.join(self._memory_path, "chroma_db")

        if not os.path.exists(knowledge_base_path):
            os.makedirs(knowledge_base_path, exist_ok=True)
            return "Error: Knowledge base directory is empty. Please add documents to .tmp/knowledge_base/"

        client = chromadb.PersistentClient(path=chroma_path)
        collection_name = "knowledge_base"

        try:
            collection = client.get_collection(name=collection_name)
        except Exception:
            collection = client.create_collection(name=collection_name)

            docs, ids = await self._read_kb_documents_offthread(knowledge_base_path)
            if docs:
                try:
                    collection.add(documents=docs, ids=ids)
                except Exception as e:
                    if logger:
                        logger.warning(f"ChromaDB add() failed: {e}")
            else:
                return "Error: No documents found in knowledge base directory"

        try:
            results = collection.query(query_texts=[search_query], n_results=3)
        except Exception as e:
            if logger:
                logger.error(f"ChromaDB query failed: {e}")
            return f"Search error: {str(e)}"

        if results.get("documents") and results["documents"][0]:
            joined = "\n\n".join(results["documents"][0])
            return "Found relevant information from ChromaDB:\n\n" + joined
        else:
            return (
                f"No relevant information found in ChromaDB for query: {search_query}"
            )

    async def _read_kb_documents_offthread(
        self, kb_path: str
    ) -> Tuple[List[str], List[str]]:
        """Read .md/.txt documents from disk off-thread to avoid blocking the event loop."""

        def _read() -> Tuple[List[str], List[str]]:
            documents: List[str] = []
            ids: List[str] = []
            for filename in os.listdir(kb_path):
                if filename.endswith((".md", ".txt")):
                    filepath = os.path.join(kb_path, filename)
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            content = f.read()
                    except Exception:
                        continue
                    chunks = content.split("\n\n")
                    for i, chunk in enumerate(chunks):
                        chunk = chunk.strip()
                        if chunk:
                            documents.append(chunk)
                            ids.append(f"{filename}_chunk_{i}")
            return documents, ids

        return await asyncio.to_thread(_read)

    async def _safe_count_tokens(
        self,
        system_message: str,
        user_message: str,
        assistant_message: str,
        model: str,
        logger: Optional[logging.Logger] = None,
    ) -> Tuple[int, int]:
        """Compute token counts defensively; never fail the request."""
        try:
            from ingenious.utils.token_counter import num_tokens_from_messages

            msgs = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ]
            total = num_tokens_from_messages(msgs, model)
            prompt = num_tokens_from_messages(msgs[:-1], model)
            completion = total - prompt
            return total, completion
        except Exception as e:
            if logger:
                logger.warning(f"Token counting failed: {e}")
            return 0, 0

    def _static_system_message(self, memory_context: str) -> str:
        prefix = "You are a knowledge base search assistant that uses Azure AI Search or local ChromaDB.\n\n"
        if memory_context:
            prefix += memory_context
        prefix += (
            "Always base your responses on knowledge base search results. "
            "If nothing is found, clearly state that and suggest rephrasing the query. "
            "TERMINATE your response when the task is complete."
        )
        return prefix

    def _assist_system_message(self, memory_context: str) -> str:
        """Richer prompt for assist mode (summarization + guidelines + citation hint)."""
        parts = [
            "You are a knowledge base search assistant that can use both Azure AI Search and local ChromaDB storage.\n",
        ]
        if memory_context:
            parts.append(memory_context)

        parts.append(
            "IMPORTANT: If there is previous conversation context above, you MUST:\n"
            "- Reference it when answering follow-up questions\n"
            "- Use information from previous searches to inform new searches\n"
            "- Maintain context about what information has already been discussed\n"
            '- Answer questions that refer to "it", "that", "those" etc. based on previous context\n\n'
            "Tasks:\n"
            "- Help users find information by searching the knowledge base\n"
            "- Use the search_tool to look up information\n"
            "- Always base your responses on search results from the knowledge base\n"
            "- Always consider and reference previous conversation when relevant\n"
            "- If no information is found, clearly state that and suggest rephrasing the query\n\n"
            "Guidelines for search queries:\n"
            "- Use specific, relevant keywords\n"
            "- Try different phrasings if initial search doesn't return results\n"
            "- Focus on topics that are relevant to the knowledge base content\n\n"
            "Format your responses clearly and cite the knowledge base when providing information.\n"
            "TERMINATE your response when the task is complete."
        )
        return "".join(parts)

    def _streaming_system_message(self, memory_context: str) -> str:
        """
        Streaming prompt with guidance, topics, and citation directive.
        """
        parts: List[str] = [
            "You are a knowledge base search assistant that can use both Azure AI Search and local ChromaDB storage.\n\n"
        ]
        if memory_context:
            parts.append(memory_context)

        parts.append(
            "IMPORTANT: Maintain context and base your responses on search results.\n\n"
            "Guidelines for search queries:\n"
            "- Use specific, relevant keywords\n"
            "- Try different phrasings if initial search doesn't return results\n"
            "- Focus on topics that are relevant to the knowledge base content\n\n"
            "Knowledge base contains documents about:\n"
            "- Azure configuration and setup\n"
            "- Workplace safety guidelines\n"
            "- Health information and nutrition\n"
            "- Emergency procedures\n"
            "- Mental health and wellbeing\n"
            "- First aid basics\n"
            "- General informational content\n\n"
            "Format your responses clearly and cite the knowledge base when providing information."
        )
        return "".join(parts)
