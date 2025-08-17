from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class IChatRequest(BaseModel):
    thread_id: Optional[str] = None
    user_prompt: str
    event_type: Optional[str] = None
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    topic: Optional[str] = None
    memory_record: Optional[bool] = True
    conversation_flow: Optional[str] = None
    # avoid mutable default
    thread_chat_history: Optional[dict[str, str]] = Field(default_factory=dict)
    thread_memory: Optional[str] = None
    stream: Optional[bool] = False
    # Optional overrides for KB behavior
    kb_top_k: Optional[int] = None
    parameters: Optional[Dict[str, Any]] = None


class IChatResponse(BaseModel):
    thread_id: Optional[str]
    message_id: Optional[str]
    agent_response: Optional[str]
    followup_questions: Optional[dict[str, str]] = Field(default_factory=dict)
    token_count: Optional[int]
    max_token_count: Optional[int]
    topic: Optional[str] = None
    memory_summary: Optional[str] = None
    event_type: Optional[str] = None


class ChatRequest(IChatRequest):
    pass


class ChatResponse(IChatResponse):
    pass


class Action(BaseModel):
    name: str
    description: Optional[str] = None


class KnowledgeBaseLink(BaseModel):
    title: str
    url: str
    description: Optional[str] = None


class Product(BaseModel):
    name: str
    description: Optional[str] = None
    price: Optional[float] = None


class ChatResponseChunk(BaseModel):
    thread_id: Optional[str]
    message_id: Optional[str]
    chunk_type: (
        str  # "content", "token_count", "memory_summary", "followup_questions", "final"
    )
    content: Optional[str] = None
    token_count: Optional[int] = None
    max_token_count: Optional[int] = None
    topic: Optional[str] = None
    memory_summary: Optional[str] = None
    followup_questions: Optional[dict[str, str]] = None
    event_type: Optional[str] = None
    is_final: bool = False


class StreamingChatResponse(BaseModel):
    """Response model for streaming chat endpoints"""

    event: str  # "data", "error", "done"
    data: Optional[ChatResponseChunk] = None
    error: Optional[str] = None
