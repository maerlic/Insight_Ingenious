"""Tests memory context building and logging for the Knowledge Base Agent.

This module contains tests for the `_build_memory_context` method within the
Knowledge Base Agent's conversation flow. It specifically verifies:
1.  Correct truncation of the conversation history to the last 10 messages.
2.  Truncation of individual long messages to 100 characters.
3.  Throttling of warning logs when fetching chat history fails repeatedly,
    preventing log spam.

Tests use pytest fixtures like `monkeypatch`, `tmp_path`, and `caplog` to
isolate behavior and inspect outcomes.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, NoReturn

import pytest

import ingenious.services.chat_services.multi_agent.conversation_flows.knowledge_base_agent.knowledge_base_agent as kb
from ingenious.models.chat import ChatRequest

if TYPE_CHECKING:
    from pathlib import Path

    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch


@pytest.mark.asyncio
async def test_memory_context_truncates_last_10_and_100_chars(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    """Verify memory context truncates history to the last 10 messages and long messages to 100 chars."""
    # Prepare 12 messages, one of them long (>100 chars)
    msgs: list[SimpleNamespace] = []
    for i in range(12):
        content: str = (f"msg{i}-" + "X" * 120) if i == 11 else f"msg{i}"
        msgs.append(SimpleNamespace(role="user", content=content))

    async def _get_thread_messages(tid: str) -> list[SimpleNamespace]:
        """Mock message retrieval to return the predefined list."""
        return msgs

    repo = SimpleNamespace(get_thread_messages=_get_thread_messages)
    svc = SimpleNamespace(chat_history_repository=repo)

    flow = kb.ConversationFlow.__new__(kb.ConversationFlow)
    flow._config = SimpleNamespace(models=[SimpleNamespace(model="gpt-4o")])
    flow._chat_service = svc
    flow._memory_path = str(tmp_path)

    ctx: str = await flow._build_memory_context(
        ChatRequest(user_prompt="q", thread_id="t1")
    )

    # Only last 10 messages (drop msg0 and msg1); avoid 'msg1' vs 'msg10' collision
    assert "user: msg0..." not in ctx
    assert "user: msg1..." not in ctx
    assert "user: msg2..." in ctx and "user: msg11" in ctx
    # Long one is truncated to first 100 chars (including the 'msg11-' prefix), then "..."
    long_prefix: str = "msg11-"
    expected_trunc: str = long_prefix + ("X" * (100 - len(long_prefix)))
    assert (expected_trunc + "...") in ctx
    assert ctx.startswith("Previous conversation:\n")


@pytest.mark.asyncio
async def test_memory_warning_throttled_to_once_within_60s(
    monkeypatch: MonkeyPatch, caplog: LogCaptureFixture, tmp_path: Path
) -> None:
    """Verify memory retrieval failure warnings are throttled to once per 60 seconds.

    This test ensures that if fetching chat history fails, the resulting
    warning log is not spammed. It should only be logged once within a 60-second
    window, with subsequent failures in that window logged at a DEBUG level.
    """

    async def _always_fail(_tid: str) -> NoReturn:
        """Mock message retrieval that always fails."""
        # Raising any Exception exercises the same error path the test is asserting.
        raise RuntimeError("db down")

    repo = SimpleNamespace(get_thread_messages=_always_fail)
    svc = SimpleNamespace(chat_history_repository=repo)
    flow = kb.ConversationFlow.__new__(kb.ConversationFlow)
    flow._config = SimpleNamespace(models=[SimpleNamespace(model="gpt-4o")])
    flow._chat_service = svc
    flow._memory_path = str(tmp_path)

    # Freeze time to keep delta < 60s between calls
    t: list[float] = [1000.0]
    monkeypatch.setattr(kb.time, "monotonic", lambda: t[0])

    req = ChatRequest(user_prompt="q", thread_id="t1")

    caplog.set_level(logging.DEBUG)
    _ = await flow._build_memory_context(req)  # first call -> warning
    # second call within <60s -> debug-level suppression
    t[0] = 1010.0
    _ = await flow._build_memory_context(req)

    warns: list[logging.LogRecord] = [
        r for r in caplog.records if r.levelno >= logging.WARNING
    ]
    debugs: list[logging.LogRecord] = [
        r for r in caplog.records if r.levelno == logging.DEBUG
    ]
    assert any("Failed to retrieve thread memory:" in r.getMessage() for r in warns)
    assert any("suppressed" in r.getMessage() for r in debugs)
