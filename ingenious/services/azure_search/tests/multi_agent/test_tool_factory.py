# tests/multi_agent/test_tool_factory.py
import json
from types import SimpleNamespace

import pytest

from ingenious.services.chat_services.multi_agent.tool_factory import ToolFunctions


@pytest.mark.asyncio
async def test_tool_factory_aisearch_success(monkeypatch):
    # Patch get_config to return stub
    monkeypatch.setattr(
        "ingenious.config.get_config",
        lambda: SimpleNamespace(),
    )

    # Patch AzureSearchProvider to return cleaned chunks
    class FakeProvider:
        def __init__(self, *_):
            pass

        async def retrieve(self, query, top_k=5):
            return [
                {
                    "id": "1",
                    "content": "A",
                    "title": "T1",
                    "_final_score": 1.23,
                    "vector": [0.1],
                },  # vector ignored by tool
                {"id": "2", "content": "B", "_final_score": 0.5},
            ]

        async def close(self):
            pass

    monkeypatch.setattr(
        "ingenious.services.azure_search.provider.AzureSearchProvider",
        FakeProvider,
    )

    out = await ToolFunctions.aisearch("my query")
    data = json.loads(out)
    assert data["@odata.count"] == 2
    assert data["value"][0]["@search.score"] == 1.23
    assert data["value"][0]["content"] == "A"
    assert data["value"][0]["title"] == "T1"  # uses title if present
    # second item title falls back to id
    assert data["value"][1]["title"] == "2"


@pytest.mark.asyncio
async def test_tool_factory_aisearch_failure_returns_error(monkeypatch):
    monkeypatch.setattr(
        "ingenious.config.get_config",
        lambda: SimpleNamespace(),
    )

    class BadProvider:
        def __init__(self, *_):
            pass

        async def retrieve(self, *_args, **_kwargs):
            raise RuntimeError("boom")

        async def close(self):
            pass

    monkeypatch.setattr(
        "ingenious.services.azure_search.provider.AzureSearchProvider",
        BadProvider,
    )

    out = await ToolFunctions.aisearch("q")
    data = json.loads(out)
    # Now expects a mock response instead of an error
    assert "@odata.count" in data
    assert data["@odata.count"] == 1
    assert len(data["value"]) == 1
    assert "Mock search result" in data["value"][0]["content"]
    assert data["value"][0]["title"] == "Mock Document q"
