"""
FILE TEST PLAN

    DynamicRankFuser:
        _calculate_alpha cases incl. rounding
        _parse_dat_scores robust regex & out-of-range fallback
        _normalize_scores paths (standard, same, zeros, invalid, empty)
        _perform_dat success (prompts/params/truncation) + exceptions → 0.5 fallback
        fuse end-to-end with overlap/metadata + one-list-empty fast paths + missing ids handling
        close() closes underlying client
"""

import math
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ingenious.services.azure_search.components.fusion import DynamicRankFuser
from ingenious.services.azure_search.config import SearchConfig


@pytest.fixture
def fuser(config: SearchConfig):
    return DynamicRankFuser(config)


@pytest.mark.parametrize(
    "sv,sl,exp",
    [
        (0, 0, 0.5),
        (5, 2, 1.0),
        (1, 5, 0.0),
        (3, 3, 0.5),
        (4, 1, 0.8),
        (1, 4, 0.2),
        (5, 5, 0.5),
        (1, 2, 0.3),
        (2, 1, 0.7),
    ],
)
def test_calculate_alpha_cases(fuser: DynamicRankFuser, sv, sl, exp):
    assert fuser._calculate_alpha(sv, sl) == exp


@pytest.mark.parametrize(
    "out,expected",
    [
        ("3 4", (3, 4)),
        (" 5 1 ", (5, 1)),
        ("Dense=5; BM=2 (ok)", (5, 2)),
        ("0 0", (0, 0)),
        ("3 4 5", (3, 4)),
    ],
)
def test_parse_scores_ok(fuser: DynamicRankFuser, out, expected):
    assert fuser._parse_dat_scores(out) == expected


@pytest.mark.parametrize("out", ["", "3", "A B"])
def test_parse_scores_bad(fuser: DynamicRankFuser, out):
    assert fuser._parse_dat_scores(out) == (0, 0)


@pytest.mark.parametrize("out", ["6 4", "3 9", "-1 3"])
def test_parse_scores_oob(fuser: DynamicRankFuser, out):
    assert fuser._parse_dat_scores(out) == (0, 0)


def test_normalize_scores_paths(fuser: DynamicRankFuser):
    rows = [
        {"id": "A", "_retrieval_score": 20.0},
        {"id": "B", "_retrieval_score": 15.0},
        {"id": "C", "_retrieval_score": 10.0},
    ]
    fuser._normalize_scores(rows)
    assert rows[0]["_normalized_score"] == 1.0
    assert rows[1]["_normalized_score"] == 0.5
    assert rows[2]["_normalized_score"] == 0.0

    same = [{"id": "A", "_retrieval_score": 5.0}, {"id": "B", "_retrieval_score": 5.0}]
    fuser._normalize_scores(same)
    assert same[0]["_normalized_score"] == 0.5 == same[1]["_normalized_score"]

    zeros = [{"id": "A", "_retrieval_score": 0.0}]
    fuser._normalize_scores(zeros)
    # Spec-compliant degeneracy: neutral 0.5 even when all scores are zero
    assert zeros[0]["_normalized_score"] == 0.5

    empty = []
    fuser._normalize_scores(empty)
    assert empty == []

    invalid = [
        {"id": "A", "_retrieval_score": 10.0},
        {"id": "B", "_retrieval_score": None},
        {"id": "C", "_retrieval_score": "bad"},
        {"id": "D"},
        {"id": "E", "_retrieval_score": 5.0},
    ]
    fuser._normalize_scores(invalid)
    assert invalid[0]["_normalized_score"] == 1.0
    assert invalid[1]["_normalized_score"] == 0.0
    assert invalid[2]["_normalized_score"] == 0.0
    assert invalid[3]["_normalized_score"] == 0.0
    assert invalid[4]["_normalized_score"] == 0.5


@pytest.mark.asyncio
async def test_perform_dat_success(
    fuser: DynamicRankFuser, config: SearchConfig, monkeypatch
):
    # Replace client method to return desired "5 3"
    async_client = fuser._llm_client
    async_client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="5 3"))]
        )
    )
    long = "X" * 2000
    a = await fuser._perform_dat(
        "Q", {config.content_field: "A"}, {config.content_field: long}
    )
    assert a == 1.0
    async_client.chat.completions.create.assert_awaited()


@pytest.mark.asyncio
async def test_perform_dat_error_fallback(fuser: DynamicRankFuser, monkeypatch):
    async_client = fuser._llm_client
    async_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))
    a = await fuser._perform_dat("Q", {"content": "a"}, {"content": "b"})
    assert a == 0.5


@pytest.mark.asyncio
async def test_fuse_e2e_and_fast_paths(fuser: DynamicRankFuser, monkeypatch):
    # Alpha = 0.6
    monkeypatch.setattr(fuser, "_perform_dat", AsyncMock(return_value=0.6))
    lexical = [
        {"id": "A", "content": "A", "_retrieval_score": 10.0, "_retrieval_type": "lex"},
        {"id": "B", "content": "B", "_retrieval_score": 8.0, "_retrieval_type": "lex"},
        {"id": "C", "content": "C", "_retrieval_score": 5.0, "_retrieval_type": "lex"},
    ]
    vector = [
        {"id": "C", "content": "C", "_retrieval_score": 0.9, "_retrieval_type": "vec"},
        {"id": "D", "content": "D", "_retrieval_score": 0.8, "_retrieval_type": "vec"},
        {"id": "E", "content": "E", "_retrieval_score": 0.7, "_retrieval_type": "vec"},
    ]
    fused = await fuser.fuse("Q", lexical, vector)
    assert [r["id"] for r in fused] == ["C", "A", "D", "B", "E"]
    assert math.isclose(fused[0]["_fused_score"], 0.6)
    assert fused[0]["_retrieval_type"].startswith("hybrid_dat_alpha")

    # one-list-empty paths
    vec_only = await fuser.fuse("Q", [], [{"id": "V1", "_retrieval_score": 0.9}])
    assert 0.0 <= vec_only[0]["_fused_score"] <= 1.0
    # and check order:
    assert [r["id"] for r in vec_only] == ["V1"]  # or the expected top-N sequence
    lex_only = await fuser.fuse("Q", [{"id": "L1", "_retrieval_score": 1.1}], [])
    # Single-list case uses per-method min–max; degenerate positive → 0.5, α=0.0 → fused 0.5
    assert lex_only[0]["_fused_score"] == 0.5

    # missing id ignored
    out = await fuser.fuse(
        "Q",
        [{"id": "L1", "_retrieval_score": 1.0}, {"_retrieval_score": 0.1}],
        [{"id": "V1", "_retrieval_score": 0.2}],
    )
    assert {d.get("id") for d in out} == {"L1", "V1"}


@pytest.mark.asyncio
async def test_fuser_close(fuser: DynamicRankFuser):
    await fuser.close()
    fuser._llm_client.close.assert_awaited()
