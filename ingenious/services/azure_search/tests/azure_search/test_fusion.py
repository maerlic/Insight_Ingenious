import pytest
from unittest.mock import AsyncMock, MagicMock
from types import SimpleNamespace
import math

# Imports rely on the structure defined in conftest.py
try:
    from ingenious.services.azure_search.config import SearchConfig
    from ingenious.services.azure_search.components.fusion import DynamicRankFuser
except ImportError:
     pytest.skip("Could not import Fusion components", allow_module_level=True)

@pytest.fixture
def mock_dat_response():
    """Factory fixture to mock the OpenAI chat completion API response for DAT."""
    def _factory(content: str):
        # Mocking the structure: response.choices[0].message.content
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content)
                )
            ]
        )
    return _factory

def test_fuser_initialization(config, mock_openai_client):
    """Test that the fuser initializes the LLM client using the configuration."""
    # The fuser fixture handles initialization, we just verify the state
    fuser = DynamicRankFuser(config)
    # The client should be the mocked instance injected by the fixture/patching
    assert fuser._llm_client == mock_openai_client

# --- Test _calculate_alpha (Core DAT Logic) ---

@pytest.mark.parametrize("score_v, score_l, expected_alpha", [
    # Case 1: Both methods fail (0, 0) -> 0.5
    (0, 0, 0.5),
    # Case 2: Vector is direct hit (5) and Lexical is not -> 1.0
    (5, 4, 1.0),
    (5, 0, 1.0),
    # Case 3: Lexical is direct hit (5) and Vector is not -> 0.0
    (4, 5, 0.0),
    (0, 5, 0.0),
    # Case 4: Proportional weighting
    (3, 3, 0.5),
    (4, 1, 0.8), # 4/(4+1)
    (1, 4, 0.2), # 1/(1+4)
    (5, 5, 0.5), # Handled by Case 4: 5/(5+5)
    # Test rounding to one decimal place (Section 4.2 requirement)
    (1, 2, 0.3), # 1/3 = 0.333... -> 0.3
    (2, 1, 0.7), # 2/3 = 0.666... -> 0.7
])
def test_calculate_alpha_logic(fuser: DynamicRankFuser, score_v, score_l, expected_alpha):
    """Test the case-aware logic for calculating alpha based on DAT scores (Eq. 6)."""
    alpha = fuser._calculate_alpha(score_v, score_l)
    assert alpha == expected_alpha

# --- Test _parse_dat_scores (LLM Output Handling) ---

@pytest.mark.parametrize("llm_output, expected_scores", [
    ("3 4", (3, 4)),
    (" 5 1 ", (5, 1)),
    ("Dense: 5, BM25: 2 (Explanation)", (5, 2)), # Robust regex matching
    ("0 0", (0, 0)),
    ("3 4 5", (3, 4)), # Takes first two numbers
])
def test_parse_dat_scores_valid(fuser: DynamicRankFuser, llm_output, expected_scores):
    """Test parsing valid LLM outputs."""
    scores = fuser._parse_dat_scores(llm_output)
    assert scores == expected_scores

@pytest.mark.parametrize("llm_output", [
    "invalid output",
    "3",          # Only one number
    "A B",        # Non-numeric
    "",           # Empty string
])
def test_parse_dat_scores_invalid_format(fuser: DynamicRankFuser, llm_output):
    """Test parsing invalid LLM output formats (fallback to 0, 0)."""
    scores = fuser._parse_dat_scores(llm_output)
    assert scores == (0, 0)

@pytest.mark.parametrize("llm_output", [
    "6 4", # Vector > 5
    "3 7", # Lexical > 5
    "-1 2", # Negative scores
])
def test_parse_dat_scores_out_of_range(fuser: DynamicRankFuser, llm_output):
    """Test parsing scores outside the 0-5 range (fallback to 0, 0)."""
    scores = fuser._parse_dat_scores(llm_output)
    assert scores == (0, 0)

# --- Test _normalize_scores (Min-Max Normalization) ---

def test_normalize_scores_standard(fuser: DynamicRankFuser):
    """Test Min-Max normalization with a standard set of scores."""
    results = [
        {"id": "A", "_retrieval_score": 20.0},
        {"id": "B", "_retrieval_score": 15.0},
        {"id": "C", "_retrieval_score": 10.0},
    ]
    fuser._normalize_scores(results)

    # Min=10, Max=20, Range=10
    assert results[0]["_normalized_score"] == 1.0 # (20-10)/10
    assert results[1]["_normalized_score"] == 0.5 # (15-10)/10
    assert results[2]["_normalized_score"] == 0.0 # (10-10)/10

def test_normalize_scores_all_same(fuser: DynamicRankFuser):
    """Test normalization when all scores are identical (Min=Max)."""
    results = [
        {"id": "A", "_retrieval_score": 5.0},
        {"id": "B", "_retrieval_score": 5.0},
    ]
    fuser._normalize_scores(results)
    # Should result in 1.0 if score > 0
    assert results[0]["_normalized_score"] == 1.0
    assert results[1]["_normalized_score"] == 1.0

def test_normalize_scores_all_zero(fuser: DynamicRankFuser):
    """Test normalization when all scores are zero."""
    results = [{"id": "A", "_retrieval_score": 0.0}]
    fuser._normalize_scores(results)
    # Should result in 0.0 if score == 0
    assert results[0]["_normalized_score"] == 0.0

def test_normalize_scores_empty_list(fuser: DynamicRankFuser):
    """Test normalization with an empty list."""
    results = []
    fuser._normalize_scores(results)
    assert results == []

def test_normalize_scores_invalid_types(fuser: DynamicRankFuser):
    """Test robust handling of None, missing, or non-numeric scores."""
    results = [
        {"id": "A", "_retrieval_score": 10.0},
        {"id": "B", "_retrieval_score": None},
        {"id": "C", "_retrieval_score": "invalid"},
        {"id": "D"}, # Missing score key
        {"id": "E", "_retrieval_score": 5.0},
    ]
    fuser._normalize_scores(results)

    # Invalid/missing scores are treated as 0.0. Min=0, Max=10.
    assert results[0]["_normalized_score"] == 1.0
    assert results[1]["_normalized_score"] == 0.0 # None -> 0.0
    assert results[2]["_normalized_score"] == 0.0 # "invalid" -> 0.0
    assert results[3]["_normalized_score"] == 0.0 # Missing -> 0.0
    assert results[4]["_normalized_score"] == 0.5

# --- Test _perform_dat (LLM Interaction) ---

@pytest.mark.asyncio
async def test_perform_dat_success(fuser: DynamicRankFuser, config: SearchConfig, mock_openai_client, mock_dat_response):
    """Test successful execution of the DAT LLM call and parsing."""
    query = "what is the capital of France?"
    # Use the configured content field name
    top_lexical = {"id": "L1", config.content_field: "Paris is a city."}
    # Test truncation (1500 chars)
    long_vector_content = "France's capital is Paris. " + "A"*1600
    top_vector = {"id": "V1", config.content_field: long_vector_content}

    # Mock LLM response: Vector (5), Lexical (3)
    mock_openai_client.chat.completions.create.return_value = mock_dat_response("5 3")

    alpha = await fuser._perform_dat(query, top_lexical, top_vector)

    # Expected alpha for (5, 3) is 1.0 (Case 2)
    assert alpha == 1.0

    # Verify the call to the LLM
    mock_openai_client.chat.completions.create.assert_called_once()
    args, kwargs = mock_openai_client.chat.completions.create.call_args

    assert kwargs['model'] == config.generation_deployment_name
    assert kwargs['temperature'] == 0.0
    assert kwargs['max_tokens'] == 10

    messages = kwargs['messages']
    assert messages[0]['role'] == 'system'
    assert messages[0]['content'] == config.dat_prompt
    user_prompt = messages[1]['content']
    
    # Check prompt content and truncation
    assert query in user_prompt
    assert top_lexical[config.content_field] in user_prompt
    assert long_vector_content[:1500] in user_prompt
    assert long_vector_content not in user_prompt # Ensure it was truncated

@pytest.mark.asyncio
async def test_perform_dat_llm_exception(fuser: DynamicRankFuser, mock_openai_client):
    """Test fallback behavior when the LLM API call fails."""
    mock_openai_client.chat.completions.create.side_effect = Exception("API Error")

    alpha = await fuser._perform_dat("query", {"content": "doc1"}, {"content": "doc2"})

    # Should fall back to equal weight (0.5) on error
    assert alpha == 0.5

@pytest.mark.asyncio
async def test_perform_dat_parsing_failure(fuser: DynamicRankFuser, mock_openai_client, mock_dat_response):
    """Test behavior when LLM returns unparseable output."""
    # Mock LLM response that cannot be parsed
    mock_openai_client.chat.completions.create.return_value = mock_dat_response("I cannot decide.")

    alpha = await fuser._perform_dat("query", {"content": "doc1"}, {"content": "doc2"})

    # Parsing failure results in scores (0, 0), which leads to alpha 0.5
    assert alpha == 0.5

# --- Test fuse (Integration of all steps) ---

@pytest.mark.asyncio
async def test_fuse_happy_path(fuser: DynamicRankFuser, mocker):
    """Test the complete fusion process with dynamic alpha tuning and overlapping results."""
    query = "test fusion"

    # Setup: Lexical (BM25 scores): A=10, B=8, C=5
    lexical_results = [
        {"id": "A", "content": "Doc A", "_retrieval_score": 10.0, "_retrieval_type": "lexical"},
        {"id": "B", "content": "Doc B", "_retrieval_score": 8.0, "_retrieval_type": "lexical"},
        {"id": "C", "content": "Doc C", "_retrieval_score": 5.0, "_retrieval_type": "lexical"},
    ]
    # Vector (Similarity scores): C=0.9, D=0.8, E=0.7
    vector_results = [
        {"id": "C", "content": "Doc C", "_retrieval_score": 0.9, "_retrieval_type": "vector"},
        {"id": "D", "content": "Doc D", "_retrieval_score": 0.8, "_retrieval_type": "vector"},
        {"id": "E", "content": "Doc E", "_retrieval_score": 0.7, "_retrieval_type": "vector"},
    ]

    # Mock the DAT step: Assume Alpha = 0.6 (favors Vector/Dense)
    mock_perform_dat = mocker.patch.object(fuser, '_perform_dat', return_value=0.6)

    fused_results = await fuser.fuse(query, lexical_results, vector_results)

    # Verify DAT was called with the top results
    mock_perform_dat.assert_called_once_with(query, lexical_results[0], vector_results[0])

    # Verify Calculations:
    # Lexical Norm (Min=5, Max=10, Range=5): A=1.0, B=0.6, C=0.0
    # Vector Norm (Min=0.7, Max=0.9, Range=0.2): C=1.0, D=0.5, E=0.0
    # Fusion (Alpha=0.6, 1-Alpha=0.4): R = α · S_dense + (1 - α) · S_BM25

    # Doc C (Hybrid): (0.6 * 1.0) + (0.4 * 0.0) = 0.6
    # Doc A (Lexical): (0.6 * 0) + (0.4 * 1.0) = 0.4
    # Doc D (Vector): (0.6 * 0.5) + (0.4 * 0) = 0.3
    # Doc B (Lexical): (0.6 * 0) + (0.4 * 0.6) = 0.24
    # Doc E (Vector): (0.6 * 0.0) + (0.4 * 0) = 0.0

    # Expected Order: C, A, D, B, E
    assert len(fused_results) == 5
    assert [r["id"] for r in fused_results] == ["C", "A", "D", "B", "E"]

    # Verify scores
    assert math.isclose(fused_results[0]["_fused_score"], 0.6)
    assert math.isclose(fused_results[1]["_fused_score"], 0.4)

    # Verify metadata update for hybrid document
    assert fused_results[0]["_retrieval_type"] == "hybrid_dat_alpha_0.6"
    # Verify metadata retention for non-hybrid documents
    assert fused_results[1]["_retrieval_type"] == "lexical"

@pytest.mark.asyncio
async def test_fuse_empty_lists(fuser: DynamicRankFuser):
    """Test fusion when both input lists are empty."""
    results = await fuser.fuse("query", [], [])
    assert results == []

@pytest.mark.asyncio
async def test_fuse_one_list_empty(fuser: DynamicRankFuser, mocker, fake_results):
    """Test fusion when one list is empty (should skip DAT and normalization)."""
    vector_results = fake_results(3, prefix="V", start_score=0.9)
    lexical_results = fake_results(3, prefix="L", start_score=10.0)
    
    # Ensure DAT and normalization are not called
    mock_perform_dat = mocker.patch.object(fuser, '_perform_dat')
    mock_normalize = mocker.patch.object(fuser, '_normalize_scores')

    # Case 1: Lexical empty
    results_v = await fuser.fuse("query", [], vector_results)
    assert results_v == vector_results
    assert results_v[0]["_fused_score"] == 0.9 # Fused score copied directly

    # Case 2: Vector empty
    results_l = await fuser.fuse("query", lexical_results, [])
    assert results_l == lexical_results
    assert results_l[0]["_fused_score"] == 10.0

    mock_perform_dat.assert_not_called()
    mock_normalize.assert_not_called()

@pytest.mark.asyncio
async def test_fuse_missing_ids(fuser: DynamicRankFuser, mocker):
    """Test that documents missing the ID field are skipped during fusion merge."""
    # Ensure DAT still runs if top docs are valid
    mocker.patch.object(fuser, '_perform_dat', return_value=0.5)

    lexical = [{"id": "L1", "_retrieval_score": 10}, {"content": "No ID", "_retrieval_score": 5}]
    vector = [{"id": "V1", "_retrieval_score": 0.8}]

    results = await fuser.fuse("query", lexical, vector)

    # Should only contain L1 and V1
    assert len(results) == 2
    assert {r.get("id") for r in results} == {"L1", "V1"}

@pytest.mark.asyncio
async def test_fuser_close(fuser: DynamicRankFuser, mock_openai_client):
    """Test that the close method closes the underlying LLM client."""
    await fuser.close()
    mock_openai_client.close.assert_awaited_once()