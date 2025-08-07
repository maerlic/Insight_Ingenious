import pytest
from unittest.mock import MagicMock, AsyncMock, patch

# Imports rely on the structure defined in conftest.py
try:
    from ingenious.services.azure_search.config import SearchConfig
    from ingenious.services.azure_search.pipeline import build_search_pipeline, AdvancedSearchPipeline
    # Attempt to import Azure SDK models
    from azure.search.documents.models import QueryType
except ImportError:
    # Mock if SDK not installed
    QueryType = MagicMock()
    QueryType.SEMANTIC = "semantic"

# Helper class for mocking async iterators (used in semantic ranking)
class AsyncIter:
    def __init__(self, items):
        self.items = items
    async def __aiter__(self):
        for item in self.items:
            yield item

# --- Test Factory (build_search_pipeline) ---

# We need to patch the components where they are imported into the pipeline module.
# We rely on the patching strategy defined in conftest (apply_monkeypatch) to handle this implicitly
# when the fixtures are active. However, for testing the factory in isolation, we patch explicitly here.

# Define potential paths for patching
PIPELINE_MODULE_PATHS = [
    'ingenious.services.azure_search.pipeline',
    # 'pipeline' # Fallback path if needed, but less likely in a structured test suite
]

def apply_factory_patches(mocker, class_name, mock_instance):
    patches = []
    for path in PIPELINE_MODULE_PATHS:
        try:
            patches.append(mocker.patch(f'{path}.{class_name}', mock_instance))
        except ImportError:
            continue
    return patches

def test_build_search_pipeline_success(config: SearchConfig, mocker):
    """Test successful construction of the pipeline via the factory."""
    MockRetriever = MagicMock()
    MockFuser = MagicMock()
    MockGenerator = MagicMock()

    # Apply patches
    apply_factory_patches(mocker, 'AzureSearchRetriever', MockRetriever)
    apply_factory_patches(mocker, 'DynamicRankFuser', MockFuser)
    apply_factory_patches(mocker, 'AnswerGenerator', MockGenerator)

    pipeline = build_search_pipeline(config)

    assert isinstance(pipeline, AdvancedSearchPipeline)
    # Verify components were initialized with the config
    MockRetriever.assert_called_with(config)
    MockFuser.assert_called_with(config)
    MockGenerator.assert_called_with(config)

    # Verify the pipeline holds the instances
    assert pipeline.retriever == MockRetriever.return_value
    assert pipeline._config == config

def test_build_search_pipeline_validation_error(config_no_semantic: SearchConfig):
    """Test validation error when semantic ranking is enabled but config name is missing."""
    # Modify the config to create the invalid state: use_semantic_ranking=True, semantic_configuration_name=None
    # Handle Pydantic V1/V2 differences for creating a copy
    if hasattr(config_no_semantic, 'model_dump'):
         config_data = config_no_semantic.model_dump(exclude={'search_key', 'openai_key'})
    else:
         config_data = config_no_semantic.dict(exclude={'search_key', 'openai_key'})

    config_data['use_semantic_ranking'] = True
    config_data['semantic_configuration_name'] = None
    # Re-add secrets
    config_data['search_key'] = config_no_semantic.search_key
    config_data['openai_key'] = config_no_semantic.openai_key
    invalid_config = SearchConfig(**config_data)

    with pytest.raises(ValueError) as excinfo:
        build_search_pipeline(invalid_config)

    assert "Configuration Error: 'use_semantic_ranking' is True, but 'semantic_configuration_name' is not provided." in str(excinfo.value)

# --- Test AdvancedSearchPipeline Initialization and Helpers ---

def test_pipeline_initialization(config, retriever, fuser, generator, mock_search_client):
    """Test the initialization of the pipeline class."""
    # The mock_search_client fixture ensures that the internal _rerank_client initialization
    # uses the patched SearchClient class, returning the mock_search_client instance.

    pipeline = AdvancedSearchPipeline(config, retriever, fuser, generator)

    assert pipeline.retriever == retriever
    # Verify the dedicated _rerank_client is initialized (it will be the mock instance)
    assert pipeline._rerank_client == mock_search_client


def test_clean_sources(pipeline: AdvancedSearchPipeline, config: SearchConfig):
    """Test the cleaning of internal metadata from the final source chunks."""
    chunks = [
        {
            config.id_field: "1",
            config.content_field: "Data",
            # Internal pipeline scores
            "_retrieval_score": 10.0,
            "_normalized_score": 1.0,
            "_fused_score": 0.8,
            "_final_score": 3.5,
            "_retrieval_type": "hybrid",
            # Vector data (should be removed)
            config.vector_field: [0.1, 0.2],
            # Azure Search metadata (should be removed)
            "@search.score": 10.0,
            "@search.reranker_score": 3.5,
            "@search.captions": "Caption text",
        }
    ]

    cleaned = pipeline._clean_sources(chunks)

    assert len(cleaned) == 1
    result = cleaned[0]

    # Ensure kept fields remain
    assert result[config.id_field] == "1"
    assert result["_final_score"] == 3.5
    assert result["_retrieval_type"] == "hybrid"

    # Ensure removed fields are gone
    assert "_retrieval_score" not in result
    assert "_normalized_score" not in result
    assert "_fused_score" not in result
    assert config.vector_field not in result
    assert "@search.score" not in result
    assert "@search.reranker_score" not in result
    assert "@search.captions" not in result

# --- Test _apply_semantic_ranking (L2 Reranking) ---

@pytest.mark.asyncio
async def test_apply_semantic_ranking_success(pipeline: AdvancedSearchPipeline, config: SearchConfig, mock_search_client):
    """Test successful application of semantic ranking (L2) and data merging."""
    query = "semantic query"
    fused_results = [
        {"id": "A", "content": "Doc A", "_fused_score": 0.8, "_retrieval_type": "hybrid"},
        {"id": "B", "content": "Doc B", "_fused_score": 0.7, "_retrieval_type": "vector"},
    ]

    # Mock the response from the rerank client (SearchClient)
    # The ranker reorders and assigns @search.reranker_score
    rerank_response = [
        {"id": "B", "@search.reranker_score": 3.0, "content": "Doc B (updated)"},
        {"id": "A", "@search.reranker_score": 2.5, "content": "Doc A (updated)"},
    ]
    # The pipeline uses its internal _rerank_client, which is the mock_search_client instance
    mock_search_client.search.return_value = AsyncIter(rerank_response)

    reranked_results = await pipeline._apply_semantic_ranking(query, fused_results)

    # Verify the call to the rerank client
    # Check the filter construction (the workaround technique)
    expected_filter = f"search.in({config.id_field}, 'A,B', ',')"
    mock_search_client.search.assert_called_once_with(
        search_text=query,
        filter=expected_filter,
        query_type=QueryType.SEMANTIC,
        semantic_configuration_name=config.semantic_configuration_name,
        top=len(fused_results),
    )

    # Verify reordering and merging
    assert len(reranked_results) == 2
    assert [r["id"] for r in reranked_results] == ["B", "A"]

    # Verify scores and metadata merging (Crucial: ensuring original data is preserved/merged)
    # Doc B
    assert reranked_results[0]["_final_score"] == 3.0
    assert reranked_results[0]["_fused_score"] == 0.7 # Original fused score retained
    assert reranked_results[0]["_retrieval_type"] == "vector" # Original type retained
    assert reranked_results[0]["content"] == "Doc B (updated)" # Content updated from semantic response

@pytest.mark.asyncio
async def test_apply_semantic_ranking_truncation(pipeline: AdvancedSearchPipeline, mock_search_client):
    """Test that only the top 50 documents (MAX_RERANK_DOCS) are sent for semantic ranking."""
    query = "test truncation"
    # Create 55 fused results
    fused_results = [{"id": f"doc_{i}", "_fused_score": 1.0} for i in range(55)]

    # Mock response only needs the top 50
    rerank_response = [{"id": f"doc_{i}", "@search.reranker_score": 3.0} for i in range(50)]
    mock_search_client.search.return_value = AsyncIter(rerank_response)

    results = await pipeline._apply_semantic_ranking(query, fused_results)

    # Verify the call parameters reflect truncation
    args, kwargs = mock_search_client.search.call_args
    assert kwargs['top'] == 50
    # The filter should contain IDs doc_0 through doc_49
    assert "doc_49" in kwargs['filter']
    assert "doc_50" not in kwargs['filter']

    # Verify the results contain all 55 documents (50 reranked + 5 remaining)
    assert len(results) == 55
    # Check the top 50 have the new score
    assert results[0]["_final_score"] == 3.0
    # Check the remaining 5 (doc_50 to doc_54) are appended at the end
    assert results[50]["id"] == "doc_50"
    # The remaining docs should not have _final_score yet (as they weren't processed by the merge loop)
    assert "_final_score" not in results[50]

@pytest.mark.asyncio
async def test_apply_semantic_ranking_edge_cases(pipeline: AdvancedSearchPipeline, mock_search_client):
    """Test semantic ranking with empty input or missing IDs."""
    # Empty input
    results_empty = await pipeline._apply_semantic_ranking("query", [])
    assert results_empty == []
    mock_search_client.search.assert_not_called()

    # Missing IDs (id_field not present in data)
    pipeline._config = pipeline._config.model_copy(update={"id_field": "non_existent_id"})
    fused_results = [{"id": "A", "content": "Data"}]
    results_missing_id = await pipeline._apply_semantic_ranking("query", fused_results)
    assert results_missing_id == fused_results
    mock_search_client.search.assert_not_called()

@pytest.mark.asyncio
async def test_apply_semantic_ranking_api_failure(pipeline: AdvancedSearchPipeline, mock_search_client):
    """Test fallback behavior when the Semantic Ranking API call fails."""
    fused_results = [{"id": "A", "_fused_score": 0.8}]
    mock_search_client.search.side_effect = Exception("API Error")

    results = await pipeline._apply_semantic_ranking("query", fused_results)

    # Should return the original list, but with _final_score set to _fused_score (fallback logic)
    assert results == fused_results
    assert results[0]["_final_score"] == 0.8

# --- Test get_answer (End-to-End Pipeline Execution) ---

@pytest.mark.asyncio
async def test_get_answer_e2e_with_semantic(pipeline: AdvancedSearchPipeline, config: SearchConfig, mocker):
    """Test the end-to-end pipeline execution with semantic ranking enabled."""
    query = "E2E test query"
    assert pipeline._config.use_semantic_ranking is True

    # Mock the component methods (L1 -> Fusion -> L2 -> RAG)
    # L1 (Executed in parallel via asyncio.gather)
    mock_retriever_lexical = mocker.patch.object(pipeline.retriever, 'search_lexical', return_value=[{"id": "L1"}])
    mock_retriever_vector = mocker.patch.object(pipeline.retriever, 'search_vector', return_value=[{"id": "V1"}])
    # Fusion
    mock_fuser_fuse = mocker.patch.object(pipeline.fuser, 'fuse', return_value=[{"id": "F1", "_fused_score": 0.9}])
    # L2 (Semantic Ranking)
    mock_apply_semantic = mocker.patch.object(pipeline, '_apply_semantic_ranking', return_value=[{"id": "S1", "_final_score": 3.0, "vector": [0.1]}])
    # RAG
    mock_generator_generate = mocker.patch.object(pipeline.answer_generator, 'generate', return_value="The final answer.")

    result = await pipeline.get_answer(query)

    # Verify execution flow and data passing
    mock_retriever_lexical.assert_called_once_with(query)
    mock_retriever_vector.assert_called_once_with(query)
    mock_fuser_fuse.assert_called_once_with(query, [{"id": "L1"}], [{"id": "V1"}])
    mock_apply_semantic.assert_called_once_with(query, [{"id": "F1", "_fused_score": 0.9}])

    # RAG called with Top N results (config.top_n_final=3)
    expected_context = [{"id": "S1", "_final_score": 3.0, "vector": [0.1]}]
    mock_generator_generate.assert_called_once_with(query, expected_context)

    # Verify final result structure and cleaning
    assert result["answer"] == "The final answer."
    assert len(result["source_chunks"]) == 1
    # Sources should be cleaned (e.g., vector field removed)
    assert result["source_chunks"][0] == {"id": "S1", "_final_score": 3.0}

@pytest.mark.asyncio
async def test_get_answer_e2e_no_semantic(pipeline_no_semantic: AdvancedSearchPipeline, config_no_semantic: SearchConfig, mocker):
    """Test the E2E pipeline execution with semantic ranking disabled."""
    pipeline = pipeline_no_semantic
    query = "E2E no semantic"

    assert pipeline._config.use_semantic_ranking is False

    # Mock components
    mocker.patch.object(pipeline.retriever, 'search_lexical', return_value=[])
    mocker.patch.object(pipeline.retriever, 'search_vector', return_value=[])
    fused_results = [
        {"id": "F1", "_fused_score": 0.9},
        {"id": "F2", "_fused_score": 0.8}
    ]
    mocker.patch.object(pipeline.fuser, 'fuse', return_value=fused_results)
    # Ensure L2 is not called
    mock_apply_semantic = mocker.patch.object(pipeline, '_apply_semantic_ranking')
    mock_generator_generate = mocker.patch.object(pipeline.answer_generator, 'generate', return_value="Answer without L2.")

    result = await pipeline.get_answer(query)

    # Verify L2 was skipped
    mock_apply_semantic.assert_not_called()

    # Verify fused scores were promoted to final scores before RAG (Step 3 else branch)
    expected_context = [
        {"id": "F1", "_fused_score": 0.9, "_final_score": 0.9},
        {"id": "F2", "_fused_score": 0.8, "_final_score": 0.8}
    ]
    # Check the arguments passed to the generator
    mock_generator_generate.assert_called_once_with(query, expected_context)

    assert result["answer"] == "Answer without L2."
    assert result["source_chunks"][0]["_final_score"] == 0.9

@pytest.mark.asyncio
async def test_get_answer_no_results_found(pipeline: AdvancedSearchPipeline, mocker):
    """Test the pipeline when L1 retrieval and fusion yield no results."""
    query = "no results"
    mocker.patch.object(pipeline.retriever, 'search_lexical', return_value=[])
    mocker.patch.object(pipeline.retriever, 'search_vector', return_value=[])
    mocker.patch.object(pipeline.fuser, 'fuse', return_value=[])
    mock_generator_generate = mocker.patch.object(pipeline.answer_generator, 'generate')

    result = await pipeline.get_answer(query)

    # Verify the specific message when no context is found (Step 4 check)
    assert "I could not find any relevant information" in result["answer"]
    assert result["source_chunks"] == []
    # Generator should not be called
    mock_generator_generate.assert_not_called()

# --- Test Pipeline Error Handling ---

@pytest.mark.asyncio
async def test_get_answer_retrieval_failure(pipeline: AdvancedSearchPipeline, mocker):
    """Test error handling during the L1 retrieval phase (Step 1)."""
    # Simulate failure in one of the parallel L1 tasks (asyncio.gather propagates the first exception)
    mocker.patch.object(pipeline.retriever, 'search_lexical', side_effect=Exception("L1 Error"))
    mocker.patch.object(pipeline.retriever, 'search_vector', return_value=[])

    # Verify exception is caught and re-raised as RuntimeError
    with pytest.raises(RuntimeError) as excinfo:
        await pipeline.get_answer("query")

    assert "L1 Retrieval failed." in str(excinfo.value)

@pytest.mark.asyncio
async def test_get_answer_fusion_failure(pipeline: AdvancedSearchPipeline, mocker):
    """Test error handling during the DAT fusion phase (Step 2)."""
    mocker.patch.object(pipeline.retriever, 'search_lexical', return_value=[{"id": "L1"}])
    mocker.patch.object(pipeline.retriever, 'search_vector', return_value=[{"id": "V1"}])
    mocker.patch.object(pipeline.fuser, 'fuse', side_effect=Exception("DAT Error"))

    with pytest.raises(RuntimeError) as excinfo:
        await pipeline.get_answer("query")

    assert "DAT Fusion failed." in str(excinfo.value)

@pytest.mark.asyncio
async def test_get_answer_generation_failure(pipeline: AdvancedSearchPipeline, mocker):
    """Test error handling during the RAG generation phase (Step 5)."""
    mocker.patch.object(pipeline.retriever, 'search_lexical', return_value=[{"id": "L1"}])
    mocker.patch.object(pipeline.retriever, 'search_vector', return_value=[{"id": "V1"}])
    mocker.patch.object(pipeline.fuser, 'fuse', return_value=[{"id": "F1"}])
    # Assuming L2 succeeds
    mocker.patch.object(pipeline, '_apply_semantic_ranking', return_value=[{"id": "S1"}])
    mocker.patch.object(pipeline.answer_generator, 'generate', side_effect=Exception("RAG Error"))

    with pytest.raises(RuntimeError) as excinfo:
        await pipeline.get_answer("query")

    assert "Answer Generation failed." in str(excinfo.value)

# --- Test Pipeline Close ---

@pytest.mark.asyncio
async def test_pipeline_close(pipeline: AdvancedSearchPipeline, mocker):
    """Test that the close method closes all underlying components and the rerank client."""
    # Mock the close methods of the components (which are injected via fixtures)
    mock_retriever_close = mocker.patch.object(pipeline.retriever, 'close', new_callable=AsyncMock)
    mock_fuser_close = mocker.patch.object(pipeline.fuser, 'close', new_callable=AsyncMock)
    mock_generator_close = mocker.patch.object(pipeline.answer_generator, 'close', new_callable=AsyncMock)
    # Mock the close method of the internal rerank client
    mock_rerank_client_close = mocker.patch.object(pipeline._rerank_client, 'close', new_callable=AsyncMock)

    await pipeline.close()

    # Verify all async close methods were awaited
    mock_retriever_close.assert_awaited_once()
    mock_fuser_close.assert_awaited_once()
    mock_generator_close.assert_awaited_once()
    mock_rerank_client_close.assert_awaited_once()