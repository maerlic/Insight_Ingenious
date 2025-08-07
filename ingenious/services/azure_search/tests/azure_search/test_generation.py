import pytest
from unittest.mock import AsyncMock, MagicMock
from types import SimpleNamespace

# Imports rely on the structure defined in conftest.py
try:
    from ingenious.services.azure_search.config import SearchConfig
    from ingenious.services.azure_search.components.generation import AnswerGenerator, DEFAULT_RAG_PROMPT
except ImportError:
    pytest.skip("Could not import Generation components", allow_module_level=True)

@pytest.fixture
def mock_rag_response():
    """Mocks the OpenAI chat completion API response for RAG."""
    # Mocking structure: response.choices[0].message.content
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="This is the generated answer. [Source 1]")
            )
        ]
    )

def test_generator_initialization(config, mock_openai_client):
    """Test that the generator initializes the LLM client and prompt."""
    generator = AnswerGenerator(config)
    assert generator._llm_client == mock_openai_client
    assert generator.rag_prompt_template == DEFAULT_RAG_PROMPT

# --- Test _format_context ---

def test_format_context(generator: AnswerGenerator, config: SearchConfig):
    """Test the formatting of context chunks for the RAG prompt."""
    chunks = [
        {"id": "1", config.content_field: "Content A."},
        {"id": "2", config.content_field: "Content B."},
    ]

    formatted_context = generator._format_context(chunks)

    # Define expected output
    expected_output = (
        "[Source 1]\nContent A.\n"
        "\n---\n"
        "[Source 2]\nContent B.\n"
    )
    # Use repr() to ensure exact match including newlines
    assert repr(formatted_context) == repr(expected_output)

def test_format_context_missing_content(generator: AnswerGenerator, config: SearchConfig):
    """Test context formatting when the content field is missing."""
    chunks = [{"id": "1", "other_field": "Data"}]
    formatted_context = generator._format_context(chunks)
    # Should use "N/A" as fallback
    assert "[Source 1]\nN/A\n" in repr(formatted_context)

def test_format_context_empty(generator: AnswerGenerator):
    """Test formatting an empty list."""
    assert generator._format_context([]) == ""

# --- Test generate ---

@pytest.mark.asyncio
async def test_generate_success(generator: AnswerGenerator, config: SearchConfig, mock_openai_client, mock_rag_response):
    """Test successful answer generation."""
    query = "Tell me about X."
    chunks = [{"id": "1", config.content_field: "X is important."}]

    mock_openai_client.chat.completions.create.return_value = mock_rag_response

    answer = await generator.generate(query, chunks)

    assert answer == "This is the generated answer. [Source 1]"

    # Verify the call to the LLM
    mock_openai_client.chat.completions.create.assert_called_once()
    args, kwargs = mock_openai_client.chat.completions.create.call_args

    assert kwargs['model'] == config.generation_deployment_name
    assert kwargs['temperature'] == 0.1
    assert kwargs['max_tokens'] == 1500

    messages = kwargs['messages']
    assert messages[0]['role'] == 'system'
    system_prompt = messages[0]['content']
    
    # Check prompt structure and context inclusion
    assert DEFAULT_RAG_PROMPT.split('{context}')[0] in system_prompt
    assert "[Source 1]" in system_prompt
    assert "X is important." in system_prompt

    assert messages[1]['role'] == 'user'
    assert messages[1]['content'] == f"Question: {query}"

@pytest.mark.asyncio
async def test_generate_empty_context(generator: AnswerGenerator, mock_openai_client):
    """Test generation when no context chunks are provided."""
    query = "Tell me about Y."
    answer = await generator.generate(query, [])

    # Verify specific message for no context
    assert "I could not find any relevant information" in answer
    # Ensure the LLM was not called (optimization path)
    mock_openai_client.chat.completions.create.assert_not_called()

@pytest.mark.asyncio
async def test_generate_llm_exception(generator: AnswerGenerator, config: SearchConfig, mock_openai_client):
    """Test error handling when the LLM API call fails."""
    chunks = [{"id": "1", config.content_field: "Context."}]

    mock_openai_client.chat.completions.create.side_effect = Exception("API Failure")

    answer = await generator.generate("query", chunks)

    # Verify the specific error message returned by the component
    assert answer == "An error occurred while generating the answer."

@pytest.mark.asyncio
async def test_generator_close(generator: AnswerGenerator, mock_openai_client):
    """Test that the close method closes the underlying LLM client."""
    await generator.close()
    mock_openai_client.close.assert_awaited_once()