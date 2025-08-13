from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from openai import AsyncAzureOpenAI

from .config import SearchConfig


def make_search_client(cfg: SearchConfig) -> SearchClient:
    return SearchClient(
        endpoint=cfg.search_endpoint,
        index_name=cfg.search_index_name,
        credential=AzureKeyCredential(cfg.search_key.get_secret_value()),
        # Removed retry_policy - use Azure SDK defaults
    )


def make_async_openai_client(cfg: SearchConfig) -> AsyncAzureOpenAI:
    # openai-py has its own retry; we set a small max_retries
    return AsyncAzureOpenAI(
        azure_endpoint=cfg.openai_endpoint,
        api_key=cfg.openai_key.get_secret_value(),
        api_version=cfg.openai_version,
        max_retries=3,
    )
