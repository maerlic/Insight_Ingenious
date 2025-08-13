import logging
import urllib.parse
from types import SimpleNamespace
from typing import Any, Optional, Protocol, Tuple, runtime_checkable

from pydantic import SecretStr

from ingenious.config import IngeniousSettings
from ingenious.services.azure_search import SearchConfig

log = logging.getLogger("ingenious.services.azure_search.builders")

# Constants
EMBEDDING_ROLES = frozenset(["embedding"])
EMBEDDING_NAME_PATTERNS = frozenset(["embedding", "embed"])
CHAT_ROLES = frozenset(["chat", "completion", "generation"])
CHAT_NAME_PATTERNS = frozenset(["gpt", "4o"])
DEFAULT_API_VERSION = "2024-02-15-preview"
DEFAULT_SEMANTIC_CONFIG = "default"
DEFAULT_TOP_K_RETRIEVAL = 20
DEFAULT_TOP_N_FINAL = 5
DEFAULT_ID_FIELD = "id"
DEFAULT_CONTENT_FIELD = "content"
DEFAULT_VECTOR_FIELD = "vector"


class ConfigError(ValueError):
    """User-actionable configuration error."""

    pass


@runtime_checkable
class ModelConfig(Protocol):
    """Type protocol for model configuration objects."""

    role: Optional[str]
    model: Optional[str]
    deployment: Optional[str]
    endpoint: Optional[str]
    base_url: Optional[str]
    key: Optional[Any]  # Can be str or SecretStr
    api_key: Optional[Any]  # Can be str or SecretStr
    api_version: Optional[str]


@runtime_checkable
class AzureSearchService(Protocol):
    """Type protocol for Azure Search service configuration."""

    endpoint: Optional[str]
    key: Optional[Any]  # Can be str or SecretStr
    api_key: Optional[Any]  # Can be str or SecretStr
    index_name: Optional[str]
    use_semantic_ranking: Optional[bool]
    semantic_ranking: Optional[bool]
    semantic_configuration: Optional[str]
    semantic_configuration_name: Optional[str]
    top_k_retrieval: Optional[int]
    top_n_final: Optional[int]
    id_field: Optional[str]
    content_field: Optional[str]
    vector_field: Optional[str]


# -------------------- Validation helpers --------------------


def _validate_endpoint(endpoint: str, name: str) -> str:
    """Validate endpoint URL format."""
    endpoint = endpoint.strip()
    if not endpoint:
        raise ConfigError(f"{name} cannot be empty")

    try:
        result = urllib.parse.urlparse(endpoint)
        if not all([result.scheme, result.netloc]):
            raise ConfigError(f"{name} must be a valid URL with scheme and host")
        if result.scheme not in ["http", "https"]:
            raise ConfigError(f"{name} must use http or https scheme")
    except Exception as e:
        raise ConfigError(f"Invalid {name}: {e}")

    return endpoint


def _first_non_empty(*vals: Optional[str]) -> Optional[str]:
    """Return the first string that is not None/empty/whitespace."""
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _get(obj: Any, *names: str) -> Optional[Any]:
    """Return the first existing attribute value by name (alias-friendly)."""
    for n in names:
        val = getattr(obj, n, None)
        if val is not None:
            return val
    return None


def _ensure_nonempty(value: Optional[str], field_name: str) -> str:
    """Raise ConfigError if missing/empty."""
    s = _first_non_empty(value)
    if not s:
        raise ConfigError(f"{field_name} is required and was not provided.")
    return s


def _extract_secret_value(value: Optional[str | SecretStr]) -> Optional[str]:
    """Extract string value from SecretStr or similar objects."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if hasattr(value, "get_secret_value"):
        try:
            return value.get_secret_value()
        except (AttributeError, TypeError) as e:
            log.debug(f"Failed to extract secret value: {e}")
    return None


# -------------------- Model configuration helpers --------------------
def _model_endpoint(model: ModelConfig) -> Optional[str]:
    """Allow `endpoint` or `base_url` on model entries."""
    return _first_non_empty(
        getattr(model, "endpoint", None), getattr(model, "base_url", None)
    )


def _model_key(model: ModelConfig) -> Optional[str]:
    """Allow `key` or `api_key` on model entries; accept SecretStr from settings."""
    key = _get(model, "key", "api_key")
    return _extract_secret_value(key)


def _is_embedding_model(model: ModelConfig) -> bool:
    """Check if a model is configured for embeddings."""
    role = (getattr(model, "role", "") or "").lower()
    if role in EMBEDDING_ROLES:
        return True
    name = (getattr(model, "model", "") or "").lower()
    return any(pattern in name for pattern in EMBEDDING_NAME_PATTERNS)


def _is_chat_model(model: ModelConfig) -> bool:
    """Check if a model is configured for chat/generation."""
    role = (getattr(model, "role", "") or "").lower()
    if role in CHAT_ROLES:
        return True
    name = (getattr(model, "model", "") or "").lower()
    return any(pattern in name for pattern in CHAT_NAME_PATTERNS)


# -------------------- Model selection --------------------


def _select_models(models: list[ModelConfig]) -> Tuple[ModelConfig, ModelConfig]:
    """
    Select embedding and chat models from the list.

    Returns:
        Tuple of (embedding_config, chat_config)

    Raises:
        ConfigError: If models cannot be properly identified
    """
    emb_cfg = next((m for m in models if _is_embedding_model(m)), None)
    chat_cfg = next((m for m in models if _is_chat_model(m)), None)

    if emb_cfg and chat_cfg:
        return emb_cfg, chat_cfg

    # Handle single model case
    if len(models) == 1:
        log.warning(
            "Single ModelSettings provided; reusing credentials for both roles. "
            "Note: On Azure OpenAI you must configure TWO deployments "
            "(one embedding, one chat). Using one deployment will fail."
        )
        if emb_cfg:
            return emb_cfg, emb_cfg
        if chat_cfg:
            return chat_cfg, chat_cfg
        raise ConfigError(
            "Unable to identify model type. Please specify 'role' attribute "
            "or use recognizable model names containing 'embedding' or 'gpt'."
        )

    # Multiple models but missing one role
    if not emb_cfg:
        raise ConfigError(
            "No embedding model configured (expected role 'embedding' or model containing 'embedding')."
        )
    if not chat_cfg:
        raise ConfigError(
            "No chat/generation model configured (expected role 'chat' or model containing 'gpt'/'4o')."
        )

    raise ConfigError(
        "Unable to select models: none look like embedding ('embedding') or chat ('gpt'/'4o')."
    )


def _pick_models(settings: IngeniousSettings) -> Tuple[str, str, str, str, str]:
    """
    Extract and validate model configurations from settings.

    Returns:
        Tuple of (openai_endpoint, openai_key, openai_version,
                  embedding_deployment, generation_deployment)

    Raises:
        ConfigError: If configuration is invalid or incomplete
    """
    models = getattr(settings, "models", None) or []
    if not models:
        raise ConfigError("No models configured (IngeniousSettings.models is empty).")

    emb_cfg, chat_cfg = _select_models(models)

    # Extract deployment names (required for Azure)
    emb_dep = _ensure_nonempty(
        getattr(emb_cfg, "deployment", None), "Embedding deployment"
    )
    gen_dep = _ensure_nonempty(
        getattr(chat_cfg, "deployment", None), "Generation deployment"
    )

    # Extract and validate endpoint
    endpoint = _first_non_empty(_model_endpoint(chat_cfg), _model_endpoint(emb_cfg))
    endpoint = _ensure_nonempty(endpoint, "OpenAI endpoint")
    endpoint = _validate_endpoint(endpoint, "OpenAI endpoint")

    # Extract API key
    key = _first_non_empty(_model_key(chat_cfg), _model_key(emb_cfg))
    key = _ensure_nonempty(key, "OpenAI API key")

    # Extract API version with fallback
    version_candidate = _first_non_empty(
        getattr(chat_cfg, "api_version", None),
        getattr(emb_cfg, "api_version", None),
        DEFAULT_API_VERSION,
    )
    version = _ensure_nonempty(version_candidate, "OpenAI API version")

    return endpoint, key, version, emb_dep, gen_dep


# -------------------- Azure Search configuration --------------------
def _extract_search_config(svc: AzureSearchService) -> dict[str, Any]:
    """Extract and validate Azure Search configuration."""
    # Extract and validate endpoint
    search_endpoint = _ensure_nonempty(_get(svc, "endpoint"), "Azure Search endpoint")
    search_endpoint = _validate_endpoint(search_endpoint, "Azure Search endpoint")

    # Extract API key
    raw_key = _extract_secret_value(_get(svc, "key", "api_key"))
    search_key = _ensure_nonempty(raw_key, "Azure Search key")

    # Extract index name
    index_name = _ensure_nonempty(_get(svc, "index_name"), "Azure Search index_name")

    # Extract semantic ranking settings
    use_semantic_ranking = _get(svc, "use_semantic_ranking")
    if use_semantic_ranking is None:
        use_semantic_ranking = bool(_get(svc, "semantic_ranking") or False)

    semantic_configuration_name = _first_non_empty(
        _get(svc, "semantic_configuration"),
        _get(svc, "semantic_configuration_name"),
        DEFAULT_SEMANTIC_CONFIG,
    )

    # Extract optional parameters with defaults and validate
    top_k_retrieval = getattr(svc, "top_k_retrieval", DEFAULT_TOP_K_RETRIEVAL)
    top_n_final = getattr(svc, "top_n_final", DEFAULT_TOP_N_FINAL)

    # Validate that top_k and top_n are positive
    if top_k_retrieval <= 0:
        raise ValueError(f"top_k_retrieval must be positive, got {top_k_retrieval}")
    if top_n_final <= 0:
        raise ValueError(f"top_n_final must be positive, got {top_n_final}")

    return {
        "search_endpoint": search_endpoint,
        "search_key": SecretStr(search_key),
        "search_index_name": index_name,
        "use_semantic_ranking": bool(use_semantic_ranking),
        "semantic_configuration_name": semantic_configuration_name,
        "top_k_retrieval": top_k_retrieval,
        "top_n_final": top_n_final,
        "id_field": getattr(svc, "id_field", DEFAULT_ID_FIELD),
        "content_field": getattr(svc, "content_field", DEFAULT_CONTENT_FIELD),
        "vector_field": getattr(svc, "vector_field", DEFAULT_VECTOR_FIELD),
    }


# -------------------- OpenAI property helper --------------------


def _ensure_openai_property_on_config_class() -> None:
    """
    Add a backward-compatible 'openai' property to SearchConfig if not already present.
    This maintains compatibility with existing tests and code that expect cfg.openai.
    """
    if hasattr(SearchConfig, "openai"):
        return

    def _openai_property(self: "SearchConfig") -> SimpleNamespace:
        """Backward compatibility property for OpenAI configuration access."""
        # First, figure out the correct value for key_val
        if isinstance(self.openai_key, SecretStr):
            key_val = self.openai_key.get_secret_value()
        else:
            key_val = self.openai_key

        # Then, use it in a single, final return statement
        return SimpleNamespace(
            endpoint=self.openai_endpoint,
            key=key_val,
            version=self.openai_version,
            embedding_deployment_name=self.embedding_deployment_name,
            generation_deployment_name=self.generation_deployment_name,
        )

    try:
        # Add property to the class
        SearchConfig.openai = property(_openai_property)  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        # If SearchConfig is immutable or doesn't allow attribute injection,
        # log a warning but continue
        log.warning(
            "Unable to add 'openai' property to SearchConfig for backward compatibility"
        )


# -------------------- Main builder function --------------------


def build_search_config_from_settings(settings: IngeniousSettings) -> SearchConfig:
    """
    Build a SearchConfig from settings with validation and alias handling.

    Enforces that embedding and chat must use different Azure OpenAI deployments
    to prevent a common misconfiguration.

    Args:
        settings: Application settings containing Azure Search and model configurations

    Returns:
        Configured SearchConfig instance

    Raises:
        ConfigError: If configuration is invalid or incomplete
    """
    # Validate Azure Search services configuration
    services = getattr(settings, "azure_search_services", None) or []
    if not services or not services[0]:
        raise ConfigError(
            "Azure Search is not configured (azure_search_services[0] missing)."
        )

    # Extract Azure Search configuration
    search_config = _extract_search_config(services[0])

    # Extract and validate model configurations
    openai_endpoint, openai_key, openai_version, emb_dep, gen_dep = _pick_models(
        settings
    )

    # Enforce distinct deployments for embedding and chat
    if emb_dep == gen_dep:
        raise ConfigError(
            "Embedding and chat deployments must not be the same. "
            "Configure distinct Azure OpenAI deployments for embeddings and chat."
        )

    # Ensure backward compatibility
    _ensure_openai_property_on_config_class()

    # Build final configuration
    return SearchConfig(
        # Azure Search settings
        **search_config,
        # OpenAI / Azure OpenAI settings
        openai_endpoint=openai_endpoint,
        openai_key=SecretStr(openai_key),
        openai_version=openai_version,
        embedding_deployment_name=emb_dep,
        generation_deployment_name=gen_dep,
    )
