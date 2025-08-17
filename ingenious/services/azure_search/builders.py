"""Builds a validated SearchConfig from application settings.

This module centralizes the complex logic of interpreting user settings for
Azure Search and Azure OpenAI services. It is responsible for parsing configurations,
handling various field name aliases (e.g., `key` vs. `api_key`), applying
sensible defaults, and validating critical inputs like endpoints. Its primary
goal is to produce a valid `SearchConfig` object that guarantees distinct Azure
OpenAI deployments for embedding and chat tasks, preventing common runtime errors.

The main entry point is `build_search_config_from_settings`.
"""

from __future__ import annotations

import logging
import urllib.parse
from types import SimpleNamespace
from typing import Any, Optional, Protocol, Tuple, cast, runtime_checkable

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
    """Type protocol for model configuration objects.

    This defines the expected shape of a configuration object for a single
    Azure OpenAI model (either embedding or chat), allowing for static analysis
    without requiring a specific class implementation.
    """

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
    """Type protocol for Azure Search service configuration.

    This defines the expected shape of the Azure Search service configuration,
    allowing for flexible, duck-typed configuration sources while ensuring
    all necessary attributes are checkable.
    """

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
    """Validate that an endpoint string is a well-formed http/https URL.

    This prevents runtime errors from malformed URLs by checking for a scheme
    and network location before the endpoint is used in a network request.

    Args:
        endpoint: The URL string to validate.
        name: A human-readable name for the endpoint for use in error messages.

    Returns:
        The validated and stripped endpoint URL.

    Raises:
        ConfigError: If the endpoint is empty or not a valid URL.
    """
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
    """Return the first string that is not None, empty, or just whitespace.

    This is a utility for gracefully handling configuration aliases, allowing
    the system to check multiple potential sources for a value and picking the
    first one that is validly provided.
    """
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _get(obj: Any, *names: str) -> Optional[Any]:
    """Return the first existing attribute value by name from an object.

    This function provides a safe way to access an attribute using multiple
    possible names (aliases), returning the first one found. This is useful for
    making configuration more flexible (e.g., accepting `key` or `api_key`).
    """
    for n in names:
        val: Optional[Any] = getattr(obj, n, None)
        if val is not None:
            return val
    return None


def _ensure_nonempty(value: Optional[str], field_name: str) -> str:
    """Raise a ConfigError if a string value is missing or empty.

    This helper enforces that a required configuration field has been provided
    with a non-whitespace value, improving configuration robustness.
    """
    s = _first_non_empty(value)
    if not s:
        raise ConfigError(f"{field_name} is required and was not provided.")
    return s


def _extract_secret_value(value: Optional[str | SecretStr]) -> Optional[str]:
    """Extract the string value from a SecretStr or return a string directly.

    This function centralizes the logic for handling values that might be
    wrapped in Pydantic's `SecretStr` for security. It safely unwraps the
    secret or returns the original value if it's already a string.
    """
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
    """Get the model endpoint, trying both `endpoint` and `base_url` attributes.

    This exists to provide user flexibility, as different libraries and conventions
    use different attribute names for the same concept (an API base URL).
    """
    return _first_non_empty(
        getattr(model, "endpoint", None), getattr(model, "base_url", None)
    )


def _model_key(model: ModelConfig) -> Optional[str]:
    """Get the model API key, trying `key` and `api_key` and handling SecretStr.

    This function allows users to specify an API key using common aliases
    and ensures that if a Pydantic `SecretStr` is provided (e.g., from settings),
    its underlying string value is correctly extracted.
    """
    key: Optional[str | SecretStr] = _get(model, "key", "api_key")
    return _extract_secret_value(key)


def _is_embedding_model(model: ModelConfig) -> bool:
    """Determine if a model is intended for embedding.

    This function identifies an embedding model by checking its assigned 'role'
    or by looking for common embedding-related patterns in its name. This is
    critical for selecting the correct deployment for embedding tasks.
    """
    role: str = (getattr(model, "role", "") or "").lower()
    if role in EMBEDDING_ROLES:
        return True
    name: str = (getattr(model, "model", "") or "").lower()
    return any(pattern in name for pattern in EMBEDDING_NAME_PATTERNS)


def _is_chat_model(model: ModelConfig) -> bool:
    """Determine if a model is intended for chat/completion.

    This function identifies a chat/generation model by checking its 'role'
    or by looking for common patterns in its name (e.g., 'gpt', '4o'). This is
    critical for selecting the correct deployment for generation tasks.
    """
    role: str = (getattr(model, "role", "") or "").lower()
    if role in CHAT_ROLES:
        return True
    name: str = (getattr(model, "model", "") or "").lower()
    return any(pattern in name for pattern in CHAT_NAME_PATTERNS)


# -------------------- Model selection --------------------


def _select_models(models: list[ModelConfig]) -> Tuple[ModelConfig, ModelConfig]:
    """Select and return the embedding and chat models from a list.

    This function implements the core logic for identifying the two required
    model roles (embedding and chat) from the user's configuration. It handles
    the simple two-model case as well as the special case where a single model
    configuration is provided, which is then used for both roles.

    Returns:
        A tuple containing (embedding_config, chat_config).

    Raises:
        ConfigError: If the models cannot be properly identified or if one of
            the required roles is missing from the configuration.
    """
    emb_cfg: Optional[ModelConfig] = next(
        (m for m in models if _is_embedding_model(m)), None
    )
    chat_cfg: Optional[ModelConfig] = next(
        (m for m in models if _is_chat_model(m)), None
    )

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

    # This case should be unreachable given the logic above, but serves as a safeguard.
    raise ConfigError(
        "Unable to select models: none look like embedding ('embedding') or chat ('gpt'/'4o')."
    )


def _pick_models(settings: IngeniousSettings) -> Tuple[str, str, str, str, str]:
    """Extract and validate all required model configurations from settings.

    This function orchestrates the process of selecting the correct models,
    extracting their essential properties (endpoint, key, deployments),
    and validating that all required values are present and correctly formatted.

    Returns:
        A tuple of (openai_endpoint, openai_key, openai_version,
                 embedding_deployment, generation_deployment).

    Raises:
        ConfigError: If any part of the model configuration is invalid or
            incomplete.
    """
    models: list[ModelConfig] = getattr(settings, "models", None) or []
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
    endpoint_candidate = _first_non_empty(
        _model_endpoint(chat_cfg), _model_endpoint(emb_cfg)
    )
    endpoint = _ensure_nonempty(endpoint_candidate, "OpenAI endpoint")
    endpoint = _validate_endpoint(endpoint, "OpenAI endpoint")

    # Extract API key
    key_candidate = _first_non_empty(_model_key(chat_cfg), _model_key(emb_cfg))
    key = _ensure_nonempty(key_candidate, "OpenAI API key")

    # Extract API version with fallback
    version_candidate = _first_non_empty(
        getattr(chat_cfg, "api_version", None),
        getattr(emb_cfg, "api_version", None),
        DEFAULT_API_VERSION,
    )
    # The default value ensures this is always a string.
    version = cast(str, version_candidate)

    return endpoint, key, version, emb_dep, gen_dep


# -------------------- Azure Search configuration --------------------
def _extract_search_config(svc: AzureSearchService) -> dict[str, Any]:
    """Extract and validate Azure Search configuration from a service object.

    This function gathers all Azure Search-specific settings, applies defaults
    for optional parameters, validates their values, and returns them in a
    dictionary ready to be passed to the `SearchConfig` constructor.
    """
    # Extract and validate endpoint
    search_endpoint_candidate: Optional[str] = _get(svc, "endpoint")
    search_endpoint = _ensure_nonempty(
        search_endpoint_candidate, "Azure Search endpoint"
    )
    search_endpoint = _validate_endpoint(search_endpoint, "Azure Search endpoint")

    # Extract API key
    raw_key = _extract_secret_value(_get(svc, "key", "api_key"))
    search_key = _ensure_nonempty(raw_key, "Azure Search key")

    # Extract index name
    index_name_candidate: Optional[str] = _get(svc, "index_name")
    index_name = _ensure_nonempty(index_name_candidate, "Azure Search index_name")

    # Extract semantic ranking settings
    use_semantic_ranking: bool | None = _get(svc, "use_semantic_ranking")
    if use_semantic_ranking is None:
        use_semantic_ranking = bool(_get(svc, "semantic_ranking") or False)

    semantic_configuration_name_candidate = _first_non_empty(
        _get(svc, "semantic_configuration"),
        _get(svc, "semantic_configuration_name"),
        DEFAULT_SEMANTIC_CONFIG,
    )
    # The default ensures this is always a string.
    semantic_configuration_name = cast(str, semantic_configuration_name_candidate)

    # Extract optional parameters with defaults and validate
    top_k_retrieval: int = getattr(svc, "top_k_retrieval", DEFAULT_TOP_K_RETRIEVAL)
    top_n_final: int = getattr(svc, "top_n_final", DEFAULT_TOP_N_FINAL)

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
    """Add a backward-compatible `openai` property to SearchConfig if needed.

    This function dynamically patches the `SearchConfig` class to add an
    `openai` property. This is done to maintain backward compatibility with
    older code that expects to access OpenAI settings via a nested `cfg.openai`
    object, preventing breaking changes for consumers of the configuration object.
    """
    if hasattr(SearchConfig, "openai"):
        return

    def _openai_property(self: "SearchConfig") -> SimpleNamespace:
        """Provide backward compatibility for accessing OpenAI configuration.

        This property emulates the old nested structure `config.openai` by
        dynamically creating a namespace from the flattened attributes on the
        main `SearchConfig` object.
        """
        key_val: str
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
        # Add property to the class. Mypy disallows this on imported classes.
        SearchConfig.openai = property(_openai_property)  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        # If SearchConfig is immutable or doesn't allow attribute injection,
        # log a warning but continue
        log.warning(
            "Unable to add 'openai' property to SearchConfig for backward compatibility"
        )


# -------------------- Main builder function --------------------


def build_search_config_from_settings(settings: IngeniousSettings) -> SearchConfig:
    """Build a complete and validated SearchConfig from application settings.

    This is the main public entry point for creating a `SearchConfig` instance.
    It orchestrates all validation, alias handling, and default application logic.
    A critical function is enforcing that the embedding and chat models use
    different Azure OpenAI deployments, which is a common and required setup.

    Args:
        settings: The application settings object containing Azure Search and
            model configurations.

    Returns:
        A fully configured and validated SearchConfig instance.

    Raises:
        ConfigError: If the configuration is invalid, incomplete, or violates
            a key constraint (like using the same deployment for two roles).
    """
    # Validate Azure Search services configuration
    services: list[AzureSearchService] = (
        getattr(settings, "azure_search_services", None) or []
    )
    if not services or not services[0]:
        raise ConfigError(
            "Azure Search is not configured (azure_search_services[0] missing)."
        )

    # Extract Azure Search configuration
    search_config: dict[str, Any] = _extract_search_config(services[0])

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

    # Ensure backward compatibility for consumers expecting cfg.openai
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
