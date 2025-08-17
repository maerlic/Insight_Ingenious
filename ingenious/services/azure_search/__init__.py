# ingenious/services/azure_search/__init__.py

from typing import TYPE_CHECKING, Any

from ingenious.services.retrieval.errors import GenerationDisabledError  # noqa: F401

# Export the light model directly – safe to import anytime
from .config import SearchConfig  # noqa: F401


# Add type hints to the function signature
def build_search_pipeline(*args: Any, **kwargs: Any) -> "AdvancedSearchPipeline":
    """
    Lazy proxy so importing this package does NOT pull Azure SDKs.
    The real import happens only when the function is actually called.
    """
    from .components.pipeline import build_search_pipeline as _impl

    return _impl(*args, **kwargs)


if TYPE_CHECKING:
    # Only for type checkers; doesn't run at runtime
    from .components.pipeline import AdvancedSearchPipeline  # noqa: F401

__all__ = [
    "SearchConfig",
    "build_search_pipeline",
    "AdvancedSearchPipeline",
    "GenerationDisabledError",
]
