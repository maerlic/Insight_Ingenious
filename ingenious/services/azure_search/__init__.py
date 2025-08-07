# Insight_Ingenious/ingenious/services/azure_search/__init__.py

# Expose only the primary public symbols for the service facade

# We use explicit imports (relative or absolute depending on package installation status)
try:
    from ingenious.services.azure_search.config import SearchConfig
    from ingenious.services.azure_search.pipeline import build_search_pipeline, AdvancedSearchPipeline
except ImportError:
    from .config import SearchConfig
    from .pipeline import build_search_pipeline, AdvancedSearchPipeline


__all__ = [
    "SearchConfig", 
    "build_search_pipeline",
    "AdvancedSearchPipeline" # Included for type hinting in consuming services
]