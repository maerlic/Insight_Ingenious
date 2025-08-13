"""
Tool factory module for multi-agent conversation flows.
Contains utility functions for AI search, SQL operations, and other tools.

Notes:
- `ToolFunctions.aisearch` prefers the Azure provider but supports a deterministic
  mock fallback via the `SEARCH_USE_MOCK` env var (true/yes/on/1) or on provider errors.
- SQL helpers return **mock data by default** for testing (see `SQL_USE_MOCK` below).
  Set `SQL_USE_MOCK=false` to disable mocks and surface explicit errors in production.
- The `index_name` parameter is attempted to be passed through to the provider. If the
  provider does not support it, the value is ignored and a DeprecationWarning may be issued.
"""

import json
import os
import warnings
from typing import Any, List, Protocol, Tuple


class SearchToolProtocol(Protocol):
    """Protocol for search tool functions."""

    async def aisearch(self, search_query: str, index_name: str = "default") -> str:
        """
        Perform AI search against the knowledge base.

        Parameters:
            search_query (str): The natural-language query.
            index_name (str): Logical index to target. The wrapper will attempt to pass this
                through to the provider (`index_name` or `index`). If unsupported by the provider,
                it will be ignored; a DeprecationWarning may be issued for non-default values.

        Returns:
            str: A JSON string with shape:
                {
                  "value": [
                    {
                      "@search.score": <float|None>,
                      "content": <str>,
                      "title": <str>,
                      "url": <str|None>
                    },
                    ...
                  ],
                  "@odata.count": <int>
                }
        """
        ...


class SQLToolProtocol(Protocol):
    """Protocol for SQL tool functions."""

    def get_db_attr(self, config_obj: Any) -> Tuple[str, List[str]]:
        """
        Get table attributes for a local SQLite database.

        Parameters:
            config_obj (Any): Configuration object (implementation-specific).

        Returns:
            (str, List[str]): (table_name, column_names).
        """
        ...

    async def execute_sql_local(self, sql_query: str) -> str:
        """
        Execute a SQL query against a local SQLite database.

        Parameters:
            sql_query (str): The SQL query to execute.

        Returns:
            str: JSON string containing:
                 {
                   "results": <List[dict]>,
                   "row_count": <int>,
                   "query": <str>
                 }
        """
        ...


class ToolFunctions:
    """Tool functions for knowledge base and search operations."""

    @staticmethod
    def _should_use_mock() -> bool:
        """
        Determine whether to use the mock search path.

        Controlled via env var `SEARCH_USE_MOCK` in {true, yes, on, 1}.

        Returns:
            bool: True when the mock should be used.
        """
        flag = os.getenv("SEARCH_USE_MOCK", "").strip().lower()
        return flag in {"1", "true", "yes", "on"}

    @staticmethod
    def _mock_search_response(search_query: str, index_name: str = "default") -> str:
        """
        Deterministic mock response for offline/dev/CI use.

        Parameters:
            search_query (str): The query string.
            index_name (str): Logical index name (reflected in the mock content).

        Returns:
            str: JSON string with one stable, predictable result.
        """
        # For testing: return a deterministic mock search result.
        mock_results = {
            "value": [
                {
                    "@search.score": 0.95,
                    "content": (
                        f"Mock search result for query: '{search_query}' in index "
                        f"'{index_name}'. This is a simulated search result for testing purposes."
                    ),
                    "title": f"Mock Document {search_query[:20]}",
                    "url": f"https://mock-docs.com/search/{search_query.replace(' ', '-')}",
                }
            ],
            "@odata.count": 1,
        }
        return json.dumps(mock_results)

    @staticmethod
    def _extract_url(chunk: dict) -> Any:
        """
        Best-effort URL extraction from a provider chunk. Returns None if not present.

        Checks common fields and nested metadata without assuming a specific schema.

        Parameters:
            chunk (dict): A single provider result chunk.

        Returns:
            Any: URL string if available, else None.
        """
        possible_keys = [
            "url",
            "source_url",
            "source",
            "uri",
            "document_url",
            "web_url",
            "link",
            "path",
        ]
        for key in possible_keys:
            if key in chunk and chunk.get(key):
                return chunk.get(key)

        meta = chunk.get("metadata") or chunk.get("_metadata") or {}
        if isinstance(meta, dict):
            for key in possible_keys:
                if key in meta and meta.get(key):
                    return meta.get(key)
        return None

    @staticmethod
    async def aisearch(search_query: str, index_name: str = "default") -> str:
        """
        Perform AI search via the unified Azure Search provider.

        Behavior:
            - Uses AzureSearchProvider when available.
            - Falls back to a deterministic mock when `SEARCH_USE_MOCK` is set or when
              provider import/config/retrieval fails.
            - Attempts to pass `index_name` to the provider. If unsupported, it is ignored and a
              DeprecationWarning may be issued for non-default values. This parameter may be
              deprecated in a future release.

        Parameters:
            search_query (str): The natural-language query.
            index_name (str): Logical index to target (provider-dependent).

        Returns:
            str: JSON string with keys:
                 - "value": list of objects each containing "@search.score", "content",
                   "title", and "url" (url may be None).
                 - "@odata.count": integer count of returned items.
        """
        # Feature-flagged mock (explicit)
        if ToolFunctions._should_use_mock():
            return ToolFunctions._mock_search_response(search_query, index_name)

        # Provider path with graceful degrade to mock on any import/runtime error
        try:
            from ingenious.config import get_config
            from ingenious.services.azure_search.provider import AzureSearchProvider

            settings = get_config()
            provider = AzureSearchProvider(settings)
            try:
                # Try to pass index_name through; if provider doesn't support it, fall back.
                try:
                    chunks = await provider.retrieve(
                        search_query, top_k=5, index_name=index_name
                    )
                except TypeError:
                    try:
                        # Some providers may use `index` instead.
                        chunks = await provider.retrieve(
                            search_query, top_k=5, index=index_name
                        )
                    except TypeError:
                        # Provider does not support index selection.
                        chunks = await provider.retrieve(search_query, top_k=5)
                        if index_name and index_name != "default":
                            warnings.warn(
                                "`index_name` is not supported by the current search provider and will be ignored. "
                                "This parameter may be deprecated in a future release.",
                                category=DeprecationWarning,
                                stacklevel=2,
                            )
            finally:
                await provider.close()

            value = []
            for c in chunks or []:
                value.append(
                    {
                        "@search.score": c.get(
                            "_final_score", c.get("@search.score", None)
                        ),
                        "content": c.get("content", ""),
                        "title": c.get("title", c.get("id", "")),
                        # Reintroduced URL for downstream consumers; may be None if unavailable.
                        "url": ToolFunctions._extract_url(c),
                    }
                )
            return json.dumps({"value": value, "@odata.count": len(value)})
        except Exception:
            # Graceful degrade: return the deterministic mock instead of an error object.
            return ToolFunctions._mock_search_response(search_query, index_name)


class SQL_ToolFunctions:
    """Tool functions for SQL database operations (local SQLite and Azure SQL)."""

    # Warn once when using default mock behavior without explicit env configuration.
    _MOCK_WARNED: bool = False

    @staticmethod
    def _use_mock_sql() -> bool:
        """
        Determine whether SQL helpers should return mock data.

        Controlled via env var `SQL_USE_MOCK` in {true, yes, on, 1}. Defaults to **True**
        for backward compatibility. When unset, a one-time RuntimeWarning is emitted.

        Returns:
            bool: True to use mock SQL paths; False to disable mocks.
        """
        val = os.getenv("SQL_USE_MOCK", None)
        if val is None:
            # Default to mock for testing; warn once to avoid accidental production reliance.
            if not SQL_ToolFunctions._MOCK_WARNED:
                warnings.warn(
                    "SQL helpers are returning mock data by default. "
                    "Set SQL_USE_MOCK=false to disable mocks in production.",
                    category=RuntimeWarning,
                    stacklevel=2,
                )
                SQL_ToolFunctions._MOCK_WARNED = True
            return True
        return val.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def get_db_attr(config_obj: Any) -> Tuple[str, List[str]]:
        """
        Get database table attributes for a local SQLite database.

        For testing: returns a mock table structure when `SQL_USE_MOCK` is enabled
        (default). If mocks are disabled, this function raises to avoid accidental
        reliance on test data.

        Parameters:
            config_obj (Any): Configuration object (implementation-specific).

        Returns:
            (str, List[str]): (table_name, column_names).

        Raises:
            RuntimeError: If `SQL_USE_MOCK=false` (real implementation not provided).
        """
        # Gate mocks to avoid accidental production reliance.
        if not SQL_ToolFunctions._use_mock_sql():
            raise RuntimeError(
                "SQL_USE_MOCK=false: real database attribute retrieval is not implemented "
                "in SQL_ToolFunctions.get_db_attr()."
            )
        try:
            # For testing: return mock table structure.
            table_name = "test_table"
            column_names = ["id", "name", "value", "created_at"]
            return table_name, column_names
        except Exception:
            return "unknown_table", ["id", "data"]

    @staticmethod
    def get_azure_db_attr(config_obj: Any) -> Tuple[str, str, List[str]]:
        """
        Get database attributes for an Azure SQL database.

        For testing: returns a mock database/table/columns triple when `SQL_USE_MOCK`
        is enabled (default). If mocks are disabled, this function raises to avoid
        accidental reliance on test data.

        Parameters:
            config_obj (Any): Configuration object (implementation-specific).

        Returns:
            (str, str, List[str]): (database_name, table_name, column_names).

        Raises:
            RuntimeError: If `SQL_USE_MOCK=false` (real implementation not provided).
        """
        # Gate mocks to avoid accidental production reliance.
        if not SQL_ToolFunctions._use_mock_sql():
            raise RuntimeError(
                "SQL_USE_MOCK=false: real Azure DB attribute retrieval is not implemented "
                "in SQL_ToolFunctions.get_azure_db_attr()."
            )
        try:
            # For testing: return mock Azure SQL structure.
            database_name = "test_database"
            table_name = "test_table"
            column_names = ["id", "name", "value", "created_at"]
            return database_name, table_name, column_names
        except Exception:
            return "unknown_db", "unknown_table", ["id", "data"]

    @staticmethod
    async def execute_sql_local(sql_query: str) -> str:
        """
        Execute a SQL query on a local SQLite database.

        For testing: returns a mock result set when `SQL_USE_MOCK` is enabled (default).
        If mocks are disabled, returns an explicit error payload instead of mock data.

        Parameters:
            sql_query (str): The SQL query to execute.

        Returns:
            str: JSON string containing:
                 {
                   "results": <List[dict]>,
                   "row_count": <int>,
                   "query": <str>
                 }
        """
        # Gate mocks to avoid accidental production reliance.
        if not SQL_ToolFunctions._use_mock_sql():
            warnings.warn(
                "SQL_USE_MOCK=false: SQL mocks disabled; returning error payload.",
                category=RuntimeWarning,
                stacklevel=2,
            )
            return json.dumps(
                {
                    "error": "SQL_USE_MOCK=false: real local SQL execution not implemented.",
                    "results": [],
                    "row_count": 0,
                    "query": sql_query,
                }
            )
        try:
            # For testing: return mock SQL results.
            mock_results = [
                {
                    "id": 1,
                    "name": "Test Item 1",
                    "value": 100,
                    "created_at": "2024-01-01",
                },
                {
                    "id": 2,
                    "name": "Test Item 2",
                    "value": 200,
                    "created_at": "2024-01-02",
                },
            ]
            return json.dumps(
                {
                    "results": mock_results,
                    "row_count": len(mock_results),
                    "query": sql_query,
                }
            )
        except Exception as e:
            return json.dumps(
                {"error": f"SQL execution failed: {str(e)}", "results": []}
            )

    @staticmethod
    async def execute_sql_azure(sql_query: str) -> str:
        """
        Execute a SQL query on an Azure SQL database.

        For testing: returns a mock result set when `SQL_USE_MOCK` is enabled (default).
        If mocks are disabled, returns an explicit error payload instead of mock data.

        Parameters:
            sql_query (str): The SQL query to execute.

        Returns:
            str: JSON string containing:
                 {
                   "results": <List[dict]>,
                   "row_count": <int>,
                   "query": <str>,
                   "source": "azure_sql"
                 }
        """
        # Gate mocks to avoid accidental production reliance.
        if not SQL_ToolFunctions._use_mock_sql():
            warnings.warn(
                "SQL_USE_MOCK=false: SQL mocks disabled; returning error payload.",
                category=RuntimeWarning,
                stacklevel=2,
            )
            return json.dumps(
                {
                    "error": "SQL_USE_MOCK=false: real Azure SQL execution not implemented.",
                    "results": [],
                    "row_count": 0,
                    "query": sql_query,
                    "source": "azure_sql",
                }
            )
        try:
            # For testing: return mock Azure SQL results.
            mock_results = [
                {
                    "id": 1,
                    "name": "Azure Test Item 1",
                    "value": 150,
                    "created_at": "2024-01-01",
                },
                {
                    "id": 2,
                    "name": "Azure Test Item 2",
                    "value": 250,
                    "created_at": "2024-01-02",
                },
            ]
            return json.dumps(
                {
                    "results": mock_results,
                    "row_count": len(mock_results),
                    "query": sql_query,
                    "source": "azure_sql",
                }
            )
        except Exception as e:
            return json.dumps(
                {"error": f"Azure SQL execution failed: {str(e)}", "results": []}
            )
