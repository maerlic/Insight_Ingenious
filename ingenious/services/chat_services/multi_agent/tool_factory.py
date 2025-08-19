"""
Tool factory module for multi-agent conversation flows.
Contains utility functions for AI search, SQL operations, and other tools.
"""

import json
from typing import Any, List, Protocol, Tuple


class SearchToolProtocol(Protocol):
    """Protocol for search tool functions."""

    async def aisearch(self, search_query: str, index_name: str = "default") -> str:
        """Perform AI search."""
        ...


class SQLToolProtocol(Protocol):
    """Protocol for SQL tool functions."""

    def get_db_attr(self, config_obj: Any) -> Tuple[str, List[str]]:
        """Get database attributes."""
        ...

    async def execute_sql_local(self, sql_query: str) -> str:
        """Execute SQL query locally."""
        ...


class ToolFunctions:
    """Tool functions for knowledge base and search operations."""

    @staticmethod
    async def aisearch(search_query: str, index_name: str = "default") -> str:
        """
        Perform AI search using Azure Search service.

        Args:
            search_query: The search query string
            index_name: Name of the search index to query

        Returns:
            JSON string with search results
        """
        try:
            # For testing with mock services, return a mock response
            mock_results = {
                "value": [
                    {
                        "@search.score": 0.95,
                        "content": f"Mock search result for query: '{search_query}' in index '{index_name}'. This is a simulated search result for testing purposes.",
                        "title": f"Mock Document {search_query[:20]}",
                        "url": f"https://mock-docs.com/search/{search_query.replace(' ', '-')}",
                    }
                ],
                "@odata.count": 1,
            }
            return json.dumps(mock_results)
        except Exception as e:
            return json.dumps({"error": f"Search failed: {str(e)}", "value": []})


class SQL_ToolFunctions:
    """Tool functions for SQL database operations."""

    @staticmethod
    def get_db_attr(config_obj: Any) -> Tuple[str, List[str]]:
        """
        Get database table attributes for local SQLite database.

        Args:
            config_obj: Configuration object

        Returns:
            Tuple of (table_name, column_names)
        """
        try:
            # For testing, return mock table structure
            table_name = "test_table"
            column_names = ["id", "name", "value", "created_at"]
            return table_name, column_names
        except Exception:
            return "unknown_table", ["id", "data"]

    @staticmethod
    def get_azure_db_attr(config_obj: Any) -> Tuple[str, str, List[str]]:
        """
        Get database attributes for Azure SQL database.

        Args:
            config_obj: Configuration object

        Returns:
            Tuple of (database_name, table_name, column_names)
        """
        try:
            # For testing, return mock Azure SQL structure
            database_name = "test_database"
            table_name = "test_table"
            column_names = ["id", "name", "value", "created_at"]
            return database_name, table_name, column_names
        except Exception:
            return "unknown_db", "unknown_table", ["id", "data"]

    @staticmethod
    async def execute_sql_local(sql_query: str) -> str:
        """
        Execute SQL query on local SQLite database.

        Args:
            sql_query: SQL query string

        Returns:
            JSON string with query results
        """
        try:
            # For testing, return mock SQL results
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
        Execute SQL query on Azure SQL database.

        Args:
            sql_query: SQL query string

        Returns:
            JSON string with query results
        """
        try:
            # For testing, return mock Azure SQL results
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
