"""
Pydantic models for configuration settings.

This module contains all the BaseModel classes that define
the structure and validation for different configuration sections.
"""

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from ingenious.common.enums import AuthenticationMethod


class ChatHistorySettings(BaseModel):
    """Configuration for chat history storage.

    Supports both SQLite (local) and Azure SQL (cloud) storage options.
    For local development, SQLite is recommended. For production, use Azure SQL.
    """

    database_type: str = Field(
        "sqlite",
        description="Type of database: 'sqlite' for local, 'azuresql' for cloud",
    )
    database_path: str = Field(
        "./tmp/high_level_logs.db",
        description="File path for SQLite database (ignored for Azure SQL)",
    )
    database_connection_string: str = Field(
        "", description="Connection string for Azure SQL (leave empty for SQLite)"
    )
    database_name: str = Field(
        "", description="Database name for Azure SQL (ignored for SQLite)"
    )
    memory_path: str = Field(
        "./tmp", description="Path for memory storage and temporary files"
    )


class ModelSettings(BaseModel):
    """Configuration for AI models.

    Defines which AI models to use and how to connect to them.
    Supports Azure OpenAI, OpenAI, and other compatible endpoints.
    """

    model: str = Field(
        ..., description="Model name (e.g., 'gpt-4.1-nano', 'gpt-3.5-turbo')"
    )
    api_type: str = Field("rest", description="API type: 'rest' for HTTP APIs")
    api_version: str = Field(
        "2023-03-15-preview", description="API version for Azure OpenAI"
    )
    deployment: str = Field("", description="Azure OpenAI deployment name (optional)")
    api_key: str = Field("", description="API key for the model service")
    base_url: str = Field("", description="Base URL for the API endpoint")
    client_id: str = Field(
        "", description="Azure client ID for MSI authentication (optional)"
    )
    client_secret: str = Field(
        "",
        description="Azure client secret for CLIENT_ID_AND_SECRET authentication (optional)",
    )
    tenant_id: str = Field(
        "",
        description="Azure tenant ID for CLIENT_ID_AND_SECRET authentication (optional)",
    )
    authentication_method: AuthenticationMethod = Field(
        AuthenticationMethod.DEFAULT_CREDENTIAL,
        description="OpenAI SAS Authentication Method",
    )

    @model_validator(mode="after")
    def _validate_auth_requires_api_key(self) -> "ModelSettings":
        from ingenious.common.enums import AuthenticationMethod

        if (
            self.authentication_method == AuthenticationMethod.TOKEN
            and not self.api_key
        ):
            raise ValueError(
                "API key is required when authentication_method is 'token'. "
                "Set AZURE_OPENAI_API_KEY (or the appropriate key) or supply a value."
            )
        return self

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str, info: ValidationInfo) -> str:
        """Validate that API key is provided when using token authentication."""
        # Get authentication_method from the values being validated
        auth_mode = info.data.get(
            "authentication_method", AuthenticationMethod.DEFAULT_CREDENTIAL
        )

        # Check for placeholder values
        if v and "placeholder" in v.lower():
            raise ValueError(
                "API key is required. Set the appropriate environment variable "
                "(e.g., AZURE_OPENAI_API_KEY) or provide a valid key."
            )

        # If authentication mode is token, api_key is required
        if auth_mode == AuthenticationMethod.TOKEN and not v:
            raise ValueError(
                "API key is required when authentication_method is 'token'. "
                "Set the appropriate environment variable (e.g., AZURE_OPENAI_API_KEY) "
                "or provide a valid key."
            )

        return v

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v: str) -> str:
        """Validate that base URL is provided and properly formatted."""
        if v and "placeholder" in v.lower():
            raise ValueError(
                "Base URL is required. Set the appropriate environment variable "
                "(e.g., AZURE_OPENAI_BASE_URL) or provide a valid URL."
            )
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("Base URL must start with 'http://' or 'https://'")
        return v

    @model_validator(mode="after")
    def validate_client_credentials(self) -> "ModelSettings":
        """Validate client credentials for CLIENT_ID_AND_SECRET authentication."""
        if self.authentication_method == AuthenticationMethod.CLIENT_ID_AND_SECRET:
            if not self.client_id:
                raise ValueError(
                    "client_id is required when authentication_method is 'client_id_and_secret'. "
                    "Set the appropriate environment variable (e.g., AZURE_CLIENT_ID) "
                    "or provide a valid client ID."
                )
            if not self.client_secret:
                raise ValueError(
                    "client_secret is required when authentication_method is 'client_id_and_secret'. "
                    "Set the appropriate environment variable (e.g., AZURE_CLIENT_SECRET) "
                    "or provide a valid client secret."
                )
            # For tenant_id, we allow it to be empty if AZURE_TENANT_ID env var is available
            if not self.tenant_id:
                import os

                if not os.getenv("AZURE_TENANT_ID"):
                    raise ValueError(
                        "tenant_id is required when authentication_method is 'client_id_and_secret'. "
                        "Set the appropriate environment variable (AZURE_TENANT_ID) "
                        "or provide a valid tenant ID in configuration."
                    )
        return self


class ChatServiceSettings(BaseModel):
    """Configuration for the chat service backend.

    Defines how the chat service processes conversations.
    Currently supports multi-agent workflows.
    """

    type: str = Field(
        "multi_agent",
        description="Chat service type: 'multi_agent' for agent workflows",
    )


class ToolServiceSettings(BaseModel):
    """Configuration for external tool integrations.

    Enables integration with external tools and services.
    Disable if only using built-in functionality.
    """

    enable: bool = Field(False, description="Enable external tool service integrations")


class LoggingSettings(BaseModel):
    """Configuration for application logging.

    Controls log levels and output formatting.
    Use 'debug' for development, 'info' or 'warning' for production.
    """

    root_log_level: str = Field(
        "info", description="Root logger level: 'debug', 'info', 'warning', 'error'"
    )
    log_level: str = Field(
        "info",
        description="Application logger level: 'debug', 'info', 'warning', 'error'",
    )

    @field_validator("root_log_level", "log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level values."""
        valid_levels = {"debug", "info", "warning", "error", "critical"}
        if v.lower() not in valid_levels:
            raise ValueError(f"Log level must be one of: {', '.join(valid_levels)}")
        return v.lower()


class AzureSearchSettings(BaseModel):
    """Configuration for Azure Cognitive Search integration.

    Enables document search and retrieval capabilities.
    Optional - leave empty if not using Azure Search.
    """

    service: str = Field("", description="Azure Search service name")
    endpoint: str = Field("", description="Azure Search service endpoint URL")
    key: str = Field("", description="Azure Search service API key")
    index_name: str = Field("", description="Azure Search index name (required)")
    semantic_configuration_name: str | None = Field(
        "default", description="Semantic configuration name for L2 re-ranking"
    )
    # Optional knobs
    use_semantic_ranking: bool = Field(
        True, description="Enable L2 semantic re-ranking"
    )
    top_k_retrieval: int = Field(20, description="K for lexical/vector retrieval")
    top_n_final: int = Field(5, description="N final chunks for RAG")
    id_field: str = Field("id", description="Index id field")
    content_field: str = Field("content", description="Index content field")
    vector_field: str = Field("vector", description="Index vector field")


class AzureSqlSettings(BaseModel):
    """Configuration for Azure SQL integration.

    Enables SQL database operations and queries.
    Optional - leave empty if not using Azure SQL.
    """

    database_name: str = Field("", description="Azure SQL database name")
    table_name: str = Field("", description="Default table name for operations")
    database_connection_string: str = Field(
        "", description="Azure SQL connection string"
    )


class WebAuthenticationSettings(BaseModel):
    """Authentication settings for web interfaces."""

    enable: bool = Field(False, description="Enable web authentication")
    username: str = Field("admin", description="Username for basic authentication")
    password: str = Field("", description="Password for basic authentication")
    type: str = Field(
        "basic", description="Authentication type: 'basic' for HTTP basic auth"
    )


class WebSettings(BaseModel):
    """Configuration for web server and API endpoints.

    Controls how the web service is exposed and secured.
    Adjust IP and port based on deployment requirements.
    """

    ip_address: str = Field(
        "0.0.0.0",
        description="IP address to bind the web server (0.0.0.0 for all interfaces)",
    )
    port: int = Field(80, description="Port number for the web server")
    type: str = Field(
        "fastapi", description="Web framework type: 'fastapi' for FastAPI"
    )
    asynchronous: bool = Field(
        False, description="Enable asynchronous response handling"
    )
    enable_streaming: bool = Field(
        True, description="Enable streaming responses for chat endpoints"
    )
    streaming_chunk_size: int = Field(
        100, description="Maximum characters per streaming chunk"
    )
    streaming_delay_ms: int = Field(
        50, description="Delay between streaming chunks in milliseconds"
    )
    authentication: WebAuthenticationSettings = WebAuthenticationSettings()

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate port number range."""
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v


class LocalSqlSettings(BaseModel):
    """Configuration for local SQLite database operations.

    Used for local development and testing with sample data.
    Points to CSV files and local database paths.
    """

    database_path: str = Field(
        "/tmp/sample_sql_db", description="Path to local SQLite database file"
    )
    sample_csv_path: str = Field(
        "", description="Path to sample CSV files for data loading"
    )
    sample_database_name: str = Field(
        "sample_sql_db", description="Name for the sample database"
    )


class FileStorageContainerSettings(BaseModel):
    """Configuration for file storage containers.

    Supports both local file system and Azure Blob Storage.
    """

    enable: bool = Field(True, description="Enable this storage container")
    storage_type: str = Field(
        "local",
        description="Storage type: 'local' for filesystem, 'azure' for blob storage",
    )
    container_name: str = Field(
        "", description="Container name for Azure storage (ignored for local)"
    )
    path: str = Field(
        "./", description="Storage path (local directory or Azure blob prefix)"
    )
    add_sub_folders: bool = Field(
        True, description="Create subdirectories for organization"
    )
    url: str = Field(
        "", description="Azure storage account URL (for Azure storage only)"
    )
    client_id: str = Field(
        "", description="Azure client ID for authentication (for Azure storage only)"
    )
    token: str = Field("", description="Azure access token (for Azure storage only)")
    authentication_method: AuthenticationMethod = Field(
        AuthenticationMethod.DEFAULT_CREDENTIAL,
        description="Authentication method for Azure: 'default_credential', 'msi', etc.",
    )


class FileStorageSettings(BaseModel):
    """Configuration for file storage system.

    Manages storage for revisions and data files.
    Supports local and cloud storage options.
    """

    revisions: FileStorageContainerSettings = Field(
        default_factory=lambda: FileStorageContainerSettings(
            enable=True,
            storage_type="local",
            container_name="",
            path="./",
            add_sub_folders=True,
            url="",
            client_id="",
            token="",
            authentication_method=AuthenticationMethod.DEFAULT_CREDENTIAL,
        )
    )
    data: FileStorageContainerSettings = Field(
        default_factory=lambda: FileStorageContainerSettings(
            enable=True,
            storage_type="local",
            container_name="",
            path="./",
            add_sub_folders=True,
            url="",
            client_id="",
            token="",
            authentication_method=AuthenticationMethod.DEFAULT_CREDENTIAL,
        )
    )


class ReceiverSettings(BaseModel):
    """Configuration for external event receivers.

    Enables integration with external systems that send events.
    Optional - disable if not using external integrations.
    """

    enable: bool = Field(False, description="Enable external event receiver")
    api_url: str = Field("", description="URL for receiving external events")
    api_key: str = Field(
        "DevApiKey", description="API key for authenticating external events"
    )
