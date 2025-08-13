from typing import List, Optional

from pydantic import BaseModel, Field, RootModel

from ingenious.common.enums import AuthenticationMethod


class ModelConfig(BaseModel):
    model: str
    api_key: str = ""
    base_url: str
    deployment: str = ""
    api_version: str = ""
    client_id: str = ""
    client_secret: str = ""
    tenant_id: str = ""
    authentication_method: AuthenticationMethod = (
        AuthenticationMethod.DEFAULT_CREDENTIAL
    )


class ChatHistoryConfig(BaseModel):
    database_connection_string: str = ""


class AzureSqlConfig(BaseModel):
    database_connection_string: str = ""


class AzureSearchConfig(BaseModel):
    service: str = "default"
    key: str = ""


class ToolServiceConfig(BaseModel):
    pass


class ChatServiceConfig(BaseModel):
    pass


class WebAuthConfig(BaseModel):
    type: str = "basic"
    enable: bool = False
    username: str = "admin"
    password: str = ""


class WebConfig(BaseModel):
    authentication: WebAuthConfig = Field(default_factory=WebAuthConfig)


class ReceiverConfig(BaseModel):
    enable: bool = False
    api_url: str = ""
    api_key: str = "DevApiKey"


class LoggingConfig(BaseModel):
    pass


class FileStorageContainer(BaseModel):
    url: str = Field("", description="File Storage SAS URL")
    client_id: str = Field("", description="File Storage SAS Client ID")
    token: str = Field("", description="File Storage SAS Token")
    authentication_method: AuthenticationMethod = Field(
        AuthenticationMethod.DEFAULT_CREDENTIAL,
        description="File Storage SAS Authentication Method",
    )


class FileStorage(BaseModel):
    revisions: FileStorageContainer = Field(
        default_factory=lambda: FileStorageContainer(
            url="",
            client_id="",
            token="",
            authentication_method=AuthenticationMethod.DEFAULT_CREDENTIAL,
        ),
        description="File Storage configuration for revisions",
    )
    data: FileStorageContainer = Field(
        default_factory=lambda: FileStorageContainer(
            url="",
            client_id="",
            token="",
            authentication_method=AuthenticationMethod.DEFAULT_CREDENTIAL,
        ),
        description="File Storage configuration for data",
    )


class Profile(BaseModel):
    """
    This is the class for the profiles.yml file. It contains only secret information.
    """

    name: str
    models: List[ModelConfig]
    chat_history: ChatHistoryConfig
    chat_service: ChatServiceConfig = Field(
        default_factory=ChatServiceConfig, description="Chat service configuration"
    )
    azure_search_services: Optional[List[AzureSearchConfig]] = Field(
        default=None, description="Azure Search services configuration (optional)"
    )
    azure_sql_services: Optional[AzureSqlConfig] = Field(
        default=None, description="Azure SQL services configuration (optional)"
    )
    web_configuration: WebConfig
    receiver_configuration: ReceiverConfig
    tool_service: ToolServiceConfig = Field(
        default_factory=ToolServiceConfig, description="Tool service configuration"
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig, description="Logging configuration"
    )
    file_storage: FileStorage = Field(
        default_factory=FileStorage, description="File Storage configuration"
    )


class Profiles(RootModel[List[Profile]]):
    pass
