# File: ingenious/services/azure_search/tests/azure_search/test_builders_fixes.py

import pytest

from ingenious.services.azure_search.builders import ConfigError, _validate_endpoint


@pytest.mark.parametrize(
    "endpoint, expected_output",
    [
        ("https://valid.search.windows.net", "https://valid.search.windows.net"),
        ("http://localhost:8080/api", "http://localhost:8080/api"),
        (
            " https://spaced-url.com/path ",
            "https://spaced-url.com/path",
        ),  # Trimming verification
    ],
)
def test_validate_endpoint_valid_formats(endpoint, expected_output):
    """Test valid URL formats for endpoints are accepted and normalized."""
    assert _validate_endpoint(endpoint, "TestService") == expected_output


@pytest.mark.parametrize(
    "endpoint, error_message_substring",
    [
        ("", "cannot be empty"),
        ("   ", "cannot be empty"),
        ("ftp://invalid.scheme.com", "must use http or https scheme"),
        ("sftp://invalid.scheme.com", "must use http or https scheme"),
        ("just-the-hostname.com", "must be a valid URL with scheme and host"),
        ("https://", "must be a valid URL with scheme and host"),  # Missing host/netloc
        ("://missing.scheme/path", "must be a valid URL with scheme and host"),
    ],
)
def test_validate_endpoint_invalid_formats(endpoint, error_message_substring):
    """Test invalid URL formats raise ConfigError with specific messages."""
    with pytest.raises(ConfigError) as excinfo:
        _validate_endpoint(endpoint, "TestService")

    # Verify the error message contains the expected diagnostic substring
    assert error_message_substring in str(excinfo.value)
