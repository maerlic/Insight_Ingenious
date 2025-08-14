# tests/azure_search/test_builders_edge_cases.py

import pytest

from ingenious.services.azure_search.builders import ConfigError, _validate_endpoint


@pytest.mark.parametrize(
    "invalid_url",
    [
        "example.com",  # Missing scheme
        "ftp://example.com",  # Invalid scheme
        "https://",  # Missing host
        "",  # Empty
        "  ",  # Whitespace
    ],
)
def test_builders_validate_endpoint_malformed_urls(invalid_url):
    """
    P3: Test _validate_endpoint with inputs missing scheme or host, or invalid schemes.
    """
    with pytest.raises(ConfigError):
        _validate_endpoint(invalid_url, "Test Endpoint")


def test_builders_validate_endpoint_valid_url():
    """
    Ensure valid URLs pass and are stripped.
    """
    valid_url = "  https://valid.example.com/path  "
    expected = "https://valid.example.com/path"
    assert _validate_endpoint(valid_url, "Test Endpoint") == expected
