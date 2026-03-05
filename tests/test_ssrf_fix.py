import os
import urllib.parse
import pytest

def is_valid_url(url: str) -> bool:
    """Validate that the URL hostname is in the allowlist."""
    try:
        parsed = urllib.parse.urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return False

        hostname = parsed.hostname
        if not hostname:
            return False

        # Default allowed hosts for development and docker-compose
        allowed_hosts = {"localhost", "127.0.0.1", "api"}

        # Allow extending the allowlist via environment variable
        env_allowed = os.getenv("ALLOWED_API_HOSTS")
        if env_allowed:
            allowed_hosts.update(h.strip() for h in env_allowed.split(",") if h.strip())

        return hostname in allowed_hosts
    except Exception:
        return False

def test_is_valid_url():
    # Allowed hosts
    assert is_valid_url("http://localhost:8001") is True
    assert is_valid_url("http://127.0.0.1:8001") is True
    assert is_valid_url("http://api:8000") is True

    # Disallowed hosts
    assert is_valid_url("http://evil.com") is False
    assert is_valid_url("https://google.com") is False
    assert is_valid_url("http://192.168.1.1") is False

    # Invalid URLs
    assert is_valid_url("not-a-url") is False
    assert is_valid_url("http://") is False

def test_is_valid_url_with_env(monkeypatch):
    monkeypatch.setenv("ALLOWED_API_HOSTS", "internal.server, other.host")

    # Still allowed
    assert is_valid_url("http://localhost:8001") is True

    # Newly allowed
    assert is_valid_url("http://internal.server") is True
    assert is_valid_url("http://other.host:9000") is True

    # Still disallowed
    assert is_valid_url("http://evil.com") is False
