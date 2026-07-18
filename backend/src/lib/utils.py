"""Generic HTTP and URL utilities shared across routes."""

from urllib.parse import urlparse

from fastapi import HTTPException

from .gold_standard.urls import is_supported_domain


def domain_of(url: str) -> str:
    """Return the netloc (hostname) of a URL."""
    return urlparse(url).netloc


def assert_supported_domain(domain: str) -> None:
    """Raise HTTP 400 if the domain has no gold standard data."""
    if not is_supported_domain(domain):
        raise HTTPException(status_code=400, detail=f"Unsupported domain: {domain}")
