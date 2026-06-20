"""Domain whitelist enforcement.

Every outbound HTTP request made by the downloader must pass through
``assert_domain_allowed`` first. This is the single choke point that keeps
the system from ever fetching content from an untrusted host.
"""
from __future__ import annotations

from urllib.parse import urlparse

from research_rag.config import ALLOWED_DOMAINS


class DomainNotAllowedError(Exception):
    """Raised when a URL's host is not on the whitelist."""


def is_domain_allowed(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    if not host:
        return False
    return any(host == d or host.endswith("." + d) for d in ALLOWED_DOMAINS)


def assert_domain_allowed(url: str) -> None:
    if not is_domain_allowed(url):
        raise DomainNotAllowedError(
            f"Refusing to contact non-whitelisted host for URL: {url}"
        )
