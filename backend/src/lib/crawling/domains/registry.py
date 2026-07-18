"""Crawler-config registry: maps URL hostnames to their CrawlerRunConfig.

To add a new domain:
  1. Create a module in this package (e.g. domains/reddit.py) that exposes CONFIG.
  2. Import it here and add an entry in DOMAIN_CONFIGS.
"""

from urllib.parse import urlparse

from crawl4ai import CrawlerRunConfig

from . import cnbc, espn, wikipedia, xe

DOMAIN_CONFIGS: dict[str, CrawlerRunConfig] = {
    "it.wikipedia.org": wikipedia.CONFIG,
    "www.espn.com": espn.CONFIG,
    "www.cnbc.com": cnbc.CONFIG,
    "www.xe.com": xe.CONFIG,
}


def config_for(url: str) -> CrawlerRunConfig | None:
    """Return the domain-specific crawler config, or None for unknown domains.

    Falls back to the default config defined in crawler.py.
    """
    return DOMAIN_CONFIGS.get(urlparse(url).netloc.lower())
