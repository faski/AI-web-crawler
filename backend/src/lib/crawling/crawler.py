"""Web crawling via Crawl4AI.

Fetches a page by URL (or from raw HTML) and returns a PageContent with
the page title, raw HTML, and converted markdown.
"""

import re
from dataclasses import dataclass

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.async_configs import CacheMode

from .domains.registry import config_for as _domain_config

_DEFAULT_CONFIG = CrawlerRunConfig(magic=True)

# We start one browser the first time it is needed and reuse it for every
# crawl, instead of opening and closing Chromium on each request (which was
# the slowest part).
_crawler: AsyncWebCrawler | None = None


async def get_crawler() -> AsyncWebCrawler:
    """Return the shared crawler, starting its browser on first use."""
    global _crawler
    if _crawler is None:
        _crawler = AsyncWebCrawler(config=BrowserConfig(headless=True))
        await _crawler.start()
    return _crawler


async def close_crawler() -> None:
    """Close the shared crawler and its browser (called on shutdown)."""
    global _crawler
    if _crawler is not None:
        await _crawler.close()
        _crawler = None


def _config_for(url: str) -> CrawlerRunConfig:
    """Return the domain-specific config, or the default magic config."""
    return _domain_config(url) or _DEFAULT_CONFIG


def _html_only_config(url: str) -> CrawlerRunConfig:
    """Return a config for processing HTML we already have, without a browser.

    We keep the domain-specific filters (excluded_tags, excluded_selector,
    remove_forms) but drop ``magic`` and anything else that would start the
    browser, since the HTML is already downloaded and we only need to turn it
    into markdown.
    """
    base = _config_for(url)
    return CrawlerRunConfig(
        excluded_tags=base.excluded_tags,
        excluded_selector=base.excluded_selector,
        remove_forms=base.remove_forms,
    )


@dataclass
class PageContent:
    """Result returned by the crawler for a single page."""

    title: str
    html_text: str       # raw HTML of the page
    markdown_text: str   # markdown converted from the raw HTML


def _title(html: str, metadata: dict) -> str:
    """Get the page title from Crawl4AI metadata, or from the <title> tag."""
    title = (metadata or {}).get("title", "")
    if not title:
        match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        title = match.group(1).strip() if match else ""
    return title


async def fetch_page(url: str) -> PageContent:
    """Crawl a URL and return its title, raw HTML, and markdown.

    Raises:
        RuntimeError: if Crawl4AI reports a failed crawl.
    """
    run_config = _config_for(url)
    run_config.cache_mode = CacheMode.BYPASS
    crawler = await get_crawler()
    result = await crawler.arun(url=url, config=run_config)

    if not result.success:
        raise RuntimeError(f"Crawl failed for {url}: {result.error_message}")

    return PageContent(
        title=_title(result.html, result.metadata),
        html_text=result.html,
        markdown_text=result.markdown,
    )


async def fetch_page_from_html(url: str, html_text: str) -> PageContent:
    """Turn a stored HTML string into markdown, without using the browser.

    We hand the HTML to Crawl4AI through its ``raw:`` scheme with a browser-free
    config, so it skips the browser and only runs the HTML-to-markdown step. The
    domain-specific filters are still applied; the browser/JS rendering is left
    off on purpose because we already have the HTML.

    Raises:
        RuntimeError: if Crawl4AI fails to process the HTML.
    """
    run_config = _html_only_config(url)
    run_config.cache_mode = CacheMode.BYPASS
    crawler = await get_crawler()
    result = await crawler.arun(url=f"raw:{html_text}", config=run_config)

    if not result.success:
        raise RuntimeError(f"HTML processing failed: {result.error_message}")

    return PageContent(
        title=_title(html_text, result.metadata),
        html_text=html_text,
        markdown_text=result.markdown,
    )
