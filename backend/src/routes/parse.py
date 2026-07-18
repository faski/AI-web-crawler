"""Route handler for POST /parse."""

from fastapi import APIRouter, HTTPException

from ..lib import (
    assert_supported_domain,
    domain_of,
    fetch_page,
    fetch_page_from_html,
    get_parser_for_url,
)
from ..lib.evaluation import strip_markdown
from ..lib.db import queries
from ..lib.db.models import WebResource
from ..schemas import ParseRequest, ParseResponse

router = APIRouter()


@router.post("/parse", response_model=ParseResponse)
async def parse(body: ParseRequest):
    """Parse a URL, preferring the copy already cached in the DB.

    The ``local`` flag has three cases:

    * ``True``  - always read the HTML stored in the ``web_resources`` table.
    * ``False`` - always crawl the page live (falling back to the stored HTML
      only if the live crawl fails, e.g. because the site blocks bots).
    * not set (default) - use the cached DB copy when we already have it, and
      crawl live only the first time we see the URL. A freshly crawled page is
      saved to the DB (without a gold text) so the next /parse reads it from
      the cache instead of crawling again.

    Preferring the cached copy keeps results stable and avoids re-crawling
    sites that block bots (for example ESPN), which would otherwise return a
    bot-protection page instead of the real article.
    """
    if body.local is True:
        return await _parse_local(body.url)
    if body.local is False:
        return await _parse_live(body.url)
    return await _parse_cached_or_live(body.url)


async def _parse_cached_or_live(url: str) -> ParseResponse:
    """Parse the cached DB copy if present, otherwise crawl live and cache it."""
    domain = domain_of(url)
    assert_supported_domain(domain)
    resource = queries.get_resource(url)
    if resource is not None:
        return await _parse_stored_resource(url, domain, resource)
    # First time we see this URL: crawl it live and store the HTML for next time.
    return await _parse_live_and_cache(url, domain)


async def _parse_live_and_cache(url: str, domain: str) -> ParseResponse:
    """Crawl a not-yet-cached URL live, save its HTML, and parse it.

    The fetched HTML is written to ``web_resources`` without a gold text, so a
    later /parse on the same URL reads from the DB. A gold text is only added
    later through the gold-standard page for that specific URL.
    """
    try:
        page = await fetch_page(url)
    except RuntimeError as crawl_error:
        # The first crawl failed and there is nothing cached to fall back to,
        # so the page is genuinely unreachable.
        raise HTTPException(status_code=502, detail=str(crawl_error))
    queries.add_resource(url, page.html_text, page.title)
    parsed_text = get_parser_for_url(url).parse(url, page.markdown_text)
    return ParseResponse(
        url=url,
        domain=domain,
        title=page.title,
        html_text=page.html_text,
        parsed_text=parsed_text,
        cleaned_text=strip_markdown(parsed_text),
    )


async def _parse_live(url: str) -> ParseResponse:
    """Crawl the URL live; if the crawl fails, fall back to the stored HTML."""
    domain = domain_of(url)
    assert_supported_domain(domain)
    try:
        page = await fetch_page(url)
    except RuntimeError as crawl_error:
        # The live crawl failed (network error, bot protection, ...). If we have
        # the page saved in the DB we use that copy instead of giving up.
        return await _parse_live_with_db_fallback(url, domain, crawl_error)

    parsed_text = get_parser_for_url(url).parse(url, page.markdown_text)
    return ParseResponse(
        url=url,
        domain=domain,
        title=page.title,
        html_text=page.html_text,
        parsed_text=parsed_text,
        cleaned_text=strip_markdown(parsed_text),
    )


async def _parse_live_with_db_fallback(
    url: str, domain: str, crawl_error: RuntimeError
) -> ParseResponse:
    """Use the stored HTML after a failed live crawl, or report the crawl error."""
    resource = queries.get_resource(url)
    if resource is None:
        # No saved copy to fall back to, so the page really is unreachable.
        raise HTTPException(status_code=502, detail=str(crawl_error))
    return await _parse_stored_resource(url, domain, resource, fallback_used=True)


async def _parse_local(url: str) -> ParseResponse:
    """Read the stored HTML from the DB and run it through the domain parser."""
    domain = domain_of(url)
    assert_supported_domain(domain)
    resource = queries.get_resource(url)
    if resource is None:
        raise HTTPException(status_code=404, detail=f"URL not found in DB: {url}")
    return await _parse_stored_resource(url, domain, resource)


async def _parse_stored_resource(
    url: str, domain: str, resource: WebResource, fallback_used: bool = False
) -> ParseResponse:
    """Convert the stored HTML to markdown and run the domain parser on it.

    ``fallback_used`` is True only when we reached this function because a live
    crawl was blocked, so the UI can warn that the result comes from the DB.
    """
    try:
        page = await fetch_page_from_html(url, resource.html_text)
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error))
    parsed_text = get_parser_for_url(url).parse(url, page.markdown_text)
    return ParseResponse(
        url=url,
        domain=domain,
        title=page.title or resource.title,
        html_text=resource.html_text,
        parsed_text=parsed_text,
        cleaned_text=strip_markdown(parsed_text),
        fallback_used=fallback_used,
    )
