"""Route handler for POST /parse.

Two pipelines answer here. ``crawl4ai`` is the base project's: Crawl4AI turns
the page into markdown and a domain parser cleans it. ``llm`` is the one this
project adds, where the raw HTML goes to a model - the same path the batch
runs measure, reachable from the page so it can be shown and not only
reported.

The LLM pipeline is refused unless PARSE_WITH_LLM is on, because a page
anyone can load must not spend credit or hang for minutes by accident. The
scripts guard the same thing with --openrouter.
"""

import os
import time

from fastapi import APIRouter, HTTPException

from ..lib import (
    assert_supported_domain,
    domain_of,
    fetch_page,
    fetch_page_from_html,
    get_parser_for_url,
)
from ..lib.evaluation import strip_markdown
from ..lib.llm import client
from ..lib.db import llm_queries, queries
from ..lib.db.models import WebResource
from ..lib.parsers.llm_parser import (
    EmptyExtractionError,
    HtmlTooLongError,
    LlmParser,
    TruncatedAnswerError,
)
from ..schemas import ParseRequest, ParseResponse

router = APIRouter()


def llm_parsing_enabled() -> bool:
    """Return whether the page is allowed to parse with the model."""
    return os.environ.get("PARSE_WITH_LLM", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _extract(
    url: str, markdown_text: str, html_text: str, parser: str, fresh: bool = False
) -> dict:
    """Run one of the two pipelines and return the text plus what it took.

    The extra fields stay empty for Crawl4AI: it runs here and costs nothing
    per call, and a zero in a cost column reads as a price, not an absence.

    For the LLM the stored extraction comes first. The finished runs hold the
    Markdown for all forty gold standard pages, so the page can show what the
    model really produced without paying for a call or making anyone wait
    minutes. ``fresh`` asks for a new call anyway.
    """
    if parser != "llm":
        return {
            "parsed_text": get_parser_for_url(url).parse(url, markdown_text),
            "parser": "crawl4ai",
        }

    if not fresh:
        stored = llm_queries.get_stored_extraction(url)
        if stored is not None:
            return {
                "parsed_text": stored["parsed_text"],
                "parser": "llm",
                "source": "stored",
                "run_label": stored["run_label"],
                "run_date": stored["run_date"],
                "model": stored["model"],
                "seconds": stored["seconds"],
                "cost_eur": stored["cost_eur"],
                "fragments": stored["fragments"],
                "grounded": stored["grounded"],
            }

    if not llm_parsing_enabled():
        # Two reasons to keep it off, and picking the wrong one is how a
        # page warns about money while a free local model hangs for an hour.
        motivo = (
            "ogni richiesta costa credito"
            if client.get_provider() == "openrouter"
            else f"il modello locale {client.get_model_name()} impiega minuti "
                 "per pagina e la richiesta resterebbe appesa"
        )
        raise HTTPException(
            status_code=403,
            detail=f"Il parser LLM e' spento: {motivo}. Per accenderlo, "
                   "PARSE_WITH_LLM=1 nell'ambiente del backend.",
        )

    started = time.monotonic()
    try:
        outcome = LlmParser().parse_html_detailed(url, html_text)
    except HtmlTooLongError as error:
        raise HTTPException(
            status_code=413,
            detail=f"L'HTML e' di circa {error.estimated_tokens:,} token, oltre "
                   f"il budget di {error.budget_tokens:,}. Attiva il chunking "
                   "con LLM_PARSER_CHUNKING=1 per leggerla a pezzi.",
        )
    except TruncatedAnswerError as error:
        raise HTTPException(status_code=502, detail=str(error))
    except EmptyExtractionError as error:
        # An empty document would look like a parsed page with nothing on it.
        # It is the opposite: the page has content and the model refused it.
        raise HTTPException(
            status_code=502,
            detail=f"Il modello non ha restituito alcun contenuto per {error.url}"
                   + (f", in nessuno dei {error.fragments} pezzi in cui la "
                      "pagina e' stata divisa." if error.fragments > 1 else "."),
        )
    except RuntimeError as error:
        # Missing key, empty answer, provider down: the model gave nothing
        # usable, and none of these is a bad page.
        raise HTTPException(status_code=502, detail=f"{type(error).__name__}: {error}")

    return {
        "parsed_text": outcome.text,
        "parser": "llm",
        "seconds": round(time.monotonic() - started, 1),
        "cost_eur": outcome.cost_eur,
        "fragments": outcome.fragments,
        "grounded": outcome.grounded,
        "model": client.get_model_name(),
        "provider": ", ".join(outcome.providers_used) or None,
        "source": "live",
    }


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
        return await _parse_local(body.url, body.parser, body.fresh)
    if body.local is False:
        return await _parse_live(body.url, body.parser, body.fresh)
    return await _parse_cached_or_live(body.url, body.parser, body.fresh)


async def _parse_cached_or_live(
    url: str, parser: str = "crawl4ai", fresh: bool = False
) -> ParseResponse:
    """Parse the cached DB copy if present, otherwise crawl live and cache it."""
    domain = domain_of(url)
    assert_supported_domain(domain)
    resource = queries.get_resource(url)
    if resource is not None:
        return await _parse_stored_resource(
            url, domain, resource, parser=parser, fresh=fresh
        )
    # First time we see this URL: crawl it live and store the HTML for next time.
    return await _parse_live_and_cache(url, domain, parser, fresh)


async def _parse_live_and_cache(
    url: str, domain: str, parser: str = "crawl4ai", fresh: bool = False
) -> ParseResponse:
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
    extraction = _extract(url, page.markdown_text, page.html_text, parser, fresh)
    return ParseResponse(
        url=url,
        domain=domain,
        title=page.title,
        html_text=page.html_text,
        cleaned_text=strip_markdown(extraction["parsed_text"]),
        **extraction,
    )


async def _parse_live(
    url: str, parser: str = "crawl4ai", fresh: bool = False
) -> ParseResponse:
    """Crawl the URL live; if the crawl fails, fall back to the stored HTML."""
    domain = domain_of(url)
    assert_supported_domain(domain)
    try:
        page = await fetch_page(url)
    except RuntimeError as crawl_error:
        # The live crawl failed (network error, bot protection, ...). If we have
        # the page saved in the DB we use that copy instead of giving up.
        return await _parse_live_with_db_fallback(
            url, domain, crawl_error, parser, fresh
        )

    extraction = _extract(url, page.markdown_text, page.html_text, parser, fresh)
    return ParseResponse(
        url=url,
        domain=domain,
        title=page.title,
        html_text=page.html_text,
        cleaned_text=strip_markdown(extraction["parsed_text"]),
        **extraction,
    )


async def _parse_live_with_db_fallback(
    url: str,
    domain: str,
    crawl_error: RuntimeError,
    parser: str = "crawl4ai",
    fresh: bool = False,
) -> ParseResponse:
    """Use the stored HTML after a failed live crawl, or report the crawl error."""
    resource = queries.get_resource(url)
    if resource is None:
        # No saved copy to fall back to, so the page really is unreachable.
        raise HTTPException(status_code=502, detail=str(crawl_error))
    return await _parse_stored_resource(
        url, domain, resource, fallback_used=True, parser=parser, fresh=fresh
    )


async def _parse_local(
    url: str, parser: str = "crawl4ai", fresh: bool = False
) -> ParseResponse:
    """Read the stored HTML from the DB and run it through the domain parser."""
    domain = domain_of(url)
    assert_supported_domain(domain)
    resource = queries.get_resource(url)
    if resource is None:
        raise HTTPException(status_code=404, detail=f"URL not found in DB: {url}")
    return await _parse_stored_resource(
        url, domain, resource, parser=parser, fresh=fresh
    )


async def _parse_stored_resource(
    url: str,
    domain: str,
    resource: WebResource,
    fallback_used: bool = False,
    parser: str = "crawl4ai",
    fresh: bool = False,
) -> ParseResponse:
    """Convert the stored HTML to markdown and run the domain parser on it.

    ``fallback_used`` is True only when we reached this function because a live
    crawl was blocked, so the UI can warn that the result comes from the DB.
    """
    try:
        page = await fetch_page_from_html(url, resource.html_text)
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error))
    # The LLM parser gets the stored HTML, not the markdown Crawl4AI made
    # from it: raw HTML in is the point of the condition being measured.
    extraction = _extract(url, page.markdown_text, resource.html_text, parser, fresh)
    return ParseResponse(
        url=url,
        domain=domain,
        title=page.title or resource.title,
        html_text=resource.html_text,
        cleaned_text=strip_markdown(extraction["parsed_text"]),
        fallback_used=fallback_used,
        **extraction,
    )
