"""Route handlers for gold standard endpoints (read + write + delete)."""

import re

from fastapi import APIRouter, HTTPException, Query

from ..lib import (
    assert_supported_domain,
    domain_of,
    get_entry_for_url,
    get_urls_for_domain,
)
from ..lib.db import queries
from ..schemas import (
    AddGoldStandardRequest,
    AddWebResourceRequest,
    DeleteUrlRequest,
    GoldStandardResponse,
    GoldStandardUrlsResponse,
    StatusResponse,
)

router = APIRouter()

TITLE_PATTERN = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


def _extract_title_from_html(html: str) -> str:
    """Return the text inside the first <title>...</title> tag, or empty string."""
    match = TITLE_PATTERN.search(html)
    return match.group(1).strip() if match else ""


# ─── Read ────────────────────────────────────────────────────────────────────

@router.get("/gold_standard", response_model=GoldStandardResponse)
def gold_standard(url: str = Query(...)):
    """Return the gold standard entry for a URL."""
    assert_supported_domain(domain_of(url))
    entry = get_entry_for_url(url)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"URL not found in gold standard: {url}")
    return GoldStandardResponse(
        url=entry["url"],
        domain=entry["domain"],
        title=entry.get("title", ""),
        html_text=entry.get("html_text", ""),
        gold_text=entry.get("gold_text", ""),
    )


@router.get("/gold_standard_urls", response_model=GoldStandardUrlsResponse)
def gold_standard_urls(domain: str = Query(...)):
    """Return all gold standard URLs for the given domain."""
    assert_supported_domain(domain)
    return GoldStandardUrlsResponse(gold_standard_urls=get_urls_for_domain(domain))


# ─── Write ───────────────────────────────────────────────────────────────────

@router.post("/add_web_resource", response_model=StatusResponse)
def add_web_resource(body: AddWebResourceRequest):
    """Insert (or overwrite) a web_resources row. Domain and title are derived from the input."""
    title = _extract_title_from_html(body.html_text)
    queries.add_resource(body.url, body.html_text, title=title)
    return StatusResponse(status="ok")


@router.post("/add_gold_standard", response_model=StatusResponse)
def add_gold_standard(body: AddGoldStandardRequest):
    """Insert (or overwrite) a gold_standard row.

    Returns ``status=error`` if the matching web_resource does not exist.
    """
    inserted = queries.add_gold_standard(body.url, body.gold_text)
    return StatusResponse(status="ok" if inserted else "error")


# ─── Delete ──────────────────────────────────────────────────────────────────

@router.delete("/web_resource", response_model=StatusResponse)
def delete_web_resource(body: DeleteUrlRequest):
    """Remove a web_resources row (cascades to the matching gold_standard row).

    Returns ``status=ok`` if a row was removed, ``status=error`` otherwise.
    """
    removed = queries.delete_resource(body.url)
    return StatusResponse(status="ok" if removed else "error")


@router.delete("/gold_standard", response_model=StatusResponse)
def delete_gold_standard(body: DeleteUrlRequest):
    """Remove only the gold_standard row, leaving the web_resource in place.

    Returns ``status=ok`` if a row was removed, ``status=error`` otherwise.
    """
    removed = queries.delete_gold_standard(body.url)
    return StatusResponse(status="ok" if removed else "error")
