"""Pydantic schemas for the parse endpoint."""

from pydantic import BaseModel


class ParseRequest(BaseModel):
    """Request body for POST /parse.

    If ``local`` is True the HTML is read from the local database; otherwise
    the page is crawled live.
    """

    url: str
    local: bool | None = None


class ParseResponse(BaseModel):
    """Response for GET /parse and POST /parse."""

    url: str
    domain: str
    title: str
    html_text: str    # raw HTML of the page
    parsed_text: str  # clean text produced by the domain-specific parser
    cleaned_text: str  # parsed_text with markdown removed
    # True only when a live crawl was blocked (e.g. bot protection) and we fell
    # back to the HTML already stored in the DB. The UI uses it to warn the user.
    fallback_used: bool = False
