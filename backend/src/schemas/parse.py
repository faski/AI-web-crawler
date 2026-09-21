"""Pydantic schemas for the parse endpoint."""

from typing import Literal

from pydantic import BaseModel


class ParseRequest(BaseModel):
    """Request body for POST /parse.

    If ``local`` is True the HTML is read from the local database; otherwise
    the page is crawled live.
    """

    url: str
    local: bool | None = None
    # Which pipeline extracts the text: "crawl4ai" is markdown plus a domain
    # parser, "llm" is the raw HTML straight to a model. The server refuses
    # the second unless PARSE_WITH_LLM is on.
    parser: Literal["crawl4ai", "llm"] = "crawl4ai"
    # Ask the model again instead of reading the extraction a finished run
    # already stored for this URL. Only this path spends anything, so only
    # this path goes through the gate.
    fresh: bool = False


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
    # Which pipeline produced parsed_text, echoed back so the page showing
    # the result can say where it came from instead of assuming.
    parser: str = "crawl4ai"
    # What the extraction took. Only the LLM parser fills these in: a zero
    # for Crawl4AI would sit in the same column as a price.
    seconds: float | None = None
    cost_eur: float | None = None
    fragments: int | None = None
    grounded: float | None = None
    # The model that did the work, and the company that served it.
    # ``provider`` is empty on the local path, which has no upstream, so
    # there the model name is what identifies the run.
    model: str | None = None
    provider: str | None = None
    # "live" if the model was called now, "stored" if the text comes from a
    # run that finished earlier. A page that showed stored text as a fresh
    # parse would be claiming a call it never made.
    source: str | None = None
    run_label: str | None = None
    run_date: str | None = None
