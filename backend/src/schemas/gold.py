"""Pydantic schemas for gold standard endpoints."""

from pydantic import BaseModel


class GoldStandardResponse(BaseModel):
    """Response for GET /gold_standard: crawled content + manually curated gold text."""

    url: str
    domain: str
    title: str
    html_text: str   # raw HTML of the page
    gold_text: str   # manually curated clean text — ground truth for evaluation


class GoldStandardUrlsResponse(BaseModel):
    """Response for GET /gold_standard_urls: list of GS URLs for a domain."""

    gold_standard_urls: list[str]


class AddWebResourceRequest(BaseModel):
    """Request body for POST /add_web_resource."""

    url: str
    html_text: str


class AddGoldStandardRequest(BaseModel):
    """Request body for POST /add_gold_standard."""

    url: str
    gold_text: str


class DeleteUrlRequest(BaseModel):
    """Request body for DELETE /web_resource and DELETE /gold_standard."""

    url: str


class StatusResponse(BaseModel):
    """Response for the add/delete endpoints: ``ok`` or ``error``."""

    status: str
