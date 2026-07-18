"""Pydantic schemas for the domains endpoint."""

from pydantic import BaseModel


class DomainsResponse(BaseModel):
    """Response for GET /domains: list of supported domain names."""

    domains: list[str]
