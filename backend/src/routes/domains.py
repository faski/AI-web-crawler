"""Route handler for GET /domains."""

from fastapi import APIRouter

from ..schemas import DomainsResponse
from ..lib import get_domains

router = APIRouter()


@router.get("/domains", response_model=DomainsResponse)
def domains():
    """Return the list of domains that have gold standard data."""
    return DomainsResponse(domains=get_domains())
