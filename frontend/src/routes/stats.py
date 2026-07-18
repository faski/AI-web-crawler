"""Stats page (GET /stats)."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from ..client import BackendUnavailable, get_db_stats, get_domains
from ..templates import templates

router = APIRouter()


@router.get("/stats", response_class=HTMLResponse)
def stats(request: Request):
    """Render aggregated statistics for every supported domain."""
    try:
        domains = get_domains()
        db_stats = get_db_stats()
    except BackendUnavailable:
        return templates.TemplateResponse(
            request=request, name="error.html.jinja", status_code=503
        )

    return templates.TemplateResponse(
        request=request,
        name="stats.html.jinja",
        context={"domains": domains, "db_stats": db_stats},
    )
