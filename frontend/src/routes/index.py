"""Home page (GET /)."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from ..client import BackendUnavailable, get_domains, get_status
from ..templates import templates

router = APIRouter()


GROUP_MEMBERS = [
    {"name": "Lapo Siciliani", "id": "2007890"},
    {"name": "Fabio Priori", "id": "1938446"},
    {"name": "Tamer Hayek", "id": "1897438"},
]


@router.get("/", response_class=HTMLResponse)
def index(request: Request):
    """Render the home page with system status and supported domains."""
    try:
        status = get_status()
        domains = get_domains()
    except BackendUnavailable:
        status, domains = {"backend": "error", "database": "error", "ollama": "error"}, []
    return templates.TemplateResponse(
        request=request,
        name="index.html.jinja",
        context={
            "group_members": GROUP_MEMBERS,
            "status": status,
            "domains": domains,
        },
    )
