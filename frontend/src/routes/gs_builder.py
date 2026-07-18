"""Gold Standard Builder page (GET /gs-builder + POST /gs-builder/save + POST /gs-builder/delete)."""

from fastapi import APIRouter, Form, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from ..client import (
    BackendUnavailable,
    add_gold_standard,
    add_web_resource,
    delete_gold_standard,
    get_domains,
    get_gold_standard,
    get_gs_urls,
    parse_url,
)
from ..templates import templates

router = APIRouter()


@router.get("/gs-builder", response_class=HTMLResponse)
def gs_builder(
    request: Request,
    domain: str = Query(default=""),
    url: str = Query(default=""),
    message: str = Query(default=""),
    error: str = Query(default=""),
):
    """Render the builder page with optional pre-loaded HTML for a URL."""
    try:
        domains = get_domains()
        selected_domain = domain or (domains[0] if domains else "")
        existing_urls = get_gs_urls(selected_domain) if selected_domain else []

        loaded_html = ""
        loaded_gold_text = ""
        load_error = ""
        if url.strip():
            data, parse_error = parse_url(url.strip(), local=False)
            if parse_error:
                load_error = parse_error
            else:
                loaded_html = data.get("html_text", "")
            # Pre-fill the gold text if the URL is already in the gold standard
            # (edit mode); otherwise the textarea stays empty (create mode).
            existing_entry = get_gold_standard(url.strip())
            if existing_entry:
                loaded_gold_text = existing_entry.get("gold_text", "")
    except BackendUnavailable:
        return templates.TemplateResponse(
            request=request, name="error.html.jinja", status_code=503
        )

    return templates.TemplateResponse(
        request=request,
        name="gs_builder.html.jinja",
        context={
            "domains": domains,
            "selected_domain": selected_domain,
            "selected_url": url.strip(),
            "loaded_html": loaded_html,
            "loaded_gold_text": loaded_gold_text,
            "load_error": load_error,
            "existing_urls": existing_urls,
            "message": message,
            "error": error,
        },
    )


@router.post("/gs-builder/save")
def gs_builder_save(
    domain: str = Form(...),
    url: str = Form(...),
    html_text: str = Form(...),
    gold_text: str = Form(...),
):
    """Persist the HTML + gold_text for a URL, then redirect back to the builder."""
    try:
        resource_ok, resource_error = add_web_resource(url, html_text)
        if not resource_ok:
            return RedirectResponse(
                url=f"/gs-builder?domain={domain}&error={resource_error}",
                status_code=303,
            )
        gold_ok, gold_error = add_gold_standard(url, gold_text)
        if not gold_ok:
            return RedirectResponse(
                url=f"/gs-builder?domain={domain}&error={gold_error}",
                status_code=303,
            )
    except BackendUnavailable:
        return RedirectResponse(url="/gs-builder?error=Backend+not+available", status_code=303)
    return RedirectResponse(
        url=f"/gs-builder?domain={domain}&message=Saved",
        status_code=303,
    )


@router.post("/gs-builder/delete")
def gs_builder_delete(domain: str = Form(...), url: str = Form(...)):
    """Remove a URL from the gold standard, then redirect back to the builder."""
    try:
        ok, error = delete_gold_standard(url)
        if not ok:
            return RedirectResponse(
                url=f"/gs-builder?domain={domain}&error={error}",
                status_code=303,
            )
    except BackendUnavailable:
        return RedirectResponse(url="/gs-builder?error=Backend+not+available", status_code=303)
    return RedirectResponse(
        url=f"/gs-builder?domain={domain}&message=Removed",
        status_code=303,
    )
