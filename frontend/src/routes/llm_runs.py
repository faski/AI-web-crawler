"""LLM parser runs and comparison page (GET /llm-runs)."""

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse

from ..client import (
    BackendUnavailable,
    compare_llm_runs,
    compare_llm_text,
    get_llm_run_domains,
    get_llm_runs,
    get_llm_self_check,
)
from ..templates import templates

router = APIRouter()


def _default_pair(runs: list[dict], left: int, right: int) -> tuple[int, int]:
    """Pick which two runs to compare when the request does not say.

    Comparing a run with itself shows nothing useful, so the first visit lands
    on the two most recent runs instead of twice the same one.
    """
    ids = [run["id"] for run in runs]
    if left in ids and right in ids and left != right:
        return left, right
    if len(ids) < 2:
        return 0, 0
    if left in ids and left != ids[0]:
        return ids[0], left
    return ids[0], ids[1]


@router.get("/llm-runs", response_class=HTMLResponse)
def llm_runs_page(
    request: Request,
    left: int = Query(default=0),
    right: int = Query(default=0),
):
    """Show the stored runs, and two of them side by side.

    Everything here is read from the database. Opening or reloading this page
    never calls a model and never costs anything.
    """
    try:
        runs = get_llm_runs()
        left, right = _default_pair(runs, left, right)
        # The self-check is a separate, paid pass, so most runs do not have
        # one. The first run that does is shown; without this the section
        # would appear empty on a page whose runs were simply never checked.
        self_check = None
        for run in runs:
            found = get_llm_self_check(run["id"])
            if found:
                self_check = dict(found, label=run["label"])
                break
        by_domain = {}
        comparison = []
        if left and right:
            by_domain = {
                "left": get_llm_run_domains(left),
                "right": get_llm_run_domains(right),
            }
            comparison = compare_llm_runs(left, right)
    except BackendUnavailable:
        return templates.TemplateResponse(
            request=request, name="error.html.jinja", status_code=503
        )

    runs_by_id = {run["id"]: run for run in runs}
    return templates.TemplateResponse(
        request=request,
        name="llm_runs.html.jinja",
        context={
            "runs": runs,
            "left": left,
            "right": right,
            "left_run": runs_by_id.get(left),
            "right_run": runs_by_id.get(right),
            "by_domain": by_domain,
            "comparison": comparison,
            "self_check": self_check,
        },
    )


@router.get("/llm-runs/text", response_class=HTMLResponse)
def llm_text_page(request: Request, left: int, right: int, url: str):
    """Show the gold text and what the two runs produced for one page.

    Like the rest of this section it only reads stored rows: no page is parsed
    again and no model is called.
    """
    try:
        runs = get_llm_runs()
        data = compare_llm_text(left, right, url)
    except BackendUnavailable:
        return templates.TemplateResponse(
            request=request, name="error.html.jinja", status_code=503
        )

    runs_by_id = {run["id"]: run for run in runs}
    return templates.TemplateResponse(
        request=request,
        name="llm_text.html.jinja",
        context={
            "data": data,
            "url": url,
            "left": left,
            "right": right,
            "left_run": runs_by_id.get(left),
            "right_run": runs_by_id.get(right),
        },
    )
