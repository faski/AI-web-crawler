"""Route handlers for the stored LLM-parser runs.

Every endpoint here only reads rows that an explicit run already wrote. None
of them calls a model, so opening the comparison page costs nothing and takes
no time, whichever provider is configured.
"""

from fastapi import APIRouter, HTTPException

from ..lib.db import llm_queries
from ..schemas.llm_runs import (
    LlmComparisonRow,
    LlmDomainRow,
    LlmPageRow,
    LlmRunSummary,
    LlmTextComparison,
    LlmTextSide,
    SaveRunRequest,
)

router = APIRouter()


def _side(run_id: int, url: str) -> LlmTextSide | None:
    """Build one side of the text comparison from the stored row."""
    row = llm_queries.get_page_result(run_id, url)
    if row is None:
        return None
    return LlmTextSide(
        run_id=run_id,
        label=row["label"],
        model_name=row["model_name"],
        condition_name=row["condition_name"],
        status=row["status"],
        f1=row["f1"],
        precision=row["precision"],
        recall=row["recall"],
        excess_ratio=row["excess_ratio"],
        seconds=row["seconds"],
        parsed_text=row["parsed_text"] or "",
    )


@router.get("/llm_runs", response_model=list[LlmRunSummary])
def list_llm_runs():
    """Return every stored run with its headline numbers."""
    return llm_queries.list_runs()


@router.get("/llm_runs/{run_id}/domains", response_model=list[LlmDomainRow])
def llm_run_domains(run_id: int):
    """Return the per-domain averages of one run."""
    rows = llm_queries.get_run_by_domain(run_id)
    if not rows:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return rows


@router.get("/llm_runs/{run_id}/pages", response_model=list[LlmPageRow])
def llm_run_pages(run_id: int):
    """Return every page of one run, worst first."""
    rows = llm_queries.get_run_pages(run_id)
    if not rows:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return rows


@router.get("/llm_runs/{run_id}/text")
def llm_run_page_text(run_id: int, url: str):
    """Return the markdown one run produced for one page."""
    text = llm_queries.get_page_text(run_id, url)
    if text is None:
        raise HTTPException(
            status_code=404,
            detail="No stored text for this page (the run was made without --keep-text)",
        )
    return {"url": url, "parsed_text": text}


@router.get("/llm_runs/compare", response_model=list[LlmComparisonRow])
def compare_llm_runs(left: int, right: int):
    """Return the pages of two runs side by side, biggest difference first."""
    return llm_queries.compare_runs(left, right)


@router.post("/llm_runs")
def save_llm_run(body: SaveRunRequest):
    """Store a finished run. Used by the import script, not by the pages."""
    run_id = llm_queries.save_run(
        body.label,
        body.model_name,
        body.provider,
        body.condition_name,
        body.budget_tokens,
        body.records,
    )
    return {"id": run_id, "label": body.label, "pages": len(body.records)}


@router.get("/llm_runs/compare_text", response_model=LlmTextComparison)
def compare_llm_text(left: int, right: int, url: str):
    """Return the text two runs produced for one page.

    Reads stored rows only: no parsing and no model call happen here.
    """
    return LlmTextComparison(
        url=url,
        left=_side(left, url),
        right=_side(right, url),
    )
