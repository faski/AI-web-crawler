"""Route handlers for the stored LLM-parser runs.

Every endpoint here only reads rows that an explicit run already wrote. None
of them calls a model, so opening the comparison page costs nothing and takes
no time, whichever provider is configured.
"""

from fastapi import APIRouter, HTTPException

from ..lib.db import llm_queries
from ..lib.evaluation import local_cost
from ..schemas.llm_runs import (
    LlmComparisonRow,
    LlmDomainRow,
    LlmPageRow,
    LlmRunSummary,
    LlmSelfCheckRow,
    LlmSelfCheckSummary,
    LlmTextComparison,
    LlmTextSide,
    SaveRunRequest,
)

router = APIRouter()


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Return Pearson's r, or None when one of the two series does not vary.

    Used to ask whether a verdict carries information about the score. A
    constant series has no correlation to report, and returning 0 for it would
    read as "measured, and no relation" instead of "not measurable".
    """
    if len(xs) < 2:
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    sx = sum(d * d for d in dx) ** 0.5
    sy = sum(d * d for d in dy) ** 0.5
    if not sx or not sy:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / (sx * sy)


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
    """Return every stored run with its headline numbers.

    The local cost is derived here rather than stored: it is a function of the
    run's duration and of four parameters about one machine, and those
    parameters change without the run changing. Freezing a figure into the
    database would mean a corrected electricity price could never reach the
    runs already recorded.

    It stays empty for a run made against a remote provider, whose seconds
    were spent waiting rather than computing.
    """
    runs = llm_queries.list_runs()
    for run in runs:
        spent = local_cost(run.get("wall_seconds"), run.get("provider"))
        if spent is not None:
            run["local_cost_eur"] = spent.total_eur
            run["local_cost_electricity_eur"] = spent.electricity_eur
            run["local_cost_hardware_eur"] = spent.hardware_eur
    return runs


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


@router.get("/llm_runs/{run_id}/self_check")
def llm_run_self_check(run_id: int):
    """Return the run's self-check summary and the pages it judged.

    404 when the run exists but was never checked: an empty summary would read
    as "the model found nothing wrong", which is the opposite of "nobody
    asked it".
    """
    summary = llm_queries.get_self_check_summary(run_id)
    if summary is None:
        raise HTTPException(
            status_code=404, detail=f"Run {run_id} has no self-check results"
        )
    pages = llm_queries.get_self_check_pages(run_id)
    scored = [p for p in pages if p["f1"] is not None]
    # The verdict is the coherence answer alone; the completeness answer is
    # correlated too, and kept, because "we measured it and dropped it" is a
    # result and "we never looked" is not.
    summary["r_check"] = _pearson(
        [float(p["check_coherent"]) for p in scored], [p["f1"] for p in scored]
    )
    summary["r_complete"] = _pearson(
        [float(p["check_complete"]) for p in scored], [p["f1"] for p in scored]
    )
    anchored = [p for p in scored if p["grounded"] is not None]
    summary["r_grounded"] = _pearson(
        [p["grounded"] for p in anchored], [p["f1"] for p in anchored]
    )
    return {
        "summary": LlmSelfCheckSummary(**summary),
        "pages": [LlmSelfCheckRow(**row) for row in pages],
    }


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
