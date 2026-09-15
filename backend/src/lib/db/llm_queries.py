"""SQL over the LLM-parser evaluation runs.

A run is one pass of the LLM parser over the gold standard with a given model
and condition. Results are written once, by an explicit run, and read back
many times by the comparison page: looking at the numbers must never cost a
model call.
"""

import json

from .connection import fetch_all, fetch_one, get_connection

# Columns of llm_page_results in the order used by the insert and the reads,
# kept in one place so the two cannot drift apart.
_PAGE_FIELDS = (
    "url", "domain", "status", "input_tokens", "html_kb", "seconds",
    "cpu_seconds", "chars", "fragments", "empty_fragments",
    "cost_usd", "cost_eur", "call_costs_usd", "prompt_tokens", "completion_tokens",
    "grounded",
    "precision_val", "recall_val", "f1", "cosine", "jaccard", "excess_ratio",
    "extracted_count", "sample_count", "parsed_text", "note",
)


def save_run(
    label: str,
    model_name: str,
    provider: str,
    condition_name: str,
    budget_tokens: int,
    records: list[dict],
    eur_per_usd: float | None = None,
) -> int:
    """Store one run and its pages, replacing a run with the same label.

    Re-importing the same results file updates the run in place instead of
    creating a second copy, so the comparison page never shows the same run
    twice.

    Returns:
        The id of the stored run.
    """
    connection = get_connection()
    command = connection.cursor()
    command.execute("DELETE FROM llm_runs WHERE label = ?", (label,))
    command.execute(
        """
        INSERT INTO llm_runs
            (label, model_name, provider, condition_name, budget_tokens,
             eur_per_usd)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (label, model_name, provider, condition_name, budget_tokens,
         eur_per_usd),
    )
    run_id = command.lastrowid
    placeholders = ", ".join(["?"] * (len(_PAGE_FIELDS) + 1))
    columns = ", ".join(("run_id",) + _PAGE_FIELDS)
    for record in records:
        command.execute(
            f"INSERT INTO llm_page_results ({columns}) VALUES ({placeholders})",
            (
                run_id,
                record["url"],
                record["domain"],
                record["status"],
                record.get("input_tokens"),
                record.get("html_kb"),
                record.get("seconds"),
                record.get("cpu_seconds"),
                record.get("chars"),
                record.get("fragments"),
                record.get("empty_fragments"),
                record.get("cost_usd"),
                record.get("cost_eur"),
                # Stored as JSON text: it is a list read back whole for
                # display, never filtered or summed in SQL.
                json.dumps(record["call_costs_usd"])
                if record.get("call_costs_usd")
                else None,
                record.get("prompt_tokens"),
                record.get("completion_tokens"),
                record.get("grounded"),
                record.get("precision"),
                record.get("recall"),
                record.get("f1"),
                record.get("cosine"),
                record.get("jaccard"),
                record.get("excess_ratio"),
                record.get("extracted_count"),
                record.get("sample_count"),
                record.get("parsed_text"),
                record.get("error"),
            ),
        )
    connection.commit()
    connection.close()
    return run_id


def list_runs() -> list[dict]:
    """Return every stored run with its headline numbers, newest first.

    The averages cover only the pages that were actually parsed: a page the
    model could not fit would otherwise drag a mean towards zero and hide the
    real quality of the pages it did handle. How many were skipped is reported
    separately, as ``too_long``.
    """
    rows = fetch_all(
        """
        SELECT r.id, r.label, r.model_name, r.provider, r.condition_name,
               r.budget_tokens, r.created_at,
               SUM(p.status = 'ok')        AS ok_count,
               SUM(p.status = 'too_long')  AS too_long_count,
               SUM(p.status = 'error')     AS error_count,
               AVG(CASE WHEN p.status='ok' THEN p.precision_val END),
               AVG(CASE WHEN p.status='ok' THEN p.recall_val END),
               AVG(CASE WHEN p.status='ok' THEN p.f1 END),
               AVG(CASE WHEN p.status='ok' THEN p.excess_ratio END),
               AVG(CASE WHEN p.status='ok' THEN p.seconds END),
               SUM(CASE WHEN p.status='ok' THEN p.cpu_seconds END),
               -- Only the pages that were actually sent. A page skipped for
               -- being too long has a token count too, but it never reached
               -- the model and must not appear as consumption.
               SUM(CASE WHEN p.status='ok' THEN p.input_tokens END),
               SUM(CASE WHEN p.status<>'ok' THEN p.input_tokens END),
               -- Chunking's own cost: one model call per piece, so a run that
               -- cut ten pages into four pieces each paid thirty calls more
               -- than the page count suggests.
               SUM(CASE WHEN p.status='ok' THEN COALESCE(p.fragments, 1) END),
               SUM(CASE WHEN p.status='ok' AND p.fragments > 1 THEN 1 END),
               -- The per-page sum when there is one, otherwise the figure
               -- recorded for the run as a whole.
               COALESCE(SUM(p.cost_eur), r.measured_cost_eur),
               r.eur_per_usd,
               SUM(p.cost_eur) IS NULL AND r.measured_cost_eur IS NOT NULL
        FROM llm_runs r
        LEFT JOIN llm_page_results p ON p.run_id = r.id
        GROUP BY r.id
        ORDER BY r.created_at DESC, r.id DESC
        """
    )
    return [
        {
            "id": row[0], "label": row[1], "model_name": row[2],
            "provider": row[3], "condition_name": row[4],
            "budget_tokens": row[5], "created_at": str(row[6]),
            "ok": int(row[7] or 0), "too_long": int(row[8] or 0),
            "errors": int(row[9] or 0),
            "precision": row[10], "recall": row[11], "f1": row[12],
            "excess_ratio": row[13], "seconds": row[14],
            "cpu_seconds": row[15],
            "input_tokens": int(row[16] or 0),
            "skipped_tokens": int(row[17] or 0),
            "calls": int(row[18] or 0),
            "chunked_pages": int(row[19] or 0),
            "cost_eur": row[20],
            "eur_per_usd": row[21],
            "cost_is_aggregate": bool(row[22]),
        }
        for row in rows
    ]


def set_measured_cost(label: str, cost_eur: float) -> None:
    """Record what a run cost overall, for a run with no per-call breakdown.

    Used for runs made before the per-call accounting existed: the amount is
    read from the provider's balance before and after, which is the truth for
    the run even though it cannot be split across pages.
    """
    connection = get_connection()
    command = connection.cursor()
    command.execute(
        "UPDATE llm_runs SET measured_cost_eur = ? WHERE label = ?",
        (cost_eur, label),
    )
    connection.commit()
    connection.close()


def get_run_by_domain(run_id: int) -> list[dict]:
    """Return one row per domain for a run, averaged over its parsed pages."""
    rows = fetch_all(
        """
        SELECT domain,
               SUM(status = 'ok'), SUM(status = 'too_long'), SUM(status = 'error'),
               AVG(CASE WHEN status='ok' THEN precision_val END),
               AVG(CASE WHEN status='ok' THEN recall_val END),
               AVG(CASE WHEN status='ok' THEN f1 END),
               AVG(CASE WHEN status='ok' THEN cosine END),
               AVG(CASE WHEN status='ok' THEN excess_ratio END),
               AVG(CASE WHEN status='ok' THEN seconds END)
        FROM llm_page_results
        WHERE run_id = ?
        GROUP BY domain
        ORDER BY domain
        """,
        (run_id,),
    )
    return [
        {
            "domain": row[0], "ok": int(row[1] or 0),
            "too_long": int(row[2] or 0), "errors": int(row[3] or 0),
            "precision": row[4], "recall": row[5], "f1": row[6],
            "cosine": row[7], "excess_ratio": row[8], "seconds": row[9],
        }
        for row in rows
    ]


def get_run_pages(run_id: int) -> list[dict]:
    """Return every page of a run, worst score first so failures are visible."""
    rows = fetch_all(
        """
        SELECT url, domain, status, input_tokens, seconds,
               precision_val, recall_val, f1, excess_ratio,
               fragments, cost_eur, call_costs_usd, grounded
        FROM llm_page_results
        WHERE run_id = ?
        ORDER BY (f1 IS NULL) DESC, f1 ASC
        """,
        (run_id,),
    )
    return [
        {
            "url": row[0], "domain": row[1], "status": row[2],
            "input_tokens": row[3], "seconds": row[4],
            "precision": row[5], "recall": row[6], "f1": row[7],
            "excess_ratio": row[8],
            "fragments": row[9],
            "cost_eur": row[10],
            # The per-call breakdown is stored as JSON text; a page read in
            # one call has a single entry, a page read in four has four.
            "call_costs_usd": json.loads(row[11]) if row[11] else None,
            "grounded": row[12],
        }
        for row in rows
    ]


def get_page_text(run_id: int, url: str) -> str | None:
    """Return the markdown one run produced for one page, if it was kept."""
    row = fetch_one(
        "SELECT parsed_text FROM llm_page_results WHERE run_id = ? AND url = ?",
        (run_id, url),
    )
    return row[0] if row else None


def compare_runs(left_id: int, right_id: int) -> list[dict]:
    """Return the pages of two runs side by side, biggest difference first.

    A page is listed even when only one of the two runs could parse it: those
    are exactly the cases where one condition reaches a page the other cannot,
    which is the comparison worth seeing.
    """
    rows = fetch_all(
        """
        SELECT COALESCE(a.url, b.url), COALESCE(a.domain, b.domain),
               a.status, b.status, a.f1, b.f1,
               a.precision_val, b.precision_val,
               a.recall_val, b.recall_val,
               a.input_tokens, b.input_tokens,
               a.seconds, b.seconds
        FROM (SELECT * FROM llm_page_results WHERE run_id = ?) a
        LEFT JOIN (SELECT * FROM llm_page_results WHERE run_id = ?) b
               ON a.url = b.url
        ORDER BY ABS(COALESCE(b.f1, 0) - COALESCE(a.f1, 0)) DESC
        """,
        (left_id, right_id),
    )
    return [
        {
            "url": row[0], "domain": row[1],
            "left_status": row[2], "right_status": row[3],
            "left_f1": row[4], "right_f1": row[5],
            "left_precision": row[6], "right_precision": row[7],
            "left_recall": row[8], "right_recall": row[9],
            "left_tokens": row[10], "right_tokens": row[11],
            "left_seconds": row[12], "right_seconds": row[13],
        }
        for row in rows
    ]


def get_page_result(run_id: int, url: str) -> dict | None:
    """Return one page of one run, text included, or None if it is not there."""
    row = fetch_one(
        """
        SELECT p.status, p.f1, p.precision_val, p.recall_val, p.excess_ratio,
               p.seconds, p.input_tokens, p.chars, p.parsed_text, p.note,
               r.label, r.model_name, r.provider, r.condition_name
        FROM llm_page_results p
        JOIN llm_runs r ON r.id = p.run_id
        WHERE p.run_id = ? AND p.url = ?
        """,
        (run_id, url),
    )
    if row is None:
        return None
    return {
        "status": row[0], "f1": row[1], "precision": row[2], "recall": row[3],
        "excess_ratio": row[4], "seconds": row[5], "input_tokens": row[6],
        "chars": row[7], "parsed_text": row[8], "note": row[9],
        "label": row[10], "model_name": row[11], "provider": row[12],
        "condition_name": row[13],
    }
