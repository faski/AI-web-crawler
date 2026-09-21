"""SQL over the LLM-parser evaluation runs.

A run is one pass of the LLM parser over the gold standard. Results are
written once by an import script and read many times by the comparison page,
which must never cost a model call.
"""

import json

from .connection import execute, fetch_all, fetch_one, get_connection

# Columns of llm_page_results in the order used by the insert and the reads,
# kept in one place so the two cannot drift apart.
_PAGE_FIELDS = (
    "url", "domain", "status", "input_tokens", "html_kb", "seconds",
    "cpu_seconds", "chars", "fragments", "empty_fragments",
    "cost_usd", "cost_eur", "call_costs_usd", "prompt_tokens", "completion_tokens",
    "providers", "generation_ids",
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

    Re-importing the same file replaces the run instead of adding a second
    copy. The old rows go with it, self-check verdicts included.

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
                # JSON text: read back whole for display, never used in SQL.
                json.dumps(record["call_costs_usd"])
                if record.get("call_costs_usd")
                else None,
                record.get("prompt_tokens"),
                record.get("completion_tokens"),
                # JSON text too: one entry per call, to say who served it.
                json.dumps(record["providers"]) if record.get("providers") else None,
                json.dumps(record["generation_ids"])
                if record.get("generation_ids")
                else None,
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

    The averages cover only the pages that were parsed: a page that did not
    fit would drag the mean down. The skipped ones are counted separately, as
    ``too_long``.
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
               -- Wall-clock time of the whole run, used to cost the local
               -- hardware. The average above is only for reading.
               SUM(CASE WHEN p.status='ok' THEN p.seconds END),
               -- Only the pages actually sent. A skipped page has a token
               -- count too, but it never reached the model.
               SUM(CASE WHEN p.status='ok' THEN p.input_tokens END),
               SUM(CASE WHEN p.status<>'ok' THEN p.input_tokens END),
               -- One call per piece, so a chunked run pays for more calls
               -- than it has pages.
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
            "wall_seconds": row[16],
            "input_tokens": int(row[17] or 0),
            "skipped_tokens": int(row[18] or 0),
            "calls": int(row[19] or 0),
            "chunked_pages": int(row[20] or 0),
            "cost_eur": row[21],
            "eur_per_usd": row[22],
            "cost_is_aggregate": bool(row[23]),
        }
        for row in rows
    ]


def set_measured_cost(label: str, cost_eur: float) -> None:
    """Record what a run cost overall, for a run with no per-call breakdown.

    For runs made before the per-call accounting existed. The amount comes
    from the provider's balance before and after, so it is right for the run
    but cannot be split across pages.
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


def attach_self_check(run_id: int, results: list[dict]) -> int:
    """Store the model's verdicts on the pages of a run already imported.

    The check is a second pass over an extraction that already exists, so the
    verdicts become columns of that run's pages instead of a run of their own.
    Only with the verdict and the F1 side by side can you ask whether the
    model caught its own bad pages.

    ``grounded`` is filled in at the same time: the check recomputes it, and
    older runs have nothing in that column.

    Returns:
        How many pages were updated.
    """
    checked = [r for r in results if r.get("status") == "ok"]
    # Counted before writing, by asking which URLs this run has. The driver's
    # UPDATE rowcount is no use here: it came back 0 on a write that landed.
    known = {
        row[0]
        for row in fetch_all(
            "SELECT url FROM llm_page_results WHERE run_id = ?", (run_id,)
        )
    }
    for result in checked:
        execute(
            """
            UPDATE llm_page_results
               SET check_complete = ?, check_coherent = ?, check_notes = ?,
                   check_fragments = ?, check_cost_eur = ?,
                   check_coherence = ?, check_coherence_evidence = ?,
                   check_claimed_missing = ?, check_quote_verdicts = ?,
                   check_omissions = ?,
                   grounded = COALESCE(?, grounded)
             WHERE run_id = ? AND url = ?
            """,
            (
                int(bool(result["complete"])),
                int(bool(result["coherent"])),
                result.get("notes"),
                result.get("check_fragments"),
                result.get("cost_eur"),
                result.get("coherence"),
                result.get("coherence_evidence"),
                json.dumps(result["claimed_missing"])
                if result.get("claimed_missing")
                else None,
                json.dumps(result["quote_checks"])
                if result.get("quote_checks")
                else None,
                result.get("omissions"),
                result.get("grounded"),
                run_id,
                result["url"],
            ),
        )
    return sum(1 for r in checked if r["url"] in known)


def get_run_pages(run_id: int) -> list[dict]:
    """Return every page of a run, worst score first so failures are visible."""
    rows = fetch_all(
        """
        SELECT url, domain, status, input_tokens, seconds,
               precision_val, recall_val, f1, excess_ratio,
               fragments, cost_eur, call_costs_usd, grounded,
               check_complete, check_coherent, check_notes, check_cost_eur
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
            # JSON text: one entry per call.
            "call_costs_usd": json.loads(row[11]) if row[11] else None,
            "grounded": row[12],
            # NULL when the run was never self-checked, which is not the same
            # as a page that was checked and passed.
            "check_complete": None if row[13] is None else bool(row[13]),
            "check_coherent": None if row[14] is None else bool(row[14]),
            "check_notes": row[15],
            "check_cost_eur": row[16],
        }
        for row in rows
    ]


def get_self_check_summary(run_id: int) -> dict | None:
    """Return how the model's own verdicts line up with the gold standard.

    The verdict is ``check_coherent`` alone. The other question the model was
    asked, "is anything missing", tracked the gold standard at r = +0.06, so
    combining the two with AND only threw information away. ``not_complete``
    is still reported, as the measurement that led to dropping it.

    What to look at is not how often the check is right - one that passes
    everything agrees with a corpus that is mostly good - but the distance
    between the two mean F1s, promoted pages against failed ones. A gap means
    the verdict says something.

    Returns None when this run was never checked.
    """
    row = fetch_one(
        """
        SELECT COUNT(*),
               SUM(check_coherent),
               AVG(CASE WHEN check_coherent THEN f1 END),
               AVG(CASE WHEN NOT check_coherent THEN f1 END),
               SUM(NOT check_complete),
               SUM(NOT check_coherent),
               SUM(check_cost_eur),
               AVG(f1),
               -- The same verdict as a 0-5 score. Averaged only over the
               -- pages that have one: older runs stored just the boolean.
               AVG(check_coherence),
               COUNT(check_coherence),
               -- Omissions the code confirmed, not claims the model made.
               SUM(check_omissions)
        FROM llm_page_results
        WHERE run_id = ? AND check_coherent IS NOT NULL
        """,
        (run_id,),
    )
    if row is None or not row[0]:
        return None
    return {
        "checked": int(row[0]),
        "passed": int(row[1] or 0),
        "failed": int(row[0]) - int(row[1] or 0),
        "f1_passed": row[2],
        "f1_failed": row[3],
        "not_complete": int(row[4] or 0),
        "not_coherent": int(row[5] or 0),
        "cost_eur": row[6],
        "f1_all": row[7],
        "coherence_mean": row[8],
        "scored": int(row[9] or 0),
        "omissions": int(row[10] or 0),
    }


def get_self_check_pages(run_id: int) -> list[dict]:
    """Return the checked pages, the ones the check failed first.

    Ordered so the disagreements come first: a page failed with a high F1 is
    a false alarm, one passed with a low F1 is a miss. The order follows the
    coherence answer, which is the verdict; completeness comes along but does
    not decide anything.

    The quotes come with the row too: what the model said was missing, and
    what the code found when it looked for it.
    """
    rows = fetch_all(
        """
        SELECT url, domain, f1, grounded, check_complete, check_coherent,
               check_notes, check_fragments, check_cost_eur,
               check_coherence, check_coherence_evidence,
               check_claimed_missing, check_quote_verdicts, check_omissions
        FROM llm_page_results
        WHERE run_id = ? AND check_coherent IS NOT NULL
        ORDER BY check_coherent ASC, f1 ASC
        """,
        (run_id,),
    )
    return [
        {
            "url": row[0], "domain": row[1], "f1": row[2], "grounded": row[3],
            "check_complete": bool(row[4]), "check_coherent": bool(row[5]),
            "check_notes": row[6], "check_fragments": row[7],
            "check_cost_eur": row[8],
            "check_coherence": row[9],
            "check_coherence_evidence": row[10],
            "check_claimed_missing": _decode_json(row[11], []),
            "check_quote_verdicts": _decode_json(row[12], []),
            "check_omissions": row[13],
        }
        for row in rows
    ]


def _decode_json(raw: str | None, fallback: list) -> list:
    """Return a JSON column as a list, or ``fallback`` if there is nothing.

    NULL is the normal case: a page with no claimed omissions stores no array.
    Broken JSON is treated the same way instead of raising, so one bad row
    cannot break the whole page.
    """
    if not raw:
        return fallback
    try:
        decoded = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return fallback
    return decoded if isinstance(decoded, list) else fallback


def get_page_text(run_id: int, url: str) -> str | None:
    """Return the markdown one run produced for one page, if it was kept."""
    row = fetch_one(
        "SELECT parsed_text FROM llm_page_results WHERE run_id = ? AND url = ?",
        (run_id, url),
    )
    return row[0] if row else None


def compare_runs(left_id: int, right_id: int) -> list[dict]:
    """Return the pages of two runs side by side, biggest difference first.

    A page is listed even when only one of the two runs could parse it: that
    is exactly the case worth seeing.
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
