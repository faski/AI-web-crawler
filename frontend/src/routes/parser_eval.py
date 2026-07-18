"""Parser & Evaluation page (GET /parser)."""

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse

from ..client import (
    BackendUnavailable,
    evaluate,
    evaluate_judge,
    get_domains,
    get_gold_standard,
    get_gs_urls,
    parse_url,
)
from ..templates import templates

router = APIRouter()


@router.get("/parser", response_class=HTMLResponse)
def parser_eval(
    request: Request,
    url: str = Query(default=""),
    mode: str = Query(default="live"),
):
    """Parse a URL (live or local), evaluate against GS if available, return results."""
    try:
        domains = get_domains()
        # URLs grouped by domain, so the page can show two linked dropdowns
        # (pick a domain, then pick one of its URLs).
        gs_urls_by_domain = {domain: get_gs_urls(domain) for domain in domains}
        all_gs_urls = [
            gs_url
            for urls in gs_urls_by_domain.values()
            for gs_url in urls
        ]

        result = None
        if url.strip():
            local = mode == "local"
            data, error = parse_url(url.strip(), local=local)
            result = {"data": data, "error": error, "url": url.strip(), "mode": mode}

            if not error:
                # The backend already returns the markdown-stripped text.
                result["cleaned_text"] = data.get("cleaned_text", "")

                # If the URL is in the gold standard, fetch the gold text and
                # compute both quantitative and judge evaluations.
                if url.strip() in all_gs_urls:
                    gold_entry = get_gold_standard(url.strip())
                    if gold_entry:
                        result["gold_text"] = gold_entry["gold_text"]
                        # Send the raw texts: the backend strips both sides itself.
                        parsed_text = data.get("parsed_text", "")
                        result["eval"] = evaluate(parsed_text, gold_entry["gold_text"])
                        judge = evaluate_judge(parsed_text, gold_entry["gold_text"])
                        result["judge"] = judge
                        if judge:
                            result["judge_parsed_preview"] = judge.get("parsed_preview", "")
                            result["judge_gold_preview"] = judge.get("gold_preview", "")
                            result["judge_text_cap"] = judge.get("text_cap")
    except BackendUnavailable:
        return templates.TemplateResponse(
            request=request, name="error.html.jinja", status_code=503
        )

    return templates.TemplateResponse(
        request=request,
        name="parser_eval.html.jinja",
        context={
            "gs_urls_by_domain": gs_urls_by_domain,
            "result": result,
            "selected_url": url.strip(),
            "selected_mode": mode,
        },
    )
