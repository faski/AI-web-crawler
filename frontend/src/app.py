"""FastAPI application entry point for the frontend.

Run from the frontend/ directory:
    uvicorn src.app:app --host 0.0.0.0 --port 8004

Static files are served from frontend/static/.
Templates are rendered from frontend/templates/.
"""

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

from .routes import (
    gs_builder_router,
    index_router,
    llm_runs_router,
    parser_eval_router,
    stats_router,
)

app = FastAPI(title="Creeping Crawler")

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.middleware("http")
async def no_cache_html(request: Request, call_next):
    """Tell the browser never to reuse a rendered page.

    Every page here is a view onto a database that changes: a run gets
    imported, verdicts get attached, a parser gets re-run. Without a header
    saying so, a browser is free to decide for itself how long to keep a copy,
    and it does - which shows up as a page that quietly disagrees with the
    database and sends the reader looking for a bug that is not there.

    Only the pages. Files under /static carry their own cache-busting and are
    left alone, so stylesheets and scripts stay cached.
    """
    response = await call_next(request)
    if not request.url.path.startswith("/static"):
        response.headers["Cache-Control"] = "no-store, must-revalidate"
    return response

app.include_router(index_router)
app.include_router(parser_eval_router)
app.include_router(gs_builder_router)
app.include_router(stats_router)
app.include_router(llm_runs_router)
