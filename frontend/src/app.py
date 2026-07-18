"""FastAPI application entry point for the frontend.

Run from the frontend/ directory:
    uvicorn src.app:app --host 0.0.0.0 --port 8004

Static files are served from frontend/static/.
Templates are rendered from frontend/templates/.
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .routes import gs_builder_router, index_router, parser_eval_router, stats_router

app = FastAPI(title="Creeping Crawler")

app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(index_router)
app.include_router(parser_eval_router)
app.include_router(gs_builder_router)
app.include_router(stats_router)
